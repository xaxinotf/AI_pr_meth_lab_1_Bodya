# src/train_baseline.py
# GPU-only Linear/ElasticNet regression in PyTorch + TimeSeriesSplit OOF
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit

from src.data import load_ld2011, save_parquet_versions
from src.eval import mae, rmse, mape


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "baseline"
MODELS.mkdir(parents=True, exist_ok=True)


# ----------------------
# Feature engineering (optimized, no fragmentation)
# ----------------------
def build_supervised_table(df_wide: pd.DataFrame, series_id: str, n_lags: int, horizon: int) -> pd.DataFrame:
    """
    Перетворює одновимірний ряд у табличку для регресії (оптимізовано):
      - ціль y = s.shift(-horizon)
      - лаги lag_1..lag_n формуються списком і конкатяться одним викликом
      - календарні та циклічні фічі додаються за один раз

    Повертає DataFrame з індексом datetime та колонками:
      [y, lag_1..lag_n, hour, dow, month, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos]
    """
    assert series_id in df_wide.columns, f"{series_id} нема в колонках!"
    s = df_wide[series_id].astype("float32").sort_index()

    # ціль: значення через `horizon` кроків
    y = s.shift(-horizon).rename("y")

    # лаги (створюємо список Series і конкатимо один раз)
    if n_lags > 0:
        lag_cols = [s.shift(i).rename(f"lag_{i}") for i in range(1, n_lags + 1)]
        lags_df = pd.concat(lag_cols, axis=1)
    else:
        lags_df = pd.DataFrame(index=s.index)

    # календарні фічі
    idx = s.index
    cal_df = pd.DataFrame({
        "hour": idx.hour.astype("int16"),
        "dow": idx.dayofweek.astype("int16"),
        "month": idx.month.astype("int16"),
    }, index=idx)
    cal_df["is_weekend"] = (cal_df["dow"] >= 5).astype("int8")

    # циклічні фічі
    cal_df["hour_sin"] = np.sin(2 * np.pi * cal_df["hour"] / 24)
    cal_df["hour_cos"] = np.cos(2 * np.pi * cal_df["hour"] / 24)
    cal_df["dow_sin"]  = np.sin(2 * np.pi * cal_df["dow"] / 7)
    cal_df["dow_cos"]  = np.cos(2 * np.pi * cal_df["dow"] / 7)

    # фінальний фрейм: concat одним викликом і один dropna
    df = pd.concat([y, lags_df, cal_df], axis=1).dropna()

    # типи
    num_cols = df.columns.difference(["is_weekend"])
    df[num_cols] = df[num_cols].astype("float32")

    return df


# ----------------------
# Model
# ----------------------
class LinearElasticNet(nn.Module):
    """
    y_hat = XW + b  (ElasticNet додається в лосс)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_fold(Xtr, ytr, Xva, yva, l1=0.0, l2=0.0, epochs=200, lr=1e-2):
    assert torch.cuda.is_available(), "CUDA недоступна — тренування лише на GPU."
    device = torch.device("cuda")

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(-1)
    Xva = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva = torch.tensor(yva, dtype=torch.float32, device=device).unsqueeze(-1)

    model = LinearElasticNet(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = mse(pred, ytr)
        W = model.linear.weight
        l1_term = l1 * torch.sum(torch.abs(W)) if l1 > 0 else 0.0
        l2_term = l2 * torch.sum(W * W)       if l2 > 0 else 0.0
        total = loss + l1_term + l2_term
        total.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        yhat = model(Xva).squeeze(-1).detach().cpu().numpy()
    return model, yhat


def fit_full(X, y, l1=0.0, l2=0.0, epochs=300, lr=1e-2):
    assert torch.cuda.is_available(), "CUDA недоступна — тренування лише на GPU."
    device = torch.device("cuda")
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)

    model = LinearElasticNet(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = mse(pred, y)
        W = model.linear.weight
        l1_term = l1 * torch.sum(torch.abs(W)) if l1 > 0 else 0.0
        l2_term = l2 * torch.sum(W * W)       if l2 > 0 else 0.0
        total = loss + l1_term + l2_term
        total.backward()
        opt.step()
    return model


def run_baseline(series_id: str, n_lags: int, horizon: int,
                 n_splits=5, l1=1e-6, l2=1e-6, freq: str = "hourly"):
    # Дані (hourly або 15min)
    df = load_ld2011()
    p15, ph = save_parquet_versions(df)
    if freq == "hourly":
        dfw = pd.read_parquet(ph)
    elif freq == "15min":
        dfw = pd.read_parquet(p15)
    else:
        raise ValueError("--freq must be 'hourly' or '15min'")

    data = build_supervised_table(dfw, series_id=series_id, n_lags=n_lags, horizon=horizon)

    X = data.drop(columns=["y"]).values
    y = data["y"].values
    idx = data.index  # для збереження timestamp у прогнозі

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    oof_pred = np.full_like(y, fill_value=np.nan, dtype=np.float32)

    for fold, (tr, va) in enumerate(tscv.split(X), start=1):
        model, yhat = train_fold(
            X[tr], y[tr], X[va], y[va],
            l1=l1, l2=l2, epochs=200, lr=1e-2
        )
        oof_pred[va] = yhat
        metrics.append({
            "fold": fold,
            "MAE": mae(y[va], yhat),
            "RMSE": rmse(y[va], yhat),
            "MAPE": mape(y[va], yhat),
        })

    # сукупні метрики за OOF
    mask = ~np.isnan(oof_pred)
    oof_mae = mae(y[mask], oof_pred[mask])
    oof_rmse = rmse(y[mask], oof_pred[mask])
    oof_mape = mape(y[mask], oof_pred[mask])

    # донавчання на всіх
    full_model = fit_full(X, y, l1=l1, l2=l2, epochs=300, lr=1e-2)
    out_model = MODELS / f"{series_id}.pt"
    torch.save(full_model.state_dict(), out_model)

    # збереження прогнозів/метрик
    out_dir = ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_path = out_dir / f"baseline_{series_id}_forecast.csv"
    met_path = out_dir / f"baseline_{series_id}_metrics.csv"

    pd.DataFrame({"timestamp": idx, "y_true": y, "yhat": oof_pred}).to_csv(fc_path, index=False)
    pd.DataFrame([{"MAE": oof_mae, "RMSE": oof_rmse, "MAPE": oof_mape}]).to_csv(met_path, index=False)

    return pd.DataFrame(metrics), out_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_id", type=str, default="MT_001", help="Яку серію (колонку) брати")
    ap.add_argument("--horizon", type=int, default=24, help="Горизонт прогнозу (кроків уперед)")
    ap.add_argument("--lags", type=int, default=168, help="К-сть лагових ознак")
    ap.add_argument("--splits", type=int, default=5, help="К-сть фолдів TimeSeriesSplit")
    ap.add_argument("--l1", type=float, default=1e-6, help="L1 регуляризація")
    ap.add_argument("--l2", type=float, default=1e-6, help="L2 регуляризація")
    ap.add_argument("--freq", type=str, choices=["hourly", "15min"], default="hourly",
                    help="Яку агрегацію даних використати")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA недоступна — ми тренуємось лише на GPU."

    fold_metrics, model_path = run_baseline(
        series_id=args.series_id,
        n_lags=args.lags,
        horizon=args.horizon,
        n_splits=args.splits,
        l1=args.l1,
        l2=args.l2,
        freq=args.freq,
    )

    print(fold_metrics.to_string(index=False))
    print(f"Model saved to: {model_path}")
