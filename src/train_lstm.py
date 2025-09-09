# src/train_lstm.py
# GPU-only LSTM with fast windowing + forecast/metrics saving (accepts --seq_len alias)
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_ld2011, save_parquet_versions
from src.features import build_feature_table
from src.eval import mae, rmse, mape

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "lstm"
MODELS.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True  # speed on fixed shapes


class TinyLSTM(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)         # [B, T, H]
        return self.fc(out[:, -1, :]) # [B, 1]


def make_windows(df, series_id="MT_001", T=24, horizon=1):
    """
    Швидке формування вікон:
      - X: [N, T, F] з фічей features.build_feature_table(...)
      - y: [N, 1] ціль через 'horizon' кроків
      - ts: індекси часу для кожного таргета
    """
    data = build_feature_table(df, series_id)
    Xfull = data.drop(columns=["y"]).values.astype("float32", copy=False)
    yfull = data["y"].values.astype("float32", copy=False)
    idx = data.index

    n = Xfull.shape[0] - T - horizon + 1
    if n <= 0:
        raise ValueError("Занадто велике T+horizon для довжини ряду — зменш T або horizon.")

    starts = np.arange(n)[:, None] + np.arange(T)[None, :]
    X = Xfull[starts, :]                               # [n, T, F]
    y = yfull[T + horizon - 1 : T + horizon - 1 + n]   # [n]
    ts = idx[T + horizon - 1 : T + horizon - 1 + n]

    X = torch.from_numpy(X)
    y = torch.from_numpy(y).unsqueeze(-1)
    return X, y, ts


def main(series_id="MT_001", seq_len=24, horizon=1, batch=256, epochs=10, lr=1e-3, workers=0, use_gpu_flag=False):
    # GPU-only, як ти просив
    assert torch.cuda.is_available(), "CUDA недоступна — тренування лише на GPU."
    device = torch.device("cuda")

    # Дані (hourly варіант)
    df = load_ld2011()
    _, ph = save_parquet_versions(df)
    dfw = pd.read_parquet(ph)

    # Вікна
    X, y, ts = make_windows(dfw, series_id=series_id, T=seq_len, horizon=horizon)
    pin = True  # ми на GPU, можна pin_memory
    ds = TensorDataset(X, y)
    train_dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                          pin_memory=pin, num_workers=workers)
    eval_dl  = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False,
                          pin_memory=pin, num_workers=workers)

    # Модель
    model = TinyLSTM(in_dim=X.shape[-1], hid=64, out_dim=1, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Тренування
    model.train()
    n_train = len(train_dl.dataset)
    for ep in range(1, epochs + 1):
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep}: MSE={total / n_train:.6f}")

    # Інференс (усі вікна, без shuffle)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in eval_dl:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb).cpu().numpy().ravel()
            preds.append(pred)
    yhat_all = np.concatenate(preds)[: len(y)]

    # Метрики
    y_np = y.numpy().ravel()
    m_mae, m_rmse, m_mape = mae(y_np, yhat_all), rmse(y_np, yhat_all), mape(y_np, yhat_all)

    # Збереження
    out = MODELS / f"{series_id}.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved model: {out}")

    out_dir = ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_path = out_dir / f"lstm_{series_id}_forecast.csv"
    met_path = out_dir / f"lstm_{series_id}_metrics.csv"

    pd.DataFrame({"timestamp": ts, "y_true": y_np, "yhat": yhat_all}).to_csv(fc_path, index=False)
    pd.DataFrame([{"MAE": m_mae, "RMSE": m_rmse, "MAPE": m_mape}]).to_csv(met_path, index=False)
    print(f"Saved forecast: {fc_path}")
    print(f"Saved metrics:  {met_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_id", type=str, default="MT_001")
    # приймаємо і --T, і --seq_len (seq_len має пріоритет, якщо задано)
    ap.add_argument("--T", type=int, default=24, help="Довжина історичного вікна")
    ap.add_argument("--seq_len", type=int, help="Синонім до --T")
    ap.add_argument("--horizon", type=int, default=1, help="Горизонт прогнозу (кроків уперед)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0, help="num_workers для DataLoader")
    ap.add_argument("--gpu", action="store_true", help="Прапорець для сумісності; тренування все одно лише на GPU")
    args = ap.parse_args()

    seq_len = args.seq_len if args.seq_len is not None else args.T
    main(args.series_id, seq_len, args.horizon, args.batch, args.epochs, args.lr, args.workers, args.gpu)
