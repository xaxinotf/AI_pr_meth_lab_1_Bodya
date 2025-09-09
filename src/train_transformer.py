# GPU-only Temporal Fusion Transformer (pytorch-forecasting)
# + повний horizon у прогнозах (а не тільки останній крок)
# + тихі логи, fast start, прапорець --skip
# + робастний парсер виходу model.predict (2 або 3 значення в tuple)
# + безпечна обробка випадку, коли немає series_id (ставимо "ALL")

import os
os.environ["MPLBACKEND"] = "Agg"  # вимкнути TkAgg у фонових процесах
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.data import load_ld2011, save_parquet_versions
from src.eval import mae, rmse, mape

# ---- тихі логи Lightning ----
logging.getLogger("lightning").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "transformer"
MODELS.mkdir(parents=True, exist_ok=True)

# reproducibility & speed
pl.seed_everything(42)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # Tensor Cores (RTX)

def prepare_long(df_wide: pd.DataFrame):
    """
    wide -> long та time_idx (години від t0).
    Повертає df_long і t0 (мінімальний timestamp) для відновлення дат.
    """
    df = df_wide.copy()
    df.index.name = "time"
    df = df.reset_index()
    df_long = df.melt(id_vars=["time"], var_name="series_id", value_name="value")
    t0 = df_long["time"].min()
    df_long["time_idx"] = ((df_long["time"] - t0).dt.total_seconds() // 3600).astype(int)
    return df_long, pd.to_datetime(t0)

def _unwrap_target(y):
    """Повертає таргет-тензор [B, pred_len], навіть якщо y був (y, weight) або списком."""
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y

def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def main(
    series_id_prefix: str = "MT_",
    max_series: int = 50,
    max_encoder_length: int = 168,     # 7 діб по годинах
    max_prediction_length: int = 24,   # 1 доба вперед
    epochs: int = 3,
    batch: int = 128,
    workers: int = 0,                  # Windows: 0-4; Linux: 8-16
    skip: bool = False,                # пропустити тренування/інференс
):
    # --- GPU only ---
    assert torch.cuda.is_available(), "CUDA недоступна — тренування лише на GPU."

    # --- 1) Дані: hourly ---
    df = load_ld2011()
    _, ph = save_parquet_versions(df)
    dfw = pd.read_parquet(ph).astype("float32")

    # обмежимо число серій для швидкого старту
    cols = [c for c in dfw.columns if c.startswith(series_id_prefix)][:max_series]
    dfw = dfw[cols]

    df_long, t0 = prepare_long(dfw)

    # --- 2) train/val split ---
    last_time = df_long["time_idx"].max()
    training_cutoff = last_time - max_prediction_length
    train_df = df_long[df_long["time_idx"] <= training_cutoff].copy()

    # --- 3) TimeSeriesDataSet ---
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        min_encoder_length=max(24, max_encoder_length // 2),
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["value"],
        time_varying_known_reals=["time_idx"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=False,
    )

    # validation з того самого конфігу; predict=True готує предикшени правильно
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_long,
        predict=True,
        stop_randomization=True,
    )

    # --- 4) DataLoaders ---
    dl_kwargs = dict(batch_size=batch, num_workers=workers)
    if workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, pin_memory=True))

    train_dl = training.to_dataloader(train=True, **dl_kwargs)
    val_dl   = validation.to_dataloader(train=False, **dl_kwargs)

    if len(val_dl) == 0:
        raise RuntimeError(
            "Validation set is empty. Зміни --enc / --pred / --max_series (наприклад, --enc 96 --pred 24)."
        )

    # --- ранній вихід за запитом ---
    if skip:
        print("Skip requested: пропускаю тренування та інференс (дані й лоадери підготовлено).")
        return

    # --- 5) Модель TFT ---
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,                 # QuantileLoss за замовчуванням (7 квантилів)
        loss=QuantileLoss(),
        log_interval=50,
        reduce_on_plateau_patience=2,
    )

    # --- 6) TensorBoard логер ---
    logger = TensorBoardLogger(save_dir=str(MODELS), name="tft_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        default_root_dir=str(MODELS),
        log_every_n_steps=50,
        logger=logger,
        num_sanity_val_steps=0,   # швидкий старт, без sanity-check
    )

    # --- 7) Навчання ---
    trainer.fit(model, train_dl, val_dl)

    # --- 8) Кращий чекпоінт (за val loss) ---
    best = trainer.checkpoint_callback.best_model_path
    print("Best checkpoint:", best)

    # =========================
    # 9) ІНФЕРЕНС: ВЕСЬ HORIZON
    # =========================
    model = TemporalFusionTransformer.load_from_checkpoint(best)

    # Версії PF можуть повертати (preds, x), (preds, x, index)
    res = model.predict(val_dl, return_x=True, return_index=True)
    preds, xs, index_df = None, None, None

    if isinstance(res, tuple):
        if len(res) == 3:
            preds, xs, index_df = res
        elif len(res) == 2:
            preds, xs = res
        else:
            preds = res[0]
    else:
        preds = res

    preds = _to_numpy(preds)  # очікуємо [N, pred_len] або [N]

    # --- збираємо всі прогнози з часами й серіями у плоскому вигляді ---
    pred_flat = preds.reshape(-1)

    time_flat = None
    sid_flat = None

    # Випадок A: є index_df (кращий шлях)
    if index_df is not None and "time_idx" in index_df.columns:
        time_flat = index_df["time_idx"].to_numpy().reshape(-1)
        if "series_id" in index_df.columns:
            sid_flat = index_df["series_id"].astype(str).to_numpy().reshape(-1)
        else:
            sid_flat = np.array(["ALL"] * len(time_flat))
    else:
        # Випадок B: збираємо з xs (list[dict] або dict)
        times = []
        sids = []
        def _iter_xs(xs_obj):
            if isinstance(xs_obj, list):
                for d in xs_obj:
                    yield d
            elif isinstance(xs_obj, dict):
                yield xs_obj

        if xs is not None:
            for d in _iter_xs(xs):
                if not isinstance(d, dict):
                    continue
                if "decoder_time_idx" in d:
                    ti = _to_numpy(d["decoder_time_idx"]).reshape(-1)
                    times.append(ti)
                    # series_id може бути відсутнім — тоді ставимо "ALL"
                    if "series_id" in d:
                        sid = np.array([str(s) for s in _to_numpy(d["series_id"]).reshape(-1)])
                    else:
                        sid = np.array(["ALL"] * len(ti))
                    sids.append(sid)
            if times:
                time_flat = np.concatenate(times, axis=0)
                sid_flat = np.concatenate(sids, axis=0)
        # якщо xs теж не дав інформацію — згенеруємо таймштампи пізніше з останніх кроків
    # Узгоджуємо довжини
    if time_flat is None:
        # як мінімум спробуємо дістати останній крок часу
        last_times = []
        for batch in val_dl:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, dict) and "decoder_time_idx" in x:
                last_times.append(_to_numpy(x["decoder_time_idx"])[:, -1])
        if last_times:
            time_flat = np.concatenate(last_times, axis=0).reshape(-1)
            # якщо прогнозів більше — обріжемо
            L = min(len(pred_flat), len(time_flat))
            pred_flat = pred_flat[:L]
            time_flat = time_flat[:L]
            sid_flat = np.array(["ALL"] * L)
        else:
            # крайнє fallback: нічого логічного не можемо зібрати
            time_flat = np.arange(len(pred_flat))
            sid_flat = np.array(["ALL"] * len(pred_flat))

    L = min(len(pred_flat), len(time_flat), len(sid_flat))
    pred_flat = pred_flat[:L]
    time_flat = time_flat[:L]
    sid_flat = sid_flat[:L]

    timestamps = pd.to_datetime(t0) + pd.to_timedelta(time_flat, unit="h")
    df_fore = pd.DataFrame(
        {"timestamp": timestamps, "series_id": sid_flat, "yhat": pred_flat}
    ).sort_values(["timestamp", "series_id"])

    # --- додамо y_true для останнього кроку кожного вікна (якщо доступно) ---
    y_true_list = []
    for batch in val_dl:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            yb = _unwrap_target(batch[1])
            y_true_list.append(yb[:, -1].cpu().numpy())
    if y_true_list:
        y_true_flat = np.concatenate(y_true_list).ravel()
        # узгодимо довжину з df_fore
        L2 = min(len(df_fore), len(y_true_flat))
        df_fore = df_fore.iloc[:L2].copy()
        df_fore.loc[:, "y_true"] = y_true_flat[:L2]

    # --- 10) Метрики та збереження ---
    if "y_true" in df_fore.columns:
        m_mae = mae(df_fore["y_true"].values, df_fore["yhat"].values)
        m_rmse = rmse(df_fore["y_true"].values, df_fore["yhat"].values)
        m_mape = mape(df_fore["y_true"].values, df_fore["yhat"].values)
    else:
        m_mae, m_rmse, m_mape = np.nan, np.nan, np.nan

    out_dir = ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_path = out_dir / "transformer_MT_all_forecast.csv"
    met_path = out_dir / "transformer_MT_all_metrics.csv"

    df_fore.to_csv(fc_path, index=False)
    pd.DataFrame([{"MAE": m_mae, "RMSE": m_rmse, "MAPE": m_mape}]).to_csv(met_path, index=False)

    print(f"Saved forecast: {fc_path}")
    print(f"Saved metrics:  {met_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_series", type=int, default=50, help="Скільки серій брати")
    ap.add_argument("--enc", type=int, default=168)
    ap.add_argument("--pred", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0, help="num_workers для DataLoader (Windows: 0-4)")
    ap.add_argument("--skip", action="store_true", help="Пропустити тренування та інференс")
    args = ap.parse_args()
    main(
        max_series=args.max_series,
        max_encoder_length=args.enc,
        max_prediction_length=args.pred,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        skip=args.skip,
    )
