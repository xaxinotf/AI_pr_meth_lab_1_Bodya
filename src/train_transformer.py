# src/train_transformer.py
import os
os.environ["MPLBACKEND"] = "Agg"
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

logging.getLogger("lightning").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "transformer"
MODELS.mkdir(parents=True, exist_ok=True)

pl.seed_everything(42)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def prepare_long(df_wide: pd.DataFrame):
    df = df_wide.copy()
    df.index.name = "time"
    df = df.reset_index()
    df_long = df.melt(id_vars=["time"], var_name="series_id", value_name="value")
    t0 = pd.to_datetime(df_long["time"].min())
    df_long["time_idx"] = ((df_long["time"] - t0).dt.total_seconds() // 3600).astype(int)
    return df_long, t0


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_target(y):
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y


def _iter_xs_batches(xs_obj):
    if xs_obj is None:
        return
    if isinstance(xs_obj, list):
        for d in xs_obj:
            if isinstance(d, dict):
                yield d
    elif isinstance(xs_obj, dict):
        yield xs_obj


def main(
    series_id_prefix: str = "MT_",
    max_series: int = 50,
    max_encoder_length: int = 168,
    max_prediction_length: int = 24,
    epochs: int = 3,
    batch: int = 128,
    workers: int = 0,
    skip: bool = False,
):
    assert torch.cuda.is_available(), "CUDA недоступна — тренування лише на GPU."

    df = load_ld2011()
    _, ph = save_parquet_versions(df)
    dfw = pd.read_parquet(ph).astype("float32")

    cols = [c for c in dfw.columns if c.startswith(series_id_prefix)][:max_series]
    if not cols:
        raise RuntimeError("Не знайдено жодної серії з вказаним префіксом.")
    dfw = dfw[cols]

    df_long, t0 = prepare_long(dfw)

    last_time = int(df_long["time_idx"].max())
    training_cutoff = last_time - max_prediction_length
    train_df = df_long[df_long["time_idx"] <= training_cutoff].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        min_encoder_length=max(1, min(max_encoder_length, max_encoder_length // 2)),
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

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df_long,
        predict=True,
        stop_randomization=True,
    )

    dl_kwargs = dict(batch_size=batch, num_workers=workers)
    if workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, pin_memory=True))

    train_dl = training.to_dataloader(train=True, **dl_kwargs)
    val_dl = validation.to_dataloader(train=False, **dl_kwargs)

    if len(val_dl) == 0:
        raise RuntimeError("Validation set is empty. Спробуй зменшити --enc або збільшити --pred/--max_series.")

    if skip:
        print("Skip requested: пропускаю тренування та інференс.")
        return

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=50,
        reduce_on_plateau_patience=2,
    )

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
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dl, val_dl)

    best = trainer.checkpoint_callback.best_model_path
    print("Best checkpoint:", best)

    model = TemporalFusionTransformer.load_from_checkpoint(best)

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

    preds_np = _to_numpy(preds)
    if preds_np.ndim == 1:
        preds_np = preds_np[:, None]
    N, P = preds_np.shape[0], preds_np.shape[1]

    records = []

    if index_df is not None and ("time_idx" in index_df.columns):
        time_flat = index_df["time_idx"].to_numpy().reshape(-1)
        if "series_id" in index_df.columns:
            sid_flat = index_df["series_id"].astype(str).to_numpy().reshape(-1)
        else:
            sid_flat = np.array(["ALL"] * len(time_flat))
        pred_flat = preds_np.reshape(-1)
        L = min(len(time_flat), len(sid_flat), len(pred_flat))
        time_flat = time_flat[:L]
        sid_flat = sid_flat[:L]
        pred_flat = pred_flat[:L]
        timestamps = pd.to_datetime(t0) + pd.to_timedelta(time_flat, unit="h")
        df_fore = pd.DataFrame(
            {"timestamp": timestamps, "series_id": sid_flat, "yhat": pred_flat}
        )
    else:
        off = 0
        for d in _iter_xs_batches(xs):
            if "decoder_time_idx" not in d:
                continue
            t = _to_numpy(d["decoder_time_idx"])
            if t.ndim == 1:
                t = t[None, :]
            B, PH = t.shape[0], t.shape[1]
            if off + B > N:
                B = max(0, N - off)
                if B == 0:
                    break
                t = t[:B]
            sids = None
            if "series_id" in d:
                s = _to_numpy(d["series_id"])
                if s.ndim == 1:
                    s = np.repeat(s[:, None], PH, axis=1)
                elif s.shape[1] != PH:
                    s = np.repeat(s[:, :1], PH, axis=1)
                sids = np.array([[str(v) for v in row] for row in s])
            yhat_batch = preds_np[off : off + B]
            for i in range(B):
                for j in range(min(PH, yhat_batch.shape[1])):
                    ti = int(t[i, j])
                    si = sids[i, j] if sids is not None else "ALL"
                    records.append(
                        {
                            "timestamp": pd.to_datetime(t0) + pd.to_timedelta(ti, unit="h"),
                            "series_id": si,
                            "yhat": float(yhat_batch[i, j]),
                        }
                    )
            off += B

        if not records:
            last_times = []
            for batch in val_dl:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(x, dict) and "decoder_time_idx" in x:
                    ti = _to_numpy(x["decoder_time_idx"])
                    if ti.ndim == 1:
                        ti = ti[None, :]
                    last_times.append(ti)
            if last_times:
                all_t = np.concatenate(last_times, axis=0).reshape(-1)
                pred_flat = preds_np.reshape(-1)
                L = min(len(all_t), len(pred_flat))
                timestamps = pd.to_datetime(t0) + pd.to_timedelta(all_t[:L], unit="h")
                df_fore = pd.DataFrame(
                    {"timestamp": timestamps, "series_id": "ALL", "yhat": pred_flat[:L]}
                )
            else:
                print("⚠️ Warning: records are empty — no predictions generated")
                out_dir = ROOT / "reports" / "tables"
                out_dir.mkdir(parents=True, exist_ok=True)
                fc_path = out_dir / "transformer_MT_all_forecast.csv"
                met_path = out_dir / "transformer_MT_all_metrics.csv"
                pd.DataFrame(columns=["timestamp", "series_id", "yhat"]).to_csv(fc_path, index=False)
                pd.DataFrame([{"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}]).to_csv(met_path, index=False)
                print(f"Saved forecast: {fc_path}")
                print(f"Saved metrics:  {met_path}")
                return
        else:
            df_fore = pd.DataFrame.from_records(records)

    sort_cols = [c for c in ["series_id", "timestamp"] if c in df_fore.columns]
    if sort_cols:
        df_fore = df_fore.sort_values(sort_cols)

    y_true_list = []
    for batch in val_dl:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            yb = _unwrap_target(batch[1])
            # yb може бути [B, pred_len] або [B]
            yb = yb.detach().cpu().numpy()
            y_true_list.append(yb.reshape(-1))  # розгортаємо на всі кроки
    if y_true_list:
        y_true_flat = np.concatenate(y_true_list, axis=0)
        L2 = min(len(df_fore), len(y_true_flat))
        df_fore = df_fore.iloc[:L2].copy()
        df_fore.loc[:, "y_true"] = y_true_flat[:L2]

    if "y_true" in df_fore.columns:
        m_mae = mae(df_fore["y_true"].values, df_fore["yhat"].values)
        m_rmse = rmse(df_fore["y_true"].values, df_fore["yhat"].values)
        m_mape = mape(df_fore["y_true"].values, df_fore["yhat"].values)
    else:
        m_mae = np.nan
        m_rmse = np.nan
        m_mape = np.nan

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
    ap.add_argument("--max_series", type=int, default=50)
    ap.add_argument("--enc", type=int, default=168)
    ap.add_argument("--pred", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip", action="store_true")
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
