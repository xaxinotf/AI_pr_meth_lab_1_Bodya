import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np

import dash
from dash import Dash, html, dcc, Output, Input
from dash import dash_table
import plotly.graph_objects as go

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "tables"

# ---------- helpers ----------

def tidy_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводимо до tidy-формату й сортуємо по часу.
    Якщо є series_id — нічого не агрегуємо, повертаємо по серіях.
    Якщо series_id немає — усереднюємо дублікати за timestamp по числових колонках.
    """
    if df is None or df.empty:
        return df
    if "timestamp" not in df.columns:
        return df

    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "series_id" in df.columns:
        return df.sort_values(["series_id", "timestamp"])

    cols = [c for c in ["y_true", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"] if c in df.columns]
    if not cols:
        return df.sort_values("timestamp")

    df = (
        df.groupby("timestamp", as_index=False)[cols].mean()
          .sort_values("timestamp")
    )
    return df


def _find_first(glob_pattern: Path | str) -> str | None:
    files = sorted(glob.glob(str(glob_pattern)))
    return files[0] if files else None


def load_single_model_pair(model_name: str):
    """
    Шукає:
      {model}_*_forecast.csv та {model}_*_metrics.csv
    Повертає (df_forecast, df_metrics) або (None, None) якщо не знайдено.
    """
    f_fore = _find_first(REPORTS / f"{model_name.lower()}_*_forecast.csv")
    f_metr = _find_first(REPORTS / f"{model_name.lower()}_*_metrics.csv")

    df_fore, df_metr = None, None

    if f_fore and os.path.exists(f_fore):
        df_fore = pd.read_csv(f_fore)
        df_fore = tidy_forecast(df_fore)
        df_fore["model"] = model_name

    if f_metr and os.path.exists(f_metr):
        df_metr = pd.read_csv(f_metr)
        df_metr["model"] = model_name

    return df_fore, df_metr


def load_transformer_forecast():
    f = REPORTS / "transformer_MT_all_forecast.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = tidy_forecast(df)
    df["model"] = "Transformer"
    return df


def load_transformer_metrics():
    f = REPORTS / "transformer_MT_all_metrics.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    m["model"] = "Transformer"
    return m


def merge_metrics(*dfs):
    frames = [d for d in dfs if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE"])
    out = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["model", "MAE", "RMSE", "MAPE"] if c in out.columns]
    return out[keep].copy()


def overlay_figure(df_list, title=""):
    fig = go.Figure()

    # Actual
    actual_added = False
    for df in df_list:
        if df is None or df.empty:
            continue
        if (not actual_added) and ("y_true" in df.columns):
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["y_true"],
                mode="lines", name="Actual", line=dict(width=2)
            ))
            actual_added = True

    # Predictions
    for df in df_list:
        if df is None or df.empty:
            continue
        name = df["model"].iloc[0] if "model" in df.columns and len(df) else "Model"
        if all(c in df.columns for c in ["yhat_p10", "yhat_p50", "yhat_p90"]):
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat_p50"],
                mode="lines", name=f"{name} (p50)", line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=list(df["timestamp"]) + list(df["timestamp"])[::-1],
                y=list(df["yhat_p90"]) + list(df["yhat_p10"])[::-1],
                fill="toself", fillcolor="rgba(0,0,0,0.08)",
                line=dict(width=0), name=f"{name} (p10–p90)"
            ))
        elif "yhat" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat"],
                mode="lines", name=name, line=dict(width=2)
            ))

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Time", yaxis_title="Load",
        legend=dict(orientation="h", y=1.15),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def one_panel_figure(df, title):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title=f"{title} (no data)", template="plotly_white")
        return fig

    if "y_true" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["y_true"],
            mode="lines", name="Actual", line=dict(width=2)
        ))

    if all(c in df.columns for c in ["yhat_p10", "yhat_p50", "yhat_p90"]):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_p50"], mode="lines",
            name="Pred (p50)", line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(df["timestamp"]) + list(df["timestamp"])[::-1],
            y=list(df["yhat_p90"]) + list(df["yhat_p10"])[::-1],
            fill="toself", fillcolor="rgba(0,0,0,0.08)",
            line=dict(width=0), name="p10–p90"
        ))
    elif "yhat" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat"],
            mode="lines", name="Pred", line=dict(width=2)
        ))

    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Time", yaxis_title="Load",
        hovermode="x unified",
    )
    return fig


def warn_missing(baseline_fore, baseline_metr, lstm_fore, lstm_metr, tft_fore, tft_metr):
    msgs = []
    if baseline_fore is None:
        msgs.append("⚠️ Baseline: не знайдено forecast CSV (reports/tables/baseline_*_forecast.csv).")
    if baseline_metr is None:
        msgs.append("⚠️ Baseline: не знайдено metrics CSV (reports/tables/baseline_*_metrics.csv).")
    if lstm_fore is None:
        msgs.append("⚠️ LSTM: не знайдено forecast CSV (reports/tables/lstm_*_forecast.csv).")
    if lstm_metr is None:
        msgs.append("⚠️ LSTM: не знайдено metrics CSV (reports/tables/lstm_*_metrics.csv).")
    if tft_fore is None:
        msgs.append("⚠️ Transformer: не знайдено transformer_MT_all_forecast.csv.")
    if tft_metr is None:
        msgs.append("⚠️ Transformer: не знайдено transformer_MT_all_metrics.csv.")
    return msgs


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp, (series_id?), model, residual = y_true - yhat_or_p50"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "yhat" in out.columns:
        pred = out["yhat"]
    elif "yhat_p50" in out.columns:
        pred = out["yhat_p50"]
    else:
        return pd.DataFrame()
    if "y_true" not in out.columns:
        return pd.DataFrame()
    out["residual"] = out["y_true"] - pred
    keep = ["timestamp", "model", "residual"]
    if "series_id" in out.columns:
        keep.insert(1, "series_id")
    return out[keep].dropna()


def coverage_ratio(df: pd.DataFrame) -> float | None:
    """Частка, де y_true між p10 та p90. Якщо немає інтервалів — None."""
    if df is None or df.empty:
        return None
    need = all(c in df.columns for c in ["y_true", "yhat_p10", "yhat_p90"])
    if not need:
        return None
    ok = (df["y_true"] >= df["yhat_p10"]) & (df["y_true"] <= df["yhat_p90"])
    return float(ok.mean()) if len(ok) else None


def has_any_quantiles(dfs: list[pd.DataFrame]) -> bool:
    return any(
        df is not None and not df.empty and all(c in df.columns for c in ["yhat_p10", "yhat_p50", "yhat_p90"])
        for df in dfs
    )


def available_series_ids(dfs: list[pd.DataFrame]) -> list:
    ids = set()
    for df in dfs:
        if df is not None and not df.empty and "series_id" in df.columns:
            ids.update(df["series_id"].dropna().unique().tolist())
    return sorted(ids)


def filter_df(df, chosen_series, start_date, end_date):
    if df is None or df.empty:
        return df
    out = df.copy()
    if "series_id" in out.columns and chosen_series not in (None, "ALL"):
        out = out[out["series_id"] == chosen_series]
    if start_date is not None:
        out = out[out["timestamp"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out["timestamp"] <= pd.to_datetime(end_date)]
    return out


def _smooth(df: pd.DataFrame, win: int) -> pd.DataFrame:
    if df is None or df.empty or win <= 1:
        return df
    out = df.copy()
    num_cols = [c for c in out.select_dtypes(include=[np.number]).columns]
    for c in num_cols:
        out[c] = out[c].rolling(int(win), min_periods=1).mean()
    return out


def rolling_metrics(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Ковзні MAE/RMSE по часу для однієї моделі."""
    if df is None or df.empty or "y_true" not in df.columns:
        return pd.DataFrame()
    if "yhat" in df.columns:
        pred = df["yhat"]
    elif "yhat_p50" in df.columns:
        pred = df["yhat_p50"]
    else:
        return pd.DataFrame()
    e = df["y_true"] - pred
    rm = pd.DataFrame({
        "timestamp": df["timestamp"],
        "MAE": e.abs().rolling(window, min_periods=1).mean(),
        "RMSE": np.sqrt((e**2).rolling(window, min_periods=1).mean())
    })
    return rm


def residual_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot |residual| середня: weekday × hour."""
    res = compute_residuals(df)
    if res.empty:
        return pd.DataFrame()
    tmp = res.copy()
    tmp["abs_resid"] = tmp["residual"].abs()
    t = df.merge(tmp[["timestamp", "abs_resid"]], on="timestamp", how="left")
    t["weekday"] = t["timestamp"].dt.weekday
    t["hour"] = t["timestamp"].dt.hour
    g = t.groupby(["weekday", "hour"], as_index=False)["abs_resid"].mean()
    pivot = g.pivot(index="weekday", columns="hour", values="abs_resid").sort_index()
    return pivot


def acf(residuals: pd.Series, max_lag: int) -> pd.DataFrame:
    """Проста ACF для залишків (lag=1..max_lag)."""
    r = residuals.dropna().values
    n = len(r)
    if n < 2:
        return pd.DataFrame()
    r = r - r.mean()
    denom = np.sum(r**2)
    out_lags, out_vals = [], []
    for lag in range(1, max_lag + 1):
        num = np.sum(r[lag:] * r[:-lag])
        out_lags.append(lag)
        out_vals.append(num / denom if denom != 0 else np.nan)
    return pd.DataFrame({"lag": out_lags, "acf": out_vals})


# ---------- load on startup ----------
baseline_fore, baseline_metr = load_single_model_pair("Baseline")
lstm_fore, lstm_metr         = load_single_model_pair("LSTM")
tft_fore                     = load_transformer_forecast()
tft_metr                     = load_transformer_metrics()

metrics_df = merge_metrics(baseline_metr, lstm_metr, tft_metr)
messages = warn_missing(baseline_fore, baseline_metr, lstm_fore, lstm_metr, tft_fore, tft_metr)

model_frames = {"Baseline": baseline_fore, "LSTM": lstm_fore, "Transformer": tft_fore}
available_models = [m for m, df in model_frames.items() if df is not None and not df.empty]
series_choices = available_series_ids([baseline_fore, lstm_fore, tft_fore])
show_coverage_tab = has_any_quantiles([baseline_fore, lstm_fore, tft_fore])

# ---------- Dash app ----------
app = Dash(__name__)
app.title = "Load Forecast: Baseline vs LSTM vs Transformer"

def tabs_def():
    tabs = [
        dcc.Tab(label="Compare (overlay)", value="tab-compare"),
        dcc.Tab(label="Side-by-side", value="tab-side"),
        dcc.Tab(label="Metrics", value="tab-metrics"),
        dcc.Tab(label="Errors (residuals)", value="tab-errors"),
        dcc.Tab(label="Distribution", value="tab-dist"),
        dcc.Tab(label="Scatter y_true vs yhat", value="tab-scatter"),
        dcc.Tab(label="Rolling metrics", value="tab-rolling"),
        dcc.Tab(label="Residuals: calendar heatmap", value="tab-cal"),
        dcc.Tab(label="Residuals: weekday boxplots", value="tab-box"),
    ]
    if show_coverage_tab:
        tabs.append(dcc.Tab(label="Coverage (p10–p90)", value="tab-coverage"))
    return tabs

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px", "fontFamily": "Inter, system-ui, sans-serif"},
    children=[
        html.H2("Electricity Load Forecast — Baseline vs LSTM vs Transformer"),
        html.Div([html.Div(m) for m in messages],
                 style={"color": "#b45309", "marginBottom": "8px"} if messages else {}),
        # --- controls ---
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "12px", "alignItems": "end",
                   "marginBottom": "12px"},
            children=[
                html.Div([
                    html.Label("Models"),
                    dcc.Checklist(
                        id="models-ckl",
                        options=[{"label": m, "value": m} for m in ["Baseline", "LSTM", "Transformer"]],
                        value=available_models, inline=True
                    )
                ]),
                html.Div([
                    html.Label("Series (optional)"),
                    dcc.Dropdown(
                        id="series-dd",
                        options=([{"label": "ALL", "value": "ALL"}] + [{"label": str(s), "value": s} for s in series_choices]) if series_choices else [{"label": "ALL", "value": "ALL"}],
                        value="ALL", clearable=False
                    )
                ]),
                html.Div([
                    html.Label("Date range"),
                    dcc.DatePickerRange(id="date-range", display_format="YYYY-MM-DD")
                ]),
                html.Div([
                    html.Label("Smoothing (rolling window)"),
                    dcc.Slider(id="smooth-win", min=1, max=24, step=1, value=1,
                               tooltip={"always_visible": False, "placement": "bottom"})
                ]),
                html.Div([
                    html.Label("Rolling window (metrics)"),
                    dcc.Slider(id="roll-win", min=4, max=288, step=1, value=48,
                               tooltip={"always_visible": False, "placement": "bottom"})
                ]),
                html.Div([
                    html.Label("ACF max lag"),
                    dcc.Slider(id="acf-lag", min=6, max=336, step=1, value=96,
                               tooltip={"always_visible": False, "placement": "bottom"})
                ]),
            ]
        ),
        dcc.Tabs(id="tabs", value="tab-compare", children=tabs_def()),
        html.Div(id="tab-content")
    ]
)

# ---------- callbacks ----------

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("models-ckl", "value"),
    Input("series-dd", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("smooth-win", "value"),
    Input("roll-win", "value"),
    Input("acf-lag", "value"),
)
def render_tab(tab, models_selected, series_id, start_date, end_date, smooth_win, roll_win, acf_lag):
    # відфільтруємо + згладимо
    dfs = []
    per_model = {}
    for m in ["Baseline", "LSTM", "Transformer"]:
        df = model_frames.get(m)
        if df is None or df.empty or (m not in models_selected):
            continue
        f = filter_df(df, series_id, start_date, end_date)
        f = _smooth(f, int(smooth_win or 1))
        dfs.append(f)
        per_model[m] = f

    if tab == "tab-compare":
        return dcc.Graph(figure=overlay_figure(dfs, title="Overlay comparison"), style={"height": "70vh"})

    elif tab == "tab-side":
        blocks = []
        for m in ["Baseline", "LSTM", "Transformer"]:
            df = per_model.get(m)
            if df is None:
                continue
            blocks.append(dcc.Graph(figure=one_panel_figure(df, m), style={"height": "28vh"}))
        return html.Div(blocks) if blocks else html.Div("Немає даних для відображення.")

    elif tab == "tab-metrics":
        if metrics_df is None or metrics_df.empty:
            return html.Div("Немає метрик для відображення.")
        mshow = metrics_df[metrics_df["model"].isin(models_selected)] if models_selected else metrics_df
        # форматування MAPE як відсотків до 1 знаку
        mshow_fmt = mshow.copy()
        if "MAPE" in mshow_fmt.columns:
            mshow_fmt["MAPE"] = (mshow_fmt["MAPE"] * 100).round(1)
        table = dash_table.DataTable(
            data=mshow_fmt.round(4).to_dict("records"),
            columns=[{"name": c, "id": c} for c in mshow_fmt.columns],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontSize": "14px"},
            style_header={"fontWeight": "bold"},
        )
        melt = mshow.melt(
            id_vars="model",
            value_vars=[c for c in ["MAE", "RMSE", "MAPE"] if c in mshow.columns],
            var_name="metric", value_name="value"
        )
        figm = go.Figure()
        for metric in melt["metric"].unique():
            sub = melt[melt["metric"] == metric]
            figm.add_trace(go.Bar(x=sub["model"], y=sub["value"], name=metric))
        figm.update_layout(
            barmode="group", template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis_title="Value"
        )
        return html.Div([html.Div(table, style={"marginBottom": "16px"}), dcc.Graph(figure=figm, style={"height": "50vh"})])

    elif tab == "tab-errors":
        figs, rows = [], []
        for m, df in per_model.items():
            res = compute_residuals(df)
            if res.empty:
                continue
            f = go.Figure()
            f.add_trace(go.Scatter(x=res["timestamp"], y=res["residual"], mode="lines", name=f"{m} residual"))
            f.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                            xaxis_title="Time", yaxis_title="Residual (y_true - yhat)")
            figs.append(dcc.Graph(figure=f, style={"height": "35vh"}))
            # зірочкою — перераховано на поточному фільтрі
            mae = float(np.mean(np.abs(res["residual"])))
            rmse = float(np.sqrt(np.mean(np.square(res["residual"]))))
            mape = float(np.mean(np.abs(res["residual"] / df["y_true"].replace(0, np.nan)))) if "y_true" in df.columns else np.nan
            rows.append({"model": m, "MAE*": mae, "RMSE*": rmse, "MAPE*": mape})
        note = html.Div("(*) метрики перераховано на вибраному фільтрі.", style={"color": "#475569", "marginBottom": "8px"})
        table = dash_table.DataTable(
            data=pd.DataFrame(rows).round(4).to_dict("records") if rows else [],
            columns=[{"name": c, "id": c} for c in (["model","MAE*","RMSE*","MAPE*"])],
            style_cell={"padding": "6px", "fontSize": "14px"},
            style_header={"fontWeight": "bold"},
        )
        return html.Div([note, html.Div(table, style={"marginBottom": "12px"}), *figs]) if figs else html.Div("Немає залишків для побудови.")

    elif tab == "tab-dist":
        fig = go.Figure()
        added = False
        for m, df in per_model.items():
            res = compute_residuals(df)
            if res.empty:
                continue
            fig.add_trace(go.Histogram(x=res["residual"], nbinsx=60, name=m, opacity=0.55))
            added = True
        if not added:
            return html.Div("Немає даних для гістограм.")
        fig.update_layout(barmode="overlay", template="plotly_white",
                          margin=dict(l=40, r=20, t=40, b=40),
                          xaxis_title="Residual", yaxis_title="Count")
        return dcc.Graph(figure=fig, style={"height": "70vh"})

    elif tab == "tab-scatter":
        fig = go.Figure()
        added = False
        for m, df in per_model.items():
            if df is None or df.empty or "y_true" not in df.columns:
                continue
            if "yhat" in df.columns:
                pred = df["yhat"]
            elif "yhat_p50" in df.columns:
                pred = df["yhat_p50"]
            else:
                continue
            fig.add_trace(go.Scatter(
                x=df["y_true"], y=pred, mode="markers", name=m,
                marker=dict(size=5, opacity=0.6)
            ))
            added = True
        if not added:
            return html.Div("Немає даних для розсіювання.")
        # лінія y=x
        all_true = pd.concat([per_model[m]["y_true"] for m in per_model if "y_true" in per_model[m].columns])
        if not all_true.empty:
            lo, hi = float(np.nanmin(all_true)), float(np.nanmax(all_true))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(dash="dash")))
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                          xaxis_title="y_true", yaxis_title="yhat / p50", hovermode="closest")
        return dcc.Graph(figure=fig, style={"height": "70vh"})

    elif tab == "tab-rolling":
        win = int(roll_win or 48)
        fig = go.Figure()
        added = False
        for m, df in per_model.items():
            rm = rolling_metrics(df, win)
            if rm.empty:
                continue
            fig.add_trace(go.Scatter(x=rm["timestamp"], y=rm["MAE"], mode="lines", name=f"{m} MAE"))
            fig.add_trace(go.Scatter(x=rm["timestamp"], y=rm["RMSE"], mode="lines", name=f"{m} RMSE"))
            added = True
        if not added:
            return html.Div("Немає даних для обчислення ковзних метрик.")
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                          xaxis_title="Time", yaxis_title=f"Rolling (win={win})")
        return dcc.Graph(figure=fig, style={"height": "70vh"})

    elif tab == "tab-cal":
        figs = []
        for m, df in per_model.items():
            pivot = residual_calendar(df)
            if pivot.empty:
                continue
            heat = go.Figure(data=go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                coloraxis="coloraxis",
                hovertemplate="hour=%{x}, weekday=%{y}<br>|resid|=%{z:.3f}<extra></extra>"
            ))
            heat.update_layout(title=f"{m}: |residual| by weekday×hour", template="plotly_white",
                               margin=dict(l=40, r=20, t=40, b=40), coloraxis=dict(colorbar_title="|resid|"))
            figs.append(dcc.Graph(figure=heat, style={"height": "55vh"}))
        return html.Div(figs) if figs else html.Div("Немає даних для heatmap.")

    elif tab == "tab-box":
        fig = go.Figure()
        added = False
        weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        for m, df in per_model.items():
            res = compute_residuals(df)
            if res.empty:
                continue
            tmp = res.copy()
            tmp["abs_resid"] = tmp["residual"].abs()
            tmp["weekday"] = pd.to_datetime(tmp["timestamp"]).dt.weekday
            fig.add_trace(go.Box(
                x=[weekdays[w] for w in tmp["weekday"]],
                y=tmp["abs_resid"],
                name=m,
                boxmean=True
            ))
            added = True
        if not added:
            return html.Div("Немає даних для boxplot.")
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                          xaxis_title="Weekday", yaxis_title="|residual|")
        return dcc.Graph(figure=fig, style={"height": "70vh"})

    else:  # tab-coverage (показуємо тільки якщо є квантілі; табу може не бути)
        rows, bars_x, bars_y = [], [], []
        for m, df in per_model.items():
            r = coverage_ratio(df)
            if r is None:
                continue
            rows.append({"model": m, "coverage_p10_p90": r})
            bars_x.append(m); bars_y.append(r)
        if not rows:
            return html.Div("Для coverage потрібні інтервальні прогнози (p10/p90).")
        table = dash_table.DataTable(
            data=pd.DataFrame(rows).round(4).to_dict("records"),
            columns=[{"name": "model", "id": "model"}, {"name": "coverage_p10_p90", "id": "coverage_p10_p90"}],
            style_cell={"padding": "6px", "fontSize": "14px"},
            style_header={"fontWeight": "bold"},
        )
        fig = go.Figure(go.Bar(x=bars_x, y=bars_y, name="Coverage"))
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
                          yaxis_title="Share within [p10, p90]", yaxis=dict(tickformat=".0%"))
        return html.Div([html.Div(table, style={"marginBottom": "12px"}), dcc.Graph(figure=fig, style={"height": "50vh"})])

if __name__ == "__main__":
    # Dash >= 2.17
    app.run(debug=True, host="127.0.0.1", port=8050)
