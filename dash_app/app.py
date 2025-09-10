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
    Усунути дублікати за timestamp: якщо series_id НЕМає — агрегуємо mean по числових колонках.
    Якщо series_id Є — НІЧОГО не агрегуємо, повертаємо дані по кожній серії.
    """
    if df is None or df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # >>> ключова правка А: якщо є series_id, не агрегуємо!
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
    Шукає й завантажує (за першим знайденим файлом) пари:
      {model}_*_forecast.csv та {model}_*_metrics.csv
    Очікувані колонки у forecast: timestamp, y_true, yhat
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
    return out[keep]

def overlay_figure(df_list):
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
            # центральна лінія
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat_p50"],
                mode="lines", name=f"{name} (p50)", line=dict(width=2)
            ))
            # «віяло» p10–p90
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

# ---------- load on startup ----------
baseline_fore, baseline_metr = load_single_model_pair("Baseline")
lstm_fore, lstm_metr         = load_single_model_pair("LSTM")
tft_fore                     = load_transformer_forecast()
tft_metr                     = load_transformer_metrics()

metrics_df = merge_metrics(baseline_metr, lstm_metr, tft_metr)
messages = warn_missing(baseline_fore, baseline_metr, lstm_fore, lstm_metr, tft_fore, tft_metr)

# ---------- Dash app ----------
app = Dash(__name__)
app.title = "Load Forecast: Baseline vs LSTM vs Transformer"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Electricity Load Forecast — Baseline vs LSTM vs Transformer"),
        html.Div(
            [html.Div(m) for m in messages],
            style={"color": "#b45309", "marginBottom": "8px"} if messages else {}
        ),
        dcc.Tabs(id="tabs", value="tab-compare", children=[
            dcc.Tab(label="Compare (overlay)", value="tab-compare"),
            dcc.Tab(label="Side-by-side", value="tab-side"),
            dcc.Tab(label="Metrics", value="tab-metrics"),
        ]),
        html.Div(id="tab-content")
    ]
)

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    if tab == "tab-compare":
        fig = overlay_figure([baseline_fore, lstm_fore, tft_fore])
        return dcc.Graph(figure=fig, style={"height": "70vh"})

    elif tab == "tab-side":
        fig1 = one_panel_figure(baseline_fore, "Baseline")
        fig2 = one_panel_figure(lstm_fore, "LSTM")
        fig3 = one_panel_figure(tft_fore, "Transformer")
        return html.Div([
            dcc.Graph(figure=fig1, style={"height": "28vh"}),
            dcc.Graph(figure=fig2, style={"height": "28vh"}),
            dcc.Graph(figure=fig3, style={"height": "28vh"}),
        ])

    else:  # tab-metrics
        if metrics_df is None or metrics_df.empty:
            return html.Div("Немає метрик для відображення.")
        table = dash_table.DataTable(
            data=metrics_df.round(4).to_dict("records"),
            columns=[{"name": c, "id": c} for c in metrics_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontFamily": "Inter, system-ui, sans-serif", "fontSize": "14px"},
            style_header={"fontWeight": "bold"},
        )
        melt = metrics_df.melt(
            id_vars="model",
            value_vars=[c for c in ["MAE", "RMSE", "MAPE"] if c in metrics_df.columns],
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
        return html.Div([
            html.Div(table, style={"marginBottom": "16px"}),
            dcc.Graph(figure=figm, style={"height": "50vh"})
        ])

if __name__ == "__main__":
    # Dash >= 2.17
    app.run(debug=True, host="127.0.0.1", port=8050)
