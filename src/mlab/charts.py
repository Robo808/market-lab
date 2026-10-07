"""Charts: interactive Plotly HTML (candles + MAs + Bollinger + volume + RSI + MACD, levels overlay)
and a static PNG via matplotlib for embedding in notes. Outputs go to reports/."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ta
from .config import REPORTS_DIR


def _out(name: str, ext: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return REPORTS_DIR / f"{safe}_{pd.Timestamp.now(tz='UTC'):%Y%m%d}.{ext}"


def candle_html(df: pd.DataFrame, title: str, levels: list[float] | None = None, card=None, path: Path | None = None) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = ta.compute_all(df)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.12, 0.16, 0.17], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=x.index, open=x["open"], high=x["high"], low=x["low"], close=x["close"], name="price"), 1, 1)
    for col, color in (("sma20", "#8888ff"), ("sma50", "#ff9f1c"), ("sma200", "#e71d36")):
        fig.add_trace(go.Scatter(x=x.index, y=x[col], name=col, line=dict(width=1.2, color=color)), 1, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_upper"], name="bb", line=dict(width=0.6, color="gray"), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_lower"], name="bb", line=dict(width=0.6, color="gray"), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.08)", showlegend=False), 1, 1)
    for lv in levels or []:
        fig.add_hline(y=lv, line=dict(dash="dot", width=1, color="#2ec4b6"), row=1, col=1)
    if card is not None:
        fig.add_hrect(y0=card.entry_zone[0], y1=card.entry_zone[1], fillcolor="rgba(46,196,182,0.15)", line_width=0, row=1, col=1)
        fig.add_hline(y=card.stop, line=dict(color="red", width=1.5), annotation_text="stop", row=1, col=1)
        for t in card.targets:
            fig.add_hline(y=t, line=dict(color="green", width=1.2), annotation_text="target", row=1, col=1)
    if x["volume"].notna().any():
        fig.add_trace(go.Bar(x=x.index, y=x["volume"], name="volume", marker_color="rgba(100,100,160,0.5)"), 2, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["rsi14"], name="RSI14", line=dict(color="#6a4c93")), 3, 1)
    fig.add_hline(y=70, line=dict(dash="dot", width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(dash="dot", width=0.8), row=3, col=1)
    fig.add_trace(go.Bar(x=x.index, y=x["hist"], name="MACD hist", marker_color="rgba(120,120,120,0.6)"), 4, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd"], name="MACD", line=dict(width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["signal"], name="signal", line=dict(width=1)), 4, 1)
    src = df.attrs.get("source", "")
    fig.update_layout(title=f"{title}  <sup>{src} · last bar {df.index[-1]}</sup>", xaxis_rangeslider_visible=False,
                      height=900, template="plotly_white", legend=dict(orientation="h"))
    p = path or _out(title, "html")
    fig.write_html(p, include_plotlyjs="cdn")
    return p


def line_png(series: dict[str, pd.Series] | pd.DataFrame, title: str, path: Path | None = None, normalize=False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(series)
    if normalize:
        df = df / df.bfill().iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(ax=ax, lw=1.4)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = path or _out(title, "png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p
