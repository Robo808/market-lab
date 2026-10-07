"""Portfolio backtester: any strategy becomes a weights panel, applied to next-bar returns with costs.

Single-instrument strategies across several symbols are run per symbol and equal-weighted
(1/N each), so one engine covers both kinds. Costs: `spread_bps` charged per unit of turnover
(half on each side is already folded in: 1 unit of turnover = one side), `funding_annual` on gross
exposure (IG overnight financing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import summary
from .strategies import OVERLAYS, get


def build_weights(name: str, frames: dict[str, pd.DataFrame], params: dict | None = None,
                  overlays: dict | None = None) -> pd.DataFrame:
    """Target weights (rows = bars, cols = symbols), decided at each bar's close."""
    s, params, overlays = get(name), params or {}, overlays or {}
    if s.kind == "portfolio":
        panel = pd.DataFrame({k: v["close"] for k, v in frames.items()}).dropna(how="all").ffill()
        w = s.fn(panel, **params)
        for ov, kw in overlays.items():
            if ov == "vol_target":  # portfolio-level: scale the book to the target vol
                w = _portfolio_vol_target(w, panel, **(kw or {}))
        return w
    cols = {}
    for sym, df in frames.items():
        pos = s.fn(df, **params)
        for ov, kw in overlays.items():
            pos = OVERLAYS[ov](pos, df, **(kw or {}))
        cols[sym] = pos
    w = pd.DataFrame(cols).fillna(0.0)
    return w / max(len(frames), 1)


def _portfolio_vol_target(w, panel, target=0.15, n=60, cap=2.0, periods=252):
    r = panel.pct_change()
    port = (w.shift() * r).sum(axis=1)
    rv = port.rolling(n).std() * np.sqrt(periods)
    return w.mul((target / rv).clip(upper=cap).fillna(1.0), axis=0)


def simulate(weights: pd.DataFrame, closes: pd.DataFrame, spread_bps: float = 0.0, funding_annual: float = 0.0,
             periods: int = 252) -> dict:
    closes = closes.reindex(weights.index).ffill()
    rets = closes.pct_change().fillna(0.0)
    held = weights.shift().fillna(0.0)                       # decided at t, earns t+1
    gross = (held * rets).sum(axis=1)
    turnover = weights.diff().fillna(weights).abs().sum(axis=1).shift().fillna(0.0)
    cost = turnover * spread_bps / 1e4
    fund = held.abs().sum(axis=1) * funding_annual / periods
    net = gross - cost - fund
    equity = (1 + net).cumprod()
    nz = np.sign(weights)
    entries = int(((nz != nz.shift().fillna(0)) & (nz != 0)).sum().sum())
    bench = closes.div(closes.bfill().iloc[0]).mean(axis=1)
    return {"returns": net, "equity": equity, "weights": weights, "trades": entries,
            "exposure": float((held.abs().sum(axis=1) > 0).mean()), "turnover_ann": float(turnover.mean() * periods),
            "cost_drag": float((cost + fund).sum()), "stats": summary(equity, periods=periods),
            "benchmark": summary(bench, periods=periods)}


def backtest(name: str, frames: dict[str, pd.DataFrame], params: dict | None = None, overlays: dict | None = None,
             spread_bps: float = 0.0, funding_annual: float = 0.0, periods: int = 252) -> dict:
    w = build_weights(name, frames, params, overlays)
    closes = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    out = simulate(w, closes, spread_bps, funding_annual, periods)
    out.update({"strategy": name, "params": params or {}, "overlays": overlays or {}, "symbols": list(frames)})
    return out


def latest_signal(name: str, frames: dict[str, pd.DataFrame], params: dict | None = None,
                  overlays: dict | None = None) -> pd.DataFrame:
    """Today's target weights and when each last changed (signal mode)."""
    w = build_weights(name, frames, params, overlays)
    last = w.iloc[-1]
    changed = {c: (w[c].ne(w[c].shift())).iloc[::-1].idxmax() if w[c].nunique() > 1 else w.index[0] for c in w}
    closes = {k: v["close"].iloc[-1] for k, v in frames.items()}
    return pd.DataFrame({"target_weight": last, "prev_weight": w.iloc[-2] if len(w) > 1 else 0.0,
                         "since": pd.Series(changed).astype(str).str[:10], "last_close": pd.Series(closes),
                         "as_of": str(w.index[-1])[:10]})
