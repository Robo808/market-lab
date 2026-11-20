"""Context tags stamped on every forward row, so results can be scored by circumstance, not just in aggregate.

Market-wide (as of each bar, carried forward across calendars): S&P 500 trend vs its 200-day average, VIX bucket
and term structure, the desk's index gate (LESSONS L-027: NDX, SOX, BTC above their 20-day low and 50-day
average), 10-year yield direction and the dollar trend. Per instrument: trend vs 50/200-day averages, 20-day
realised-vol percentile over a year, RSI(14) bucket and ATR as a share of price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ta


def _day_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
    out = df.copy()
    out.index = pd.DatetimeIndex(idx).normalize()
    return out[~out.index.duplicated(keep="last")]


def gate_ok(close: pd.Series) -> pd.Series:
    """L-027 gate: close above the lowest close of the prior 20 bars and above the 50-bar average."""
    low20 = close.shift().rolling(20).min()
    return (close > low20) & (close > ta.sma(close, 50))


def market(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily market-wide tags (index = naive dates)."""
    c = {k: _day_index(v)["close"] for k, v in frames.items() if v is not None and len(v)}
    idx = sorted(set().union(*[s.index for s in c.values()])) if c else []
    out = pd.DataFrame(index=pd.DatetimeIndex(idx))

    def put(name, s):
        out[name] = s.reindex(out.index, method="ffill")

    if "^GSPC" in c:
        s = c["^GSPC"]
        put("spx_trend", pd.Series(np.where(s > ta.sma(s, 200), "up", "down"), index=s.index))
    if "^VIX" in c:
        v = c["^VIX"]
        put("vix", v.round(2))
        put("vix_bucket", pd.Series(pd.cut(v, [0, 15, 25, 1e9], labels=["calm", "normal", "stressed"]).astype(str),
                                    index=v.index))
        if "^VIX3M" in c:
            ratio = v / c["^VIX3M"].reindex(v.index, method="ffill")
            put("vix_term", pd.Series(np.where(ratio > 1, "backwardation", "contango"), index=v.index))
    for key, name in (("^NDX", "gate_ndx"), ("^SOX", "gate_sox"), ("crypto:BTC", "gate_btc")):
        if key in c:
            put(name, gate_ok(c[key]))
    if "^TNX" in c:
        t = c["^TNX"]
        put("rates", pd.Series(np.where(t.diff(20) > 0, "rising", "falling"), index=t.index))
    if "DX-Y.NYB" in c:
        d = c["DX-Y.NYB"]
        put("usd", pd.Series(np.where(d > ta.sma(d, 50), "strong", "weak"), index=d.index))
    gates = [g for g in ("gate_ndx", "gate_sox", "gate_btc") if g in out]
    if gates:
        out["gate_all_fail"] = ~out[gates].fillna(True).astype(bool).any(axis=1)
    return out


def instrument(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument tags on its own calendar (index = naive dates)."""
    d = _day_index(df)
    c = d["close"]
    s50, s200 = ta.sma(c, 50), ta.sma(c, 200)
    trend = np.where((c > s50) & (c > s200), "up", np.where((c < s50) & (c < s200), "down", "mixed"))
    rv = c.pct_change().rolling(20).std() * np.sqrt(252)
    pct = rv.rolling(252, min_periods=60).rank(pct=True)
    r = ta.rsi(c, 14)
    atr = ta.atr(d, 14) if {"high", "low"} <= set(d.columns) else c.pct_change().abs().rolling(14).mean() * c
    return pd.DataFrame({
        "trend": trend,
        "rv20": rv.round(4),
        "rv_bucket": pd.cut(pct, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"]).astype(str),
        "rsi_bucket": pd.cut(r, [-1, 30, 70, 101], labels=["oversold", "neutral", "overbought"]).astype(str),
        "atr_pct": (atr / c).round(5),
    }, index=d.index)


def row(mkt: pd.DataFrame, inst: pd.DataFrame | None, day: pd.Timestamp) -> dict:
    """Merge market tags (as of `day`, carried forward) with an instrument's tags on `day`."""
    out = {}
    if mkt is not None and len(mkt):
        m = mkt.loc[:day]
        if len(m):
            out.update({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in m.iloc[-1].items()
                        if not (isinstance(v, float) and np.isnan(v))})
    if inst is not None and day in inst.index:
        out.update({k: v for k, v in inst.loc[day].items() if not (isinstance(v, float) and np.isnan(v))})
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}
