"""Algorithmic trading pattern library.

Two kinds of strategy:
- single: f(df OHLCV, **params) -> position Series in [-1, 1]
- portfolio: f(panel of closes, **params) -> weights DataFrame (rows = dates, cols = symbols)

Every decision uses data up to and including the close of bar t; the engine applies it to the
return of bar t+1, so nothing here may look ahead (tests check this by truncation).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import ta


# ---- helpers ------------------------------------------------------------------------
def _state(enter_long, exit_long, enter_short=None, exit_short=None, hold: int | None = None) -> np.ndarray:
    """Stateful position from boolean entry/exit arrays (long wins ties). hold: exit after N bars."""
    el, xl = np.asarray(enter_long, bool), np.asarray(exit_long, bool)
    n = len(el)
    es = np.zeros(n, bool) if enter_short is None else np.asarray(enter_short, bool)
    xs = np.zeros(n, bool) if exit_short is None else np.asarray(exit_short, bool)
    pos, cur, age = np.zeros(n), 0, 0
    for i in range(n):
        if cur == 1 and (xl[i] or (hold and age >= hold)):
            cur = 0
        elif cur == -1 and (xs[i] or (hold and age >= hold)):
            cur = 0
        if cur == 0:
            if el[i]:
                cur, age = 1, 0
            elif es[i]:
                cur, age = -1, 0
        else:
            age += 1
        pos[i] = cur
    return pos


def _ser(a, df) -> pd.Series:
    return pd.Series(a, index=df.index, dtype=float).fillna(0.0)


def _month_ends(index: pd.DatetimeIndex) -> pd.Series:
    """True on bars where the next business day falls in a new month. Calendar-based, so the
    latest bar is not mistaken for a month-end just because it is the last row."""
    from pandas.tseries.offsets import BDay
    idx = index.tz_localize(None) if index.tz is not None else index
    idx = pd.DatetimeIndex(idx).normalize()
    return pd.Series((idx + BDay(1)).month != idx.month, index=index)


# ---- trend --------------------------------------------------------------------------
def sma_cross(df, fast=50, slow=200, short=False):
    f, s = ta.sma(df["close"], fast), ta.sma(df["close"], slow)
    return _ser(np.where(s.isna(), 0, np.where(f > s, 1, -1 if short else 0)), df)


def ema_cross(df, fast=20, slow=100, short=False):
    f, s = ta.ema(df["close"], fast), ta.ema(df["close"], slow)
    return _ser(np.where(s.isna(), 0, np.where(f > s, 1, -1 if short else 0)), df)


def donchian(df, entry=55, exit=20, short=True):
    """Turtle: enter on an N-bar breakout, exit on an M-bar opposite break."""
    c = df["close"]
    hi_e, lo_e = df["high"].rolling(entry).max().shift(), df["low"].rolling(entry).min().shift()
    hi_x, lo_x = df["high"].rolling(exit).max().shift(), df["low"].rolling(exit).min().shift()
    return _ser(_state(c > hi_e, c < lo_x, (c < lo_e) if short else None, c > hi_x), df)


def supertrend(df, n=10, mult=3.0, short=False):
    t = ta.supertrend(df, n, mult)["trend"]
    return t.clip(lower=-1 if short else 0).fillna(0.0)


def tsmom(df, lookback=252, skip=21, short=True):
    """Time-series momentum (Moskowitz, Ooi, Pedersen 2012): sign of the 12-1 month return."""
    c = df["close"]
    r = c.shift(skip) / c.shift(lookback) - 1
    return _ser(np.where(r.isna(), 0, np.where(r > 0, 1, -1 if short else 0)), df)


def macd_trend(df, fast=12, slow=26, signal=9, short=False):
    m = ta.macd(df["close"], fast, slow, signal)
    up = (m["macd"] > m["signal"]) & (m["macd"] > 0)
    dn = (m["macd"] < m["signal"]) & (m["macd"] < 0)
    return _ser(np.where(up, 1, np.where(dn & short, -1, 0)), df)


def adx_trend(df, n=14, threshold=25, short=False):
    a = ta.adx(df, n)
    strong = a["adx"] > threshold
    up, dn = strong & (a["+di"] > a["-di"]), strong & (a["-di"] > a["+di"])
    return _ser(np.where(up, 1, np.where(dn & short, -1, 0)), df)


def keltner_breakout(df, n=20, k=2.0, short=False):
    kc, c = ta.keltner(df, n, k), df["close"]
    return _ser(_state(c > kc["upper"], c < kc["mid"], (c < kc["lower"]) if short else None, c > kc["mid"]), df)


def squeeze_breakout(df, n=20, bb_k=2.0, kc_k=1.5, short=True):
    """Volatility squeeze (Bollinger inside Keltner), then trade the first close outside the bands."""
    c = df["close"]
    bb, kc = ta.bollinger(c, n, bb_k), ta.keltner(df, n, kc_k)
    squeeze = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    recent = squeeze.rolling(5, min_periods=1).max().astype(bool)
    up, dn = recent & (c > bb["upper"]), recent & (c < bb["lower"])
    mid = bb["mid"]
    return _ser(_state(up, c < mid, dn if short else None, c > mid), df)


# ---- mean reversion -----------------------------------------------------------------
def rsi_reversion(df, low=30, n=14, exit_level=50, trend_filter=200):
    c = df["close"]
    r = ta.rsi(c, n)
    ok = c > ta.sma(c, trend_filter) if trend_filter else pd.Series(True, index=df.index)
    return _ser(_state((r < low) & ok, r > exit_level), df)


def rsi2(df, entry=10, trend=200, exit_sma=5):
    """Connors RSI(2): buy deep short-term oversold in an uptrend, exit on a close above the 5 SMA."""
    c = df["close"]
    r = ta.rsi(c, 2)
    return _ser(_state((r < entry) & (c > ta.sma(c, trend)), c > ta.sma(c, exit_sma)), df)


def bollinger_reversion(df, n=20, k=2.0, short=False):
    c = df["close"]
    bb = ta.bollinger(c, n, k)
    return _ser(_state(c < bb["lower"], c >= bb["mid"], (c > bb["upper"]) if short else None, c <= bb["mid"]), df)


def zscore_reversion(df, n=20, entry=2.0, exit=0.5, short=True):
    c = df["close"]
    z = (c - c.rolling(n).mean()) / c.rolling(n).std()
    return _ser(_state(z < -entry, z > -exit, (z > entry) if short else None, z < exit), df)


def ibs_reversion(df, low=0.2, high=0.8):
    """Internal bar strength (close-low)/(high-low): buy weak closes, sell strong ones."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ibs = ((df["close"] - df["low"]) / rng).fillna(0.5)
    return _ser(_state(ibs < low, ibs > high), df)


# ---- breakout / seasonality ---------------------------------------------------------
def nr7_breakout(df, lookback=7, hold=5, short=True):
    """After the narrowest range of the last 7 bars, go with the break of that bar's high or low."""
    rng = df["high"] - df["low"]
    nr = rng == rng.rolling(lookback).min()
    ref_hi, ref_lo = df["high"].where(nr).ffill(limit=hold), df["low"].where(nr).ffill(limit=hold)
    armed = nr.shift().rolling(hold, min_periods=1).max().fillna(0).astype(bool)
    c = df["close"]
    up, dn = armed & (c > ref_hi.shift()), armed & (c < ref_lo.shift())
    never = np.zeros(len(df), bool)
    return _ser(_state(up, never, dn if short else None, never, hold=hold), df)


def turn_of_month(df, days_before=1, days_after=3):
    """Long from the last N business days of a month through the first M of the next.
    Uses the business-day calendar (not future bars) to know where the next session falls."""
    from pandas.tseries.offsets import BDay, BMonthEnd
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    nxt = pd.DatetimeIndex(idx).normalize() + BDay(1)
    first = nxt.to_period("M").start_time
    last = nxt + BMonthEnd(0)
    pos_from_start = np.array([np.busday_count(f.date(), n.date()) for f, n in zip(first, nxt)])
    pos_from_end = np.array([np.busday_count(n.date(), l.date()) for n, l in zip(nxt, last)])
    return _ser(((pos_from_end < days_before) | (pos_from_start < days_after)).astype(float), df)


# ---- portfolio (panel of closes -> weights) -------------------------------------------
def _monthly_hold(w: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Keep weights only at month-ends, hold them in between."""
    me = _month_ends(panel.index)
    return w.where(me, np.nan).ffill().fillna(0.0)


def xs_momentum(panel, lookback=252, skip=21, top=3, short_bottom=0):
    """Cross-sectional momentum (Jegadeesh-Titman): long the top N by 12-1 return, monthly."""
    mom = panel.shift(skip) / panel.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    n_valid = mom.notna().sum(axis=1)
    w = (rank <= top).astype(float).div(top)
    if short_bottom:
        w -= (rank > n_valid.values[:, None] - short_bottom).astype(float).div(short_bottom)
    w = w.where(mom.notna(), 0.0)
    w[n_valid < top + short_bottom] = 0.0
    return _monthly_hold(w, panel)


def dual_momentum(panel, safe=None, lookback=252):
    """Antonacci: hold the best risky asset if its 12m return beats the safe asset (or 0), else safe."""
    safe = safe or panel.columns[-1]
    if safe not in panel:
        raise ValueError(f"safe asset {safe!r} is not in the panel: {list(panel.columns)}")
    mom = panel / panel.shift(lookback) - 1
    risky = [c for c in panel.columns if c != safe]
    best = mom[risky].fillna(-np.inf).idxmax(axis=1)
    best_r = mom[risky].max(axis=1)
    hurdle = mom[safe].fillna(0).clip(lower=0)
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for d in panel.index:
        if pd.isna(best_r.loc[d]):
            continue
        w.loc[d, best.loc[d] if best_r.loc[d] > hurdle.loc[d] else safe] = 1.0
    return _monthly_hold(w, panel)


def inverse_vol(panel, n=60):
    vol = panel.pct_change().rolling(n).std()
    w = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
    return _monthly_hold(w.fillna(0.0), panel)


def risk_parity(panel, n=126, iters=200):
    """Equal risk contribution weights from the trailing covariance (long-only), monthly."""
    r = panel.pct_change()
    me = _month_ends(panel.index)
    w = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for i, d in enumerate(panel.index):
        if not me.iloc[i] or i < n:
            continue
        cov = r.iloc[i - n + 1:i + 1].cov().values
        if np.isnan(cov).any():
            continue
        x = np.ones(len(cov)) / len(cov)
        for _ in range(iters):  # multiplicative fixed point for ERC
            rc = x * (cov @ x)
            x = x * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
            x /= x.sum()
        w.loc[d] = x
    return w.ffill().fillna(0.0)


def pairs(panel, n=60, entry=2.0, exit=0.5):
    """Two-asset stat-arb: rolling OLS hedge ratio on logs, trade the spread z-score.
    Long spread = +1 unit of A, -beta of B (weights normalised to gross 1)."""
    if panel.shape[1] != 2:
        raise ValueError("pairs needs exactly two symbols")
    la, lb = np.log(panel.iloc[:, 0]), np.log(panel.iloc[:, 1])
    beta = la.rolling(n).cov(lb) / lb.rolling(n).var()
    spread = la - beta * lb
    z = (spread - spread.rolling(n).mean()) / spread.rolling(n).std()
    side = pd.Series(_state(z < -entry, z > -exit, z > entry, z < exit), index=panel.index)
    gross = 1 + beta.abs()
    w = pd.DataFrame({panel.columns[0]: side / gross, panel.columns[1]: -side * beta / gross}, index=panel.index)
    return w.fillna(0.0)


# ---- overlays -----------------------------------------------------------------------
def vol_target(position: pd.Series, df: pd.DataFrame, target=0.15, n=20, cap=2.0, periods=252) -> pd.Series:
    """Scale a position so the instrument's recent realised vol maps to `target` annualised."""
    rv = df["close"].pct_change().rolling(n).std() * np.sqrt(periods)
    return (position * (target / rv).clip(upper=cap)).fillna(0.0)


def regime_filter(position: pd.Series, df: pd.DataFrame, n=200) -> pd.Series:
    """Longs only above the n-SMA, shorts only below."""
    up = df["close"] > ta.sma(df["close"], n)
    return position.where(((position > 0) & up) | ((position < 0) & ~up), 0.0)


# ---- registry -----------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    name: str
    fn: Callable
    kind: str            # single | portfolio
    family: str
    summary: str
    defaults: dict = field(default_factory=dict)


def _s(fn, kind, family, summary):
    import inspect
    d = {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect._empty}
    return Strategy(fn.__name__, fn, kind, family, summary, d)


REGISTRY: dict[str, Strategy] = {s.name: s for s in [
    _s(sma_cross, "single", "trend", "fast SMA above slow SMA"),
    _s(ema_cross, "single", "trend", "fast EMA above slow EMA"),
    _s(donchian, "single", "trend", "Turtle 55-bar breakout, 20-bar exit"),
    _s(supertrend, "single", "trend", "ATR Supertrend direction"),
    _s(tsmom, "single", "trend", "sign of 12-1 month return (MOP 2012)"),
    _s(macd_trend, "single", "trend", "MACD above signal and zero"),
    _s(adx_trend, "single", "trend", "+DI over -DI with ADX above threshold"),
    _s(keltner_breakout, "single", "breakout", "close above upper Keltner, exit at mid"),
    _s(squeeze_breakout, "single", "breakout", "BB inside KC squeeze, trade the release"),
    _s(nr7_breakout, "single", "breakout", "break of the NR7 bar, hold N bars"),
    _s(rsi_reversion, "single", "mean-reversion", "RSI14 oversold in an uptrend"),
    _s(rsi2, "single", "mean-reversion", "Connors RSI(2) < 10 above 200 SMA"),
    _s(bollinger_reversion, "single", "mean-reversion", "close below lower band, exit at mid"),
    _s(zscore_reversion, "single", "mean-reversion", "fade |z| > 2 vs rolling mean"),
    _s(ibs_reversion, "single", "mean-reversion", "internal bar strength < 0.2"),
    _s(turn_of_month, "single", "seasonality", "long last day to 3rd day of month"),
    _s(xs_momentum, "portfolio", "cross-sectional", "long top N by 12-1 return, monthly"),
    _s(dual_momentum, "portfolio", "allocation", "best risky asset if it beats safe, else safe"),
    _s(inverse_vol, "portfolio", "allocation", "weights proportional to 1/vol, monthly"),
    _s(risk_parity, "portfolio", "allocation", "equal risk contribution, monthly"),
    _s(pairs, "portfolio", "stat-arb", "rolling hedge ratio, trade spread z-score"),
]}

OVERLAYS = {"vol_target": vol_target, "regime_filter": regime_filter}


def get(name: str) -> Strategy:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; `mlab algo list` shows {len(REGISTRY)}") from None


def table() -> pd.DataFrame:
    return pd.DataFrame([{"strategy": s.name, "kind": s.kind, "family": s.family, "rule": s.summary,
                          "defaults": ", ".join(f"{k}={v}" for k, v in s.defaults.items())}
                         for s in REGISTRY.values()]).set_index("strategy")
