"""The full Investopedia technical-analysis catalogue on top of ta.py: the remaining indicators, candlestick
patterns, rule-based chart patterns on causal zigzag pivots, alternative chart types (Heikin Ashi, Renko,
point & figure), level sets (pivots, market profile) and a CATALOG registry the agent can list and explain.

Conventions: every function takes an OHLCV frame (lowercase columns) or a close Series, is pure pandas/numpy,
and never looks ahead unless its docstring says so (Renko/P&F box sizes and gap fill status are the exceptions)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _swv

from . import ta
from .ta import atr, ema, sma, true_range, wilder


# ---- helpers ----------------------------------------------------------------------
def _roll(s: pd.Series, n: int, f) -> pd.Series:
    """Apply a vectorised function to (len-n+1, n) trailing windows; result aligned to the window end."""
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        out[n - 1:] = f(_swv(v, n))
    return pd.Series(out, index=s.index)


def _hlcov(df):
    return df["high"], df["low"], df["close"], df["open"], df["volume"].fillna(0)


def _intraday(df) -> bool:
    return len(df) > 2 and pd.Series(df.index).diff().median() < pd.Timedelta(hours=20)


# ---- moving averages -----------------------------------------------------------------
def wma(s: pd.Series, n: int = 20) -> pd.Series:
    """Linearly weighted moving average (newest weight n)."""
    w = np.arange(1, n + 1, dtype=float)
    return _roll(s, n, lambda x: x @ w / w.sum())


def hma(s: pd.Series, n: int = 20) -> pd.Series:
    """Hull MA: WMA(2*WMA(n/2) - WMA(n), sqrt n)."""
    return wma(2 * wma(s, max(n // 2, 1)) - wma(s, n), max(int(np.sqrt(n)), 1))


def dema(s: pd.Series, n: int = 20) -> pd.Series:
    e = ema(s, n)
    return 2 * e - ema(e, n)


def tema(s: pd.Series, n: int = 20) -> pd.Series:
    e1 = ema(s, n)
    e2 = ema(e1, n)
    return 3 * e1 - 3 * e2 + ema(e2, n)


def zlema(s: pd.Series, n: int = 20) -> pd.Series:
    """Zero-lag EMA: EMA of price plus its momentum over (n-1)/2 bars."""
    lag = (n - 1) // 2
    return ema(s + (s - s.shift(lag)), n)


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive MA: smoothing constant scaled by the efficiency ratio."""
    c = close.to_numpy(float)
    er = ((close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum().replace(0, np.nan)).fillna(0).to_numpy()
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n] = c[n]
        for i in range(n + 1, len(c)):
            out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def envelopes(close: pd.Series, n: int = 20, pct: float = 0.025) -> pd.DataFrame:
    m = sma(close, n)
    return pd.DataFrame({"mid": m, "upper": m * (1 + pct), "lower": m * (1 - pct)})


# ---- momentum -----------------------------------------------------------------------
def aroon(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Aroon up/down = 100*(n - bars since n-bar high/low)/n; oscillator = up - down."""
    def since(w):  # bars since the latest extreme in each window (ties -> most recent)
        return w[:, ::-1]
    up = _roll(df["high"], n + 1, lambda w: 100 * (n - np.argmax(since(w), axis=1)) / n)
    dn = _roll(df["low"], n + 1, lambda w: 100 * (n - np.argmin(since(w), axis=1)) / n)
    return pd.DataFrame({"up": up, "down": dn, "osc": up - dn})


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    md = _roll(tp, n, lambda w: np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1))
    return (tp - sma(tp, n)) / (0.015 * md.replace(0, np.nan))


def coppock(close: pd.Series, r1: int = 14, r2: int = 11, n: int = 10) -> pd.Series:
    return wma(100 * (ta.roc(close, r1) + ta.roc(close, r2)), n)


def elder_ray(df: pd.DataFrame, n: int = 13) -> pd.DataFrame:
    e = ema(df["close"], n)
    return pd.DataFrame({"bull": df["high"] - e, "bear": df["low"] - e})


def kst(close: pd.Series, signal: int = 9) -> pd.DataFrame:
    """Know Sure Thing: weighted sum of smoothed ROC(10,15,20,30)."""
    k = sum(w * sma(100 * ta.roc(close, r), s) for r, s, w in ((10, 10, 1), (15, 10, 2), (20, 10, 3), (30, 15, 4)))
    return pd.DataFrame({"kst": k, "signal": sma(k, signal)})


def _pct_osc(s: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    line = 100 * (ema(s, fast) - ema(s, slow)) / ema(s, slow).replace(0, np.nan)
    sig = ema(line, signal)
    return pd.DataFrame({"line": line, "signal": sig, "hist": line - sig})


def ppo(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    return _pct_osc(close, fast, slow, signal)


def trix(close: pd.Series, n: int = 15, signal: int = 9) -> pd.DataFrame:
    t = 100 * ema(ema(ema(close, n), n), n).pct_change()
    return pd.DataFrame({"trix": t, "signal": ema(t, signal)})


def tsi(close: pd.Series, r: int = 25, s: int = 13, signal: int = 7) -> pd.DataFrame:
    m = close.diff()
    t = 100 * ema(ema(m, r), s) / ema(ema(m.abs(), r), s).replace(0, np.nan)
    return pd.DataFrame({"tsi": t, "signal": ema(t, signal)})


def ultimate(df: pd.DataFrame, a: int = 7, b: int = 14, c: int = 28) -> pd.Series:
    pc = df["close"].shift()
    lo = np.minimum(df["low"], pc)
    bp, tr = df["close"] - lo, np.maximum(df["high"], pc) - lo
    av = lambda n: bp.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)  # noqa: E731
    return 100 * (4 * av(a) + 2 * av(b) + av(c)) / 7


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]; identical to stochastic %K - 100."""
    return ta.stoch(df, k=n)["k"] - 100


def stoch_rsi(close: pd.Series, n: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = ta.rsi(close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    kk = 100 * sma((r - lo) / (hi - lo).replace(0, np.nan), k)
    return pd.DataFrame({"k": kk, "d": sma(kk, d)})


def cmo(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    su, sd = d.clip(lower=0).rolling(n).sum(), (-d.clip(upper=0)).rolling(n).sum()
    return 100 * (su - sd) / (su + sd).replace(0, np.nan)


def dpo(close: pd.Series, n: int = 20) -> pd.Series:
    """Detrended price oscillator: price n/2+1 bars ago minus the current n-SMA (causal form)."""
    return close.shift(n // 2 + 1) - sma(close, n)


def bop(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)


def schaff(close: pd.Series, fast: int = 23, slow: int = 50, n: int = 10, f: float = 0.5) -> pd.Series:
    """Schaff trend cycle: double-smoothed stochastic of MACD, 0-100."""
    def st(x):
        lo, hi = x.rolling(n).min(), x.rolling(n).max()
        return (100 * (x - lo) / (hi - lo).replace(0, np.nan)).ffill()
    d1 = st(ema(close, fast) - ema(close, slow)).ewm(alpha=f, adjust=False).mean()
    return st(d1).ewm(alpha=f, adjust=False).mean()


# ---- volume -----------------------------------------------------------------------------
def _mfm(df):
    h, l, c, _, _ = _hlcov(df)
    return (((c - l) - (h - c)) / (h - l).replace(0, np.nan)).fillna(0)


def ad_line(df: pd.DataFrame) -> pd.Series:
    """Accumulation/distribution: cumulative money-flow multiplier x volume."""
    return (_mfm(df) * df["volume"].fillna(0)).cumsum()


def chaikin_osc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    ad = ad_line(df)
    return ema(ad, fast) - ema(ad, slow)


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    v = df["volume"].fillna(0)
    return (_mfm(df) * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)


def force_index(df: pd.DataFrame, n: int = 13) -> pd.Series:
    return ema(df["close"].diff() * df["volume"].fillna(0), n)


def ease_of_movement(df: pd.DataFrame, n: int = 14, scale: float = 1e6) -> pd.Series:
    h, l, _, _, v = _hlcov(df)
    emv = ((h + l) / 2).diff() * (h - l) * scale / v.replace(0, np.nan)
    return sma(emv.replace([np.inf, -np.inf], np.nan), n)


def pvo(volume: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    return _pct_osc(volume.astype(float), fast, slow, signal)


def nvi(df: pd.DataFrame, signal: int = 255) -> pd.DataFrame:
    """Negative volume index: compounds returns only on volume-down days."""
    r, v = df["close"].pct_change().fillna(0), df["volume"]
    x = 1000 * pd.Series(np.where(v < v.shift(), 1 + r, 1.0), index=df.index).cumprod()
    return pd.DataFrame({"nvi": x, "signal": ema(x, signal)})


def pvi(df: pd.DataFrame, signal: int = 255) -> pd.DataFrame:
    r, v = df["close"].pct_change().fillna(0), df["volume"]
    x = 1000 * pd.Series(np.where(v > v.shift(), 1 + r, 1.0), index=df.index).cumprod()
    return pd.DataFrame({"pvi": x, "signal": ema(x, signal)})


def pvt(df: pd.DataFrame) -> pd.Series:
    return (df["close"].pct_change().fillna(0) * df["volume"].fillna(0)).cumsum()


def klinger(df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.DataFrame:
    """Klinger volume oscillator (Investopedia definition with cumulative measurement)."""
    h, l, c, _, v = (x.to_numpy(float) for x in _hlcov(df))
    hlc = h + l + c
    trend = np.where(np.r_[np.nan, np.diff(hlc)] > 0, 1.0, -1.0)
    dm = h - l
    cm = dm.copy()
    for i in range(1, len(dm)):
        cm[i] = (cm[i - 1] if trend[i] == trend[i - 1] else dm[i - 1]) + dm[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        vf = v * np.abs(2 * dm / np.where(cm == 0, np.nan, cm) - 1) * trend * 100
    vf = pd.Series(vf, index=df.index).fillna(0)
    k = ema(vf, fast) - ema(vf, slow)
    return pd.DataFrame({"kvo": k, "signal": ema(k, signal)})


# ---- volatility ------------------------------------------------------------------------
def mass_index(df: pd.DataFrame, n: int = 9, s: int = 25) -> pd.Series:
    e1 = ema(df["high"] - df["low"], n)
    return (e1 / ema(e1, n)).rolling(s).sum()


def ulcer_index(close: pd.Series, n: int = 14) -> pd.Series:
    dd = 100 * (close / close.rolling(n).max() - 1)
    return np.sqrt((dd ** 2).rolling(n).mean())


def vortex(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tr = true_range(df).rolling(n).sum().replace(0, np.nan)
    return pd.DataFrame({"plus": (df["high"] - df["low"].shift()).abs().rolling(n).sum() / tr,
                         "minus": (df["low"] - df["high"].shift()).abs().rolling(n).sum() / tr})


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """100 = pure chop, 0 = straight line; >61.8 range, <38.2 trend."""
    rng = (df["high"].rolling(n).max() - df["low"].rolling(n).min()).replace(0, np.nan)
    return 100 * np.log10(true_range(df).rolling(n).sum() / rng) / np.log10(n)


def hist_vol(df: pd.DataFrame, n: int = 20, ann: int = 252) -> pd.DataFrame:
    """Annualised historical volatility: close-close, Parkinson, Garman-Klass, Yang-Zhang."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    hl2, co = np.log(h / l) ** 2, np.log(c / o)
    cc = np.log(c / c.shift()).rolling(n).std() * np.sqrt(ann)
    park = np.sqrt(hl2.rolling(n).mean() / (4 * np.log(2)) * ann)
    gk = np.sqrt((0.5 * hl2 - (2 * np.log(2) - 1) * co ** 2).rolling(n).mean() * ann)
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = np.sqrt((np.log(o / c.shift()).rolling(n).var() + k * co.rolling(n).var() + (1 - k) * rs) * ann)
    return pd.DataFrame({"close": cc, "parkinson": park, "garman_klass": gk, "yang_zhang": yz})


def linreg(close: pd.Series, n: int = 50, k: float = 2.0) -> pd.DataFrame:
    """Rolling least-squares fit: endpoint value, slope (per bar and % of price), R^2 and a +-k sigma channel."""
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    mt, my = t.rolling(n).mean(), close.rolling(n).mean()
    slope = ((t * close).rolling(n).mean() - mt * my) / ((n * n - 1) / 12)
    val = my + slope * (t - mt)
    r2 = close.rolling(n).corr(t) ** 2
    sd = close.rolling(n).std(ddof=0) * np.sqrt((1 - r2).clip(lower=0))
    return pd.DataFrame({"value": val, "slope": slope, "slope_pct": slope / val, "r2": r2,
                         "upper": val + k * sd, "lower": val - k * sd})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Same rules as ta.supertrend but seeded at the first valid ATR. ta.supertrend never leaves its NaN
    seed (bands stay NaN, trend stays +1), so the catalogue and signals use this version."""
    a = atr(df, n)
    out = pd.DataFrame({"supertrend": np.nan, "trend": np.nan}, index=df.index)
    first = a.first_valid_index()
    if first is None:
        return out
    k = df.index.get_loc(first)
    out.iloc[k:] = _st_core(df.iloc[k:], a.iloc[k:], mult)
    return out


def _st_core(df, a, mult):
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    up_b, lo_b, close = hl2 + mult * a.to_numpy(), hl2 - mult * a.to_numpy(), df["close"].to_numpy()
    fu, fl, trend = up_b.copy(), lo_b.copy(), np.ones(len(df))
    for i in range(1, len(df)):
        fu[i] = up_b[i] if (up_b[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo_b[i] if (lo_b[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        trend[i] = 1 if close[i] > fu[i - 1] else -1 if close[i] < fl[i - 1] else trend[i - 1]
    return np.c_[np.where(trend > 0, fl, fu), trend]


# ---- alternative chart types --------------------------------------------------------------
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """HA close = OHLC/4; HA open = midpoint of the previous HA candle body."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    hc = (o + h + l + c) / 4
    seed = hc.shift()
    if len(df):
        seed.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    ho = seed.ewm(alpha=0.5, adjust=False).mean()
    return pd.DataFrame({"open": ho, "high": pd.concat([h, ho, hc], axis=1).max(axis=1),
                         "low": pd.concat([l, ho, hc], axis=1).min(axis=1), "close": hc})


def renko(df: pd.DataFrame, box: float | None = None, atr_n: int = 14) -> pd.DataFrame:
    """Close-based Renko bricks. Default box = latest ATR (uses the whole window: for display, not signals)."""
    c = df["close"].to_numpy(float)
    if box is None:
        box = float(atr(df, atr_n).iloc[-1]) if len(df) > atr_n else float(np.nanstd(np.diff(c)) or 1)
    rows = []
    if not len(c) or not np.isfinite(box) or box <= 0:
        return pd.DataFrame(columns=["time", "open", "close", "dir"])
    hi = lo = c[0]
    for i in range(1, len(c)):
        if c[i] >= hi + box:
            k = int((c[i] - hi) // box)
            rows += [(df.index[i], hi + j * box, hi + (j + 1) * box, 1) for j in range(k)]
            lo, hi = hi + (k - 1) * box, hi + k * box
        elif c[i] <= lo - box:
            k = int((lo - c[i]) // box)
            rows += [(df.index[i], lo - j * box, lo - (j + 1) * box, -1) for j in range(k)]
            hi, lo = lo - (k - 1) * box, lo - k * box
    out = pd.DataFrame(rows, columns=["time", "open", "close", "dir"])
    out.attrs["box"] = box
    return out


def point_figure(df: pd.DataFrame, box: float | None = None, reversal: int = 3, atr_n: int = 14) -> pd.DataFrame:
    """High/low point & figure columns (X rising, O falling). Default box = latest ATR."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    if box is None:
        box = float(atr(df, atr_n).iloc[-1]) if len(df) > atr_n else float(np.nanmean(h - l) or 1)
    cols, kind = [], 0
    if not len(h) or not np.isfinite(box) or box <= 0:
        return pd.DataFrame(columns=["kind", "start", "end", "low", "high", "boxes"])
    ref, top, bot, start = (h[0] + l[0]) / 2, None, None, df.index[0]
    for i in range(1, len(h)):
        t = df.index[i]
        if kind == 0:
            if h[i] >= ref + box:
                kind, bot, top, start = 1, ref, ref + ((h[i] - ref) // box) * box, t
            elif l[i] <= ref - box:
                kind, top, bot, start = -1, ref, ref - ((ref - l[i]) // box) * box, t
        elif kind == 1:
            if h[i] >= top + box:
                top += ((h[i] - top) // box) * box
            elif l[i] <= top - reversal * box:
                cols.append(("X", start, df.index[i - 1], bot, top))
                kind, start = -1, t
                bot, top = top - ((top - l[i]) // box) * box, top - box
        else:
            if l[i] <= bot - box:
                bot -= ((bot - l[i]) // box) * box
            elif h[i] >= bot + reversal * box:
                cols.append(("O", start, df.index[i - 1], bot, top))
                kind, start = 1, t
                bot, top = bot + box, bot + ((h[i] - bot) // box) * box
    if kind:
        cols.append(("X" if kind == 1 else "O", start, df.index[-1], bot, top))
    out = pd.DataFrame(cols, columns=["kind", "start", "end", "low", "high"])
    out["boxes"] = ((out["high"] - out["low"]) / box).round().astype(int) + 1 if len(out) else []
    out.attrs["box"] = box
    return out


# ---- levels --------------------------------------------------------------------------------
def pivot_levels(df: pd.DataFrame, method: str = "classic") -> dict:
    """Floor pivots from the last completed bar: classic | camarilla | woodie | fibonacci."""
    if method == "classic":
        return ta.pivots(df)
    h, l, c = (float(df[k].iloc[-1]) for k in ("high", "low", "close"))
    r = h - l
    if method == "camarilla":
        out = {"P": (h + l + c) / 3}
        for i, f in enumerate((12, 6, 4, 2), 1):
            out[f"R{i}"], out[f"S{i}"] = c + r * 1.1 / f, c - r * 1.1 / f
        return out
    if method == "woodie":
        p = (h + l + 2 * c) / 4
        return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + r, "S2": p - r}
    if method == "fibonacci":
        p = (h + l + c) / 3
        return {"P": p, **{f"R{i}": p + f * r for i, f in enumerate((0.382, 0.618, 1.0), 1)},
                **{f"S{i}": p - f * r for i, f in enumerate((0.382, 0.618, 1.0), 1)}}
    raise ValueError(method)


def market_profile(df: pd.DataFrame, bins: int = 50, lookback: int | None = None, value_area: float = 0.7) -> dict:
    """Volume-at-price approximation: each bar's volume spread evenly over its high-low range.
    Returns POC, value-area high/low (70% default) and the histogram. Falls back to TPO counts without volume."""
    w = df.tail(lookback) if lookback else df
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if not np.isfinite(lo) or hi <= lo:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "hist": pd.DataFrame(columns=["price", "volume"])}
    edges = np.linspace(lo, hi, bins + 1)
    L, H = w["low"].to_numpy(float)[:, None], w["high"].to_numpy(float)[:, None]
    rng = H - L
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(rng > 0, np.clip(np.minimum(H, edges[1:]) - np.maximum(L, edges[:-1]), 0, None) / rng, 0.0)
    flat = (rng[:, 0] <= 0)
    if flat.any():
        j = np.clip(np.searchsorted(edges, w["close"].to_numpy(float)[flat], "right") - 1, 0, bins - 1)
        share[np.flatnonzero(flat), j] = 1.0
    vol = w["volume"].fillna(0).to_numpy(float)
    if vol.sum() <= 0:
        vol = np.ones(len(w))
    hist = (share * vol[:, None]).sum(axis=0)
    poc = int(np.argmax(hist))
    a, b, tot = poc, poc, hist[poc]
    while tot < value_area * hist.sum() and (a > 0 or b < bins - 1):
        dn, up = (hist[a - 1] if a > 0 else -1), (hist[b + 1] if b < bins - 1 else -1)
        if up >= dn:
            b += 1
            tot += hist[b]
        else:
            a -= 1
            tot += hist[a]
    mids = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(mids[poc]), "vah": float(edges[b + 1]), "val": float(edges[a]),
            "hist": pd.DataFrame({"price": mids, "volume": hist})}


# ---- zigzag / swing structure -----------------------------------------------------------------
def _threshold(df, pct, atr_mult, atr_n):
    if pct is not None:
        return (pct * df["close"]).to_numpy(float)
    a = atr(df, atr_n).fillna(true_range(df).expanding().mean())
    return (atr_mult * a).to_numpy(float)


def _zigzag(df, pct=None, atr_mult=2.0, atr_n=14):
    """Causal zigzag: a pivot is recorded only once price has reversed by the threshold (confirm bar)."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    thr = _threshold(df, pct, atr_mult, atr_n)
    piv, trend, hi_i, lo_i, ext = [], 0, 0, 0, 0
    for i in range(1, len(h)):
        t = thr[i]
        if not np.isfinite(t) or t <= 0:
            continue
        if trend == 0:
            if h[i] > h[hi_i]:
                hi_i = i
            if l[i] < l[lo_i]:
                lo_i = i
            if h[hi_i] - l[lo_i] >= t:
                if lo_i < hi_i:
                    piv.append((lo_i, l[lo_i], -1, i))
                    trend, ext = 1, hi_i
                else:
                    piv.append((hi_i, h[hi_i], 1, i))
                    trend, ext = -1, lo_i
        elif trend == 1:
            if h[i] > h[ext]:
                ext = i
            elif h[ext] - l[i] >= t:
                piv.append((ext, h[ext], 1, i))
                trend, ext = -1, ext + 1 + int(np.argmin(l[ext + 1:i + 1]))
        else:
            if l[i] < l[ext]:
                ext = i
            elif h[i] - l[ext] >= t:
                piv.append((ext, l[ext], -1, i))
                trend, ext = 1, ext + 1 + int(np.argmax(h[ext + 1:i + 1]))
    return piv, thr


def zigzag(df: pd.DataFrame, pct: float | None = None, atr_mult: float = 2.0, atr_n: int = 14) -> pd.DataFrame:
    """Confirmed swing pivots (percentage or ATR reversal) labelled HH/LH/HL/LL. Elliott-style swing
    labelling only: no wave counts are claimed. `confirm_time` is when the pivot became knowable."""
    piv, _ = _zigzag(df, pct, atr_mult, atr_n)
    out = pd.DataFrame(piv, columns=["i", "price", "kind", "confirm_i"])
    out["time"] = df.index[out["i"].to_numpy(int)] if len(out) else pd.Series(dtype="datetime64[ns, UTC]")
    out["confirm_time"] = df.index[out["confirm_i"].to_numpy(int)] if len(out) else out["time"]
    labels, last = [], {1: None, -1: None}
    for p, k in zip(out["price"], out["kind"]):
        prev = last[k]
        labels.append(("H" if k == 1 else "L") if prev is None else
                      ("HH" if p > prev else "LH") if k == 1 else ("HL" if p > prev else "LL"))
        last[k] = p
    out["label"] = labels
    out["kind"] = out["kind"].map({1: "high", -1: "low"})
    return out[["time", "price", "kind", "label", "confirm_time", "i", "confirm_i"]]


def dow_trend(df: pd.DataFrame, pct: float | None = None, atr_mult: float = 2.0) -> pd.Series:
    """Dow theory state per bar from confirmed swings: +1 HH+HL, -1 LH+LL, 0 mixed (causal)."""
    zz = zigzag(df, pct, atr_mult)
    state = np.full(len(df), np.nan)
    last = {"high": None, "low": None}
    for lab, kind, ci in zip(zz["label"], zz["kind"], zz["confirm_i"]):
        last[kind] = lab
        s = 1 if (last["high"] == "HH" and last["low"] == "HL") else -1 if (last["high"] == "LH" and last["low"] == "LL") else 0
        state[ci:] = s
    return pd.Series(state, index=df.index, name="dow_trend")


def wyckoff_phase(df: pd.DataFrame) -> pd.Series:
    """Heuristic Wyckoff phase: accumulation/distribution = choppy range after a down/up leg,
    markup/markdown = Dow trend aligned with the 50-SMA. A label, not a forecast."""
    c = df["close"]
    chop, dt, s50 = choppiness(df, 20), dow_trend(df), sma(c, 50)
    prior = c.shift(20) / c.shift(140) - 1
    out = pd.Series("transition", index=df.index, dtype=object)
    out[(dt == 1) & (c > s50)] = "markup"
    out[(dt == -1) & (c < s50)] = "markdown"
    rng = chop > 55
    out[rng & (prior < 0)] = "accumulation"
    out[rng & (prior > 0)] = "distribution"
    out[s50.isna()] = None
    return out


# ---- candlestick patterns ------------------------------------------------------------------------
CANDLES = {  # name: (direction, Investopedia term, definition)
    "doji": (0, "Doji", "body <= 10% of range"),
    "doji_dragonfly": (1, "Dragonfly Doji", "doji, upper shadow <= 10%, lower shadow >= 60% of range"),
    "doji_gravestone": (-1, "Gravestone Doji", "doji, lower shadow <= 10%, upper shadow >= 60% of range"),
    "doji_long_legged": (0, "Long-Legged Doji", "doji, both shadows >= 30% of range, range >= 1.3x 10-bar avg"),
    "hammer": (1, "Hammer Candlestick", "after downtrend: body <= 35% range, lower shadow >= 2x body, upper <= 15%"),
    "hanging_man": (-1, "Hanging Man", "hammer shape after an uptrend"),
    "inverted_hammer": (1, "Inverted Hammer", "after downtrend: upper shadow >= 2x body, lower <= 15% range"),
    "shooting_star": (-1, "Shooting Star", "inverted-hammer shape after an uptrend"),
    "bullish_engulfing": (1, "Bullish Engulfing Pattern", "downtrend, bear candle then larger bull body engulfing it"),
    "bearish_engulfing": (-1, "Bearish Engulfing Pattern", "uptrend, bull candle then larger bear body engulfing it"),
    "bullish_harami": (1, "Bullish Harami", "downtrend, long bear body then body <= 50% inside it"),
    "bearish_harami": (-1, "Bearish Harami", "uptrend, long bull body then body <= 50% inside it"),
    "piercing_line": (1, "Piercing Pattern", "downtrend, long bear, bull opens below prior close and closes above its midpoint"),
    "dark_cloud_cover": (-1, "Dark Cloud Cover", "uptrend, long bull, bear opens above prior close and closes below its midpoint"),
    "morning_star": (1, "Morning Star", "long bear, small star body at/below its close, bull closing above first midpoint"),
    "evening_star": (-1, "Evening Star", "long bull, small star body at/above its close, bear closing below first midpoint"),
    "three_white_soldiers": (1, "Three White Soldiers", "after downtrend: 3 rising bulls, bodies >= 60% range, opens inside prior body"),
    "three_black_crows": (-1, "Three Black Crows", "after uptrend: 3 falling bears, bodies >= 60% range, opens inside prior body"),
    "tweezer_top": (-1, "Tweezer Tops", "uptrend, bull then bear with equal highs (within 10% avg range)"),
    "tweezer_bottom": (1, "Tweezer Bottoms", "downtrend, bear then bull with equal lows (within 10% avg range)"),
    "bullish_marubozu": (1, "Marubozu", "long bull body, each shadow <= 5% of range"),
    "bearish_marubozu": (-1, "Marubozu", "long bear body, each shadow <= 5% of range"),
    "spinning_top": (0, "Spinning Top", "body 10-30% of range with both shadows longer than the body"),
    "three_inside_up": (1, "Three Inside Up/Down", "bullish harami then bull close above the first open"),
    "three_inside_down": (-1, "Three Inside Up/Down", "bearish harami then bear close below the first open"),
    "three_outside_up": (1, "Three Outside Up/Down", "bullish engulfing then a higher bull close"),
    "three_outside_down": (-1, "Three Outside Up/Down", "bearish engulfing then a lower bear close"),
    "rising_three_methods": (1, "Rising Three Methods", "uptrend, long bull, 3 small bodies inside its range, long bull closing higher"),
    "falling_three_methods": (-1, "Falling Three Methods", "downtrend, long bear, 3 small bodies inside its range, long bear closing lower"),
    "bullish_kicker": (1, "Kicker Pattern", "long bear then long bull opening above the bear's open"),
    "bearish_kicker": (-1, "Kicker Pattern", "long bull then long bear opening below the bull's open"),
    "bullish_abandoned_baby": (1, "Abandoned Baby", "downtrend, long bear, doji gapped below, bull gapped above the doji"),
    "bearish_abandoned_baby": (-1, "Abandoned Baby", "uptrend, long bull, doji gapped above, bear gapped below the doji"),
}


def trend_context(df: pd.DataFrame, n: int = 10, lb: int = 5) -> pd.DataFrame:
    """Short-term trend used as candle context: close vs n-SMA and vs close lb bars ago."""
    c = df["close"]
    m = sma(c, n)
    return pd.DataFrame({"up": (c > m) & (c > c.shift(lb)), "down": (c < m) & (c < c.shift(lb))})


def candles(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, one column per CANDLES entry; True on the bar that completes the pattern.
    Prior-trend context is measured on the bar before the pattern's first candle (first candle for 2-bar ones)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body, rng = (c - o).abs(), h - l
    top, bot = np.maximum(o, c), np.minimum(o, c)
    ush, lsh = h - top, bot - l
    bull, bear = c > o, c < o
    abody = body.rolling(10, min_periods=5).mean().shift()
    arng = rng.rolling(10, min_periods=5).mean().shift()
    long_ = body >= abody
    tc = trend_context(df)
    S = lambda s, k: s.shift(k)  # noqa: E731
    B = lambda s, k: s.shift(k, fill_value=False).astype(bool)  # noqa: E731
    up_ = lambda k: B(tc["up"], k)  # noqa: E731
    dn_ = lambda k: B(tc["down"], k)  # noqa: E731

    doji = (body <= 0.1 * rng) & (rng > 0)
    out = {"doji": doji,
           "doji_dragonfly": doji & (ush <= 0.1 * rng) & (lsh >= 0.6 * rng),
           "doji_gravestone": doji & (lsh <= 0.1 * rng) & (ush >= 0.6 * rng),
           "doji_long_legged": doji & (ush >= 0.3 * rng) & (lsh >= 0.3 * rng) & (rng >= 1.3 * arng)}
    ham = ~doji & (body <= 0.35 * rng) & (lsh >= 2 * body) & (ush <= 0.15 * rng)
    inv = ~doji & (body <= 0.35 * rng) & (ush >= 2 * body) & (lsh <= 0.15 * rng)
    out |= {"hammer": ham & dn_(1), "hanging_man": ham & up_(1),
            "inverted_hammer": inv & dn_(1), "shooting_star": inv & up_(1)}

    o1, h1, l1, c1, b1 = S(o, 1), S(h, 1), S(l, 1), S(c, 1), S(body, 1)
    bull1, bear1, long1 = B(bull, 1), B(bear, 1), B(long_, 1)
    out["bullish_engulfing"] = bear1 & bull & (o <= c1) & (c >= o1) & (body > b1) & dn_(1)
    out["bearish_engulfing"] = bull1 & bear & (o >= c1) & (c <= o1) & (body > b1) & up_(1)
    out["bullish_harami"] = bear1 & long1 & (top <= o1) & (bot >= c1) & (body <= 0.5 * b1) & dn_(1)
    out["bearish_harami"] = bull1 & long1 & (top <= c1) & (bot >= o1) & (body <= 0.5 * b1) & up_(1)
    out["piercing_line"] = bear1 & long1 & bull & (o < c1) & (c > (o1 + c1) / 2) & (c < o1) & dn_(1)
    out["dark_cloud_cover"] = bull1 & long1 & bear & (o > c1) & (c < (o1 + c1) / 2) & (c > o1) & up_(1)

    o2, c2, b2, h2, l2 = S(o, 2), S(c, 2), S(body, 2), S(h, 2), S(l, 2)
    bull2, bear2, long2 = B(bull, 2), B(bear, 2), B(long_, 2)
    small1 = b1 <= 0.3 * b2
    out["morning_star"] = bear2 & long2 & small1 & (S(top, 1) <= c2) & bull & (c > (o2 + c2) / 2) & dn_(2)
    out["evening_star"] = bull2 & long2 & small1 & (S(bot, 1) >= c2) & bear & (c < (o2 + c2) / 2) & up_(2)

    def soldiers(sign):
        ok = pd.Series(True, index=df.index)
        for k in range(3):
            ck, ok_, bk, rk = S(c, k), S(o, k), S(body, k), S(rng, k)
            ok &= B(bull if sign > 0 else bear, k) & (bk >= 0.6 * rk)
            if k < 2:
                pc_, po_ = S(c, k + 1), S(o, k + 1)
                ok &= (sign * (ck - pc_) > 0) & (ok_ >= np.minimum(po_, pc_)) & (ok_ <= np.maximum(po_, pc_))
        return ok & (dn_(3) if sign > 0 else up_(3))
    out["three_white_soldiers"], out["three_black_crows"] = soldiers(1), soldiers(-1)

    tol = 0.1 * arng
    out["tweezer_top"] = up_(1) & bull1 & bear & ((h - h1).abs() <= tol)
    out["tweezer_bottom"] = dn_(1) & bear1 & bull & ((l - l1).abs() <= tol)
    maru = (ush <= 0.05 * rng) & (lsh <= 0.05 * rng) & long_ & (rng > 0)
    out["bullish_marubozu"], out["bearish_marubozu"] = maru & bull, maru & bear
    out["spinning_top"] = ~doji & (body <= 0.3 * rng) & (ush > body) & (lsh > body)
    out["three_inside_up"] = B(out["bullish_harami"], 1) & bull & (c > o2)
    out["three_inside_down"] = B(out["bearish_harami"], 1) & bear & (c < o2)
    out["three_outside_up"] = B(out["bullish_engulfing"], 1) & bull & (c > c1)
    out["three_outside_down"] = B(out["bearish_engulfing"], 1) & bear & (c < c1)

    def three_methods(sign):
        b4, h4, l4, c4 = S(body, 4), S(h, 4), S(l, 4), S(c, 4)
        ok = B(bull if sign > 0 else bear, 4) & B(long_, 4) & (bull if sign > 0 else bear) & long_ & (sign * (c - c4) > 0)
        for k in (1, 2, 3):
            ok &= (S(body, k) < 0.5 * b4) & (S(h, k) <= h4) & (S(l, k) >= l4)
        return ok & (up_(4) if sign > 0 else dn_(4))
    out["rising_three_methods"], out["falling_three_methods"] = three_methods(1), three_methods(-1)
    out["bullish_kicker"] = bear1 & long1 & bull & long_ & (o > o1)
    out["bearish_kicker"] = bull1 & long1 & bear & long_ & (o < o1)
    doji1 = B(doji, 1)
    out["bullish_abandoned_baby"] = bear2 & long2 & doji1 & (h1 < l2) & bull & (l > h1) & dn_(2)
    out["bearish_abandoned_baby"] = bull2 & long2 & doji1 & (l1 > h2) & bear & (h < l1) & up_(2)
    return pd.DataFrame({k: out[k].fillna(False).astype(bool) for k in CANDLES}, index=df.index)


# ---- bar-level patterns ------------------------------------------------------------------------
BAR_PATTERNS = {
    "high_52w_breakout": (1, "52-Week High/Low", "close above the prior 252-bar high"),
    "low_52w_breakdown": (-1, "52-Week High/Low", "close below the prior 252-bar low"),
    "inside_bar": (0, "Inside Day", "high below and low above the previous bar's"),
    "outside_bar": (0, "Outside Days", "high above and low below the previous bar's"),
    "nr4": (0, "Narrow Range (NR4/NR7)", "narrowest high-low range of the last 4 bars"),
    "nr7": (0, "Narrow Range (NR4/NR7)", "narrowest high-low range of the last 7 bars"),
    "gap_up": (1, "Gap", "low above the previous high"),
    "gap_down": (-1, "Gap", "high below the previous low"),
}


def bar_patterns(df: pd.DataFrame, lookback52: int = 252) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    rng = h - l
    out = {"high_52w_breakout": c > h.rolling(lookback52).max().shift(),
           "low_52w_breakdown": c < l.rolling(lookback52).min().shift(),
           "inside_bar": (h < h.shift()) & (l > l.shift()), "outside_bar": (h > h.shift()) & (l < l.shift()),
           "nr4": rng <= rng.rolling(4).min(), "nr7": rng <= rng.rolling(7).min(),
           "gap_up": l > h.shift(), "gap_down": h < l.shift()}
    return pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in out.items()}, index=df.index)


# ---- chart patterns -------------------------------------------------------------------------------
CHART_PATTERNS = {  # name: (bias, Investopedia term, rule)
    "double_top": (-1, "Double Top", "H-L-H with tops within 1 ATR after a rise; breaks the trough (neckline)"),
    "double_bottom": (1, "Double Bottom", "L-H-L with bottoms within 1 ATR after a fall; breaks the peak (neckline)"),
    "triple_top": (-1, "Triple Top", "three highs within 1 ATR; breaks the lower of the two troughs"),
    "triple_bottom": (1, "Triple Bottom", "three lows within 1 ATR; breaks the higher of the two peaks"),
    "head_shoulders": (-1, "Head and Shoulders Pattern", "head > both shoulders by 1 ATR, similar shoulders; breaks sloped neckline"),
    "inverse_head_shoulders": (1, "Inverse Head and Shoulders", "mirror of head and shoulders"),
    "ascending_triangle": (1, "Ascending Triangle", "flat highs, rising lows over 5 pivots; breaks the flat top"),
    "descending_triangle": (-1, "Descending Triangle", "falling highs, flat lows; breaks the flat bottom"),
    "symmetrical_triangle": (0, "Symmetrical Triangle", "falling highs, rising lows; break either side"),
    "rising_wedge": (-1, "Rising Wedge", "both boundaries rising and converging; breaks support"),
    "falling_wedge": (1, "Falling Wedge", "both boundaries falling and converging; breaks resistance"),
    "rectangle": (0, "Rectangle Formation", "flat highs and flat lows; break either side"),
    "bull_flag": (1, "Flag", "pole >= 2 zigzag thresholds in <= 15 bars, pullback <= 50% in <= 20 bars; breaks the pole high"),
    "bear_flag": (-1, "Flag", "mirror of bull flag"),
    "bull_pennant": (1, "Pennant", "bull flag whose consolidation ranges contract"),
    "bear_pennant": (-1, "Pennant", "bear flag whose consolidation ranges contract"),
    "cup_handle": (1, "Cup and Handle", "rims within 2 ATR, rounded cup >= 20 bars, handle in the upper half; breaks the right rim"),
    "breakaway_gap": (0, "Breakaway Gap", "gap clearing the 20-bar range from a non-trending base (ADX<25)"),
    "runaway_gap": (0, "Runaway Gap", "gap in the direction of an established trend (vs 20/50 SMA)"),
    "exhaustion_gap": (0, "Exhaustion Gap", "gap after an extended move (RSI>70/<30 or >3 ATR from 20 SMA); bias against the gap"),
    "common_gap": (0, "Gap", "any other true gap"),
}


def _line(i0, v0, i1, v1):
    s = (v1 - v0) / (i1 - i0) if i1 != i0 else 0.0
    return s, lambda i: v0 + s * (i - i0)


def _detect_tops(pi, pv, pk, j, det, tol, thr, h, l, c):
    """Top/bottom family on pivots ending at j: triple > head&shoulders > double (one per j)."""
    sgn = pk[j]  # +1: tops, -1: bottoms
    lvl = lambda k: pv[k] * sgn  # noqa: E731  (mirror bottoms into tops)
    if j >= 5 and pk[j - 5] == -sgn:
        tops, necks = [j - 4, j - 2, j], [j - 3, j - 1]
        tv = [lvl(k) for k in tops]
        if max(tv) - min(tv) <= tol and lvl(j - 5) < min(lvl(k) for k in necks):
            neck = min(lvl(k) for k in necks) * sgn
            ext = max(tv) * sgn
            return dict(pattern="triple_top" if sgn > 0 else "triple_bottom", bias=-sgn, s=j - 4,
                        up=(ext if sgn > 0 else neck, 0.0), lo=(neck if sgn > 0 else ext, 0.0),
                        height=abs(np.mean([pv[k] for k in tops]) - neck),
                        levels=f"{'tops' if sgn > 0 else 'bottoms'} {'/'.join(f'{pv[k]:.4g}' for k in tops)}; neckline {neck:.4g}")
        ls, hd, rs = lvl(j - 4), lvl(j - 2), lvl(j)
        n1, n2 = lvl(j - 3), lvl(j - 1)
        if hd > max(ls, rs) + tol and abs(ls - rs) <= max(2 * tol, 0.35 * (hd - max(n1, n2))) \
                and min(ls, rs) > max(n1, n2) and lvl(j - 5) < min(n1, n2):
            s, f = _line(pi[j - 3], pv[j - 3], pi[j - 1], pv[j - 1])
            neck_det = f(det)
            return dict(pattern="head_shoulders" if sgn > 0 else "inverse_head_shoulders", bias=-sgn, s=j - 4,
                        up=(pv[j - 2], 0.0) if sgn > 0 else (neck_det, s), lo=(neck_det, s) if sgn > 0 else (pv[j - 2], 0.0),
                        height=abs(pv[j - 2] - f(pi[j - 2])),
                        levels=f"shoulders {pv[j - 4]:.4g}/{pv[j]:.4g}; head {pv[j - 2]:.4g}; neckline {neck_det:.4g} (slope {s:+.3g}/bar)")
    if j >= 3 and pk[j - 3] == -sgn:
        t1, t2, nk, pre = lvl(j - 2), lvl(j), lvl(j - 1), lvl(j - 3)
        if abs(t1 - t2) <= tol and pre < nk and 5 <= pi[j] - pi[j - 2] <= 150:
            neck, ext = pv[j - 1], max(t1, t2) * sgn
            return dict(pattern="double_top" if sgn > 0 else "double_bottom", bias=-sgn, s=j - 2,
                        up=(ext if sgn > 0 else neck, 0.0), lo=(neck if sgn > 0 else ext, 0.0),
                        height=abs((pv[j] + pv[j - 2]) / 2 - neck),
                        levels=f"{'tops' if sgn > 0 else 'bottoms'} {pv[j - 2]:.4g}/{pv[j]:.4g}; neckline {neck:.4g}")
    return None


def _detect_converging(pi, pv, pk, j, det, tol, thr, h, l, c):
    """Triangles, wedges and rectangles from the last 5 pivots (lines through the outer points of each side)."""
    if j < 4:
        return None
    ks = range(j - 4, j + 1)
    span = pi[j] - pi[j - 4]
    if span < 10:
        return None

    def fit(side):
        P = [(pi[k], pv[k]) for k in ks if pk[k] == side]
        s, f = _line(*P[0], *P[-1])
        if len(P) == 3 and abs(P[1][1] - f(P[1][0])) > tol:
            return None
        return s, f
    fh, fl = fit(1), fit(-1)
    if fh is None or fl is None:
        return None
    (sh, uh), (sl, ul) = fh, fl
    w0, w1 = uh(pi[j - 4]) - ul(pi[j - 4]), uh(det) - ul(det)
    if w0 <= 0 or w1 <= 0:
        return None
    cls = lambda d: 0 if abs(d) <= tol else (1 if d > 0 else -1)  # noqa: E731
    ch, cl, conv = cls(sh * span), cls(sl * span), w1 < 0.85 * w0
    name = {(0, 1): "ascending_triangle", (-1, 0): "descending_triangle", (-1, 1): "symmetrical_triangle",
            (0, 0): "rectangle"}.get((ch, cl))
    if name is None and conv and ch == cl == 1:
        name = "rising_wedge"
    if name is None and conv and ch == cl == -1:
        name = "falling_wedge"
    if name is None:
        return None
    return dict(pattern=name, bias=CHART_PATTERNS[name][0], s=j - 4, up=(uh(det), sh), lo=(ul(det), sl), height=w0,
                wait=int(np.clip(span, 10, 60)),
                levels=f"upper {uh(det):.4g} ({sh:+.3g}/bar); lower {ul(det):.4g} ({sl:+.3g}/bar); height {w0:.4g}")


def _detect_flag(pi, pv, pk, j, det, tol, thr, h, l, c):
    if j < 2:
        return None
    a, b = j - 2, j - 1
    sgn = pk[b]  # +1 bull (pole up to a high), -1 bear
    pole = abs(pv[b] - pv[a])
    if pole < 2 * thr[pi[b]] or not 1 <= pi[b] - pi[a] <= 15:
        return None
    if abs(pv[b] - pv[j]) / pole > 0.5 or not 2 <= pi[j] - pi[b] <= 20:
        return None
    r = (h - l)[pi[b]:pi[j] + 1]
    half = len(r) // 2
    pennant = len(r) >= 4 and r[half:].mean() < 0.75 * r[:half].mean()
    name = ("bull_" if sgn > 0 else "bear_") + ("pennant" if pennant else "flag")
    return dict(pattern=name, bias=sgn, s=a, up=(pv[b] if sgn > 0 else pv[j], 0.0), lo=(pv[j] if sgn > 0 else pv[b], 0.0),
                height=pole, wait=15, levels=f"pole {pv[a]:.4g}->{pv[b]:.4g}; consolidation extreme {pv[j]:.4g}")


def _detect_cup(pi, pv, pk, j, det, tol, thr, h, l, c):
    if j < 3 or pk[j] != -1:
        return None
    h0, l1, h2, l3 = j - 3, j - 2, j - 1, j
    depth = (pv[h0] + pv[h2]) / 2 - pv[l1]
    width = pi[h2] - pi[h0]
    if depth <= 0 or abs(pv[h2] - pv[h0]) > 2 * tol or not 20 <= width <= 400:
        return None
    if pv[h2] - pv[l3] > 0.5 * depth or pi[l3] - pi[h2] > width / 2:
        return None
    seg = c[pi[h0]:pi[h2] + 1]
    if np.mean(seg < pv[l1] + depth / 3) < 0.25:  # rounded, not a V
        return None
    return dict(pattern="cup_handle", bias=1, s=h0, up=(pv[h2], 0.0), lo=(pv[l3], 0.0), height=depth,
                wait=int(np.clip(2 * (pi[l3] - pi[h2]), 10, 40)),
                levels=f"rims {pv[h0]:.4g}/{pv[h2]:.4g}; cup low {pv[l1]:.4g}; handle low {pv[l3]:.4g}")


_DETECTORS = (_detect_tops, _detect_converging, _detect_flag, _detect_cup)


def _resolve(c, det, up, lo, bias, wait):
    """Walk forward from the detection bar: confirm on a close beyond the boundary in the bias direction
    (either side for neutral patterns), fail on a close beyond the opposite boundary."""
    end = min(len(c), det + wait + 1)
    k = np.arange(det, end) - det
    U, L = up[0] + up[1] * k, lo[0] + lo[1] * k
    cc = c[det:end]
    above, below = np.flatnonzero(cc > U), np.flatnonzero(cc < L)
    fa, fb = (above[0] if len(above) else None), (below[0] if len(below) else None)
    first = lambda a, b: a is not None and (b is None or a < b)  # noqa: E731
    if bias > 0:
        hit = ("confirmed", det + fa, 1, U[fa]) if first(fa, fb) else ("failed", det + fb, 0, np.nan) if fb is not None else None
    elif bias < 0:
        hit = ("confirmed", det + fb, -1, L[fb]) if first(fb, fa) else ("failed", det + fa, 0, np.nan) if fa is not None else None
    else:
        hit = ("confirmed", det + fa, 1, U[fa]) if first(fa, fb) else ("confirmed", det + fb, -1, L[fb]) if fb is not None else None
    if hit:
        return hit
    return ("pending", None, 0, np.nan) if end == len(c) else ("expired", end - 1, 0, np.nan)


def _gap_rows(df):
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    a = atr(df).shift()
    up, dn = l > h.shift(), h < l.shift()
    s20, s50 = sma(c, 20).shift(), sma(c, 50).shift()
    r = ta.rsi(c).shift()
    ax = ta.adx(df)["adx"].shift()
    hh, ll = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    c1 = c.shift()
    rows = []
    hv, lv = h.to_numpy(), l.to_numpy()
    for i in np.flatnonzero((up | dn).to_numpy()):
        g = 1 if up.iloc[i] else -1
        ext = (g * (c1.iloc[i] - s20.iloc[i]) > 3 * a.iloc[i]) or (r.iloc[i] > 70 if g > 0 else r.iloc[i] < 30)
        clears = (l.iloc[i] > hh.iloc[i]) if g > 0 else (h.iloc[i] < ll.iloc[i])
        trend = (g * (c1.iloc[i] - s50.iloc[i]) > 0) and (g * (s20.iloc[i] - s50.iloc[i]) > 0)
        kind = "exhaustion" if ext else "breakaway" if (clears and ax.iloc[i] < 25) else "runaway" if trend else "common"
        edge = hv[i - 1] if g > 0 else lv[i - 1]
        fut = np.flatnonzero(lv[i + 1:] <= edge) if g > 0 else np.flatnonzero(hv[i + 1:] >= edge)
        bias = -g if kind == "exhaustion" else g
        rows.append(dict(pattern=f"{kind}_gap", bias=bias, direction=bias, status="filled" if len(fut) else "open",
                         confirmed=True, start_i=i - 1, end_i=i, det_i=i, confirm_i=i, breakout_level=edge, target=np.nan,
                         invalidation=edge, levels=f"gap {'up' if g > 0 else 'down'} {edge:.4g}->{(lv[i] if g > 0 else hv[i]):.4g}"
                         f" ({abs((lv[i] if g > 0 else hv[i]) - edge) / a.iloc[i]:.2f} ATR)" if np.isfinite(a.iloc[i]) else ""))
    return rows


_PCOLS = ["pattern", "bias", "direction", "status", "confirmed", "start", "end", "detected", "confirm_time",
          "breakout_level", "target", "invalidation", "levels", "start_i", "end_i", "det_i", "confirm_i"]


def chart_patterns(df: pd.DataFrame, atr_mult: float = 2.0, pct: float | None = None, gaps: bool = True) -> pd.DataFrame:
    """Rule-based chart patterns on confirmed zigzag pivots (no look-ahead at detection time).
    One row per detection: start/end (first/last pivot), detected (bar the last pivot was confirmed),
    breakout level and measured-move target, invalidation, and status (confirmed/failed/pending/expired).
    `confirmed` is True when a close went beyond the neckline/boundary; confirm_time is that bar.
    A growing pattern of the same type that is still unresolved is superseded by its newer detection.
    Gap rows: `status` open/filled uses later bars (informational; the gap event itself is causal)."""
    piv, thr = _zigzag(df, pct, atr_mult)
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    pi = np.array([p[0] for p in piv], dtype=int)
    pv = np.array([p[1] for p in piv], dtype=float)
    pk = np.array([p[2] for p in piv], dtype=int)
    pc = np.array([p[3] for p in piv], dtype=int)
    rows, live = [], {}
    for j in range(len(piv)):
        det = int(pc[j])
        tol = 0.5 * thr[det]
        for f in _DETECTORS:
            d = f(pi, pv, pk, j, det, tol, thr, h, l, c)
            if d is None:
                continue
            status, ci, direc, brk = _resolve(c, det, d["up"], d["lo"], d["bias"], d.get("wait", int(np.clip(pi[j] - pi[d["s"]], 10, 60))))
            if not np.isfinite(brk):
                brk = d["up"][0] if d["bias"] > 0 else d["lo"][0] if d["bias"] < 0 else np.nan
            dd = direc or d["bias"]
            row = dict(pattern=d["pattern"], bias=d["bias"], direction=direc if status == "confirmed" else d["bias"],
                       status=status, confirmed=status == "confirmed", start_i=int(pi[d["s"]]), end_i=int(pi[j]), det_i=det,
                       confirm_i=ci if status == "confirmed" else None, resolve_i=ci, breakout_level=brk,
                       target=brk + dd * d["height"] if dd else np.nan,
                       invalidation=d["lo"][0] if d["bias"] > 0 else d["up"][0] if d["bias"] < 0 else np.nan, levels=d["levels"])
            prev = live.get(d["pattern"])
            if prev is not None and row["start_i"] < prev["end_i"] and (prev["resolve_i"] is None or prev["resolve_i"] > det):
                prev["drop"] = True
            live[d["pattern"]] = row
            rows.append(row)
    rows = [r for r in rows if not r.get("drop")]
    if gaps:
        rows += _gap_rows(df)
    out = pd.DataFrame(rows, columns=[k for k in _PCOLS if k not in ("start", "end", "detected", "confirm_time")])
    if out.empty:
        return pd.DataFrame(columns=_PCOLS)
    idx = df.index
    out["confirm_i"] = out["confirm_i"].astype("Int64")
    out["start"], out["end"], out["detected"] = idx[out["start_i"].astype(int)], idx[out["end_i"].astype(int)], idx[out["det_i"].astype(int)]
    out["confirm_time"] = [idx[int(i)] if pd.notna(i) else pd.NaT for i in out["confirm_i"]]
    return out[_PCOLS].sort_values(["det_i", "pattern"]).reset_index(drop=True)


def active_patterns(df: pd.DataFrame, within: int = 10, patterns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pending chart patterns plus those confirmed in the last `within` bars."""
    p = chart_patterns(df) if patterns is None else patterns
    if p.empty:
        return p
    n = len(df)
    recent = p["confirm_i"].fillna(-1).astype(int) >= n - within
    return p[(p["status"] == "pending") | (p["confirmed"] & recent)]


# ---- catalogue ---------------------------------------------------------------------------------------
def _cl(f, *a, **k):
    return lambda df: f(df["close"], *a, **k)


def _vwap(df):
    if _intraday(df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ta.vwap(df, "D")
    return ta.anchored_vwap(df, _naive(df["low"].tail(252).idxmin()))


def _naive(t):
    """ta.anchored_vwap localises the anchor itself, so pass it tz-naive."""
    t = pd.Timestamp(t)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _sel(f, col):
    return lambda df: f(df)[col]


def _chart(name):
    def f(df):
        p = chart_patterns(df)
        return p[p["pattern"] == name]
    return f


def _mk():
    I, Cn, Ch = "indicator", "candle", "chart"
    rows = [
        # ta.py
        ("sma", I, "trend", lambda df: pd.DataFrame({f"sma{n}": ta.sma(df["close"], n) for n in (20, 50, 200)}), "Simple moving averages 20/50/200", "Simple Moving Average (SMA)"),
        ("ema", I, "trend", _cl(ta.ema, 21), "Exponential moving average (21)", "Exponential Moving Average (EMA)"),
        ("macd", I, "trend", _cl(ta.macd), "MACD 12/26/9 line, signal, histogram", "Moving Average Convergence Divergence (MACD)"),
        ("adx_dmi", I, "trend", ta.adx, "ADX trend strength with +DI/-DI (DMI)", "Average Directional Index (ADX) / Directional Movement Index (DMI)"),
        ("supertrend", I, "trend", supertrend, "ATR trailing trend line (10, 3)", "Supertrend Indicator"),
        ("ichimoku", I, "trend", ta.ichimoku, "Tenkan/Kijun/Senkou spans/Chikou", "Ichimoku Cloud"),
        ("regime", I, "trend", ta.regime, "Trend/vol regime summary (SMA50/200, ADX, vol percentile)", "Trend Analysis"),
        ("rsi", I, "momentum", _cl(ta.rsi), "Wilder RSI(14); >70 overbought, <30 oversold", "Relative Strength Index (RSI)"),
        ("stochastic", I, "momentum", ta.stoch, "Stochastic %K(14)/%D(3); >80/<20", "Stochastic Oscillator"),
        ("roc", I, "momentum", _cl(ta.roc, 10), "Rate of change over 10 bars", "Price Rate of Change (ROC)"),
        ("atr", I, "volatility", ta.atr, "Wilder average true range (14)", "Average True Range (ATR)"),
        ("bollinger", I, "volatility", _cl(ta.bollinger), "20-SMA +-2 sd bands, %b and width (squeeze)", "Bollinger Bands"),
        ("keltner", I, "volatility", ta.keltner, "20-EMA +-2 ATR channel", "Keltner Channel"),
        ("donchian", I, "volatility", ta.donchian, "20-bar highest high / lowest low", "Donchian Channels"),
        ("obv", I, "volume", ta.obv, "On-balance volume", "On-Balance Volume (OBV)"),
        ("vwap", I, "volume", _vwap, "Session VWAP intraday; anchored at the 252-bar low on daily+", "Volume-Weighted Average Price (VWAP)"),
        ("anchored_vwap", I, "volume", lambda df: ta.anchored_vwap(df, _naive(df.index[max(len(df) - 252, 0)])), "VWAP anchored 252 bars back (any anchor via ta.anchored_vwap)", "Anchored VWAP"),
        ("mfi", I, "volume", ta.mfi, "Money flow index (volume-weighted RSI, 14)", "Money Flow Index (MFI)"),
        ("pivot_classic", I, "levels", ta.pivots, "Classic floor pivots from the last bar", "Pivot Point"),
        ("swing_points", I, "levels", ta.swing_points, "Fractal swing highs/lows (centred window: hindsight only)", "Swing High / Swing Low"),
        ("support_resistance", I, "levels", ta.support_resistance, "Clustered swing levels ranked by touches", "Support and Resistance"),
        ("fib_retracements", I, "levels", ta.fib_retracements, "Fibonacci retracements of the last 120-bar leg", "Fibonacci Retracement Levels"),
        # catalogue additions
        ("wma", I, "trend", _cl(wma, 20), "Linearly weighted MA (20)", "Weighted Moving Average (WMA)"),
        ("hma", I, "trend", _cl(hma, 20), "Hull MA (20): low-lag WMA blend", "Hull Moving Average (HMA)"),
        ("kama", I, "trend", _cl(kama, 10), "Kaufman adaptive MA (10, 2, 30)", "Kaufman's Adaptive Moving Average (KAMA)"),
        ("dema", I, "trend", _cl(dema, 20), "Double EMA (20)", "Double Exponential Moving Average (DEMA)"),
        ("tema", I, "trend", _cl(tema, 20), "Triple EMA (20)", "Triple Exponential Moving Average (TEMA)"),
        ("zlema", I, "trend", _cl(zlema, 20), "Zero-lag EMA (20)", "Zero-Lag Exponential Moving Average"),
        ("ma_envelopes", I, "trend", _cl(envelopes), "20-SMA +-2.5% envelopes", "Envelope (Moving Average Envelope)"),
        ("aroon", I, "trend", aroon, "Aroon up/down/oscillator (25)", "Aroon Indicator / Aroon Oscillator"),
        ("parabolic_sar", I, "trend", lambda df: psar(df), "Parabolic stop-and-reverse (0.02/0.2)", "Parabolic SAR"),
        ("vortex", I, "trend", vortex, "Vortex VI+/VI- (14)", "Vortex Indicator (VI)"),
        ("schaff", I, "trend", _cl(schaff), "Schaff trend cycle (23/50/10); >75/<25", "Schaff Trend Cycle (STC)"),
        ("linreg", I, "trend", _cl(linreg), "50-bar regression value, slope, R^2, +-2 sd channel", "Linear Regression Channel"),
        ("heikin_ashi", I, "trend", heikin_ashi, "Heikin Ashi candles (smoothed OHLC)", "Heikin-Ashi Technique"),
        ("renko", I, "trend", renko, "Renko bricks, ATR or fixed box", "Renko Chart"),
        ("point_figure", I, "trend", point_figure, "Point & figure X/O columns (box, 3-box reversal)", "Point-and-Figure (P&F) Chart"),
        ("dow_theory", I, "trend", dow_trend, "Trend state from zigzag swings: HH/HL up, LH/LL down", "Dow Theory"),
        ("wyckoff_phase", I, "trend", wyckoff_phase, "Heuristic accumulation/markup/distribution/markdown label", "Wyckoff Method"),
        ("cci", I, "momentum", cci, "Commodity channel index (20); +-100", "Commodity Channel Index (CCI)"),
        ("coppock", I, "momentum", _cl(coppock), "WMA10 of ROC14+ROC11; turn up below zero = buy", "Coppock Curve"),
        ("elder_ray", I, "momentum", elder_ray, "Bull/bear power vs 13-EMA", "Elder-Ray Index"),
        ("kst", I, "momentum", _cl(kst), "Know Sure Thing with 9-SMA signal", "Know Sure Thing (KST)"),
        ("ppo", I, "momentum", _cl(ppo), "Percentage price oscillator 12/26/9", "Percentage Price Oscillator (PPO)"),
        ("trix", I, "momentum", _cl(trix), "1-bar % change of triple EMA (15) with signal", "Triple Exponential Average (TRIX)"),
        ("tsi", I, "momentum", _cl(tsi), "True strength index 25/13 with 7 signal", "True Strength Index (TSI)"),
        ("ultimate_oscillator", I, "momentum", ultimate, "Ultimate oscillator 7/14/28; >70/<30", "Ultimate Oscillator"),
        ("williams_r", I, "momentum", williams_r, "Williams %R (14); >-20 overbought, <-80 oversold", "Williams %R"),
        ("stoch_rsi", I, "momentum", _cl(stoch_rsi), "Stochastic of RSI (14,3,3); >80/<20", "Stochastic RSI (StochRSI)"),
        ("cmo", I, "momentum", _cl(cmo), "Chande momentum oscillator (14); +-50", "Chande Momentum Oscillator"),
        ("dpo", I, "momentum", _cl(dpo), "Detrended price oscillator (20)", "Detrended Price Oscillator (DPO)"),
        ("balance_of_power", I, "momentum", bop, "(close-open)/(high-low), smoothed 14", "Balance of Power (BOP)"),
        ("ad_line", I, "volume", ad_line, "Accumulation/distribution line", "Accumulation/Distribution Indicator (A/D)"),
        ("chaikin_osc", I, "volume", chaikin_osc, "3/10 EMA difference of the A/D line", "Chaikin Oscillator"),
        ("cmf", I, "volume", cmf, "Chaikin money flow (20); >0 accumulation", "Chaikin Money Flow (CMF)"),
        ("force_index", I, "volume", force_index, "13-EMA of price change x volume", "Force Index"),
        ("ease_of_movement", I, "volume", ease_of_movement, "Ease of movement (14)", "Ease of Movement (EMV)"),
        ("pvo", I, "volume", lambda df: pvo(df["volume"]), "Percentage volume oscillator 12/26/9", "Percentage Volume Oscillator (PVO)"),
        ("nvi", I, "volume", nvi, "Negative volume index with 255-EMA", "Negative Volume Index (NVI)"),
        ("pvi", I, "volume", pvi, "Positive volume index with 255-EMA", "Positive Volume Index (PVI)"),
        ("pvt", I, "volume", pvt, "Price volume trend", "Price Volume Trend (PVT)"),
        ("klinger", I, "volume", klinger, "Klinger volume oscillator 34/55/13", "Klinger Oscillator"),
        ("mass_index", I, "volatility", mass_index, "Mass index (9/25); reversal bulge >27 then <26.5", "Mass Index"),
        ("ulcer_index", I, "volatility", _cl(ulcer_index), "Ulcer index (14): RMS drawdown", "Ulcer Index (UI)"),
        ("choppiness", I, "volatility", choppiness, "Choppiness index (14); >61.8 range, <38.2 trend", "Choppiness Index"),
        ("hv_close", I, "volatility", _sel(hist_vol, "close"), "Close-to-close historical vol (20, annualised)", "Historical Volatility (HV)"),
        ("hv_parkinson", I, "volatility", _sel(hist_vol, "parkinson"), "Parkinson high-low vol (20)", "Historical Volatility (HV)"),
        ("hv_garman_klass", I, "volatility", _sel(hist_vol, "garman_klass"), "Garman-Klass OHLC vol (20)", "Historical Volatility (HV)"),
        ("hv_yang_zhang", I, "volatility", _sel(hist_vol, "yang_zhang"), "Yang-Zhang overnight+intraday vol (20)", "Historical Volatility (HV)"),
        ("pivot_camarilla", I, "levels", lambda df: pivot_levels(df, "camarilla"), "Camarilla pivots R1-R4/S1-S4", "Camarilla Pivot Points"),
        ("pivot_woodie", I, "levels", lambda df: pivot_levels(df, "woodie"), "Woodie pivots (close-weighted)", "Woodie's Pivot Points"),
        ("pivot_fibonacci", I, "levels", lambda df: pivot_levels(df, "fibonacci"), "Fibonacci pivots (0.382/0.618/1.0 of range)", "Fibonacci Pivot Points"),
        ("market_profile", I, "levels", market_profile, "Volume-at-price: POC and 70% value area", "Market Profile / Point of Control"),
        ("zigzag", I, "levels", zigzag, "ATR/percentage zigzag swings labelled HH/HL/LH/LL (no wave counts)", "Zig Zag Indicator / Elliott Wave Theory"),
    ]
    rows += [(k, Cn, "pattern", _sel(candles, k), v[2], v[1]) for k, v in CANDLES.items()]
    rows += [(k, Ch, "pattern", _chart(k), v[2], v[1]) for k, v in CHART_PATTERNS.items()]
    rows += [(k, Ch, "pattern", _sel(bar_patterns, k), v[2], v[1]) for k, v in BAR_PATTERNS.items() if k not in ("gap_up", "gap_down")]
    return {n: {"kind": k, "fn": f, "group": g, "summary": s, "investopedia": t} for n, k, g, f, s, t in rows}


def psar(df: pd.DataFrame, step: float = 0.02, max_af: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR with trend (+1 long / -1 short)."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    sar, trend = np.full(n, np.nan), np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"sar": sar, "trend": trend}, index=df.index)
    up = c[1] >= c[0]
    ep, s, af = (h[0], l[0], step) if up else (l[0], h[0], step)
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_af)
        sar[i], trend[i] = s, 1 if up else -1
    return pd.DataFrame({"sar": sar, "trend": trend}, index=df.index)


CATALOG = _mk()
GROUPS = ("trend", "momentum", "volume", "volatility", "levels", "pattern")


def catalog_table(group: str | None = None, kind: str | None = None) -> pd.DataFrame:
    t = pd.DataFrame([{"name": k, **{c: v[c] for c in ("kind", "group", "investopedia", "summary")}} for k, v in CATALOG.items()])
    if group:
        t = t[t["group"] == group]
    if kind:
        t = t[t["kind"] == kind]
    return t.set_index("name")


# ---- one-frame compute & latest readings -------------------------------------------------------------
def compute_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + ta.compute_all columns + every catalogue indicator as prefixed columns."""
    c = df["close"]
    base = ta.compute_all(df)
    base[["st_supertrend", "st_trend"]] = supertrend(df).to_numpy()  # NaN-safe replacement (see supertrend)
    st = ta.stoch(df)
    ich = ta.ichimoku(df).drop(columns="chikou")  # chikou is shifted forward (look-ahead); excluded
    parts = [
        pd.DataFrame({"stoch_k": st["k"], "stoch_d": st["d"], "roc10": ta.roc(c, 10), "obv": ta.obv(df), "mfi14": ta.mfi(df),
                      "vwap": _vwap(df), "wma20": wma(c), "hma20": hma(c), "kama10": kama(c), "dema20": dema(c),
                      "tema20": tema(c), "zlema20": zlema(c), "cci20": cci(df), "coppock": coppock(c), "uo": ultimate(df),
                      "willr14": williams_r(df), "cmo14": cmo(c), "dpo20": dpo(c), "bop": bop(df), "bop14": sma(bop(df), 14),
                      "ad": ad_line(df), "chaikin_osc": chaikin_osc(df), "cmf20": cmf(df), "force13": force_index(df),
                      "eom14": ease_of_movement(df), "pvt": pvt(df), "mass_index": mass_index(df), "ulcer14": ulcer_index(c),
                      "chop14": choppiness(df), "stc": schaff(c), "dow_trend": dow_trend(df), "wyckoff": wyckoff_phase(df)},
                     index=df.index),
        ta.keltner(df).add_prefix("kc_"), ta.donchian(df, 20).add_prefix("dc20_"), ta.donchian(df, 55).add_prefix("dc55_"),
        ich.add_prefix("ich_"), envelopes(c).add_prefix("env_"), aroon(df).add_prefix("aroon_"), elder_ray(df).add_prefix("elder_"),
        kst(c).add_prefix("kst_"), psar(df).add_prefix("psar_"), ppo(c).add_prefix("ppo_"), pvo(df["volume"]).add_prefix("pvo_"),
        trix(c).add_prefix("trix_"), tsi(c).add_prefix("tsi_"), vortex(df).add_prefix("vi_"), stoch_rsi(c).add_prefix("srsi_"),
        nvi(df).add_prefix("nvi_"), pvi(df).add_prefix("pvi_"), klinger(df).add_prefix("kvo_"), hist_vol(df).add_prefix("hv_"),
        linreg(c).add_prefix("lr_"), heikin_ashi(df).add_prefix("ha_"),
    ]
    return pd.concat([base] + parts, axis=1)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _cross(a, b, a0, b0, names=("bullish cross", "bearish cross", "above", "below")):
    if a > b and a0 <= b0:
        return names[0]
    if a < b and a0 >= b0:
        return names[1]
    return names[2] if a > b else names[3]


def _zone(v, hi, lo, labels=("overbought", "oversold", "neutral")):
    return labels[0] if v >= hi else labels[1] if v <= lo else labels[2]


def _pctile(s: pd.Series, n: int = 252) -> float:
    w = s.dropna().tail(n)
    return float((w < w.iloc[-1]).mean()) if len(w) > 20 else np.nan


def latest_readings(df: pd.DataFrame) -> dict:
    """{name: {group, value, state}} for the last bar across the indicator catalogue.
    `state` is a plain description (overbought, bullish cross, above zero...), 'n/a' when history is too short."""
    x = compute_catalog(df)
    out = {}
    if len(x) < 2:
        return out
    r, p = x.iloc[-1], x.iloc[-2]
    c = _f(r["close"])

    def put(name, group, value, state, *need):
        vals = [value, *need] if not isinstance(value, str) else list(need)
        ok = all(np.isfinite(_f(v)) for v in vals)
        try:
            st = state() if ok else "n/a (insufficient history)"
        except Exception:  # noqa: BLE001  - a reading must never break the table
            st = "n/a"
        out[name] = {"group": group, "value": value if isinstance(value, str) else _f(value), "state": st}

    g = lambda k: _f(r[k])  # noqa: E731
    q = lambda k: _f(p[k])  # noqa: E731
    above = lambda v: f"price {'above' if c > v else 'below'} ({c / v - 1:+.1%})"  # noqa: E731
    slope = lambda k: "rising" if g(k) > q(k) else "falling"  # noqa: E731
    sign = lambda v, pos="above zero", neg="below zero": pos if v > 0 else neg  # noqa: E731

    # trend
    for k in ("sma20", "sma50", "sma200"):
        put(k, "trend", g(k), lambda k=k: f"{above(g(k))}, {slope(k)}")
    put("sma50_vs_200", "trend", g("sma50") - g("sma200"),
        lambda: _cross(g("sma50"), g("sma200"), q("sma50"), q("sma200"), ("golden cross", "death cross", "50 above 200", "50 below 200")),
        q("sma50"), q("sma200"))
    put("ema21", "trend", g("ema21"), lambda: f"{above(g('ema21'))}, {slope('ema21')}")
    for k in ("wma20", "hma20", "kama10", "dema20", "tema20", "zlema20"):
        put(k, "trend", g(k), lambda k=k: f"{above(g(k))}, {slope(k)}", q(k))
    put("macd", "trend", g("macd"), lambda: f"{_cross(g('macd'), g('signal'), q('macd'), q('signal'))} signal, {sign(g('macd'))}", g("signal"), q("signal"))
    put("adx_dmi", "trend", g("adx"), lambda: f"{'trending' if g('adx') > 25 else 'weak/no trend'}, {'+DI' if g('+di') > g('-di') else '-DI'} leads"
        + (" (DI cross today)" if np.sign(g('+di') - g('-di')) != np.sign(q('+di') - q('-di')) else ""), g("+di"), g("-di"))
    put("supertrend", "trend", g("st_supertrend"), lambda: ("uptrend" if g("st_trend") > 0 else "downtrend")
        + (" (flipped today)" if g("st_trend") != q("st_trend") else ""), g("st_trend"), q("st_trend"))
    put("parabolic_sar", "trend", g("psar_sar"), lambda: ("bullish (SAR below)" if g("psar_trend") > 0 else "bearish (SAR above)")
        + (" (flipped today)" if g("psar_trend") != q("psar_trend") else ""), g("psar_trend"), q("psar_trend"))
    put("ichimoku", "trend", g("ich_kijun"), lambda: ("above cloud" if c > max(g("ich_span_a"), g("ich_span_b")) else
                                                       "below cloud" if c < min(g("ich_span_a"), g("ich_span_b")) else "inside cloud")
        + f", TK {_cross(g('ich_tenkan'), g('ich_kijun'), q('ich_tenkan'), q('ich_kijun'))}",
        g("ich_span_a"), g("ich_span_b"), g("ich_tenkan"), q("ich_tenkan"), q("ich_kijun"))
    put("aroon", "trend", g("aroon_osc"), lambda: ("strong uptrend" if g("aroon_osc") > 50 else "strong downtrend" if g("aroon_osc") < -50 else "consolidating")
        + f", up/down {_cross(g('aroon_up'), g('aroon_down'), q('aroon_up'), q('aroon_down'))}", q("aroon_up"), q("aroon_down"))
    put("vortex", "trend", g("vi_plus") - g("vi_minus"), lambda: _cross(g("vi_plus"), g("vi_minus"), q("vi_plus"), q("vi_minus"),
                                                                       ("bullish cross", "bearish cross", "VI+ leads (bullish)", "VI- leads (bearish)")), q("vi_plus"), q("vi_minus"))
    put("ma_envelopes", "trend", g("env_mid"), lambda: "above upper envelope" if c > g("env_upper") else "below lower envelope" if c < g("env_lower") else "inside envelopes")
    put("linreg", "trend", g("lr_slope_pct"), lambda: f"slope {g('lr_slope_pct'):+.2%}/bar, R2 {g('lr_r2'):.2f}"
        + (" (strong fit)" if g("lr_r2") > 0.6 else "") + (", above +2sd" if c > g("lr_upper") else ", below -2sd" if c < g("lr_lower") else ""), g("lr_r2"))
    put("schaff", "trend", g("stc"), lambda: _zone(g("stc"), 75, 25, ("overbought (>75)", "oversold (<25)", "mid-cycle"))
        + (", crossed up through 25" if g("stc") > 25 >= q("stc") else ", crossed down through 75" if g("stc") < 75 <= q("stc") else ""), q("stc"))
    put("dow_theory", "trend", g("dow_trend"), lambda: {1: "uptrend (HH+HL)", -1: "downtrend (LH+LL)", 0: "mixed swings"}[int(g("dow_trend"))])
    put("wyckoff_phase", "trend", str(r["wyckoff"]) if isinstance(r["wyckoff"], str) else "n/a", lambda: "heuristic label")
    ha_bull = g("ha_close") > g("ha_open")
    put("heikin_ashi", "trend", g("ha_close"), lambda: ("bullish" if ha_bull else "bearish")
        + (" (no lower wick: strong)" if ha_bull and g("ha_low") >= g("ha_open") else " (no upper wick: strong)" if not ha_bull and g("ha_high") <= g("ha_open") else ""))
    rk = renko(df)
    if len(rk):
        d = rk["dir"].to_numpy()
        streak = int(len(d) - np.flatnonzero(d != d[-1])[-1] - 1) if (d != d[-1]).any() else len(d)
        put("renko", "trend", float(rk["close"].iloc[-1]), lambda: f"{'up' if d[-1] > 0 else 'down'} bricks x{streak} (box {rk.attrs['box']:.4g})")
    pf = point_figure(df)
    if len(pf):
        put("point_figure", "trend", float(pf["high"].iloc[-1] if pf["kind"].iloc[-1] == "X" else pf["low"].iloc[-1]),
            lambda: f"current column {pf['kind'].iloc[-1]} ({pf['boxes'].iloc[-1]} boxes, box {pf.attrs['box']:.4g})")
    # momentum
    put("rsi", "momentum", g("rsi14"), lambda: _zone(g("rsi14"), 70, 30))
    put("stochastic", "momentum", g("stoch_k"), lambda: f"{_zone(g('stoch_k'), 80, 20)}, %K {_cross(g('stoch_k'), g('stoch_d'), q('stoch_k'), q('stoch_d'))} %D", g("stoch_d"), q("stoch_k"), q("stoch_d"))
    put("williams_r", "momentum", g("willr14"), lambda: _zone(g("willr14"), -20, -80))
    put("stoch_rsi", "momentum", g("srsi_k"), lambda: _zone(g("srsi_k"), 80, 20))
    put("cci", "momentum", g("cci20"), lambda: _zone(g("cci20"), 100, -100, ("above +100 (strong/overbought)", "below -100 (weak/oversold)", "neutral")))
    put("cmo", "momentum", g("cmo14"), lambda: _zone(g("cmo14"), 50, -50))
    put("ultimate_oscillator", "momentum", g("uo"), lambda: _zone(g("uo"), 70, 30))
    put("mfi", "volume", g("mfi14"), lambda: _zone(g("mfi14"), 80, 20))
    put("roc", "momentum", g("roc10"), lambda: sign(g("roc10")))
    put("dpo", "momentum", g("dpo20"), lambda: sign(g("dpo20"), "above cycle mean", "below cycle mean"))
    put("coppock", "momentum", g("coppock"), lambda: f"{sign(g('coppock'))}, {slope('coppock')}"
        + (" (turned up below zero: classic buy)" if g("coppock") < 0 and g("coppock") > q("coppock") and q("coppock") <= _f(x["coppock"].iloc[-3]) else ""), q("coppock"))
    put("ppo", "momentum", g("ppo_line"), lambda: f"{_cross(g('ppo_line'), g('ppo_signal'), q('ppo_line'), q('ppo_signal'))} signal, {sign(g('ppo_line'))}", g("ppo_signal"), q("ppo_line"), q("ppo_signal"))
    for k, a_, b_ in (("trix", "trix_trix", "trix_signal"), ("kst", "kst_kst", "kst_signal"), ("tsi", "tsi_tsi", "tsi_signal")):
        put(k, "momentum", g(a_), lambda a_=a_, b_=b_: f"{_cross(g(a_), g(b_), q(a_), q(b_))} signal, {sign(g(a_))}", g(b_), q(a_), q(b_))
    put("elder_ray", "momentum", g("elder_bull"), lambda: "bulls in control (both powers > 0)" if g("elder_bear") > 0 else
        "bears in control (both powers < 0)" if g("elder_bull") < 0 else
        f"balanced; bear power {'rising' if g('elder_bear') > q('elder_bear') else 'falling'}", g("elder_bear"), q("elder_bear"))
    put("balance_of_power", "momentum", g("bop14"), lambda: sign(g("bop14"), "buyers dominate", "sellers dominate"))
    # volume
    for k, col in (("obv", "obv"), ("ad_line", "ad"), ("pvt", "pvt")):
        put(k, "volume", g(col), lambda col=col: "rising vs 20-bar avg" if g(col) > _f(x[col].tail(20).mean()) else "falling vs 20-bar avg")
    put("chaikin_osc", "volume", g("chaikin_osc"), lambda: _cross(g("chaikin_osc"), 0, q("chaikin_osc"), 0,
                                                                     ("crossed above zero", "crossed below zero", "above zero", "below zero")), q("chaikin_osc"))
    put("cmf", "volume", g("cmf20"), lambda: _zone(g("cmf20"), 0.05, -0.05, ("accumulation", "distribution", "neutral")))
    put("force_index", "volume", g("force13"), lambda: sign(g("force13"), "buying force", "selling force"))
    put("ease_of_movement", "volume", g("eom14"), lambda: sign(g("eom14"), "rising easily", "falling easily"))
    put("pvo", "volume", g("pvo_line"), lambda: sign(g("pvo_line"), "volume expanding", "volume contracting"))
    put("nvi", "volume", g("nvi_nvi"), lambda: sign(g("nvi_nvi") - g("nvi_signal"), "above 255-EMA (smart money bullish)", "below 255-EMA"), g("nvi_signal"))
    put("pvi", "volume", g("pvi_pvi"), lambda: sign(g("pvi_pvi") - g("pvi_signal"), "above 255-EMA (bullish)", "below 255-EMA"), g("pvi_signal"))
    put("klinger", "volume", g("kvo_kvo"), lambda: f"{_cross(g('kvo_kvo'), g('kvo_signal'), q('kvo_kvo'), q('kvo_signal'))} signal, {sign(g('kvo_kvo'))}", g("kvo_signal"), q("kvo_kvo"), q("kvo_signal"))
    put("vwap", "volume", g("vwap"), lambda: above(g("vwap")))
    # volatility
    put("atr", "volatility", g("atr14"), lambda: f"{g('atr14') / c:.2%} of price, {_pctile(x['atr14'] / x['close']):.0%} 1y percentile")
    sq = lambda: g("bb_width") <= _f(x["bb_width"].tail(126).quantile(0.1))  # noqa: E731
    put("bollinger", "volatility", g("bb_pct_b"), lambda: ("above upper band" if g("bb_pct_b") > 1 else "below lower band" if g("bb_pct_b") < 0 else f"%b {g('bb_pct_b'):.2f}")
        + (", squeeze (bottom-decile width)" if sq() else ""), g("bb_width"))
    put("keltner", "volatility", g("kc_mid"), lambda: "above upper channel" if c > g("kc_upper") else "below lower channel" if c < g("kc_lower") else "inside channel")
    put("donchian", "volatility", g("dc20_upper"), lambda: "at/above prior 20-bar high" if c >= q("dc20_upper") else "at/below prior 20-bar low" if c <= q("dc20_lower") else
        f"{(c - g('dc20_lower')) / (g('dc20_upper') - g('dc20_lower')):.0%} of 20-bar range", q("dc20_upper"), q("dc20_lower"))
    put("mass_index", "volatility", g("mass_index"), lambda: "reversal bulge (rose >27, now <26.5)" if _f(x["mass_index"].tail(25).max()) > 27 and g("mass_index") < 26.5
        else "bulge forming (>27)" if g("mass_index") > 27 else "normal")
    put("ulcer_index", "volatility", g("ulcer14"), lambda: f"{_pctile(x['ulcer14']):.0%} 1y percentile")
    put("choppiness", "volatility", g("chop14"), lambda: "choppy/range (>61.8)" if g("chop14") > 61.8 else "trending (<38.2)" if g("chop14") < 38.2 else "neutral")
    for k in ("close", "parkinson", "garman_klass", "yang_zhang"):
        put(f"hv_{k}", "volatility", g(f"hv_{k}"), lambda k=k: f"{_pctile(x[f'hv_{k}']):.0%} 1y percentile")
    # levels
    for m in ("classic", "camarilla", "woodie", "fibonacci"):
        lv = pivot_levels(df, m)
        put(f"pivot_{m}", "levels", lv["P"], lambda lv=lv: _between(c, lv))
    fib = ta.fib_retracements(df)
    fl = {k: v for k, v in fib.items() if k[0] == "0"}
    put("fib_retracements", "levels", c, lambda: f"{fib['leg']} leg; nearest {min(fl, key=lambda k: abs(fl[k] - c))} at {fl[min(fl, key=lambda k: abs(fl[k] - c))]:.4g}")
    mp = market_profile(df, lookback=min(len(df), 252))
    put("market_profile", "levels", mp["poc"], lambda: f"{'above value area' if c > mp['vah'] else 'below value area' if c < mp['val'] else 'inside value area'}"
        f" (VAL {mp['val']:.4g} / VAH {mp['vah']:.4g})", mp["vah"], mp["val"])
    hi52, lo52 = _f(df["high"].tail(252).max()), _f(df["low"].tail(252).min())
    put("range_52w", "levels", c / hi52 - 1, lambda: f"{c / hi52 - 1:+.1%} from 52w high, {c / lo52 - 1:+.1%} from 52w low")
    zz = zigzag(df)
    if len(zz):
        lp = zz.iloc[-1]
        put("zigzag", "levels", float(lp["price"]), lambda: f"last confirmed swing {lp['label']} ({lp['kind']}) at {lp['time']:%Y-%m-%d}; leg now {'down' if lp['kind'] == 'high' else 'up'}")
    sr = ta.support_resistance(df)
    if len(sr):
        sup, res = sr[sr["kind"] == "support"], sr[sr["kind"] == "resistance"]
        put("support_resistance", "levels", c, lambda: "; ".join(
            ([f"nearest support {sup['level'].max():.4g}"] if len(sup) else []) +
            ([f"nearest resistance {res['level'].min():.4g}"] if len(res) else [])) or "none")
    return out


def _between(c, lv: dict) -> str:
    items = sorted(((v, k) for k, v in lv.items() if np.isfinite(_f(v))))
    below = [k for v, k in items if v <= c]
    above_ = [k for v, k in items if v > c]
    return f"between {below[-1] if below else '-'} and {above_[0] if above_ else '-'}"


def readings_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.from_dict(latest_readings(df), orient="index")[["group", "value", "state"]]


def recent_candles(df: pd.DataFrame, bars: int = 5, cd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candles that fired in the last `bars` bars with direction and definition."""
    cd = candles(df) if cd is None else cd
    w = cd.tail(bars)
    rows = [{"time": t, "pattern": k, "bias": {1: "bullish", -1: "bearish", 0: "neutral"}[CANDLES[k][0]], "rule": CANDLES[k][2]}
            for t, row in w.iterrows() for k in w.columns if row[k]]
    return pd.DataFrame(rows, columns=["time", "pattern", "bias", "rule"])


# ---- CLI ----------------------------------------------------------------------------------------------
_SHOW_PCOLS = ["pattern", "direction", "status", "start", "end", "confirm_time", "breakout_level", "target", "invalidation", "levels"]


def _ptable(p: pd.DataFrame) -> pd.DataFrame:
    t = p[_SHOW_PCOLS].copy()
    for k in ("start", "end", "confirm_time"):
        t[k] = [x.strftime("%Y-%m-%d %H:%M").replace(" 00:00", "") if pd.notna(x) else "" for x in t[k]]
    t["direction"] = t["direction"].map({1: "bull", -1: "bear", 0: "either"})
    return t.set_index("pattern")


def cmd_catalog(a):
    from .cli import show
    t = catalog_table(a.group, a.kind)
    if a.json:
        show(t, True)
        return
    for g in GROUPS:
        sub = t[t["group"] == g]
        if len(sub):
            show(sub.drop(columns="group"), title=f"{g} ({len(sub)})")
    print(f"\n{len(t)} entries · " + " · ".join(f"{k}: {v}" for k, v in t["kind"].value_counts().items()))


def cmd_tafull(a):
    from .cli import show, src_line
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period, refresh=a.refresh)
    rd = readings_table(df)
    if a.json:
        show({"readings": rd, "candles": recent_candles(df), "patterns": active_patterns(df)}, True)
        return
    for g in ("trend", "momentum", "volume", "volatility", "levels"):
        show(rd[rd["group"] == g].drop(columns="group"), title=f"{a.symbol} {a.interval} · {g}")
    show(recent_candles(df).set_index("time"), title="Candlestick patterns in the last 5 bars")
    bp = bar_patterns(df).tail(5)
    fired = [f"{t:%Y-%m-%d}: {k}" for t, row in bp.iterrows() for k in bp.columns if row[k]]
    print("\n**Bar patterns (last 5 bars):** " + ("; ".join(fired) if fired else "none"))
    ap = active_patterns(df)
    show(_ptable(ap) if len(ap) else ap, title="Active chart patterns (pending, or confirmed in the last 10 bars)")
    src_line(df)


def cmd_patterns(a):
    from .cli import show, src_line
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period, refresh=a.refresh)
    cd = candles(df)
    cnt = cd.sum()
    last = {k: cd.index[cd[k].to_numpy()][-1] for k in cd.columns if cnt[k]}
    ct = pd.DataFrame({"count": cnt, "last": pd.Series(last), "bias": {k: v[0] for k, v in CANDLES.items()}})
    ct = ct[ct["count"] > 0].sort_values("count", ascending=False)
    p = chart_patterns(df)
    if a.json:
        show({"candles": ct, "chart_patterns": p}, True)
        return
    show(ct, title=f"{a.symbol} {a.interval} candlestick patterns ({len(df)} bars)")
    if not a.gaps:
        p = p[~p["pattern"].str.endswith("_gap")]
    show(_ptable(p.tail(a.tail)) if len(p) else p, title=f"Chart patterns (last {a.tail} detections; gaps {'on' if a.gaps else 'off'})")
    if len(p):
        print("\n" + " · ".join(f"{k}: {v}" for k, v in p["status"].value_counts().items()))
    src_line(df)


def _win(q, period):
    q.add_argument("--interval", "-i", default="1d")
    q.add_argument("--period", "-p", default=period, help="1mo 3mo 6mo 1y 2y 5y 10y max")
    q.add_argument("--start")
    q.add_argument("--refresh", action="store_true", help="bypass cache")


def register(add):
    q = add("catalog", cmd_catalog, "list the technical-analysis catalogue (indicators, candles, chart patterns)")
    q.add_argument("--group", choices=GROUPS)
    q.add_argument("--kind", choices=["indicator", "candle", "chart"])
    q = add("tafull", cmd_tafull, "latest readings across the whole TA catalogue + recent candles + active chart patterns")
    q.add_argument("symbol"); _win(q, "2y")  # noqa: E702
    q = add("patterns", cmd_patterns, "candlestick and chart-pattern history over the window")
    q.add_argument("symbol"); _win(q, "2y")  # noqa: E702
    q.add_argument("--tail", type=int, default=40)
    q.add_argument("--gaps", action="store_true", help="include gap rows")
