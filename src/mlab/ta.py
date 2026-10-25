"""Technical analysis: pure pandas/numpy indicators (no TA-Lib needed), levels, regime, and a
one-call snapshot used for trade timing. All functions take an OHLCV frame (lowercase columns)."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- moving averages & momentum ---------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stoch(df: pd.DataFrame, k=14, d=3) -> pd.DataFrame:
    lo, hi = df["low"].rolling(k).min(), df["high"].rolling(k).max()
    kline = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"k": kline, "d": kline.rolling(d).mean()})


def roc(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n)


# ---- volatility & trend strength ----------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    m, sd = sma(close, n), close.rolling(n).std(ddof=0)  # population std, as Bollinger and TA-Lib define it
    up, lo = m + k * sd, m - k * sd
    return pd.DataFrame({"mid": m, "upper": up, "lower": lo, "pct_b": (close - lo) / (up - lo), "width": (up - lo) / m})


def keltner(df: pd.DataFrame, n=20, k=2.0) -> pd.DataFrame:
    m, a = ema(df["close"], n), atr(df, n)
    return pd.DataFrame({"mid": m, "upper": m + k * a, "lower": m - k * a})


def donchian(df: pd.DataFrame, n=20) -> pd.DataFrame:
    return pd.DataFrame({"upper": df["high"].rolling(n).max(), "lower": df["low"].rolling(n).min()})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    pdi, mdi = 100 * wilder(plus, n) / tr, 100 * wilder(minus, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "+di": pdi, "-di": mdi})


def supertrend(df: pd.DataFrame, n=10, mult=3.0) -> pd.DataFrame:
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    up_b, lo_b = (hl2 + mult * a).values, (hl2 - mult * a).values
    close = df["close"].values
    fu, fl = up_b.copy(), lo_b.copy()
    trend = np.ones(len(df))
    valid = np.flatnonzero(~np.isnan(up_b))
    first = valid[0] if len(valid) else len(df)
    for i in range(first + 1, len(df)):
        fu[i] = up_b[i] if (up_b[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lo_b[i] if (lo_b[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        trend[i] = 1 if close[i] > fu[i - 1] else -1 if close[i] < fl[i - 1] else trend[i - 1]
    line = np.where(trend > 0, fl, fu)
    out = pd.DataFrame({"supertrend": line, "trend": trend}, index=df.index)
    out.loc[a.isna(), :] = np.nan
    return out


def ichimoku(df: pd.DataFrame, t=9, k=26, s=52) -> pd.DataFrame:
    mid = lambda n: (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2  # noqa: E731
    ten, kij = mid(t), mid(k)
    return pd.DataFrame({"tenkan": ten, "kijun": kij, "span_a": ((ten + kij) / 2).shift(k),
                         "span_b": mid(s).shift(k), "chikou": df["close"].shift(-k)})


# ---- volume ----------------------------------------------------------------------
def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"].fillna(0)).cumsum()


def vwap(df: pd.DataFrame, anchor: str | None = "D") -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv, v = tp * df["volume"], df["volume"]
    if anchor is None:
        return pv.cumsum() / v.cumsum()
    g = df.index.to_period(anchor)
    return pv.groupby(g).cumsum() / v.groupby(g).cumsum()


def anchored_vwap(df: pd.DataFrame, anchor_time) -> pd.Series:
    sub = df[df.index >= pd.Timestamp(anchor_time, tz=df.index.tz)]
    return vwap(sub, None).reindex(df.index)


def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(), 0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(), 0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


# ---- levels ------------------------------------------------------------------------
def pivots(df: pd.DataFrame) -> dict:
    """Classic floor pivots from the last completed bar (use daily for intraday, weekly for swing)."""
    h, l, c = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l),
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)}


def swing_points(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    hi, lo = df["high"], df["low"]
    is_hi = hi == hi.rolling(2 * k + 1, center=True).max()
    is_lo = lo == lo.rolling(2 * k + 1, center=True).min()
    return pd.DataFrame({"swing_high": hi.where(is_hi), "swing_low": lo.where(is_lo)})


def support_resistance(df: pd.DataFrame, k: int = 5, tol_atr: float = 0.5, top: int = 6) -> pd.DataFrame:
    """Cluster swing highs/lows within tol_atr*ATR; rank by touches and recency."""
    sp = swing_points(df, k)
    a = atr(df).iloc[-1]
    pts = pd.concat([sp["swing_high"].dropna(), sp["swing_low"].dropna()]).sort_values()
    if pts.empty or np.isnan(a):
        return pd.DataFrame(columns=["level", "touches", "last_touch", "kind"])
    clusters, cur = [], [pts.index[0]]
    vals = pts.values
    for i in range(1, len(vals)):
        if vals[i] - vals[i - 1] <= tol_atr * a:
            cur.append(pts.index[i])
        else:
            clusters.append(cur)
            cur = [pts.index[i]]
    clusters.append(cur)
    last = df["close"].iloc[-1]
    rows = []
    for c in clusters:
        lv = float(np.mean(pts[pts.index.isin(c)]))
        rows.append({"level": lv, "touches": len(c), "last_touch": max(c),
                     "kind": "resistance" if lv > last else "support"})
    out = pd.DataFrame(rows)
    out["dist_atr"] = (out["level"] - last) / a
    out["score"] = out["touches"] - out["dist_atr"].abs() * 0.3
    return out.sort_values("score", ascending=False).head(top).sort_values("level")


def fib_retracements(df: pd.DataFrame, lookback: int = 120) -> dict:
    w = df.tail(lookback)
    hi, lo = w["high"].max(), w["low"].min()
    up = w["high"].idxmax() > w["low"].idxmin()  # last leg up -> retrace down from high
    rng = hi - lo
    lv = {r: (hi - r * rng) if up else (lo + r * rng) for r in (0.236, 0.382, 0.5, 0.618, 0.786)}
    return {"swing_high": hi, "swing_low": lo, "leg": "up" if up else "down", **{f"{k:.3f}": v for k, v in lv.items()}}


# ---- regime & snapshot -------------------------------------------------------------
def regime(df: pd.DataFrame) -> dict:
    c = df["close"]
    s50, s200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
    ax = adx(df)["adx"].iloc[-1]
    rv20 = c.pct_change().tail(20).std() * np.sqrt(252)
    rv_hist = c.pct_change().rolling(20).std().mul(np.sqrt(252))
    vol_pct = (rv_hist.dropna() < rv20).mean() if rv_hist.notna().sum() > 50 else np.nan
    trend = ("up" if c.iloc[-1] > s200 and s50 > s200 else "down" if c.iloc[-1] < s200 and s50 < s200 else "mixed") \
        if not np.isnan(s200) else ("up" if c.iloc[-1] > s50 else "down") if not np.isnan(s50) else "n/a"
    return {"trend": trend, "trending": bool(ax > 25) if not np.isnan(ax) else None, "adx": ax,
            "realized_vol_20d": rv20, "vol_percentile": vol_pct}


def snapshot(df: pd.DataFrame) -> dict:
    """Everything needed to time an entry, as one dict of latest values."""
    c = df["close"]
    last = float(c.iloc[-1])
    a = float(atr(df).iloc[-1])
    m = macd(c).iloc[-1]
    bb = bollinger(c).iloc[-1]
    st = supertrend(df).iloc[-1]
    hi52, lo52 = df["high"].tail(252).max(), df["low"].tail(252).min()
    out = {
        "last": last, "last_bar": str(df.index[-1]),
        "chg_1d": c.pct_change().iloc[-1], "chg_1w": roc(c, 5).iloc[-1], "chg_1m": roc(c, 21).iloc[-1],
        "chg_3m": roc(c, 63).iloc[-1], "chg_1y": roc(c, 252).iloc[-1] if len(c) > 252 else np.nan,
        "sma20": sma(c, 20).iloc[-1], "sma50": sma(c, 50).iloc[-1], "sma200": sma(c, 200).iloc[-1],
        "ema21": ema(c, 21).iloc[-1],
        "rsi14": rsi(c).iloc[-1], "macd": m["macd"], "macd_signal": m["signal"], "macd_hist": m["hist"],
        "stoch_k": stoch(df)["k"].iloc[-1], "atr14": a, "atr_pct": a / last,
        "bb_pct_b": bb["pct_b"], "bb_width": bb["width"],
        "supertrend": st["supertrend"], "supertrend_dir": "up" if st["trend"] > 0 else "down",
        "high_52w": hi52, "low_52w": lo52, "pct_from_52w_high": last / hi52 - 1,
        **{f"regime_{k}": v for k, v in regime(df).items()},
    }
    if df["volume"].notna().sum() > 20 and df["volume"].tail(20).sum() > 0:
        out["rel_volume"] = df["volume"].iloc[-1] / df["volume"].tail(21).iloc[:-1].mean()
        out["mfi14"] = mfi(df).iloc[-1]
    out["signals"] = signals(df, out)
    return out


def signals(df: pd.DataFrame, snap: dict | None = None) -> list[str]:
    """Plain-language flags. Facts only; the call is made by whoever reads them."""
    c = df["close"]
    s = snap or {}
    flags = []
    s50, s200 = sma(c, 50), sma(c, 200)
    if len(c) > 201:
        if s50.iloc[-1] > s200.iloc[-1] and s50.iloc[-6] <= s200.iloc[-6]:
            flags.append("golden cross within 5 bars")
        if s50.iloc[-1] < s200.iloc[-1] and s50.iloc[-6] >= s200.iloc[-6]:
            flags.append("death cross within 5 bars")
    r = s.get("rsi14", rsi(c).iloc[-1])
    if r >= 70:
        flags.append(f"RSI overbought ({r:.0f})")
    elif r <= 30:
        flags.append(f"RSI oversold ({r:.0f})")
    h = macd(c)["hist"]
    if h.iloc[-1] > 0 >= h.iloc[-2]:
        flags.append("MACD bullish cross")
    if h.iloc[-1] < 0 <= h.iloc[-2]:
        flags.append("MACD bearish cross")
    dc = donchian(df, 55)
    if c.iloc[-1] >= dc["upper"].shift().iloc[-1]:
        flags.append("55-bar breakout high (turtle)")
    if c.iloc[-1] <= dc["lower"].shift().iloc[-1]:
        flags.append("55-bar breakdown low (turtle)")
    bb = bollinger(c)
    if bb["width"].iloc[-1] <= bb["width"].tail(126).quantile(0.1):
        flags.append("Bollinger squeeze (bottom decile width, 6m)")
    # RSI divergence over the last ~30 bars
    w = df.tail(30)
    if len(w) == 30:
        rr = rsi(c).tail(30)
        if w["close"].iloc[-1] >= w["close"].max() * 0.995 and rr.iloc[-1] < rr.max() - 5:
            flags.append("bearish RSI divergence (price at high, RSI lower)")
        if w["close"].iloc[-1] <= w["close"].min() * 1.005 and rr.iloc[-1] > rr.min() + 5:
            flags.append("bullish RSI divergence (price at low, RSI higher)")
    return flags


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Frame with the common indicator columns appended (for charts, screens, backtests)."""
    out = df.copy()
    c = df["close"]
    for n in (20, 50, 200):
        out[f"sma{n}"] = sma(c, n)
    out["ema21"] = ema(c, 21)
    out["rsi14"] = rsi(c)
    out = out.join(macd(c)).join(bollinger(c).add_prefix("bb_")).join(adx(df))
    out["atr14"] = atr(df)
    out = out.join(supertrend(df).add_prefix("st_"))
    return out
