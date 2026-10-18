"""Point-in-time feature cube: symbol x time x feature.

Every feature at bar t uses only bars up to and including t (decided at t's close). Features are scale-free
(returns, ATR units, ratios, bounded oscillators) so instruments pool into one phase space. Higher-timeframe
features (weekly for daily bars, daily for intraday bars) come from completed higher-timeframe bars only.

The look-ahead gate (`check_point_in_time`) recomputes the cube on truncated history and demands that every
value it already produced stays identical: a feature that changes when the future is cut off was using it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ta

# name -> (low, high) bounds the schema contract enforces; None = unbounded but finite
FEATURES: dict[str, tuple[float, float] | None] = {
    "ret_1": None,            # log return, 1 bar
    "ret_5": None,
    "ret_21": None,
    "ret_63": None,           # medium-term momentum
    "rsi_14": (-1.0, 1.0),    # (RSI - 50) / 50
    "macd_atr": None,         # MACD histogram in ATR units
    "atr_pct": (0.0, None),   # ATR(14) / close
    "rv_21": (0.0, None),     # realised vol of log returns, per bar
    "rv_ratio": None,         # log(rv_5 / rv_63): vol expanding (>0) or contracting (<0)
    "vol_z": None,            # log volume z-score over 20 bars (NaN where the venue has no volume)
    "dist_sma20": None,       # (close - SMA) / ATR
    "dist_sma50": None,
    "dist_sma200": None,
    "bb_pctb": None,          # Bollinger %b centred on 0 (-0.5 lower band, +0.5 upper band)
    "adx_14": (0.0, 1.0),
    "er_21": (0.0, 1.0),      # Kaufman efficiency ratio: 1 = straight line, 0 = pure chop
    "donch_55": (-0.5, 0.5),  # position in the 55-bar high-low channel
    "skew_63": None,
    "ac1_63": (-1.0, 1.0),    # lag-1 autocorrelation of returns over 63 bars
    "pe_63": (0.0, 1.0),      # normalised permutation entropy (order 3) of closes over 63 bars
    "htf_rsi": (-1.0, 1.0),   # higher timeframe, completed bars only
    "htf_dist_sma10": None,
    "htf_ret_4": None,
}

WARMUP = 200  # bars before every feature (SMA 200) is defined


def permutation_entropy(x: pd.Series, window: int = 63, order: int = 3) -> pd.Series:
    """Rolling normalised permutation entropy (Bandt & Pompe 2002). 0 = fully ordered, 1 = white noise."""
    v = x.to_numpy(float)
    n = len(v)
    out = np.full(n, np.nan)
    if n < order + window - 1:
        return pd.Series(out, index=x.index)
    lagged = np.column_stack([v[i:n - order + 1 + i] for i in range(order)])   # rows end at t = order-1..n-1
    ranks = np.argsort(lagged, axis=1, kind="stable")
    code = (ranks * order ** np.arange(order)[::-1]).sum(axis=1)
    _, idx = np.unique(code, return_inverse=True)
    onehot = np.zeros((len(code), idx.max() + 1))
    onehot[np.arange(len(code)), idx] = 1.0
    cs = np.vstack([np.zeros((1, onehot.shape[1])), onehot.cumsum(axis=0)])
    counts = cs[window:] - cs[:-window]                                           # windows of `window` patterns
    p = counts / window
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1) / np.log(float(np.prod(range(1, order + 1))))
    out[order - 1 + window - 1:] = h
    return pd.Series(out, index=x.index)


def load(ref: str, symbols: list[str] | None = None, split: str = "train+validation", min_bars: int = 750,
         log=print) -> dict[str, pd.DataFrame]:
    """Frozen dataset members that exist on disk with enough history; OHLC rescaled to the dividend-adjusted
    close where the snapshot carries one. The test split stays locked here: research runs on train+validation."""
    from .. import datasets
    if "test" in split.split("+"):
        raise PermissionError("phase research runs on train/validation only; the test split is graded once, "
                              "through `mlab hypo test`, on Cezar's word")
    name, ver, root = datasets.resolve(ref)
    man = datasets.manifest(ref)
    want = symbols or list(man["members"])
    present = [s for s in want if s in man["members"] and (root / man["members"][s]["path"]).exists()]
    missing = sorted(set(want) - set(present))
    if missing:
        log(f"_{name}@{ver}: skipped members not on disk or not in the snapshot: {', '.join(missing)}_")
    out = {}
    for s, df in datasets.load(ref, present, split).items():
        if "adj_close" in df and df["adj_close"].notna().all():
            k = df["adj_close"] / df["close"]
            df = df.assign(**{c: df[c] * k for c in ("open", "high", "low", "close")})
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) > 2 and pd.Series(df.index).diff().median() >= pd.Timedelta(hours=20):
            # daily bars carry exchange-local close times (05:00, 08:00 UTC...): key them by session date so
            # instruments share one calendar
            df = df.set_axis(df.index.normalize())
            df = df[~df.index.duplicated(keep="last")]
        hi, lo = df[["open", "high", "close"]].max(axis=1), df[["open", "low", "close"]].min(axis=1)
        fixed = int(((hi != df["high"]) | (lo != df["low"])).sum())
        if fixed:   # snapshot bars with the close or open outside high-low (quote feeds do this): widen the range
            log(f"_{s}: {fixed} bars had open/close outside high-low; range widened to contain them_")
            df = df.assign(high=hi, low=lo)
        if len(df) >= min_bars:
            out[s] = df
        else:
            log(f"_{s}: {len(df)} bars in {split}, below {min_bars}: skipped_")
    return out


def _htf_rule(index: pd.DatetimeIndex) -> tuple[str, bool]:
    step = pd.Series(index).diff().median()
    return ("1D", True) if step < pd.Timedelta(hours=20) else ("W-FRI", False)


def higher_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly features for daily bars, daily features for intraday bars, from completed buckets only.
    A bucket labelled L (its end) is visible on rows at or after L, so a partial bucket never leaks."""
    rule, intraday = _htf_rule(df.index)
    d = df.copy()
    if not intraday:
        d.index = d.index.normalize()       # daily stamps like 05:00 UTC must fall in their own calendar week
    kw = {"closed": "left", "label": "right"} if intraday else {"closed": "right", "label": "right"}
    agg = d.resample(rule, **kw).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h = pd.DataFrame(index=agg.index)
    h["htf_rsi"] = (ta.rsi(agg["close"], 14) - 50) / 50
    h["htf_dist_sma10"] = (agg["close"] - ta.sma(agg["close"], 10)) / ta.atr(agg, 14).replace(0, np.nan)
    h["htf_ret_4"] = np.log(agg["close"]).diff(4)
    keys = pd.DataFrame({"key": d.index, "row": np.arange(len(d))})
    right = h.reset_index().rename(columns={h.index.name or "index": "key"})
    right["key"] = right["key"].astype(keys["key"].dtype)
    m = pd.merge_asof(keys.sort_values("key"), right.sort_values("key"), on="key", direction="backward")
    m = m.sort_values("row")
    return pd.DataFrame(m[list(h.columns)].to_numpy(), index=df.index, columns=h.columns)


def features(df: pd.DataFrame, htf: bool = True) -> pd.DataFrame:
    """All features for one instrument's OHLCV frame (lowercase columns)."""
    c = df["close"].astype(float)
    lc = np.log(c)
    r = lc.diff()
    a = ta.atr(df, 14).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 21, 63):
        f[f"ret_{n}"] = lc.diff(n)
    f["rsi_14"] = (ta.rsi(c, 14) - 50) / 50
    f["macd_atr"] = ta.macd(c)["hist"] / a
    f["atr_pct"] = a / c
    f["rv_21"] = r.rolling(21).std()
    with np.errstate(divide="ignore"):
        f["rv_ratio"] = np.log(r.rolling(5).std() / r.rolling(63).std())
    vol = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    lv = np.log(vol.where(vol > 0))
    f["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = (c - ta.sma(c, n)) / a
    f["bb_pctb"] = ta.bollinger(c, 20)["pct_b"] - 0.5
    f["adx_14"] = ta.adx(df, 14)["adx"] / 100
    f["er_21"] = (c - c.shift(21)).abs() / c.diff().abs().rolling(21).sum()
    hi, lo = df["high"].rolling(55).max(), df["low"].rolling(55).min()
    f["donch_55"] = (c - lo) / (hi - lo).replace(0, np.nan) - 0.5
    f["skew_63"] = r.rolling(63).skew()
    f["ac1_63"] = r.rolling(63).corr(r.shift())
    f["pe_63"] = permutation_entropy(c, 63, 3)
    if htf:
        f = f.join(higher_timeframe(df))
    else:
        for k in ("htf_rsi", "htf_dist_sma10", "htf_ret_4"):
            f[k] = np.nan
    return f.replace([np.inf, -np.inf], np.nan)[list(FEATURES)].astype("float64")


def build(frames: dict[str, pd.DataFrame], htf: bool = True, drop_warmup: bool = True) -> pd.DataFrame:
    """The cube as a long frame indexed by (symbol, time)."""
    parts = []
    for sym, df in frames.items():
        f = features(df, htf)
        if drop_warmup:
            f = f.iloc[WARMUP:]
        f.index.name = "time"
        parts.append(pd.concat({sym: f}, names=["symbol"]))
    return pd.concat(parts).sort_index()


def to_array(cube: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DatetimeIndex, list[str]]:
    """Dense ticker x time x feature array (NaN where a ticker has no bar), with its axes."""
    syms = list(cube.index.get_level_values("symbol").unique())
    times = cube.index.get_level_values("time").unique().sort_values()
    arr = np.full((len(syms), len(times), cube.shape[1]), np.nan)
    for i, s in enumerate(syms):
        arr[i] = cube.xs(s, level="symbol").reindex(times).to_numpy()
    return arr, syms, times, list(cube.columns)


def validate(cube: pd.DataFrame) -> list[str]:
    """Schema contract. Returns violations; empty means the cube is fit for the regime model and evaluator."""
    bad = []
    if list(cube.index.names) != ["symbol", "time"]:
        bad.append(f"index names {cube.index.names} != ['symbol', 'time']")
    if list(cube.columns) != list(FEATURES):
        bad.append(f"columns differ from the contract: missing {sorted(set(FEATURES) - set(cube.columns))}, "
                   f"extra {sorted(set(cube.columns) - set(FEATURES))}")
    if not cube.index.is_unique:
        bad.append("duplicate (symbol, time) rows")
    for sym, g in cube.groupby(level="symbol"):
        t = g.index.get_level_values("time")
        if not t.is_monotonic_increasing:
            bad.append(f"{sym}: time not increasing")
        if t.tz is None:
            bad.append(f"{sym}: naive timestamps (want UTC)")
    for col, bounds in FEATURES.items():
        if col not in cube:
            continue
        s = cube[col]
        if s.dtype != "float64":
            bad.append(f"{col}: dtype {s.dtype}")
        if np.isinf(s.to_numpy()).any():
            bad.append(f"{col}: infinite values")
        if bounds:
            lo, hi = bounds
            v = s.dropna()
            if lo is not None and (v < lo - 1e-9).any():
                bad.append(f"{col}: {int((v < lo - 1e-9).sum())} values below {lo}")
            if hi is not None and (v > hi + 1e-9).any():
                bad.append(f"{col}: {int((v > hi + 1e-9).sum())} values above {hi}")
    return bad


def check_point_in_time(fn, df: pd.DataFrame, cuts: list[int] | None = None, rtol: float = 1e-9) -> dict[str, int]:
    """Look-ahead gate. `fn(frame) -> frame of features`. Recompute on df.iloc[:k] for each cut and compare
    every row before the cut with the full-history value. Returns {column: rows that changed}; empty = clean."""
    full = fn(df)
    n = len(df)
    cuts = cuts or [int(n * q) for q in (0.35, 0.6, 0.85)] + [n - 1]
    leaks: dict[str, int] = {}
    for k in cuts:
        part = fn(df.iloc[:k])
        a, b = part.to_numpy(float), full.iloc[:k][part.columns].to_numpy(float)
        same = np.isclose(a, b, rtol=rtol, atol=1e-12, equal_nan=True)
        for j, col in enumerate(part.columns):
            diff = int((~same[:, j]).sum())
            if diff:
                leaks[col] = max(leaks.get(col, 0), diff)
    return leaks
