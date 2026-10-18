"""Signal lab: discrete trading signals as boolean event series, event studies against the unconditional
base rate, and multiple-testing-aware scorecards: "what is firing now, and has it actually worked here?".

Every signal on bar t uses data up to and including bar t only. Returns are measured from the close of t
(or the next open) to the close of t+h, multiplied by the signal direction (+1 long, -1 short)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _st

from . import ta
from . import ta_catalog as tc


@dataclass(frozen=True)
class Signal:
    name: str
    direction: int  # +1 long, -1 short
    group: str
    desc: str
    fn: Callable  # (Ctx) -> bool Series


SIGNALS: dict[str, Signal] = {}


def _add(name, direction, group, desc, fn):
    SIGNALS[name] = Signal(name, direction, group, desc, fn)


# ---- lazy indicator cache --------------------------------------------------------------
_IND = {
    "c": lambda d: d["close"], "h": lambda d: d["high"], "l": lambda d: d["low"],
    "sma20": lambda d: ta.sma(d["close"], 20), "sma50": lambda d: ta.sma(d["close"], 50), "sma200": lambda d: ta.sma(d["close"], 200),
    "macd": lambda d: ta.macd(d["close"]), "rsi": lambda d: ta.rsi(d["close"]), "stoch": ta.stoch, "adx": ta.adx,
    "st": tc.supertrend, "psar": tc.psar, "aroon": tc.aroon, "ich": ta.ichimoku, "bb": lambda d: ta.bollinger(d["close"]),
    "kc": ta.keltner, "obv": ta.obv, "cmf": tc.cmf, "chaikin": tc.chaikin_osc, "kvo": tc.klinger, "willr": tc.williams_r,
    "cci": tc.cci, "srsi": lambda d: tc.stoch_rsi(d["close"]), "mfi": ta.mfi, "uo": tc.ultimate,
    "trix": lambda d: tc.trix(d["close"]), "kst": lambda d: tc.kst(d["close"]), "tsi": lambda d: tc.tsi(d["close"]),
    "vortex": tc.vortex, "coppock": lambda d: tc.coppock(d["close"]), "stc": lambda d: tc.schaff(d["close"]),
    "bars": tc.bar_patterns, "candles": tc.candles, "charts": tc.chart_patterns, "dow": tc.dow_trend,
    "trend": tc.trend_context,
}


class Ctx(dict):
    """Per-frame indicator cache shared by all signal functions (computed on first use)."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def __missing__(self, k):
        v = self[k] = _IND[k](self.df)
        return v


def xup(a: pd.Series, b) -> pd.Series:
    """a crosses above b on this bar (b may be a scalar)."""
    bp = b.shift() if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift() <= bp)


def xdn(a: pd.Series, b) -> pd.Series:
    bp = b.shift() if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift() >= bp)


def _first(cond: pd.Series) -> pd.Series:
    """True only on the first bar of a run of True."""
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(fill_value=False).astype(bool)


def _flip(trend: pd.Series, to: int) -> pd.Series:
    return (trend == to) & (trend.shift() == -to)


def _divergence(x: Ctx, bull: bool, k: int = 3, lookback: int = 60) -> pd.Series:
    """Causal RSI divergence: a swing extreme of the close at t-k is confirmed at t (k bars either side);
    bullish = lower low with RSI at least 2 pts higher than at the previous swing low (prev RSI < 40)."""
    c, r = x["c"], x["rsi"]
    win = c.rolling(2 * k + 1)
    conf = c.shift(k) == (win.min() if bull else win.max())
    pos = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    pv, pr, pp = c.shift(k).where(conf), r.shift(k).where(conf), (pos - k).where(conf)
    prv, prr, prp = pv.shift().ffill(), pr.shift().ffill(), pp.shift().ffill()
    near = (pp - prp) <= lookback
    if bull:
        return conf & near & (pv < prv) & (pr > prr + 2) & (prr < 40)
    return conf & near & (pv > prv) & (pr < prr - 2) & (prr > 60)


def _squeeze_break(x: Ctx, up: bool) -> pd.Series:
    bb = x["bb"]
    q = bb["width"].rolling(126, min_periods=60).quantile(0.2)
    recent = (bb["width"] <= q).astype(float).shift().rolling(5).sum() > 0
    return (xup(x["c"], bb["upper"]) if up else xdn(x["c"], bb["lower"])) & recent


def _chart_events(x: Ctx, pattern: str, direction: int | None) -> pd.Series:
    p = x["charts"]
    m = (p["pattern"] == pattern) & p["confirmed"]
    if direction is not None:
        m &= p["direction"] == direction
    ev = np.zeros(len(x.df), dtype=bool)
    ev[p.loc[m, "confirm_i"].dropna().astype(int).to_numpy()] = True
    return pd.Series(ev, index=x.df.index)


def _build():
    A = _add
    # trend
    A("macd_bull_cross", 1, "trend", "MACD line crosses above signal", lambda x: xup(x["macd"]["macd"], x["macd"]["signal"]))
    A("macd_bear_cross", -1, "trend", "MACD line crosses below signal", lambda x: xdn(x["macd"]["macd"], x["macd"]["signal"]))
    A("macd_zero_up", 1, "trend", "MACD line crosses above zero", lambda x: xup(x["macd"]["macd"], 0))
    A("macd_zero_down", -1, "trend", "MACD line crosses below zero", lambda x: xdn(x["macd"]["macd"], 0))
    A("golden_cross", 1, "trend", "SMA50 crosses above SMA200", lambda x: xup(x["sma50"], x["sma200"]))
    A("death_cross", -1, "trend", "SMA50 crosses below SMA200", lambda x: xdn(x["sma50"], x["sma200"]))
    A("close_x_sma50_up", 1, "trend", "close crosses above SMA50", lambda x: xup(x["c"], x["sma50"]))
    A("close_x_sma50_down", -1, "trend", "close crosses below SMA50", lambda x: xdn(x["c"], x["sma50"]))
    A("close_x_sma200_up", 1, "trend", "close crosses above SMA200", lambda x: xup(x["c"], x["sma200"]))
    A("close_x_sma200_down", -1, "trend", "close crosses below SMA200", lambda x: xdn(x["c"], x["sma200"]))
    A("supertrend_flip_up", 1, "trend", "Supertrend(10,3) flips to up", lambda x: _flip(x["st"]["trend"], 1))
    A("supertrend_flip_down", -1, "trend", "Supertrend(10,3) flips to down", lambda x: _flip(x["st"]["trend"], -1))
    A("psar_flip_up", 1, "trend", "Parabolic SAR flips below price", lambda x: _flip(x["psar"]["trend"], 1))
    A("psar_flip_down", -1, "trend", "Parabolic SAR flips above price", lambda x: _flip(x["psar"]["trend"], -1))
    A("adx_di_bull", 1, "trend", "+DI crosses above -DI with ADX > 25", lambda x: xup(x["adx"]["+di"], x["adx"]["-di"]) & (x["adx"]["adx"] > 25))
    A("adx_di_bear", -1, "trend", "-DI crosses above +DI with ADX > 25", lambda x: xdn(x["adx"]["+di"], x["adx"]["-di"]) & (x["adx"]["adx"] > 25))
    A("aroon_bull_cross", 1, "trend", "Aroon up crosses above Aroon down", lambda x: xup(x["aroon"]["up"], x["aroon"]["down"]))
    A("aroon_bear_cross", -1, "trend", "Aroon down crosses above Aroon up", lambda x: xdn(x["aroon"]["up"], x["aroon"]["down"]))
    A("ichimoku_tk_bull", 1, "trend", "Tenkan crosses above Kijun", lambda x: xup(x["ich"]["tenkan"], x["ich"]["kijun"]))
    A("ichimoku_tk_bear", -1, "trend", "Tenkan crosses below Kijun", lambda x: xdn(x["ich"]["tenkan"], x["ich"]["kijun"]))
    A("kumo_breakout_up", 1, "trend", "close crosses above the cloud top",
      lambda x: xup(x["c"], np.maximum(x["ich"]["span_a"], x["ich"]["span_b"])))
    A("kumo_breakout_down", -1, "trend", "close crosses below the cloud bottom",
      lambda x: xdn(x["c"], np.minimum(x["ich"]["span_a"], x["ich"]["span_b"])))
    for n in (20, 55):
        A(f"donchian{n}_breakout", 1, "trend", f"first close above the prior {n}-bar high",
          lambda x, n=n: _first(x["c"] > x["h"].rolling(n).max().shift()))
        A(f"donchian{n}_breakdown", -1, "trend", f"first close below the prior {n}-bar low",
          lambda x, n=n: _first(x["c"] < x["l"].rolling(n).min().shift()))
    A("high_52w_breakout", 1, "trend", "first close above the prior 252-bar high", lambda x: _first(x["bars"]["high_52w_breakout"]))
    A("low_52w_breakdown", -1, "trend", "first close below the prior 252-bar low", lambda x: _first(x["bars"]["low_52w_breakdown"]))
    A("dow_turn_up", 1, "trend", "Dow swing state turns up (HH+HL)", lambda x: (x["dow"] == 1) & (x["dow"].shift() != 1) & x["dow"].shift().notna())
    A("dow_turn_down", -1, "trend", "Dow swing state turns down (LH+LL)", lambda x: (x["dow"] == -1) & (x["dow"].shift() != -1) & x["dow"].shift().notna())
    A("vortex_bull_cross", 1, "trend", "VI+ crosses above VI-", lambda x: xup(x["vortex"]["plus"], x["vortex"]["minus"]))
    A("vortex_bear_cross", -1, "trend", "VI- crosses above VI+", lambda x: xdn(x["vortex"]["plus"], x["vortex"]["minus"]))
    A("schaff_up", 1, "trend", "Schaff trend cycle crosses up through 25", lambda x: xup(x["stc"], 25))
    A("schaff_down", -1, "trend", "Schaff trend cycle crosses down through 75", lambda x: xdn(x["stc"], 75))
    # momentum
    for nm, key, a, b in (("trix", "trix", "trix", "signal"), ("kst", "kst", "kst", "signal"), ("tsi", "tsi", "tsi", "signal")):
        A(f"{nm}_bull_cross", 1, "momentum", f"{nm.upper()} crosses above its signal", lambda x, k=key, a=a, b=b: xup(x[k][a], x[k][b]))
        A(f"{nm}_bear_cross", -1, "momentum", f"{nm.upper()} crosses below its signal", lambda x, k=key, a=a, b=b: xdn(x[k][a], x[k][b]))
    A("coppock_turn_up", 1, "momentum", "Coppock turns up while below zero",
      lambda x: (x["coppock"] < 0) & (x["coppock"] > x["coppock"].shift()) & (x["coppock"].shift() <= x["coppock"].shift(2)))
    A("coppock_turn_down", -1, "momentum", "Coppock turns down while above zero",
      lambda x: (x["coppock"] > 0) & (x["coppock"] < x["coppock"].shift()) & (x["coppock"].shift() >= x["coppock"].shift(2)))
    A("rsi_os_enter", 1, "momentum", "RSI(14) drops below 30 (mean-reversion long)", lambda x: xdn(x["rsi"], 30))
    A("rsi_os_exit", 1, "momentum", "RSI(14) climbs back above 30", lambda x: xup(x["rsi"], 30))
    A("rsi_ob_enter", -1, "momentum", "RSI(14) rises above 70 (mean-reversion short)", lambda x: xup(x["rsi"], 70))
    A("rsi_ob_exit", -1, "momentum", "RSI(14) falls back below 70", lambda x: xdn(x["rsi"], 70))
    A("rsi_bull_divergence", 1, "momentum", "close lower low, RSI higher low (swing confirmed 3 bars later)", lambda x: _divergence(x, True))
    A("rsi_bear_divergence", -1, "momentum", "close higher high, RSI lower high (swing confirmed 3 bars later)", lambda x: _divergence(x, False))
    A("stoch_bull_os", 1, "momentum", "%K crosses above %D with %D < 20", lambda x: xup(x["stoch"]["k"], x["stoch"]["d"]) & (x["stoch"]["d"] < 20))
    A("stoch_bear_ob", -1, "momentum", "%K crosses below %D with %D > 80", lambda x: xdn(x["stoch"]["k"], x["stoch"]["d"]) & (x["stoch"]["d"] > 80))
    A("willr_os_exit", 1, "momentum", "Williams %R crosses up through -80", lambda x: xup(x["willr"], -80))
    A("willr_ob_exit", -1, "momentum", "Williams %R crosses down through -20", lambda x: xdn(x["willr"], -20))
    A("cci_up100", 1, "momentum", "CCI crosses above +100 (momentum)", lambda x: xup(x["cci"], 100))
    A("cci_down100", -1, "momentum", "CCI crosses below -100 (momentum)", lambda x: xdn(x["cci"], -100))
    A("cci_os_exit", 1, "momentum", "CCI crosses back above -100", lambda x: xup(x["cci"], -100))
    A("cci_ob_exit", -1, "momentum", "CCI crosses back below +100", lambda x: xdn(x["cci"], 100))
    A("stochrsi_os_exit", 1, "momentum", "StochRSI %K crosses up through 20", lambda x: xup(x["srsi"]["k"], 20))
    A("stochrsi_ob_exit", -1, "momentum", "StochRSI %K crosses down through 80", lambda x: xdn(x["srsi"]["k"], 80))
    A("uo_os_exit", 1, "momentum", "Ultimate oscillator crosses up through 30", lambda x: xup(x["uo"], 30))
    A("uo_ob_exit", -1, "momentum", "Ultimate oscillator crosses down through 70", lambda x: xdn(x["uo"], 70))
    # volatility
    A("bb_squeeze_breakout_up", 1, "volatility", "close crosses upper band within 5 bars of a squeeze (width <= 20th pct, 6m)", lambda x: _squeeze_break(x, True))
    A("bb_squeeze_breakout_down", -1, "volatility", "close crosses lower band within 5 bars of a squeeze", lambda x: _squeeze_break(x, False))
    A("bb_reentry_up", 1, "volatility", "close back inside after closing below the lower band", lambda x: xup(x["c"], x["bb"]["lower"]))
    A("bb_reentry_down", -1, "volatility", "close back inside after closing above the upper band", lambda x: xdn(x["c"], x["bb"]["upper"]))
    A("keltner_breakout_up", 1, "volatility", "close crosses above the upper Keltner channel", lambda x: xup(x["c"], x["kc"]["upper"]))
    A("keltner_breakout_down", -1, "volatility", "close crosses below the lower Keltner channel", lambda x: xdn(x["c"], x["kc"]["lower"]))
    A("nr7_breakout_up", 1, "volatility", "close above the high of an NR7 bar (next bar)", lambda x: x["bars"]["nr7"].shift(fill_value=False).astype(bool) & (x["c"] > x["h"].shift()))
    A("nr7_breakout_down", -1, "volatility", "close below the low of an NR7 bar (next bar)", lambda x: x["bars"]["nr7"].shift(fill_value=False).astype(bool) & (x["c"] < x["l"].shift()))
    A("gap_up", 1, "volatility", "true gap up (low above prior high)", lambda x: x["bars"]["gap_up"])
    A("gap_down", -1, "volatility", "true gap down (high below prior low)", lambda x: x["bars"]["gap_down"])
    # volume
    A("obv_confirmed_breakout", 1, "volume", "first close above prior 20-bar high with OBV at a 20-bar high",
      lambda x: _first(x["c"] > x["h"].rolling(20).max().shift()) & (x["obv"] >= x["obv"].rolling(20).max()))
    A("obv_confirmed_breakdown", -1, "volume", "first close below prior 20-bar low with OBV at a 20-bar low",
      lambda x: _first(x["c"] < x["l"].rolling(20).min().shift()) & (x["obv"] <= x["obv"].rolling(20).min()))
    A("cmf_confirmed_sma50_up", 1, "volume", "close crosses above SMA50 with CMF(20) > 0", lambda x: xup(x["c"], x["sma50"]) & (x["cmf"] > 0))
    A("cmf_confirmed_sma50_down", -1, "volume", "close crosses below SMA50 with CMF(20) < 0", lambda x: xdn(x["c"], x["sma50"]) & (x["cmf"] < 0))
    A("chaikin_zero_up", 1, "volume", "Chaikin oscillator crosses above zero", lambda x: xup(x["chaikin"], 0))
    A("chaikin_zero_down", -1, "volume", "Chaikin oscillator crosses below zero", lambda x: xdn(x["chaikin"], 0))
    A("klinger_bull_cross", 1, "volume", "Klinger KVO crosses above its signal", lambda x: xup(x["kvo"]["kvo"], x["kvo"]["signal"]))
    A("klinger_bear_cross", -1, "volume", "Klinger KVO crosses below its signal", lambda x: xdn(x["kvo"]["kvo"], x["kvo"]["signal"]))
    A("mfi_os_exit", 1, "volume", "MFI crosses up through 20", lambda x: xup(x["mfi"], 20))
    A("mfi_ob_exit", -1, "volume", "MFI crosses down through 80", lambda x: xdn(x["mfi"], 80))
    # candles: directional ones as-is, neutral ones signed by the prior trend (reversal reading)
    for k, (d, _, rule) in tc.CANDLES.items():
        if d:
            A(f"candle_{k}", d, "candle", rule, lambda x, k=k: x["candles"][k])
        else:
            A(f"candle_{k}_after_up", -1, "candle", f"{rule}; after an uptrend (reversal short)",
              lambda x, k=k: x["candles"][k] & x["trend"]["up"].shift(fill_value=False).astype(bool))
            A(f"candle_{k}_after_down", 1, "candle", f"{rule}; after a downtrend (reversal long)",
              lambda x, k=k: x["candles"][k] & x["trend"]["down"].shift(fill_value=False).astype(bool))
    # chart patterns: confirmed breakouts (event on the confirmation bar)
    for k, (bias, _, rule) in tc.CHART_PATTERNS.items():
        if k.endswith("_gap"):
            for d, s in ((1, "up"), (-1, "down")):
                A(f"chart_{k}_{s}", d, "chart", f"{k.replace('_', ' ')} {s} ({rule})", lambda x, k=k, d=d: _chart_events(x, k, d))
        elif bias:
            A(f"chart_{k}", bias, "chart", f"confirmed breakout: {rule}", lambda x, k=k: _chart_events(x, k, None))
        else:
            for d, s in ((1, "up"), (-1, "down")):
                A(f"chart_{k}_{s}", d, "chart", f"confirmed {s} breakout: {rule}", lambda x, k=k, d=d: _chart_events(x, k, d))


_build()


def events(df: pd.DataFrame, names=None, ctx: Ctx | None = None) -> pd.DataFrame:
    """Boolean event frame, one column per signal. Signals that fail on this frame are all-False and
    listed in .attrs['errors']."""
    x = ctx or Ctx(df)
    names = list(SIGNALS) if names is None else list(names)
    cols, errs = {}, {}
    for n in names:
        try:
            s = SIGNALS[n].fn(x)
            cols[n] = pd.Series(s, index=df.index).fillna(False).astype(bool).to_numpy()
        except Exception as e:  # noqa: BLE001  - one broken signal must not kill the scorecard
            cols[n] = np.zeros(len(df), dtype=bool)
            errs[n] = f"{type(e).__name__}: {e}"[:120]
    out = pd.DataFrame(cols, index=df.index)
    out.attrs["errors"] = errs
    return out


# ---- event study ------------------------------------------------------------------------------
def _forward(df: pd.DataFrame, horizons, entry: str = "close", atr_n: int = 14) -> dict:
    """Per horizon: raw forward return from entry to close t+h, and adverse excursion (in ATR at t) for
    longs (lowest low) and shorts (highest high) over bars t+1..t+h."""
    c, o = df["close"].to_numpy(float), df["open"].to_numpy(float)
    e = c if entry == "close" else np.r_[o[1:], np.nan]
    if entry not in ("close", "next_open"):
        raise ValueError("entry must be 'close' or 'next_open'")
    a = ta.atr(df, atr_n).to_numpy(float)
    n, out = len(c), {}
    for h in horizons:
        ex = np.r_[c[h:], np.full(min(h, n), np.nan)][:n]
        lo = df["low"].rolling(h).min().shift(-h).to_numpy(float)
        hi = df["high"].rolling(h).max().shift(-h).to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[h] = (ex / e - 1, np.maximum(0, (e - lo) / a), np.maximum(0, (hi - e) / a))
    return out


def _decluster(pos: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 1 or len(pos) < 2:
        return pos
    keep, last = [], -10 ** 9
    for p in pos:
        if p - last >= gap:
            keep.append(p)
            last = p
    return np.asarray(keep, dtype=int)


def _collect(df, ev: pd.DataFrame, dirs: dict, horizons, entry, min_gap, atr_n, acc: dict | None = None) -> dict:
    """Append per-event arrays to acc[(signal, h)] = dict(r, e, bh, mae). r = direction-adjusted return,
    e = r minus this instrument's unconditional mean (same horizon/direction), bh = base hit rate."""
    fw = _forward(df, horizons, entry, atr_n)
    acc = {} if acc is None else acc
    base = {}
    for h in horizons:
        ret = fw[h][0]
        v = ret[np.isfinite(ret)]
        base[h] = (v.mean() if len(v) else np.nan, (v > 0).mean() if len(v) else np.nan, (v < 0).mean() if len(v) else np.nan)
    for name in ev.columns:
        d = dirs[name]
        pos = _decluster(np.flatnonzero(ev[name].to_numpy()), min_gap)
        for h in horizons:
            ret, ml, ms = fw[h]
            r = ret[pos] * d
            ok = np.isfinite(r)
            bm, bhl, bhs = base[h]
            slot = acc.setdefault((name, h), {"r": [], "e": [], "bh": [], "mae": []})
            slot["r"].append(r[ok])
            slot["e"].append(r[ok] - bm * d)
            slot["bh"].append(np.full(ok.sum(), bhl if d > 0 else bhs))
            slot["mae"].append((ml if d > 0 else ms)[pos][ok])
    acc.setdefault("_base", []).append({h: base[h] for h in horizons})
    return acc


def _stats(slot: dict) -> dict:
    r, e, bh, mae = (np.concatenate(slot[k]) if slot[k] else np.array([]) for k in ("r", "e", "bh", "mae"))
    n = len(r)
    if n == 0:
        return dict(n=0, hit=np.nan, base_hit=np.nan, mean=np.nan, median=np.nan, base_mean=np.nan, edge=np.nan,
                    t=np.nan, p=np.nan, pf=np.nan, mae_atr=np.nan)
    sd = e.std(ddof=1) if n > 1 else np.nan
    if n > 1 and sd > 0:
        t = e.mean() / (sd / np.sqrt(n))
        p = 2 * _st.t.sf(abs(t), n - 1)
    else:
        t, p = np.nan, (0.0 if n > 1 and e.mean() != 0 else np.nan)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=n, hit=(r > 0).mean(), base_hit=bh.mean(), mean=r.mean(), median=np.median(r), base_mean=(r - e).mean(),
                edge=e.mean(), t=t, p=p, pf=gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan),
                mae_atr=np.nanmean(mae) if np.isfinite(mae).any() else np.nan)


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs kept as NaN, excluded from the family)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = q[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def _resolve_signal(df, signal, direction):
    if isinstance(signal, str):
        sig = SIGNALS[signal]
        ev = events(df, [signal])[signal]
        return signal, ev, direction or sig.direction
    ev = pd.Series(signal, copy=False).reindex(df.index).fillna(False).astype(bool)
    return getattr(signal, "name", None) or "signal", ev, direction or 1


def event_study(df: pd.DataFrame, signal, horizons=(1, 5, 10, 20), direction: int | None = None, entry: str = "close",
                min_gap: int = 0, atr_n: int = 14, return_events: bool = False):
    """Forward-return study of one signal (registry name or boolean Series) at each horizon.
    Columns: n, hit (share of direction-adjusted returns > 0), base_hit, mean, median, base_mean (all bars,
    same horizon and direction), edge = mean - base_mean, t/p (one-sample t on returns minus base mean),
    pf (profit factor), mae_atr (mean max adverse excursion before the horizon, in ATR at entry).
    min_gap: keep an event only if >= min_gap bars after the previously kept one (de-clustering).
    entry: 'close' (close of the event bar) or 'next_open'. With return_events, also a per-event table."""
    name, ev, d = _resolve_signal(df, signal, direction)
    horizons = tuple(horizons)
    acc = _collect(df, pd.DataFrame({name: ev.to_numpy()}, index=df.index), {name: d}, horizons, entry, min_gap, atr_n)
    out = pd.DataFrame({h: _stats(acc[(name, h)]) for h in horizons}).T
    out.index.name = "horizon"
    out.attrs.update(signal=name, direction=d, entry=entry, min_gap=min_gap)
    if not return_events:
        return out
    pos = _decluster(np.flatnonzero(ev.to_numpy()), min_gap)
    fw = _forward(df, horizons, entry, atr_n)
    tab = pd.DataFrame({f"ret_{h}": fw[h][0][pos] * d for h in horizons}, index=df.index[pos])
    return out, tab


# ---- scorecards ---------------------------------------------------------------------------------
def _table(acc: dict, names, horizons, min_count: int, rank_h: int, alpha: float) -> pd.DataFrame:
    rows = []
    for nm in names:
        for h in horizons:
            s = _stats(acc[(nm, h)])
            rows.append({"signal": nm, "h": h, **s})
    long = pd.DataFrame(rows)
    elig = long["n"] >= min_count
    long["p_adj"] = np.nan
    long.loc[elig, "p_adj"] = bh_adjust(long.loc[elig, "p"].to_numpy())
    wide = long[long["h"] == rank_h].set_index("signal")
    out = pd.DataFrame({"dir": [SIGNALS[n].direction if n in SIGNALS else 1 for n in wide.index],
                        "group": [SIGNALS[n].group if n in SIGNALS else "custom" for n in wide.index],
                        "n": wide["n"].astype(int)}, index=wide.index)
    for h in horizons:
        sub = long[long["h"] == h].set_index("signal")
        out[f"hit_{h}"] = sub["hit"]
        out[f"edge_{h}"] = sub["edge"]
    for k in ("base_hit", "mean", "base_mean", "t", "p", "p_adj", "pf", "mae_atr"):
        out[f"{k}_{rank_h}"] = wide[k]

    def verdict(r):
        if r["n"] < min_count:
            return f"too few (n<{min_count})"
        pa, p, e = r[f"p_adj_{rank_h}"], r[f"p_{rank_h}"], r[f"edge_{rank_h}"]
        if pa < alpha:
            return "edge (BH)" if e > 0 else "anti-edge (fade it)"
        if p < alpha:
            return "raw p<.05, fails BH"
        return "no edge"
    out["verdict"] = out.apply(verdict, axis=1) if len(out) else []
    out["_ok"] = out["n"] >= min_count
    out = out.sort_values(["_ok", f"edge_{rank_h}"], ascending=[False, False]).drop(columns="_ok")
    out.attrs["long"] = long
    return out


def _rank_h(horizons, rank_h):
    horizons = tuple(horizons)
    return rank_h if rank_h in horizons else horizons[len(horizons) // 2]


def scorecard(df: pd.DataFrame, signals=None, horizons=(5, 10, 20), min_count: int = 15, min_gap: int = 5,
              entry: str = "close", rank_h: int | None = None, alpha: float = 0.05, atr_n: int = 14,
              ev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank every signal by edge vs the instrument's base rate. hit_h/edge_h per horizon; t, p, BH-adjusted p,
    profit factor and MAE at the ranking horizon (middle horizon by default). BH runs across every
    (signal, horizon) test with n >= min_count, so luck across ~100 signals is not shown as edge."""
    horizons = tuple(horizons)
    rh = _rank_h(horizons, rank_h)
    ev = events(df, signals) if ev is None else (ev if signals is None else ev[list(signals)])
    acc = _collect(df, ev, {n: SIGNALS[n].direction for n in ev.columns}, horizons, entry, min_gap, atr_n)
    out = _table(acc, list(ev.columns), horizons, min_count, rh, alpha)
    out.attrs.update(base={h: dict(zip(("mean", "hit_long", "hit_short"), acc["_base"][0][h])) for h in horizons},
                     rank_h=rh, horizons=horizons, min_count=min_count, min_gap=min_gap, entry=entry, bars=len(df), n_signals=len(out),
                     errors=ev.attrs.get("errors", {}))
    return out


def multi_asset_scorecard(symbols, interval: str = "1d", period: str = "10y", start=None, signals=None,
                          horizons=(5, 10, 20), min_count: int = 30, min_gap: int = 5, entry: str = "close",
                          rank_h: int | None = None, alpha: float = 0.05, refresh: bool = False, frames: dict | None = None) -> pd.DataFrame:
    """Pool events across instruments (via data.get_prices, or pre-loaded `frames` {symbol: df}).
    Each event's excess return is taken vs its own instrument's base rate before pooling."""
    horizons = tuple(horizons)
    rh = _rank_h(horizons, rank_h)
    acc, used, errs, counts = {}, [], {}, {}
    names = list(SIGNALS) if signals is None else list(signals)
    for s in symbols:
        try:
            if frames is not None:
                df = frames[s]
            else:
                from .data import get_prices
                df = get_prices(s, interval, start, None, period, refresh=refresh)
            ev = events(df, names)
            _collect(df, ev, {n: SIGNALS[n].direction for n in names}, horizons, entry, min_gap, 14, acc)
            counts[s] = ev.sum()
            used.append(s)
        except Exception as e:  # noqa: BLE001
            errs[s] = f"{type(e).__name__}: {e}"[:120]
    if not used:
        raise LookupError(f"no data for any symbol: {errs}")
    out = _table(acc, names, horizons, min_count, rh, alpha)
    cnt = pd.DataFrame(counts)
    out.insert(3, "assets", (cnt > 0).sum(axis=1).reindex(out.index))
    out.attrs.update(symbols=used, errors=errs, n_signals=len(out), rank_h=rh, horizons=horizons, min_count=min_count, min_gap=min_gap, entry=entry)
    return out


def active_now(df: pd.DataFrame, lookback: int = 3, horizons=(5, 10, 20), min_count: int = 15, min_gap: int = 5,
               entry: str = "close", rank_h: int | None = None) -> pd.DataFrame:
    """Signals that fired in the last `lookback` bars, joined with their historical stats on this instrument
    (scorecard over all signals so the BH family is the full catalogue)."""
    ev = events(df)
    sc = scorecard(df, horizons=horizons, min_count=min_count, min_gap=min_gap, entry=entry, rank_h=rank_h, ev=ev)
    rh = sc.attrs["rank_h"]
    tail = ev.tail(lookback)
    fired = [c for c in tail.columns if tail[c].any()]
    cols = ["n"] + [f"hit_{h}" for h in horizons] + [f"edge_{h}" for h in horizons] + [f"base_hit_{rh}", f"p_adj_{rh}", f"pf_{rh}", f"mae_atr_{rh}", "verdict"]
    rows = []
    for c in fired:
        last = int(np.flatnonzero(tail[c].to_numpy())[-1])
        rows.append({"signal": c, "dir": "long" if SIGNALS[c].direction > 0 else "short", "fired": tail.index[last],
                     "bars_ago": len(tail) - 1 - last, **sc.loc[c, cols].to_dict(), "rule": SIGNALS[c].desc})
    out = pd.DataFrame(rows, columns=["signal", "dir", "fired", "bars_ago"] + cols + ["rule"])
    if len(out):
        out = out.sort_values(["bars_ago", f"edge_{rh}"], ascending=[True, False]).set_index("signal")
    out.attrs.update(sc.attrs)
    return out


# ---- CLI -----------------------------------------------------------------------------------------
def pretty(sc: pd.DataFrame) -> pd.DataFrame:
    """Display copy: hit rates and returns as percentages, p-values rounded."""
    t = sc.copy()
    for k in t.columns:
        if k.startswith(("hit_", "base_hit_")):
            t[k] = [f"{v:.0%}" if pd.notna(v) else "n/a" for v in t[k]]
        elif k.startswith(("edge_", "mean_", "base_mean_")):
            t[k] = [f"{v:+.2%}" if pd.notna(v) else "n/a" for v in t[k]]
        elif k.startswith(("p_", "p_adj_")):
            t[k] = [f"{v:.3f}" if pd.notna(v) else "n/a" for v in t[k]]
        elif k.startswith(("t_", "pf_", "mae_atr_")):
            t[k] = [f"{v:.2f}" if pd.notna(v) else "n/a" for v in t[k]]
        elif k == "fired":
            t[k] = [x.strftime("%Y-%m-%d %H:%M").replace(" 00:00", "") for x in t[k]]
    if "dir" in t and pd.api.types.is_numeric_dtype(t["dir"]):
        t["dir"] = t["dir"].map({1: "long", -1: "short"})
    return t


def _notes(sc):
    rh, b = sc.attrs["rank_h"], sc.attrs.get("base", {})
    if b:
        print("\nBase rate (all bars, long): " + " · ".join(f"{h}b mean {v['mean']:+.2%}, up {v['hit_long']:.0%}" for h, v in b.items()))
    print(f"edge = mean direction-adjusted return minus that base; t/p vs base; p_adj = Benjamini-Hochberg across all "
          f"{sc.attrs.get('n_signals', len(sc))} signals x {len(sc.attrs['horizons'])} horizons with n >= {sc.attrs['min_count']}; ranked at {rh} bars; "
          f"events de-clustered to >= {sc.attrs['min_gap']} bars apart; entry at {sc.attrs['entry']}; overlapping horizons make t-stats optimistic.")


def cmd_signals(a):
    from .cli import show, src_line
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period, refresh=a.refresh)
    sc = scorecard(df, horizons=tuple(a.horizons), min_count=a.min_count, min_gap=a.min_gap, entry=a.entry)
    if a.json:
        show(sc, True)
        return
    rh = sc.attrs["rank_h"]
    ok = sc[sc["n"] >= a.min_count]
    cols = ["dir", "group", "n"] + [f"{k}_{h}" for h in sc.attrs["horizons"] for k in ("hit", "edge")] + \
        [f"t_{rh}", f"p_adj_{rh}", f"pf_{rh}", f"mae_atr_{rh}", "verdict"]
    show(pretty(ok if a.all else ok.head(a.top))[cols], title=f"{a.symbol} {a.interval} signal scorecard ({len(df)} bars, ranked by edge at {rh} bars)")
    if not a.all and len(ok) > a.top:
        show(pretty(ok.tail(min(a.top, len(ok) - a.top)))[cols], title="Worst signals (fade candidates if anti-edge)")
    print(f"\n{len(ok)} of {len(sc)} signals have n >= {a.min_count}; BH-significant: "
          + (", ".join(ok.index[ok['verdict'].str.startswith(('edge', 'anti'))]) or "none"))
    _notes(sc)
    src_line(df)


def cmd_firing(a):
    from .cli import show, src_line
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period, refresh=a.refresh)
    t = active_now(df, a.lookback, tuple(a.horizons), a.min_count, a.min_gap, a.entry)
    if a.json:
        show(t, True)
        return
    show(pretty(t) if len(t) else t, title=f"{a.symbol} {a.interval}: signals fired in the last {a.lookback} bars, with their history on this instrument")
    _notes(t)
    src_line(df)


def register(add):
    def common(q):
        q.add_argument("symbol")
        q.add_argument("--interval", "-i", default="1d")
        q.add_argument("--period", "-p", default="10y", help="history for the stats (1y 2y 5y 10y max)")
        q.add_argument("--start")
        q.add_argument("--refresh", action="store_true", help="bypass cache")
        q.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
        q.add_argument("--min-count", type=int, default=15)
        q.add_argument("--min-gap", type=int, default=5, help="min bars between counted events (de-clustering)")
        q.add_argument("--entry", choices=["close", "next_open"], default="close")
    q = add("signals", cmd_signals, "backtested scorecard of every TA signal vs base rate (BH-adjusted p-values)")
    common(q)
    q.add_argument("--top", type=int, default=25)
    q.add_argument("--all", action="store_true", help="show every signal with n >= min-count")
    q = add("firing", cmd_firing, "signals firing now + their historical hit rate and edge on this instrument")
    common(q)
    q.add_argument("--lookback", type=int, default=3)
