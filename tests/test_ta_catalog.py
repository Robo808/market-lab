"""Offline tests for mlab.ta_catalog on synthetic OHLCV (closed-form identities, planted patterns, causality)."""
import numpy as np
import pandas as pd
import pytest
from conftest import from_path, synth_ohlcv

from mlab import ta
from mlab import ta_catalog as tc


@pytest.fixture(scope="module")
def df():
    return synth_ohlcv()


def _bars(rows):
    idx = pd.date_range("2021-01-04", periods=len(rows), freq="B", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1e6)


def _downtrend(n=15, start=110.0):
    rows, c = [], start
    for _ in range(n):
        o, c = c + 0.2, c - 1.0
        rows.append((o, o + 0.1, c - 0.1, c))
    return rows, c


# ---- closed-form identities ---------------------------------------------------------
def test_williams_r_is_stoch_minus_100(df):
    w = tc.williams_r(df)
    assert np.allclose((w - (ta.stoch(df)["k"] - 100)).dropna(), 0)
    hh, ll = df["high"].rolling(14).max(), df["low"].rolling(14).min()
    assert np.allclose((w - (-100 * (hh - df["close"]) / (hh - ll))).dropna(), 0)
    assert w.dropna().between(-100, 0).all()


def test_heikin_ashi(df):
    ha = tc.heikin_ashi(df)
    assert np.allclose(ha["close"], df[["open", "high", "low", "close"]].mean(axis=1))
    assert ha["open"].iloc[0] == pytest.approx((df["open"].iloc[0] + df["close"].iloc[0]) / 2)
    assert np.allclose(ha["open"].iloc[1:].to_numpy(), ((ha["open"] + ha["close"]) / 2).iloc[:-1].to_numpy())
    assert (ha["high"] >= ha[["open", "close"]].max(axis=1) - 1e-12).all()


def test_moving_averages_on_constant_and_linear():
    s = pd.Series(np.full(200, 50.0))
    for f in (tc.wma, tc.hma, tc.dema, tc.tema, tc.zlema, tc.kama):
        assert np.allclose(f(s).dropna(), 50.0), f.__name__
    line = pd.Series(np.arange(300, dtype=float) * 0.5 + 10)
    lr = tc.linreg(line, 50).dropna()
    assert np.allclose(lr["slope"], 0.5) and np.allclose(lr["r2"], 1.0) and np.allclose(lr["value"], line[lr.index])
    w = tc.wma(line, 3).iloc[-1]
    assert w == pytest.approx((line.iloc[-3] + 2 * line.iloc[-2] + 3 * line.iloc[-1]) / 6)


def test_oscillator_ranges(df):
    rng = lambda s, lo, hi: s.dropna().between(lo - 1e-9, hi + 1e-9).all()  # noqa: E731
    a = tc.aroon(df)
    assert rng(a["up"], 0, 100) and rng(a["down"], 0, 100) and rng(a["osc"], -100, 100)
    assert rng(tc.cmf(df), -1, 1) and rng(tc.ultimate(df), 0, 100) and rng(tc.cmo(df["close"]), -100, 100)
    assert rng(tc.stoch_rsi(df["close"])["k"], 0, 100) and rng(tc.schaff(df["close"]), 0, 100)
    assert rng(tc.choppiness(df), 0, 100) and rng(tc.bop(df), -1, 1)
    v = tc.vortex(df).dropna()
    assert (v > 0).all().all()


def test_hist_vol_recovers_sigma():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    d = from_path(c, spread=0.01)
    hv = tc.hist_vol(d, n=2000)["close"].iloc[-1]
    assert hv == pytest.approx(0.01 * np.sqrt(252), rel=0.08)
    assert (tc.hist_vol(synth_ohlcv(600)).dropna() > 0).all().all()


def test_aroon_and_cci_closed_form():
    up = pd.Series(np.arange(60, dtype=float))
    d = pd.DataFrame({"open": up, "high": up + 1, "low": up - 1, "close": up, "volume": 1.0})
    a = tc.aroon(d, 25).dropna()
    assert (a["up"] == 100).all() and (a["down"] == 0).all()
    tp = (d["high"] + d["low"] + d["close"]) / 3
    md = tp.rolling(20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    assert np.allclose(tc.cci(d).dropna(), ((tp - tp.rolling(20).mean()) / (0.015 * md)).dropna())


def test_pivots_profile_renko():
    d = _bars([(10, 12, 8, 11)])
    cam = tc.pivot_levels(d, "camarilla")
    assert cam["R4"] == pytest.approx(11 + 4 * 1.1 / 2) and cam["S1"] == pytest.approx(11 - 4 * 1.1 / 12)
    assert tc.pivot_levels(d, "woodie")["P"] == pytest.approx((12 + 8 + 22) / 4)
    assert tc.pivot_levels(d, "fibonacci")["R2"] == pytest.approx(31 / 3 + 0.618 * 4)
    assert tc.pivot_levels(d, "classic") == ta.pivots(d)
    rows = [(100, 101, 99, 100)] * 30 + [(110, 111, 109, 110)] * 300
    mp = tc.market_profile(_bars(rows), bins=60)
    assert 109 <= mp["poc"] <= 111 and mp["val"] >= 108.9 and mp["vah"] <= 111 + 1e-9  # all volume of the 110 bars
    c = np.linspace(100, 120.5, 50)
    r = tc.renko(from_path(c), box=1.0)
    assert (r["dir"] == 1).all() and len(r) == 20


def test_zigzag_causal_and_labels(df):
    zz = tc.zigzag(df)
    assert (zz["confirm_i"] >= zz["i"]).all() and len(zz) > 20
    assert set(zz["kind"].iloc[::2]) | set(zz["kind"].iloc[1::2]) == {"high", "low"}
    assert (zz["kind"] != zz["kind"].shift()).iloc[1:].all()  # alternates
    cut = 1500
    zt = tc.zigzag(df.iloc[:cut])
    known = zz[zz["confirm_i"] < cut]
    assert np.array_equal(zt["i"].to_numpy()[:len(known)], known["i"].to_numpy())
    # stair-step uptrend -> Dow uptrend
    path = np.concatenate([np.linspace(a, b, 12) for a, b in [(100, 110), (110, 105), (105, 116), (116, 111), (111, 123), (123, 118), (118, 130)]])
    assert tc.dow_trend(from_path(path, spread=0.2)).iloc[-1] == 1


# ---- candles ---------------------------------------------------------------------------
def test_planted_bullish_engulfing():
    rows, c = _downtrend()
    rows += [(c + 0.5, c + 0.6, c - 0.6, c - 0.5), (c - 0.7, c + 1.4, c - 0.8, c + 1.2)]
    cd = tc.candles(_bars(rows))
    assert cd["bullish_engulfing"].iloc[-1] and not cd["bearish_engulfing"].any()
    assert cd["bullish_engulfing"].sum() == 1


def test_planted_hammer_and_morning_star():
    rows, c = _downtrend()
    cd = tc.candles(_bars(rows + [(c - 0.5, c + 0.05, c - 3.0, c)]))
    assert cd["hammer"].iloc[-1] and not cd["hanging_man"].iloc[-1]
    star = [(c, c + 0.1, c - 3.1, c - 3.0), (c - 3.3, c - 3.1, c - 3.6, c - 3.25), (c - 3.2, c - 0.9, c - 3.3, c - 1.0)]
    cd = tc.candles(_bars(rows + star))
    assert cd["morning_star"].iloc[-1]


def test_candles_frame(df):
    cd = tc.candles(df)
    assert list(cd.columns) == list(tc.CANDLES) and (cd.dtypes == bool).all()
    assert not (cd["hammer"] & cd["hanging_man"]).any()
    assert not (cd["bullish_engulfing"] & cd["bearish_engulfing"]).any()


# ---- chart patterns ----------------------------------------------------------------------
def _segments(points, n=15, seed=1):
    rng = np.random.default_rng(seed)
    path = np.concatenate([np.linspace(a, b, n, endpoint=False) for a, b in zip(points[:-1], points[1:])] + [[points[-1]] * 10])
    return path + rng.normal(0, 0.05, len(path))


def test_planted_double_bottom():
    d = from_path(_segments([90, 100, 80, 90, 80.5, 100]), spread=0.3)
    p = tc.chart_patterns(d)
    db = p[(p["pattern"] == "double_bottom") & p["confirmed"]]
    assert len(db) == 1
    r = db.iloc[0]
    assert r["breakout_level"] == pytest.approx(90, abs=1.0) and r["target"] == pytest.approx(99.7, abs=1.5)
    assert r["confirm_time"] > r["end"] and d.loc[r["confirm_time"], "close"] > r["breakout_level"]
    between = d["close"].iloc[int(r["det_i"]):int(r["confirm_i"])]
    assert (between <= r["breakout_level"] + 1e-9).all()  # confirm_time is the first close beyond the neckline


def test_planted_head_and_shoulders():
    d = from_path(_segments([80, 100, 90, 110, 90.5, 100.5, 80]), spread=0.3)
    p = tc.chart_patterns(d)
    hs = p[(p["pattern"] == "head_shoulders") & p["confirmed"]]
    assert len(hs) == 1 and hs.iloc[0]["direction"] == -1
    assert hs.iloc[0]["breakout_level"] == pytest.approx(90.3, abs=1.2)
    assert not ((p["pattern"] == "double_top") & p["confirmed"]).any()


def test_chart_patterns_schema(df):
    p = tc.chart_patterns(df)
    assert list(p.columns) == tc._PCOLS and len(p) > 10
    assert set(p["pattern"]) <= set(tc.CHART_PATTERNS)
    conf = p[p["confirmed"]]
    assert (conf["confirm_i"].astype(int) >= conf["det_i"]).all() and (conf["det_i"] >= conf["end_i"]).all()
    assert set(p["status"]) <= {"confirmed", "failed", "pending", "expired", "open", "filled"}
    bp = tc.bar_patterns(df)
    assert (bp.dtypes == bool).all() and not (bp["inside_bar"] & bp["outside_bar"]).any()
    assert (bp["nr7"] <= bp["nr4"]).all()


# ---- catalogue / readings -----------------------------------------------------------------
def test_catalog_complete(df):
    kinds = pd.Series({k: v["kind"] for k, v in tc.CATALOG.items()}).value_counts()
    assert kinds["indicator"] >= 60 and kinds["candle"] == len(tc.CANDLES) and kinds["chart"] >= 25
    for name, e in tc.CATALOG.items():
        assert set(e) == {"kind", "fn", "group", "summary", "investopedia"} and e["group"] in tc.GROUPS, name
    for name in ("rsi", "macd", "adx_dmi", "ichimoku", "parabolic_sar", "williams_r", "market_profile", "dow_theory", "double_top", "hammer"):
        assert name in tc.CATALOG
    small = df.tail(400)
    for e in tc.CATALOG.values():
        e["fn"](small)  # every entry runs


def test_compute_catalog_and_readings(df):
    x = tc.compute_catalog(df)
    assert len(x) == len(df) and x.columns.is_unique and "chikou" not in " ".join(x.columns)
    rd = tc.latest_readings(df)
    assert len(rd) >= 70 and all(set(v) == {"group", "value", "state"} for v in rd.values())
    assert rd["rsi"]["state"] in {"overbought", "oversold", "neutral"}
    short = tc.latest_readings(df.head(30))
    assert short["sma200"]["state"].startswith("n/a") and short["rsi"]["state"] != "n/a"


def test_supertrend_valid(df):
    st = tc.supertrend(df).dropna()
    assert len(st) > len(df) - 20 and set(st["trend"].unique()) == {1.0, -1.0}
    up = st["trend"] > 0
    assert (df["close"][up.index[up]] >= st["supertrend"][up] - 1e-9).mean() > 0.95
