"""Offline tests for mlab.signal_lab: event-study arithmetic, base-rate subtraction, BH, causality."""
import time

import numpy as np
import pandas as pd
import pytest
from conftest import from_path, synth_ohlcv

from mlab import signal_lab as sl
from mlab import ta_catalog as tc


@pytest.fixture(scope="module")
def df():
    return synth_ohlcv()


def test_registry(df):
    assert len(sl.SIGNALS) >= 80
    assert all(s.direction in (1, -1) for s in sl.SIGNALS.values())
    names = set(sl.SIGNALS)
    for k, (d, _, _) in tc.CANDLES.items():
        assert (f"candle_{k}" in names) if d else {f"candle_{k}_after_up", f"candle_{k}_after_down"} <= names
    for k in ("macd_bull_cross", "golden_cross", "rsi_bull_divergence", "bb_squeeze_breakout_up", "donchian55_breakout",
              "supertrend_flip_up", "psar_flip_down", "ichimoku_tk_bull", "kumo_breakout_up", "nr7_breakout_up",
              "high_52w_breakout", "gap_down", "chart_double_bottom", "chart_symmetrical_triangle_up", "obv_confirmed_breakout"):
        assert k in names
    ev = sl.events(df)
    assert not ev.attrs["errors"] and (ev.dtypes == bool).all()
    assert (ev.sum() > 0).mean() > 0.85  # nearly every signal fires on 10y of data


def test_guaranteed_future_up_move_hits_100pct(df):
    sig = df["close"].shift(-5) > df["close"]  # constructed with hindsight on purpose
    es = sl.event_study(df, sig, horizons=(5,))
    assert es.loc[5, "hit"] == 1.0 and es.loc[5, "base_hit"] < 0.6
    assert es.loc[5, "edge"] > 0 and es.loc[5, "p"] < 1e-6 and np.isinf(es.loc[5, "pf"])
    short = sl.event_study(df, sig, horizons=(5,), direction=-1)
    assert short.loc[5, "hit"] == 0.0


def test_base_rate_subtraction():
    c = 100 * 1.001 ** np.arange(400)
    d = from_path(c, spread=0.01)
    sig = pd.Series(np.arange(400) % 7 == 0, index=d.index)
    for direction in (1, -1):
        es = sl.event_study(d, sig, horizons=(1, 10), direction=direction)
        assert np.allclose(es["mean"], es["base_mean"]) and np.allclose(es["edge"], 0, atol=1e-12)
        assert np.sign(es.loc[10, "mean"]) == direction
        assert es.loc[10, "mean"] == pytest.approx(direction * (1.001 ** 10 - 1))
    es = sl.event_study(d, sig, horizons=(10,))
    assert es.loc[10, "n"] == sum(1 for i in range(0, 390) if i % 7 == 0)


def test_planted_edge_is_found():
    rng = np.random.default_rng(5)
    n = 2000
    r = rng.normal(0, 0.01, n)
    ev = np.zeros(n, bool)
    ev[np.arange(50, n - 30, 40)] = True
    for i in np.flatnonzero(ev):
        r[i + 1:i + 6] += 0.004  # +2% drift over the 5 bars after each event
    d = from_path(100 * np.exp(np.cumsum(r)), spread=0.2)
    es = sl.event_study(d, pd.Series(ev, index=d.index), horizons=(5, 20))
    assert es.loc[5, "edge"] == pytest.approx(0.02, abs=0.006) and es.loc[5, "p"] < 1e-4 and es.loc[5, "hit"] > 0.75


def test_entry_next_open_and_mae():
    c = np.r_[np.full(30, 100.0), 100, 99, 98, 103, 104, 105, np.full(10, 105.0)]
    d = from_path(c, spread=0.0001)
    d["open"] = d["close"].shift().fillna(100) + 0.5
    sig = pd.Series(False, index=d.index)
    sig.iloc[30] = True
    _, tab = sl.event_study(d, sig, horizons=(3,), return_events=True)
    assert tab["ret_3"].iloc[0] == pytest.approx(103 / 100 - 1)
    _, tab = sl.event_study(d, sig, horizons=(3,), entry="next_open", return_events=True)
    assert tab["ret_3"].iloc[0] == pytest.approx(103 / d["open"].iloc[31] - 1)
    es = sl.event_study(d, sig, horizons=(3,))
    assert es.loc[3, "mae_atr"] > 0  # dipped to ~98 before the horizon


def test_declustering(df):
    sig = pd.Series(False, index=df.index)
    sig.iloc[100:110] = True  # ten consecutive bars
    assert sl.event_study(df, sig, horizons=(5,)).loc[5, "n"] == 10
    assert sl.event_study(df, sig, horizons=(5,), min_gap=5).loc[5, "n"] == 2
    assert sl.event_study(df, sig, horizons=(5,), min_gap=20).loc[5, "n"] == 1


def test_bh_adjust_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.random.default_rng(0).uniform(0, 0.2, 50)
    assert np.allclose(sl.bh_adjust(p), multipletests(p, method="fdr_bh")[1])
    assert np.isnan(sl.bh_adjust([np.nan, 0.01])[0])


def test_no_look_ahead(df):
    """Shuffling every bar after t must not change any signal on bars <= t."""
    base = sl.events(df)
    for t in (900, 1700, 2300):
        d2 = df.copy()
        d2.iloc[t + 1:] = df.iloc[t + 1:].sample(frac=1, random_state=t).to_numpy()
        ev2 = sl.events(d2)
        diff = (base.iloc[:t + 1] != ev2.iloc[:t + 1]).any()
        assert not diff.any(), f"look-ahead at t={t}: {list(diff[diff].index)}"


def test_scorecard(df):
    t0 = time.time()
    sc = sl.scorecard(df)
    assert time.time() - t0 < 5
    assert len(sc) == len(sl.SIGNALS) and {"hit_5", "edge_20", "p_adj_10", "verdict"} <= set(sc.columns)
    ok = sc[sc["n"] >= 15]
    assert ok["edge_10"].is_monotonic_decreasing and (ok["p_adj_10"] >= ok["p_10"] - 1e-12).all()
    assert sc[sc["n"] < 15]["verdict"].str.startswith("too few").all()
    assert not sc["verdict"].eq("edge (BH)").any()  # a random walk has no real edge
    assert set(sc.attrs["base"]) == {5, 10, 20}
    assert "%" in sl.pretty(sc)["hit_5"].iloc[0]


def test_multi_asset_and_active_now(df):
    other = synth_ohlcv(1500, seed=3)
    m = sl.multi_asset_scorecard(["A", "B"], signals=["macd_bull_cross", "rsi_os_exit"], frames={"A": df, "B": other})
    single = [sl.scorecard(x, signals=["macd_bull_cross"], min_count=1).loc["macd_bull_cross", "n"] for x in (df, other)]
    assert m.loc["macd_bull_cross", "n"] == sum(single) and m.loc["macd_bull_cross", "assets"] == 2
    an = sl.active_now(df, lookback=5)
    ev = sl.events(df).tail(5)
    assert set(an.index) == set(ev.columns[ev.any()]) and (an["bars_ago"] < 5).all()
    assert {"n", "hit_10", "edge_10", "verdict", "rule"} <= set(an.columns)


def test_cli_registers():
    from mlab.cli import build_parser
    p = build_parser()
    a = p.parse_args(["signals", "X", "--horizons", "5", "20", "--min-count", "10"])
    assert a.horizons == [5, 20] and a.min_count == 10 and a.period == "10y"
    assert p.parse_args(["firing", "X"]).lookback == 3
    assert p.parse_args(["tafull", "X"]).period == "2y" and p.parse_args(["catalog", "--group", "volume"]).group == "volume"
