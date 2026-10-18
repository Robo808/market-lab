"""Gates for the phase-space research layer. The look-ahead and schema tests are the deterministic gates:
a feature, regime filter or walk-forward fold that uses future data fails the build."""
import numpy as np
import pandas as pd
import pytest
from conftest import synth_ohlcv

from mlab.phase import evaluate as E
from mlab.phase import features as F
from mlab.phase import physics as P
from mlab.phase.regimes import GaussianHMM, RegimeModel


@pytest.fixture(scope="module")
def frames():
    return {f"S{i}": synth_ohlcv(1600, seed=10 + i, start="2004-01-02") for i in range(3)}


def _hourly(n=3000, seed=1):
    df = synth_ohlcv(n, seed=seed)
    df.index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return df


# ---- look-ahead gate --------------------------------------------------------------------------------------
@pytest.mark.parametrize("make", [lambda: synth_ohlcv(1200, seed=3), lambda: _hourly(2000, seed=4)],
                         ids=["daily+weekly", "hourly+daily"])
def test_every_feature_is_point_in_time(make):
    df = make()
    cuts = [300, 517, 801, 1111, len(df) - 1]
    assert F.check_point_in_time(F.features, df, cuts) == {}


def test_lookahead_gate_catches_planted_leaks():
    df = synth_ohlcv(900, seed=5)

    def leaky(d):
        f = F.features(d)
        f["peek_next_close"] = d["close"].shift(-1) / d["close"]
        f["centred_mean"] = d["close"].rolling(21, center=True).mean() / d["close"]
        f["full_sample_z"] = (d["close"] - d["close"].mean()) / d["close"].std()
        return f
    leaks = F.check_point_in_time(leaky, df)
    assert set(leaks) == {"peek_next_close", "centred_mean", "full_sample_z"}


def test_partial_higher_timeframe_bar_never_leaks():
    df = synth_ohlcv(400, seed=6)
    h = F.higher_timeframe(df)
    # a Wednesday must not see the week it sits in: its weekly value equals the previous Friday's
    wed = df.index[df.index.dayofweek == 2][10]
    fri_before = df.index[(df.index < wed) & (df.index.dayofweek == 4)][-1]
    pd.testing.assert_series_equal(h.loc[wed], h.loc[fri_before], check_names=False)


# ---- schema contract --------------------------------------------------------------------------------------
def test_cube_meets_schema_contract(frames):
    cube = F.build(frames)
    assert F.validate(cube) == []
    arr, syms, times, _ = F.to_array(cube)
    assert arr.shape == (3, len(times), len(F.FEATURES)) and syms == ["S0", "S1", "S2"]


def test_schema_contract_catches_violations(frames):
    cube = F.build(frames).copy()
    cube.iloc[5, cube.columns.get_loc("rsi_14")] = 3.0
    cube.iloc[6, cube.columns.get_loc("ret_1")] = np.inf
    bad = F.validate(cube.drop(columns="pe_63"))
    assert any("pe_63" in b for b in bad)
    bad = F.validate(cube)
    assert any("rsi_14" in b for b in bad) and any("ret_1: infinite" in b for b in bad)


def test_permutation_entropy_bounds():
    up = pd.Series(np.arange(300.0))
    assert F.permutation_entropy(up, 63).dropna().max() == pytest.approx(0.0)
    noise = pd.Series(np.random.default_rng(0).normal(size=3000))
    assert F.permutation_entropy(noise, 500).dropna().mean() > 0.97


# ---- regimes ------------------------------------------------------------------------------------------------
def _two_state(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < 0.97 else 1 - s[t - 1]
    mu = np.array([[0.0, 0.0], [3.0, -2.0]])
    return mu[s] + rng.normal(size=(n, 2)), s


def test_hmm_recovers_planted_regimes():
    x, s = _two_state()
    hmm = GaussianHMM(k=2, seed=0).fit([x[:800], x[800:]])
    lab = hmm.filter(x).argmax(1)
    acc = max((lab == s).mean(), (lab != s).mean())         # labels are arbitrary up to permutation
    assert acc > 0.95
    assert np.diag(hmm.A).min() > 0.9


def test_hmm_filter_is_point_in_time():
    x, _ = _two_state(600, seed=1)
    hmm = GaussianHMM(k=2, seed=0).fit([x])
    full = hmm.filter(x)
    for k in (100, 350, 599):
        np.testing.assert_allclose(hmm.filter(x[:k]), full[:k], rtol=1e-10, atol=1e-12)


def test_regime_model_probabilities_are_point_in_time(frames):
    cube = F.build(frames)
    rm = RegimeModel(k=2, seed=0).fit(cube)
    full = rm.probs(cube)
    cut = cube.index.get_level_values("time")[len(cube) // 2]
    part = rm.probs(cube[cube.index.get_level_values("time") <= cut])
    pd.testing.assert_frame_equal(part, full.loc[part.index])
    assert len(rm.names) == 2 and rm.profile["share"].sum() == pytest.approx(1.0)


# ---- evaluator ----------------------------------------------------------------------------------------------
def test_train_mask_purges_label_window():
    t = pd.date_range("2010-01-01", periods=100, freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([["A"], t], names=["symbol", "time"])
    m = E.train_mask(idx, t[60], gap=10)
    assert m.sum() == 50 and not m[50:].any()


def test_walk_forward_fold_ignores_later_data(frames):
    """Predictions for a test year must not change when every bar after that year is deleted."""
    S = E.prepare(frames, "__test_wf__", min_train_years=3)
    year = S.years[0]
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    S2 = E.prepare({k: v[v.index < end] for k, v in frames.items()}, "__test_wf__", min_train_years=3)
    for cfg in ({"model": "ridge", "horizon": 5}, {"model": "regime", "horizon": 5, "k": 2}):
        y1, y2 = E.forward_target(S, 5), E.forward_target(S2, 5)
        p1 = E.predict_fold(S, E.FoldCache(S, 5, log=lambda *_: None), cfg, year, y1)
        p2 = E.predict_fold(S2, E.FoldCache(S2, 5, log=lambda *_: None), cfg, year, y2)
        # the last h bars of the year have no label in the truncated set, but their predictions use features only
        pd.testing.assert_series_equal(p1.loc[p2.index], p2, check_names=False)


def test_simulate_charges_costs_and_respects_delay(frames):
    S = E.prepare(frames, "__test_sim__", min_train_years=3)
    w = pd.DataFrame(0.0, index=S.closes.index, columns=S.closes.columns)
    w.iloc[100:, 0] = 1.0
    r0 = E.simulate(S, w, delay=0, funding_annual=0.0)
    r1 = E.simulate(S, w, delay=1, funding_annual=0.0)
    ret = S.closes.iloc[:, 0].pct_change()
    assert r0.iloc[101] == pytest.approx(ret.iloc[101] - S.costs.iloc[0] / 1e4)
    assert r0.iloc[100] == 0.0 and r1.iloc[101] == 0.0
    assert r1.iloc[102] == pytest.approx(ret.iloc[102] - S.costs.iloc[0] / 1e4)


def test_nw_tstat_matches_plain_t_on_iid():
    x = pd.Series(np.random.default_rng(1).normal(0.05, 1, 5000))
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert E.nw_tstat(x, lags=5) == pytest.approx(plain, rel=0.1)


def test_pbo_flags_noise_and_clears_a_real_edge():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
    assert 0.25 < E.pbo_cscv(noise)["pbo"] < 0.8
    real = noise.copy()
    real[0] += 0.003
    assert E.pbo_cscv(real)["pbo"] < 0.1


def test_experiment_log_counts_each_config_once():
    fam = "__test_log__"
    E.log_experiment(fam, {"model": "x", "h": 1}, {"sharpe": 0.1}, {})
    E.log_experiment(fam, {"model": "x", "h": 1}, {"sharpe": 0.2}, {})
    E.log_experiment(fam, {"model": "x", "h": 2}, {"sharpe": -0.1}, {})
    assert E.trial_count(fam) == 2
    log = E.experiments(fam)
    assert len(log) == 2 and set(log["h"]) == {1, 2}


def test_random_walk_does_not_pass(frames):
    S = E.prepare(frames, "__test_rw__", min_train_years=3)
    res = E.evaluate(S, [{"model": "ridge", "horizon": 5}, {"model": "tsmom", "horizon": 5}], log=lambda *_: None)
    assert not res["verdict"].startswith("PASS")
    assert {"always_long", "coin_flip seed=0"} <= set(res["stubs"]["stub"])
    assert res["trials"] >= 2


# ---- physics ------------------------------------------------------------------------------------------------
def test_fisher_g_finds_a_planted_cycle_only():
    rng = np.random.default_rng(3)
    t = np.arange(2048)
    assert P.fisher_g(np.sin(2 * np.pi * t / 32) + rng.normal(0, 0.5, len(t)))[2] < 1e-6
    _, _, p = P.fisher_g(rng.normal(size=len(t)))
    assert p > 0.01


def test_forces_detect_momentum_and_reversion():
    rng = np.random.default_rng(4)
    out = {}
    for name, phi in (("mom", 0.3), ("rev", -0.3)):
        e = rng.normal(0, 0.01, 4000)
        r = np.zeros_like(e)
        for i in range(1, len(e)):
            r[i] = phi * r[i - 1] + e[i]
        c = 100 * np.exp(np.cumsum(r))
        idx = pd.date_range("2000-01-03", periods=len(c), freq="B", tz="UTC")
        df = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)
        cube = F.build({name: df})
        reg = pd.Series(0, index=cube.index)
        out[name] = P.forces_by_regime({name: df}, cube, reg, ["all"], qs=(1,))
    assert out["mom"].loc[0, "corr_q1"] > 0.2 and out["rev"].loc[0, "corr_q1"] < -0.2


def test_synthetic_frames_keep_bar_shape(frames):
    syn = P.synthetic_frames(frames, block=21, seed=0)
    for s, d in syn.items():
        assert (d["high"] >= d[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (d["low"] <= d[["open", "close"]].min(axis=1) + 1e-9).all()
        r0, r1 = np.log(frames[s]["close"]).diff().std(), np.log(d["close"]).diff().std()
        assert r1 == pytest.approx(r0, rel=0.15)


def test_stress_ranges_are_ordered():
    r = pd.Series(np.random.default_rng(5).normal(0.0004, 0.01, 1500))
    st = P.stress(r, n_paths=200)
    assert st["sharpe_5_50_95"][0] < st["sharpe_5_50_95"][1] < st["sharpe_5_50_95"][2]
    assert st["max_dd_5_50_95"][2] <= 0
