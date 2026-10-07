"""Offline tests for the quant layer: no look-ahead in any strategy, engine maths, validation
statistics, paper books and pre-registered hypotheses."""
import numpy as np
import pandas as pd
import pytest

from conftest import synth_ohlcv
from mlab.quant import engine, hypothesis, paper, strategies, validation


@pytest.fixture(scope="module")
def frames():
    return {f"S{i}": synth_ohlcv(1500, seed=i) for i in range(4)}


@pytest.mark.parametrize("name", [n for n, s in strategies.REGISTRY.items() if s.kind == "single"])
def test_single_strategies_no_lookahead(name):
    df = synth_ohlcv(900, seed=3)
    s = strategies.get(name)
    full = s.fn(df)
    assert full.index.equals(df.index) and full.abs().max() <= 1 + 1e-9
    for k in (400, 650, 899):
        part = s.fn(df.iloc[:k])
        pd.testing.assert_series_equal(part, full.iloc[:k], check_names=False, obj=f"{name} truncated at {k}")


@pytest.mark.parametrize("name", ["xs_momentum", "dual_momentum", "inverse_vol", "risk_parity"])
def test_portfolio_strategies_no_lookahead(name, frames):
    panel = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    fn = strategies.get(name).fn
    kw = {"top": 2} if name == "xs_momentum" else {}
    full = fn(panel, **kw)
    k = 1100
    part = fn(panel.iloc[:k], **kw)
    # month-end rows are decided at that close, so compare only rows strictly before the last month-end held
    pd.testing.assert_frame_equal(part.iloc[:k - 25], full.iloc[:k - 25])
    if name != "dual_momentum":
        assert (full.sum(axis=1).round(9) <= 1.0 + 1e-9).all()


def test_pairs_market_neutral(frames):
    a = frames["S0"]["close"]
    noise = np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.004, len(a))) * 0.3)
    panel = pd.DataFrame({"A": a, "B": a ** 0.8 * noise})  # positively hedged pair
    w = strategies.pairs(panel)
    active = w.abs().sum(axis=1) > 0
    assert active.any()
    assert np.allclose(w[active].abs().sum(axis=1), 1.0)
    assert (np.sign(w.loc[active, "A"]) != np.sign(w.loc[active, "B"])).all()


def test_engine_buy_and_hold_matches_price():
    df = synth_ohlcv(500, seed=5)
    w = pd.DataFrame({"X": 1.0}, index=df.index)
    res = engine.simulate(w, df[["close"]].rename(columns={"close": "X"}))
    assert res["equity"].iloc[-1] == pytest.approx(df["close"].iloc[-1] / df["close"].iloc[0], rel=1e-9)
    costly = engine.simulate(w, df[["close"]].rename(columns={"close": "X"}), spread_bps=10, funding_annual=0.05)
    assert costly["equity"].iloc[-1] < res["equity"].iloc[-1]


def test_engine_single_strategy_equal_weights(frames):
    res = engine.backtest("sma_cross", frames, {"fast": 20, "slow": 100})
    assert res["weights"].max().max() <= 1 / len(frames) + 1e-12
    assert res["trades"] > 0 and np.isfinite(res["equity"].iloc[-1])


def test_overlays(frames):
    df = frames["S0"]
    pos = strategies.sma_cross(df, 20, 100, short=True)
    vt = strategies.vol_target(pos, df, target=0.10)
    assert (vt.abs() <= 2.0 + 1e-9).all()
    rf = strategies.regime_filter(pos, df)
    up = df["close"] > df["close"].rolling(200).mean()
    assert not ((rf < 0) & up).any() and not ((rf > 0) & ~up).any()


def test_psr_dsr_behave():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 2000))
    flat = pd.Series(rng.normal(0.0, 0.01, 2000))
    assert validation.psr(good) > 0.95 > validation.psr(flat)
    assert validation.dsr(good, trials=100) < validation.dsr(good, trials=1)
    assert validation.expected_max_sharpe(1, 0.01) == 0


def test_bootstrap_ci_brackets_sharpe():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    b = validation.bootstrap_sharpe(r, n_boot=200)
    assert b["ci_low"] < b["sharpe"] < b["ci_high"]


def test_permutation_detects_foresight_not_noise():
    df = synth_ohlcv(1200, seed=9)
    closes = df[["close"]].rename(columns={"close": "X"})
    nxt = np.sign(closes["X"].pct_change().shift(-1)).fillna(0)
    cheat = pd.DataFrame({"X": nxt})
    rng = np.random.default_rng(2)
    noise = pd.DataFrame({"X": rng.choice([-1.0, 1.0], len(df))}, index=df.index)
    assert validation.permutation_test(cheat, closes, n_perm=200)["perm_p"] < 0.01
    assert validation.permutation_test(noise, closes, n_perm=200)["perm_p"] > 0.01


def test_walk_forward_and_sensitivity(frames):
    one = {"S0": frames["S0"]}
    grid = {"fast": [10, 20], "slow": [100, 150]}
    sens = validation.sensitivity("sma_cross", one, grid)
    assert len(sens) == 4 and sens["sharpe"].notna().all()
    wf = validation.walk_forward("sma_cross", one, grid, train=500, test=200)
    assert len(wf["picks"]) >= 3 and np.isfinite(wf["oos_sharpe"])


def test_signal_mode(frames):
    sig = engine.latest_signal("tsmom", frames)
    assert set(sig.index) == set(frames) and sig["target_weight"].abs().max() <= 0.25 + 1e-12


def test_paper_book_idempotent_per_bar(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    s1 = paper.rebalance("t", "tsmom", frames, capital=10_000)
    s2 = paper.rebalance("t", "tsmom", frames, capital=10_000)
    assert s1["fills"] == s2["fills"] and s2["bars"] == 1
    # next bar: book trades only the delta and marks to market
    later = {k: pd.concat([v, v.iloc[[-1]].set_axis([v.index[-1] + pd.Timedelta(days=1)])]) for k, v in frames.items()}
    s3 = paper.rebalance("t", "tsmom", later)
    assert s3["bars"] == 2
    with pytest.raises(ValueError):
        paper.rebalance("t", "sma_cross", frames)
    assert "t" in paper.status().index


def test_hypothesis_lifecycle(frames, tmp_path):
    p = hypothesis.new("trend on synth", "12-1 momentum beats flat on synthetic data",
                       {"strategy": "tsmom", "symbols": list(frames)[:2],
                        "pass": {"min_trades": 1, "perm_p_max": 1.0}}, directory=tmp_path)
    assert hypothesis.listing(tmp_path).iloc[0]["verdict"] == "PENDING"
    sub = {k: frames[k] for k in list(frames)[:2]}
    res = hypothesis.test(p, frames=sub, n_perm=50, n_boot=50)
    assert res["verdict"] == "PASS"
    txt = p.read_text()
    assert "Verdict: PASS" in txt and "## Results" in txt and "Graded" in txt
    with pytest.raises(ValueError, match="already graded"):
        hypothesis.test(p, frames=sub)
    assert hypothesis.trial_count(tmp_path) == 1


def test_hypothesis_refuses_spec_edited_after_registration(frames, tmp_path):
    p = hypothesis.new("tamper", "x", {"strategy": "sma_cross", "symbols": ["S0"]}, directory=tmp_path)
    p.write_text(p.read_text().replace('"oos_sharpe_min": 0.5', '"oos_sharpe_min": -5'))
    with pytest.raises(ValueError, match="spec changed"):
        hypothesis.test(p, frames={"S0": frames["S0"]})


def test_grade_fails_on_any_missed_bar():
    v, rows = hypothesis.grade({"oos_sharpe": 0.8, "trades": 10, "max_dd": -0.4},
                               {"oos_sharpe_min": 0.5, "min_trades": 30, "max_dd_max": 0.3})
    assert v == "FAIL" and [r["pass"] for r in rows] == [True, False, False]
