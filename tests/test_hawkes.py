"""Hawkes branching-ratio estimator: recovers known parameters, and features never look ahead. Synthetic data only."""
import numpy as np
import pandas as pd

from mlab.quant import hawkes as hk


def test_fit_recovers_branching_ratio():
    t = hk.simulate(mu=0.05, n=0.6, beta=0.1, T=6000, seed=3)
    post = hk.fit(t, 6000.0, (1.0, 200.0), iters=3000, burn=1000, seed=1)
    assert 0.45 < post["n_mean"] < 0.75 and post["n_lo"] < 0.6 < post["n_hi"]
    assert 5 < post["decay_bars"] < 20


def test_fit_sees_no_excitation_in_poisson_events():
    t = np.sort(np.random.default_rng(4).uniform(0, 6000, 300))
    assert hk.fit(t, 6000.0, (1.0, 20.0), iters=3000, burn=1000, seed=1)["n_mean"] < 0.2


def test_loglik_matches_direct_integral():
    t, T, mu, n, beta = np.array([1.0, 2.5, 7.0]), 10.0, 0.3, 0.5, 0.8
    lam = [mu + n * beta * sum(np.exp(-beta * (ti - tj)) for tj in t if tj < ti) for ti in t]
    grid = np.linspace(0, T, 200001)
    lam_g = mu + n * beta * sum(np.where(grid > tj, np.exp(-beta * (grid - tj)), 0) for tj in t)
    direct = np.log(lam).sum() - np.trapezoid(lam_g, grid)
    assert abs(hk.loglik(mu, n, beta, t, T) - direct) < 1e-3


def _close(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.01)), index=idx)


SPEC = {"sigma_window": 100, "k_sigma": 3.0, "fit_window": 500, "horizon": 24, "every": 200,
        "decay_bounds": [1, 100], "mcmc": {"iters": 400, "burn": 100, "step": 0.3}}


def test_events_and_features_use_only_past_bars():
    c = _close()
    full = hk.extreme_events(c, 100, 3.0)
    assert full.iloc[:2000].equals(hk.extreme_events(c.iloc[:2000], 100, 3.0))
    a = hk.panel(c, SPEC)
    b = hk.panel(c.iloc[:2200], SPEC)  # truncating the future changes nothing already computed
    cols = ["n", "events", "log_rv1", "log_rv4"]
    pd.testing.assert_frame_equal(a[cols].iloc[:len(b)], b[cols])
