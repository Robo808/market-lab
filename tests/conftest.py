"""Shared synthetic market data for offline tests (no network)."""
import numpy as np
import pandas as pd


def synth_ohlcv(n: int = 2520, seed: int = 7, start: str = "2015-01-02") -> pd.DataFrame:
    """Geometric random walk with drift/vol regimes; OHLC built around the close, lognormal volume."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    regimes = [(0.0006, 0.010), (-0.0005, 0.018), (0.0, 0.008), (0.0008, 0.012), (-0.0002, 0.022)]
    mu, sig = np.empty(n), np.empty(n)
    i = 0
    while i < n:
        m, s = regimes[rng.integers(len(regimes))]
        L = int(rng.integers(60, 250))
        mu[i:i + L], sig[i:i + L] = m, s
        i += L
    r = rng.normal(mu, sig)
    c = 100 * np.exp(np.cumsum(r))
    gap = rng.normal(0, sig * 0.3)
    o = np.r_[c[0], c[:-1]] * np.exp(gap)
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, sig * 0.5)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, sig * 0.5)))
    v = rng.lognormal(13, 0.4, n) * (1 + 20 * np.abs(r))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


def from_path(closes, spread: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """OHLCV frame following a given close path (open = previous close) with small wicks."""
    rng = np.random.default_rng(seed)
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, spread, len(c))) * 0.5 + spread * 0.25
    idx = pd.date_range("2020-01-01", periods=len(c), freq="B", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w, "close": c,
                         "volume": np.full(len(c), 1e6)}, index=idx)
