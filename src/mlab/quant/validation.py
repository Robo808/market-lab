"""Is the edge real? Out-of-sample split, walk-forward, probabilistic and deflated Sharpe,
block-bootstrap confidence interval, Monte Carlo permutation test and parameter sensitivity.

References: Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (2012, PSR) and
"The Deflated Sharpe Ratio" (2014, DSR); Politis & Romano (1994) stationary bootstrap;
Aronson, "Evidence-Based Technical Analysis" (2006) for permutation tests.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import engine

EULER = 0.5772156649015329


def sharpe(r: pd.Series, periods: int = 252) -> float:
    r = pd.Series(r).dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd and len(r) > 1 else float("nan")


def psr(r: pd.Series, sr_benchmark: float = 0.0, periods: int = 252) -> float:
    """Probability the true Sharpe exceeds `sr_benchmark` (annualised), allowing for skew and fat tails."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 3 or not r.std():
        return float("nan")
    sr = r.mean() / r.std()                       # per-period
    sr0 = sr_benchmark / np.sqrt(periods)
    g3, g4 = r.skew(), r.kurt() + 3               # raw kurtosis
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trials: int, sr_var: float) -> float:
    """Expected maximum of `trials` Sharpe estimates under the null (per-period units)."""
    if trials <= 1:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / trials) + EULER * norm.ppf(1 - 1 / (trials * np.e))))


def dsr(r: pd.Series, trials: int = 1, sr_var: float | None = None, periods: int = 252) -> float:
    """Deflated Sharpe: PSR against the Sharpe you would expect from the best of `trials` random tries.
    sr_var: variance of the per-period Sharpe across the trials; defaults to the estimator variance 1/n."""
    r = pd.Series(r).dropna()
    if len(r) < 3:
        return float("nan")
    v = sr_var if sr_var is not None else 1 / len(r)
    return psr(r, expected_max_sharpe(trials, v) * np.sqrt(periods), periods)


def bootstrap_sharpe(r: pd.Series, n_boot: int = 1000, block: int = 20, periods: int = 252, seed: int = 0) -> dict:
    """Stationary block bootstrap (geometric block lengths) CI for the annualised Sharpe."""
    x = pd.Series(r).dropna().values
    n = len(x)
    if n < 2 * block:
        return {"sharpe": sharpe(pd.Series(x), periods), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    p = 1 / block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, int)
        idx[0] = rng.integers(n)
        jumps = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for i in range(1, n):
            idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
        s = x[idx]
        out[b] = s.mean() / s.std() * np.sqrt(periods) if s.std() else 0.0
    return {"sharpe": sharpe(pd.Series(x), periods), "ci_low": float(np.quantile(out, 0.05)),
            "ci_high": float(np.quantile(out, 0.95)), "p_sharpe_le_0": float((out <= 0).mean())}


def permutation_test(weights: pd.DataFrame, closes: pd.DataFrame, n_perm: int = 500, seed: int = 0,
                     periods: int = 252, **sim_kw) -> dict:
    """Shuffle the market's returns against the strategy's positions: how often does luck match the real Sharpe?
    Keeps the strategy's exposure profile; destroys any timing skill."""
    real = engine.simulate(weights, closes, periods=periods, **sim_kw)["returns"]
    sr = sharpe(real, periods)
    rets = closes.reindex(weights.index).ffill().pct_change().fillna(0.0).values
    held = weights.shift().fillna(0.0).values
    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(n_perm):
        perm = rets[rng.permutation(len(rets))]
        pr = (held * perm).sum(axis=1)
        if pr.std() and pr.mean() / pr.std() * np.sqrt(periods) >= sr:
            beats += 1
    return {"sharpe": sr, "perm_p": (beats + 1) / (n_perm + 1), "n_perm": n_perm}


def split_stats(returns: pd.Series, oos_start: str | None, periods: int = 252) -> dict:
    r = returns.dropna()
    if not oos_start:
        cut = r.index[int(len(r) * 0.7)]
    else:
        cut = pd.Timestamp(oos_start, tz=r.index.tz)
    ins, oos = r[r.index < cut], r[r.index >= cut]
    def block(x):
        eq = (1 + x).cumprod()
        dd = float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")
        return {"sharpe": sharpe(x, periods), "ann_return": float(x.mean() * periods), "max_dd": dd, "bars": len(x)}
    return {"oos_start": str(cut)[:10], "in_sample": block(ins), "out_of_sample": block(oos)}


def sensitivity(name: str, frames: dict, grid: dict[str, list], periods: int = 252, **bt_kw) -> pd.DataFrame:
    """Sharpe over a parameter grid. A robust edge is a plateau, a fragile one is a lone spike."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(keys, combo))
        try:
            res = engine.backtest(name, frames, p, periods=periods, **bt_kw)
            rows.append({**p, "sharpe": sharpe(res["returns"], periods), "trades": res["trades"],
                         "max_dd": res["stats"].get("max_drawdown")})
        except Exception as e:  # invalid combos (e.g. fast >= slow) are reported, not fatal
            rows.append({**p, "sharpe": float("nan"), "error": str(e)[:60]})
    return pd.DataFrame(rows)


def walk_forward(name: str, frames: dict, grid: dict[str, list], train: int = 756, test: int = 126,
                 periods: int = 252, **bt_kw) -> dict:
    """Re-pick params on each train window by Sharpe, stitch the untouched test windows together."""
    closes = pd.DataFrame({k: v["close"] for k, v in frames.items()}).dropna(how="all")
    n = len(closes)
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())] or [{}]
    pieces, picks = [], []
    for start in range(0, n - train - test + 1, test):
        tr_idx = closes.index[start:start + train]
        te_idx = closes.index[start + train:start + train + test]
        def score(p):
            r = engine.backtest(name, frames, p, periods=periods, **bt_kw)["returns"]
            return np.nan_to_num(sharpe(r.loc[tr_idx[0]:tr_idx[-1]], periods), nan=-9)
        best = max(combos, key=score)
        r = engine.backtest(name, frames, best, periods=periods, **bt_kw)["returns"]  # signals use only past data
        pieces.append(r.loc[te_idx[0]:te_idx[-1]])
        picks.append({"test_start": str(te_idx[0])[:10], **best})
    oos = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return {"oos_returns": oos, "oos_sharpe": sharpe(oos, periods), "picks": pd.DataFrame(picks)}


def report(name: str, frames: dict, params: dict | None = None, overlays: dict | None = None, trials: int = 1,
           oos_start: str | None = None, n_perm: int = 500, n_boot: int = 500, periods: int = 252, **bt_kw) -> dict:
    res = engine.backtest(name, frames, params, overlays, periods=periods, **bt_kw)
    r = res["returns"]
    closes = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    first = res["weights"].abs().sum(axis=1).gt(0).idxmax()   # skip warm-up bars with no position possible
    r_live = r.loc[first:]
    split = split_stats(r_live, oos_start, periods)
    return {
        "strategy": name, "params": params or {}, "trades": res["trades"], "exposure": res["exposure"],
        "sharpe": sharpe(r_live, periods), "psr_vs_0": psr(r_live, 0.0, periods),
        "dsr": dsr(r_live, trials, periods=periods), "trials": trials,
        "max_dd": res["stats"].get("max_drawdown"), "cagr": res["stats"].get("cagr"),
        "cost_drag": res["cost_drag"], "split": split,
        "bootstrap": bootstrap_sharpe(r_live, n_boot, periods=periods),
        "permutation": permutation_test(res["weights"], closes, n_perm, periods=periods,
                                        **{k: v for k, v in bt_kw.items() if k in ("spread_bps", "funding_annual")}),
        "benchmark": res["benchmark"], "result": res,
    }
