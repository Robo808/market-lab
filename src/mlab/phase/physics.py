"""Physics layer: which force dominates in each regime, how predictable the series is, whether the cycles are
real, and how the strategy behaves on histories that did not happen.

- Inertia vs restoring force per regime: correlation of the past q-bar return with the next q-bar return
  (momentum > 0, mean reversion < 0), and an Ornstein-Uhlenbeck fit of the deviation from the 63-bar mean,
  with its half-life measured against the same fit on return-shuffled paths (a deviation from a moving average
  mean-reverts mechanically, so the null matters; Lo & MacKinlay 1988, Poterba & Summers 1988).
- Permutation entropy (Bandt & Pompe 2002; Zunino et al. 2009 on market efficiency): forecast skill by entropy
  bucket, buckets set on the first training years only.
- Spectral check: Fisher's g test on periodograms of raw log prices vs returns. Raw prices are red noise and
  "find" long cycles; returns are where a real cycle would have to show up (Granger 1966).
- Monte Carlo: stationary block bootstrap (Politis & Romano 1994) of the out-of-sample returns for drawdown and
  Sharpe ranges, and synthetic markets rebuilt from resampled bars, run through the same walk-forward, to see what
  the pipeline "finds" when there is no structure beyond the block length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..quant import validation


def _dev(c: pd.Series, n: int = 63) -> pd.Series:
    lc = np.log(c)
    return lc - lc.rolling(n).mean()


def forces_by_regime(frames: dict[str, pd.DataFrame], cube: pd.DataFrame, regime: pd.Series, names: list[str],
                     qs=(1, 5, 21), seed: int = 0) -> pd.DataFrame:
    """Per regime: lead-lag correlation of q-bar returns (inertia > 0, restoring < 0) and OU half-life vs null."""
    rng = np.random.default_rng(seed)
    parts = []
    for s, df in frames.items():
        c = df["close"]
        lc = np.log(c)
        rv = cube["rv_21"].xs(s, level="symbol").reindex(df.index) if s in cube.index.get_level_values(0) else None
        if rv is None:
            continue
        d = pd.DataFrame(index=df.index)
        for q in qs:
            d[f"past_{q}"] = (lc - lc.shift(q)) / (rv * np.sqrt(q))
            d[f"next_{q}"] = (lc.shift(-q) - lc) / (rv * np.sqrt(q))
        x = _dev(c)
        d["x"] = x / rv
        d["dx"] = (x.shift(-1) - x) / rv
        r = lc.diff().dropna().to_numpy()                  # null: same returns, shuffled order
        xn = _dev(pd.Series(np.exp(np.r_[lc.iloc[0], lc.iloc[0] + np.cumsum(rng.permutation(r))]), index=df.index))
        d["x0"] = xn / rv
        d["dx0"] = (xn.shift(-1) - xn) / rv
        d["regime"] = regime.xs(s, level="symbol").reindex(df.index) if s in regime.index.get_level_values(0) else np.nan
        parts.append(d)
    D = pd.concat(parts).dropna(subset=["regime"])
    rows = []
    for j, g in D.groupby("regime"):
        row = {"regime": int(j), "name": names[int(j)], "bars": len(g)}
        for q in qs:
            gg = g[[f"past_{q}", f"next_{q}"]].dropna()
            rho = gg.corr().iloc[0, 1] if len(gg) > 30 else np.nan
            row[f"corr_q{q}"] = rho
            row[f"t_q{q}"] = rho * np.sqrt(max(len(gg) / q - 2, 1)) / np.sqrt(max(1 - rho ** 2, 1e-9))
        for tag, xc, dc in (("ou", "x", "dx"), ("null", "x0", "dx0")):
            gg = g[[xc, dc]].dropna()
            if len(gg) > 30:
                b = np.polyfit(gg[xc], gg[dc], 1)[0]
                kappa = -np.log1p(b) if b > -1 else np.nan
                row[f"{tag}_halflife"] = np.log(2) / kappa if kappa and kappa > 0 else np.inf
                if tag == "ou":
                    row["ou_b"] = b
                else:
                    row["null_b"] = b
        tq = {q: row.get(f"t_q{q}", 0.0) for q in qs}
        q_star = max(tq, key=lambda q: abs(np.nan_to_num(tq[q])))
        t_star = tq[q_star]
        row["force"] = (f"inertia at {q_star} bars (t {t_star:+.1f})" if t_star >= 3 else
                        f"restoring at {q_star} bars (t {t_star:+.1f})" if t_star <= -3 else "neither (|t|<3)")
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def entropy_skill(pred: pd.Series, y: pd.Series, pe: pd.Series, calib_end: pd.Timestamp) -> pd.DataFrame:
    """Out-of-sample IC split by permutation-entropy tercile, tercile edges from data before `calib_end`."""
    t = pe.index.get_level_values("time")
    edges = pe[t < calib_end].dropna().quantile([1 / 3, 2 / 3]).to_numpy()
    d = pd.DataFrame({"p": pred, "y": y.reindex(pred.index), "pe": pe.reindex(pred.index)}).dropna()
    d["bucket"] = np.select([d["pe"] <= edges[0], d["pe"] <= edges[1]], ["low entropy", "mid entropy"], "high entropy")
    rows = []
    for b in ("low entropy", "mid entropy", "high entropy"):
        g = d[d["bucket"] == b]
        rows.append({"bucket": b, "pe_range": f"<= {edges[0]:.3f}" if b.startswith("low") else
                     f"{edges[0]:.3f}-{edges[1]:.3f}" if b.startswith("mid") else f"> {edges[1]:.3f}",
                     "rows": len(g), "ic": spearmanr(g["p"], g["y"]).statistic if len(g) > 30 else np.nan,
                     "hit": float((np.sign(g["p"]) == np.sign(g["y"])).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def fisher_g(x: np.ndarray) -> tuple[float, float, float]:
    """Fisher's g test for the largest periodogram peak. Returns (period in bars, g, p-value)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    I = np.abs(np.fft.rfft(x)) ** 2
    I = I[1:len(x) // 2]
    m = len(I)
    if m < 2:
        return float("nan"), float("nan"), float("nan")
    k = int(I.argmax())
    g = I[k] / I.sum()
    # first term of Fisher's (1929) exact distribution: accurate where it matters (small p), capped at 1
    p = m * (1 - g) ** (m - 1)
    return float(len(x) / (k + 1)), float(g), float(min(max(p, 0.0), 1.0))


def spectral_check(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in frames.items():
        lc = np.log(df["close"]).dropna()
        p_raw, _, pv_raw = fisher_g(lc.to_numpy())
        p_ret, _, pv_ret = fisher_g(lc.diff().dropna().to_numpy())
        rows.append({"symbol": s, "raw_peak_period": round(p_raw, 1), "raw_p": pv_raw,
                     "returns_peak_period": round(p_ret, 1), "returns_p": pv_ret})
    return pd.DataFrame(rows).set_index("symbol")


def stationary_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, int)
    idx[0] = rng.integers(n)
    jump = rng.random(n) < 1 / block
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jump[i] else (idx[i - 1] + 1) % n
    return idx


def stress(r: pd.Series, n_paths: int = 2000, block: float = 21, seed: int = 0) -> dict:
    """Block-bootstrapped alternative histories of the out-of-sample returns: Sharpe, max drawdown and worst
    year ranges (5th/50th/95th percentiles) and the probability of a losing 12 months."""
    x = r.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    sh, dd, worst = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    for b in range(n_paths):
        s = x[stationary_indices(n, block, rng)]
        sh[b] = s.mean() / s.std() * np.sqrt(252) if s.std() else 0.0
        eq = np.cumprod(1 + s)
        dd[b] = (eq / np.maximum.accumulate(eq) - 1).min()
        yr = np.add.reduceat(s, np.arange(0, n, 252))[:-1] if n > 504 else s.sum()[None]
        worst[b] = np.min(yr)
    q = lambda a: [float(np.quantile(a, p)) for p in (0.05, 0.5, 0.95)]
    return {"sharpe_5_50_95": q(sh), "max_dd_5_50_95": q(dd), "worst_year_5_50_95": q(worst),
            "p_sharpe_le_0": float((sh <= 0).mean()), "paths": n_paths}


def synthetic_frames(frames: dict[str, pd.DataFrame], block: float = 21, seed: int = 0) -> dict[str, pd.DataFrame]:
    """A market that did not happen: each instrument's bars (close-to-close return, open/high/low relative to
    the close, volume) resampled in stationary blocks and re-chained from the first close. Keeps volatility
    clustering up to the block length; destroys longer trends, regimes and cross-asset timing."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, df in frames.items():
        c = df["close"].to_numpy(float)
        lr = np.r_[0.0, np.diff(np.log(c))]
        rel = np.log(df[["open", "high", "low"]].to_numpy(float) / c[:, None])
        idx = stationary_indices(len(c), block, rng)
        nc = c[0] * np.exp(np.cumsum(lr[idx]))
        o, h, lo = (nc[:, None] * np.exp(rel[idx])).T
        d = pd.DataFrame({"open": o, "high": np.maximum.reduce([o, h, nc]), "low": np.minimum.reduce([o, lo, nc]),
                          "close": nc}, index=df.index)
        if "volume" in df:
            d["volume"] = df["volume"].to_numpy()[idx]
        if "spread" in df:
            d["spread"] = (df["spread"] / df["close"]).to_numpy()[idx] * nc
        out[s] = d
    return out


def null_world(frames: dict[str, pd.DataFrame], cfg: dict, n: int = 10, block: float = 21, embargo: int = 5,
               delay: int = 0, funding: float = 0.02, log=print) -> pd.DataFrame:
    """Run one fixed configuration through the full walk-forward on n synthetic markets (nothing is logged as a
    trial: these are calibration runs of the method, not candidate strategies)."""
    from . import evaluate as E
    rows = []
    for i in range(n):
        syn = synthetic_frames(frames, block, seed=1000 + i)
        S = E.prepare(syn, family="__null_world__")
        cache = E.FoldCache(S, embargo, log=lambda *_: None)
        res = E.run_config(S, cache, cfg, delay, funding)
        m = E.summarize(res["returns"], lags=max(5, cfg["horizon"]))
        rows.append({"world": i, "sharpe": m["sharpe"], "t_nw": m["t_nw"], "ic": res["ic"]})
        log(f"  synthetic market {i}: Sharpe {m['sharpe']:+.2f}, t {m['t_nw']:+.2f}, IC {res['ic']:+.4f}")
    return pd.DataFrame(rows)


def sharpe(r: pd.Series) -> float:
    return validation.sharpe(r)
