"""Bayesian self-exciting (Hawkes) model of extreme moves: the branching ratio as a reflexivity gauge.

Seismology's ETAS aftershock model (Ogata 1988) is a marked Hawkes process. Filimonov & Sornette (2012, PRE,
arXiv:1201.3572) read the branching ratio n of a Hawkes fit to market events as the share of activity triggered by
earlier activity rather than by outside news. Here n is estimated by random-walk Metropolis on trailing windows of
extreme-move times (bars where |log return| beats k trailing sigmas), and the pre-registered study asks whether n adds
to a HAR-style baseline when forecasting the next period's realised volatility.

    python -m mlab.quant.hawkes research/hypotheses/H-20261014-hawkes-branching-vol.md [--test]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def extreme_events(close: pd.Series, sigma_window: int, k: float) -> pd.Series:
    """True on bars whose |log return| exceeds k times the std of the previous `sigma_window` returns (bar excluded)."""
    r = np.log(close.astype(float)).diff()
    sig = r.rolling(sigma_window, min_periods=sigma_window).std().shift(1)
    return (r.abs() > k * sig).fillna(False)


def loglik(mu: float, n: float, beta: float, t: np.ndarray, T: float) -> float:
    """Exponential-kernel Hawkes log likelihood on [0, T]: lambda(s) = mu + n*beta*sum exp(-beta (s - t_i))."""
    if len(t) == 0:
        return -mu * T
    d = t[:, None] - t[None, :]
    a = np.where(d > 0, np.exp(-beta * np.clip(d, 0, None)), 0.0).sum(1)
    lam = mu + n * beta * a
    return float(np.log(lam).sum() - mu * T - n * np.sum(1.0 - np.exp(-beta * (T - t))))


def fit(t: np.ndarray, T: float, decay_bounds: tuple[float, float], iters: int = 4000, burn: int = 1000,
        step: float = 0.3, seed: int = 0) -> dict:
    """Posterior of (mu, n, beta) by random-walk Metropolis on (log mu, logit n, log beta).

    Priors: flat on log mu; n ~ Uniform(0, 1) (stationary); 1/beta log-uniform within decay_bounds (in bars).
    """
    t = np.sort(np.asarray(t, float))
    rng = np.random.default_rng(seed)
    b_lo, b_hi = 1.0 / decay_bounds[1], 1.0 / decay_bounds[0]

    def logpost(z):
        mu, n, beta = np.exp(z[0]), 1.0 / (1.0 + np.exp(-z[1])), np.exp(z[2])
        if not (b_lo <= beta <= b_hi):
            return -np.inf
        return loglik(mu, n, beta, t, T) + np.log(n) + np.log1p(-n)  # Jacobian of logit for the uniform prior on n

    z = np.array([np.log(max(len(t), 1) / T * 0.5), 0.0, 0.5 * (np.log(b_lo) + np.log(b_hi))])
    lp = logpost(z)
    draws, acc = [], 0
    for i in range(iters):
        prop = z + step * rng.standard_normal(3)
        lp2 = logpost(prop)
        if np.log(rng.random()) < lp2 - lp:
            z, lp = prop, lp2
            acc += i >= burn
        if i >= burn:
            draws.append(z.copy())
    d = np.array(draws)
    n = 1.0 / (1.0 + np.exp(-d[:, 1]))
    return {"n_mean": float(n.mean()), "n_lo": float(np.quantile(n, 0.05)), "n_hi": float(np.quantile(n, 0.95)),
            "p_n_gt_0_5": float((n > 0.5).mean()), "mu_mean": float(np.exp(d[:, 0]).mean()),
            "decay_bars": float(np.median(1.0 / np.exp(d[:, 2]))), "accept": acc / max(iters - burn, 1),
            "events": int(len(t))}


def simulate(mu: float, n: float, beta: float, T: float, seed: int = 0) -> np.ndarray:
    """Ogata thinning; used by the tests to check that `fit` recovers known parameters."""
    rng = np.random.default_rng(seed)
    out, s, excite = [], 0.0, 0.0  # excite = n*beta*sum exp(-beta (s - t_i)) at time s
    while True:
        lam_bar = mu + excite
        w = rng.exponential(1.0 / lam_bar)
        s += w
        excite *= np.exp(-beta * w)
        if s > T:
            return np.array(out)
        if rng.random() * lam_bar <= mu + excite:
            out.append(s)
            excite += n * beta


def panel(close: pd.Series, spec: dict, seed: int = 0) -> pd.DataFrame:
    """One row per evaluation bar i (every `every` bars): Hawkes posterior from events in (i-L, i], HAR features from
    returns up to i, and the target log realised vol over (i, i+H]. Uses only bars <= i for features."""
    close = close.dropna().astype(float)
    r = np.log(close).diff().to_numpy()
    ev = extreme_events(close, spec["sigma_window"], spec["k_sigma"]).to_numpy()
    L, H, every = spec["fit_window"], spec["horizon"], spec["every"]
    start = spec["sigma_window"] + L + 1
    rows = []
    for j, i in enumerate(range(start, len(close) - H, every)):
        idx = np.flatnonzero(ev[i - L + 1:i + 1]).astype(float) + 1.0  # times in (0, L]
        post = fit(idx, float(L), tuple(spec["decay_bounds"]), spec["mcmc"]["iters"], spec["mcmc"]["burn"],
                   spec["mcmc"]["step"], seed=seed + j)
        rv1 = np.sqrt(np.nansum(r[i - H + 1:i + 1] ** 2))
        rv4 = np.sqrt(np.nansum(r[i - 4 * H + 1:i + 1] ** 2) / 4)
        fwd = r[i + 1:i + H + 1]
        rows.append({"time": close.index[i], "fwd_end": close.index[i + H], "n": post["n_mean"],
                     "n_lo": post["n_lo"], "n_hi": post["n_hi"], "events": post["events"],
                     "decay_bars": post["decay_bars"], "accept": post["accept"],
                     "log_rv1": np.log(rv1), "log_rv4": np.log(rv4), "y": np.log(np.sqrt(np.nansum(fwd ** 2))),
                     "fwd_dd": float(min(np.nancumsum(fwd).min(), 0.0))})
    return pd.DataFrame(rows)


def _split_rows(p: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    d0 = pd.Timestamp(lo, tz="UTC")
    d1 = pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)
    t, e = pd.to_datetime(p["time"], utc=True), pd.to_datetime(p["fwd_end"], utc=True)
    return p[(t >= d0) & (e < d1)]  # forward window must end inside the split


def _design(p: pd.DataFrame, with_n: bool, n_mu: float, n_sd: float) -> pd.DataFrame:
    x = pd.get_dummies(p["symbol"], dtype=float)  # one intercept per symbol
    x["log_rv1"], x["log_rv4"] = p["log_rv1"].to_numpy(), p["log_rv4"].to_numpy()
    if with_n:
        x["n_z"] = ((p["n"] - n_mu) / n_sd).to_numpy()
    return x


def regress(p: pd.DataFrame, n_mu: float, n_sd: float, lags: int):
    """Pooled OLS, Driscoll-Kraay errors (robust to cross-symbol and serial correlation)."""
    import statsmodels.api as sm
    p = p.sort_values("time")
    tcode = pd.factorize(p["time"].dt.floor("D"))[0]
    return sm.OLS(p["y"].to_numpy(), _design(p, True, n_mu, n_sd)).fit(
        cov_type="hac-groupsum", cov_kwds={"time": tcode, "maxlags": lags})


def oos_mse_gain(train: pd.DataFrame, test: pd.DataFrame, n_mu: float, n_sd: float) -> float:
    """1 - MSE(base + n) / MSE(base) on `test`, both models fitted on `train`."""
    def mse(with_n):
        xtr, xte = _design(train, with_n, n_mu, n_sd), _design(test, with_n, n_mu, n_sd)
        xte = xte.reindex(columns=xtr.columns, fill_value=0.0)
        b = np.linalg.lstsq(xtr.to_numpy(), train["y"].to_numpy(), rcond=None)[0]
        return float(np.mean((test["y"].to_numpy() - xte.to_numpy() @ b) ** 2))
    return 1.0 - mse(True) / mse(False)


def run_leg(leg: dict, splits: list[str], open_test: bool, note: str, log=print) -> dict:
    from .. import datasets
    split = "train+validation" + ("+test" if open_test else "")
    frames = datasets.load(leg["dataset"], leg["symbols"], split, allow_test=open_test, note=note)
    man = datasets.manifest(leg["dataset"])["splits"]
    parts = []
    for s, df in frames.items():
        log(f"  {s}: {len(df)} bars {str(df.index[0])[:10]}..{str(df.index[-1])[:10]}")
        p = panel(df["close"], leg)
        p["symbol"] = s
        parts.append(p)
    p = pd.concat(parts, ignore_index=True)
    p["time"] = pd.to_datetime(p["time"], utc=True)
    sp = {k: _split_rows(p, *man[k]) for k in splits}
    tr = sp["train"]
    n_mu, n_sd = float(tr["n"].mean()), float(tr["n"].std())
    out = {"rows": {k: len(v) for k, v in sp.items()}, "n_train_mean": n_mu, "n_train_sd": n_sd,
           "mean_events_per_window": float(tr["events"].mean()), "mean_accept": float(tr["accept"].mean())}
    for k, v in sp.items():
        m = regress(v, n_mu, n_sd, leg["hac_lags"])
        out[k] = {"coef_n_z": float(m.params["n_z"]), "t_n_z": float(m.tvalues["n_z"]), "r2": float(m.rsquared),
                  "nobs": int(m.nobs)}
        q = pd.qcut(v["n"], 3, labels=["low", "mid", "high"])
        out[k]["by_n_tercile"] = {str(g): {"n": round(float(x["n"].mean()), 3),
                                           "fwd_rv_mean": round(float(np.exp(x["y"]).mean()), 5),
                                           "fwd_dd_mean": round(float(x["fwd_dd"].mean()), 5),
                                           "fwd_dd_p10": round(float(x["fwd_dd"].quantile(0.1)), 5)}
                                  for g, x in v.groupby(q, observed=True)}
    out["validation"]["oos_mse_gain"] = oos_mse_gain(tr, sp["validation"], n_mu, n_sd)
    if "test" in sp:
        trv = pd.concat([tr, sp["validation"]])
        out["test"]["oos_mse_gain"] = oos_mse_gain(trv, sp["test"], n_mu, n_sd)
    return out


def grade_leg(res: dict, bars: dict, stage: str) -> tuple[bool, list[dict]]:
    rows = []
    if stage == "train+validation":
        b = bars["train"]
        rows.append({"bar": "train t(n_z)", "need": f">= {b['t_min']}", "got": res["train"]["t_n_z"],
                     "pass": res["train"]["t_n_z"] >= b["t_min"]})
        b = bars["validation"]
        rows.append({"bar": "validation coef(n_z)", "need": "> 0", "got": res["validation"]["coef_n_z"],
                     "pass": res["validation"]["coef_n_z"] > 0})
        rows.append({"bar": "validation OOS MSE gain", "need": f">= {b['oos_mse_gain_min']}",
                     "got": res["validation"]["oos_mse_gain"],
                     "pass": res["validation"]["oos_mse_gain"] >= b["oos_mse_gain_min"]})
    else:
        b = bars["test"]
        rows.append({"bar": "test coef(n_z)", "need": "> 0", "got": res["test"]["coef_n_z"],
                     "pass": res["test"]["coef_n_z"] > 0})
        rows.append({"bar": "test OOS MSE gain", "need": f"> {b['oos_mse_gain_min']}",
                     "got": res["test"]["oos_mse_gain"], "pass": res["test"]["oos_mse_gain"] > b["oos_mse_gain_min"]})
    return all(r["pass"] for r in rows), rows


def run(path: Path, open_test: bool = False, legs: list[str] | None = None, log=print) -> dict:
    from ..textlab.studies import read_spec
    spec = read_spec(path)
    splits = ["train", "validation"] + (["test"] if open_test else [])
    out = {}
    for name, leg in spec["legs"].items():
        if legs and name not in legs:
            continue
        log(f"leg {name}")
        res = run_leg(leg, splits, open_test, f"{Path(path).name} test split", log=log)
        ok, rows = grade_leg(res, spec["pass"], "train+validation")
        res["verdict_train_validation"], res["checks"] = ("PASS" if ok else "FAIL"), rows
        if open_test:
            ok_t, rows_t = grade_leg(res, spec["pass"], "test")
            res["verdict_test"], res["checks_test"] = ("PASS" if ok_t else "FAIL"), rows_t
        out[name] = res
    return out


if __name__ == "__main__":
    legs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--legs=")]
    print(json.dumps(run(Path(sys.argv[1]), "--test" in sys.argv, legs[0].split(",") if legs else None,
                         log=lambda m: print(m, file=sys.stderr)),
                     indent=2, default=str))
