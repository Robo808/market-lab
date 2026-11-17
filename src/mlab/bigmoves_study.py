"""Exploratory study on the bigmoves table (issue #78): can the state of the world before a big move tell it apart from
an ordinary day, call its direction, or call its newspaper-coded cause? And do pre-move states form real clusters?

Every question is a counted trial with the same bar, fixed before the run:
- statistic at least 3 standard deviations above its permutation null (the desk's t >= 3 bar), and
- Benjamini-Hochberg adjusted permutation p <= 0.01 across all trials.
The bar is checked with null stubs first (L-012): seeded coin-flip labels pushed through the identical pipeline must
all fail it, otherwise the bar is REFUSED and no result counts.

Run log (trials are counted, none are dropped):
- v1 (2026-11-17): T1-T4. T4 used 100 permutations, so its smallest possible p (0.0099) could never clear the BH
  bar: a harness defect, fixed by giving every trial the same permutation count.
- v2 (2026-11-17, added after seeing v1, so they count as extra trials): T5 checks T1 on the first move of each episode
  only (the day after a big move is itself a high-risk day); T6 is T3 on every BBDS-coded jump day 1990-2019 instead of
  the 47 that are also z >= 3 moves; T7 replaces T4's weak null (shuffling each column destroys the correlation any
  real feature set has) with ordinary days: are pre-move states more clustered than random days of the same size?

This is exploration on the train and validation years only (the 2020+ test split is never read): a PASS here is a
candidate to pre-register with `mlab hypo`, not a finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize

CORE = ["pre_ret_1d", "pre_ret_5d", "pre_ret_20d", "pre_ret_60d", "pre_rv_20d", "pre_rv_ratio_5_60", "pre_dd_252d",
        "pre_dist_sma50", "pre_dist_sma200", "pre_up_days_10", "pre_vix", "pre_vix_chg5", "pre_vix_z252",
        "pre_tnx_chg20", "pre_curve_10y_3m", "pre_dxy_ret20", "pre_gprd_z", "pre_epu_us_z", "pre_days_to_fomc",
        "pre_days_since_fomc"]
BAR = {"z_vs_null": 3.0, "bh_p": 0.01}
K_CLUSTERS = 4
N_FOLDS = 6


# =============================================================================== small, exact building blocks
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity (ties get average ranks)."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """L2-penalised logistic regression (intercept unpenalised); returns [b0, w...]."""
    Xb = np.column_stack([np.ones(len(X)), X])
    pen = np.r_[0.0, np.full(X.shape[1], lam)]

    def f(b):
        z = Xb @ b
        ll = np.sum(np.logaddexp(0, z) - y * z)
        p = 1 / (1 + np.exp(-z))
        return ll + 0.5 * np.sum(pen * b * b), Xb.T @ (p - y) + pen * b

    return minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x


def blocked_folds(dates: pd.Series, k: int = N_FOLDS) -> np.ndarray:
    """Fold id per row: k contiguous time blocks with about equal row counts."""
    order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
    fold = np.empty(len(dates), int)
    fold[order] = np.minimum(np.arange(len(dates)) * k // max(len(dates), 1), k - 1)
    return fold


def cv_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray, lam: float = 1.0) -> np.ndarray:
    s = np.empty(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2:
            s[te] = y[tr].mean()
            continue
        b = fit_logit(X[tr], y[tr], lam)
        s[te] = b[0] + X[te] @ b[1:]
    return s


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    ks = np.unique(labels)
    if len(ks) < 2:
        return np.nan
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        n_own = own.sum() - 1
        a = D[i, own].sum() / n_own if n_own > 0 else 0.0
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = 0.0 if n_own == 0 else (b - a) / max(a, b)
    return float(s.mean())


def cluster(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    best, best_inertia = None, np.inf
    for r in range(5):  # restarts, keep the tightest
        c, lab = kmeans2(X, k, minit="++", seed=seed * 100 + r)
        inertia = ((X - c[lab]) ** 2).sum()
        if inertia < best_inertia and len(np.unique(lab)) == k:
            best, best_inertia = lab, inertia
    return best if best is not None else lab


def bh(p: list[float]) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    adj[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


def design(df: pd.DataFrame, cols: list[str], max_missing: float = 0.3) -> tuple[np.ndarray, pd.DataFrame]:
    """Standardise each feature within market, fill gaps with the market mean (0), drop rows missing > max_missing."""
    cols = [c for c in cols if c in df]
    keep = df[cols].isna().mean(axis=1) <= max_missing
    d = df[keep].copy()
    Z = d.groupby("market")[cols].transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
    return Z.fillna(0.0).to_numpy(float), d


# =============================================================================== trials
def _perm_trial(stat_fn, y: np.ndarray, n_perm: int, rng) -> dict:
    obs = stat_fn(y)
    null = np.array([stat_fn(rng.permutation(y)) for _ in range(n_perm)])
    sd = null.std(ddof=1)
    return {"stat": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z_vs_null": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            "perm_p": float((1 + (null >= obs).sum()) / (n_perm + 1))}


def classification_trial(X, y, dates, n_perm, rng, lam=1.0) -> dict:
    folds = blocked_folds(dates)
    out = _perm_trial(lambda yy: auc(yy, cv_scores(X, yy.astype(float), folds, lam)), y.astype(float), n_perm, rng)
    out["n"], out["positives"] = int(len(y)), int(y.sum())
    return out


def cluster_trial(X, n_perm, rng, k=K_CLUSTERS, seed=0) -> dict:
    def stat(Xp):
        return silhouette(Xp, cluster(Xp, k, seed))
    obs = stat(X)
    null = np.array([stat(np.column_stack([rng.permutation(c) for c in X.T])) for _ in range(n_perm)])
    sd = null.std(ddof=1)
    return {"stat": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z_vs_null": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            "perm_p": float((1 + (null >= obs).sum()) / (n_perm + 1)), "n": int(len(X))}


def cluster_vs_controls(Xm, Xc, n_perm, rng, k=K_CLUSTERS, seed=0) -> dict:
    """Silhouette of the move states against silhouettes of random same-size samples of ordinary days."""
    def stat(X):
        return silhouette(X, cluster(X, k, seed))
    obs = stat(Xm)
    n = len(Xm)
    null = np.array([stat(Xc[rng.choice(len(Xc), n, replace=False)]) for _ in range(n_perm)])
    sd = null.std(ddof=1)
    return {"stat": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z_vs_null": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
            "perm_p": float((1 + (null >= obs).sum()) / (n_perm + 1)), "n": int(n)}


def run(tables: dict[str, pd.DataFrame], n_perm: int = 500, seed: int = 11, n_stubs: int = 5,
        stub_perm: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    moves, ctrls = tables["moves"], tables["controls"]
    assert pd.to_datetime(moves["date"]).max() < pd.Timestamp("2020-01-01"), "test split rows in the study input"
    trials = {}

    both = pd.concat([moves.assign(y=1), ctrls.assign(y=0)], ignore_index=True)
    X1, d1 = design(both, CORE)
    trials["T1 move vs ordinary day"] = classification_trial(X1, d1["y"].to_numpy(), d1["date"], n_perm, rng)

    X2, d2 = design(moves, CORE)
    trials["T2 direction of the move"] = classification_trial(X2, (d2["day_direction"] > 0).to_numpy(), d2["date"], n_perm, rng)

    us = moves[moves.get("bbds_match", pd.Series(False, index=moves.index)).fillna(False).astype(bool)]
    us = us.sort_values("market", key=lambda s: s.map({"SPX": 0, "NDX": 1})).drop_duplicates("date")
    X3, d3 = design(us, CORE)
    trials["T3 policy-coded cause (BBDS)"] = classification_trial(X3, (d3["bbds_policy"] >= 0.5).to_numpy(), d3["date"],
                                                                  n_perm, rng)

    trials["T4 clusters among pre-move states"] = cluster_trial(X2, n_perm, rng, seed=seed)

    first = moves[moves["episode_rank"].notna()].sort_values("date").groupby(["market", "episode"]).head(1)
    X5, d5 = design(pd.concat([first.assign(y=1), ctrls.assign(y=0)], ignore_index=True), CORE)
    trials["T5 T1 on first move of each episode"] = classification_trial(X5, d5["y"].to_numpy(), d5["date"], n_perm, rng)

    lab = tables.get("labelled", pd.DataFrame())
    if len(lab):
        X6, d6 = design(lab, CORE)
        trials["T6 policy-coded cause, all BBDS days"] = classification_trial(X6, (d6["bbds_policy"] >= 0.5).to_numpy(),
                                                                              d6["date"], n_perm, rng)

    Xc, _ = design(ctrls, CORE)
    trials["T7 moves more clustered than ordinary days"] = cluster_vs_controls(X2, Xc, n_perm, rng, seed=seed)

    adj = bh([t["perm_p"] for t in trials.values()])
    for (name, t), a in zip(trials.items(), adj):
        t["bh_p"] = float(a)
        t["verdict"] = "PASS" if (t["z_vs_null"] >= BAR["z_vs_null"] and a <= BAR["bh_p"]) else "FAIL"

    # null stubs: coin-flip labels with T1's base rate through the T1 pipeline must all fail the bar
    stubs = []
    base = d1["y"].mean()
    for s in range(n_stubs):
        r = np.random.default_rng(1000 + s)
        yy = (r.random(len(d1)) < base).astype(float)
        st = classification_trial(X1, yy, d1["date"], stub_perm, r)
        st["passes_bar"] = bool(st["z_vs_null"] >= BAR["z_vs_null"] and st["perm_p"] * len(trials) <= BAR["bh_p"])
        stubs.append(st)
    gate = "OK" if not any(s["passes_bar"] for s in stubs) else "REFUSED"

    # description only (not trials): cluster profiles and what the newspapers said about each cluster's moves
    lab = cluster(X2, K_CLUSTERS, seed)
    d2 = d2.assign(cluster=lab)
    prof = pd.DataFrame(X2, columns=[c for c in CORE if c in moves]).assign(cluster=lab).groupby("cluster").median().T
    mix = d2.groupby("cluster").agg(n=("day_ret", "size"), up_share=("day_direction", lambda s: (s > 0).mean()),
                                    med_ret=("day_ret", "median"))
    if "bbds_primary" in d2:
        top = d2.dropna(subset=["bbds_primary"]).groupby("cluster")["bbds_primary"].agg(
            lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().head(3).items()))
        mix = mix.join(top.rename("bbds_top_causes"))
    coefs = pd.Series(fit_logit(X1, d1["y"].to_numpy(float))[1:], index=[c for c in CORE if c in both])
    vol_only = auc(d1["y"].to_numpy(), d1["pre_rv_20d"].fillna(d1["pre_rv_20d"].median()).to_numpy())
    return {"trials": trials, "stubs": stubs, "gate": gate, "n_trials": len(trials), "bar": BAR,
            "profiles": prof, "mix": mix, "t1_coefs": coefs.sort_values(), "vol_only_auc_T1": vol_only, "n_perm": n_perm, "seed": seed,
            "markets": moves.groupby("market").size().to_dict()}


def report(res: dict) -> str:
    rows = [{"trial": k, "n": v["n"], "stat": round(v["stat"], 3), "null": f"{v['null_mean']:.3f}±{v['null_sd']:.3f}",
             "z": round(v["z_vs_null"], 2), "perm_p": round(v["perm_p"], 4), "bh_p": round(v["bh_p"], 4),
             "verdict": v["verdict"]} for k, v in res["trials"].items()]
    lines = [f"Null-stub gate: {res['gate']} ({sum(s['passes_bar'] for s in res['stubs'])}/{len(res['stubs'])} coin-flip "
             f"stubs clear the bar; stub z: {', '.join(f'{s['z_vs_null']:.2f}' for s in res['stubs'])})",
             f"Bar: z vs permutation null >= {res['bar']['z_vs_null']} and BH p <= {res['bar']['bh_p']}; "
             f"trials {res['n_trials']}; permutations {res['n_perm']} each; seed {res['seed']}",
             "", pd.DataFrame(rows).to_markdown(index=False), "",
             f"Vol-only baseline for T1 (AUC of prior 20-day realised vol alone): {res['vol_only_auc_T1']:.3f}", "",
             "T1 model weights on standardised features (full sample, description):",
             res["t1_coefs"].round(3).to_frame("weight").to_markdown(), "",
             "Cluster mix (description, not a test):", res["mix"].round(4).to_markdown(), "",
             "Cluster profiles, median standardised pre-move features:", res["profiles"].round(2).to_markdown()]
    return "\n".join(lines)
