"""Honest evaluator for phase-space models.

- Purged, embargoed, anchored walk-forward (López de Prado, Advances in Financial Machine Learning, 2018,
  ch. 7): each test year is predicted by models fitted only on bars whose label window ends before the test
  starts (purge = horizon h) plus an embargo gap. Scalers, HMMs and regressions are all refitted per fold.
- Decisions at bar t's close earn from t+1+delay on (`delay` models execution latency); positions are the
  average of the last h signals (overlapping cohorts, Jegadeesh & Titman 1993), vol-targeted per instrument.
- Costs: half the quoted spread plus slippage on every unit of turnover (from the snapshot's own spread
  column where it has one), and overnight funding on gross exposure.
- Statistics: Newey-West t-stat, probabilistic and deflated Sharpe (Bailey & López de Prado 2012, 2014) with
  the trial count from the experiment log, probability of backtest overfitting by CSCV (Bailey, Borwein,
  López de Prado & Zhu 2017), out-of-sample IC by horizon, Sharpe by year (edge decay).
- Null stubs (always long, seeded coin flips matched to the candidate's exposure and holding period) go
  through the same costs and bars. If a stub clears the bars, the bars are refused.
- Every configuration is written once to the experiment log (`$MLAB_DATA_DIR/research/phase/experiments/`),
  failures included, so the deflated Sharpe knows how many things were tried.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess  # nosec B404 - fixed git arguments only
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import DATA_DIR, ROOT
from ..quant import validation
from . import features as F
from .regimes import RegimeModel

LOG_DIR = DATA_DIR / "research" / "phase" / "experiments"
RIDGE_FEATURES = [c for c in F.FEATURES if c not in ("vol_z",)]   # volume is missing on FX and CFD quotes

# one-side cost in bps where the snapshot has no spread column (Yahoo): conservative retail CFD/spread-bet levels
CLASS_COST_BPS = {"index": 1.5, "fx": 1.0, "commodity": 3.0, "etf": 2.0, "crypto": 8.0, "stock": 3.0, "rate": 3.0}


def asset_class(sym: str) -> str:
    s = sym.upper()
    if s.endswith("=X") or (len(s) == 6 and s[:3] in {"EUR", "GBP", "USD", "AUD", "NZD", "CAD", "CHF", "JPY"}):
        return "fx"
    if s.endswith("=F") or s in {"XAUUSD", "XAGUSD", "BRENT", "WTI", "GLD", "SGLN.L"}:
        return "commodity"
    if s.startswith("^") or s.startswith("_") or s in {"UK100", "US500", "USTEC", "DE40", "EU50", "JP225", "US30"}:
        return "rate" if s in {"^TNX", "^IRX", "_TNX", "_IRX"} else "index"
    if "BTC" in s or "ETH" in s:
        return "crypto"
    if s in {"SPY", "QQQ", "TLT", "IEF", "SHY", "HYG", "LQD", "IWM", "DIA"} or s.endswith(".L") or s.startswith("XL"):
        return "etf"
    return "stock"


def cost_bps(frames: dict[str, pd.DataFrame], slippage_bps: float = 0.5) -> pd.Series:
    """One-side cost per unit of turnover, per symbol: half the median quoted spread + slippage, floored at the
    asset-class default (which is also used when the snapshot has no spread)."""
    out = {}
    for s, df in frames.items():
        if "spread" in df and df["spread"].notna().any():
            quoted = float((df["spread"] / df["close"]).median() * 1e4 / 2 + slippage_bps)
            out[s] = max(quoted, CLASS_COST_BPS[asset_class(s)])   # free feeds quote some CFDs near zero spread
        else:
            out[s] = CLASS_COST_BPS[asset_class(s)]
    return pd.Series(out)


# ---- statistics ---------------------------------------------------------------------------------------
def nw_tstat(r: pd.Series, lags: int = 5) -> float:
    """Newey-West (Bartlett) t-stat of the mean: overlapping positions make daily returns autocorrelated."""
    x = pd.Series(r).dropna().to_numpy()
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def pbo_cscv(M: pd.DataFrame, blocks: int = 16) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation. M = T x N returns
    of N configurations. PBO = share of IS/OOS splits where the IS-best config ranks below the OOS median."""
    M = M.dropna(how="all").fillna(0.0)
    T, N = M.shape
    if N < 2 or T < blocks * 5:
        return {"pbo": float("nan"), "splits": 0}
    edges = np.linspace(0, T, blocks + 1).astype(int)
    X = M.to_numpy()
    s1 = np.stack([X[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    s2 = np.stack([(X[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges)
    logits = []
    for ins in itertools.combinations(range(blocks), blocks // 2):
        m = np.zeros(blocks, bool)
        m[list(ins)] = True

        def sr(mask):
            n = cnt[mask].sum()
            mu = s1[mask].sum(0) / n
            sd = np.sqrt(np.maximum(s2[mask].sum(0) / n - mu ** 2, 1e-18))
            return mu / sd
        best = int(np.argmax(sr(m)))
        oos = sr(~m)
        w = (oos < oos[best]).sum() / (N - 1)          # relative rank of the IS winner out of sample
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    lg = np.array(logits)
    return {"pbo": float((lg <= 0).mean()), "splits": len(lg), "median_logit": float(np.median(lg))}


def summarize(r: pd.Series, trials: int = 1, sr_var: float | None = None, lags: int = 5, periods: int = 252) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"sharpe": validation.sharpe(r, periods), "t_nw": nw_tstat(r, lags),
            "ann_return": float(r.mean() * periods), "ann_vol": float(r.std() * np.sqrt(periods)),
            "max_dd": float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan"),
            "psr": validation.psr(r, 0.0, periods), "dsr": validation.dsr(r, trials, sr_var, periods),
            "bars": len(r)}


# ---- experiment log -------------------------------------------------------------------------------------
def _git() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,  # nosec
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def config_id(family: str, cfg: dict) -> str:
    return hashlib.sha256(json.dumps({"family": family, **cfg}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def log_experiment(family: str, cfg: dict, metrics: dict, context: dict) -> str:
    """One record per (family, config): re-running the identical config is the same trial, so the count does
    not grow, but the record keeps when it was first tried, how often it ran, and the latest metrics."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cid = config_id(family, cfg)
    p = LOG_DIR / f"{family}.{cid}.json"
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    first, runs = now, 0
    if p.exists():
        try:
            old = json.loads(p.read_text())
            first, runs = old.get("first_at", old.get("at", now)), int(old.get("runs", 1))
        except (OSError, json.JSONDecodeError):
            pass
    rec = {"id": cid, "family": family, "first_at": first, "at": now, "runs": runs + 1, "git": _git(),
           "config": cfg, "metrics": metrics, **context}
    tmp = p.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp")
    tmp.write_text(json.dumps(rec, default=float, indent=1))
    tmp.replace(p)
    return cid


def experiments(family: str | None = None) -> pd.DataFrame:
    rows = []
    for p in sorted(LOG_DIR.glob("*.json")) if LOG_DIR.exists() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if family and r.get("family") != family:
            continue
        rows.append({"id": r["id"], "family": r["family"], "at": r["at"], **r["config"],
                     **{k: r["metrics"].get(k) for k in ("sharpe", "t_nw", "dsr", "ic")}})
    return pd.DataFrame(rows)


def trial_count(family: str) -> int:
    return max(1, len(list(LOG_DIR.glob(f"{family}.*.json")))) if LOG_DIR.exists() else 1


# ---- walk-forward ---------------------------------------------------------------------------------------
@dataclass
class Setup:
    frames: dict
    cube: pd.DataFrame
    closes: pd.DataFrame
    rv: pd.DataFrame
    costs: pd.Series
    years: list[int]
    family: str
    context: dict = field(default_factory=dict)
    block: int = 1                 # years per test fold


def prepare(frames: dict[str, pd.DataFrame], family: str, min_train_years: int = 5, context: dict | None = None,
            block: int = 1) -> Setup:
    cube = F.build(frames)
    closes = pd.DataFrame({s: df["close"] for s, df in frames.items()}).sort_index()
    rv = cube["rv_21"].unstack("symbol").reindex(closes.index)
    t = cube.index.get_level_values("time")
    y0, y1 = t.min().year, t.max().year
    years = list(range(y0 + min_train_years, y1 + 1, block))
    return Setup(frames, cube, closes, rv, cost_bps(frames), years, family, context or {}, block)


def forward_target(S: Setup, h: int) -> pd.Series:
    """Vol-normalised forward log return over h bars, aligned to the cube's (symbol, time) rows."""
    out = []
    for s, df in S.frames.items():
        lc = np.log(df["close"])
        y = (lc.shift(-h) - lc)
        out.append(pd.concat({s: y}, names=["symbol"]))
    y = pd.concat(out).reindex(S.cube.index)
    y.index.names = ["symbol", "time"]
    return y / (S.cube["rv_21"] * np.sqrt(h))


def train_mask(index: pd.MultiIndex, test_start: pd.Timestamp, gap: int) -> np.ndarray:
    """Rows usable for a label-based fit: each symbol's bars at least `gap` bars before its first test bar."""
    t = index.get_level_values("time")
    sym = index.get_level_values("symbol")
    df = pd.DataFrame({"sym": sym, "pre": t < test_start})
    pos = df.groupby("sym").cumcount().to_numpy()
    first = df.groupby("sym")["pre"].transform("sum").to_numpy()   # position of first test bar
    return (pos < first - gap) & df["pre"].to_numpy()


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    P = alpha * np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + P, Xb.T @ y)


class FoldCache:
    """Regime models are label-free, so one fit per (fold, k) serves every horizon and model."""
    def __init__(self, S: Setup, embargo: int, log=print):
        self.S, self.embargo, self.log, self.store = S, embargo, log, {}

    def regimes(self, year: int, k: int) -> tuple[RegimeModel, pd.DataFrame]:
        key = (year, k)
        if key not in self.store:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            m = train_mask(self.S.cube.index, start, self.embargo)
            rm = RegimeModel(k=k, seed=year).fit(self.S.cube[m])
            self.store[key] = (rm, rm.probs(self.S.cube))
            self.log(f"  fold {year} k={k}: HMM fitted on {int(m.sum())} rows; regimes {', '.join(rm.names)}")
        return self.store[key]


def predict_fold(S: Setup, cache: FoldCache, cfg: dict, year: int, y: pd.Series) -> pd.Series:
    """Out-of-sample predictions for one test year from models fitted on purged earlier data."""
    h, emb = cfg["horizon"], cache.embargo
    start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + S.block}-01-01", tz="UTC")
    idx = S.cube.index
    t = idx.get_level_values("time")
    test = (t >= start) & (t < end)
    tr = train_mask(idx, start, h + emb) & y.notna().to_numpy()
    model = cfg["model"]
    if model == "tsmom":                       # literature benchmark: sign of the trailing 63-bar return
        return np.sign(S.cube.loc[test, "ret_63"])
    if model in ("regime", "regime_ridge"):
        _, probs = cache.regimes(year, cfg["k"])
        P = probs[[f"p{j}" for j in range(cfg["k"])]].reindex(idx)
    if model == "regime":
        ok = tr & P.notna().all(axis=1).to_numpy()
        w = P[ok].to_numpy()
        m = (w * y[ok].to_numpy()[:, None]).sum(0) / w.sum(0)     # regime-conditional mean, probability-weighted
        return pd.Series(P[test].to_numpy() @ m, index=idx[test])
    X = S.cube[RIDGE_FEATURES].copy()
    X["rv_21"] = np.log(X["rv_21"].clip(lower=1e-6))
    if model == "regime_ridge":
        X = X.join(P.iloc[:, 1:])              # drop one probability column (they sum to 1)
    ok = tr & X.notna().all(axis=1).to_numpy()
    mu, sd = X[ok].mean(), X[ok].std().replace(0, 1.0)
    Z = ((X - mu) / sd).clip(-5, 5)
    beta = _ridge(Z[ok].to_numpy(), y[ok].clip(-5, 5).to_numpy(), alpha=cfg.get("alpha", 100.0))
    Zt = Z[test]
    pred = pd.Series(np.nan, index=idx[test])
    good = Zt.notna().all(axis=1).to_numpy()
    pred[good] = beta[0] + Zt[good].to_numpy() @ beta[1:]
    return pred


def positions(S: Setup, pred: pd.Series, h: int, long_only: bool, target_vol: float = 0.10,
              cap: float = 3.0) -> pd.DataFrame:
    """Signal -> weights: sign of the forecast, averaged over the last h signals, vol-targeted, 1/N of capital."""
    sig = np.sign(pred).unstack("symbol")
    if long_only:
        sig = sig.clip(lower=0)
    sig = sig.rolling(h, min_periods=1).mean()              # over the symbol's own bars
    lev = (target_vol / (S.rv * np.sqrt(252))).clip(upper=cap)
    # on the union calendar a symbol without a bar that day (its market shut) keeps yesterday's position
    w = (sig.reindex(S.closes.index) * lev).ffill(limit=5)
    return (w / len(S.frames)).fillna(0.0)


def simulate(S: Setup, w: pd.DataFrame, delay: int = 0, funding_annual: float = 0.02, costs: bool = True) -> pd.Series:
    rets = S.closes.pct_change(fill_method=None).fillna(0.0).reindex(w.index).fillna(0.0)
    held = w.shift(1 + delay).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turn = w.diff().abs().fillna(w.abs())
    cost = (turn.shift(1 + delay).fillna(0.0) * S.costs.reindex(w.columns).fillna(3.0) / 1e4).sum(axis=1) * costs
    fund = held.abs().sum(axis=1) * funding_annual / 252
    return gross - cost - fund


def run_config(S: Setup, cache: FoldCache, cfg: dict, delay: int = 0, funding: float = 0.02) -> dict:
    y = forward_target(S, cfg["horizon"])
    preds = pd.concat([predict_fold(S, cache, cfg, yr, y) for yr in S.years]).sort_index()
    w = positions(S, preds, cfg["horizon"], cfg.get("long_only", False))
    first = pd.Timestamp(f"{S.years[0]}-01-01", tz="UTC")
    w = w[w.index >= first]
    r = simulate(S, w, delay, funding)
    yy = y.reindex(preds.index)
    ok = preds.notna() & yy.notna()
    ic = float(spearmanr(preds[ok], yy[ok]).statistic) if ok.sum() > 100 else float("nan")
    hit = float((np.sign(preds[ok]) == np.sign(yy[ok])).mean()) if ok.sum() else float("nan")
    on = w.abs().gt(1e-12)
    entries = int((on & ~on.shift(fill_value=False)).sum().sum())
    pre = validation.sharpe(simulate(S, w, delay, 0.0, costs=False))
    spread_only = validation.sharpe(simulate(S, w, delay, 0.0))     # futures/cash venues carry no funding line
    return {"returns": r, "weights": w, "pred": preds, "ic": ic, "hit": hit, "sharpe_pre_cost": pre,
            "sharpe_no_funding": spread_only,
            "exposure": float(on.to_numpy().mean()), "hold": float(on.to_numpy().sum() / max(entries, 1)),
            "turnover_ann": float(w.diff().abs().sum(axis=1).mean() * 252)}


def default_grid(quick: bool = False) -> list[dict]:
    grid = []
    hs = [21] if quick else [5, 21]
    ks = [4] if quick else [3, 4]
    for h in hs:
        for lo in (False, True):
            grid.append({"model": "tsmom", "horizon": h, "long_only": lo})
            grid.append({"model": "ridge", "horizon": h, "long_only": lo})
            for k in ks:
                grid.append({"model": "regime", "horizon": h, "k": k, "long_only": lo})
                grid.append({"model": "regime_ridge", "horizon": h, "k": k, "long_only": lo})
    return grid


def null_stubs(S: Setup, shape: dict, seeds: int = 5, delay: int = 0, funding: float = 0.02) -> dict[str, pd.Series]:
    """Always long and seeded coin flips with the candidate's exposure and average hold, same sizing and costs."""
    first = pd.Timestamp(f"{S.years[0]}-01-01", tz="UTC")
    idx = S.closes.index[S.closes.index >= first]
    lev = (0.10 / (S.rv * np.sqrt(252))).clip(upper=3.0).reindex(idx).ffill(limit=5) / len(S.frames)
    have = lev.notna()
    out = {"always_long": simulate(S, (lev * have).fillna(0.0), delay, funding)}
    hold = max(1, int(round(shape["hold"])))
    p = min(max(shape["exposure"], 0.05), 1.0)
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        n = len(idx) // hold + 1
        side = rng.choice([-1.0, 1.0], size=(n, len(S.frames))) * (rng.random((n, len(S.frames))) < p)
        sig = pd.DataFrame(np.repeat(side, hold, axis=0)[:len(idx)], index=idx, columns=list(S.frames))
        if shape.get("long_only"):
            sig = sig.abs()
        out[f"coin_flip seed={sd}"] = simulate(S, (sig * lev * have).fillna(0.0), delay, funding)
    return out


def bars_pass(m: dict, bars: dict) -> bool:
    return bool(m["t_nw"] >= bars["t_min"] and m["dsr"] >= bars["dsr_min"])


def evaluate(S: Setup, grid: list[dict], embargo: int = 5, delay: int = 0, funding: float = 0.02,
             bars: dict | None = None, log=print) -> dict:
    """Run every config through the purged walk-forward, log each, then grade the best against null stubs."""
    bars = bars or {"t_min": 3.0, "dsr_min": 0.95, "pbo_max": 0.5}
    cache = FoldCache(S, embargo, log)
    runs = {}
    for cfg in grid:
        res = run_config(S, cache, cfg, delay, funding)
        runs[config_id(S.family, cfg)] = (cfg, res)
        log(f"  {json.dumps(cfg)}: Sharpe {validation.sharpe(res['returns']):.2f}, IC {res['ic']:+.3f}")
    R = pd.DataFrame({cid: res["returns"] for cid, (cfg, res) in runs.items()})
    per = R.mean() / R.std()
    sr_var = float(per.var()) if len(per) > 1 else None
    # log every config first, so the trial count includes this whole run before anyone is graded
    for cid, (cfg, res) in runs.items():
        m = summarize(res["returns"], 1, None, lags=max(5, cfg["horizon"]))
        m.update({k: res[k] for k in ("ic", "hit", "exposure", "hold", "turnover_ann", "sharpe_pre_cost",
                                               "sharpe_no_funding")})
        log_experiment(S.family, {**cfg, "embargo": embargo, "delay": delay, "funding": funding}, m, S.context)
    trials = trial_count(S.family)
    rows = []
    for cid, (cfg, res) in runs.items():
        m = summarize(res["returns"], trials, sr_var, lags=max(5, cfg["horizon"]))
        rows.append({"id": cid, **cfg, **{k: m[k] for k in ("sharpe", "t_nw", "dsr", "ann_return", "max_dd")},
                     "sharpe_pre_cost": res["sharpe_pre_cost"], "sharpe_no_funding": res["sharpe_no_funding"],
                     "ic": res["ic"], "hit": res["hit"],
                     "turnover": res["turnover_ann"]})
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    best_id = table.iloc[0]["id"]
    bcfg, bres = runs[best_id]
    pbo = pbo_cscv(R)
    shape = {"exposure": bres["exposure"], "hold": bres["hold"], "long_only": bcfg.get("long_only", False)}
    stubs = null_stubs(S, shape, delay=delay, funding=funding)
    srows = []
    for name, r in stubs.items():
        m = summarize(r, trials, sr_var)
        srows.append({"stub": name, **{k: m[k] for k in ("sharpe", "t_nw", "dsr", "ann_return", "max_dd")},
                      "passes_bars": bars_pass(m, bars)})
    stub_tbl = pd.DataFrame(srows)
    gate = "REFUSED" if stub_tbl["passes_bars"].any() else "OK"
    bm = summarize(bres["returns"], trials, sr_var, lags=max(5, bcfg["horizon"]))
    checks = [
        {"bar": "Newey-West t of mean daily return", "need": f">= {bars['t_min']}", "got": round(bm["t_nw"], 2),
         "pass": bm["t_nw"] >= bars["t_min"]},
        {"bar": f"deflated Sharpe probability ({trials} trials)", "need": f">= {bars['dsr_min']}",
         "got": round(bm["dsr"], 3), "pass": bm["dsr"] >= bars["dsr_min"]},
        {"bar": "probability of backtest overfitting (CSCV)", "need": f"< {bars['pbo_max']}",
         "got": round(pbo["pbo"], 3), "pass": pbo["pbo"] < bars["pbo_max"]},
        {"bar": "beats always-long Sharpe", "need": "> stub",
         "got": f"{bm['sharpe']:.2f} vs {stub_tbl.iloc[0]['sharpe']:.2f}",
         "pass": bm["sharpe"] > stub_tbl.iloc[0]["sharpe"]},
    ]
    verdict = "REFUSED (a null stub clears the bars)" if gate == "REFUSED" else (
        "PASS" if all(c["pass"] for c in checks) else "FAIL")
    by_year = bres["returns"].groupby(bres["returns"].index.year).apply(lambda r: validation.sharpe(r))
    return {"table": table, "best": bcfg, "best_id": best_id, "best_returns": bres["returns"], "best_pred": bres["pred"],
            "best_metrics": bm, "pbo": pbo, "stubs": stub_tbl, "gate": gate, "checks": pd.DataFrame(checks),
            "verdict": verdict, "trials": trials, "by_year": by_year, "decay_slope": _theil_sen(by_year),
            "returns": R, "cache": cache}


def _theil_sen(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 3:
        return float("nan")
    x, v = np.asarray(s.index, float), s.to_numpy()
    slopes = [(v[j] - v[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
    return float(np.median(slopes))


def skill_by_horizon(S: Setup, cache: FoldCache, horizons=(1, 2, 5, 10, 21, 42, 63), models=("ridge", "regime", "tsmom"),
                     k: int = 4, block: int = 63, n_boot: int = 300, log=print) -> pd.DataFrame:
    """Out-of-sample rank IC between forecast and realised vol-normalised return, by horizon, with a
    90% moving-block bootstrap interval over time. Chaos limits show up as IC decaying toward 0."""
    rows = []
    for h in horizons:
        y = forward_target(S, h)
        for model in models:
            cfg = {"model": model, "horizon": h, "k": k}
            p = pd.concat([predict_fold(S, cache, cfg, yr, y) for yr in S.years]).sort_index()
            d = pd.DataFrame({"p": p, "y": y.reindex(p.index)}).dropna()
            ic = float(spearmanr(d["p"], d["y"]).statistic)
            # bootstrap over contiguous time blocks
            times = d.index.get_level_values("time").unique().sort_values()
            tb = np.searchsorted(times, d.index.get_level_values("time"))
            nb = len(times) // block
            rng = np.random.default_rng(h)
            groups = [np.flatnonzero(tb // block == b) for b in range(nb + 1)]
            groups = [g for g in groups if len(g)]
            boots = []
            pv, yv = d["p"].to_numpy(), d["y"].to_numpy()
            for _ in range(n_boot):
                pick = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
                boots.append(spearmanr(pv[pick], yv[pick]).statistic)
            lo, hi = np.nanquantile(boots, [0.05, 0.95])
            hit = float((np.sign(d["p"]) == np.sign(d["y"])).mean())
            rows.append({"horizon": h, "model": model, "ic": ic, "ic_lo": float(lo), "ic_hi": float(hi), "hit": hit,
                         "n": len(d)})
            log(f"  h={h:>2} {model:<7} IC {ic:+.4f} [{lo:+.4f}, {hi:+.4f}] hit {hit:.3f}")
    return pd.DataFrame(rows)


def skill_plot(tbl: pd.DataFrame, path, title: str) -> str:
    import plotly.graph_objects as go

    from .regimes import PALETTE
    fig = go.Figure()
    for i, (m, g) in enumerate(tbl.groupby("model", sort=False)):
        c = PALETTE[i]
        fig.add_trace(go.Scatter(x=g["horizon"], y=g["ic"], mode="lines+markers", name=m, line={"color": c, "width": 2},
                                 marker={"size": 8},
                                 error_y={"type": "data", "symmetric": False, "array": g["ic_hi"] - g["ic"],
                                          "arrayminus": g["ic"] - g["ic_lo"], "color": c, "thickness": 1},
                                 hovertemplate=m + " h=%{x}: IC %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line={"color": "#888", "width": 1})
    fig.update_layout(title=title, xaxis={"title": "horizon (bars, log scale)", "type": "log"},
                      yaxis={"title": "out-of-sample rank IC (90% block-bootstrap CI)"}, hovermode="x unified")
    fig.write_html(str(path), include_plotlyjs="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js")
    return str(path)
