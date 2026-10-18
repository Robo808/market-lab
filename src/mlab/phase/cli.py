"""`mlab phase ...`: phase-space regime research on frozen datasets (train + validation only).

  cube      build the feature cube, run the schema contract and the look-ahead gate on every symbol
  regimes   fit HMM regimes on the train split, filter train+validation, write the 3D phase-space view
  evaluate  purged walk-forward over a config grid, null stubs, deflated Sharpe, PBO, skill vs horizon
  physics   inertia vs restoring force per regime, entropy buckets, spectral check, Monte Carlo
  log       the experiment log (every configuration ever tried, per dataset)
"""
from __future__ import annotations

import json

import pandas as pd


def _out(name: str):
    from ..config import REPORTS_DIR
    d = REPORTS_DIR / "phase"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _stamp() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")


def _load(a):
    from . import features as F
    return F.load(a.dataset, a.symbols or None, a.split)


def cmd_phase(a):
    from ..cli import show
    from . import evaluate as E
    from . import features as F
    from . import physics as P
    from .regimes import RegimeModel, phase_plot

    if a.action == "log":
        show(E.experiments(a.dataset), a.json, title=f"Phase experiment log ({E.LOG_DIR})")
        return
    if not a.dataset:
        raise SystemExit("usage: mlab phase ACTION DATASET [SYMBOLS ...]")
    frames = _load(a)
    src = next(iter(frames.values())).attrs.get("source", "?")
    print(f"_{len(frames)} instruments from {src}_\n")

    if a.action == "cube":
        cube = F.build(frames)
        bad = F.validate(cube)
        leaks = {s: F.check_point_in_time(F.features, df) for s, df in frames.items()}
        leaks = {s: v for s, v in leaks.items() if v}
        show(cube.describe().T[["count", "mean", "std", "min", "max"]], a.json,
             title=f"Feature cube {cube.shape[0]} rows x {cube.shape[1]} features")
        print(f"\nschema contract: {'ok' if not bad else 'VIOLATED: ' + '; '.join(bad)}")
        print(f"look-ahead gate: {'ok on every symbol' if not leaks else 'LEAK: ' + json.dumps(leaks)}")
        if bad or leaks:
            raise SystemExit(1)
        return

    if a.action == "regimes":
        cube = F.build(frames)
        t = cube.index.get_level_values("time")
        from ..datasets import manifest
        train_end = pd.Timestamp(manifest(a.dataset)["splits"]["train"][1], tz="UTC") + pd.Timedelta(days=1)
        rm = RegimeModel(k=a.k, seed=a.seed).fit(cube[t < train_end])
        probs = rm.probs(cube)
        prof = rm.profile.copy()
        show(prof, a.json, title=f"Regimes (HMM k={a.k}, fitted on bars before {train_end.date()})")
        show(pd.DataFrame(rm.hmm.A, index=rm.names, columns=rm.names), a.json, title="Transition matrix")
        last = probs.groupby(level="symbol").tail(1).reset_index()
        last["regime"] = [rm.names[r] for r in last["regime"]]
        last["p_max"] = probs.groupby(level="symbol").tail(1)[[f"p{j}" for j in range(a.k)]].max(axis=1).to_numpy()
        show(last[["symbol", "time", "regime", "p_max"]], a.json, title="Filtered regime at the last bar of the split")
        path = phase_plot(cube, probs, rm.names, _out(f"phase_space_{a.dataset}_{_stamp()}.html"), every=a.every,
                          title=f"Phase space: {a.dataset} {a.split}, HMM k={a.k} fitted to {train_end.date()}")
        print(f"\n3D phase-space view: {path}")
        return

    fam = a.family or a.dataset.split("@")[0]
    S = E.prepare(frames, fam, context={"dataset": src, "split": a.split, "symbols": sorted(frames)},
                  min_train_years=a.min_train, block=a.block)
    print(f"walk-forward test folds start {', '.join(map(str, S.years))} ({a.block}y each), one-side costs (bps): "
          + ", ".join(f"{k} {v:.1f}" for k, v in S.costs.items()) + f"; funding {a.funding:.1%}/yr on gross\n")

    if a.action == "evaluate":
        grid = json.loads(a.grid) if a.grid and a.grid.startswith("[") else E.default_grid(quick=a.grid == "quick")
        res = E.evaluate(S, grid, a.embargo, a.delay, a.funding,
                         bars={"t_min": a.t_min, "dsr_min": 0.95, "pbo_max": 0.5})
        tbl = res["table"].drop(columns="id")
        show(tbl, a.json, title=f"Walk-forward, out of sample {S.years[0]}-{S.years[-1]} (after costs)")
        show(res["stubs"], a.json, title=f"Null stubs through the same bars (gate: {res['gate']})")
        show(res["checks"], a.json, title=f"Best config {json.dumps(res['best'])}: {res['verdict']}")
        print(f"\nPBO {res['pbo']['pbo']:.2f} over {res['pbo']['splits']} CSCV splits · trials in log for "
              f"'{fam}': {res['trials']} · Sharpe by year: "
              + ", ".join(f"{y} {v:+.2f}" for y, v in res["by_year"].dropna().items())
              + f" · decay slope {res['decay_slope']:+.3f}/yr")
        if a.skill:
            tblh = E.skill_by_horizon(S, res["cache"], k=a.k)
            show(tblh, a.json, title="Forecast skill vs horizon (out-of-sample rank IC)")
            p = E.skill_plot(tblh, _out(f"skill_horizon_{fam}_{_stamp()}.html"), f"Skill vs horizon: {fam}")
            print(f"\nchart: {p}")
        return

    if a.action == "physics":
        cache = E.FoldCache(S, a.embargo)
        # regimes for the forces table: fitted on the first fold's training data, filtered forward
        rm, probs = cache.regimes(S.years[0], a.k)
        t = probs.index.get_level_values("time")
        oos = probs[t >= pd.Timestamp(f"{S.years[0]}-01-01", tz="UTC")]
        forces = P.forces_by_regime(frames, S.cube, oos["regime"], rm.names)
        show(forces, a.json, title=f"Inertia vs restoring force by regime, {S.years[0]}-{S.years[-1]} "
                                   f"(regimes fitted before {S.years[0]})")
        cfg = json.loads(a.config) if a.config else {"model": "ridge", "horizon": 21}
        res = E.run_config(S, cache, cfg, a.delay, a.funding)
        y = E.forward_target(S, cfg["horizon"])
        ent = P.entropy_skill(res["pred"], y, S.cube["pe_63"], pd.Timestamp(f"{S.years[0]}-01-01", tz="UTC"))
        show(ent, a.json, title=f"Skill by permutation-entropy bucket ({json.dumps(cfg)})")
        show(P.spectral_check(frames), a.json, title="Spectral check: Fisher's g on raw log prices vs returns")
        st = P.stress(res["returns"], a.paths)
        show(pd.DataFrame({k: v for k, v in st.items() if isinstance(v, list)}, index=["p5", "p50", "p95"]).T,
             a.json, title=f"Monte Carlo: {st['paths']} block-bootstrapped histories of the OOS returns "
                           f"(P(Sharpe <= 0) = {st['p_sharpe_le_0']:.2f})")
        if a.worlds:
            nw = P.null_world(frames, cfg, a.worlds, embargo=a.embargo, delay=a.delay, funding=a.funding)
            real = E.summarize(res["returns"], lags=max(5, cfg["horizon"]))
            show(nw.describe().T, a.json, title=f"Same pipeline on {a.worlds} synthetic markets (no structure "
                                                f"beyond 21-bar blocks)")
            print(f"\nreal Sharpe {real['sharpe']:+.2f} vs synthetic worlds: share at or above it "
                  f"{(nw['sharpe'] >= real['sharpe']).mean():.2f}")
        return


def register(add):
    q = add("phase", cmd_phase, "phase-space regime research: cube | regimes | evaluate | physics | log (frozen datasets)")
    q.add_argument("action", choices=["cube", "regimes", "evaluate", "physics", "log"])
    q.add_argument("dataset", nargs="?", help="frozen dataset NAME[@VERSION], e.g. duka-1d (mlab datasets list)")
    q.add_argument("symbols", nargs="*")
    q.add_argument("--split", default="train+validation", help="train | validation | train+validation (test stays locked)")
    q.add_argument("--k", type=int, default=4, help="number of HMM regimes")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--every", type=int, default=3, help="regimes: plot every Nth bar")
    q.add_argument("--grid", default="full", help="evaluate: full | quick | JSON list of configs")
    q.add_argument("--embargo", type=int, default=5, help="bars between the last label and the test start")
    q.add_argument("--delay", type=int, default=0, help="extra bars of execution latency after the decision close")
    q.add_argument("--funding", type=float, default=0.02, help="annual funding on gross exposure")
    q.add_argument("--block", type=int, default=1, help="years per walk-forward test fold")
    q.add_argument("--min-train", type=int, default=5, help="years of history before the first test fold")
    q.add_argument("--t-min", type=float, default=3.0, help="pass bar on the Newey-West t-stat")
    q.add_argument("--skill", action="store_true", help="evaluate: also chart skill vs horizon")
    q.add_argument("--family", help="experiment-log family (default: dataset name)")
    q.add_argument("--config", help='physics: JSON config, default {"model":"ridge","horizon":21}')
    q.add_argument("--paths", type=int, default=2000, help="physics: Monte Carlo paths")
    q.add_argument("--worlds", type=int, default=0, help="physics: synthetic markets through the full pipeline")
