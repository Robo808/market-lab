"""`mlab algo ...` (strategy library: list, backtest, validate, signal, paper) and `mlab hypo ...`
(pre-registered hypothesis tests). Backtest, signal and paper modes only: no live execution."""
from __future__ import annotations

import json

import pandas as pd


def _frames(a):
    if getattr(a, "dataset", None):  # frozen offline snapshot: no network, fixed split
        from ..datasets import load
        return load(a.dataset, a.symbols or None, a.split, allow_test=a.allow_test, note=f"algo {a.action} {a.strategy}")
    from ..data import get_prices
    return {s: get_prices(s, a.interval, a.start, None, a.period, refresh=getattr(a, "refresh", False)) for s in a.symbols}


def _src(frames):
    for s, df in frames.items():
        print(f"_{s}: {df.attrs.get('source', '?')} · fetched {df.attrs.get('fetched_at', '?')} · "
              f"last bar {df.attrs.get('last_bar', str(df.index[-1])[:10])}_")


def _overlays(a):
    ov = {}
    if getattr(a, "vol_target", None):
        ov["vol_target"] = {"target": a.vol_target}
    if getattr(a, "regime_filter", False):
        ov["regime_filter"] = {}
    return ov


def cmd_algo(a):
    from ..cli import show
    from . import engine, paper, strategies, validation
    act = a.action
    if act == "list":
        show(strategies.table(), a.json, title=f"Strategy library ({len(strategies.REGISTRY)})")
        print("\nOverlays: " + ", ".join(strategies.OVERLAYS) + " (--vol-target 0.15, --regime-filter)")
        return
    if act == "paper-status":
        show(paper.status(a.book), a.json, title=f"Paper book {a.book}" if a.book else "Paper books")
        return
    if not a.strategy or not a.symbols:
        raise SystemExit(f"usage: mlab algo {act} STRATEGY SYMBOL [SYMBOL ...]")
    params = json.loads(a.params) if a.params else {}
    frames = _frames(a)
    kw = {"spread_bps": a.spread_bps, "funding_annual": a.funding}
    if act == "backtest":
        res = engine.backtest(a.strategy, frames, params, _overlays(a), **kw)
        show(pd.DataFrame({"strategy": res["stats"], "buy_hold_ew": res["benchmark"]}), a.json,
             title=f"{a.strategy} on {', '.join(frames)}")
        print(f"\ntrades {res['trades']} · exposure {res['exposure']:.0%} · turnover/yr {res['turnover_ann']:.1f}x · "
              f"cost drag {res['cost_drag']:.2%}")
    elif act == "validate":
        rep = validation.report(a.strategy, frames, params, _overlays(a), trials=a.trials, oos_start=a.oos_start,
                                n_perm=a.n_perm, n_boot=a.n_perm, **kw)
        flat = {k: v for k, v in rep.items() if k not in ("result", "split", "bootstrap", "permutation", "benchmark", "params")}
        flat.update({"is_sharpe": rep["split"]["in_sample"]["sharpe"], "oos_sharpe": rep["split"]["out_of_sample"]["sharpe"],
                     "oos_start": rep["split"]["oos_start"], "boot_ci_low": rep["bootstrap"]["ci_low"],
                     "boot_ci_high": rep["bootstrap"]["ci_high"], "perm_p": rep["permutation"]["perm_p"]})
        show(flat, a.json, title=f"Validation: {a.strategy} on {', '.join(frames)}")
        if a.grid:
            grid = json.loads(a.grid)
            show(validation.sensitivity(a.strategy, frames, grid, **kw), a.json, title="Parameter sensitivity")
            wf = validation.walk_forward(a.strategy, frames, grid, **kw)
            print(f"\nWalk-forward out-of-sample Sharpe: {wf['oos_sharpe']:.2f}")
            show(wf["picks"], a.json, title="Walk-forward picks")
    elif act == "signal":
        show(engine.latest_signal(a.strategy, frames, params, _overlays(a)), a.json, title=f"{a.strategy}: target now")
    elif act == "paper":
        if not a.book:
            raise SystemExit("--book NAME is required for paper")
        show(paper.rebalance(a.book, a.strategy, frames, params, _overlays(a), a.capital, a.spread_bps or 5.0),
             a.json, title=f"Paper book {a.book}")
    _src(frames)


def cmd_hypo(a):
    from ..cli import show
    from . import hypothesis
    if a.action == "new":
        spec = json.loads(a.spec) if a.spec else {}
        if a.strategy:
            spec["strategy"] = a.strategy
        if a.symbols:
            spec["symbols"] = a.symbols
        spec.setdefault("trials", max(1, hypothesis.trial_count() + 1))
        p = hypothesis.new(a.slug, a.claim or a.slug, spec, branch=a.branch)
        print(f"registered {p}\nEdit Mechanism and the spec now, commit, THEN run: bash mlab hypo test {p}")
    elif a.action == "test":
        res = hypothesis.test(a.slug, force=a.force, n_perm=a.n_perm, n_boot=a.n_perm)
        show(res["checks"], a.json, title=f"Verdict: {res['verdict']}")
        print(f"\nresults appended to {res['path']}")
    elif a.action == "list":
        show(hypothesis.listing(), a.json, title=f"Hypotheses (graded trials so far: {hypothesis.trial_count()})")


def register(add):
    q = add("algo", cmd_algo, "algorithmic strategies: list | backtest | validate | signal | paper | paper-status")
    q.add_argument("action", choices=["list", "backtest", "validate", "signal", "paper", "paper-status"])
    q.add_argument("strategy", nargs="?"); q.add_argument("symbols", nargs="*")
    q.add_argument("--params", help='JSON, e.g. \'{"lookback":126}\'')
    q.add_argument("--spread-bps", type=float, default=2.0, help="cost per unit turnover in basis points")
    q.add_argument("--funding", type=float, default=0.0, help="annual overnight funding on gross exposure")
    q.add_argument("--vol-target", type=float); q.add_argument("--regime-filter", action="store_true")
    q.add_argument("--grid", help='validate: JSON grid, e.g. \'{"lookback":[126,252],"skip":[0,21]}\'')
    q.add_argument("--trials", type=int, default=1, help="variants tried so far (deflated Sharpe)")
    q.add_argument("--oos-start"); q.add_argument("--n-perm", type=int, default=500)
    q.add_argument("--book"); q.add_argument("--capital", type=float, default=10_000.0)
    q.add_argument("--interval", "-i", default="1d"); q.add_argument("--period", "-p", default="max")
    q.add_argument("--start"); q.add_argument("--refresh", action="store_true")
    q.add_argument("--dataset", help="run offline on a frozen snapshot, NAME[@VERSION] (mlab datasets list)")
    q.add_argument("--split", default="train", help="train | validation | test | train+validation (with --dataset)")
    q.add_argument("--allow-test", action="store_true", help="unlock the held-out test split (logged)")

    q = add("hypo", cmd_hypo, "pre-registered hypotheses: new SLUG --claim .. | test FILE | list")
    q.add_argument("action", choices=["new", "test", "list"]); q.add_argument("slug", nargs="?", help="slug (new) or file (test)")
    q.add_argument("--claim"); q.add_argument("--strategy"); q.add_argument("--symbols", nargs="*")
    q.add_argument("--spec", help="JSON overrides for the spec block"); q.add_argument("--branch", action="store_true",
                                                                                     help="also create the hypo/ git branch")
    q.add_argument("--force", action="store_true"); q.add_argument("--n-perm", type=int, default=500)
