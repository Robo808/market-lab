"""`mlab fwd ...`: the forward paper-testing lab. Paper only: nothing here can place an order.

    mlab fwd register            add new matrix cells to research/forward/registry.json (with backtest priors)
    mlab fwd gate                the pass bars must refuse every null stub on trailing backtests
    mlab fwd run                 process every complete bar since the last run; mark option books live
    mlab fwd status              cells, last bars, store size, last run's per-item status
    mlab fwd score [--by KEY]    verdicts (PASS / FAIL / PENDING), context breakdown, desk-filter value
    mlab fwd audit               registry hashes, frozen copies, store schema, ledger chain
"""
from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import time

import numpy as np
import pandas as pd

from ..config import ROOT
from . import SCHEMA_VERSION, context, costs, linear, optsim, score, specs, store

CODE_FILES = ["src/mlab/quant/strategies.py", "src/mlab/quant/engine.py", "src/mlab/ta.py", "src/mlab/options.py"]


def provenance() -> dict:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                             timeout=10).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True, check=False,
                                    text=True, timeout=10).stdout.strip())
    except Exception:
        sha, dirty = "unknown", None
    h = hashlib.sha256()
    for f in sorted(CODE_FILES + [str(p.relative_to(ROOT)) for p in (ROOT / "src/mlab/forward").glob("*.py")]):
        p = ROOT / f
        if p.exists():
            h.update(f.encode() + p.read_bytes())
    return {"code_sha": sha[:12], "code_dirty": dirty, "code_hash": h.hexdigest()[:16]}


def _day(df: pd.DataFrame) -> pd.DataFrame:
    return context._day_index(df)


def load_frames(symbols: list[str], period: str = "5y") -> tuple[dict, dict]:
    from ..data import get_prices
    frames, status = {}, {}
    for s in symbols:
        try:
            df = get_prices(s, "1d", period=period)
            frames[s] = _day(df)
            status[s] = {"status": "ok", "source": df.attrs.get("source", "?"), "last_bar": str(frames[s].index[-1].date())}
        except Exception as e:
            status[s] = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return frames, status


def earnings_calendar(symbols: list[str]) -> dict:
    from ..earnings import fetch_earnings_dates
    out = {}
    for s in symbols:
        try:
            d = fetch_earnings_dates(s, 8)
            idx = pd.DatetimeIndex(d.index)
            idx = idx.tz_localize(None) if idx.tz is not None else idx
            out[s] = sorted(str(x.date()) for x in idx.normalize())
        except Exception:
            out[s] = None
    return out


def _priors(reg_specs: list[dict], frames: dict) -> dict:
    out = {}
    for c in reg_specs:
        if c["kind"] != "linear" or not all(s in frames for s in c["symbols"]):
            continue
        fr = {s: frames[s].tail(756 + 260) for s in c["symbols"]}
        spread = float(np.mean([costs.round_trip_bps(s) / 2 for s in c["symbols"]]))
        try:
            bt = __import__("mlab.quant.engine", fromlist=["backtest"]).backtest(
                c["strategy"], fr, c.get("params"), c.get("overlays"), spread_bps=spread)
            r = bt["returns"].tail(756)
            from ..quant import validation
            out[c["id"]] = {"window": f"{r.index[0].date()}..{r.index[-1].date()}", "days": len(r),
                            "sharpe": round(validation.sharpe(r), 3), "total": round(float((1 + r).prod() - 1), 4),
                            "trades": int(bt["trades"]), "spread_bps": round(spread, 2),
                            "note": "trailing 3y backtest at registration, no funding; shortlist evidence only"}
        except Exception as e:
            out[c["id"]] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    return out


def cmd_register(a):
    reg = specs.load()
    new = [c for c in specs.generate() if c["id"] not in {x["id"] for x in reg["cells"]}]
    priors = {}
    if new and not a.no_priors:
        syms = sorted({s for c in new for s in c["symbols"]})
        frames, _ = load_frames(syms)
        priors = _priors(new, frames)
    res = specs.register(priors=priors)
    reg = specs.load()
    print(f"registered {len(res['added'])} new cells, {res['unchanged']} unchanged; totals {reg['counts']}")
    print(f"registry: {specs.REGISTRY_PATH.relative_to(ROOT)} (commit it: the PR is the pre-registration record)")


def us_options_open(now: pd.Timestamp) -> bool:
    """Options books fill and mark only on live two-sided quotes: weekdays 09:45-16:00 New York."""
    ny = (now if now.tzinfo else now.tz_localize("UTC")).tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and 9 * 60 + 45 <= mins <= 16 * 60


def run(force_options: bool = False, only: str | None = None, skip_options: bool = False, now=None) -> dict:
    """One forward run. Returns the run record (also written to events/runs)."""
    t0 = time.time()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    run_id = now.strftime("%Y-%m-%dT%H%MZ") + "-" + secrets.token_hex(2)
    cutoff = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    reg = specs.load()
    bad = specs.verify(reg)
    if bad:
        raise ValueError(f"registry hash check failed for {bad[:5]}: refusing to run")
    reg_sha, _ = store.freeze_registry(reg)
    prov = {"run_id": run_id, "registry_sha": reg_sha[:16], **provenance()}
    cells = [c for c in reg["cells"] if not c.get("retired_at") and (not only or only in c["id"])]
    with store.RunLock(run_id):
        state, prev_run = store.load_state()
        frozen = state.setdefault("frozen", {})
        syms = sorted({s for c in cells for s in c["symbols"]} | set(specs.CONTEXT_SYMBOLS))
        frames, data_status = load_frames(syms)
        mkt = context.market({k: frames[k] for k in specs.CONTEXT_SYMBOLS if k in frames})
        inst = {s: context.instrument(df) for s, df in frames.items()}
        stocks = sorted({s for c in cells for s in c["symbols"] if costs.asset_class(s) in ("us_share", "uk_share")}
                        | {c["symbols"][0] for c in cells if c["kind"] == "option"
                           and costs.asset_class(c["symbols"][0]) == "us_share"})
        earnings = earnings_calendar(stocks)
        sig, trig, fil, grk, vol = [], [], [], [], []
        items = {}
        for c in cells:
            if c["kind"] != "linear":
                continue
            if frozen.get(c["id"], c["spec_sha"]) != c["spec_sha"]:
                items[c["id"]] = {"status": "refused", "why": "spec hash differs from the one frozen at first signal"}
                continue
            missing = [s for s in c["symbols"] if s not in frames]
            if missing:
                items[c["id"]] = {"status": "failed", "why": f"no data: {missing}"}
                continue
            try:
                st, s1, t1, f1 = linear.step_cell(c, state["cells"].get(c["id"]), frames, mkt, inst, earnings,
                                                  cutoff, prov)
                state["cells"][c["id"]] = st
                frozen.setdefault(c["id"], c["spec_sha"])
                sig += s1
                trig += t1
                fil += f1
                items[c["id"]] = {"status": "ok", "bars": len(s1)}
            except Exception as e:
                items[c["id"]] = {"status": "failed", "why": f"{type(e).__name__}: {str(e)[:120]}"}
        opt_cells = [c for c in cells if c["kind"] == "option"]
        chains = []
        if opt_cells and not skip_options and not us_options_open(now) and not force_options:
            for c in opt_cells:
                items[c["id"]] = {"status": "skipped", "why": "US options market closed: no live bid/ask to fill or mark"}
        elif opt_cells and not skip_options:
            from .. import options as opt
            r = opt.risk_free_rate()
            today = str(cutoff.date())
            for und in sorted({c["symbols"][0] for c in opt_cells}):
                mine = [c for c in opt_cells if c["symbols"][0] == und]
                held = sorted({lg["expiry"] for c in mine for lg in
                               ((state["options"].get(c["id"]) or {}).get("pos") or {}).get("legs", [])
                               if lg["type"] != "U"})
                held = [h for h in held]
                try:
                    spot, raw = optsim.fetch_selected(und, [7, 30, 35, 45, 60, 90], held, now)
                    ch = opt.normalize_chain(raw, spot, r)
                except Exception as e:
                    for c in mine:
                        items[c["id"]] = {"status": "failed", "why": f"chain: {type(e).__name__}: {str(e)[:100]}"}
                    continue
                vs = optsim.vol_snapshot(ch, spot, frames[und]["close"] if und in frames else None)
                vol.append({"underlying": und, "bar": today, **vs, **prov})
                if vs.get("atm_iv30"):
                    try:
                        opt.record_atm_iv(und, vs["atm_iv30"], spot, now)
                    except Exception as e:  # IV history is a by-product; never fail the run on it
                        vs["iv_history_error"] = f"{type(e).__name__}"
                near = ch[(ch["moneyness"].between(0.75, 1.25)) & (ch["dte"] <= 100)]
                chains.append(near.assign(underlying=und, spot=spot, run_id=run_id)[
                    ["underlying", "spot", "expiry", "dte", "type", "strike", "bid", "ask", "mid", "last", "volume",
                     "oi", "iv", "iv_src", "delta", "gamma", "run_id"]])
                for c in mine:
                    if frozen.get(c["id"], c["spec_sha"]) != c["spec_sha"]:
                        items[c["id"]] = {"status": "refused", "why": "spec hash changed since first signal"}
                        continue
                    stc = state["options"].get(c["id"])
                    try:
                        st, s1, t1, f1, g1 = optsim.step_option(c, stc, ch, spot, r, vs, today, prov,
                                                                earnings.get(und))
                        state["options"][c["id"]] = st
                        frozen.setdefault(c["id"], c["spec_sha"])
                        if s1:
                            sig.append(s1)
                        trig += t1
                        fil += f1
                        grk += g1
                        items[c["id"]] = {"status": "ok", "bars": 1 if s1 else 0}
                    except Exception as e:
                        items[c["id"]] = {"status": "failed", "why": f"{type(e).__name__}: {str(e)[:120]}"}
        written = {}
        for kind, rows in (("signals", sig), ("triggers", trig), ("fills", fil), ("greeks", grk), ("vol", vol)):
            p = store.append_events(kind, run_id, [_clean(r) for r in rows])
            written[kind] = len(rows) if p else 0
        if chains:
            store.save_chain(run_id, pd.concat(chains, ignore_index=True))
        state["last_run"] = run_id
        store.save_state(run_id, state)
        counts = pd.Series([v["status"] for v in items.values()]).value_counts().to_dict() if items else {}
        rec = {"run_id": run_id, "at": store.now_iso(), "prev_run": prev_run, "cutoff": str(cutoff.date()),
               "cells": len(cells), "item_counts": counts, "rows": written, "data": data_status,
               "items_not_ok": {k: v for k, v in items.items() if v["status"] != "ok"},
               "earnings_fetched": {k: v is not None for k, v in earnings.items()},
               "seconds": round(time.time() - t0, 1), "schema_version": SCHEMA_VERSION, **prov}
        store.append_events("runs", run_id, [_clean(rec)])
    try:
        from .. import ledger
        ledger.snapshot(note=f"forward run {run_id}")
    except Exception as e:
        rec["ledger"] = f"failed: {e}"
    return rec


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def cmd_run(a):
    rec = run(force_options=a.force_options, only=a.only, skip_options=a.skip_options)
    print(f"run {rec['run_id']} ({rec['seconds']}s, code {rec['code_sha']}, registry {rec['registry_sha']})")
    print(f"cells {rec['cells']}: {rec['item_counts']}; rows {rec['rows']}")
    bad_data = {k: v for k, v in rec["data"].items() if v["status"] != "ok"}
    if bad_data:
        print(f"data failures: {bad_data}")
    if rec["items_not_ok"]:
        print(f"items not ok ({len(rec['items_not_ok'])}): " + json.dumps(dict(list(rec["items_not_ok"].items())[:10])))


def cmd_status(a):
    from ..cli import show
    reg = specs.load()
    state, last = store.load_state()
    runs = store.read_events("runs")
    print(f"registry: {reg.get('counts')}, bars sha {reg.get('bars_sha', '')[:12]}")
    print(f"last run: {last}; state cells {len(state['cells'])}, option books {len(state['options'])}")
    u = store.usage()
    print("store bytes: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in u.items()))
    if not runs.empty:
        show(runs[["run_id", "cutoff", "cells", "item_counts", "rows", "seconds", "code_sha"]].tail(a.tail), a.json,
             title="Recent runs")


def cmd_score(a):
    from ..cli import show
    reg = specs.load()
    sig = store.read_events("signals")
    if sig.empty:
        print("no forward signals yet: run `mlab fwd run` first")
        return
    if a.by:
        show(score.by_context(sig, a.by, a.book).head(a.top), a.json, title=f"Pooled by {a.by} ({a.book} book)")
        return
    if a.filters:
        show(score.filter_value(sig), a.json, title="Desk filters graded: desk minus raw, per strategy (bp/day)")
        return
    if a.books:
        b = score.grade_books(sig, reg, book=a.book)
        print(f"book verdicts: {b['verdict'].value_counts().to_dict()} (trials {b.attrs['trials']})")
        cols = ["strategy", "cells", "n_days", "trades", "total", "sharpe", "dsr", "q", "excess_psr", "verdict"]
        show(b.sort_values("t", ascending=False)[cols].head(a.top), a.json, title=f"Forward books ({a.book})")
        return
    g = score.grade(sig, reg, book=a.book)
    print(f"verdicts: {g['verdict'].value_counts().to_dict()} (trials {sum(g['role'] == 'candidate')})")
    cols = ["strategy", "group", "n_days", "trades", "total", "sharpe", "dsr", "q", "excess_psr", "prior_sharpe",
            "verdict"]
    show(g[g["role"] == "candidate"].sort_values("t", ascending=False)[cols].head(a.top), a.json,
         title=f"Forward cells, best t first ({a.book} book; provisional until {reg['bars']['min_days']} bars)")


def cmd_gate(a):
    reg = specs.load()
    syms = sorted({s for c in reg["cells"] if c["role"] == "null" for s in c["symbols"]})
    frames, _ = load_frames(syms)
    g = score.gate(reg, frames, days=a.days)
    print(f"gate {g['status']}: {g.get('pass_full', 0)} of {g.get('nulls', 0)} nulls pass the full bar set "
          f"(trials {g.get('trials')}); without the excess bar {g.get('pass_without_excess_bar')} would pass; "
          f"per bar {g.get('pass_each')}")
    if g.get("passing"):
        print("nulls passing: " + ", ".join(g["passing"][:20]))


def cmd_audit(a):
    reg = specs.load()
    bad = specs.verify(reg)
    frozen_dir = store.root() / "registry"
    frozen = sorted(p.name for p in frozen_dir.glob("*.json")) if frozen_dir.exists() else []
    state, last = store.load_state()
    drift = [cid for cid, sha in state.get("frozen", {}).items()
             if cid in {c["id"]: c for c in reg["cells"]} and {c["id"]: c for c in reg["cells"]}[cid]["spec_sha"] != sha]
    schemas = {}
    for kind in store.KINDS:
        df = store.read_events(kind)
        schemas[kind] = sorted(df["schema"].unique().tolist()) if not df.empty else []
    from .. import ledger
    lv = ledger.verify()
    print(f"registry hashes: {'ok' if not bad else 'BAD ' + str(bad[:5])}")
    print(f"frozen registry copies: {len(frozen)}; cells frozen at first signal: {len(state.get('frozen', {}))}; "
          f"spec drift since freeze: {drift or 'none'}")
    print(f"event schemas: {schemas}")
    print(f"ledger: {lv['status']} ({lv['manifests']} manifests, {lv['objects_checked']} object checks)")
    print(f"store: {store.usage()['total'] / 1e6:.2f} MB")


def cmd_fwd(a):
    {"register": cmd_register, "run": cmd_run, "status": cmd_status, "score": cmd_score, "gate": cmd_gate,
     "audit": cmd_audit}[a.action](a)


def register(add):
    q = add("fwd", cmd_fwd, "forward paper-testing lab: register | gate | run | status | score | audit (paper only)")
    q.add_argument("action", choices=["register", "gate", "run", "status", "score", "audit"])
    q.add_argument("--only", help="run: only cells whose id contains this text")
    q.add_argument("--skip-options", action="store_true"); q.add_argument("--force-options", action="store_true",
                                                                                    help="mark option books outside US hours (stale quotes)")
    q.add_argument("--no-priors", action="store_true", help="register: skip the trailing backtest priors")
    q.add_argument("--by", help="score: pool by a context key (vix_bucket, trend, rv_bucket, spx_trend, gate_ndx...)")
    q.add_argument("--filters", action="store_true", help="score: grade the desk filters (desk minus raw)")
    q.add_argument("--books", action="store_true", help="score: primary grade, equal-weight books across instruments")
    q.add_argument("--book", default="raw", choices=["raw", "desk"])
    q.add_argument("--top", type=int, default=30); q.add_argument("--tail", type=int, default=10)
    q.add_argument("--days", type=int, default=504, help="gate: trailing backtest length")
