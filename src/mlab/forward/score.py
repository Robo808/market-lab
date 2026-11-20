"""Grade forward cells against bars that scale with the number of cells tested.

A candidate cell PASSES only when, on its forward record net of costs:
- it has at least `min_days` forward bars and `min_trades` entries (allocation books: `min_trades_allocation`);
- its deflated Sharpe, with trials = every registered candidate cell, is at least `dsr_min`;
- its one-sided t-test on mean daily return survives Benjamini-Hochberg at `fdr_q` across all candidates;
- its daily excess over the same instrument's (or basket's) always_long control has PSR >= `excess_psr_min`.
Before `min_days` the verdict is PENDING and every number is provisional.

`gate` runs the same bars on trailing backtests of the null stubs; a bar set that passes a null is refused.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..quant import engine, validation
from . import costs

ALLOCATION = {"dual_momentum", "inverse_vol", "risk_parity", "xs_momentum"}


def null_of(cell_id: str, kind: str = "linear") -> str:
    target = cell_id.split(":", 1)[1]
    return f"always_long.v1:opt-{target}" if kind == "option" else f"always_long.v1:{target}"


def returns_panel(signals: pd.DataFrame, book: str = "raw") -> pd.DataFrame:
    """bars x cells of daily returns (the first, unfilled bar of each cell dropped; last run per bar wins)."""
    if signals.empty:
        return pd.DataFrame()
    s = signals[~signals.get("first", False).fillna(False).astype(bool)] if "first" in signals else signals
    s = s.sort_values("run_id").drop_duplicates(["cell", "bar"], keep="last")
    return s.pivot(index="bar", columns="cell", values=f"ret_{book}").sort_index()


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted q-values."""
    p = p.dropna()
    if p.empty:
        return p
    order = p.sort_values()
    m = len(order)
    q = order * m / np.arange(1, m + 1)
    q = q[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def metrics(r: pd.Series, trials: int) -> dict:
    r = pd.Series(r).dropna()
    n = len(r)
    sd = r.std() if n > 1 else float("nan")
    t = float(r.mean() / sd * math.sqrt(n)) if sd and sd > 0 else float("nan")
    return {"n_days": n, "total": float((1 + r).prod() - 1) if n else 0.0,
            "sharpe": validation.sharpe(r) if n > 2 else float("nan"),
            "psr": validation.psr(r) if n > 2 else float("nan"),
            "dsr": validation.dsr(r, trials=max(trials, 1)) if n > 2 else float("nan"),
            "t": t, "p": float(1 - norm.cdf(t)) if np.isfinite(t) else 1.0}


def grade(signals: pd.DataFrame, registry: dict, bars: dict | None = None, book: str = "raw") -> pd.DataFrame:
    bars = bars or registry["bars"]
    cells = {c["id"]: c for c in registry["cells"]}
    trials = sum(c["role"] == "candidate" for c in registry["cells"])
    panel = returns_panel(signals, book)
    last = signals.sort_values("run_id").drop_duplicates("cell", keep="last").set_index("cell") \
        if not signals.empty else pd.DataFrame()
    rows = []
    for cid, c in cells.items():
        r = panel[cid] if cid in panel else pd.Series(dtype=float)
        m = metrics(r, trials)
        nid = null_of(cid, c["kind"])
        if c["role"] == "candidate" and nid in panel and len(r.dropna()) > 2:
            ex = (r - panel[nid]).dropna()
            m["excess_sharpe"] = validation.sharpe(ex)
            m["excess_psr"] = validation.psr(ex)
        else:
            m["excess_sharpe"] = m["excess_psr"] = float("nan")
        tcol = f"trades_{book}"
        m["trades"] = int(last.loc[cid, tcol]) if (cid in last.index and tcol in last and
                                                    pd.notna(last.loc[cid, tcol])) else 0
        m.update({"cell": cid, "role": c["role"], "kind": c["kind"], "group": c["group"],
                  "strategy": c.get("strategy") or c.get("structure"),
                  "prior_sharpe": (c.get("prior") or {}).get("sharpe", float("nan"))})
        rows.append(m)
    df = pd.DataFrame(rows).set_index("cell")
    cand = df["role"] == "candidate"
    df["q"] = np.nan
    df.loc[cand, "q"] = bh(df.loc[cand, "p"])
    df["verdict"] = [verdict(r, bars) for _, r in df.iterrows()]
    return df


def book_key(cell: dict) -> str:
    """The book a cell belongs to: its matrix row (id before ':') for single-instrument and option cells,
    itself for basket strategies."""
    if cell["group"] == "basket":
        return cell["id"]
    return cell["id"].split(":", 1)[0] + (":options" if cell["kind"] == "option" else ":all")


def grade_books(signals: pd.DataFrame, registry: dict, bars: dict | None = None, book: str = "raw") -> pd.DataFrame:
    """Primary grade: equal-weight books across instruments, trials = number of candidate books."""
    bars = bars or registry["bars"]
    panel = returns_panel(signals, book)
    if panel.empty:
        return pd.DataFrame()
    cells = [c for c in registry["cells"] if c["id"] in panel]
    groups: dict[str, list[str]] = {}
    for c in cells:
        groups.setdefault(book_key(c), []).append(c["id"])
    by_id = {c["id"]: c for c in registry["cells"]}
    cand = {k: v for k, v in groups.items() if by_id[v[0]]["role"] == "candidate"}
    trials = len({book_key(c) for c in registry["cells"] if c["role"] == "candidate"})
    last = signals.sort_values("run_id").drop_duplicates("cell", keep="last").set_index("cell")
    rows = []
    for k, ids in cand.items():
        r = panel[ids].mean(axis=1, skipna=True)
        m = metrics(r, trials)
        nulls = [null_of(i, by_id[i]["kind"]) for i in ids]
        nulls = [n for n in nulls if n in panel]
        if nulls and len(r.dropna()) > 2:
            ex = (r - panel[nulls].mean(axis=1)).dropna()
            m["excess_sharpe"], m["excess_psr"] = validation.sharpe(ex), validation.psr(ex)
        else:
            m["excess_sharpe"] = m["excess_psr"] = float("nan")
        tcol = f"trades_{book}"
        m["trades"] = int(last.reindex(ids)[tcol].fillna(0).sum()) if tcol in last else 0
        c0 = by_id[ids[0]]
        m.update({"book": k, "cells": len(ids), "role": "candidate", "kind": c0["kind"],
                  "strategy": c0.get("strategy") or c0.get("structure")})
        rows.append(m)
    df = pd.DataFrame(rows).set_index("book")
    df["q"] = bh(df["p"])
    df["verdict"] = [verdict(r, bars) for _, r in df.iterrows()]
    df.attrs["trials"] = trials
    return df


def verdict(r, bars: dict) -> str:
    if r["role"] != "candidate":
        return "control"
    if r["n_days"] < bars["min_days"]:
        return "PENDING"
    min_tr = bars["min_trades_allocation"] if r["strategy"] in ALLOCATION else bars["min_trades"]
    ok = (r["trades"] >= min_tr and r["dsr"] >= bars["dsr_min"] and r["q"] <= bars["fdr_q"]
          and r["excess_psr"] >= bars["excess_psr_min"])
    return "PASS" if ok else "FAIL"


def bar_checks(r, bars: dict) -> dict:
    return {"dsr": bool(r["dsr"] >= bars["dsr_min"]), "fdr": bool(r["q"] <= bars["fdr_q"]),
            "excess": bool(r["excess_psr"] >= bars["excess_psr_min"])}


def by_context(signals: pd.DataFrame, key: str, book: str = "raw", role: str = "candidate") -> pd.DataFrame:
    """Pooled daily returns per strategy and context value: where does each rule earn its keep?"""
    if signals.empty:
        return pd.DataFrame()
    s = signals[(signals["role"] == role) & ~signals.get("first", False).fillna(False).astype(bool)].copy()
    s[key] = s["context"].map(lambda c: (c or {}).get(key))
    g = s.groupby(["strategy", key])[f"ret_{book}"]
    out = pd.DataFrame({"n": g.size(), "mean_bp": g.mean() * 1e4, "sd_bp": g.std() * 1e4})
    out["t"] = out["mean_bp"] / out["sd_bp"] * np.sqrt(out["n"])
    return out.sort_values("t", ascending=False)


def filter_value(signals: pd.DataFrame) -> pd.DataFrame:
    """Desk filters graded: per strategy, the desk book's return minus the raw book's."""
    if signals.empty:
        return pd.DataFrame()
    s = signals[(signals["role"] == "candidate") & ~signals.get("first", False).fillna(False).astype(bool)]
    s = s.assign(diff=s["ret_desk"] - s["ret_raw"])
    g = s.groupby("strategy")["diff"]
    out = pd.DataFrame({"n": g.size(), "mean_bp": g.mean() * 1e4, "sd_bp": g.std() * 1e4})
    out["t"] = out["mean_bp"] / out["sd_bp"] * np.sqrt(out["n"])
    return out.sort_values("t", ascending=False)


def gate(registry: dict, frames: dict[str, pd.DataFrame], days: int = 504, bars: dict | None = None) -> dict:
    """Apply the pass bars to trailing backtests of every null control (and, for contrast, with the excess bar
    switched off). OK only if no null passes the full bar set."""
    bars = {**(bars or registry["bars"]), "min_days": 0}
    trials = sum(c["role"] == "candidate" for c in registry["cells"])
    rets, rows = {}, []
    nulls = [c for c in registry["cells"] if c["role"] == "null" and all(s in frames for s in c["symbols"])]
    for c in nulls:
        fr = {s: frames[s].tail(days + 260) for s in c["symbols"]}
        spread = float(np.mean([costs.round_trip_bps(s) / 2 for s in c["symbols"]]))
        bt = engine.backtest(c["strategy"], fr, c.get("params"), c.get("overlays"), spread_bps=spread)
        rets[c["id"]] = bt["returns"].tail(days)
    for c in nulls:
        r = rets[c["id"]]
        m = metrics(r, trials)
        base = null_of(c["id"]) if c["strategy"] != "always_long" else c["id"]
        ex = (r - rets[base]).dropna() if base in rets else pd.Series(dtype=float)
        m["excess_psr"] = validation.psr(ex) if len(ex) > 2 and ex.std() > 0 else 0.0
        m.update({"cell": c["id"], "strategy": c["strategy"], "trades": 999, "role": "candidate"})
        rows.append(m)
    df = pd.DataFrame(rows).set_index("cell") if rows else pd.DataFrame()
    if df.empty:
        return {"status": "NO DATA", "table": df}
    df["q"] = bh(df["p"])
    checks = pd.DataFrame([bar_checks(r, bars) for _, r in df.iterrows()], index=df.index)
    full = checks.all(axis=1)
    no_excess = checks[["dsr", "fdr"]].all(axis=1)
    return {"status": "OK" if not full.any() else "REFUSED", "nulls": len(df), "trials": trials,
            "pass_full": int(full.sum()), "pass_without_excess_bar": int(no_excess.sum()),
            "pass_each": checks.sum().to_dict(), "passing": list(df.index[full]),
            "table": df.join(checks.add_prefix("bar_"))}
