"""Linear strategy books run forward bar by bar: decide at the close of bar t, fill at the open of t+1.

Each cell keeps two books on the same signal stream:
- raw: takes every target the rule produces;
- desk: takes an exposure-increasing trigger only if the desk filters pass (LESSONS L-019 cost in R, L-020
  scheduled earnings, L-027 index gate, L-028 all-gates-fail), and follows exits and resizing.
Every trigger is logged with each filter's verdict, so skipped trades are scored as well as taken ones and the
filters themselves get graded (desk minus raw).

Forward means forward: a cell's first processed bar is the last complete bar at its first run. Nothing is
backfilled, and a missed day is caught up bar by bar on the next run.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..quant import engine
from . import context, costs

R_ATR_MULT = 2.0          # desk cards stop about 2 ATR away: 1R = 2 x ATR(14) (assumption for the cost filter)
COST_R_MAX = 0.2          # LESSONS L-019
EARNINGS_DAYS = 3         # LESSONS L-020: two sessions, counted as up to three calendar days ahead
GATE_SOX = {"SMH", "NVDA", "AMD", "AVGO", "^SOX"}
GATE_NDX = {"^NDX", "QQQ", "XLK", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"}


def _sign(x: float) -> int:
    return 0 if abs(x) < 1e-12 else (1 if x > 0 else -1)


def gate_key(sym: str) -> str | None:
    if sym.startswith("crypto:"):
        return "gate_btc"
    if sym in GATE_SOX:
        return "gate_sox"
    if sym in GATE_NDX:
        return "gate_ndx"
    return None


def filters(sym: str, side: int, ctx: dict, inst_ctx: dict, day: pd.Timestamp, earnings: dict) -> dict:
    """Desk filter verdicts for an exposure-increasing trigger: {name: True pass | False fail | None n/a}."""
    out = {}
    g = gate_key(sym)
    out["L027_index_gate"] = (bool(ctx[g]) if g in ctx else None) if (g and side > 0) else None
    out["L028_all_gates_fail"] = (not bool(ctx["gate_all_fail"]) if "gate_all_fail" in ctx else None) \
        if side > 0 else None
    atr = inst_ctx.get("atr_pct")
    if atr and atr > 0:
        cost_r = costs.round_trip_bps(sym, side) / (R_ATR_MULT * atr * 1e4)
        out["L019_cost_in_R"] = cost_r <= COST_R_MAX
        out["cost_R"] = round(cost_r, 3)
    else:
        out["L019_cost_in_R"] = None
    if sym in earnings:
        dates = earnings[sym]
        if dates is None:
            out["L020_earnings"] = None
        else:
            ahead = [d for d in dates if 0 <= (pd.Timestamp(d) - day).days <= EARNINGS_DAYS]
            out["L020_earnings"] = not ahead
            if ahead:
                out["earnings_date"] = str(pd.Timestamp(ahead[0]).date())
    return out


def passes(f: dict) -> bool:
    return all(v is not False for k, v in f.items() if k.startswith("L0"))


def desk_target(raw_prev: float, raw_now: float, desk_now: float, ok: bool) -> float:
    s_raw, s_desk = _sign(raw_now), _sign(desk_now)
    if s_raw == s_desk:
        return raw_now                    # same side (or both flat): follow the sizing
    if s_raw == 0:
        return 0.0                        # exit
    if _sign(raw_prev) != s_raw and ok:
        return raw_now                    # a fresh trigger that passed the filters
    return 0.0 if s_desk != 0 else desk_now   # blocked flip goes flat; a missed entry stays flat


def new_book() -> dict:
    return {"cash": 1.0, "units": {}, "pending": {}, "equity": 1.0, "trades": 0}


def _equity(book: dict, px: dict) -> float:
    return book["cash"] + sum(u * px[s] for s, u in book["units"].items() if s in px and np.isfinite(px[s]))


def _fill(book: dict, opens: dict, closes_prev: dict, cell_id: str, bar: str, label: str, fills: list) -> float:
    """Move the book to its pending weights at the bar's open. Returns cost paid (equity units)."""
    if not book["pending"]:
        return 0.0
    px = {s: (opens[s] if np.isfinite(opens.get(s, np.nan)) else closes_prev.get(s, np.nan)) for s in opens}
    eq = _equity(book, px)
    paid = 0.0
    for s, w_to in book["pending"].items():
        p = px.get(s, np.nan)
        if not np.isfinite(p) or p <= 0:
            continue
        u0 = book["units"].get(s, 0.0)
        w_from = u0 * p / eq if eq else 0.0
        if abs(w_to - w_from) < 1e-9:
            continue
        cost, venues = costs.fill_cost(s, w_from, w_to, eq)
        u1 = w_to * eq / p
        book["cash"] -= (u1 - u0) * p + cost
        book["units"][s] = u1
        paid += cost
        if _sign(w_to) != 0 and _sign(w_to) != _sign(w_from):
            book["trades"] += 1
        fills.append({"cell": cell_id, "book": label, "bar": bar, "symbol": s, "price": p, "w_from": round(w_from, 6),
                      "w_to": round(w_to, 6), "units": u1 - u0, "cost": cost, "venues": venues,
                      "price_src": "open" if np.isfinite(opens.get(s, np.nan)) else "prev_close"})
    book["pending"] = {}
    return paid


def _funding(book: dict, px_prev: dict, days: float) -> float:
    eq = _equity(book, px_prev)
    f = 0.0
    for s, u in book["units"].items():
        p = px_prev.get(s, np.nan)
        if u and np.isfinite(p) and eq:
            f += costs.funding_cost(s, u * p / eq, eq, days)
    book["cash"] -= f
    return f


def step_cell(cell: dict, st: dict | None, frames: dict[str, pd.DataFrame], mkt: pd.DataFrame,
              inst_ctx: dict[str, pd.DataFrame], earnings: dict, cutoff: pd.Timestamp, prov: dict) -> tuple:
    """Advance one cell over every complete bar it has not processed. Returns (state, signals, triggers, fills)."""
    syms = cell["symbols"]
    fr = {s: frames[s] for s in syms}
    w = engine.build_weights(cell["strategy"], fr, cell.get("params"), cell.get("overlays"))
    w = w[w.index < cutoff]
    closes = pd.DataFrame({s: fr[s]["close"] for s in syms}).reindex(w.index).ffill()
    opens = pd.DataFrame({s: fr[s]["open"] if "open" in fr[s] else fr[s]["close"] for s in syms}).reindex(w.index)
    if w.empty:
        return st, [], [], []
    if st is None:
        st = {"spec_sha": cell["spec_sha"], "first_bar": None, "last_bar": None, "raw": new_book(),
              "desk": new_book(), "target": {s: 0.0 for s in syms}}
        todo = [w.index[-1]]
    else:
        last = pd.Timestamp(st["last_bar"])
        todo = [d for d in w.index if d > last]
    sig, trig, fil = [], [], []
    for d in todo:
        bar = str(d.date())
        i = w.index.get_loc(d)
        prev_day = w.index[i - 1] if i > 0 else None
        px_prev = closes.iloc[i - 1].to_dict() if i > 0 else closes.iloc[i].to_dict()
        px = closes.iloc[i].to_dict()
        op = opens.iloc[i].to_dict()
        days = (d - prev_day).days if (prev_day is not None and st["last_bar"]) else 0
        row_cost = {}
        for label in ("raw", "desk"):
            b = st[label]
            eq_prev = b["equity"]
            fund = _funding(b, px_prev, days) if days else 0.0
            cost = _fill(b, op, px_prev, cell["id"], bar, label, fil)
            b["equity"] = _equity(b, px)
            row_cost[label] = (eq_prev, cost, fund)
        target = {s: float(w.iloc[i][s]) if np.isfinite(w.iloc[i][s]) else 0.0 for s in syms}
        desk_w = {}
        for s in syms:
            raw_prev, raw_now = st["target"].get(s, 0.0), target[s]
            desk_now = _desk_weight(st["desk"], s, px)
            ictx = context.row(None, inst_ctx.get(s), d)
            mctx = context.row(mkt, None, d)
            is_trigger = _sign(raw_prev) != _sign(raw_now)
            increasing = _sign(raw_now) != 0 and _sign(raw_now) != _sign(raw_prev)
            f = filters(s, _sign(raw_now), mctx, ictx, d, earnings) if increasing else {}
            ok = passes(f)
            desk_w[s] = desk_target(raw_prev, raw_now, desk_now, ok)
            if is_trigger:
                trig.append({"cell": cell["id"], "bar": bar, "symbol": s, "from": raw_prev, "to": raw_now,
                             "kind": "entry" if raw_prev == 0 else ("exit" if raw_now == 0 else "flip"),
                             "taken_raw": True, "taken_desk": _sign(desk_w[s]) == _sign(raw_now),
                             "filters": f, "close": px.get(s), "context": {**mctx, **ictx}, **prov,
                             "spec_sha": cell["spec_sha"]})
        st["raw"]["pending"] = target
        st["desk"]["pending"] = desk_w
        one = syms[0] if len(syms) == 1 else None
        ctx_row = {**context.row(mkt, None, d), **(context.row(None, inst_ctx.get(one), d) if one else {})}
        sig.append({
            "cell": cell["id"], "bar": bar, "symbols": syms, "role": cell["role"], "strategy": cell["strategy"],
            "group": cell["group"], "target": target, "desk_target": desk_w,
            "held_raw": {s: _desk_weight(st["raw"], s, px) for s in syms},
            "held_desk": {s: _desk_weight(st["desk"], s, px) for s in syms},
            "close": {s: px[s] for s in syms},
            "equity_raw": st["raw"]["equity"], "equity_desk": st["desk"]["equity"],
            "ret_raw": st["raw"]["equity"] / row_cost["raw"][0] - 1 if st["last_bar"] else 0.0,
            "ret_desk": st["desk"]["equity"] / row_cost["desk"][0] - 1 if st["last_bar"] else 0.0,
            "cost_raw": row_cost["raw"][1], "funding_raw": row_cost["raw"][2],
            "cost_desk": row_cost["desk"][1], "funding_desk": row_cost["desk"][2],
            "trades_raw": st["raw"]["trades"], "trades_desk": st["desk"]["trades"],
            "first": st["last_bar"] is None, "context": ctx_row, "spec_sha": cell["spec_sha"], **prov})
        st["target"] = target
        st["first_bar"] = st["first_bar"] or bar
        st["last_bar"] = bar
    return st, sig, trig, fil


def _desk_weight(book: dict, s: str, px: dict) -> float:
    eq = _equity(book, px)
    p = px.get(s, np.nan)
    u = book["units"].get(s, 0.0)
    return float(u * p / eq) if (eq and np.isfinite(p) and u) else 0.0


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
