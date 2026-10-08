"""Paper books: run a strategy forward on real closes with no money and no orders.

A book lives in $MLAB_DATA_DIR/paper/<book>.json. Each `rebalance` marks the book to the latest
close, moves holdings to the strategy's current target weights at that close (charging the
spread), and logs the fills. Running it twice on the same bar only marks to market.
There is no execution path: this module never talks to a broker.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, env
from ..storage import atomic_write_text, file_lock
from . import engine

PAPER_DIR = Path(env("MLAB_PAPER_DIR") or DATA_DIR / "paper")


def _path(book: str) -> Path:
    return PAPER_DIR / f"{book}.json"


def load(book: str) -> dict | None:
    p = _path(book)
    return json.loads(p.read_text()) if p.exists() else None


def _save(state: dict):
    atomic_write_text(_path(state["book"]), json.dumps(state, indent=1, default=str))


def rebalance(book: str, strategy: str, frames: dict[str, pd.DataFrame], params: dict | None = None,
              overlays: dict | None = None, capital: float = 10_000.0, spread_bps: float = 5.0) -> dict:
    with file_lock(_path(book)):  # one book has one writer: the lock stops a second run in this container
        return _rebalance(book, strategy, frames, params, overlays, capital, spread_bps)


def _rebalance(book, strategy, frames, params, overlays, capital, spread_bps) -> dict:
    state = load(book) or {"book": book, "strategy": strategy, "params": params or {}, "overlays": overlays or {},
                           "symbols": list(frames), "capital": capital, "cash": capital, "units": {},
                           "created": datetime.now(UTC).isoformat(timespec="seconds"),
                           "history": [], "fills": []}
    if state["strategy"] != strategy:
        raise ValueError(f"book {book!r} runs {state['strategy']}; use another --book name")
    sig = engine.latest_signal(strategy, frames, state["params"], state["overlays"])
    px = sig["last_close"].to_dict()
    bar = sig["as_of"].iloc[0]
    units = {k: float(v) for k, v in state["units"].items()}
    equity = state["cash"] + sum(units.get(s, 0.0) * px[s] for s in px)
    already = state["history"] and state["history"][-1]["bar"] == bar
    if not already:
        for sym, w in sig["target_weight"].items():
            target = w * equity / px[sym]
            delta = target - units.get(sym, 0.0)
            if abs(delta * px[sym]) < 1e-6 * equity:
                continue
            cost = abs(delta * px[sym]) * spread_bps / 1e4
            state["cash"] -= delta * px[sym] + cost
            units[sym] = target
            state["fills"].append({"bar": bar, "symbol": sym, "units": delta, "price": px[sym], "cost": cost,
                                   "target_weight": w})
        equity = state["cash"] + sum(units.get(s, 0.0) * px[s] for s in px)
    state["units"] = units
    snap = {"bar": bar, "equity": equity, "marked_at": datetime.now(UTC).isoformat(timespec="seconds")}
    if already:
        state["history"][-1] = snap
    else:
        state["history"].append(snap)
    _save(state)
    return status(book)


def status(book: str | None = None) -> pd.DataFrame | dict:
    if book is None:
        rows = []
        for p in sorted(PAPER_DIR.glob("*.json")) if PAPER_DIR.exists() else []:
            s = json.loads(p.read_text())
            eq = s["history"][-1]["equity"] if s["history"] else s["capital"]
            rows.append({"book": s["book"], "strategy": s["strategy"], "symbols": ",".join(s["symbols"]),
                         "since": s["created"][:10], "bars": len(s["history"]), "equity": eq,
                         "return": eq / s["capital"] - 1, "fills": len(s["fills"])})
        return pd.DataFrame(rows).set_index("book") if rows else pd.DataFrame()
    s = load(book)
    if s is None:
        raise FileNotFoundError(f"no paper book {book!r} in {PAPER_DIR}")
    h = pd.DataFrame(s["history"])
    eq = h["equity"] if len(h) else pd.Series([s["capital"]])
    return {"book": book, "strategy": s["strategy"], "params": s["params"], "symbols": s["symbols"],
            "equity": float(eq.iloc[-1]), "return": float(eq.iloc[-1] / s["capital"] - 1),
            "max_dd": float((eq / eq.cummax() - 1).min()), "bars": len(h), "fills": len(s["fills"]),
            "holdings": pd.DataFrame({"units": s["units"]}), "recent_fills": pd.DataFrame(s["fills"][-10:])}
