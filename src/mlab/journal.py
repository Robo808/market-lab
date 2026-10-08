"""Trade journal, event-sourced so threads in different containers never lose each other's entries.

Every change (a new card, a status change, a fill, an exit) is one new write-once file in
journal/events/. The journal is the fold of journal/trades.jsonl (entries written before events existed,
read-only now) plus those events in time order. JOURNAL.md is a rendered view, regenerated on every write.

Lifecycle: idea -> open -> closed (or idea -> passed). Every entry keeps the trade card,
the lens, and the sources, so reviews can score which lens actually makes money.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .storage import append_event, atomic_write_text, read_events

JOURNAL_DIR = Path(os.environ.get("MLAB_JOURNAL_DIR", DATA_DIR / "journal"))
FILE = JOURNAL_DIR / "trades.jsonl"  # legacy base, read-only
EVENTS = JOURNAL_DIR / "events"
STATUSES = ("idea", "open", "closed", "passed")


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def _base() -> list[dict]:
    if not FILE.exists():
        return []
    return [json.loads(line) for line in FILE.read_text().splitlines() if line.strip()]


def _apply(rows: dict[str, dict], ev: dict) -> None:
    if ev["op"] == "add":
        rows.setdefault(ev["row"]["id"], ev["row"])
        return
    r = rows.get(ev["id"])
    if r is None:  # update for an entry this reader cannot see (yet): skip, the fold is re-run on every read
        return
    if ev.get("status"):
        r["status"] = ev["status"]
    for k in ("fill", "size"):
        if ev.get(k) is not None:
            r[k] = ev[k]
    if ev.get("exit") is not None:
        r["exit"], r["closed"] = ev["exit"], ev["t"]
        r["r_multiple"] = r_multiple(r)
    r["events"].append({"t": ev["t"], "status": r["status"], "note": ev.get("note", ""),
                        **({"fill": ev["fill"]} if ev.get("fill") is not None else {}),
                        **({"exit": ev["exit"]} if ev.get("exit") is not None else {})})


def _read() -> list[dict]:
    rows = {r["id"]: r for r in _base()}
    for ev in read_events(EVENTS):
        _apply(rows, ev)
    return list(rows.values())


def add(card: dict, status: str = "idea", note: str = "") -> dict:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    row = {"id": uuid.uuid4().hex[:8], "created": _now(), "status": status, "card": card,
           "events": [{"t": _now(), "status": status, "note": note}]}
    append_event(EVENTS, {"op": "add", "t": row["created"], "row": row})
    render()
    return row


def update(trade_id: str, status: str | None = None, fill: float | None = None, exit_price: float | None = None,
           size: float | None = None, note: str = "") -> dict:
    if status and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    rows = {r["id"]: r for r in _read()}
    if trade_id not in rows:
        raise KeyError(trade_id)
    ev = {"op": "update", "t": _now(), "id": trade_id, "status": status, "fill": fill, "size": size,
          "exit": exit_price, "note": note}
    append_event(EVENTS, ev)
    _apply(rows, ev)
    render()
    return rows[trade_id]


def r_multiple(r: dict) -> float | None:
    c = r["card"]
    entry = r.get("fill") or sum(c["entry_zone"]) / 2
    risk = abs(entry - c["stop"])
    if not risk or r.get("exit") is None:
        return None
    sign = 1 if c["bias"].upper().startswith("L") else -1
    return round(sign * (r["exit"] - entry) / risk, 2)


def table(status: str | None = None) -> pd.DataFrame:
    rows = [r for r in _read() if status in (None, r["status"])]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "id": r["id"], "created": r["created"][:10], "status": r["status"], "bias": r["card"]["bias"],
        "instrument": r["card"]["instrument"], "lens": r["card"].get("lens", ""),
        "entry": "-".join(f"{x:g}" for x in r["card"]["entry_zone"]), "stop": r["card"]["stop"],
        "targets": ",".join(f"{t:g}" for t in r["card"]["targets"]), "conv": r["card"].get("conviction"),
        "fill": r.get("fill"), "exit": r.get("exit"), "R": r.get("r_multiple")} for r in rows])


def review() -> dict:
    df = table("closed")
    if df.empty or df["R"].isna().all():
        return {"closed_trades": 0}
    R = df["R"].dropna()
    wins, losses = R[R > 0], R[R <= 0]
    by_lens = df.groupby("lens")["R"].agg(["count", "mean", "sum"]).round(2).to_dict("index")
    by_conv = df.groupby("conv")["R"].agg(["count", "mean"]).round(2).to_dict("index")
    return {"closed_trades": len(R), "win_rate": round((R > 0).mean(), 3), "avg_win_R": round(wins.mean(), 2) if len(wins) else None,
            "avg_loss_R": round(losses.mean(), 2) if len(losses) else None, "expectancy_R": round(R.mean(), 3),
            "total_R": round(R.sum(), 2), "best_R": R.max(), "worst_R": R.min(), "by_lens": by_lens, "by_conviction": by_conv}


def render() -> Path:
    out = JOURNAL_DIR / "JOURNAL.md"
    parts = ["# Trade journal\n", "_Generated from trades.jsonl + events/ by `mlab journal`. Edit via the CLI, not by hand._\n"]
    for st in ("open", "idea", "closed", "passed"):
        df = table(st)
        if not df.empty:
            parts.append(f"\n## {st.title()} ({len(df)})\n\n" + df.to_markdown(index=False) + "\n")
    rv = review()
    if rv.get("closed_trades"):
        parts.append("\n## Review\n\n```\n" + json.dumps(rv, indent=2, default=str) + "\n```\n")
    return atomic_write_text(out, "".join(parts))
