"""Scheduled-buy planner (dollar-cost averaging) for accounts no API can trade.

IG's public API serves spread bet and CFD accounts only: share dealing, ISA and SIPP accounts
answer `public-api.failure.stockbroking-not-supported`, and IG's share dealing agreement asks for
prior written consent before any automated software touches its platforms. So this module never
places orders. It keeps a plan of buy tranches (resting limits and dated market buys), checks the
plan against daily prices, and says what to do today; the order itself is one tap on the platform.

Plan files are JSON in `$MLAB_DCA_DIR` (default `$MLAB_DATA_DIR/plans/dca`), never in the repo.

    mlab dca init world --symbol SPY --tranche 500:2027-01-04 --tranche 500@480:2027-03-01
    mlab dca status world            # what is due, which limits probably filled
    mlab dca mark world t2 --filled --price 480 --date 2027-02-10

Tranche rules:
- `limit` only: rests until the price trades through it.
- `due` only: buy at market on that date (weekends roll to the next weekday).
- both: the limit rests until `due`; if still unfilled then, buy at market and cancel the limit.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

DCA_DIR = Path(os.environ.get("MLAB_DCA_DIR", DATA_DIR / "plans" / "dca"))
TRANCHE_SPEC = re.compile(r"^(?P<amount>\d+(?:\.\d+)?)(?:@(?P<limit>\d+(?:\.\d+)?))?(?::(?P<due>\d{4}-\d{2}-\d{2}))?$")


@dataclass
class Tranche:
    id: str
    amount: float
    limit: float | None = None
    due: str | None = None            # ISO date
    status: str = "pending"           # pending | filled | cancelled
    fill_price: float | None = None
    fill_date: str | None = None
    note: str = ""


@dataclass
class Plan:
    name: str
    symbol: str
    start: str                        # ISO date the plan (and its resting limits) began
    venue: str = ""
    account: str = ""
    holidays: list[str] = field(default_factory=list)   # extra non-trading dates for due-date rolls
    tranches: list[Tranche] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        d = dict(d)
        d["tranches"] = [Tranche(**t) for t in d.get("tranches", [])]
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def tranche(self, tid: str) -> Tranche:
        for t in self.tranches:
            if t.id == tid:
                return t
        raise KeyError(f"no tranche '{tid}' in plan '{self.name}' ({', '.join(t.id for t in self.tranches)})")


# ---- storage --------------------------------------------------------------------------
def plan_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("plan name: letters, digits, dot, dash, underscore only")
    return DCA_DIR / f"{name}.json"


def load(name: str) -> Plan:
    p = plan_path(name)
    if not p.exists():
        raise FileNotFoundError(f"no plan '{name}' at {p}; create it with `mlab dca init`")
    return Plan.from_dict(json.loads(p.read_text()))


def save(plan: Plan) -> Path:
    p = plan_path(plan.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    return p


def parse_tranche(spec: str, idx: int) -> Tranche:
    """'1000' | '1000@480' | '1000:2027-01-04' | '1000@480:2027-01-04'."""
    m = TRANCHE_SPEC.match(spec.strip())
    if not m:
        raise ValueError(f"bad tranche '{spec}': use AMOUNT[@LIMIT][:YYYY-MM-DD]")
    if not m["limit"] and not m["due"]:
        raise ValueError(f"tranche '{spec}' needs a limit, a due date, or both")
    if m["due"]:
        dt.date.fromisoformat(m["due"])
    return Tranche(id=f"t{idx}", amount=float(m["amount"]),
                   limit=float(m["limit"]) if m["limit"] else None, due=m["due"])


# ---- logic ----------------------------------------------------------------------------
def trading_day(d: dt.date, holidays: list[str] | None = None) -> dt.date:
    """Roll a date forward past weekends and listed holidays."""
    hol = set(holidays or [])
    while d.weekday() >= 5 or d.isoformat() in hol:
        d += dt.timedelta(days=1)
    return d


def _bar_dates(prices: pd.DataFrame) -> pd.Series:
    idx = prices.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    return pd.Series(pd.DatetimeIndex(idx).date, index=prices.index)


def evaluate(plan: Plan, prices: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """One row per tranche: state, the action for today, and the evidence behind it.

    States: filled / cancelled (recorded), limit_touched (daily low traded at or below the limit
    since the plan start: probably filled, confirm on the platform), due (market buy today),
    overdue (due date passed, still pending), waiting.
    """
    start = dt.date.fromisoformat(plan.start)
    dates = _bar_dates(prices)
    window = prices[(dates >= start).values & (dates <= today).values]
    last = float(prices["close"].iloc[-1]) if len(prices) else math.nan
    rows = []
    for t in plan.tranches:
        row = {"id": t.id, "amount": t.amount, "limit": t.limit, "due": t.due, "state": t.status,
               "action": "", "evidence": "", "est_shares": math.nan}
        if t.status in ("filled", "cancelled"):
            if t.status == "filled":
                row["evidence"] = f"filled {t.fill_date or '?'} at {t.fill_price if t.fill_price is not None else '?'}"
            rows.append(row)
            continue
        due = trading_day(dt.date.fromisoformat(t.due), plan.holidays) if t.due else None
        touched = window[window["low"] <= t.limit] if t.limit is not None and len(window) else window.iloc[0:0]
        if t.limit is not None and len(touched):
            row.update(state="limit_touched", action="confirm the fill on the platform, then `mlab dca mark`",
                       evidence=f"low {float(touched['low'].iloc[0]):.2f} <= {t.limit:.2f} on {_bar_dates(touched).iloc[0]}")
            row["est_shares"] = math.floor(t.amount / t.limit)
        elif due is not None and today >= due:
            row["state"] = "due" if today == due else "overdue"
            row["action"] = f"buy {t.amount:,.0f} at market" + (" and cancel the resting limit" if t.limit else "")
            row["evidence"] = f"due {due}" + (f", last close {last:.2f}" if not math.isnan(last) else "")
            row["est_shares"] = math.floor(t.amount / last) if last and not math.isnan(last) else math.nan
        else:
            row["state"] = "waiting"
            bits = []
            if t.limit is not None and not math.isnan(last):
                bits.append(f"limit {(t.limit / last - 1) * 100:+.1f}% from last close {last:.2f}")
                row["est_shares"] = math.floor(t.amount / t.limit)
            if due is not None:
                bits.append(f"{(due - today).days} days to {due}")
            row["evidence"] = "; ".join(bits)
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def summary(plan: Plan, table: pd.DataFrame) -> dict:
    total = sum(t.amount for t in plan.tranches)
    filled = sum(t.amount for t in plan.tranches if t.status == "filled")
    actions = table[table["state"].isin(["due", "overdue", "limit_touched"])]
    return {"plan": plan.name, "symbol": plan.symbol, "venue": plan.venue, "account": plan.account,
            "total": total, "filled": filled, "pending": total - filled - sum(t.amount for t in plan.tranches if t.status == "cancelled"),
            "actions_today": int(len(actions))}


# ---- CLI ------------------------------------------------------------------------------
def cmd_dca(a):
    from .cli import show
    if a.action == "list":
        names = sorted(p.stem for p in DCA_DIR.glob("*.json")) if DCA_DIR.exists() else []
        print("\n".join(names) if names else f"(no plans in {DCA_DIR})")
        return
    if not a.name:
        raise SystemExit("plan name required")
    if a.action == "init":
        if plan_path(a.name).exists() and not a.force:
            raise SystemExit(f"plan '{a.name}' exists; use --force to overwrite")
        if not a.symbol or not a.tranche:
            raise SystemExit("init needs --symbol and at least one --tranche")
        plan = Plan(name=a.name, symbol=a.symbol, start=a.start or dt.date.today().isoformat(),
                    venue=a.venue or "", account=a.account or "", holidays=a.holiday or [],
                    tranches=[parse_tranche(s, i + 1) for i, s in enumerate(a.tranche)])
        print(f"saved {save(plan)}")
        return
    plan = load(a.name)
    if a.action == "mark":
        if not a.tranche_id:
            raise SystemExit("mark needs a tranche id, e.g. t2")
        t = plan.tranche(a.tranche_id)
        if a.filled:
            t.status, t.fill_price, t.fill_date = "filled", a.price, a.date or dt.date.today().isoformat()
        elif a.cancelled:
            t.status = "cancelled"
        else:
            t.status, t.fill_price, t.fill_date = "pending", None, None
        if a.note:
            t.note = a.note
        save(plan)
        print(f"{plan.name}/{t.id}: {t.status}")
        return
    # status
    from .data import get_prices
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    start = dt.date.fromisoformat(plan.start)
    prices = get_prices(plan.symbol, "1d", start=(start - dt.timedelta(days=10)).isoformat(), period=None, refresh=a.refresh)
    table = evaluate(plan, prices, today)
    info = summary(plan, table)
    info["source"] = prices.attrs.get("source", "?")
    info["fetched_at"] = prices.attrs.get("fetched_at", "?")
    info["last_bar"] = str(prices.index[-1]) if len(prices) else "?"
    info["last_close"] = float(prices["close"].iloc[-1]) if len(prices) else math.nan
    info["as_of"] = today.isoformat()
    if getattr(a, "json", False):
        recs = table.reset_index().astype(object).where(table.reset_index().notna(), None).to_dict(orient="records")
        info = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in info.items()}
        print(json.dumps({**info, "tranches": recs}, default=str))
        return
    show(info, title=f"DCA plan {plan.name} ({plan.symbol})")
    show(table, title="Tranches")
    print("\n_No orders are placed by mlab: act on the platform, then `mlab dca mark`._")


def register(add):
    q = add("dca", cmd_dca, "scheduled-buy plan for API-less accounts (ISA/share dealing): init | status | mark | list")
    q.add_argument("action", choices=["init", "status", "mark", "list"])
    q.add_argument("name", nargs="?"); q.add_argument("tranche_id", nargs="?")
    q.add_argument("--symbol"); q.add_argument("--tranche", action="append", help="AMOUNT[@LIMIT][:YYYY-MM-DD], repeatable")
    q.add_argument("--start"); q.add_argument("--venue"); q.add_argument("--account")
    q.add_argument("--holiday", action="append", help="extra non-trading date YYYY-MM-DD, repeatable")
    q.add_argument("--force", action="store_true")
    q.add_argument("--filled", action="store_true"); q.add_argument("--cancelled", action="store_true")
    q.add_argument("--price", type=float); q.add_argument("--date"); q.add_argument("--note")
    q.add_argument("--today", help="evaluate as of this date (YYYY-MM-DD)"); q.add_argument("--refresh", action="store_true")
