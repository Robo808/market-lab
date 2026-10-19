"""`mlab newsstore`: one command over the GDELT stores (articles, political events, theme x country). Kept apart
from the store modules so they never import each other in a cycle."""
from __future__ import annotations

from . import gdelt, gdelt_events, gdelt_themes


def cmd_newsstore(a):
    from ..cli import show
    store = {"gdelt": gdelt, "events": gdelt_events, "themes": gdelt_themes}.get(a.action)
    if store is not None:
        print(store.ingest(a.since or "1d", a.until, a.workers))
        return
    if a.action == "show":
        if not a.symbol:
            raise SystemExit("symbol required")
        df = gdelt.daily(a.symbol.upper(), a.since, a.until)
        if df.empty:
            print(f"no GDELT articles stored for {a.symbol} (run `mlab newsstore gdelt --since ...` first)")
            return
        show(df.reset_index(names="day"), a.json, title=f"{a.symbol.upper()} GDELT daily · {gdelt.CITATION}")


def register(add):
    q = add("newsstore", cmd_newsstore, "point-in-time news store: gdelt (GKG articles) | events (CAMEO political "
            "events) | themes (GKG theme x country) | show SYMBOL")
    q.add_argument("action", choices=["gdelt", "events", "themes", "show"])
    q.add_argument("symbol", nargs="?")
    q.add_argument("--since", help="start: date, timestamp or relative like 3d / 12h (default 1d)")
    q.add_argument("--until", help="end (default: latest published file)")
    q.add_argument("--workers", type=int, default=4)
