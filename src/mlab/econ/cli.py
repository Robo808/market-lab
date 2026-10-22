"""`mlab econ`: browse and traverse the indicator dictionary and industry graph.

    mlab econ summary                          # node and edge counts, validation problems
    mlab econ search gnome                     # find nodes by id, name, theme or note
    mlab econ show garden_ornaments            # one node with every edge in and out
    mlab econ walk ind:pallet_networks -d 2    # breadth-first neighbourhood (filter --types, --kind)
    mlab econ sideways src:ons                 # alternative sources / sibling nodes of the same kind
    mlab econ path sig:uk_mortgage_approvals ind:retail_tiles_flooring
    mlab econ sources ind:retail_bathroom_online -d 2   # who publishes the numbers, access, licence
    mlab econ check                            # live-check every FRED and ONS series id
    mlab econ panel ind:retail_home_improvement -d 2 --geo US   # ML feature matrix (in memory)
    mlab econ export --out DIR                 # nodes.csv + edges.csv for the influence-map network
    mlab econ docs --out docs/wiki/Econ-Dictionary.md
    mlab econ supply-demand                    # supply/demand category map and where each stands now
    mlab econ sd-study --split train           # categories vs industry returns: explain, predict, rotation
"""
from __future__ import annotations

import pandas as pd

from .graph import load


def _edge_row(g, e, other, how):
    n = g.nodes[other]
    return {"dir": "->" if how == "out" else "<-", "type": e.type, "node": other, "name": n.name[:70],
            "sign": e.attrs.get("sign", ""), "lag_m": e.attrs.get("lag", ""), "evidence": e.attrs.get("evidence", ""),
            "series": e.attrs.get("series", "")}


def cmd_econ(a):
    from ..cli import show
    act = a.action
    if act in ("supply-demand", "sd-study"):
        _supply_demand(a)
        return
    g = load()
    if act == "summary":
        s = g.summary()
        show(pd.DataFrame({"count": s["nodes"]}), a.json, title="Nodes by kind")
        show(pd.DataFrame({"count": s["edges"]}), a.json, title="Edges by type")
        if g.problems:
            show(pd.DataFrame({"problem": g.problems}), a.json, title="Validation problems")
        else:
            print("\nValidation: no problems.")
        return
    if act == "list":
        rows = [{"id": n.id, "name": n.name[:80], **({"geo": n.data.get("geo"), "freq": n.data.get("freq"),
                 "cycle": n.data.get("cycle")} if n.kind == "sig" else {})}
                for n in g.nodes.values() if not a.kind or n.kind == a.kind]
        show(pd.DataFrame(rows).set_index("id"), a.json, title=f"{len(rows)} nodes")
        return
    if act == "search":
        hits = g.search(a.node, a.kind)
        show(pd.DataFrame([{"id": n.id, "name": n.name[:90]} for n in hits]).set_index("id") if hits else pd.DataFrame(),
             a.json, title=f"Matches for {a.node!r}")
        return
    if act == "check":
        from .fetch import check
        df = check(g, log=(lambda *x: None) if a.json else print)
        bad = df[~df["ok"].astype(bool)]
        print(f"\n{int(df['ok'].sum())}/{len(df)} series ids returned data or metadata "
              f"({pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC).")
        if len(bad):
            show(bad.set_index("indicator"), a.json, title="Failed")
        return
    if act == "export":
        import os
        nodes, edges = g.frames()
        os.makedirs(a.out, exist_ok=True)
        nodes.to_csv(f"{a.out}/nodes.csv", index=False)
        edges.to_csv(f"{a.out}/edges.csv", index=False)
        print(f"wrote {len(nodes)} nodes and {len(edges)} edges to {a.out}/")
        return
    if act == "docs":
        from .docs import render
        md = render(g)
        if a.out:
            with open(a.out, "w") as f:
                f.write(md)
            print(f"wrote {a.out} ({len(md.splitlines())} lines)")
        else:
            print(md)
        return

    if not a.node:
        raise SystemExit(f"econ {act} needs a node (id, key or unique part of a name)")
    nid = g.resolve(a.node)
    n = g.nodes[nid]
    types = set(a.types.split(",")) if a.types else None
    if act == "show":
        info = {k: v for k, v in n.data.items() if not isinstance(v, (list, dict))}
        for k, v in n.data.items():
            if isinstance(v, list) and v and not isinstance(v[0], dict):
                info[k] = ", ".join(map(str, v))
        show({"id": nid, **info}, a.json, title=n.name)
        if n.data.get("series"):
            show(pd.DataFrame(n.data["series"]), a.json, title="Series")
        rows = [_edge_row(g, e, o, h) for e, o, h in g.neighbors(nid, types)]
        show(pd.DataFrame(rows), a.json, title=f"{len(rows)} edges")
    elif act == "walk":
        rows = [{"depth": d, **_edge_row(g, e, o, h)} for d, e, o, h in
                g.walk(nid, a.depth, types, a.direction, {a.kind} if a.kind else None)]
        show(pd.DataFrame(rows), a.json, title=f"Walk from {nid}, depth {a.depth}")
    elif act == "sideways":
        rows = [{"peer": p, "name": g.nodes[p].name[:70], "score": k, "via": ", ".join(v[:6])}
                for p, k, v in g.sideways(nid, a.hops, a.limit)]
        show(pd.DataFrame(rows), a.json, title=f"Sideways from {nid} (same kind, ranked by random-walk weight, top = 1)")
    elif act == "path":
        if not a.target:
            raise SystemExit("econ path needs two nodes")
        p = g.path(nid, a.target, types)
        if p is None:
            print("no path")
            return
        show(pd.DataFrame([{"from": e.src, "type": e.type, "to": e.dst, "sign": e.attrs.get("sign", ""),
                            "evidence": e.attrs.get("evidence", "")} for e in p]), a.json, title=f"{nid} to {g.resolve(a.target)}")
    elif act == "sources":
        rows = []
        for sid, inds in g.sources_for(nid, a.depth).items():
            s = g.nodes.get(sid)
            d = s.data if s else {}
            rows.append({"source": sid, "access": d.get("access"), "cost": str(d.get("cost") or "")[:50],
                         "commercial": d.get("commercial_use"), "storage": d.get("storage_allowed"),
                         "verified": d.get("verified"), "licence": str(d.get("licence") or "")[:60],
                         "indicators": len(inds)})
        df = pd.DataFrame(rows).sort_values(["access", "source"]) if rows else pd.DataFrame()
        show(df, a.json, title=f"Sources for the numbers within {a.depth} hop(s) of {nid}")
    elif act == "panel":
        from .fetch import panel
        df, prov = panel(g, nid, a.depth, a.geo, a.since, None if a.no_target else a.target_label, a.out,
                         log=(lambda *x: None) if a.json else print)
        show(prov, a.json, title="Columns and provenance")
        if not a.json:
            print(f"\nPanel: {df.shape[0]} months x {df.shape[1]} columns, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}"
                  if len(df) else "\nPanel is empty")
            show(df.tail(a.tail).iloc[::-1], a.json, title=f"Last {a.tail} months")
        if a.out:
            print(f"\nwrote {a.out}")
        print(f"\n_Fetched live {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC; FRED data is never cached or stored._")


def _supply_demand(a):
    from ..cli import show
    from . import supply_demand as sd
    m = sd.load_map()
    raw = sd.fetch(m, refresh=a.refresh, log=(lambda *x: None) if a.json else print)
    if a.action == "supply-demand":
        show(sd.map_frame(m), a.json, title=f"{len(m['categories'])} supply/demand categories")
        show(sd.latest(raw, m), a.json, title="Latest published month: yoy growth % (s = supply volume, d = nominal "
                                                "demand, gap = d - s, acc = change over 3 months)")
    else:
        res = sd.study(raw, a.split, m, allow_test=a.allow_test)
        P = res["predict"]
        show(res["explain"].sort_values("t", key=abs, ascending=False).round(2).set_index(["category", "feature"]),
             a.json, title=f"Explain ({a.split}): 12m excess return vs same-window growth, and price leading growth")
        show(P[P["t"].abs() >= 3].round(3).set_index(["category", "feature", "h"]), a.json,
             title=f"Predict ({a.split}): {int((P['t'].abs() >= 3).sum())} of {len(P)} tests with |t| >= 3 "
                   f"(null shifts clear t >= 3 {100 * P['null_t3'].mean():.1f}% of the time)")
        show(res["xs"].round(3).set_index("feature"), a.json, title="Rotation: long top third / short bottom third, "
                                                                      "1 month, vs random rankings")
    print(f"\n_Sources: Federal Reserve G.17, Census MRTS, Ken French 49 industries; fetched {raw.asof or 'from cache'}_")


def register(add):
    q = add("econ", cmd_econ, "indicator dictionary + industry graph: summary | list | search | show | walk | sideways | "
                              "path | sources | check | panel | export | docs")
    q.add_argument("action", choices=["summary", "list", "search", "show", "walk", "sideways", "path", "sources", "check",
                                      "panel", "export", "docs", "supply-demand", "sd-study"])
    q.add_argument("node", nargs="?", help="node id (ind:/sig:/kpi:/src:/mkt:/cls:), bare key, or search text")
    q.add_argument("target", nargs="?", help="second node for `path`")
    q.add_argument("--depth", "-d", type=int, default=1)
    q.add_argument("--types", help="comma-separated edge types to follow, e.g. drives,leads,input_to")
    q.add_argument("--direction", choices=["out", "in", "both"], default="both")
    q.add_argument("--kind", choices=["ind", "sig", "kpi", "src", "mkt", "cls"], help="filter results to one node kind")
    q.add_argument("--hops", type=int, help="sideways: hops to search (default 2, sources 4)")
    q.add_argument("--limit", type=int, default=25)
    q.add_argument("--geo", help="panel: keep indicators for one geography (US, UK, ...)")
    q.add_argument("--since", default="2000-01-01")
    q.add_argument("--target-label", default="sig:us_recession_nber", help="panel: label column")
    q.add_argument("--no-target", action="store_true")
    q.add_argument("--tail", type=int, default=6)
    q.add_argument("--out", help="export: directory; docs/panel: file (panel refuses storage-restricted sources)")
    q.add_argument("--split", choices=["train", "validation", "test"], default="train", help="sd-study window")
    q.add_argument("--allow-test", action="store_true", help="sd-study: open the locked 2020+ test split")
    q.add_argument("--refresh", action="store_true", help="supply-demand / sd-study: refetch instead of the cache")
