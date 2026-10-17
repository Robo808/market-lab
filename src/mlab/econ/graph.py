"""Load the YAML dictionary in econ/data/ into a typed graph and walk it.

Node ids carry their kind as a prefix:

    ind:<key>   industry (industries.yaml)          sig:<key>  indicator (indicators.yaml)
    kpi:<key>   unit-economics KPI                  src:<key>  data source (sources.yaml)
    mkt:<key>   tradable proxy (instruments.yaml)   cls:<system>:<code>  classification code

Edges are explicit (edges.yaml) or generated from the node files: child_of, measures, published_by,
originates_from, applies_to, proxies, discloses, classified_as, maps_to.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

DATA = Path(__file__).with_name("data")
FILES = {"industries": "ind", "indicators": "sig", "unit_economics": "kpi", "sources": "src", "instruments": "mkt"}
KINDS = {v: k for k, v in FILES.items()} | {"cls": "classifications"}
EXPLICIT_TYPES = {"drives", "leads", "input_to"}
LAYERS = {"macro", "meso", "micro"}
CYCLES = {"leading", "coincident", "lagging", "target"}
FREQS = {"D", "W", "M", "Q", "A"}
TRANSFORMS = {"level", "yoy", "mom", "diff", "log", "z"}
CLASS_FIELDS = ("naics", "isic", "uksic", "coicop", "hs")
HUBS = {"ind:economy"}  # the root links everything; sideways walks never pass through it

# ISIC Rev.4 / NACE Rev.2 / UK SIC 2007 share these division -> section ranges.
_SECTIONS = [("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35), ("E", 36, 39), ("F", 41, 43), ("G", 45, 47),
             ("H", 49, 53), ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68), ("M", 69, 75), ("N", 77, 82),
             ("O", 84, 84), ("P", 85, 85), ("Q", 86, 88), ("R", 90, 93), ("S", 94, 96), ("T", 97, 98), ("U", 99, 99)]
_NAICS_SECTOR = {"31": "31-33", "32": "31-33", "33": "31-33", "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}


def section_of(code: str) -> str | None:
    """ISIC / UK SIC section letter for a division or class code ("47.52" -> "G"); letters pass through."""
    if code.isalpha():
        return code
    div = int(code.split(".")[0][:2])
    return next((s for s, lo, hi in _SECTIONS if lo <= div <= hi), None)


def naics_sector(code: str) -> str:
    return code if "-" in code else _NAICS_SECTOR.get(code[:2], code[:2])


@dataclass
class Node:
    id: str
    kind: str
    data: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.id.split(":", 1)[1]

    @property
    def name(self) -> str:
        return self.data.get("name", self.key)


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    attrs: dict = field(default_factory=dict)


class EconGraph:
    def __init__(self, nodes: dict[str, Node], edges: list[Edge], problems: list[str]):
        self.nodes, self.edges, self.problems = nodes, edges, problems
        self.out, self.inn = defaultdict(list), defaultdict(list)
        for e in edges:
            self.out[e.src].append(e)
            self.inn[e.dst].append(e)

    # ---- lookup ----------------------------------------------------------------------------------
    def resolve(self, name: str) -> str:
        """Full id, bare key (unique across kinds), or a unique substring of a key or name."""
        if name in self.nodes:
            return name
        hits = [n for n in self.nodes.values() if n.key == name]
        if len(hits) == 1:
            return hits[0].id
        low = name.lower()
        hits = hits or [n for n in self.nodes.values() if low in n.key.lower() or low in n.name.lower()]
        if len(hits) == 1:
            return hits[0].id
        if not hits:
            raise KeyError(f"no node matches {name!r}")
        raise KeyError(f"{name!r} is ambiguous: " + ", ".join(sorted(h.id for h in hits)[:12]))

    def search(self, text: str, kind: str | None = None) -> list[Node]:
        low = text.lower()
        out = []
        for n in self.nodes.values():
            if kind and n.kind != kind:
                continue
            blob = " ".join(str(v) for v in [n.id, n.name, n.data.get("themes", ""), n.data.get("note", ""),
                                             n.data.get("drivers", "")]).lower()
            if low in blob:
                out.append(n)
        return out

    def of_kind(self, kind: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]

    # ---- traversal -------------------------------------------------------------------------------
    def neighbors(self, nid: str, types=None, direction: str = "both"):
        """(edge, other node id, 'out'|'in') for every edge touching nid."""
        res = []
        if direction in ("out", "both"):
            res += [(e, e.dst, "out") for e in self.out[nid] if not types or e.type in types]
        if direction in ("in", "both"):
            res += [(e, e.src, "in") for e in self.inn[nid] if not types or e.type in types]
        return res

    def walk(self, start: str, depth: int = 2, types=None, direction: str = "both", kinds=None):
        """Breadth-first walk: list of (depth, edge, node id, direction) in discovery order."""
        start = self.resolve(start)
        seen, q, out = {start}, deque([(start, 0)]), []
        while q:
            nid, d = q.popleft()
            if d >= depth:
                continue
            for e, other, how in self.neighbors(nid, types, direction):
                if other in seen:
                    continue
                seen.add(other)
                if not kinds or self.nodes[other].kind in kinds:
                    out.append((d + 1, e, other, how))
                q.append((other, d + 1))
        return out

    def path(self, a: str, b: str, types=None) -> list[Edge]:
        """Shortest undirected path from a to b as a list of edges ([] when a == b, None when unreachable)."""
        a, b = self.resolve(a), self.resolve(b)
        prev, q = {a: None}, deque([a])
        while q:
            nid = q.popleft()
            if nid == b:
                break
            for e, other, _ in self.neighbors(nid, types):
                if other not in prev:
                    prev[other] = (nid, e)
                    q.append(other)
        if b not in prev:
            return None
        out, cur = [], b
        while prev[cur]:
            cur, e = prev[cur]
            out.append(e)
        return out[::-1]

    def sideways(self, start: str, hops: int | None = None, limit: int = 25):
        """Peers of the same kind reached through shared neighbours: alternative sources for the same
        numbers, sibling industries, indicators of the same industry from other publishers.

        Ranked by random-walk weight (each step splits the weight over the node's edges), so links through
        hubs like ind:economy count for little and a shared indicator or niche industry counts for a lot.
        `via` names the shared middle nodes. Sources default to 4 hops (src-sig-ind-sig-src)."""
        start = self.resolve(start)
        kind = self.nodes[start].kind
        hops = hops or (4 if kind == "src" else 2)
        skip = {"classified_as", "maps_to"} if kind != "cls" else set()

        def links(nid):
            return [(e, o) for e, o, _ in self.neighbors(nid) if e.type not in skip and (o not in HUBS or o == start)]

        frontier = {start: (1.0, ())}  # node -> (weight, path of intermediate nodes)
        seen, score, via = {start}, Counter(), defaultdict(Counter)
        for _ in range(hops):
            nxt: dict[str, list] = {}
            for nid, (w, trail) in frontier.items():
                ls = links(nid)
                for _, other in ls:
                    if other in seen:
                        continue
                    cur = nxt.setdefault(other, [0.0, trail + (other,)])
                    cur[0] += w / len(ls)
            seen |= set(nxt)
            frontier = {}
            for nid, (w, trail) in nxt.items():
                if self.nodes[nid].kind == kind:  # a peer: score it, don't walk through it
                    score[nid] += w
                    mid = trail[:-1]
                    if mid:
                        via[nid][mid[(len(mid) - 1) // 2 if len(mid) > 1 else 0]] += w
                else:
                    frontier[nid] = (w, trail)
        ranked = score.most_common(limit)
        top = max((w for _, w in ranked), default=1.0)
        return [(nid, round(w / top, 3), [m for m, _ in via[nid].most_common(6)]) for nid, w in ranked]

    # ---- sources and series ----------------------------------------------------------------------
    def indicators_near(self, start: str, depth: int = 1) -> list[str]:
        start = self.resolve(start)
        ids = [start] if self.nodes[start].kind == "sig" else []
        skip = {"published_by", "originates_from", "classified_as", "maps_to"}
        types = {e.type for e in self.edges} - skip
        ids += [nid for _, _, nid, _ in self.walk(start, depth, types=types) if self.nodes[nid].kind == "sig"]
        return ids

    def series_for(self, start: str, depth: int = 1):
        """(indicator id, series dict, source node) for every series of every indicator near start."""
        out = []
        for sid in self.indicators_near(start, depth):
            for s in self.nodes[sid].data.get("series", []):
                out.append((sid, s, self.nodes.get(f"src:{s['source']}")))
        return out

    def sources_for(self, start: str, depth: int = 1) -> dict[str, list[str]]:
        """Source id -> indicator ids it covers, for the indicators near start."""
        res = defaultdict(list)
        for sid, s, src in self.series_for(start, depth):
            res[f"src:{s['source']}"].append(sid)
            if s.get("via") and f"src:{s['via']}" in self.nodes:
                res[f"src:{s['via']}"].append(sid)
        return {k: sorted(set(v)) for k, v in res.items()}

    # ---- export ----------------------------------------------------------------------------------
    def frames(self):
        """(nodes, edges) DataFrames, ready for the lead-lag network in the influence map."""
        import pandas as pd
        nodes = pd.DataFrame([{"id": n.id, "kind": n.kind, "name": n.name,
                               **{k: _flat(n.data.get(k)) for k in ("layer", "geo", "freq", "cycle", "themes", "parent",
                                                                     "access", "storage_allowed", "ticker")}}
                              for n in self.nodes.values()])
        edges = pd.DataFrame([{"src": e.src, "dst": e.dst, "type": e.type,
                               **{k: _flat(e.attrs.get(k)) for k in ("sign", "lag", "evidence", "cite", "series", "note")}}
                              for e in self.edges])
        return nodes, edges

    def summary(self) -> dict:
        return {"nodes": dict(Counter(n.kind for n in self.nodes.values())),
                "edges": dict(Counter(e.type for e in self.edges)), "problems": len(self.problems)}


def _flat(v):
    if isinstance(v, (list, tuple)):
        return ",".join(map(str, v))
    return v


def _read(name: str, data_dir: Path) -> dict:
    with open(data_dir / f"{name}.yaml") as f:
        return yaml.safe_load(f) or {}


def build(data_dir: Path = DATA) -> EconGraph:
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    problems: list[str] = []
    raw = {name: _read(name, data_dir) for name in FILES}
    for name, prefix in FILES.items():
        for key, d in raw[name].items():
            nodes[f"{prefix}:{key}"] = Node(f"{prefix}:{key}", prefix, d or {})

    cls = _read("classifications", data_dir)
    for system, codes in (cls.get("codes") or {}).items():
        for code, title in codes.items():
            nodes[f"cls:{system}:{code}"] = Node(f"cls:{system}:{code}", "cls", {"name": title, "system": system})

    def cls_node(system: str, code: str) -> str:
        nid = f"cls:{system}:{code}"
        if nid not in nodes:
            nodes[nid] = Node(nid, "cls", {"name": f"{system.upper()} {code}", "system": system})
            parent = None
            if system == "naics" and len(code) > 2 and "-" not in code:
                parent = naics_sector(code)
            elif system in ("isic", "uksic") and not code.isalpha():
                parent = section_of(code)
            elif system == "hs" and len(code) > 4:
                parent = code[:4]
            if parent and parent != code:
                edges.append(Edge(nid, cls_node(system, parent), "child_of"))
        return nid

    def add(src, dst, typ, **attrs):
        if dst not in nodes:
            problems.append(f"{src}: {typ} -> unknown node {dst}")
            return
        edges.append(Edge(src, dst, typ, attrs))

    for nid, n in list(nodes.items()):
        d = n.data
        if n.kind == "ind":
            if d.get("parent"):
                add(nid, f"ind:{d['parent']}", "child_of")
            for system in CLASS_FIELDS:
                for code in d.get(system) or []:
                    edges.append(Edge(nid, cls_node(system, str(code)), "classified_as"))
        elif n.kind == "sig":
            for k in ("layer", "cycle", "freq"):
                allowed = {"layer": LAYERS, "cycle": CYCLES, "freq": FREQS}[k]
                if d.get(k) not in allowed:
                    problems.append(f"{nid}: {k}={d.get(k)!r} not in {sorted(allowed)}")
            for t in d.get("transform") or []:
                if t not in TRANSFORMS:
                    problems.append(f"{nid}: unknown transform {t!r}")
            for ind in d.get("industries") or []:
                add(nid, f"ind:{ind}", "measures")
            if not d.get("series"):
                problems.append(f"{nid}: no series")
            for s in d.get("series") or []:
                add(nid, f"src:{s.get('source')}", "published_by", series=s.get("id"), via=s.get("via"))
                if s.get("via"):
                    add(nid, f"src:{s['via']}", "originates_from", series=s.get("id"))
        elif n.kind == "kpi":
            for ind in d.get("applies_to") or []:
                add(nid, f"ind:{ind}", "applies_to")
        elif n.kind == "mkt":
            for ind in d.get("industries") or []:
                add(nid, f"ind:{ind}", "proxies")
            for k in d.get("discloses") or []:
                add(nid, f"kpi:{k}", "discloses")
        elif n.kind == "src":
            for k in ("access", "commercial_use", "storage_allowed", "licence"):
                if d.get(k) in (None, ""):
                    problems.append(f"{nid}: missing {k}")

    for a, sections in ((cls.get("crosswalk") or {}).get("naics_to_isic") or {}).items():
        for s in sections:
            edges.append(Edge(cls_node("naics", str(a)), cls_node("isic", s), "maps_to"))
    for code in list(nodes):
        if code.startswith("cls:uksic:") and code.split(":")[2].isalpha():
            edges.append(Edge(code, cls_node("isic", code.split(":")[2]), "maps_to"))

    for e in _read("edges", data_dir).get("edges") or []:
        e = dict(e)
        src, dst, typ = e.pop("from"), e.pop("to"), e.pop("type")
        if typ not in EXPLICIT_TYPES:
            problems.append(f"edge {src}->{dst}: unknown type {typ}")
        if src not in nodes:
            problems.append(f"edge {src}->{dst}: unknown node {src}")
            continue
        if typ != "input_to" and e.get("evidence") not in ("literature", "prior", "tested"):
            problems.append(f"edge {src}->{dst}: evidence must be literature, prior or tested")
        if e.get("evidence") == "literature" and not e.get("cite"):
            problems.append(f"edge {src}->{dst}: literature edge without cite")
        add(src, dst, typ, **e)

    return EconGraph(nodes, edges, problems)


@lru_cache(maxsize=1)
def load() -> EconGraph:
    return build()
