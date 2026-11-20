"""Version history for desk state that lives outside git (journal, lessons, assumption register, paper log).

`snapshot` stores each tracked file's bytes once, content-addressed (objects/<sha[:2]>/<sha>.gz), and writes a
manifest listing every tracked file's hash. Each manifest carries the hash of the previous one, so the chain is
tamper-evident: `verify` re-hashes every object and walks the chain. Unchanged files cost nothing. `log PATH`
lists a file's versions; `show PATH SHA` prints one.

Layout: $MLAB_DATA_DIR/ledger/{objects,manifests}. Never modified in place; nothing here goes to the repo.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .config import DATA_DIR, env

TRACKED = ["journal/trades.jsonl", "journal/JOURNAL.md", "LESSONS.md", "research/assumptions.md",
           "reports/paper_book_log.md"]
TRACKED_GLOBS = ["journal/events/**/*", "paper/*.json"]


def root() -> Path:
    return Path(env("MLAB_LEDGER_DIR") or DATA_DIR / "ledger")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _files(base: Path) -> list[Path]:
    out = [base / f for f in TRACKED if (base / f).is_file()]
    for g in TRACKED_GLOBS:
        out += [p for p in base.glob(g) if p.is_file()]
    return sorted(set(out))


def _manifests() -> list[Path]:
    d = root() / "manifests"
    return sorted(d.glob("*/*.json")) if d.exists() else []


def snapshot(base: Path | None = None, note: str = "") -> dict:
    """Record the current version of every tracked file. Returns the manifest (or the last one if unchanged)."""
    from .forward.store import now_iso, write_once
    base = Path(base or DATA_DIR)
    files = {}
    for p in _files(base):
        data = p.read_bytes()
        sha = _sha(data)
        obj = root() / "objects" / sha[:2] / f"{sha}.gz"
        if not obj.exists():
            write_once(obj, gzip.compress(data, mtime=0))
        files[str(p.relative_to(base))] = {"sha": sha, "size": len(data)}
    prev = _manifests()
    prev_doc = json.loads(prev[-1].read_text()) if prev else None
    if prev_doc and prev_doc["files"] == files:
        return {**prev_doc, "unchanged": True}
    ts = now_iso()
    doc = {"schema": "ledger.manifest/1", "at": ts, "note": note, "files": files,
           "prev": _sha(prev[-1].read_bytes()) if prev else None}
    body = json.dumps(doc, indent=1, sort_keys=True).encode()
    name = f"{len(prev) + 1:08d}-" + ts.replace(":", "").replace("+00:00", "Z") + "-" + _sha(body)[:8]
    write_once(root() / "manifests" / ts[:7] / f"{name}.json", body)
    return doc


def verify() -> dict:
    """Re-hash every object and walk the manifest chain."""
    problems, prev_sha, n_obj = [], None, 0
    for m in _manifests():
        raw = m.read_bytes()
        doc = json.loads(raw)
        if doc.get("prev") != prev_sha:
            problems.append(f"chain break at {m.name}")
        prev_sha = _sha(raw)
        for path, meta in doc["files"].items():
            obj = root() / "objects" / meta["sha"][:2] / f"{meta['sha']}.gz"
            if not obj.exists():
                problems.append(f"missing object for {path} in {m.name}")
                continue
            n_obj += 1
            if _sha(gzip.decompress(obj.read_bytes())) != meta["sha"]:
                problems.append(f"corrupt object {meta['sha'][:12]} ({path})")
    return {"manifests": len(_manifests()), "objects_checked": n_obj, "problems": problems,
            "status": "ok" if not problems else "BROKEN"}


def log(path: str) -> list[dict]:
    out, last = [], None
    for m in _manifests():
        doc = json.loads(m.read_text())
        meta = doc["files"].get(path)
        if meta and meta["sha"] != last:
            out.append({"at": doc["at"], "sha": meta["sha"], "size": meta["size"], "note": doc.get("note", "")})
            last = meta["sha"]
    return out


def show(sha: str) -> bytes:
    hits = list((root() / "objects").glob(f"{sha[:2]}/{sha}*.gz"))
    if not hits:
        raise FileNotFoundError(f"no object {sha}")
    return gzip.decompress(hits[0].read_bytes())


def cmd_ledger(a):
    import pandas as pd

    from .cli import show as show_table
    if a.action == "snapshot":
        d = snapshot(note=a.note or "")
        print(("unchanged since " if d.get("unchanged") else "snapshot ") + d["at"] + f": {len(d['files'])} files")
    elif a.action == "verify":
        v = verify()
        print(f"ledger {v['status']}: {v['manifests']} manifests, {v['objects_checked']} object checks")
        for p in v["problems"]:
            print("  " + p)
    elif a.action == "log":
        show_table(pd.DataFrame(log(a.path)), a.json, title=f"History of {a.path}")
    elif a.action == "show":
        print(show(a.path).decode(errors="replace"))


def register(add):
    q = add("ledger", cmd_ledger, "desk-state history: snapshot | verify | log PATH | show SHA")
    q.add_argument("action", choices=["snapshot", "verify", "log", "show"])
    q.add_argument("path", nargs="?", help="log: file path relative to the data dir; show: object sha")
    q.add_argument("--note")
