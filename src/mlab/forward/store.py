"""Append-only, write-once event store for the forward lab.

Layout under $MLAB_DATA_DIR/forward (override MLAB_FORWARD_DIR):
    registry/<registry_sha>.json            frozen registry copies (write-once)
    events/<kind>/<YYYY-MM>/<run_id>.jsonl.gz   one file per run and kind (write-once)
    chains/<YYYY-MM>/<run_id>.parquet       option chain snapshots near the money (write-once)
    state/<run_id>.json.gz                  book state after a run (write-once); the newest is the live one
    RUNNING                                 lock while a run is in progress

Files are never modified after they are written. Each write is read back and its SHA-256 checked, because
the shared folder can fail silently when full (LESSONS L-015). Concurrent runs are refused by the lock.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, env
from . import SCHEMA_VERSION

KINDS = {
    "signals": "one row per cell per bar: target, held position, equity, P&L, context",
    "triggers": "every entry, exit or flip a cell's rule produced, with the desk filters' verdict (taken or skipped)",
    "fills": "simulated fills at the next bar's open (or option bid/ask), with cost and its source",
    "greeks": "per option book and leg: price, IV, delta, gamma, theta, vega and the day's P&L attribution",
    "vol": "per options underlying: spot, ATM IV30, 25-delta skew, term slope, realised vol, IV/RV",
    "runs": "one row per run: code commit, registry hash, bars processed, per-item status",
}


def root() -> Path:
    return Path(env("MLAB_FORWARD_DIR") or DATA_DIR / "forward")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_once(path: Path, data: bytes) -> str:
    """Write bytes to a new file; refuse to overwrite; read back and verify. Returns the SHA-256."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"write-once file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    want = sha256_bytes(data)
    got = sha256_bytes(path.read_bytes())
    if got != want:
        raise OSError(f"read-back mismatch on {path} (folder full? LESSONS L-015)")
    return want


def _rows_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write(json.dumps(r, default=str, separators=(",", ":")).encode() + b"\n")
    return buf.getvalue()


def append_events(kind: str, run_id: str, rows: list[dict], month: str | None = None) -> Path | None:
    """Write one run's rows of a kind as a new gzip JSONL file. Each row gets schema and kind stamps."""
    if kind not in KINDS:
        raise ValueError(f"unknown event kind {kind!r}; known: {', '.join(KINDS)}")
    if not rows:
        return None
    stamped = [{"schema": f"fwd.{kind}/{SCHEMA_VERSION}", **r} for r in rows]
    month = month or run_id[:7]
    p = root() / "events" / kind / month / f"{run_id}.jsonl.gz"
    write_once(p, _rows_bytes(stamped))
    return p


def read_events(kind: str, since: str | None = None) -> pd.DataFrame:
    """All rows of a kind (optionally from runs on or after `since`, YYYY-MM-DD) as a DataFrame."""
    base = root() / "events" / kind
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(base.glob("*/*.jsonl.gz")):
        if since and p.name[:10] < since:
            continue
        with gzip.open(p, "rt") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return pd.DataFrame(rows)


def save_state(run_id: str, state: dict) -> Path:
    p = root() / "state" / f"{run_id}.json.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(state, default=str).encode())
    write_once(p, buf.getvalue())
    return p


def load_state() -> tuple[dict, str | None]:
    """Newest state snapshot (run ids sort by time) or an empty state."""
    d = root() / "state"
    files = sorted(d.glob("*.json.gz")) if d.exists() else []
    if not files:
        return {"cells": {}, "options": {}, "frozen": {}}, None
    with gzip.open(files[-1], "rt") as fh:
        return json.load(fh), files[-1].name.split(".")[0]


def save_chain(run_id: str, df: pd.DataFrame) -> Path | None:
    if df is None or df.empty:
        return None
    p = root() / "chains" / run_id[:7] / f"{run_id}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    write_once(p, buf.getvalue())
    return p


def freeze_registry(registry: dict) -> tuple[str, Path]:
    """Store a write-once copy of the registry under its own hash; returns (sha, path)."""
    data = canonical(registry)
    sha = sha256_bytes(data)
    p = root() / "registry" / f"{sha}.json"
    if not p.exists():
        write_once(p, data)
    return sha, p


class RunLock:
    """Refuses a second concurrent run. A lock older than `stale_h` hours is taken over (crashed run)."""

    def __init__(self, run_id: str, stale_h: float = 3.0):
        self.path, self.run_id, self.stale_h = root() / "RUNNING", run_id, stale_h

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age_h = (datetime.now().timestamp() - self.path.stat().st_mtime) / 3600
            if age_h < self.stale_h:
                raise RuntimeError(f"another forward run holds the lock ({self.path.read_text().strip()}, "
                                   f"{age_h:.1f}h old)")
        self.path.write_text(f"{self.run_id} {now_iso()}\n")
        return self

    def __exit__(self, *exc):
        try:
            if self.path.read_text().split()[0] == self.run_id:
                self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def usage() -> dict:
    """Bytes used per top-level folder of the forward store (the shared folder has a 10 GB cap)."""
    out = {}
    r = root()
    for sub in ("registry", "events", "chains", "state"):
        d = r / sub
        out[sub] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0
    out["total"] = sum(out.values())
    return out
