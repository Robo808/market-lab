"""Parquet cache for time series. Incremental: new bars are merged with what is on disk.

Layout: data/cache/<provider>/<interval>/<SYMBOL>.parquet
Cached history is what makes IG's weekly data allowance go a long way.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, CACHE_MAX_MB


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def path_for(provider: str, symbol: str, interval: str) -> Path:
    return CACHE_DIR / _safe(provider) / _safe(interval) / f"{_safe(symbol)}.parquet"


def load(provider: str, symbol: str, interval: str) -> pd.DataFrame | None:
    p = path_for(provider, symbol, interval)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        os.utime(p)  # mark as recently used for pruning
        return df
    except Exception:
        return None


def save(provider: str, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
    """Merge df into the cached frame (new rows win) and write it back."""
    if df is None or df.empty:
        return df
    old = load(provider, symbol, interval)
    if old is not None and not old.empty:
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    p = path_for(provider, symbol, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.replace(p)
    prune()
    return df


def covers(df: pd.DataFrame | None, start: pd.Timestamp | None, end: pd.Timestamp | None,
           max_age: pd.Timedelta) -> bool:
    """True when the cached frame spans [start, end] and is fresh enough."""
    if df is None or df.empty:
        return False
    idx = df.index
    if start is not None and idx[0] > start + pd.Timedelta(days=4):  # weekends/holidays slack
        return False
    now = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
    target_end = min(end, now) if end is not None else now
    return idx[-1] >= target_end - max_age


def stats() -> dict:
    files = list(CACHE_DIR.rglob("*.parquet")) if CACHE_DIR.exists() else []
    size = sum(f.stat().st_size for f in files)
    return {"dir": str(CACHE_DIR), "files": len(files), "mb": round(size / 1e6, 2), "cap_mb": CACHE_MAX_MB}


def prune(max_mb: int = CACHE_MAX_MB) -> int:
    """Delete least-recently-used files until the cache is under max_mb. Returns files removed."""
    if not CACHE_DIR.exists():
        return 0
    files = sorted(CACHE_DIR.rglob("*.parquet"), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    removed = 0
    while files and total > max_mb * 1e6:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
        removed += 1
    return removed


def clear(provider: str | None = None) -> int:
    root = CACHE_DIR / _safe(provider) if provider else CACHE_DIR
    n = 0
    for f in root.rglob("*.parquet"):
        f.unlink(missing_ok=True)
        n += 1
    return n


class TTLMemo:
    """Tiny in-process memo for JSON endpoints (quotes, search) within one run."""

    def __init__(self, ttl: float = 60):
        self.ttl, self._d = ttl, {}

    def get(self, key):
        v = self._d.get(key)
        return v[1] if v and time.time() - v[0] < self.ttl else None

    def put(self, key, value):
        self._d[key] = (time.time(), value)
        return value
