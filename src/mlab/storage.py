"""Safe writes for desk state in the shared folder.

The shared folder is written by several threads at once, each in its own container, through a FUSE
mount of an object store. Two rules follow (see docs/wiki/Design-Standards.md, "Concurrency"):

1. Never leave a half-written file: write to a unique temp file in the same directory, then rename.
2. Never read-modify-write a file that another container may also rewrite. A lock only coordinates
   processes in one container, so state that must not lose an update is stored as write-once event
   files (one new file per change, unique name) and folded on read.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

try:  # POSIX advisory locks; Windows falls back to no lock (single-user desktop use)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

log = logging.getLogger(__name__)


def atomic_write(path: Path, write: Callable[[Path], None]) -> Path:
    """Call write(tmp) on a uniquely named temp file next to `path`, then rename it over `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write(path, lambda t: t.write_text(text, encoding="utf-8"))


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> Path:
    return atomic_write(path, lambda t: df.to_parquet(t))


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock on `<path>.lock` for a read-modify-write within this container."""
    if fcntl is None:
        yield
        return
    lock = Path(f"{path}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def event_name(now: pd.Timestamp | None = None) -> str:
    """Sortable, collision-free file stem: UTC time to the microsecond plus a random suffix."""
    now = now or pd.Timestamp.now(tz="UTC")
    return f"{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}"


def append_event(directory: Path, event: dict) -> Path:
    """Write one event as its own new JSON file. Safe across containers: nothing is ever overwritten."""
    p = Path(directory) / f"{event_name()}.json"
    return atomic_write_text(p, json.dumps(event, default=str))


def read_events(directory: Path) -> list[dict]:
    """All events in name (= time) order. Unreadable files (an upload still in flight) are skipped."""
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            log.warning("skipping unreadable event %s: %s", p.name, e)
    return out
