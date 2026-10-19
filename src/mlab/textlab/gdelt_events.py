"""GDELT 2.0 Events (CAMEO) store: who did what to whom, where, every 15 minutes since 2015-02-19.

    mlab newsstore events --since 2015-02-19 --until 2015-12-31 --workers 8   # backfill, resumable
    mlab newsstore events --since 2d                                          # top up

The political half of the influence map (#58): conflict and cooperation between countries, protests, sanctions,
threats, aid, diplomacy. One file per UTC day, $MLAB_DATA_DIR/data/news/gdelt/events/YYYY-MM-DD.parquet, one row per
(publication bucket, actor-1 country, actor-2 country, where it happened, CAMEO root code, quad class):
- avail: publication bucket of the 15-minute files (DATEADDED): 00:00 = published before 20:00 UTC that day,
  20:00 = published from 20:00 UTC on (see bucket()); never earlier than GDELT published them;
- a1, a2: actor country (CAMEO 3-letter, e.g. USA, CHN; "" when the actor has no country, e.g. "police");
- geo: action location country (FIPS 2-letter, e.g. US, CH); root: CAMEO root code "01".."20"
  (14 protest, 16 reduce relations incl. sanctions, 17 coerce, 18 assault, 19 fight, 20 mass violence);
- quad: 1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict;
- events, root_events, articles (sum of NumArticles), goldstein and tone (article-weighted means).
Raw columns follow the GDELT 2.0 Event codebook (http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf).
GDELT is free for academic, commercial and governmental use with a citation and a link to https://www.gdeltproject.org.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import gdelt

STORE = gdelt.STORE / "events"
KEY = ["avail", "a1", "a2", "geo", "root", "quad"]
QUAD = {1: "verbal cooperation", 2: "material cooperation", 3: "verbal conflict", 4: "material conflict"}


def parse(blob: bytes, ts: pd.Timestamp) -> pd.DataFrame:
    """Aggregated rows from one zipped events export file (61 tab-separated columns)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    rows = [r for r in (line.split("\t") for line in raw.splitlines()) if len(r) == 61]
    if not rows:
        return pd.DataFrame(columns=KEY)
    a = np.array([[r[7], r[17], r[53], r[28], r[29], r[25], r[30], r[33], r[34]] for r in rows], dtype=object)
    df = pd.DataFrame(a, columns=["a1", "a2", "geo", "root", "quad", "is_root", "goldstein", "articles", "tone"])
    for c in ("goldstein", "articles", "tone"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["quad"] = pd.to_numeric(df["quad"], errors="coerce").fillna(0).astype("int8")
    df["is_root"] = df["is_root"] == "1"
    df["articles"] = df["articles"].fillna(1).clip(lower=1)
    df["gw"], df["tw"] = df["goldstein"] * df["articles"], df["tone"] * df["articles"]
    g = df.groupby(["a1", "a2", "geo", "root", "quad"], sort=False)
    out = g.agg(events=("articles", "size"), root_events=("is_root", "sum"), articles=("articles", "sum"),
                gw=("gw", "sum"), tw=("tw", "sum")).reset_index()
    out.insert(0, "avail", bucket(ts))
    return out


def bucket(ts: pd.Timestamp) -> pd.Timestamp:
    """Two buckets per UTC day: files published before the 20:00 UTC cutoff (at or before the US close all year)
    are stamped 00:00, later ones 20:00. A daily decision at the close of day t may use the 00:00 bucket of t."""
    d = ts.floor("D")
    return d if ts.hour < 20 else d + pd.Timedelta(hours=20)


def _path(day: str):
    return STORE / f"{day}.parquet"


def _done(day: str) -> set:
    """15-minute stamps already ingested for a day (kept in the file's metadata column `files`)."""
    p = _path(day)
    if not p.exists():
        return set()
    done = pd.read_parquet(p, columns=["files"])["files"].dropna()
    return set(pd.to_datetime(done.iloc[0].split(","), utc=True)) if len(done) else set()


def _fetch(ts: pd.Timestamp, s) -> pd.DataFrame | None:
    r = s.get(f"{gdelt.BASE}/{ts:%Y%m%d%H%M%S}.export.CSV.zip", timeout=60)
    if r.status_code == 404:  # missing at the source: recorded as done so it isn't retried
        return None
    r.raise_for_status()
    return parse(r.content, ts)


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    g = df.groupby(KEY, sort=False)
    return g[["events", "root_events", "articles", "gw", "tw"]].sum().reset_index()


def ingest_day(day: str, stamps_: list[pd.Timestamp], workers: int = 4, session=None, log=print) -> dict:
    from ..net import session as mk
    s = session or mk("market-lab/0.1")
    out = {"ok": 0, "missing": 0, "failed": 0, "events": 0}
    got, done = [], []

    def one(t):
        try:
            return t, _fetch(t, s), None
        except Exception as e:  # count and carry on; a rerun picks the file up again
            return t, None, f"{type(e).__name__}: {str(e)[:80]}"
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for t, res, err in ex.map(one, stamps_):
            if err:
                out["failed"] += 1
                log(f"  {t:%Y-%m-%d %H:%M} events failed {err}")
                continue
            done.append(t)
            if res is None:
                out["missing"] += 1
                continue
            out["ok"] += 1
            out["events"] += int(res["events"].sum()) if len(res) else 0
            got.append(res)
    if not done:
        return out
    p = _path(day)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    files = sorted(_done(day) | set(done))
    old = old.dropna(subset=["avail"]).drop(columns="files") if "avail" in old else pd.DataFrame()
    parts = ([old] if len(old) else []) + got
    df = _combine(parts) if any(len(x) for x in parts) else pd.DataFrame(columns=KEY + ["events", "root_events",
                                                                                         "articles", "gw", "tw"])
    df = df.sort_values(KEY).reset_index(drop=True)
    df["files"] = None
    if len(df):
        df.loc[0, "files"] = ",".join(f"{t:%Y-%m-%dT%H:%M:%SZ}" for t in files)
    else:  # keep the done-marker even for a day with no events at all
        df = pd.DataFrame({"files": [",".join(f"{t:%Y-%m-%dT%H:%M:%SZ}" for t in files)]})
    STORE.mkdir(parents=True, exist_ok=True)
    gdelt._write(df, p)
    return out


def ingest(since, until=None, workers: int = 4, log=print) -> dict:
    by_day: dict[str, list] = {}
    for t in gdelt.stamps(since, until):
        by_day.setdefault(f"{t:%Y-%m-%d}", []).append(t)
    todo = {d: [t for t in ts if t not in _done(d)] for d, ts in by_day.items()}
    todo = {d: ts for d, ts in todo.items() if ts}
    log(f"gdelt events: {sum(map(len, todo.values()))} files over {len(todo)} days to fetch into {STORE}")
    out = {"ok": 0, "missing": 0, "failed": 0, "events": 0}
    for i, (d, ts) in enumerate(sorted(todo.items())):
        r = ingest_day(d, ts, workers, log=log)
        out = {k: out[k] + r[k] for k in out}
        if (i + 1) % 14 == 0 or i + 1 == len(todo):
            log(f"  {d}: {i + 1}/{len(todo)} days, {out['events']} events")
    return out


def load(since=None, until=None, columns=None) -> pd.DataFrame:
    """Event aggregates per publication bucket, with goldstein and tone as article-weighted means. Rows only, no done-markers."""
    files = sorted(STORE.glob("????-??-??.parquet")) if STORE.exists() else []
    if since is not None:
        files = [f for f in files if f.stem >= str(pd.Timestamp(since).date())]
    if until is not None:
        files = [f for f in files if f.stem <= str(pd.Timestamp(until).date())]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    df = df.dropna(subset=["avail"]).drop(columns="files", errors="ignore")
    if {"gw", "articles"} <= set(df):
        df["goldstein"] = df["gw"] / df["articles"]
    if {"tw", "articles"} <= set(df):
        df["tone"] = df["tw"] / df["articles"]
    df.attrs["source"] = f"{gdelt.CITATION} Events 2.0 raw files, {STORE}"
    return df
