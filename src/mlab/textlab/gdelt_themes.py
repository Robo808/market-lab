"""GDELT GKG theme x country intensities: how much the world's news talks about tariffs, sanctions, elections, war,
regulation, energy or inflation, and where (#58).

    mlab newsstore themes --since 2015-02-19 --until 2015-08-31 --workers 8   # backfill, resumable

A second pass over the GKG 15-minute files (the firm/macro tables keep only company articles and global ECON_/EPU_
counts). One file per UTC day, $MLAB_DATA_DIR/data/news/gdelt/themes/YYYY-MM-DD.parquet, one row per (publication
bucket, theme, country): n articles, mean tone, plus a done-marker row per file (theme "_FILE", n = rows parsed,
-1 when the file is missing at the source).
- avail: same two buckets as the events table (published before 20:00 UTC -> 00:00, else 20:00; see
  gdelt_events.bucket);
- theme: GKG V1 theme, kept when it starts with one of THEMES (policy, conflict, economy, environment);
- country: FIPS code of the article's most-mentioned location country, "_ALL" for every article.
GDELT is free for academic, commercial and governmental use with a citation and a link to https://www.gdeltproject.org.
"""
from __future__ import annotations

import io
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import gdelt
from .gdelt_events import bucket

STORE = gdelt.STORE / "themes"
# Prefixes picked from the GKG theme list (checked against a 2019 file): economy and policy uncertainty, trade,
# sanctions, elections and politics, conflict and unrest, energy and environment, disasters and health.
THEMES = ("ECON_", "EPU_", "ENV_", "UNREST_", "SANCTIONS", "ELECTION", "LEADER", "LEGISLATION", "GENERAL_GOVERNMENT",
          "POLITICAL_TURMOIL", "PROTEST", "STRIKE", "REBELLION", "COUP", "MILITARY", "ARMEDCONFLICT", "KILL", "TERROR",
          "CYBER_ATTACK", "BLOCKADE", "SEIZE", "SECURITY_SERVICES", "WB_698_TRADE", "WB_775_TRADE_POLICY",
          "WB_778_NON_TARIFF", "WB_2462_POLITICAL_VIOLENCE", "WB_507_ENERGY", "WB_1235_CENTRAL_BANKS", "IMMIGRATION",
          "NATURAL_DISASTER", "MANMADE_DISASTER", "HEALTH_PANDEMIC", "FOOD_SECURITY", "CORRUPTION", "TAX_FNCACT_CEO")


def _country(locations: str) -> str:
    """Most-mentioned country (FIPS) among an article's V1 locations ('type#name#CC#...;...'), '' if none."""
    c = Counter(p.split("#")[2] for p in locations.split(";") if p.count("#") >= 2 and p.split("#")[2])
    return c.most_common(1)[0][0] if c else ""


def parse(blob: bytes, ts: pd.Timestamp) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    acc: dict[tuple, list] = defaultdict(lambda: [0, 0.0])
    n = 0
    for r in (line.split("\t") for line in raw.splitlines()):
        if len(r) != len(gdelt.COLS):
            continue
        try:
            tone = float(r[15].split(",")[0])
        except ValueError:
            continue
        n += 1
        themes = {t for t in r[7].split(";") if t.startswith(THEMES)}
        if not themes:
            continue
        cc = _country(r[9])
        for t in themes:
            for c in ("_ALL", cc) if cc else ("_ALL",):
                a = acc[(t, c)]
                a[0] += 1
                a[1] += tone
    b = bucket(ts)
    rows = [{"avail": b, "theme": t, "country": c, "n": v[0], "tone_sum": v[1]} for (t, c), v in acc.items()]
    rows.append({"avail": b, "theme": "_FILE", "country": f"{ts:%Y%m%d%H%M%S}", "n": n, "tone_sum": 0.0})
    return pd.DataFrame(rows)


def _path(day: str):
    return STORE / f"{day}.parquet"


def _done(day: str) -> set:
    p = _path(day)
    if not p.exists():
        return set()
    d = pd.read_parquet(p, columns=["theme", "country"])
    return set(pd.to_datetime(d.loc[d["theme"] == "_FILE", "country"], format="%Y%m%d%H%M%S", utc=True))


def _fetch(ts: pd.Timestamp, s) -> pd.DataFrame:
    r = s.get(f"{gdelt.BASE}/{ts:%Y%m%d%H%M%S}.gkg.csv.zip", timeout=60)
    if r.status_code == 404:
        return pd.DataFrame([{"avail": bucket(ts), "theme": "_FILE", "country": f"{ts:%Y%m%d%H%M%S}", "n": -1,
                              "tone_sum": 0.0}])
    r.raise_for_status()
    return parse(r.content, ts)


def ingest_day(day: str, stamps_: list[pd.Timestamp], workers: int = 4, session=None, log=print) -> dict:
    from ..net import session as mk
    s = session or mk("market-lab/0.1")
    out = {"ok": 0, "missing": 0, "failed": 0, "articles": 0}
    got = []

    def one(t):
        try:
            return t, _fetch(t, s), None
        except Exception as e:  # count and carry on; a rerun picks the file up again
            return t, None, f"{type(e).__name__}: {str(e)[:80]}"
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for t, res, err in ex.map(one, stamps_):
            if err:
                out["failed"] += 1
                log(f"  {t:%Y-%m-%d %H:%M} themes failed {err}")
                continue
            marker = res[res["theme"] == "_FILE"]["n"].iloc[0]
            out["missing" if marker < 0 else "ok"] += 1
            out["articles"] += max(int(marker), 0)
            got.append(res)
    if not got:
        return out
    p = _path(day)
    parts = ([pd.read_parquet(p)] if p.exists() else []) + got
    df = pd.concat(parts, ignore_index=True)
    files = df[df["theme"] == "_FILE"].drop_duplicates("country")
    agg = df[df["theme"] != "_FILE"].groupby(["avail", "theme", "country"], as_index=False)[["n", "tone_sum"]].sum()
    STORE.mkdir(parents=True, exist_ok=True)
    gdelt._write(pd.concat([agg, files], ignore_index=True).sort_values(["avail", "theme", "country"]), p)
    return out


def ingest(since, until=None, workers: int = 4, log=print) -> dict:
    by_day: dict[str, list] = {}
    for t in gdelt.stamps(since, until):
        by_day.setdefault(f"{t:%Y-%m-%d}", []).append(t)
    todo = {d: [t for t in ts if t not in _done(d)] for d, ts in by_day.items()}
    todo = {d: ts for d, ts in todo.items() if ts}
    log(f"gdelt themes: {sum(map(len, todo.values()))} files over {len(todo)} days to fetch into {STORE}")
    out = {"ok": 0, "missing": 0, "failed": 0, "articles": 0}
    for i, (d, ts) in enumerate(sorted(todo.items())):
        r = ingest_day(d, ts, workers, log=log)
        out = {k: out[k] + r[k] for k in out}
        if (i + 1) % 7 == 0 or i + 1 == len(todo):
            log(f"  {d}: {i + 1}/{len(todo)} days, {out['articles']} articles parsed")
    return out


def load(since=None, until=None, themes=None, countries=None) -> pd.DataFrame:
    """Theme x country rows with mean tone; `themes` / `countries` filter (exact names)."""
    files = sorted(STORE.glob("????-??-??.parquet")) if STORE.exists() else []
    if since is not None:
        files = [f for f in files if f.stem >= str(pd.Timestamp(since).date())]
    if until is not None:
        files = [f for f in files if f.stem <= str(pd.Timestamp(until).date())]
    if not files:
        return pd.DataFrame()
    filt = []
    if themes:
        filt.append(("theme", "in", list(themes)))
    if countries:
        filt.append(("country", "in", list(countries)))
    df = pd.concat([pd.read_parquet(f, filters=filt or None) for f in files], ignore_index=True)
    df = df[df["theme"] != "_FILE"]
    df["tone"] = df["tone_sum"] / df["n"]
    df.attrs["source"] = f"{gdelt.CITATION} GKG 2.0 raw files, themes x country, {STORE}"
    return df
