"""Live checks and fetchers for the dictionary's series ids (FRED, ONS), plus the ML feature panel.

Nothing here writes to disk unless asked, and `panel(..., out=...)` refuses when any column comes from a
source whose licence does not allow storage (FRED's terms forbid storing or caching its content).
"""
from __future__ import annotations

import pandas as pd

from ..net import session
from ..providers.macro import fred
from ..providers.publishers import bls, hmrc_trade
from .graph import EconGraph

ONS = "https://www.ons.gov.uk"
ONS_SEARCH = "https://api.beta.ons.gov.uk/v1/search"
FETCHABLE = {"fred", "ons", "hmrc_uktradeinfo", "bls"}
# Months from the end of the reference period to publication, when an indicator sets no
# release_lag_months. Conservative on purpose: a feature that arrives late costs a little freshness,
# one that arrives early is look-ahead.
DEFAULT_LAG = {"D": 0, "W": 0, "M": 2, "Q": 3, "A": 6}


def is_cdid(sid: str) -> bool:
    """ONS series ids are four alphanumerics (J5EK); longer ids name a release, not a series."""
    return len(sid) == 4 and sid.isalnum()


def ons_lookup(cdid: str, dataset: str | None = None) -> dict | None:
    """ONS search API entry for a time-series CDID (dataset id, title, uri), preferring `dataset` when the
    same CDID is published in several datasets; None when the CDID does not exist."""
    r = session().get(ONS_SEARCH, params={"q": cdid, "content_type": "timeseries", "limit": 50}, timeout=20)
    r.raise_for_status()
    hits = [it for it in r.json().get("items", []) if it.get("cdid", "").upper() == cdid.upper()]
    exact = [it for it in hits if dataset and it.get("dataset_id", "").upper() == dataset.upper()]
    return (exact or hits or [None])[0]


def ons(cdid: str, dataset: str | None = None) -> pd.Series:
    """ONS time series (monthly if published, else quarterly, else annual) as a float Series."""
    hit = ons_lookup(cdid, dataset)
    if not hit:
        raise KeyError(f"ONS CDID {cdid} not found")
    uri = hit["uri"]
    r = session().get(f"{ONS}{uri}/data", timeout=30)
    r.raise_for_status()
    j = r.json()
    for freq, key in (("M", "months"), ("Q", "quarters"), ("A", "years")):
        rows = j.get(key) or []
        if rows:
            break
    idx = []
    for row in rows:
        if freq == "M":
            idx.append(pd.Timestamp(f"{row['year']}-{row['month'][:3]}-01"))
        elif freq == "Q":
            idx.append(pd.Period(f"{row['year']}{row['quarter']}", "Q").to_timestamp())
        else:
            idx.append(pd.Timestamp(f"{row['year']}-01-01"))
    s = pd.Series(pd.to_numeric([row["value"] for row in rows], errors="coerce"), index=idx, name=cdid).dropna()
    s.attrs["source"] = f"ONS {cdid}/{j.get('description', {}).get('datasetId', dataset)} {ONS}{uri}"
    return s


def fetch_series(spec: dict) -> pd.Series:
    if spec["source"] == "fred":
        s = fred(spec["id"])
        s.attrs["source"] = f"FRED {spec['id']}"
        return s
    if spec["source"] == "ons":
        return ons(spec["id"], spec.get("dataset"))
    if spec["source"] == "hmrc_uktradeinfo":
        df = hmrc_trade(spec["id"], spec.get("flow", "imports"))
        s = df[spec.get("measure", "value")].rename(spec["id"])
        s.attrs["source"] = df.attrs["source"]
        return s
    if spec["source"] == "bls":
        return bls(spec["id"])
    raise NotImplementedError(f"no fetcher for source {spec['source']}")


def check(g: EconGraph, sources=("fred", "ons", "hmrc_uktradeinfo"), log=print) -> pd.DataFrame:
    """Hit every FRED / ONS id once; report ok, last date and title, or the error."""
    rows = []
    for n in g.of_kind("sig"):
        for s in n.data.get("series", []):
            if s["source"] not in sources:
                continue
            row = {"indicator": n.id, "source": s["source"], "id": s["id"], "dataset": s.get("dataset")}
            try:
                if s["source"] == "ons" and not is_cdid(s["id"]):
                    row.update(ok=True, title="release, not a series id (not checked)")
                elif s["source"] == "hmrc_uktradeinfo":
                    from ..providers.publishers import hmrc_commodities
                    com = hmrc_commodities([c.strip() for c in s["id"].split(",")])
                    row.update(ok=len(com) > 0, title=f"{len(com)} CN8 codes: " + "; ".join(
                        sorted(set(com["Hs6Description"].astype(str)))[:2]) if len(com) else None)
                    if not len(com):
                        row["error"] = "no commodity codes"
                elif s["source"] == "ons":
                    hit = ons_lookup(s["id"], s.get("dataset"))
                    if not hit:
                        raise KeyError("CDID not found")
                    row["title"] = hit.get("title")
                    row["ok"] = not s.get("dataset") or hit.get("dataset_id", "").upper() == s["dataset"].upper()
                    if not row["ok"]:
                        row["error"] = f"dataset is {hit.get('dataset_id')}"
                else:
                    ser = fetch_series(s)
                    row.update(ok=True, last=ser.index[-1].date(), value=float(ser.iloc[-1]))
            except Exception as e:  # report and keep going
                row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:80]}")
            log(f"{'ok ' if row.get('ok') else 'BAD'} {row['source']:4} {row['id']:24} {row.get('error') or row.get('title') or row.get('last')}")
            rows.append(row)
    return pd.DataFrame(rows)


def _transform(s: pd.Series, how: str, freq: str) -> pd.Series:
    per = {"D": 365, "W": 52, "M": 12, "Q": 4, "A": 1}.get(freq, 12)
    if how == "level":
        return s
    if how == "yoy":
        return s.pct_change(per, fill_method=None) * 100
    if how == "mom":
        return s.pct_change(fill_method=None) * 100
    if how == "diff":
        return s.diff()
    if how == "log":
        import numpy as np
        return np.log(s.where(s > 0))
    if how == "z":
        return (s - s.rolling(per * 5, min_periods=per * 2).mean()) / s.rolling(per * 5, min_periods=per * 2).std()
    raise ValueError(how)


def pick_series(g: EconGraph, d: dict) -> list[dict]:
    """Fetchable series of an indicator in the order to try them: storable original publishers first,
    mirrors (FRED) after, so a panel can be saved whenever the data allows it."""
    def ok(s):
        return s["source"] in FETCHABLE and (s["source"] != "ons" or is_cdid(s["id"]))

    def storable(s):
        src = g.nodes.get(f"src:{s['source']}")
        return bool(src) and src.data.get("storage_allowed") == "yes"
    cands = [s for s in d.get("series", []) if ok(s)]
    return sorted(cands, key=lambda s: not storable(s))


def panel(g: EconGraph, start: str, depth: int = 1, geo: str | None = None, since: str = "2000-01-01",
          target: str | None = "sig:us_recession_nber", out: str | None = None, realtime: bool = True,
          log=print):
    """Monthly ML feature matrix for every fetchable indicator within `depth` of `start`: one column per
    indicator x transform, aligned at month end, plus the target label. Returns (frame, provenance).

    Rows are indexed by reference period, not release date: shift each column by its publication lag
    before training, or the model sees numbers that were not yet published (look-ahead). With
    realtime=True (the default) the panel does that shift itself: each column moves forward by the
    indicator's release_lag_months (or DEFAULT_LAG for its frequency), so row t holds only what was
    published by the end of month t. The target label is never shifted."""
    ids = g.indicators_near(start, depth)
    if target:
        ids.append(g.resolve(target))
    cols, prov = {}, []
    for sid in dict.fromkeys(ids):
        d = g.nodes[sid].data
        if geo and d.get("geo") != geo and sid != target:
            continue
        s = None
        for spec in pick_series(g, d):
            try:
                s = fetch_series(spec)
                break
            except Exception as e:  # fall through to the next publisher
                log(f"{sid}: {spec['source']} failed ({type(e).__name__}: {str(e)[:60]})")
        if s is None or s.empty:
            continue
        s = s[s.index >= pd.Timestamp(since) - pd.DateOffset(years=6)]
        freq = d.get("freq", "M")
        m = s.resample("ME").last() if freq in ("D", "W", "M") else s.resample("QE").last().resample("ME").ffill()
        lag = 0 if sid == target or not realtime else int(d.get("release_lag_months", DEFAULT_LAG.get(freq, 2)))
        for how in (d.get("transform") or ["level"]) if sid != target else ["level"]:
            col = _transform(m, how, "M")
            if lag:  # run the index on to this month so the latest release lands where it was published
                col = col.reindex(pd.date_range(col.index[0], pd.Timestamp.now() + pd.offsets.MonthEnd(0),
                                                freq="ME").normalize()).shift(lag)
            cols[f"{g.nodes[sid].key}__{how}"] = col
        src = g.nodes.get(f"src:{spec['source']}")
        prov.append({"indicator": sid, "source": spec["source"], "id": spec["id"], "last": s.index[-1].date(),
                     "lag_months": lag,
                     "storage_allowed": src.data.get("storage_allowed") if src else None,
                     "origin": s.attrs.get("source")})
    df = pd.DataFrame(cols)
    df = df[df.index >= pd.Timestamp(since)]
    prov = pd.DataFrame(prov)
    if out:
        blocked = prov[prov["storage_allowed"] != "yes"] if not prov.empty else prov
        if not blocked.empty:
            raise PermissionError("not writing: storage not allowed (or unverified) for "
                                  + ", ".join(f"{r.source}:{r.id}" for r in blocked.itertuples())
                                  + ". Use --geo UK (ONS, OGL) or fetch these from their original publishers.")
        df.to_csv(out)
    return df, prov
