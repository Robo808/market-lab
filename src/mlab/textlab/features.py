"""Point-in-time text features from the GDELT store: novelty of each headline and daily firm signals.

Timing: a trading day t's features use only articles available (GDELT file time) before 20:00 UTC on t, which
is at or before the US close all year; trades happen at the open of t+1 (see events.forward_car).

Novelty (as in Ke, Kelly & Xiu): 1 - max cosine similarity between a headline and the same firm's headlines
of the previous `window` days, on hashed word unigrams + bigrams. 1.0 = nothing like it recently.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from scipy import sparse

CUTOFF_UTC_HOUR = 20
_TOKEN = re.compile(r"[a-z][a-z0-9']+")
STOP = set("the a an and or of to in on for with at by from as is are was were be been it its this that "
           "after over new says said will can could may us up out about into than more".split())


def _tokens(title: str) -> list[str]:
    w = [t for t in _TOKEN.findall(title.lower()) if t not in STOP]
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def hash_matrix(titles: list[str], dim: int = 2 ** 18) -> sparse.csr_matrix:
    """L2-normalised hashed bag of unigrams + bigrams (crc32, stable across runs)."""
    rows, cols = [], []
    for i, t in enumerate(titles):
        for tok in set(_tokens(t or "")):
            rows.append(i)
            cols.append(zlib.crc32(tok.encode()) % dim)
    m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(titles), dim))
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    return sparse.diags(1 / norm) @ m


def novelty(df: pd.DataFrame, window: str = "5D") -> pd.Series:
    """Novelty per article row of one symbol's frame (columns avail, title), vs that symbol's prior window."""
    df = df.sort_values("avail")
    x = hash_matrix(df["title"].fillna("").tolist())
    t = pd.DatetimeIndex(df["avail"]).as_unit("ns").asi8
    lo = np.searchsorted(t, t - pd.Timedelta(window).value, side="left")
    out = np.ones(len(df))
    for i in range(len(df)):
        if lo[i] < i:
            prev = x[lo[i]:i]
            sim = prev @ x[i].T
            out[i] = 1.0 - (sim.max() if sim.nnz else 0.0)
    return pd.Series(out, index=df.index, name="novelty")


def trading_day(avail: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Map availability times to the trading day whose close-time decision may use them."""
    index = pd.DatetimeIndex(index).sort_values()
    day = avail.dt.floor("D")
    day = day.where(avail.dt.hour < CUTOFF_UTC_HOUR, day + pd.Timedelta(days=1))
    pos = index.searchsorted(pd.DatetimeIndex(day))
    ok = pos < len(index)
    mapped = pd.Series(pd.NaT, index=avail.index, dtype=index.dtype)
    mapped[ok] = index[pos[ok]]
    return mapped


def firm_daily(articles: pd.DataFrame, index: pd.DatetimeIndex, fresh: float = 0.75) -> pd.DataFrame:
    """Per trading day for one symbol: article count, mean tone, fresh (novelty >= `fresh`) count and tone,
    stale count and tone. `articles` must already carry a novelty column."""
    a = articles.copy()
    a["day"] = trading_day(a["avail"], index)
    a = a.dropna(subset=["day"])
    a["is_fresh"] = a["novelty"] >= fresh
    g = a.groupby("day")
    out = pd.DataFrame({"n": g.size(), "tone": g["tone"].mean()})
    for name, mask in (("fresh", a["is_fresh"]), ("stale", ~a["is_fresh"])):
        gg = a[mask].groupby("day")
        out[f"n_{name}"] = gg.size()
        out[f"tone_{name}"] = gg["tone"].mean()
    return out.reindex(index).fillna({"n": 0, "n_fresh": 0, "n_stale": 0})
