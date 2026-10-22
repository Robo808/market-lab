"""Point-in-time text features from the GDELT store: novelty of each headline and daily firm signals.

Timing: a trading day t's features use only articles available (GDELT file time) before 20:00 UTC on t, which
is at or before the US close all year; trades happen at the open of t+1 (see events.forward_car).

Novelty (as in Ke, Kelly & Xiu): 1 - max cosine similarity between a headline and the same firm's headlines
of the previous `window` days, on hashed word unigrams + bigrams. 1.0 = nothing like it recently.
"""
from __future__ import annotations

import re
import zlib
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy import sparse

CUTOFF_UTC_HOUR = 20
_WORD = re.compile(r"[a-z]{2,}")
_URL_STOP = {"html", "htm", "php", "aspx", "asp", "shtml", "article", "index", "amp", "story", "news"}
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


def slug(url: str) -> str:
    """Headline words from a URL path: the path segment with the most words, minus file and CMS tokens. GKG only
    carries page titles from late Sep 2019, so slugs are the one headline text that spans every year (85-93% of
    company articles have a slug of 3+ words in 2015-2022)."""
    segs = [s for s in urlsplit(url or "").path.lower().split("/") if s]
    best = max(segs, key=lambda s: len(_WORD.findall(s)), default="")
    return " ".join(w for w in re.split(r"[-_.+]", best) if _WORD.fullmatch(w) and w not in _URL_STOP)


# Names that make an article *about* the company when they appear in its headline (URL slug words). GKG org lists
# also tag passing mentions ("streaming on Netflix", "posted on Facebook"), which swamp single-stock counts (#57).
SLUG_NAMES = {
    "AAPL": ["apple", "iphone", "ipad", "macbook", "tim cook"], "MSFT": ["microsoft", "xbox", "azure", "nadella"],
    "NVDA": ["nvidia"], "AMZN": ["amazon", "aws", "bezos", "jassy"], "GOOGL": ["google", "alphabet", "youtube"],
    "META": ["facebook", "instagram", "whatsapp", "zuckerberg", "meta platforms"], "TSLA": ["tesla", "elon musk"],
    "AVGO": ["broadcom"], "BRK-B": ["berkshire", "buffett"], "JPM": ["jpmorgan", "jp morgan", "dimon"],
    "LLY": ["eli lilly", "lilly"], "V": ["visa inc", "visa card", "visa earnings"], "MA": ["mastercard"],
    "XOM": ["exxon", "exxonmobil"], "UNH": ["unitedhealth", "unitedhealthcare", "optum"], "COST": ["costco"],
    "NFLX": ["netflix"], "AMD": ["amd", "advanced micro devices"], "ORCL": ["oracle"], "WMT": ["walmart", "wal mart"],
    "PLTR": ["palantir"], "CRM": ["salesforce"], "ADBE": ["adobe"], "INTC": ["intel"],
    "BAC": ["bank of america", "bofa"],
}
_SLUG_RE = {s: re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b") for s, names in SLUG_NAMES.items()}


def primary(text: str, symbol: str) -> bool:
    """True when the headline text (slug or title, lower case) names the company: the article is about it."""
    rx = _SLUG_RE.get(symbol)
    return bool(rx and rx.search((text or "").lower()))


def novelty(df: pd.DataFrame, window: str = "5D", text: str = "title", block: int = 500) -> pd.Series:
    """Novelty per article row of one symbol's frame (columns avail and `text`), vs that symbol's earlier rows in
    the prior `window` (rows earlier in avail order; same-file duplicates count only after the first)."""
    df = df.sort_values("avail", kind="stable")
    x = hash_matrix(df[text].fillna("").tolist())
    t = pd.DatetimeIndex(df["avail"]).as_unit("ns").asi8
    w = pd.Timedelta(window).value
    out = np.ones(len(df))
    for a in range(0, len(df), block):  # score rows a..b against every row that can precede them
        b = min(a + block, len(df))
        lo = int(np.searchsorted(t, t[a] - w, side="left"))
        sim = (x[a:b] @ x[lo:b].T).toarray()
        i, j = np.arange(a, b)[:, None], np.arange(lo, b)[None, :]
        sim[(j >= i) | (t[lo:b][None, :] < t[a:b][:, None] - w)] = 0.0
        out[a:b] = 1.0 - sim.max(axis=1, initial=0.0)
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
