"""News, sentiment and social desk.

Sources (each returns the same normalised frame, see COLUMNS): Google News RSS, Yahoo Finance RSS,
yfinance Ticker.news, GDELT DOC 2.0 (articles + tone/volume timelines), SEC 8-K filings, StockTwits,
Reddit search, Hacker News (Algolia), and keyed Alpha Vantage / FMP / Finnhub when their env vars are set.

Every network call lives in a small `_fetch_*` / `_get` function; every `parse_*` function is pure, so
parsing, scoring and aggregation are testable offline. `brief()` never fails because one source did:
per-source errors are returned alongside the results.

Scoring is a compact VADER-style rule-based scorer (valence lexicon, negation, boosters, but-clauses,
caps/exclamation emphasis) plus finance phrases and social slang. No NLP packages needed.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import html
import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from .config import env

# host -> purpose, for `mlab doctor` style reachability checks.
HOSTS: dict[str, str] = {
    "news.google.com": "Google News RSS search",
    "feeds.finance.yahoo.com": "Yahoo Finance RSS headlines",
    "query2.finance.yahoo.com": "yfinance Ticker.news / company name",
    "api.gdeltproject.org": "GDELT DOC 2.0 articles, tone and volume timelines",
    "data.sec.gov": "SEC EDGAR 8-K filings (via providers.sec)",
    "www.sec.gov": "SEC ticker map (via providers.sec)",
    "api.stocktwits.com": "StockTwits symbol stream (Bullish/Bearish tags)",
    "www.reddit.com": "Reddit search JSON",
    "hn.algolia.com": "Hacker News Algolia search",
    "www.alphavantage.co": "Alpha Vantage NEWS_SENTIMENT (ALPHAVANTAGE_API_KEY)",
    "financialmodelingprep.com": "FMP stock news (FMP_API_KEY)",
    "finnhub.io": "Finnhub company-news (FINNHUB_API_KEY)",
}

COLUMNS = ["time", "source", "symbol", "title", "summary", "url", "author", "engagement", "comments",
           "label", "vendor_score"]
SOCIAL = {"stocktwits", "reddit", "hn"}
SOURCE_WEIGHT = {"sec": 1.0, "alphavantage": 0.85, "finnhub": 0.85, "fmp": 0.8, "yahoo": 0.8, "yfinance": 0.8,
                 "google": 0.7, "gdelt": 0.6, "hn": 0.45, "reddit": 0.4, "stocktwits": 0.3}
SUBREDDITS = ("wallstreetbets", "stocks", "investing", "options", "UKInvesting")
TECH = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "ORCL", "CRM", "ADBE", "NFLX",
        "PLTR", "SNOW", "NET", "CRWD", "SHOP", "UBER", "ARM", "AVGO", "TSM", "ASML", "MU", "QCOM", "IBM", "COIN"}
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "GOOG": "Alphabet", "AMZN": "Amazon",
         "META": "Meta Platforms", "NVDA": "Nvidia", "TSLA": "Tesla", "AMD": "AMD", "NFLX": "Netflix",
         "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs", "XOM": "Exxon Mobil", "BRK-B": "Berkshire Hathaway",
         "PLTR": "Palantir", "AVGO": "Broadcom", "TSM": "TSMC", "ASML": "ASML", "COIN": "Coinbase", "INTC": "Intel",
         "BARC.L": "Barclays", "HSBA.L": "HSBC", "BP.L": "BP", "SHEL.L": "Shell", "AZN.L": "AstraZeneca", "VOD.L": "Vodafone",
         "LLOY.L": "Lloyds Banking", "RR.L": "Rolls-Royce", "ULVR.L": "Unilever", "GSK.L": "GSK"}
# Query templates per kind; {when} is appended by google_news().
QUERY_TEMPLATES = {
    "ticker": '"{ticker}" stock',
    "company": '"{name}" (shares OR stock OR earnings OR analyst)',
    "topic": "{query}",
}
TICKER_RE = re.compile(r"^\$?[A-Z]{1,5}([.\-][A-Z]{1,2})?$")
GDELT_PAUSE = 5.5  # GDELT asks for at most one request every 5 seconds


# =============================================================================== network layer
def _get(url: str, params: dict | None = None, kind: str = "json", headers: dict | None = None, timeout: float = 20):
    """The one HTTP call every source uses. Returns parsed JSON or text."""
    from .net import session
    s = session("Mozilla/5.0 (X11; Linux x86_64) market-lab/0.1 (research)")
    r = s.get(url, params=params, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    if kind == "text":
        return r.text
    try:
        return r.json()
    except ValueError:
        raise ValueError(f"non-JSON reply from {url.split('/')[2]}: {r.text[:120]!r}") from None


def _yf_news_raw(ticker: str) -> list:
    import yfinance as yf
    return yf.Ticker(ticker).news or []


def _yf_name(ticker: str) -> str | None:
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    return info.get("shortName") or info.get("longName")


def _sec_8k_raw(ticker: str, limit: int = 20) -> pd.DataFrame:
    from .providers import sec
    return sec.filings(ticker, ("8-K",), limit)


# =============================================================================== helpers
_TAG = re.compile(r"<[^>]+>")


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", str(s)))).strip()


def _ts(v):
    """Parse RFC-822, ISO, compact GDELT/AV stamps or unix seconds into a UTC Timestamp (NaT on failure)."""
    if v is None or v == "":
        return pd.NaT
    if isinstance(v, (int, float, np.integer, np.floating)):
        return pd.Timestamp(int(v), unit="s", tz="UTC")
    s = str(v).strip()
    m = re.fullmatch(r"(\d{8})T?(\d{4,6})?Z?", s)
    if m:
        hms = (m.group(2) or "000000").ljust(6, "0")
        return pd.Timestamp(f"{m.group(1)}T{hms}", tz="UTC")
    try:
        return pd.Timestamp(parsedate_to_datetime(s)).tz_convert("UTC")
    except (TypeError, ValueError, IndexError):
        pass
    try:
        t = pd.Timestamp(s)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (ValueError, TypeError):
        return pd.NaT


def _frame(rows: list[dict], source: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["source"] = source
    df["time"] = pd.to_datetime(df["time"].map(_ts), utc=True) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    for c in ("title", "summary", "url", "author", "label", "symbol"):
        df[c] = df[c].map(_clean).astype(object)
    for c in ("engagement", "comments", "vendor_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["title"].str.len() > 0]
    return df.sort_values("time", ascending=False, na_position="last").reset_index(drop=True)


def empty() -> pd.DataFrame:
    return _frame([], "none")


def _uid(df: pd.DataFrame) -> pd.Series:
    key = df["source"].astype(str) + "|" + df["url"].where(df["url"].str.len() > 0, df["title"]).astype(str)
    return key.map(lambda s: hashlib.sha1(s.encode(), usedforsecurity=False).hexdigest()[:16])  # dedupe key only


# =============================================================================== parsers (pure)
def parse_rss(xml_text: str, source: str, symbol: str = "") -> pd.DataFrame:
    """RSS 2.0 (Google News, Yahoo Finance). Google titles end in ' - Publisher', split into author."""
    from defusedxml import ElementTree as ET  # feeds are untrusted: block entity-expansion and XXE tricks
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    rows = []
    for it in root.iter("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()  # noqa: E731
        title, pub = g("title"), g("source")
        if source == "google" and not pub and " - " in title:
            title, pub = title.rsplit(" - ", 1)
        elif pub and title.endswith(" - " + pub):
            title = title[: -len(pub) - 3]
        summary = _clean(g("description"))
        if summary.startswith(_clean(title)):  # Google repeats the title in the description
            summary = summary[len(_clean(title)):].strip(" -")
            summary = "" if summary == pub else summary
        rows.append({"time": g("pubDate"), "title": title, "summary": summary, "url": g("link"),
                     "author": pub, "symbol": symbol})
    return _frame(rows, source)


def parse_yf_news(items: list, symbol: str) -> pd.DataFrame:
    """yfinance Ticker.news: handles both the legacy flat and the newer {'content': {...}} shapes."""
    rows = []
    for it in items or []:
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        rows.append({"time": c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"),
                     "title": c.get("title"), "summary": c.get("summary") or c.get("description") or "",
                     "url": url or (c.get("clickThroughUrl") or {}).get("url", ""),
                     "author": (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher"),
                     "symbol": symbol})
    return _frame(rows, "yfinance")


def parse_gdelt_artlist(j: dict, symbol: str = "") -> pd.DataFrame:
    rows = [{"time": a.get("seendate"), "title": a.get("title"), "url": a.get("url"), "author": a.get("domain"),
             "summary": f"{a.get('sourcecountry', '')} {a.get('language', '')}".strip(), "symbol": symbol}
            for a in (j or {}).get("articles", [])]
    return _frame(rows, "gdelt")


def parse_gdelt_timeline(j: dict, name: str) -> pd.Series:
    """TimelineTone / TimelineVolRaw JSON -> daily Series (UTC midnight index)."""
    tl = (j or {}).get("timeline") or []
    if not tl:
        return pd.Series(dtype=float, name=name)
    data = tl[0].get("data", [])
    s = pd.Series({_ts(d["date"]): float(d["value"]) for d in data if "date" in d}, name=name, dtype=float)
    s.index = pd.DatetimeIndex(s.index).floor("D")
    return s.groupby(level=0).mean() if "tone" in name else s.groupby(level=0).sum()


def parse_sec_8k(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rows = [{"time": r["filingDate"], "title": f"8-K: {r.get('primaryDocDescription') or 'current report'}",
             "summary": f"report date {r.get('reportDate', '')}", "url": r["url"], "author": "SEC EDGAR", "symbol": symbol}
            for _, r in (df if df is not None else pd.DataFrame()).iterrows()]
    return _frame(rows, "sec")


def parse_stocktwits(j: dict, symbol: str) -> pd.DataFrame:
    rows = []
    for m in (j or {}).get("messages", []):
        sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic") or ""
        user = m.get("user") or {}
        body = m.get("body") or ""
        rows.append({"time": m.get("created_at"), "title": body[:280], "summary": "", "author": user.get("username"),
                     "url": f"https://stocktwits.com/{user.get('username', '')}/message/{m.get('id', '')}",
                     "engagement": (m.get("likes") or {}).get("total", 0), "comments": (m.get("conversation") or {}).get("replies"),
                     "label": sent, "symbol": symbol})
    return _frame(rows, "stocktwits")


def parse_reddit(j: dict, symbol: str = "") -> pd.DataFrame:
    rows = []
    for ch in ((j or {}).get("data") or {}).get("children", []):
        d = ch.get("data", {})
        rows.append({"time": d.get("created_utc"), "title": d.get("title"), "summary": (d.get("selftext") or "")[:500],
                     "url": "https://www.reddit.com" + d.get("permalink", ""), "author": f"r/{d.get('subreddit', '')} u/{d.get('author', '')}",
                     "engagement": d.get("score"), "comments": d.get("num_comments"), "symbol": symbol})
    return _frame(rows, "reddit")


def parse_hn(j: dict, symbol: str = "") -> pd.DataFrame:
    rows = [{"time": h.get("created_at_i") or h.get("created_at"), "title": h.get("title") or h.get("story_title"),
             "summary": (h.get("story_text") or "")[:500], "author": h.get("author"),
             "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
             "engagement": h.get("points"), "comments": h.get("num_comments"), "symbol": symbol}
            for h in (j or {}).get("hits", [])]
    return _frame(rows, "hn")


def parse_alphavantage(j: dict, symbol: str) -> pd.DataFrame:
    if "feed" not in (j or {}):
        raise LookupError(f"alphavantage: {str(j)[:160]}")
    rows = []
    for a in j["feed"]:
        ts = next((float(t["ticker_sentiment_score"]) for t in a.get("ticker_sentiment", [])
                   if t.get("ticker", "").upper() == symbol.upper()), a.get("overall_sentiment_score"))
        rows.append({"time": a.get("time_published"), "title": a.get("title"), "summary": a.get("summary"), "url": a.get("url"),
                     "author": a.get("source"), "vendor_score": ts, "symbol": symbol})
    return _frame(rows, "alphavantage")


def parse_fmp(j: list, symbol: str) -> pd.DataFrame:
    if isinstance(j, dict):
        raise LookupError(f"fmp: {str(j)[:160]}")
    rows = [{"time": a.get("publishedDate"), "title": a.get("title"), "summary": a.get("text"), "url": a.get("url"),
             "author": a.get("site") or a.get("publisher"), "symbol": symbol} for a in j or []]
    return _frame(rows, "fmp")


def parse_finnhub(j: list, symbol: str) -> pd.DataFrame:
    if isinstance(j, dict):
        raise LookupError(f"finnhub: {str(j)[:160]}")
    rows = [{"time": a.get("datetime"), "title": a.get("headline"), "summary": a.get("summary"), "url": a.get("url"),
             "author": a.get("source"), "symbol": symbol} for a in j or []]
    return _frame(rows, "finnhub")


# =============================================================================== sources
@dataclass
class Query:
    query: str
    ticker: str | None
    name: str
    days: int = 7

    @property
    def key(self) -> str:
        return self.ticker or self.query


def company_name(ticker: str) -> str:
    """Short company name for search queries: NAMES map, then yfinance, then the ticker."""
    n = NAMES.get(ticker.upper())
    if not n:
        try:
            n = _yf_name(ticker)
        except Exception:
            n = None
    n = n or ticker
    return re.sub(r"[,.]?\s+(inc|corp|corporation|plc|ltd|limited|co|company|holdings|group|class [a-c])\.?$", "", n, flags=re.I).strip()


def make_query(q: str, days: int = 7, name: str | None = None) -> Query:
    q = q.strip()
    if TICKER_RE.match(q):
        t = q.lstrip("$")
        return Query(q, t, name or company_name(t), days)
    return Query(q, None, name or q, days)


def google_news(query: str, days: int = 7, symbol: str = "") -> pd.DataFrame:
    url = "https://news.google.com/rss/search"
    xml = _get(url, {"q": f"{query} when:{days}d", "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}, kind="text")
    return parse_rss(xml, "google", symbol)


def google_for(qc: Query) -> pd.DataFrame:
    if not qc.ticker:
        return google_news(QUERY_TEMPLATES["topic"].format(query=qc.query), qc.days, qc.query)
    frames = [google_news(QUERY_TEMPLATES["ticker"].format(ticker=qc.ticker), qc.days, qc.ticker)]
    if qc.name and qc.name != qc.ticker:
        frames.append(google_news(QUERY_TEMPLATES["company"].format(name=qc.name), qc.days, qc.ticker))
    return pd.concat(frames, ignore_index=True)


def yahoo_rss(ticker: str) -> pd.DataFrame:
    xml = _get("https://feeds.finance.yahoo.com/rss/2.0/headline", {"s": ticker, "region": "US", "lang": "en-US"}, kind="text")
    return parse_rss(xml, "yahoo", ticker)


def yfinance_news(ticker: str) -> pd.DataFrame:
    return parse_yf_news(_yf_news_raw(ticker), ticker)


def gdelt_query(qc: Query) -> str:
    if qc.ticker:
        terms = [f'"{qc.name}"'] + ([f'"{qc.ticker}"'] if len(qc.ticker) >= 3 and qc.ticker != qc.name else [])
        q = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    else:
        q = qc.query if qc.query.startswith(("(", '"')) or " " not in qc.query else f'"{qc.query}"'
    return f"{q} sourcelang:english"


def _gdelt(query: str, mode: str, timespan: str, maxrecords: int = 250) -> dict:
    params = {"query": query, "mode": mode, "format": "json", "timespan": timespan}
    if mode == "ArtList":
        params.update(maxrecords=maxrecords, sort="DateDesc")
    return _get("https://api.gdeltproject.org/api/v2/doc/doc", params)


def gdelt_articles(qc: Query) -> pd.DataFrame:
    return parse_gdelt_artlist(_gdelt(gdelt_query(qc), "ArtList", f"{qc.days}d"), qc.key)


def gdelt_timeline(qc: Query, days: int = 90, pause: float = GDELT_PAUSE) -> pd.DataFrame:
    """Daily GDELT average tone and raw article volume (DOC API covers ~3 months)."""
    q = gdelt_query(qc)
    tone = parse_gdelt_timeline(_gdelt(q, "TimelineTone", f"{days}d"), "gdelt_tone")
    time.sleep(pause)
    vol = parse_gdelt_timeline(_gdelt(q, "TimelineVolRaw", f"{days}d"), "gdelt_volume")
    return pd.concat([tone, vol], axis=1, sort=True)


def sec_8k(ticker: str) -> pd.DataFrame:
    return parse_sec_8k(_sec_8k_raw(ticker.replace(".L", "")), ticker)


def stocktwits(ticker: str) -> pd.DataFrame:
    return parse_stocktwits(_get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"), ticker)


def reddit(query: str, symbol: str = "", subreddits=SUBREDDITS, limit: int = 100, t: str = "month") -> pd.DataFrame:
    url = f"https://www.reddit.com/r/{'+'.join(subreddits)}/search.json"
    j = _get(url, {"q": query, "restrict_sr": 1, "sort": "new", "limit": limit, "t": t})
    return parse_reddit(j, symbol or query)


def hackernews(query: str, days: int = 7, symbol: str = "") -> pd.DataFrame:
    since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp())
    j = _get("https://hn.algolia.com/api/v1/search_by_date",
             {"query": query, "tags": "story", "numericFilters": f"created_at_i>{since}", "hitsPerPage": 100})
    return parse_hn(j, symbol or query)


def alphavantage(ticker: str) -> pd.DataFrame:
    j = _get("https://www.alphavantage.co/query", {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": 200,
                                                   "apikey": env("ALPHAVANTAGE_API_KEY")})
    return parse_alphavantage(j, ticker)


def fmp(ticker: str) -> pd.DataFrame:
    key = env("FMP_API_KEY")
    try:
        j = _get("https://financialmodelingprep.com/stable/news/stock", {"symbols": ticker, "limit": 100, "apikey": key})
    except Exception:
        j = _get("https://financialmodelingprep.com/api/v3/stock_news", {"tickers": ticker, "limit": 100, "apikey": key})
    return parse_fmp(j, ticker)


def finnhub(ticker: str, days: int = 7) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    j = _get("https://finnhub.io/api/v1/company-news", {"symbol": ticker, "from": f"{now - pd.Timedelta(days=days):%Y-%m-%d}",
                                                         "to": f"{now:%Y-%m-%d}", "token": env("FINNHUB_API_KEY")})
    return parse_finnhub(j, ticker)


def _reddit_q(qc: Query) -> str:
    if not qc.ticker:
        return qc.query
    base = qc.ticker.split(".")[0]
    return f'"{base}" OR "{qc.name}"' if qc.name != qc.ticker else base


# name -> (callable(Query), applies(Query))
FETCHERS = {
    "google": (google_for, lambda q: True),
    "yahoo": (lambda q: yahoo_rss(q.ticker), lambda q: bool(q.ticker)),
    "yfinance": (lambda q: yfinance_news(q.ticker), lambda q: bool(q.ticker)),
    "gdelt": (gdelt_articles, lambda q: True),
    "sec": (lambda q: sec_8k(q.ticker), lambda q: bool(q.ticker) and "." not in q.ticker),
    "stocktwits": (lambda q: stocktwits(q.ticker), lambda q: bool(q.ticker)),
    "reddit": (lambda q: reddit(_reddit_q(q), q.key, t="week" if q.days <= 7 else "month"), lambda q: True),
    "hn": (lambda q: hackernews(q.name if q.ticker else q.query, q.days, q.key), lambda q: not q.ticker or q.ticker in TECH),
    "alphavantage": (lambda q: alphavantage(q.ticker), lambda q: bool(q.ticker and env("ALPHAVANTAGE_API_KEY"))),
    "fmp": (lambda q: fmp(q.ticker), lambda q: bool(q.ticker and env("FMP_API_KEY"))),
    "finnhub": (lambda q: finnhub(q.ticker, q.days), lambda q: bool(q.ticker and env("FINNHUB_API_KEY"))),
}


def collect(qc: Query, sources=None, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Run the applicable sources in parallel. Returns (items, {source: error}) and never raises."""
    names = [s for s in (sources or FETCHERS) if s in FETCHERS and (sources or FETCHERS[s][1](qc))]
    errors = {s: "unknown source" for s in (sources or []) if s not in FETCHERS}

    def run(name):
        try:
            return name, FETCHERS[name][0](qc), None
        except Exception as e:
            return name, None, f"{type(e).__name__}: {str(e)[:140]}"

    frames = []
    with cf.ThreadPoolExecutor(max(1, min(workers, len(names) or 1))) as ex:
        for name, df, err in ex.map(run, names):
            if err:
                errors[name] = err
            elif df is not None and len(df):
                frames.append(df)
    items = pd.concat(frames, ignore_index=True) if frames else empty()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=qc.days)
    items = items[items["time"].isna() | (items["time"] >= cutoff)].reset_index(drop=True)
    return items, errors


# =============================================================================== scoring
def _lex(spec: str) -> dict[str, float]:
    out = {}
    for line in spec.strip().splitlines():
        words, v = line.rsplit(":", 1)
        for w in words.split():
            out[w] = float(v)
    return out


LEXICON = _lex("""
good well nice solid fine-tuned: 1.7
great excellent outstanding stellar superb blowout: 3.0
best: 3.1
strong stronger strongest robust resilient healthy: 1.7
upbeat optimistic confident encouraging impressive: 1.9
positive favorable favourable: 1.6
bullish: 2.2
win wins won winning: 1.8
gain gains gained gaining: 1.4
rise rises rose rising: 1.0
climb climbs climbed climbing: 1.2
jump jumps jumped jumping: 1.6
surge surges surged surging: 2.2
soar soars soared soaring: 2.4
rally rallies rallied rallying: 1.8
rebound rebounds rebounded rebounding recover recovers recovered recovery: 1.3
boost boosts boosted boosting: 1.4
beat beats beating: 1.5
top tops topped: 0.8
outperform outperforms outperformed outperforming: 1.6
upgrade upgrades upgraded upgrading: 2.0
growth grow grows growing grew expands expansion: 1.1
profit profits profitable profitability: 1.3
record: 0.6
breakthrough: 2.3
approval approved approves: 1.6
milestone: 1.3
innovative innovation: 1.2
partnership partners: 0.8
buy: 0.8
accumulate: 0.7
opportunity opportunities: 1.0
love loves loved: 2.9
happy: 2.5
undervalued cheap bargain: 1.2
tailwind tailwinds: 1.3
dividend dividends: 0.6
buyback buybacks repurchase: 1.4
bad poor: -2.3
terrible awful horrible disaster disastrous: -3.0
worst: -3.1
weak weaker weakest weakness soft softer sluggish: -1.7
disappoint disappoints disappointed disappointing disappointment: -2.2
negative unfavorable unfavourable: -1.6
bearish: -2.2
lose loses losing lost: -1.5
loss losses: -1.6
fall falls fell falling: -1.2
drop drops dropped dropping: -1.3
decline declines declined declining: -1.3
slide slides slid sliding: -1.3
dip dips dipped: -0.9
sink sinks sank sinking: -1.8
slump slumps slumped slumping: -2.2
tumble tumbles tumbled tumbling: -2.2
plunge plunges plunged plunging: -2.6
plummet plummets plummeted plummeting: -2.8
crash crashes crashed crashing collapse collapses collapsed collapsing: -3.0
tank tanks tanked tanking: -2.3
miss misses missed missing: -1.5
downgrade downgrades downgraded downgrading: -2.0
underperform underperforms underperformed: -1.6
warn warns warned warning warnings: -1.8
cut cuts cutting: -1.0
slash slashes slashed: -1.8
layoff layoffs: -1.1
lawsuit lawsuits sue sues sued litigation: -1.6
probe probes investigation investigations investigate investigating subpoena: -1.8
fraud fraudulent scandal: -3.1
fined fines penalty penalties: -1.6
recall recalls recalled: -1.8
bankruptcy bankrupt insolvency insolvent: -3.2
default defaults defaulted: -2.4
delay delays delayed: -1.2
halt halts halted suspend suspends suspended: -1.6
risk risks risky: -0.8
concern concerns worried worry worries fear fears: -1.4
headwind headwinds: -1.3
overvalued bubble: -1.4
recession: -2.0
sell selloff sell-off: -1.1
volatile volatility turmoil: -0.9
dilution dilutive: -1.5
resign resigns resigned: -0.8
""")
# Finance multi-word phrases. Matched case-insensitively on the raw text and replaced by one token.
_GAP = r"(?:(?!but\b|however\b)[\w$&'.-]+\s+){0,3}?"
_EST = r"(?:estimates?|expectations?|forecasts?|consensus|views?|street|targets?)"
_AUX = r"\s+(?:(?:was|is|were|are|has been|have been|been|gets?|got|being)\s+)?"
_GUIDE = r"(?:guidance|outlook|forecasts?|full[- ]year|fy\d*\s+(?:guidance|outlook))"
PHRASES: list[tuple[str, str, float]] = [
    (r"\b(?:beat|beats|beating|tops?|topped|tops|surpass(?:es|ed)?|exceed(?:s|ed)?|crush(?:es|ed)?)\s+" + _GAP + _EST + r"\b", "fin_beat", 2.6),
    (r"\b(?:miss(?:es|ed)?|fall(?:s)? short of|fell short of|lag(?:s|ged)?)\s+" + _GAP + _EST + r"\b", "fin_miss", -2.6),
    (r"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed)?|boost(?:s|ed)?|ups|upped|hik(?:e|es|ed)|increas(?:e|es|ed)|reaffirm(?:s|ed)?)\s+" + _GAP + _GUIDE + r"\b", "fin_guide_up", 2.4),
    (r"\b" + _GUIDE + _AUX + r"(?:raised|lifted|boosted|hiked|increased|beats?|tops)\b", "fin_guide_up", 2.4),
    (r"\b(?:cut(?:s|ting)?|lower(?:s|ed)?|slash(?:es|ed)?|trim(?:s|med)?|reduc(?:e|es|ed)|withdraw(?:s|n)?|withdrew|pull(?:s|ed)?|weak|soft)\s+" + _GAP + _GUIDE + r"\b", "fin_guide_down", -2.6),
    (r"\b" + _GUIDE + _AUX + r"(?:cut|lowered|slashed|trimmed|reduced|withdrawn|disappoints?|misses)\b", "fin_guide_down", -2.6),
    (r"\b(?:rais(?:es|ed)|lift(?:s|ed)|boost(?:s|ed)|hik(?:es|ed))\s+" + _GAP + r"price targets?\b", "fin_pt_up", 1.4),
    (r"\b(?:cut(?:s)?|lower(?:s|ed)|slash(?:es|ed)|trim(?:s|med))\s+" + _GAP + r"price targets?\b", "fin_pt_down", -1.4),
    (r"\b(?:sec|doj|ftc|fca|cma|justice department|antitrust|criminal|federal)\s+(?:probe|investigation|inquiry|charges?|subpoena|lawsuit)\b", "fin_reg_probe", -2.6),
    (r"\bclass[- ]action\b", "fin_class_action", -2.0),
    (r"\bshort[- ]sellers?(?:'s)?\s+(?:report|attack|bet)|\bshort report\b|\bshort[- ]seller\b", "fin_short_report", -2.3),
    (r"\bchapter (?:7|11)\b|\bfiles? for bankruptcy\b|\bgoing concern\b", "fin_bankrupt", -3.4),
    (r"\b(?:dividend\s+(?:hike|increase|raise|boost)|(?:raises|hikes|boosts|increases)\s+(?:its\s+|quarterly\s+)*dividend|special dividend)\b", "fin_div_up", 2.0),
    (r"\b(?:dividend\s+(?:cut|suspension)|(?:cuts|slashes|suspends|eliminates)\s+(?:its\s+|quarterly\s+)*dividend)\b", "fin_div_down", -2.4),
    (r"\b(?:share\s+)?(?:buyback|repurchase)\s+(?:program(?:me)?|plan|authori[sz]ation)\b|\bbuy back\b", "fin_buyback", 1.8),
    (r"\brecord\s+(?:revenue|sales|profits?|earnings|quarter|results|deliveries|margins?)\b", "fin_record", 2.4),
    (r"\b(?:all[- ]time|record|52[- ]week)\s+highs?\b", "fin_high", 1.5),
    (r"\b(?:52[- ]week|multi[- ]year|record)\s+lows?\b", "fin_low", -1.5),
    (r"\b(?:job cuts|cuts? jobs|cuts? \d[\d,]* jobs|mass layoffs)\b", "fin_layoffs", -1.3),
    (r"\bprofit warning\b", "fin_profit_warning", -2.8),
    (r"\b(?:accounting|audit)\s+(?:irregularities|probe|concerns|restatement)\b|\brestat(?:es|ed|ement)\b", "fin_accounting", -2.8),
    (r"\bfda\s+(?:approval|approves|clears)\b", "fin_fda_ok", 2.2),
    (r"\bfda\s+(?:rejects|rejection|declines|crl|complete response letter)\b", "fin_fda_no", -2.6),
    # social slang (unambiguous)
    (r"\bto the moon\b", "slang_moon", 3.0),
    (r"\brug[- ]?pull(?:ed)?\b", "slang_rug", -3.0),
    (r"\bdiamond hands\b", "slang_diamond", 1.8),
    (r"\bpaper hands\b", "slang_paper", -0.8),
    (r"\bshort squeeze\b|\bgamma squeeze\b", "slang_squeeze", 1.6),
    (r"\bbuy(?:ing)? the dip\b|\bbtfd\b", "slang_btd", 1.2),
    (r"\bdead cat bounce\b", "slang_dcb", -1.6),
    (r"\bbull trap\b", "slang_bulltrap", -1.6),
    (r"\bbear trap\b", "slang_beartrap", 1.0),
]
_PHRASES = [(re.compile(p, re.I), f" {tok} ", v) for p, tok, v in PHRASES]
LEXICON.update({tok: v for _, tok, v in PHRASES})
SLANG = _lex("""
moon mooning moonshot: 2.2
rocket rockets: 2.0
tendies: 1.8
squeeze squeezing: 1.2
yolo: 0.6
stonks: 0.8
lambo: 1.6
bagholder bagholders bagholding bagholders: -1.9
guh: -2.6
dump dumping dumped: -1.8
rekt wrecked: -2.4
bleeding: -1.8
""")
SLANG.update({"\U0001F680": 2.0, "\U0001F315": 1.8, "\U0001F4C8": 1.4, "\U0001F402": 1.4, "\U0001F48E": 1.2,
              "\U0001F4C9": -1.4, "\U0001F43B": -1.4, "\U0001F480": -1.0, "\U0001F921": -1.2, "\U0001FA78": -1.6})
LEXICON.update(SLANG)
# Ambiguous in news prose ("CEO calls for...", "puts pressure on"): only scored for social posts.
SOCIAL_ONLY = _lex("""
calls call: 1.2
puts put: -1.2
long: 0.8
short shorting shorted: -0.9
""")
NEGATE = {"not", "no", "never", "nor", "neither", "without", "cannot", "cant", "dont", "doesnt", "didnt", "isnt",
          "wasnt", "arent", "wont", "fails", "failed", "fail", "unable", "hardly", "barely"}
BOOST = {w: 0.293 for w in "very extremely hugely massively sharply significantly strongly deeply highly huge massive major "
         "really so most totally absolutely substantially dramatically incredibly exceptionally big biggest".split()}
BOOST.update({w: -0.293 for w in "slightly somewhat marginally modestly mildly little partly kinda sorta less".split()})
CONTRAST = {"but", "however"}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'\-]*|\$[A-Za-z]+|[^\w\s!?.,;:()\"\[\]/]")


def tokens(text: str) -> list[str]:
    t = text
    for rx, tok, _ in _PHRASES:
        t = rx.sub(tok, t)
    return _TOKEN.findall(t)


def score(text: str, social: bool = False) -> float:
    """VADER-style compound score in [-1, 1]."""
    if not text or not isinstance(text, str):
        return 0.0
    words = tokens(text)
    lex = {**LEXICON, **SOCIAL_ONLY} if social else LEXICON
    alpha = [w for w in words if w.isalpha() and len(w) > 1]
    mixed_case = any(w.isupper() for w in alpha) and not all(w.isupper() for w in alpha)
    vals = []
    for i, w in enumerate(words):
        lw = w.lower().replace("'", "")
        v = lex.get(lw)
        if v is None or lw in BOOST:
            vals.append(0.0)
            continue
        sgn = 1.0 if v > 0 else -1.0
        if mixed_case and w.isupper() and w.isalpha() and len(w) > 1:
            v += 0.733 * sgn
        negated = False
        for j, decay in ((1, 1.0), (2, 0.95), (3, 0.9)):
            if i - j < 0:
                break
            p = words[i - j].lower().replace("'", "")
            if p in BOOST:
                v += BOOST[p] * sgn * decay
            if p in NEGATE or p.endswith("nt") and p[:-2] + "n't" == words[i - j].lower():
                negated = True
            if p in CONTRAST:
                break
        if negated:
            v *= -0.74
        vals.append(v)
    lw_all = [w.lower() for w in words]
    for c in CONTRAST:
        if c in lw_all:
            k = lw_all.index(c)
            vals = [v * 0.5 for v in vals[:k]] + vals[k:k + 1] + [v * 1.5 for v in vals[k + 1:]]
            break
    s = float(sum(vals))
    if s:
        s += math.copysign(min(text.count("!"), 4) * 0.292, s)
    return float(np.clip(s / math.sqrt(s * s + 15), -1, 1))


LABEL_SCORE = {"bullish": 0.6, "bearish": -0.6}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add compound (sentiment), text_score, events and uid columns."""
    df = df.copy()
    if df.empty:
        for c, d in (("compound", float), ("text_score", float), ("events", object), ("uid", object)):
            df[c] = pd.Series(dtype=d)
        return df
    social = df["source"].isin(SOCIAL)
    text = df["title"].fillna("")
    text = text.where(~social, text + " " + df["summary"].fillna("").str[:280])
    df["text_score"] = [score(t, s) for t, s in zip(text, social)]
    lab = df["label"].fillna("").str.lower().map(LABEL_SCORE)
    df["compound"] = lab.fillna(df["text_score"]).astype(float)
    df["events"] = [",".join(tag_events(t)) for t in df["title"].fillna("")]
    df["uid"] = _uid(df)
    return df


# =============================================================================== event tagging
EVENT_RULES: dict[str, str] = {
    "earnings": r"\b(earnings|eps|quarterly (results|profit|revenue)|q[1-4] (results|earnings|revenue|sales)|results|fin_beat|fin_miss|revenue|profit)\b",
    "guidance": r"\b(guidance|outlook|fin_guide_up|fin_guide_down|forecasts?|profit warning|fin_profit_warning)\b",
    "m&a": r"\b(acquir(e|es|ed|ing)|acquisition|merger|merge|takeover|buyout|tender offer|bid for|(agrees?|deal|plans?|offers?|bids?|moves?|sets?) to (buy|acquire)|stake in|spin[- ]?off|divest)",
    "analyst": r"\b(upgrades?|upgraded|downgrades?|downgraded|price target|fin_pt_up|fin_pt_down|initiat(es|ed) coverage|overweight|underweight|outperform|underperform|reiterat(es|ed))\b",
    "regulatory_legal": r"\b(sec|doj|ftc|fca|cma|antitrust|lawsuit|sues?|sued|probe|investigation|fined?|settlement|settles|court|ruling|regulators?|fin_reg_probe|fin_class_action|subpoena|recalls?)\b",
    "management": r"\b(ceo|cfo|coo|chairman|chief executive|president)\b.*\b(resign|resigns|steps down|step down|departs|appointed|appoints|names|hired|fired|ousted|retire|retires|successor)\b"
                  r"|\b(resigns|steps down|appoints|names)\b.*\b(ceo|cfo|chairman|chief executive)\b",
    "product": r"\b(launch(es|ed)?|unveil(s|ed)?|releases?|new (product|model|chip|phone|drug)|fda|fin_fda_ok|fin_fda_no|patent|rollout|recalls?)\b",
    "macro": r"\b(fed|fomc|rate (cut|hike)s?|interest rates?|inflation|cpi|ppi|payrolls|jobs report|gdp|tariffs?|recession|ecb|boe|bank of england|treasury yields?|central bank)\b",
    "capital_return": r"\b(buybacks?|repurchase|dividends?|fin_buyback|fin_div_up|fin_div_down|special dividend)\b",
    "short_report": r"\b(fin_short_report|hindenburg|muddy waters|citron|gotham city|spruce point|culper|kerrisdale|grizzly research|blue orca|wolfpack)\b",
    "insider_trade": r"\b(insider (buying|selling|buys|sells|purchase|sale)s?|form 4|(ceo|cfo|director|chairman|founder|insider)s? (buys|sells|sold|bought|unloads|dumps|offloads))\b",
}
_EVENTS = {k: re.compile(v, re.I) for k, v in EVENT_RULES.items()}
_UP = re.compile(r"\b(upgrades?|upgraded|fin_pt_up|raises? (its )?price target|to (buy|overweight|outperform))\b", re.I)
_DOWN = re.compile(r"\b(downgrades?|downgraded|fin_pt_down|cuts? (its )?price target|to (sell|underweight|underperform|neutral|hold))\b", re.I)


def tag_events(title: str) -> list[str]:
    """Keyword-rule event classification; analyst changes carry a direction (analyst_up / analyst_down)."""
    if not title:
        return []
    t = " ".join(tokens(title))
    out = [k for k, rx in _EVENTS.items() if rx.search(t)]
    if "analyst" in out:
        d = "analyst_up" if _UP.search(t) and not _DOWN.search(t) else "analyst_down" if _DOWN.search(t) else None
        if d:
            out[out.index("analyst")] = d
    return out


# =============================================================================== aggregation
_STOP = set("a an the to of in on for and or at by with from as is are be its it this that after amid over up down vs".split())


def norm_title(t: str) -> str:
    t = re.sub(r"\s+[-|–]\s+[^-|–]{2,40}$", "", t or "")  # trailing " - Publisher"
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(w for w in t.split() if w not in _STOP)


def similar(a: str, b: str, jaccard: float = 0.6, ratio: float = 0.85) -> bool:
    if not a or not b:
        return False
    sa, sb = set(a.split()), set(b.split())
    if sa and sb and len(sa & sb) / len(sa | sb) >= jaccard:
        return True
    return SequenceMatcher(None, a, b).ratio() >= ratio


def dedupe(df: pd.DataFrame, max_items: int = 1500) -> pd.DataFrame:
    """Collapse near-identical headlines across sources. Keeps the highest-weight source's row and records
    n_sources/also_in. Social posts are not merged (each post is a separate voice)."""
    if df.empty:
        out = df.copy()
        out["n_sources"], out["also_in"] = pd.Series(dtype=int), pd.Series(dtype=object)
        return out
    df = df.copy()
    df["_w"] = df["source"].map(SOURCE_WEIGHT).fillna(0.5)
    df["_n"] = df["title"].map(norm_title)
    news = df[~df["source"].isin(SOCIAL)].sort_values(["_w", "time"], ascending=[False, False]).head(max_items)
    keep, groups = [], []
    for idx, n in news["_n"].items():
        for g in groups:
            if similar(n, g["n"]):
                g["members"].append(idx)
                break
        else:
            groups.append({"n": n, "members": [idx]})
    for g in groups:
        lead = g["members"][0]
        srcs = sorted(set(news.loc[g["members"], "source"]))
        keep.append((lead, len(g["members"]), ",".join(s for s in srcs if s != news.at[lead, "source"])))
    k = pd.DataFrame(keep, columns=["_i", "n_sources", "also_in"]).set_index("_i")
    out = df.loc[k.index].join(k)
    soc = df[df["source"].isin(SOCIAL)].assign(n_sources=1, also_in="")
    out = pd.concat([out, soc]).drop(columns=["_w", "_n"])
    return out.sort_values("time", ascending=False).reset_index(drop=True)


def daily_index(items: pd.DataFrame, timeline: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per UTC day: mean compound (news+social), mentions, social bull ratio (+ GDELT tone/volume when given)."""
    cols = ["sentiment", "news_sentiment", "mentions", "social_mentions", "bull_ratio"]
    if items is not None and len(items) and items["time"].notna().any():
        d = items.dropna(subset=["time"]).copy()
        d["day"] = d["time"].dt.floor("D")
        soc = d[d["source"].isin({"stocktwits", "reddit"})].copy()
        lab = soc["label"].fillna("").str.lower()
        soc["bull"] = np.where(lab == "bullish", 1.0, np.where(lab == "bearish", 0.0,
                               np.where(soc["compound"] > 0.05, 1.0, np.where(soc["compound"] < -0.05, 0.0, np.nan))))
        g = d.groupby("day")
        out = pd.DataFrame({"sentiment": g["compound"].mean(), "mentions": g.size()})
        out["news_sentiment"] = d[~d["source"].isin(SOCIAL)].groupby("day")["compound"].mean()
        out["social_mentions"] = soc.groupby("day").size()
        out["bull_ratio"] = soc.groupby("day")["bull"].mean()
        out = out.asfreq("D")
        out[["mentions", "social_mentions"]] = out[["mentions", "social_mentions"]].fillna(0)
    else:
        out = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="day"), dtype=float)
    out = out.reindex(columns=cols)
    if timeline is not None and len(timeline):
        tl = timeline.copy()
        tl.index = pd.DatetimeIndex(tl.index).tz_convert("UTC") if tl.index.tz is not None else pd.DatetimeIndex(tl.index).tz_localize("UTC")
        out = out.join(tl, how="outer")
        if len(out):
            out = out.asfreq("D")
        out[["mentions", "social_mentions"]] = out[["mentions", "social_mentions"]].fillna(0)
    out.index.name = "day"
    vol = out["gdelt_volume"] if "gdelt_volume" in out and out["gdelt_volume"].notna().sum() >= 10 else out["mentions"]
    out["attention_z"] = attention_z(vol)
    return out


def attention_z(volume: pd.Series, window: int = 30, min_periods: int = 10) -> pd.Series:
    """z-score of today's mention volume vs the trailing `window` days (today excluded)."""
    v = volume.astype(float)
    base = v.shift(1).rolling(window, min_periods=min_periods)
    sd = base.std().replace(0, np.nan)
    return (v - base.mean()) / sd


def _sessions(daily: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Map calendar-day signals onto the next trading session (weekend news counts for Monday)."""
    td = pd.DatetimeIndex(trading_days).normalize()
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None) if d.index.tz is not None else d.index
    pos = td.searchsorted(d.index.normalize())
    ok = pos < len(td)
    d = d[ok]
    d.index = td[pos[ok]]
    agg = {c: ("sum" if c in ("mentions", "social_mentions", "gdelt_volume") else "mean") for c in d.columns}
    return d.groupby(level=0).agg(agg)


def sentiment_vs_price(symbol: str, days: int = 90, daily: pd.DataFrame | None = None, prices: pd.DataFrame | None = None,
                       lags: range = range(-1, 6), store: bool = True) -> dict:
    """Join daily sentiment/attention with returns. lead_lag[signal][k] = corr(signal_t, return_{t+k}); k=-1 tests
    whether sentiment just follows yesterday's price. Contrarian extremes: crowd bull ratio at its high + RSI > 70
    (or capitulation + RSI < 30)."""
    from . import ta
    if daily is None:
        daily = sentiment_index(symbol, days, store=store)["daily"]
    if prices is None:
        from .data import get_prices
        prices = get_prices(symbol, "1d", period="1y")
    px = prices.copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None) if px.index.tz is not None else pd.DatetimeIndex(px.index)
    px.index = px.index.normalize()
    close = px["close"]
    j = _sessions(daily, px.index).join(pd.DataFrame({"close": close, "ret": close.pct_change(), "rsi": ta.rsi(close)}), how="right")
    j = j[j.index >= (daily.index.min().tz_localize(None) if daily.index.tz is not None else daily.index.min())] if len(daily) else j.iloc[0:0]
    signals = [c for c in ("sentiment", "news_sentiment", "gdelt_tone", "bull_ratio") if c in j and j[c].notna().sum() >= 5]
    rows = {}
    pairs = [(c, j[c], j["ret"]) for c in signals]
    if "attention_z" in j and j["attention_z"].notna().sum() >= 5:
        pairs.append(("attention_z~|ret|", j["attention_z"], j["ret"].abs()))
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, sig, ret in pairs:
            rows[name] = {f"t{k:+d}": sig.corr(ret.shift(-k)) for k in lags}
            rows[name]["n"] = int(sig.notna().sum())
    lead_lag = pd.DataFrame(rows).T
    ext = pd.DataFrame()
    if "bull_ratio" in j and j["bull_ratio"].notna().sum() >= 3:
        br = j["bull_ratio"]
        hi, lo = max(0.75, br.quantile(0.9)), min(0.35, br.quantile(0.1))
        flag = np.where((br >= hi) & (j["rsi"] >= 70), "crowd max bullish + RSI>70 (contrarian bearish)",
                        np.where((br <= lo) & (j["rsi"] <= 30), "crowd capitulation + RSI<30 (contrarian bullish)", ""))
        ext = j.assign(flag=flag)[lambda x: x["flag"] != ""][["close", "rsi", "bull_ratio", "flag"]]
    return {"symbol": symbol, "lead_lag": lead_lag, "joined": j, "extremes": ext,
            "note": "corr(signal_t, return_t+k) on trading days; weekend items roll to the next session"}


# =============================================================================== history store
def _store(key: str, items: pd.DataFrame) -> pd.DataFrame:
    """Append items to the parquet history (data/cache/news/items/<KEY>.parquet) so attention builds up."""
    from . import cache
    if items.empty:
        return items
    keep = [c for c in COLUMNS + ["compound", "text_score", "events", "uid"] if c in items]
    d = items[keep].dropna(subset=["time"]).set_index(["time", "uid"])
    return cache.save("news", key, "items", d).reset_index()


def history(key: str) -> pd.DataFrame:
    from . import cache
    h = cache.load("news", key, "items")
    return empty() if h is None else h.reset_index()


def _merge_history(qc: Query, items: pd.DataFrame, store: bool) -> pd.DataFrame:
    try:
        allx = _store(qc.key, items) if store else pd.concat([history(qc.key), items], ignore_index=True)
    except Exception:
        allx = items
    if "uid" in allx:
        allx = allx.drop_duplicates("uid", keep="last")
    return allx


def _timeline_safe(qc: Query, days: int, errors: dict) -> pd.DataFrame | None:
    try:
        return gdelt_timeline(qc, days)
    except Exception as e:
        errors["gdelt_timeline"] = f"{type(e).__name__}: {str(e)[:140]}"
        return None


def sentiment_index(q: str, days: int = 30, sources=None, store: bool = True) -> dict:
    """Fetch, score, store and aggregate: {'daily', 'items', 'errors'}."""
    qc = make_query(q, days)
    errors: dict = {}
    with cf.ThreadPoolExecutor(2) as ex:
        ft = ex.submit(_timeline_safe, qc, max(90, days), errors) if (sources is None or "gdelt" in sources) else None
        items, errs = collect(qc, sources)
        tl = ft.result() if ft else None
    errors.update(errs)
    items = enrich(items)
    allx = _merge_history(qc, items, store)
    return {"query": qc, "daily": daily_index(allx, tl), "items": items, "errors": errors}


# =============================================================================== brief
def rank_headlines(df: pd.DataFrame, now: pd.Timestamp | None = None, half_life_h: float = 24) -> pd.DataFrame:
    """rank = recency (half-life decay) x (0.3 + |compound|) x source weight x coverage bonus."""
    if df.empty:
        return df.assign(rank=pd.Series(dtype=float))
    now = now or pd.Timestamp.now(tz="UTC")
    age = ((now - df["time"]).dt.total_seconds() / 3600).fillna(24 * 7).clip(lower=0)
    cov = 1 + 0.25 * (df.get("n_sources", pd.Series(1, index=df.index)).fillna(1) - 1)
    r = 0.5 ** (age / half_life_h) * (0.3 + df["compound"].abs()) * df["source"].map(SOURCE_WEIGHT).fillna(0.5) * cov
    return df.assign(rank=r).sort_values("rank", ascending=False)


def _window_mean(daily: pd.DataFrame, col: str, days: int, now: pd.Timestamp) -> float:
    if col not in daily or daily.empty:
        return float("nan")
    s = daily.loc[daily.index >= now.floor("D") - pd.Timedelta(days=days - 1), col]
    return float(s.mean()) if s.notna().any() else float("nan")


def brief(q: str, days: int = 7, sources=None, top: int = 12, store: bool = True) -> dict:
    """One-page news/sentiment brief for a ticker or topic. Never raises because a source failed."""
    now = pd.Timestamp.now(tz="UTC")
    res = sentiment_index(q, days, sources, store=store)
    qc, items, daily, errors = res["query"], res["items"], res["daily"], res["errors"]
    dd = dedupe(items)
    news = dd[~dd["source"].isin(SOCIAL)]
    heads = rank_headlines(news, now).head(top)
    soc = items[items["source"].isin({"stocktwits", "reddit"})]
    lab = soc["label"].fillna("").str.lower() if len(soc) else pd.Series(dtype=str)
    bulls = int((lab == "bullish").sum() + ((lab == "") & (soc["compound"] > 0.05)).sum()) if len(soc) else 0
    bears = int((lab == "bearish").sum() + ((lab == "") & (soc["compound"] < -0.05)).sum()) if len(soc) else 0
    ev = news.assign(event=news["events"].str.split(",")).explode("event") if len(news) else news.assign(event=None)
    ev = ev[ev["event"].fillna("") != ""]
    events = ev.groupby("event").agg(count=("title", "size"), avg_sentiment=("compound", "mean"),
                                     latest=("title", "first")).sort_values("count", ascending=False) if len(ev) else pd.DataFrame()
    az = daily["attention_z"].dropna() if "attention_z" in daily else pd.Series(dtype=float)
    last_z = float(az.iloc[-1]) if len(az) else float("nan")
    out = {
        "query": qc.query, "name": qc.name, "as_of": now.strftime("%Y-%m-%d %H:%M UTC"),
        "items": int(len(items)), "unique_headlines": int(len(news)),
        "sentiment_7d": _window_mean(daily, "sentiment", 7, now), "sentiment_30d": _window_mean(daily, "sentiment", 30, now),
        "gdelt_tone_7d": _window_mean(daily, "gdelt_tone", 7, now), "gdelt_tone_30d": _window_mean(daily, "gdelt_tone", 30, now),
        "attention_z": last_z, "attention_spike": bool(last_z >= 2) if last_z == last_z else False,
        "social_posts": int(len(soc)), "social_bull_ratio": bulls / (bulls + bears) if bulls + bears else float("nan"),
        "headlines": heads[["time", "source", "compound", "events", "n_sources", "title", "url"]].assign(
            time=heads["time"].dt.strftime("%m-%d %H:%M")).set_index("time") if len(heads) else pd.DataFrame(),
        "events": events,
        "errors": errors,
        "sources_ok": sorted(set(items["source"])) if len(items) else [],
    }
    return out


# =============================================================================== CLI
def _social(q: str, days: int = 7, store: bool = True) -> dict:
    qc = make_query(q, days)
    items, errors = collect(qc, ["stocktwits", "reddit"])
    items = enrich(items)
    if store:
        try:
            _store(qc.key, items)
        except Exception:
            pass
    rows = []
    for src, g in items.groupby("source"):
        lab = g["label"].fillna("").str.lower()
        bull = (lab == "bullish") | ((lab == "") & (g["compound"] > 0.05))
        bear = (lab == "bearish") | ((lab == "") & (g["compound"] < -0.05))
        rows.append({"source": src, "posts": len(g), "tagged": int((lab != "").sum()), "bullish": int(bull.sum()),
                     "bearish": int(bear.sum()), "bull_ratio": bull.sum() / max(1, bull.sum() + bear.sum()),
                     "mean_compound": g["compound"].mean(), "engagement": g["engagement"].sum()})
    top = items.sort_values("engagement", ascending=False).head(15) if len(items) else items
    return {"summary": pd.DataFrame(rows).set_index("source") if rows else pd.DataFrame(),
            "top": top[["time", "source", "author", "engagement", "label", "compound", "title"]].set_index("time") if len(top) else pd.DataFrame(),
            "errors": errors}


def _srcs(a):
    return [s.strip() for s in a.sources.split(",")] if a.sources else None


def cmd_news(a):
    from .cli import show
    q = " ".join(a.query)
    r = brief(q, a.days, _srcs(a), a.top, store=not a.no_store)
    if a.json:
        return show(r, True)
    show({k: v for k, v in r.items() if k not in ("headlines", "events", "errors", "sources_ok")}, title=f"News brief: {q}")
    show(r["headlines"], title="Top headlines (recency x |sentiment| x source weight)")
    show(r["events"], title="Event tags")
    if r["errors"]:
        show(pd.DataFrame({"error": r["errors"]}), title="Unavailable sources")
    print(f"\n_Sources: {', '.join(r['sources_ok']) or 'none'} · {r['as_of']}_")


def cmd_sentiment(a):
    from .cli import show
    r = sentiment_index(a.symbol, a.days, _srcs(a), store=not a.no_store)
    d = r["daily"]
    show(d.tail(a.tail), a.json, title=f"{a.symbol} daily sentiment / attention")
    try:
        vp = sentiment_vs_price(a.symbol, a.days, daily=d)
        show(vp["lead_lag"], a.json, title="Lead/lag: corr(signal_t, return_t+k)")
        show(vp["extremes"], a.json, title="Contrarian extremes")
        print(f"\n_{vp['note']}_")
    except Exception as e:
        print(f"\n(price join unavailable: {type(e).__name__}: {str(e)[:120]})")
    if r["errors"]:
        show(pd.DataFrame({"error": r["errors"]}), a.json, title="Unavailable sources")


def cmd_social(a):
    from .cli import show
    r = _social(a.symbol, a.days, store=not a.no_store)
    show(r["summary"], a.json, title=f"{a.symbol} social (StockTwits + Reddit)")
    show(r["top"], a.json, title="Top posts by engagement")
    if r["errors"]:
        show(pd.DataFrame({"error": r["errors"]}), a.json, title="Unavailable sources")


def register(add):
    q = add("news", cmd_news, "news brief: headlines, sentiment, events, attention for a ticker or topic")
    q.add_argument("query", nargs="+"); q.add_argument("--days", type=int, default=7); q.add_argument("--top", type=int, default=12)
    q.add_argument("--sources", help=f"comma list from: {','.join(FETCHERS)}"); q.add_argument("--no-store", action="store_true")
    q = add("sentiment", cmd_sentiment, "daily sentiment index + attention z-score + lead/lag vs price")
    q.add_argument("symbol"); q.add_argument("--days", type=int, default=30); q.add_argument("--tail", type=int, default=20)
    q.add_argument("--sources"); q.add_argument("--no-store", action="store_true")
    q = add("social", cmd_social, "StockTwits + Reddit crowd: bull ratio, top posts")
    q.add_argument("symbol"); q.add_argument("--days", type=int, default=7); q.add_argument("--no-store", action="store_true")
