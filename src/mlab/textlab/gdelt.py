"""GDELT 2.0 GKG ingester (#52): 15-minute raw files -> point-in-time Parquet, no BigQuery quota.

    mlab newsstore gdelt --since 2026-09-10            # backfill, resumable, then keep running it
    mlab newsstore gdelt --since 2d --workers 4    # relative: last 2 days
    mlab newsstore show AMZN --since 2026-10-01         # daily article count and tone for a symbol

Two tables under $MLAB_DATA_DIR/data/news/gdelt/:
- firm/date=YYYY-MM-DD/<ts>.parquet: one row per article that names a watched company: avail (the file's 15-minute
  timestamp, when GDELT published it, which is the earliest we could have known), symbol, via (org:/platform:/
  person: match), source, url, title, tone,
  pos, neg, polarity, words, themes (V1 themes, ';'-joined), orgs (all organisations in the article).
- macro/date=YYYY-MM-DD/<ts>.parquet: per file, article counts and mean tone per ECON_/EPU_ theme plus "_ALL".

GDELT is free for academic, commercial and governmental use with a citation and a link to
https://www.gdeltproject.org (https://www.gdeltproject.org/about.html).
"""
from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR

BASE = "https://data.gdeltproject.org/gdeltv2"
STORE = Path(DATA_DIR) / "data" / "news" / "gdelt"
CITATION = "GDELT Project (https://www.gdeltproject.org)"

# GKG org names are lower-case free text. Aliases match a whole org entry or its leading words.
WATCH = {
    "AAPL": ["apple", "apple inc"], "MSFT": ["microsoft"], "NVDA": ["nvidia"],
    "AMZN": ["amazon", "amazon com", "amazon web services", "amazon web service", "amazon prime"],
    "GOOGL": ["google", "alphabet", "alphabet inc", "youtube"], "META": ["meta platforms", "facebook", "instagram"],
    "TSLA": ["tesla", "tesla inc", "tesla motors"], "AVGO": ["broadcom"], "BRK-B": ["berkshire hathaway"],
    "JPM": ["jpmorgan", "jpmorgan chase", "jp morgan", "j p morgan"], "LLY": ["eli lilly", "eli lilly and company"],
    "V": ["visa inc"], "MA": ["mastercard"], "XOM": ["exxon", "exxon mobil", "exxonmobil"],
    "UNH": ["unitedhealth", "unitedhealth group", "unitedhealthcare"], "COST": ["costco"], "NFLX": ["netflix"],
    "AMD": ["advanced micro devices", "amd"], "ORCL": ["oracle"], "WMT": ["walmart", "wal mart"],
    "PLTR": ["palantir", "palantir technologies"], "CRM": ["salesforce"], "ADBE": ["adobe"], "INTC": ["intel"],
    "BAC": ["bank of america"],
}
# Chief executives (GKG Persons). Kept apart from org matches in the `via` column: a CEO story is often about
# the person (Musk and politics), so analyses choose whether to include them.
PEOPLE = {"tim cook": "AAPL", "satya nadella": "MSFT", "jensen huang": "NVDA", "andy jassy": "AMZN",
          "sundar pichai": "GOOGL", "mark zuckerberg": "META", "elon musk": "TSLA", "hock tan": "AVGO",
          "warren buffett": "BRK-B", "greg abel": "BRK-B", "jamie dimon": "JPM", "david ricks": "LLY",
          "ryan mcinerney": "V", "michael miebach": "MA", "darren woods": "XOM", "ron vachris": "COST",
          "ted sarandos": "NFLX", "greg peters": "NFLX", "lisa su": "AMD", "safra catz": "ORCL",
          "doug mcmillon": "WMT", "john furner": "WMT", "alex karp": "PLTR", "marc benioff": "CRM",
          "shantanu narayen": "ADBE", "lip-bu tan": "INTC", "brian moynihan": "BAC"}
# Platform names: an article that only says "posted on Facebook" is not news about the company.
PLATFORM = {"facebook", "instagram", "youtube", "google", "amazon prime", "whatsapp"}
_ALIAS = {a: s for s, al in WATCH.items() for a in al}
DENY = {"apple daily", "apple valley", "apple records", "amazon rainforest", "amazon basin", "oracle arena",
        "visa office"}  # org names that start with an alias but are not the company
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
COLS = ["GKGRECORDID", "DATE", "SRC_COLLECTION", "SOURCE", "URL", "COUNTS", "V2COUNTS", "THEMES", "V2THEMES",
        "LOCATIONS", "V2LOCATIONS", "PERSONS", "V2PERSONS", "ORGS", "V2ORGS", "TONE", "DATES", "GCAM", "IMAGE",
        "RELATED_IMAGES", "SOCIAL_IMAGES", "SOCIAL_VIDEOS", "QUOTATIONS", "ALLNAMES", "AMOUNTS", "TRANSLATION",
        "EXTRAS"]


def match(orgs: str, persons: str = "") -> dict[str, str]:
    """{symbol: via} where via is 'org:<alias>', 'platform:<alias>' or 'person:<name>' (org beats the others)."""
    out: dict[str, str] = {}
    for o in orgs.split(";"):
        o = o.strip()
        if o in DENY:
            continue
        while o:
            if o in _ALIAS:
                kind = "platform" if o in PLATFORM else "org"
                if out.get(_ALIAS[o], "").split(":")[0] != "org":
                    out[_ALIAS[o]] = f"{kind}:{o}"
                break
            o = o.rsplit(" ", 1)[0] if " " in o else ""
    for p in persons.split(";"):
        sym = PEOPLE.get(p.strip())
        if sym and sym not in out:
            out[sym] = f"person:{p.strip()}"
    return out


def match_symbols(orgs: str, persons: str = "") -> set[str]:
    return set(match(orgs, persons))


def parse(blob: bytes, ts: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(firm rows, macro theme rows) from one zipped GKG file."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    rows = [r for r in (line.split("\t") for line in raw.splitlines()) if len(r) == len(COLS)]
    firm, macro = [], {}
    for r in rows:
        tone = (r[15].split(",") + [""] * 7)[:7]
        try:
            t, pos, neg, pol, words = float(tone[0]), float(tone[1]), float(tone[2]), float(tone[3]), float(tone[6])
        except ValueError:
            continue
        for th in {"_ALL", *(x for x in r[7].split(";") if x.startswith(("ECON_", "EPU_")))}:
            n, s = macro.get(th, (0, 0.0))
            macro[th] = (n + 1, s + t)
        syms = match(r[13], r[11]) if (r[13] or r[11]) else {}
        if syms:
            m = _TITLE.search(r[26])
            for sym, via in syms.items():
                firm.append({"avail": ts, "symbol": sym, "via": via, "source": r[3], "url": r[4],
                             "title": m.group(1).strip() if m else "", "tone": t, "pos": pos, "neg": neg,
                             "polarity": pol, "words": words, "themes": r[7], "orgs": r[13]})
    f = pd.DataFrame(firm)
    mc = pd.DataFrame([{"avail": ts, "theme": k, "n": n, "tone_mean": s / n} for k, (n, s) in macro.items()])
    return f, mc


def stamps(since, until=None) -> list[pd.Timestamp]:
    now = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15)
    def _ts(x, default):
        if x is None:
            return default
        if isinstance(x, str) and re.fullmatch(r"-?\d+[dhm]", x):  # relative: 3d, 12h, 90m (or -3d)
            return now - pd.Timedelta(x.lstrip("-"))
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tz is None else t
    lo, hi = _ts(since, now - pd.Timedelta(days=1)).floor("15min"), min(_ts(until, now), now)
    return list(pd.date_range(lo, hi, freq="15min"))


def _paths(ts: pd.Timestamp) -> tuple[Path, Path]:
    d, name = f"date={ts:%Y-%m-%d}", f"{ts:%Y%m%d%H%M%S}.parquet"
    return STORE / "firm" / d / name, STORE / "macro" / d / name


def ingest_one(ts: pd.Timestamp, session=None) -> str:
    fp, mp = _paths(ts)
    if mp.exists():
        return "skip"
    from ..net import session as mk
    s = session or mk("market-lab/0.1")
    r = s.get(f"{BASE}/{ts:%Y%m%d%H%M%S}.gkg.csv.zip", timeout=60)
    if r.status_code == 404:
        return "missing"
    r.raise_for_status()
    f, mc = parse(r.content, ts)
    fp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    if len(f):
        f.to_parquet(fp.with_suffix(".tmp"), index=False)
        fp.with_suffix(".tmp").rename(fp)
    mc.to_parquet(mp.with_suffix(".tmp"), index=False)  # macro written last: it marks the file as done
    mp.with_suffix(".tmp").rename(mp)
    return f"ok {len(f)}"


def ingest(since, until=None, workers: int = 4, log=print) -> dict:
    todo = [t for t in stamps(since, until) if not _paths(t)[1].exists()]
    log(f"gdelt: {len(todo)} files to fetch into {STORE}")
    out = {"ok": 0, "missing": 0, "failed": 0, "articles": 0}

    def one(t):
        try:
            return t, ingest_one(t)
        except Exception as e:  # count and carry on; a rerun picks the file up again
            return t, f"failed {type(e).__name__}: {str(e)[:80]}"
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, (t, res) in enumerate(ex.map(one, todo)):
            k = res.split()[0]
            out[k if k in out else "failed"] += 1
            if k == "ok":
                out["articles"] += int(res.split()[1])
            elif k == "failed":
                log(f"  {t:%Y-%m-%d %H:%M} {res}")
            if i and i % 96 == 0:
                log(f"  {i}/{len(todo)} files, {out['articles']} watched-company articles")
    return out


def load(kind: str = "firm", symbols: list[str] | None = None, since=None, until=None) -> pd.DataFrame:
    root = STORE / kind
    if not root.exists():
        return pd.DataFrame()
    days = sorted(root.glob("date=*"))
    if since is not None:
        days = [d for d in days if d.name[5:] >= str(pd.Timestamp(since).date())]
    if until is not None:
        days = [d for d in days if d.name[5:] <= str(pd.Timestamp(until).date())]
    files = [f for d in days for f in d.glob("*.parquet")]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if symbols and kind == "firm":
        df = df[df["symbol"].isin(symbols)]
    df.attrs["source"] = f"{CITATION} GKG 2.0 raw files, {root}"
    return df.sort_values("avail").reset_index(drop=True)


def daily(symbol: str, since=None, until=None, via=("org",)) -> pd.DataFrame:
    """Per-day article count, unique sources and mean tone for one symbol (UTC days of availability), counting
    only matches of the given kinds (default: company-name matches, not platform or CEO mentions)."""
    df = load("firm", [symbol], since, until)
    if df.empty:
        return df
    if via and "via" in df:
        df = df[df["via"].str.split(":").str[0].isin(via)]
    df = df.drop_duplicates(["symbol", "url"])
    g = df.groupby(df["avail"].dt.floor("D"))
    return pd.DataFrame({"articles": g.size(), "sources": g["source"].nunique(), "tone": g["tone"].mean().round(2),
                         "neg_share": g["tone"].apply(lambda x: (x < -2).mean()).round(2)})


# ---- CLI ------------------------------------------------------------------------------------
def cmd_newsstore(a):
    from ..cli import show
    if a.action == "gdelt":
        print(ingest(a.since or "1d", a.until, a.workers))
        return
    if a.action == "show":
        if not a.symbol:
            raise SystemExit("symbol required")
        df = daily(a.symbol.upper(), a.since, a.until)
        if df.empty:
            print(f"no GDELT articles stored for {a.symbol} (run `mlab newsstore gdelt --since ...` first)")
            return
        show(df.reset_index(names="day"), a.json, title=f"{a.symbol.upper()} GDELT daily · {CITATION}")


def register(add):
    q = add("newsstore", cmd_newsstore, "point-in-time news store: gdelt (ingest GKG raw files) | show SYMBOL")
    q.add_argument("action", choices=["gdelt", "show"])
    q.add_argument("symbol", nargs="?")
    q.add_argument("--since", help="start: date, timestamp or relative like 3d / 12h (default 1d)")
    q.add_argument("--until", help="end (default: latest published file)")
    q.add_argument("--workers", type=int, default=4)
