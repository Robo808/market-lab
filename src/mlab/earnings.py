"""Earnings desk: earnings history vs price reactions, earnings documents (SEC 8-K Item 2.02 press
releases, call transcripts via Alpha Vantage / FMP / saved files) and lexicon-based text analytics,
joined per quarter by align().

Quarter labels ('2025Q3') are the calendar quarter that most recently ended before the event date
(shift with fiscal_offset for off-calendar filers). Transcripts are cached as
data/transcripts/<TICKER>/<YYYYQn>.txt; SEC press releases as data/transcripts/<TICKER>/press/<filed>.txt.
Network access lives only in the fetch_* functions and the desk-level orchestrators.
"""
from __future__ import annotations

import html as _html
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, env
from .options import jsonable

HOSTS = {  # all already in net.SOURCES
    "query2.finance.yahoo.com": "Yahoo earnings dates / EPS surprise (yfinance)",
    "www.sec.gov": "SEC 8-K filing indexes and exhibit 99.1 press releases",
    "data.sec.gov": "SEC submissions (8-K items)",
    "www.alphavantage.co": "Alpha Vantage EARNINGS_CALL_TRANSCRIPT (ALPHAVANTAGE_API_KEY)",
    "financialmodelingprep.com": "FMP earning call transcripts (FMP_API_KEY)",
}
TRANSCRIPTS_DIR = Path(env("MLAB_TRANSCRIPTS_DIR") or DATA_DIR / "data" / "transcripts")
ET = "America/New_York"


# ---- earnings history ------------------------------------------------------------------
def fetch_earnings_dates(ticker: str, limit: int = 16) -> pd.DataFrame:
    """Raw yfinance get_earnings_dates (index = event timestamp, ET)."""
    import yfinance as yf

    df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if df is None or df.empty:
        raise LookupError(f"yahoo returned no earnings dates for {ticker}")
    return df


def fetch_quarterly_revenue(ticker: str) -> pd.Series:
    """Quarterly total revenue by fiscal period end (yfinance income statement; ~5 quarters)."""
    import yfinance as yf

    inc = yf.Ticker(ticker).quarterly_income_stmt
    if inc is None or inc.empty or "Total Revenue" not in inc.index:
        return pd.Series(dtype=float)
    s = inc.loc["Total Revenue"].astype(float).dropna()
    idx = pd.to_datetime(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    return s.sort_index()


def infer_timing(ts) -> str:
    """BMO (before 09:30 ET), AMC (16:00 ET or later), DMH (during market hours), unknown (midnight / date only)."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize(ET) if ts.tzinfo is None else ts.tz_convert(ET)
    if ts.hour == 0 and ts.minute == 0:
        return "unknown"
    h = ts.hour + ts.minute / 60
    return "BMO" if h < 9.5 else "AMC" if h >= 16 else "DMH"


def quarter_label(date, fiscal_offset: int = 0) -> str:
    """Calendar quarter that most recently ended before `date` (+fiscal_offset quarters), e.g. '2025Q3'."""
    p = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date)
    q = pd.Period(p, "Q") - 1 + fiscal_offset
    return f"{q.year}Q{q.quarter}"


def normalize_earnings(raw: pd.DataFrame, fiscal_offset: int = 0) -> pd.DataFrame:
    """Tidy events: event_time (ET), date, timing, quarter, eps_est, eps_act, surprise_pct (percent), reported."""
    pick = lambda *keys: next((raw[c] for c in raw.columns if any(k in c.lower() for k in keys)), pd.Series(np.nan, index=raw.index))  # noqa: E731
    t = pd.to_datetime(raw.index)
    t = t.tz_localize(ET) if t.tz is None else t.tz_convert(ET)
    df = pd.DataFrame({"event_time": t, "eps_est": pd.to_numeric(pick("estimate").to_numpy(), errors="coerce"),
                       "eps_act": pd.to_numeric(pick("reported", "actual").to_numpy(), errors="coerce"),
                       "surprise_pct": pd.to_numeric(pick("surprise").to_numpy(), errors="coerce")})
    calc = (df["eps_act"] - df["eps_est"]) / df["eps_est"].abs().replace(0, np.nan) * 100
    df["surprise_pct"] = df["surprise_pct"].fillna(calc)
    df["date"] = df["event_time"].dt.tz_localize(None).dt.normalize()
    df["timing"] = [infer_timing(x) for x in df["event_time"]]
    df["quarter"] = [quarter_label(d, fiscal_offset) for d in df["date"]]
    df["reported"] = df["eps_act"].notna()
    df = df.sort_values("event_time").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df[["event_time", "date", "timing", "quarter", "eps_est", "eps_act", "surprise_pct", "reported"]]


def attach_revenue(events: pd.DataFrame, revenue: pd.Series, max_lag_days: int = 100) -> pd.DataFrame:
    """Add revenue of the latest fiscal period ending within max_lag_days before each event, and its YoY."""
    ev = events.copy()
    ev["revenue"], ev["revenue_yoy"] = np.nan, np.nan
    if revenue is None or revenue.empty:
        return ev
    rev = revenue.sort_index()
    yoy = rev / rev.shift(4) - 1
    for i, d in ev["date"].items():
        cand = rev[(rev.index < d) & (rev.index >= d - pd.Timedelta(days=max_lag_days))]
        if len(cand):
            ev.loc[i, "revenue"], ev.loc[i, "revenue_yoy"] = cand.iloc[-1], yoy.get(cand.index[-1], np.nan)
    return ev


# ---- reaction study (pure) -------------------------------------------------------------
def _daily(prices: pd.DataFrame) -> pd.DataFrame:
    """Index prices by naive exchange session date (UTC-stamped Yahoo bars -> ET dates)."""
    df = prices.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_convert(ET).tz_localize(None) if idx.tz is not None else idx).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def event_window(sessions: pd.DatetimeIndex, date, timing: str) -> tuple[int, int]:
    """(pre, reaction) session positions. BMO/DMH: prior close -> event-day close; AMC: event-day close ->
    next-day close; unknown: prior close -> next-day close (spans both possibilities)."""
    d = pd.Timestamp(date).normalize()
    left, right = sessions.searchsorted(d, "left"), sessions.searchsorted(d, "right")
    if timing in ("BMO", "DMH"):
        return left - 1, left
    if timing == "AMC":
        return right - 1, right
    return left - 1, right


def reaction_study(events: pd.DataFrame, prices: pd.DataFrame, bench: pd.DataFrame | pd.Series | None = None,
                   implied: dict | pd.Series | None = None, atr_n: int = 14, vol_n: int = 60) -> pd.DataFrame:
    """Per reported event: gap (pre close -> reaction open), day1 (pre close -> reaction close), drift5/drift20
    (reaction close -> +5/+20 sessions), move in ATRs, z vs trailing daily vol, benchmark-adjusted returns,
    and realized/implied move ratio when `implied` (event date -> implied move as a fraction) is given."""
    from .ta import atr

    px = _daily(prices)
    sess, close, opn = px.index, px["close"].to_numpy(float), px["open"].to_numpy(float) if "open" in px else None
    a = atr(px, atr_n).to_numpy(float) if {"high", "low"} <= set(px) else np.full(len(px), np.nan)
    sd = np.log(px["close"]).diff().rolling(vol_n, min_periods=20).std().to_numpy(float)
    bc = None
    if bench is not None:
        b = bench if isinstance(bench, pd.Series) else bench["close"]
        bc = _daily(b.to_frame("close"))["close"].reindex(sess).ffill().to_numpy(float)
    imp = {pd.Timestamp(k).normalize(): float(v) for k, v in (dict(implied) if implied is not None else {}).items()}
    rows = []
    for _, ev in events.iterrows():
        if not ev.get("reported", True):
            continue
        pre, r0 = event_window(sess, ev["date"], ev["timing"])
        row = {k: ev.get(k) for k in ("date", "timing", "quarter", "eps_est", "eps_act", "surprise_pct", "revenue", "revenue_yoy") if k in ev}
        if pre < 0 or r0 >= len(sess):
            rows.append(row)
            continue
        ret = lambda arr, i, j: arr[j] / arr[i] - 1 if 0 <= i < len(arr) and j < len(arr) else np.nan  # noqa: E731
        row.update({"pre_date": sess[pre], "react_date": sess[r0], "pre_close": close[pre],
                    "gap": opn[r0] / close[pre] - 1 if opn is not None else np.nan, "day1": ret(close, pre, r0),
                    "drift5": ret(close, r0, r0 + 5), "drift20": ret(close, r0, r0 + 20)})
        row["move_atr"] = (close[r0] - close[pre]) / a[pre] if a[pre] > 0 else np.nan
        row["z"] = np.log1p(row["day1"]) / (sd[pre] * math.sqrt(r0 - pre)) if sd[pre] > 0 else np.nan
        if bc is not None:
            row["bench_day1"] = ret(bc, pre, r0)
            row["abn_day1"] = row["day1"] - row["bench_day1"]
            row["abn_drift20"] = row["drift20"] - ret(bc, r0, r0 + 20)
        im = imp.get(pd.Timestamp(ev["date"]).normalize())
        if im:
            row["implied_move"], row["realized_vs_implied"] = im, abs(row["day1"]) / im
        rows.append(row)
    return pd.DataFrame(rows)


def _corr(x, y) -> dict:
    from scipy import stats

    d = pd.DataFrame({"x": x, "y": y}).astype(float).dropna()
    n = len(d)
    if n < 3 or d["x"].std() == 0 or d["y"].std() == 0:
        return {"n": n, "pearson": np.nan, "p_value": np.nan, "spearman": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}
    r, p = stats.pearsonr(d["x"], d["y"])
    rho = stats.spearmanr(d["x"], d["y"])[0]
    lo = hi = np.nan
    if n > 3 and abs(r) < 1:
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    return {"n": n, "pearson": float(r), "p_value": float(p), "spearman": float(rho), "ci95_lo": lo, "ci95_hi": hi}


def summarize_reactions(rx: pd.DataFrame) -> dict:
    """Average/median absolute moves, hit rates, drift after beats vs misses, surprise/reaction correlation."""
    d = rx.dropna(subset=["day1"]) if "day1" in rx else rx.iloc[0:0]
    if d.empty:
        return {"n": 0}
    beat, miss = d["surprise_pct"] > 0, d["surprise_pct"] < 0
    m = lambda s: float(s.mean()) if len(s.dropna()) else np.nan  # noqa: E731
    out = {"n": len(d), "avg_abs_day1": m(d["day1"].abs()), "median_abs_day1": float(d["day1"].abs().median()),
           "avg_abs_gap": m(d["gap"].abs()), "up_rate": float((d["day1"] > 0).mean()), "down_rate": float((d["day1"] < 0).mean()),
           "avg_day1": m(d["day1"]), "avg_drift5": m(d["drift5"]), "avg_drift20": m(d["drift20"]),
           "n_beats": int(beat.sum()), "n_misses": int(miss.sum()),
           "day1_after_beats": m(d.loc[beat, "day1"]), "day1_after_misses": m(d.loc[miss, "day1"]),
           "drift20_after_beats": m(d.loc[beat, "drift20"]), "drift20_after_misses": m(d.loc[miss, "drift20"]),
           "beat_but_sold_off": int((beat & (d["day1"] < 0)).sum()), "miss_but_rallied": int((miss & (d["day1"] > 0)).sum()),
           "avg_abs_z": m(d["z"].abs()), "avg_abs_move_atr": m(d["move_atr"].abs())}
    c = _corr(d["surprise_pct"], d["day1"])
    out.update({"corr_surprise_day1": c["pearson"], "corr_n": c["n"], "corr_p": c["p_value"]})
    if "abn_day1" in d:
        out["avg_abs_abn_day1"] = m(d["abn_day1"].abs())
    if "realized_vs_implied" in d:
        out["avg_realized_vs_implied"] = m(d["realized_vs_implied"])
    return out


# ---- documents: transcripts cache ---------------------------------------------------------
def _tdir(ticker: str) -> Path:
    return TRANSCRIPTS_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", ticker.upper())


def transcript_path(ticker: str, label: str) -> Path:
    return _tdir(ticker) / f"{label.upper()}.txt"


def load_transcript_text(path) -> str:
    """Read a transcript saved by any route (connector, web, provider) as UTF-8 text."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def save_transcript(ticker: str, label: str, text: str) -> Path:
    p = transcript_path(ticker, label)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def list_transcripts(ticker: str) -> dict[str, Path]:
    """Cached transcripts {label: path}, oldest first."""
    d = _tdir(ticker)
    if not d.exists():
        return {}
    return {p.stem.upper(): p for p in sorted(d.glob("*.txt")) if re.fullmatch(r"\d{4}Q[1-4]", p.stem.upper())}


def fetch_transcript_av(ticker: str, label: str, key: str | None = None) -> str:
    """Alpha Vantage EARNINGS_CALL_TRANSCRIPT (provider's fiscal quarter label, e.g. 2024Q1)."""
    from .net import session

    key = key or env("ALPHAVANTAGE_API_KEY")
    if not key:
        raise LookupError("ALPHAVANTAGE_API_KEY not set")
    r = session().get("https://www.alphavantage.co/query", timeout=30,
                      params={"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker.upper(), "quarter": label.upper(), "apikey": key})
    r.raise_for_status()
    js = r.json()
    parts = js.get("transcript") or []
    if not parts:
        raise LookupError(f"alpha vantage: no transcript for {ticker} {label}: {str(js.get('Information') or js.get('Note') or js)[:120]}")
    return "\n\n".join(f"{p.get('speaker', '')}{' (' + p['title'] + ')' if p.get('title') else ''}: {p.get('content', '')}" for p in parts)


def fetch_transcript_fmp(ticker: str, label: str, key: str | None = None) -> str:
    """FMP earning call transcript (stable endpoint, then legacy v3)."""
    from .net import session

    key = key or env("FMP_API_KEY")
    if not key:
        raise LookupError("FMP_API_KEY not set")
    year, q = int(label[:4]), int(label[-1])
    s = session()
    for url, params in (("https://financialmodelingprep.com/stable/earning-call-transcript", {"symbol": ticker.upper()}),
                        (f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker.upper()}", {})):
        try:
            r = s.get(url, params={**params, "year": year, "quarter": q, "apikey": key}, timeout=30)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, list) and js and js[0].get("content"):
                return js[0]["content"]
        except Exception:
            continue
    raise LookupError(f"FMP: no transcript for {ticker} {label}")


def get_transcript(ticker: str, label: str, refresh: bool = False) -> tuple[str, str]:
    """(text, source): cache first, then Alpha Vantage, then FMP (each only when its key is set). Saves to cache."""
    p = transcript_path(ticker, label)
    if p.exists() and not refresh:
        return load_transcript_text(p), f"cache {p}"
    errs = []
    for name, fn in (("Alpha Vantage", fetch_transcript_av), ("FMP", fetch_transcript_fmp)):
        try:
            text = fn(ticker, label)
            save_transcript(ticker, label, text)
            return text, name
        except Exception as e:
            errs.append(f"{name}: {str(e)[:100]}")
    raise LookupError(f"no transcript for {ticker} {label}. " + " | ".join(errs))


# ---- documents: SEC 8-K Item 2.02 press releases ---------------------------------------------
def html_to_text(doc: str) -> str:
    """Strip an HTML filing to readable text (block tags -> newlines, entities unescaped)."""
    doc = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", doc)
    doc = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", doc)
    doc = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table)>", "\n", doc)
    doc = re.sub(r"(?i)</t[dh]>", " \t ", doc)
    doc = _html.unescape(re.sub(r"(?s)<[^>]+>", " ", doc)).replace("\xa0", " ")
    doc = re.sub(r"[ \t]+", " ", doc)
    return re.sub(r"\n\s*\n+", "\n\n", "\n".join(ln.strip() for ln in doc.splitlines())).strip()


def pick_exhibit_991(names: list[str]) -> str | None:
    """Exhibit 99.1 file name from a filing index (ex99-1, ex991, exhibit991, dex991 ...); first 99.x otherwise."""
    docs = [n for n in names if n.lower().endswith((".htm", ".html", ".txt")) and not n.lower().endswith("-index.htm")]
    for pat in (r"ex(hibit)?[-_]?99[-_.]?0?1(?!\d)", r"ex(hibit)?[-_]?99"):
        hit = [n for n in docs if re.search(pat, n.lower())]
        if hit:
            return sorted(hit, key=len)[0]
    return None


def _sec_text(url: str) -> str:
    from .config import SEC_USER_AGENT
    from .net import session

    r = session(SEC_USER_AGENT).get(url, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_earnings_releases(ticker: str, limit: int = 12) -> pd.DataFrame:
    """8-K / 8-K/A filings carrying Item 2.02 (results of operations), newest first, with filing index URL."""
    from .providers import sec

    c = sec.cik(ticker)
    rec = pd.DataFrame(sec._get(f"https://data.sec.gov/submissions/CIK{c}.json")["filings"]["recent"])
    rec = rec[rec["form"].isin(["8-K", "8-K/A"]) & rec["items"].astype(str).str.contains("2.02", regex=False)].head(limit).copy()
    base = f"https://www.sec.gov/Archives/edgar/data/{int(c)}/"
    rec["folder"] = base + rec["accessionNumber"].str.replace("-", "", regex=False) + "/"
    rec["filingDate"] = pd.to_datetime(rec["filingDate"])
    return rec[["filingDate", "form", "items", "accessionNumber", "primaryDocument", "folder"]].reset_index(drop=True)


def fetch_press_release(ticker: str, release: pd.Series, refresh: bool = False) -> tuple[str, str]:
    """(text, url) of exhibit 99.1 for one row of fetch_earnings_releases; cached under transcripts/<T>/press/."""
    from .providers import sec

    p = _tdir(ticker) / "press" / f"{pd.Timestamp(release['filingDate']):%Y-%m-%d}.txt"
    if p.exists() and not refresh:
        return load_transcript_text(p), f"cache {p}"
    idx = sec._get(release["folder"] + "index.json")
    names = [i["name"] for i in idx.get("directory", {}).get("item", [])]
    url = release["folder"] + (pick_exhibit_991(names) or release["primaryDocument"])
    text = html_to_text(_sec_text(url))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text, url


# ---- text analytics: lexicon -------------------------------------------------------------------
# Finance tone word lists in the spirit of Loughran-McDonald (hand-written, representative, not the LM files).
# A trailing * matches any word starting with the stem.
LEXICON: dict[str, list[str]] = {
    "positive": """able abundan* accomplish* achiev* advanc* advantag* attain* attractive* beneficial* benefit* best better
        boost* breakthrough* brilliant* compliment* confident confidence constructive creativ* delight* dependab* desirab*
        efficien* enabl* encourag* enhanc* enjoy* enthusias* excellen* exceed* exceptional* excit* favorab* gain gains gained
        gaining good great greater greatest happy highest ideal impress* improv* incredibl* innovat* insightful leadership
        leading lucrative momentum opportunit* optimis* optimal outpac* outperform* perfect* pleased pleasure popular
        positive positively profitab* progress* prosper* rebound* record resilien* reward* robust satisf* smooth* solid
        stabiliz* strength* strong stronger strongest succeed* success* superior surpass* tailwind* thriv* transformative
        tremendous upside upturn* valuable win winner* winning wins accelerat* healthy outstanding""".split(),
    "negative": """abandon* adverse* against bankrupt* burden* challeng* closure* concern* contraction* curtail* damag* decline
        declined declines declining decreas* default* deficien* deficit* delay* deteriorat* difficult* diminish* disappoint*
        disrupt* downgrad* downsiz* downturn* drag erod* erosion fail* fell fall falling falls fraud* harm* headwind* hurt*
        impair* inability inadequa* ineffective* layoff* lose loses losing loss losses lost miss missed misses negative*
        obstacle* penalt* poor* pressur* problem* recession* restat* restructur* setback* severe* shortage* shortfall* slow
        slowed slower slowing slowdown slump* soft softer softness suffer* terminat* threat* turmoil unfavorab* unprofitab*
        underperform* unsuccessful* weak* worse worsen* worst writedown* write-down* writeoff* adversely volatile
        unfortunately challenging tough""".split(),
    "uncertainty": """almost ambigu* anticipat* appear appears approximat* assum* believe* believed could depend depends
        depending doubt* estimat* eventual* exposure fluctuat* hidden imprecis* indefinite* likelihood may maybe might
        nearly occasional* perhaps possib* predict* preliminar* presum* probab* random* reassess* risk risks risky
        roughly seem* sometime* somewhat speculat* suggest* tentativ* uncertain* unclear unknown* unpredictab* unsure
        unusual* variab* volatil* vague* unproven untested""".split(),
    "litigious": """adjudicat* allegation* alleg* antitrust appeal* appellate arbitrat* attorney* claimant* complainant*
        court* defendant* deposition* indict* infring* injunct* judicial jury lawsuit* lawyer* legal* legislat* litigat*
        plaintiff* prosecut* settlement* statut* subpoena* sue sued suing testimony tribunal* verdict* investigation*
        regulator* enforcement""".split(),
    "constraining": """commit* compel* constrain* covenant* encumb* entail* impos* indebted* limit* mandat* must necessit*
        obligat* preclud* prevent* prohibit* require* requirement* restrain* restrict* confin* stipulat* bound""".split(),
    "strong_modal": """always clearly definitely highly must never undoubtedly unequivocal* undisputed* unparalleled
        absolute* inevitabl* certainly strongly will""".split(),
    "weak_modal": """almost apparently appear appears conceivabl* could depend depends depending may maybe might nearly
        occasionally perhaps possibl* seldom sometimes somewhat suggest* uncertain* hopefully""".split(),
}
NEGATORS = {"no", "not", "none", "neither", "never", "nobody", "nor", "without", "isn't", "wasn't", "aren't", "weren't",
            "don't", "didn't", "doesn't", "can't", "cannot", "won't", "haven't", "hasn't"}
_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


def _compile_lexicon(lex: dict[str, list[str]]):
    return {cat: ({w for w in ws if not w.endswith("*")}, tuple(w[:-1] for w in ws if w.endswith("*"))) for cat, ws in lex.items()}


_LEX = _compile_lexicon(LEXICON)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def _in(word: str, entry) -> bool:
    exact, pre = entry
    return word in exact or word.startswith(pre) if pre else word in exact


def tone(text: str, lexicon: dict | None = None) -> dict:
    """Word counts per category (positive words negated within 3 words count as negative), per-100-word rates,
    net tone (pos-neg)/(pos+neg) in [-1, 1], uncertainty share, weak/strong modal ratio."""
    lex = _compile_lexicon(lexicon) if lexicon else _LEX
    toks = tokenize(text)
    counts, cache = Counter(), {}
    for i, w in enumerate(toks):
        cats = cache.get(w)
        if cats is None:
            cats = cache[w] = [c for c, e in lex.items() if _in(w, e)]
        for c in cats:
            if c == "positive" and any(t in NEGATORS for t in toks[max(0, i - 3):i]):
                counts["negative"] += 1
            else:
                counts[c] += 1
    n = len(toks)
    out = {"words": n, **{c: counts.get(c, 0) for c in lex}}
    for c in lex:
        out[f"{c}_per100"] = 100 * out[c] / n if n else np.nan
    pn = out.get("positive", 0) + out.get("negative", 0)
    out["net_tone"] = (out.get("positive", 0) - out.get("negative", 0)) / pn if pn else 0.0
    out["uncertainty_share"] = out.get("uncertainty", 0) / n if n else np.nan
    out["weak_strong_ratio"] = out.get("weak_modal", 0) / out["strong_modal"] if out.get("strong_modal") else np.nan
    return out


# ---- guidance, forward-looking, Q&A, topics ---------------------------------------------------
_GUIDE_NOUN = re.compile(r"\b(guidance|outlook|forecast|full[- ]year|fiscal (year )?20\d\d|fy ?\d{2,4}|we now (expect|anticipate|see|project)|"
                         r"expectations? for (the )?(full|fiscal|year|quarter|second half|remainder))\b", re.I)
_DIRS = {
    "withdraw": re.compile(r"\b(withdr(aw|aws|awn|awing|ew)|suspend(s|ed|ing)?|pull(s|ed|ing)? (our|its|the) |no longer (provid|giv|offer)|"
                           r"not (providing|issuing|giving) (guidance|an outlook))", re.I),
    "raise": re.compile(r"\b(rais(e|es|ed|ing)|increas(e|es|ed|ing) (our|its|the)|lift(s|ed|ing)?|boost(s|ed|ing)? (our|its|the)|"
                        r"upgrad(e|es|ed|ing)|top end|above (our|the) (prior|previous))\b", re.I),
    "lower": re.compile(r"\b(lower(s|ed|ing)|lower (our|its|the|full|fiscal)|cut(s|ting)? (our|its|the)|reduc(e|es|ed|ing) (our|its|the)|"
                        r"trim(s|med|ming)?|downgrad(e|es|ed|ing)|below (our|the) (prior|previous))\b", re.I),
    "reaffirm": re.compile(r"\b(reaffirm(s|ed|ing)?|reiterat(e|es|ed|ing)|maintain(s|ed|ing)?|confirm(s|ed|ing)?|unchanged|on track|"
                           r"in line with (our|the) (prior|previous))\b", re.I),
}
_SCORE = {"raise": 1, "lower": -1, "withdraw": -1, "reaffirm": 0}
_FLS = re.compile(r"\b(will|expect\w*|anticipat\w*|intend\w*|plan(s|ned|ning)?|believ\w*|outlook|guidance|forecast\w*|project(ed|ing|ions?)|"
                  r"target\w*|aim(s|ing)?|should|going forward|next (quarter|year)|(full|fiscal)[- ]year|second half|remainder of)\b", re.I)
_QA = re.compile(r"question[- ]and[- ]answer|questions? (and|&) answers?|\bq\s*&\s*a\b|(first|next) question|"
                 r"open (it |the call |the line |the lines |the floor )?(up )?(for|to) questions|we will now (begin|take|open)", re.I)
DEFAULT_TOPICS: dict[str, str] = {
    "AI": r"(?-i:\bAI\b)|artificial intelligence|generative|\bgenai\b|\bllms?\b|machine learning|inference|accelerated computing",
    "pricing": r"\bpric(e|es|ed|ing)\b",
    "demand": r"\bdemand\b|\bbacklog\b|\border(s| book)\b|bookings",
    "margins": r"\bmargins?\b",
    "tariffs": r"\btariffs?\b|\bduties\b|trade (war|policy)",
    "China": r"\bchina\b|\bchinese\b",
    "layoffs": r"\blayoffs?\b|headcount reduction|workforce reduction|job cuts|\brestructuring\b",
    "buyback": r"\bbuybacks?\b|\brepurchas\w*",
    "guidance": r"\bguidance\b|\boutlook\b",
    "capex": r"\bcapex\b|capital expenditure",
    "consumer": r"\bconsumer\b",
    "inventory": r"\binventor(y|ies)\b",
    "competition": r"\bcompetit\w*",
    "FX": r"foreign exchange|\bcurrency\b|\bfx\b",
}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])|\n{2,}|\n(?=[A-Z][\w .'-]{0,40}:)", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and len(p.strip()) > 3]


def guidance_sentences(text: str) -> pd.DataFrame:
    """Sentences mentioning guidance/outlook, classified raise / lower / withdraw / reaffirm / mention.
    Withdraw wins; raise vs lower goes to whichever verb appears first."""
    rows = []
    for s in sentences(text):
        if not _GUIDE_NOUN.search(s):
            continue
        hits = {k: m.start() for k, rx in _DIRS.items() if (m := rx.search(s))}
        if "withdraw" in hits:
            d = "withdraw"
        elif "raise" in hits or "lower" in hits:
            d = min((k for k in ("raise", "lower") if k in hits), key=hits.get)
        elif "reaffirm" in hits:
            d = "reaffirm"
        else:
            d = "mention"
        rows.append({"direction": d, "score": _SCORE.get(d, np.nan), "sentence": s})
    return pd.DataFrame(rows, columns=["direction", "score", "sentence"])


def guidance_summary(g: pd.DataFrame) -> dict:
    """Net direction: withdraw if any; else raise/lower by count; reaffirm; mixed; none. score in {-1, 0, 1} or NaN."""
    c = g["direction"].value_counts().to_dict() if len(g) else {}
    up, dn = c.get("raise", 0), c.get("lower", 0)
    if c.get("withdraw"):
        d = "withdraw"
    elif up != dn:
        d = "raise" if up > dn else "lower"
    elif up:
        d = "mixed"
    elif c.get("reaffirm"):
        d = "reaffirm"
    else:
        d = "none"
    return {"guidance": d, "guidance_score": {"raise": 1, "lower": -1, "withdraw": -1, "reaffirm": 0, "mixed": 0}.get(d, np.nan),
            **{f"g_{k}": c.get(k, 0) for k in ("raise", "lower", "reaffirm", "withdraw", "mention")}}


def forward_looking(text: str) -> dict:
    sents = sentences(text)
    words = len(tokenize(text))
    cues = len(_FLS.findall(text))
    return {"fls_per1k": 1000 * cues / words if words else np.nan,
            "fls_sentence_share": float(np.mean([bool(_FLS.search(s)) for s in sents])) if sents else np.nan}


def split_qa(text: str, min_frac: float = 0.1) -> tuple[str, str | None]:
    """(prepared remarks, Q&A) split at the first Q&A marker after min_frac of the text; Q&A None if no marker."""
    start = int(len(text) * min_frac)
    m = _QA.search(text, start)
    if not m:
        return text, None
    cut = text.rfind("\n", 0, m.start())
    cut = m.start() if cut < start else cut
    return text[:cut], text[cut:]


def topic_counts(text: str, topics: dict[str, str] | None = None) -> pd.DataFrame:
    words = max(len(tokenize(text)), 1)
    rows = {k: len(re.findall(p, text, re.I)) for k, p in (topics or DEFAULT_TOPICS).items()}
    return pd.DataFrame({"mentions": rows, "per_1k_words": {k: 1000 * v / words for k, v in rows.items()}})


def analyze_text(text: str, topics: dict[str, str] | None = None) -> dict:
    """Tone, guidance, forward-looking density, prepared vs Q&A tone and topic counts for one document."""
    t = tone(text)
    g = guidance_sentences(text)
    prep, qa = split_qa(text)
    sec = {"prepared": tone(prep)}
    if qa:
        sec["qa"] = tone(qa)
    sections = pd.DataFrame(sec).T[["words", "net_tone", "positive_per100", "negative_per100", "uncertainty_per100"]]
    return {**t, **forward_looking(text), **guidance_summary(g), "has_qa": qa is not None,
            "qa_net_tone": sec["qa"]["net_tone"] if qa else np.nan, "prepared_net_tone": sec["prepared"]["net_tone"],
            "guidance_sentences": g, "sections": sections, "topics": topic_counts(text, topics)}


METRIC_COLS = ["words", "net_tone", "positive_per100", "negative_per100", "uncertainty_share", "litigious_per100",
               "constraining_per100", "weak_strong_ratio", "fls_per1k", "guidance", "guidance_score", "prepared_net_tone", "qa_net_tone"]


def text_metrics(a: dict) -> dict:
    """Flat numeric row of an analyze_text result (for tables and alignment)."""
    return {k: a.get(k, np.nan) for k in METRIC_COLS}


def tone_history(docs: dict[str, str], topics=None) -> pd.DataFrame:
    """Metrics per quarter label (sorted) with quarter-over-quarter changes."""
    df = pd.DataFrame({lbl: text_metrics(analyze_text(t, topics)) for lbl, t in docs.items()}).T.sort_index()
    return add_changes(df)


def add_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("net_tone", "uncertainty_share", "fls_per1k"):
        if c in df:
            df[f"{c}_chg"] = pd.to_numeric(df[c], errors="coerce").diff()
    return df


# ---- alignment ---------------------------------------------------------------------------------
PAIRS = [("surprise_pct", "day1"), ("surprise_pct", "drift20"), ("net_tone", "day1"), ("net_tone_chg", "day1"),
         ("net_tone_chg", "drift20"), ("uncertainty_share_chg", "day1"), ("guidance_score", "day1"), ("guidance_score", "drift20")]


def align_frames(rx: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """Join reaction rows (column 'quarter') with per-quarter text metrics (index = quarter label)."""
    left = rx.set_index("quarter") if "quarter" in rx else rx
    t = text if any(c.endswith("_chg") for c in text.columns) else add_changes(text)
    out = left.join(t, how="left", rsuffix="_txt").sort_index()
    out.index.name = "quarter"
    return out


def alignment_stats(df: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    rows = [{"x": x, "y": y, **_corr(df[x], df[y])} for x, y in pairs if x in df and y in df]
    return pd.DataFrame(rows).set_index(["x", "y"]) if rows else pd.DataFrame()


# ---- desk orchestration (network) -------------------------------------------------------------
def history(ticker: str, quarters: int = 12, bench: str | None = "^GSPC", fiscal_offset: int = 0, implied=None) -> dict:
    """Earnings events + reaction study + summary. Prices via mlab.data.get_prices (cached)."""
    from .data import get_prices

    ev = normalize_earnings(fetch_earnings_dates(ticker, quarters + 6), fiscal_offset)
    try:
        ev = attach_revenue(ev, fetch_quarterly_revenue(ticker))
    except Exception:
        pass
    done = ev[ev["reported"]].tail(quarters)
    start = (done["date"].min() if len(done) else pd.Timestamp.now()) - pd.Timedelta(days=150)
    px = get_prices(ticker, start=start.strftime("%Y-%m-%d"), period=None)
    bx, bsrc = None, None
    if bench:
        try:
            bx = get_prices(bench, start=start.strftime("%Y-%m-%d"), period=None)
            bsrc = bx.attrs.get("source")
        except Exception as e:
            bsrc = f"{bench} unavailable: {str(e)[:60]}"
    rx = reaction_study(done, px, bx, implied)
    upcoming = ev[~ev["reported"] & (ev["date"] >= pd.Timestamp.now().normalize())].head(1)
    return {"events": ev, "reactions": rx, "summary": summarize_reactions(rx), "upcoming": upcoming,
            "sources": ["Yahoo earnings dates (yfinance)", px.attrs.get("source"), bsrc]}


def latest_document(ticker: str, file: str | None = None, fetch: bool = True) -> tuple[str, str, str]:
    """(text, label, source): explicit file, newest cached transcript, provider transcript, else latest SEC press release."""
    if file:
        return load_transcript_text(file), Path(file).stem, f"file {file}"
    cached = list_transcripts(ticker)
    if cached:
        lbl = list(cached)[-1]
        return load_transcript_text(cached[lbl]), lbl, f"cache {cached[lbl]}"
    if not fetch:
        raise LookupError(f"no cached transcript for {ticker}")
    errs = []
    if env("ALPHAVANTAGE_API_KEY") or env("FMP_API_KEY"):
        now = pd.Timestamp.now()
        for lbl in (quarter_label(now), quarter_label(now - pd.DateOffset(months=3))):
            try:
                text, src = get_transcript(ticker, lbl)
                return text, lbl, src
            except Exception as e:
                errs.append(str(e)[:100])
    try:
        rel = fetch_earnings_releases(ticker, 1)
        if len(rel):
            text, url = fetch_press_release(ticker, rel.iloc[0])
            return text, f"8-K {rel.iloc[0]['filingDate']:%Y-%m-%d}", f"SEC 8-K ex99.1 {url}"
    except Exception as e:
        errs.append(f"SEC: {str(e)[:100]}")
    raise LookupError(f"no earnings document for {ticker}: " + " | ".join(errs))


def quarter_documents(ticker: str, events: pd.DataFrame, use_sec: bool = True, fetch_transcripts: bool = False,
                      max_days: int = 4) -> tuple[dict[str, str], dict[str, str]]:
    """Text per quarter label: cached transcript, provider transcript (opt-in), else the 8-K press release filed
    within max_days of the event date. Returns (texts, sources)."""
    texts, srcs = {}, {}
    cached = list_transcripts(ticker)
    rel = None
    for _, ev in events.iterrows():
        lbl = ev["quarter"]
        if lbl in cached:
            texts[lbl], srcs[lbl] = load_transcript_text(cached[lbl]), "transcript (cache)"
            continue
        if fetch_transcripts:
            try:
                texts[lbl], src = get_transcript(ticker, lbl)
                srcs[lbl] = f"transcript ({src})"
                continue
            except Exception:
                pass
        if use_sec:
            try:
                if rel is None:
                    rel = fetch_earnings_releases(ticker, 40)
                gap = (rel["filingDate"] - ev["date"]).dt.days.abs()
                if len(rel) and gap.min() <= max_days:
                    text, url = fetch_press_release(ticker, rel.loc[gap.idxmin()])
                    texts[lbl], srcs[lbl] = text, "8-K ex99.1"
            except Exception:
                rel = rel if rel is not None else pd.DataFrame(columns=["filingDate"])
    return texts, srcs


def align(ticker: str, quarters: int = 12, bench: str | None = "^GSPC", fiscal_offset: int = 0,
          use_sec: bool = True, fetch_transcripts: bool = False) -> dict:
    """Per quarter: surprise, tone metrics, guidance direction, price reaction; plus correlation table."""
    h = history(ticker, quarters, bench, fiscal_offset)
    texts, srcs = quarter_documents(ticker, h["events"][h["events"]["reported"]].tail(quarters), use_sec, fetch_transcripts)
    tm = pd.DataFrame({k: text_metrics(analyze_text(v)) for k, v in texts.items()}).T if texts else pd.DataFrame(columns=METRIC_COLS)
    tm["doc"] = pd.Series(srcs)
    joined = align_frames(h["reactions"], tm)
    return {"aligned": joined, "stats": alignment_stats(joined), "summary": h["summary"], "sources": h["sources"]}


# ---- CLI -----------------------------------------------------------------------------------------
def _pct_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 100
    return df


def _dates(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df


def cmd_earnings(a):
    from .cli import show
    h = history(a.ticker, a.quarters, a.bench, a.fiscal_offset)
    if a.json:
        show(jsonable(h), True)
        return
    implied_note = ""
    if len(h["upcoming"]):
        u = h["upcoming"].iloc[0]
        implied_note = f"next: {u['date']:%Y-%m-%d} {u['timing']} est EPS {u['eps_est']}"
        try:  # options-implied move for the first expiry after the event
            from . import options as op
            spot, raw = op.fetch_chain(a.ticker, 8)
            em = op.expected_move(op.normalize_chain(raw, spot, op.risk_free_rate()))
            after = em[em.index >= u["date"]]
            if len(after):
                implied_note += f" · implied move to {after.index[0]:%Y-%m-%d}: ±{after['move_pct'].iloc[0]:.2%} (ATM straddle)"
        except Exception as e:
            implied_note += f" · implied move unavailable ({type(e).__name__})"
    rx = _pct_cols(h["reactions"], ["gap", "day1", "drift5", "drift20", "bench_day1", "abn_day1", "abn_drift20", "revenue_yoy"])
    cols = [c for c in ["date", "timing", "quarter", "eps_est", "eps_act", "surprise_pct", "revenue_yoy", "gap", "day1", "drift5",
                        "drift20", "move_atr", "z", "abn_day1", "abn_drift20"] if c in rx]
    print(f"## {a.ticker} earnings desk" + (f" · {implied_note}" if implied_note else ""))
    show(_dates(rx[cols]).set_index("date"), title="Earnings reactions (returns in %; surprise in %; z vs trailing 60d daily vol)")
    show(h["summary"], title="Summary")
    print("\n_Sources: " + " · ".join(s for s in h["sources"] if s) + f" · {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def cmd_earnings_text(a):
    from .cli import show
    text, label, src = latest_document(a.ticker, a.file)
    an = analyze_text(text)
    prev = None
    cached = list_transcripts(a.ticker)
    older = [k for k in cached if k < label] if re.fullmatch(r"\d{4}Q[1-4]", label) else []
    if older:
        prev = text_metrics(analyze_text(load_transcript_text(cached[older[-1]])))
    if a.json:
        show(jsonable({"label": label, "source": src, **an, "previous": prev}), True)
        return
    m = text_metrics(an)
    if prev:
        m.update({f"{k}_chg_vs_{older[-1]}": m[k] - prev[k] for k in ("net_tone", "uncertainty_share", "fls_per1k")
                  if isinstance(m[k], (int, float)) and isinstance(prev[k], (int, float))})
    print(f"## {a.ticker} earnings text · {label}")
    show(m, title="Tone and guidance")
    show(pd.DataFrame({k: an[k] for k in LEXICON}, index=["count"]).T, title="Lexicon counts")
    show(an["sections"], title="Prepared remarks vs Q&A" if an["has_qa"] else "Whole document (no Q&A marker found)")
    show(an["topics"][an["topics"]["mentions"] > 0].sort_values("mentions", ascending=False), title="Topics")
    g = an["guidance_sentences"]
    g = g[g["direction"] != "mention"] if (g["direction"] != "mention").any() else g
    if len(g):
        show(g.assign(sentence=g["sentence"].str.slice(0, 220)).head(a.max_sentences).set_index("direction"), title="Guidance sentences")
    print(f"\n_Source: {src} · {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def cmd_earnings_align(a):
    from .cli import show
    r = align(a.ticker, a.quarters, a.bench, a.fiscal_offset, not a.no_sec, a.fetch_transcripts)
    if a.json:
        show(jsonable(r), True)
        return
    df = _pct_cols(r["aligned"], ["day1", "drift20", "abn_day1"])
    cols = [c for c in ["date", "timing", "surprise_pct", "day1", "abn_day1", "drift20", "net_tone", "net_tone_chg", "uncertainty_share",
                        "guidance", "guidance_score", "doc"] if c in df]
    show(_dates(df[cols]), title=f"{a.ticker} per-quarter alignment (returns in %)")
    st = r["stats"]
    if len(st):
        st = st.set_axis([f"{x} vs {y}" for x, y in st.index], axis=0)
    show(st, title="Correlations (n, p-value and 95% CI shown)")
    print("\n_Sources: " + " · ".join(s for s in r["sources"] if s) + f" · {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def register(add):
    q = add("earnings", cmd_earnings, "earnings history vs price reactions: gap, day-1, drift, ATR/z, abnormal return, summary")
    q.add_argument("ticker"); q.add_argument("--quarters", type=int, default=12); q.add_argument("--bench", default="^GSPC")
    q.add_argument("--fiscal-offset", type=int, default=0, help="shift quarter labels for off-calendar fiscal years")

    q = add("earnings-text", cmd_earnings_text, "tone / guidance / topics of the latest transcript or 8-K press release")
    q.add_argument("ticker"); q.add_argument("--file", help="transcript text file (else data/transcripts/<TICKER>/, providers, SEC)")
    q.add_argument("--max-sentences", type=int, default=12)

    q = add("earnings-align", cmd_earnings_align, "per-quarter join of surprise, tone, guidance and price reaction + correlations")
    q.add_argument("ticker"); q.add_argument("--quarters", type=int, default=12); q.add_argument("--bench", default="^GSPC")
    q.add_argument("--fiscal-offset", type=int, default=0); q.add_argument("--no-sec", action="store_true", help="skip 8-K press releases")
    q.add_argument("--fetch-transcripts", action="store_true", help="pull missing transcripts from Alpha Vantage/FMP (uses API quota)")
