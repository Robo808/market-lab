"""Big market moves and the state of the world before each (issue #78).

Builds a small research table of the largest daily moves in a few markets, each joined to what was known at the
prior close (price path, volatility, rates, USD, oil, gold, credit, CFTC positioning, policy-uncertainty and
geopolitical-risk indices, GDELT macro news volume and tone, central-bank calendar), plus a matched sample of ordinary
"control" days with the same features, so clustering and classification can ask what precedes and causes large moves.

Published work this builds on:
- Cutler, Poterba and Summers (1989), "What moves stock prices?", J. Portfolio Management 15(3): the 50 largest S&P
  moves 1941-87 mostly had no identifiable news.
- Baker, Bloom, Davis and Sammon, "What triggers stock market jumps?", NBER w28687 (2021, rev. 2025): next-day
  newspaper coding of the cause of every US daily move above 2.5% since 1900, public at stockmarketjumps.com. Its
  category shares are ingested as ground-truth cause labels (`bbds_*` columns).

Point-in-time rule: every `pre_*` column uses only data dated strictly before the move day (series on other calendars
are joined "as of" the last earlier date; CFTC positions are lagged to their Friday release). `day_*` columns describe
the move day itself (context and labels, never features for prediction).

Storage: $MLAB_DATA_DIR/data/bigmoves/<version>/ (parquet + manifest with sha256); raw source responses are kept in
$MLAB_DATA_DIR/data/bigmoves/raw/ so a rebuild never needs the network. Schema: docs/wiki/Big-Moves.md.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

ROOT = DATA_DIR / "data" / "bigmoves"
RAW = ROOT / "raw"
TEST_START = pd.Timestamp("2020-01-01")  # the frozen datasets' locked test split starts here

BBDS_URL = ("https://docs.google.com/spreadsheets/d/1fkIIishR6LsAtZSv-_cpvyJtuSytemiVYjGUiRgYvNI/export?format=xlsx")
BBDS_SHEET = "jumps by day (all papers)"
BBDS_CATEGORIES = ["commodities", "corporate", "elections", "exrate", "foreign", "govspend", "trade", "macro",
                   "monetary", "noart", "other", "otherpol", "reg", "sovmil", "taxes", "terror", "unknown"]
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

# market -> (dataset, member, CFTC key or None)
MARKETS = {
    "SPX": ("xasset-daily", "^GSPC", "sp500"),
    "NDX": ("xasset-daily", "^NDX", "nasdaq100"),
    "FTSE": ("xasset-daily", "^FTSE", None),
    "BTC": ("crypto-daily", "BTCUSDT", "bitcoin"),
}
CROSS = {"vix": "^VIX", "tnx": "^TNX", "irx": "^IRX", "dxy": "DX-Y.NYB", "oil": "CL=F", "gold": "GC=F", "hyg": "HYG",
         "spx": "^GSPC"}
POLICY = ["GPRD", "GPRD_ACT", "GPRD_THREAT", "EPU_US", "TPU_EPU"]
NEWS_THEMES = ["ECON_STOCKMARKET", "EPU_CATS_MONETARY_POLICY", "ECON_INFLATION", "ECON_OILPRICE", "ECON_DEBT",
               "ECON_WORLDCURRENCIES", "EPU_UNCERTAINTY", "EPU_CATS_NATIONAL_SECURITY", "EPU_CATS_TRADE_POLICY",
               "EPU_CATS_DEBT_CEILING_GOV_SHUTDOWN", "EPU_CATS_REGULATION", "EPU_CATS_TAXES"]
COT_RELEASE_LAG_DAYS = 3  # Tuesday positions, published Friday


# =============================================================================== detection (pure)
def detect(close: pd.Series, span: int = 60, z_min: float = 3.0, min_obs: int = 60, episode_gap: int = 5) -> pd.DataFrame:
    """Daily moves whose log return is at least z_min ex-ante standard deviations.

    sigma_t is the zero-mean EWMA (RiskMetrics style) of squared log returns through t-1, so the scale never sees the
    move it scores. Moves closer than `episode_gap` bars belong to one episode (kept, not dropped: a crash and its
    rebound are two different events with different causes)."""
    close = close.dropna().astype(float)
    r = np.log(close).diff()
    var = r.pow(2).ewm(span=span, adjust=False, min_periods=min_obs).mean().shift(1)
    sigma = np.sqrt(var)
    z = r / sigma
    pos = pd.Series(np.arange(len(close)), index=close.index)
    hit = z.abs() >= z_min
    out = pd.DataFrame({"close": close, "ret": r, "sigma": sigma, "z": z, "bar": pos})[hit].copy()
    out["direction"] = np.sign(out["ret"]).astype(int)
    ep = (out["bar"].diff().fillna(episode_gap + 1) > episode_gap).cumsum()
    out["episode"] = ep.astype(int)
    out["episode_rank"] = out.groupby("episode")["z"].transform(lambda s: s.abs().rank(ascending=False, method="first")).astype(int)
    return out.drop(columns="bar")


def price_features(close: pd.Series) -> pd.DataFrame:
    """Path and volatility features known at the close before each bar (row t uses closes through t-1)."""
    c = close.dropna().astype(float)
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=c.index)
    for n in (1, 5, 20, 60):
        f[f"ret_{n}d"] = lc - lc.shift(n)
    f["rv_20d"] = r.rolling(20).std() * np.sqrt(252)
    f["rv_ratio_5_60"] = r.rolling(5).std() / r.rolling(60).std()
    f["dd_252d"] = c / c.rolling(252, min_periods=60).max() - 1
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    f["up_days_10"] = (r > 0).astype(float).rolling(10).sum()
    return f.shift(1).add_prefix("pre_")


def asof(values: pd.DataFrame | pd.Series, dates: pd.DatetimeIndex, lag_days: int = 0) -> pd.DataFrame:
    """Rows of `values` from the last date strictly before each target date (after shifting availability by lag)."""
    v = values.to_frame() if isinstance(values, pd.Series) else values
    v = v.copy()
    v.index = _day(v.index) + pd.Timedelta(days=lag_days)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    left = pd.DataFrame({"t": _day(dates)}).reset_index(drop=True)
    right = v.reset_index().rename(columns={v.index.name or "index": "t"})
    m = pd.merge_asof(left.sort_values("t"), right, on="t", allow_exact_matches=False)
    m.index = dates
    return m.drop(columns="t")


def _day(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def zscore_prior(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """z of each value against the window before it (the value itself excluded from mean and std)."""
    p = s.shift(1).rolling(window, min_periods=min_periods)
    return (s - p.mean()) / p.std()


# =============================================================================== sources (parse is pure)
def parse_bbds(path: Path | str) -> pd.DataFrame:
    """Baker-Bloom-Davis-Sammon jump days (all papers): date, return, clarity, category shares, primary cause."""
    raw = pd.read_excel(path, sheet_name=BBDS_SHEET)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    df = pd.DataFrame({"date": pd.to_datetime(raw["date"]).dt.normalize(), "bbds_return": raw["return"].astype(float),
                       "bbds_clarity": raw["clarity"].astype(float), "bbds_ease": raw["EaseofCoding"].astype(float),
                       "bbds_confidence": raw["JournalistConfidence"].astype(float),
                       "bbds_coders": raw["numcoders"].astype(int), "bbds_policy": raw["policy"].astype(float)})
    for c in BBDS_CATEGORIES:
        df[f"bbds_{c}"] = raw[c].astype(float)
    shares = df[[f"bbds_{c}" for c in BBDS_CATEGORIES]]
    df["bbds_primary"] = shares.idxmax(axis=1).str.removeprefix("bbds_")
    df["bbds_primary_share"] = shares.max(axis=1)
    return df.set_index("date").sort_index()


_MONTHS = ("January February March April May June July August September October November December").split()


def parse_fomc(html: str, year: int) -> pd.DataFrame:
    """FOMC meetings and calls from a federalreserve.gov fomchistoricalYYYY page: last day of each event."""
    rows = []
    for label in re.findall(r"<h5[^>]*>\s*([^<]+?)\s*-\s*\d{4}\s*</h5>", html):
        months = [m for m in re.findall(r"[A-Z][a-z]+", label) if m in _MONTHS]
        days = [int(d) for d in re.findall(r"\d+", label)]
        if not months or not days:
            continue
        date = pd.Timestamp(year=year, month=_MONTHS.index(months[-1]) + 1, day=days[-1])
        kind = "meeting" if "Meeting" in label else ("unscheduled" if "unscheduled" in label.lower() else "call")
        rows.append({"date": date, "kind": kind, "label": label})
    return pd.DataFrame(rows, columns=["date", "kind", "label"])


def calendar_features(dates: pd.DatetimeIndex, trading_days: pd.DatetimeIndex, fomc: pd.DataFrame) -> pd.DataFrame:
    """Calendar context for each date. Scheduled FOMC dates are public in advance, so distance to the next one is a
    legitimate pre-move feature; unscheduled calls are only flagged on the day itself."""
    d = _day(dates)
    td = _day(trading_days)
    f = pd.DataFrame(index=dates)
    f["day_weekday"] = d.day_of_week
    nxt = pd.Series(td[1:].tolist() + [pd.NaT], index=td)
    f["day_month_end"] = [bool(pd.notna(nxt.get(x)) and nxt.get(x).month != x.month) for x in d]
    f["day_opex"] = [(x.weekday() == 4) and (15 <= x.day <= 21) for x in d]
    f["day_first_friday"] = [(x.weekday() == 4) and (x.day <= 7) for x in d]  # US payrolls proxy, not the BLS calendar
    sched = pd.DatetimeIndex(sorted(fomc.loc[fomc["kind"] == "meeting", "date"])) if len(fomc) else pd.DatetimeIndex([])
    allev = set(pd.DatetimeIndex(fomc["date"])) if len(fomc) else set()
    f["day_fomc"] = [x in allev for x in d]
    f["day_fomc_kind"] = [fomc.loc[fomc["date"] == x, "kind"].iloc[0] if x in allev else "" for x in d]
    tdi = pd.Series(np.arange(len(td)), index=td)

    def tdist(a, b):
        ia, ib = tdi.asof(a), tdi.asof(b)
        return float(ib - ia) if pd.notna(ia) and pd.notna(ib) else np.nan

    to_next, since_last = [], []
    for x in d:
        k = sched.searchsorted(x)  # first scheduled decision on or after x
        to_next.append(tdist(x, sched[k]) if k < len(sched) else np.nan)
        j = sched.searchsorted(x) - 1  # last scheduled decision strictly before x
        since_last.append(tdist(sched[j], x) if j >= 0 else np.nan)
    f["pre_days_to_fomc"] = to_next
    f["pre_days_since_fomc"] = since_last
    return f


def news_daily(paths: list[Path], themes: list[str] = NEWS_THEMES) -> pd.DataFrame:
    """GDELT macro 15-minute files -> one row per UTC day: article count, tone, theme shares."""
    parts = []
    for p in paths:
        try:
            d = pd.read_parquet(p, columns=["avail", "theme", "n", "tone_mean"])
        except Exception:  # noqa: BLE001  (a damaged day file is skipped and shows up as a missing day)
            continue
        d = d[d["theme"].isin(["_ALL", *themes])]
        d["day"] = _day(d["avail"])
        d["tone_x_n"] = d["tone_mean"] * d["n"]
        parts.append(d.groupby(["day", "theme"])[["n", "tone_x_n"]].sum())
    if not parts:
        return pd.DataFrame()
    g = pd.concat(parts).groupby(level=[0, 1]).sum()
    n = g["n"].unstack().fillna(0)
    tone = (g["tone_x_n"] / g["n"]).unstack()
    out = pd.DataFrame(index=n.index)
    out["news_n"] = n["_ALL"]
    out["news_tone"] = tone["_ALL"]
    if "ECON_STOCKMARKET" in tone:
        out["news_tone_stockmarket"] = tone["ECON_STOCKMARKET"]
    for t in themes:
        out[f"news_share_{t.lower()}"] = (n[t] if t in n else 0.0) / n["_ALL"]
    out.index.name = "day"
    return out


def news_features(nd: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Prior UTC day's news (pre_) z-scored against its own 60 days, plus the move day's own news (day_)."""
    if nd.empty:
        return pd.DataFrame(index=dates)
    z = nd.apply(lambda s: zscore_prior(s, 60, 20))
    full = pd.DataFrame(index=pd.date_range(nd.index.min(), nd.index.max(), freq="D"))
    raw, zz = full.join(nd), full.join(z)
    d = _day(dates)
    prev = d - pd.Timedelta(days=1)
    pre = raw.reindex(prev).add_prefix("pre_").set_axis(dates)
    pre_z = zz.reindex(prev).add_prefix("pre_z_").set_axis(dates)
    day = zz.reindex(d).add_prefix("day_z_").set_axis(dates)
    return pd.concat([pre[["pre_news_n", "pre_news_tone"]], pre_z, day], axis=1)


def cross_features(cross: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rates, vol, USD, commodities, credit, as of the last close strictly before each date."""
    der = {}
    if "vix" in cross:
        v = cross["vix"]
        der["pre_vix"] = v
        der["pre_vix_chg5"] = v - v.shift(5)
        der["pre_vix_z252"] = zscore_prior(v)
    if "tnx" in cross:
        der["pre_tnx"] = cross["tnx"]
        der["pre_tnx_chg20"] = cross["tnx"] - cross["tnx"].shift(20)
    if "tnx" in cross and "irx" in cross:
        a = pd.concat([cross["tnx"], cross["irx"]], axis=1).dropna()
        der["pre_curve_10y_3m"] = a.iloc[:, 0] - a.iloc[:, 1]
    for k in ("dxy", "oil", "gold", "hyg", "spx"):
        if k in cross:
            s = np.log(cross[k].where(cross[k] > 0))
            der[f"pre_{k}_ret20"] = s - s.shift(20)
            if k == "spx":
                der["pre_spx_ret1"] = s - s.shift(1)
    out = [asof(s.rename(name), dates) for name, s in der.items()]
    return pd.concat(out, axis=1) if out else pd.DataFrame(index=dates)


def policy_features(policy: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily GPR / EPU / TPU: 7-day mean and z against the prior year, as of the day before the move.
    Caveat: these indices are compiled from that day's newspapers and published later; fine for description,
    a forecasting use needs their real release lag."""
    der = {}
    for k, s in policy.items():
        m7 = s.rolling(7, min_periods=3).mean()
        der[f"pre_{k.lower()}_7d"] = m7
        der[f"pre_{k.lower()}_z"] = zscore_prior(m7, 365, 90)
    out = [asof(s.rename(name), dates) for name, s in der.items()]
    return pd.concat(out, axis=1) if out else pd.DataFrame(index=dates)


def cot_features(cotdf: pd.DataFrame | None, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Leveraged funds and asset managers net as % of open interest, plus 3-year COT index of leveraged net,
    available from the Friday release after each Tuesday report date."""
    cols = ["pre_cot_lev_net_oi", "pre_cot_am_net_oi", "pre_cot_lev_index"]
    if cotdf is None or cotdf.empty or "lev_funds_net" not in cotdf:
        return pd.DataFrame(np.nan, index=dates, columns=cols)
    from .cot import cot_index
    f = pd.DataFrame(index=cotdf.index)
    f[cols[0]] = cotdf["lev_funds_net"] / cotdf["open_interest"]
    f[cols[1]] = cotdf["asset_mgr_net"] / cotdf["open_interest"] if "asset_mgr_net" in cotdf else np.nan
    f[cols[2]] = cot_index(cotdf["lev_funds_net"])
    return asof(f, dates, lag_days=COT_RELEASE_LAG_DAYS)


def controls(close: pd.Series, moves: pd.DatetimeIndex, k: int, seed: int, buffer: int = 5, min_obs: int = 252,
             start=None, end=None) -> pd.DatetimeIndex:
    """k ordinary days per move, drawn at random from bars in [start, end] at least `buffer` bars from any move."""
    idx = close.dropna().index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    banned = np.zeros(len(idx), bool)
    banned[:min_obs] = True
    if start is not None:
        banned |= idx < pd.Timestamp(start)
    if end is not None:
        banned |= idx > pd.Timestamp(end)
    for p in pos.reindex(moves).dropna().astype(int):
        banned[max(0, p - buffer): p + buffer + 1] = True
    pool = idx[~banned]
    n = min(len(pool), k * len(moves))
    rng = np.random.default_rng(seed)
    return pd.DatetimeIndex(sorted(rng.choice(pool, size=n, replace=False))) if n else pd.DatetimeIndex([])


# =============================================================================== fetch raw (network, saved verbatim)
def fetch_raw(years: range, cot_keys: list[str], cot_since: str = "2006-01-01", log=print, session=None) -> dict:
    """Save each source's real response under raw/ (once). Returns {item: status}."""
    from .net import session as mk
    s = session or mk()
    RAW.mkdir(parents=True, exist_ok=True)
    st = {}
    p = RAW / "bbds_jumps.xlsx"
    if not p.exists():
        r = s.get(BBDS_URL, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    st["bbds"] = f"ok ({p.stat().st_size} bytes)"
    (RAW / "fomc").mkdir(exist_ok=True)
    got = 0
    for y in years:
        q = RAW / "fomc" / f"fomchistorical{y}.htm"
        if q.exists() and q.stat().st_size > 1000:
            got += 1
            continue
        r = s.get(FOMC_URL.format(year=y), timeout=30)
        if r.status_code == 200:
            q.write_text(r.text)
            got += 1
        else:
            log(f"  fomc {y}: HTTP {r.status_code}")
    st["fomc"] = f"{got}/{len(years)} years"
    from . import cot
    for key in cot_keys:
        q = RAW / f"cot_tff_{key}.json"
        if not q.exists():
            rows = cot.fetch_rows(key, "tff", pd.Timestamp(cot_since))
            q.write_text(json.dumps(rows))
        st[f"cot:{key}"] = f"ok ({len(json.loads(q.read_text()))} rows)"
    return st


def load_fomc(years: range) -> pd.DataFrame:
    parts = [parse_fomc((RAW / "fomc" / f"fomchistorical{y}.htm").read_text(), y)
             for y in years if (RAW / "fomc" / f"fomchistorical{y}.htm").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "kind", "label"])


def load_cot(key: str | None) -> pd.DataFrame | None:
    q = RAW / f"cot_tff_{key}.json" if key else None
    if not q or not q.exists():
        return None
    from . import cot
    return cot.parse(json.loads(q.read_text()), "tff")


# =============================================================================== build
def _reuse_news(a: pd.Timestamp, b: pd.Timestamp, last_file: pd.Timestamp | None) -> pd.DataFrame | None:
    """news_daily from an earlier build that covers [a - 90 days, b] (reading the GDELT day files takes ~10 minutes)."""
    for v in sorted(ROOT.glob("v*"), reverse=True):
        p = v / "news_daily.parquet"
        if p.exists():
            nd = pd.read_parquet(p).set_index("day")
            if len(nd) and nd.index.min() <= a and last_file is not None and nd.index.max() >= min(b, last_file):
                return nd
    return None


def _close(df: pd.DataFrame) -> pd.Series:
    s = df["close"].astype(float)
    s.index = _day(s.index)
    return s[~s.index.duplicated(keep="last")]


def build(markets: list[str] | None = None, start: str = "1990-01-01", end: str = "2019-12-31", z_min: float = 3.0,
          span: int = 60, k_controls: int = 5, seed: int = 7, news_since: str = "2015-02-19", allow_test: bool = False,
          version: str | None = None, fetch: bool = True, log=print) -> Path:
    """Build moves.parquet + controls.parquet (+ bbds, fomc, news_daily side tables) into ROOT/<version>/."""
    from . import datasets as D
    markets = markets or list(MARKETS)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if t1 >= TEST_START and not allow_test:
        raise PermissionError("end reaches the locked 2020+ test split; pass allow_test=True (CLI --allow-test)")
    split = None if allow_test else "train+validation"
    years = range(max(t0.year - 1, 1990), t1.year + 2)
    status = {}
    if fetch:
        status.update(fetch_raw(years, sorted({MARKETS[m][2] for m in markets if MARKETS[m][2]}), log=log))
    bbds = parse_bbds(RAW / "bbds_jumps.xlsx")
    fomc = load_fomc(years)

    note = "bigmoves build (#78)"
    xa = D.load("xasset-daily", symbols=list(dict.fromkeys(CROSS.values())), split=split, allow_test=allow_test, note=note)
    cross = {k: _close(xa[v]) for k, v in CROSS.items() if v in xa and len(xa[v])}
    pol_ref = D.versions("policy-daily")[-1]
    pol = D.load(f"policy-daily@{pol_ref}", symbols=POLICY, split=split, allow_test=allow_test, note=note)
    policy = {k: _close(v.rename(columns={"value": "close"})) for k, v in pol.items()}

    news_dir = DATA_DIR / "data" / "news" / "gdelt" / "macro"
    nds = pd.Timestamp(news_since)
    paths = [p for p in sorted(news_dir.glob("*.parquet")) if max(t0, nds) - pd.Timedelta(days=90) <= pd.Timestamp(p.stem) <= t1]
    nd = _reuse_news(max(t0, nds), t1, pd.Timestamp(paths[-1].stem) if paths else None)
    if nd is None:
        log(f"  news: {len(paths)} GDELT macro day files")
        nd = news_daily(paths)
    else:
        log(f"  news: reused news_daily from an earlier build ({len(nd)} days)")
    status["news"] = f"{len(nd)} days" if len(nd) else "no files"

    frames, ctrl_frames, lab_frames = [], [], []
    for m in markets:
        ds, member, cot_key = MARKETS[m]
        px = D.load(ds, symbols=[member], split=split, allow_test=allow_test, note=note)[member]
        close = _close(px)
        close = close[close.index <= t1]
        mv = detect(close, span=span, z_min=z_min)
        mv = mv[(mv.index >= t0) & (mv.index <= t1)]
        cd = controls(close, mv.index, k_controls, seed + len(frames), start=t0, end=t1)
        pf = price_features(close)
        cot_df = load_cot(cot_key)
        status[f"market:{m}"] = f"{len(mv)} moves, {len(cd)} controls"
        if cot_key:
            status[f"cot:{cot_key}"] = f"ok ({len(cot_df)} weeks from the saved response)" if cot_df is not None else "missing"
        kinds = [("move", mv.index, mv), ("control", cd, None)]
        if m == "SPX":  # every BBDS-coded jump day in the window, whatever its z: the labelled sample for cause models
            bd = bbds.index[(bbds.index >= t0) & (bbds.index <= t1)].intersection(close.index)
            kinds.append(("bbds", bd, None))
            status["labelled:SPX"] = f"{len(bd)} BBDS jump days"
        for kind, dates, base in kinds:
            if not len(dates):
                continue
            if base is None:
                r = np.log(close).diff()
                var = r.pow(2).ewm(span=span, adjust=False, min_periods=60).mean().shift(1)
                base = pd.DataFrame({"close": close, "ret": r, "sigma": np.sqrt(var)}).reindex(dates)
                base["z"] = base["ret"] / base["sigma"]
                base["direction"] = np.sign(base["ret"]).astype(int)
                base["episode"], base["episode_rank"] = -1, 0
            parts = [base.rename(columns={"close": "day_close", "ret": "day_ret", "sigma": "pre_sigma", "z": "day_z",
                                          "direction": "day_direction"}),
                     pf.reindex(dates), cross_features({k: v for k, v in cross.items() if not (m == "SPX" and k == "spx")}, dates),
                     policy_features(policy, dates), cot_features(cot_df, dates),
                     news_features(nd[nd.index >= nds] if len(nd) else nd, dates),
                     calendar_features(dates, close.index, fomc)]
            t = pd.concat(parts, axis=1)
            t.insert(0, "market", m)
            t.insert(1, "kind", kind)
            t.index.name = "date"
            if m in ("SPX", "NDX"):
                t = t.join(bbds, how="left")
                t["bbds_match"] = t["bbds_return"].notna()
            {"move": frames, "control": ctrl_frames, "bbds": lab_frames}[kind].append(t)

    moves = pd.concat(frames).reset_index().sort_values(["date", "market"]).reset_index(drop=True)
    ctrls = pd.concat(ctrl_frames).reset_index().sort_values(["date", "market"]).reset_index(drop=True)
    ctrls = ctrls[[c for c in ctrls.columns if not c.startswith("bbds_")]]
    labelled = pd.concat(lab_frames).reset_index().sort_values("date").reset_index(drop=True) if lab_frames else pd.DataFrame()
    version = version or f"v{pd.Timestamp.now(tz='UTC'):%Y%m%d}"
    out = ROOT / version
    out.mkdir(parents=True, exist_ok=True)
    tables = {"moves": moves, "controls": ctrls, "labelled": labelled, "bbds": bbds.reset_index(), "fomc": fomc,
              "news_daily": nd.reset_index() if len(nd) else pd.DataFrame()}
    files = {}
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        back = pd.read_parquet(p)  # L-015: read every shared-folder write back
        if len(back) != len(df):
            raise OSError(f"{p}: wrote {len(df)} rows, read back {len(back)}")
        files[name] = {"rows": len(df), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    man = {"name": "bigmoves", "version": version, "built_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "params": {"markets": markets, "start": start, "end": end, "z_min": z_min, "ewma_span": span,
                      "k_controls": k_controls, "seed": seed, "news_since": news_since, "allow_test": allow_test},
           "sources": {"prices": "datasets xasset-daily (Yahoo), crypto-daily (Binance BTCUSDT)",
                       "policy": f"datasets policy-daily@{pol_ref} (GPR: Caldara-Iacoviello; EPU/TPU: Baker-Bloom-Davis)",
                       "cot": "CFTC TFF via publicreporting.cftc.gov (raw/cot_tff_*.json)",
                       "news": "GDELT 2.0 GKG macro themes store (data/news/gdelt/macro)",
                       "fomc": "federalreserve.gov fomchistoricalYYYY pages (raw/fomc)",
                       "labels": "Baker, Bloom, Davis, Sammon, What triggers stock market jumps? (stockmarketjumps.com)"},
           "licences": {"yahoo": "personal, non-commercial; no redistribution", "binance": "CC BY-NC-SA",
                        "bbds": "public academic dataset; cite the paper", "gdelt": "open, cite GDELT",
                        "cftc": "US government, public domain", "federalreserve.gov": "US government, public domain"},
           "files": files, "status": status, "schema": "docs/wiki/Big-Moves.md"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    json.loads((out / "manifest.json").read_text())
    return out


def latest() -> Path:
    vs = sorted(p for p in ROOT.glob("v*") if (p / "manifest.json").exists())
    if not vs:
        raise FileNotFoundError(f"no bigmoves build under {ROOT}; run `mlab bigmoves build`")
    return vs[-1]


def load(version: str | None = None) -> dict[str, pd.DataFrame]:
    root = ROOT / version if version else latest()
    return {p.stem: pd.read_parquet(p) for p in root.glob("*.parquet")}


# =============================================================================== CLI
def cmd_bigmoves(a):
    from .cli import show
    if a.action == "build":
        out = build(markets=a.markets, start=a.start, end=a.end, z_min=a.z, k_controls=a.controls,
                    allow_test=a.allow_test, version=a.version, fetch=not a.no_fetch)
        man = json.loads((out / "manifest.json").read_text())
        show(pd.DataFrame([{"item": k, "status": v} for k, v in man["status"].items()]).set_index("item"),
             title=f"bigmoves {man['version']} -> {out}")
        show(pd.DataFrame(man["files"]).T, title="files")
    elif a.action == "show":
        t = load(a.version)["moves"]
        cols = ["date", "market", "day_ret", "day_z", "episode", "pre_ret_20d", "pre_vix", "bbds_primary"]
        t = t.reindex(t["day_z"].abs().sort_values(ascending=False).index).head(a.top)
        show(t[[c for c in cols if c in t]].set_index("date"), title=f"largest moves ({latest().name})")
    elif a.action == "study":
        from .bigmoves_study import run, report
        res = run(load(a.version), n_perm=a.perm, seed=a.seed)
        print(report(res))


def register(add):
    q = add("bigmoves", cmd_bigmoves, "big market moves + world state before each: build | show | study (issue #78)")
    q.add_argument("action", choices=["build", "show", "study"])
    q.add_argument("--markets", nargs="*", choices=list(MARKETS))
    q.add_argument("--start", default="1990-01-01"); q.add_argument("--end", default="2019-12-31")
    q.add_argument("--z", type=float, default=3.0); q.add_argument("--controls", type=int, default=5)
    q.add_argument("--allow-test", action="store_true"); q.add_argument("--version")
    q.add_argument("--no-fetch", action="store_true", help="use only saved raw responses")
    q.add_argument("--top", type=int, default=25); q.add_argument("--perm", type=int, default=500)
    q.add_argument("--seed", type=int, default=11)
