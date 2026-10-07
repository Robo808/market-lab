"""CFTC Commitments of Traders: weekly positioning for the macro desk.

Data: CFTC public reporting Socrata API (keyless; optional CFTC_APP_TOKEN raises rate limits).
  legacy  6dca-aqww  Legacy futures-only: non-commercial / commercial / non-reportable
  disagg  72hh-3qpy  Disaggregated futures-only: managed money / producer-merchant / swap dealers / other
  tff     gpe5-46if  Traders in Financial Futures futures-only: leveraged funds / asset managers / dealers / other

Analytics per trader class: net, weekly change, net % of open interest, COT index (0-100 position of net
within its trailing 156-week range) and z-score, extreme flags (>90 / <10), and positioning-vs-price
divergence using Yahoo futures prices. Weekly rows are cached to parquet so history accumulates.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import env

HOSTS: dict[str, str] = {
    "publicreporting.cftc.gov": "CFTC Commitments of Traders (Socrata API)",
    "query1.finance.yahoo.com": "Yahoo futures prices for divergence (via data.get_prices)",
    "stooq.com": "price fallback (via data.get_prices)",
}

DATASETS = {"legacy": "6dca-aqww", "disagg": "72hh-3qpy", "tff": "gpe5-46if"}
BASE = "https://publicreporting.cftc.gov/resource/{}.json"
DATE = "report_date_as_yyyy_mm_dd"

# Trader classes: (long candidates, short candidates). First field present wins; CFTC field names have quirks
# (e.g. swap__positions_short_all with a double underscore), so a fuzzy fallback also applies.
CLASSES: dict[str, dict[str, tuple[list[str], list[str]]]] = {
    "legacy": {
        "noncomm": (["noncomm_positions_long_all"], ["noncomm_positions_short_all"]),
        "comm": (["comm_positions_long_all"], ["comm_positions_short_all"]),
        "nonrept": (["nonrept_positions_long_all"], ["nonrept_positions_short_all"]),
    },
    "disagg": {
        "managed_money": (["m_money_positions_long_all", "m_money_positions_long"], ["m_money_positions_short_all", "m_money_positions_short"]),
        "producer": (["prod_merc_positions_long", "prod_merc_positions_long_all"], ["prod_merc_positions_short", "prod_merc_positions_short_all"]),
        "swap": (["swap_positions_long_all", "swap_positions_long"], ["swap__positions_short_all", "swap_positions_short_all"]),
        "other": (["other_rept_positions_long", "other_rept_positions_long_all"], ["other_rept_positions_short", "other_rept_positions_short_all"]),
        "nonrept": (["nonrept_positions_long_all"], ["nonrept_positions_short_all"]),
    },
    "tff": {
        "lev_funds": (["lev_money_positions_long", "lev_money_positions_long_all"], ["lev_money_positions_short", "lev_money_positions_short_all"]),
        "asset_mgr": (["asset_mgr_positions_long", "asset_mgr_positions_long_all"], ["asset_mgr_positions_short", "asset_mgr_positions_short_all"]),
        "dealer": (["dealer_positions_long_all", "dealer_positions_long"], ["dealer_positions_short_all", "dealer_positions_short"]),
        "other": (["other_rept_positions_long", "other_rept_positions_long_all"], ["other_rept_positions_short", "other_rept_positions_short_all"]),
        "nonrept": (["nonrept_positions_long_all"], ["nonrept_positions_short_all"]),
    },
}
SPEC = {"legacy": "noncomm", "disagg": "managed_money", "tff": "lev_funds"}  # the "speculator" class per report
FUZZY = {"noncomm": "noncomm", "comm": "^comm", "nonrept": "nonrept", "managed_money": "m_money", "producer": "prod_merc",
         "swap": "swap", "other": "other_rept", "lev_funds": "lev_money", "asset_mgr": "asset_mgr", "dealer": "dealer"}

# friendly key -> CFTC contract market code, market-name pattern (fallback), Yahoo symbol, kind (fin -> TFF, com -> disagg)
MARKETS: dict[str, dict] = {
    "sp500":     {"code": "13874A", "pattern": "E-MINI S&P 500 -", "yahoo": "ES=F", "kind": "fin", "label": "S&P 500 E-mini"},
    "nasdaq100": {"code": "209742", "pattern": "NASDAQ MINI", "yahoo": "NQ=F", "kind": "fin", "label": "Nasdaq-100 E-mini"},
    "dow":       {"code": "124603", "pattern": "DJIA x $5", "yahoo": "YM=F", "kind": "fin", "label": "Dow E-mini ($5)"},
    "russell":   {"code": "239742", "pattern": "RUSSELL E-MINI", "yahoo": "RTY=F", "kind": "fin", "label": "Russell 2000 E-mini"},
    "vix":       {"code": "1170E1", "pattern": "VIX FUTURES", "yahoo": "^VIX", "kind": "fin", "label": "VIX futures"},
    "ust2y":     {"code": "042601", "pattern": "UST 2Y NOTE", "yahoo": "ZT=F", "kind": "fin", "label": "2-year T-note"},
    "ust5y":     {"code": "044601", "pattern": "UST 5Y NOTE", "yahoo": "ZF=F", "kind": "fin", "label": "5-year T-note"},
    "ust10y":    {"code": "043602", "pattern": "UST 10Y NOTE", "yahoo": "ZN=F", "kind": "fin", "label": "10-year T-note"},
    "ust30y":    {"code": "020601", "pattern": "UST BOND", "yahoo": "ZB=F", "kind": "fin", "label": "30-year T-bond"},
    "sofr":      {"code": "134741", "pattern": "SOFR-3M", "yahoo": None, "kind": "fin", "label": "3M SOFR"},
    "eurodollar": {"code": "132741", "pattern": "EURODOLLARS-3M", "yahoo": None, "kind": "fin", "label": "Eurodollar (ended 2023)"},
    "eur":       {"code": "099741", "pattern": "EURO FX -", "yahoo": "6E=F", "kind": "fin", "label": "Euro FX"},
    "gbp":       {"code": "096742", "pattern": "BRITISH POUND", "yahoo": "6B=F", "kind": "fin", "label": "British pound"},
    "jpy":       {"code": "097741", "pattern": "JAPANESE YEN", "yahoo": "6J=F", "kind": "fin", "label": "Japanese yen"},
    "chf":       {"code": "092741", "pattern": "SWISS FRANC", "yahoo": "6S=F", "kind": "fin", "label": "Swiss franc"},
    "cad":       {"code": "090741", "pattern": "CANADIAN DOLLAR", "yahoo": "6C=F", "kind": "fin", "label": "Canadian dollar"},
    "aud":       {"code": "232741", "pattern": "AUSTRALIAN DOLLAR", "yahoo": "6A=F", "kind": "fin", "label": "Australian dollar"},
    "nzd":       {"code": "112741", "pattern": "NZ DOLLAR", "yahoo": "6N=F", "kind": "fin", "label": "New Zealand dollar"},
    "mxn":       {"code": "095741", "pattern": "MEXICAN PESO", "yahoo": "6M=F", "kind": "fin", "label": "Mexican peso"},
    "usd":       {"code": "098662", "pattern": "USD INDEX", "yahoo": "DX-Y.NYB", "kind": "fin", "label": "US dollar index"},
    "bitcoin":   {"code": "133741", "pattern": "BITCOIN - CHICAGO", "yahoo": "BTC=F", "kind": "fin", "label": "Bitcoin (CME)"},
    "ether":     {"code": "146021", "pattern": "ETHER CASH SETTLED", "yahoo": "ETH=F", "kind": "fin", "label": "Ether (CME)"},
    "gold":      {"code": "088691", "pattern": "GOLD - COMMODITY EXCHANGE", "yahoo": "GC=F", "kind": "com", "label": "Gold (COMEX)"},
    "silver":    {"code": "084691", "pattern": "SILVER - COMMODITY EXCHANGE", "yahoo": "SI=F", "kind": "com", "label": "Silver (COMEX)"},
    "copper":    {"code": "085692", "pattern": "COPPER- #1", "yahoo": "HG=F", "kind": "com", "label": "Copper (COMEX)"},
    "platinum":  {"code": "076651", "pattern": "PLATINUM - NEW YORK", "yahoo": "PL=F", "kind": "com", "label": "Platinum (NYMEX)"},
    "wti":       {"code": "067651", "pattern": "CRUDE OIL, LIGHT SWEET - NEW YORK", "yahoo": "CL=F", "kind": "com", "label": "WTI crude (NYMEX)"},
    "brent":     {"code": "06765T", "pattern": "BRENT LAST DAY", "yahoo": "BZ=F", "kind": "com", "label": "Brent last day (NYMEX)"},
    "natgas":    {"code": "023651", "pattern": "NAT GAS NYME", "yahoo": "NG=F", "kind": "com", "label": "Henry Hub natural gas (NYMEX)"},
    "gasoline":  {"code": "111659", "pattern": "GASOLINE RBOB", "yahoo": "RB=F", "kind": "com", "label": "RBOB gasoline"},
    "heatingoil": {"code": "022651", "pattern": "NY HARBOR ULSD", "yahoo": "HO=F", "kind": "com", "label": "NY Harbor ULSD / heating oil"},
    "corn":      {"code": "002602", "pattern": "CORN - CHICAGO", "yahoo": "ZC=F", "kind": "com", "label": "Corn (CBOT)"},
    "wheat":     {"code": "001602", "pattern": "WHEAT-SRW", "yahoo": "ZW=F", "kind": "com", "label": "Wheat SRW (CBOT)"},
    "soybeans":  {"code": "005602", "pattern": "SOYBEANS - CHICAGO", "yahoo": "ZS=F", "kind": "com", "label": "Soybeans (CBOT)"},
    "sugar":     {"code": "080732", "pattern": "SUGAR NO. 11", "yahoo": "SB=F", "kind": "com", "label": "Sugar No. 11 (ICE)"},
    "coffee":    {"code": "083731", "pattern": "COFFEE C", "yahoo": "KC=F", "kind": "com", "label": "Coffee C (ICE)"},
    "cocoa":     {"code": "073732", "pattern": "COCOA - ICE", "yahoo": "CC=F", "kind": "com", "label": "Cocoa (ICE)"},
    "cotton":    {"code": "033661", "pattern": "COTTON NO. 2", "yahoo": "CT=F", "kind": "com", "label": "Cotton No. 2 (ICE)"},
    "cattle":    {"code": "057642", "pattern": "LIVE CATTLE", "yahoo": "LE=F", "kind": "com", "label": "Live cattle (CME)"},
}
ALIASES = {
    "es": "sp500", "spx": "sp500", "s&p": "sp500", "s&p500": "sp500", "sp": "sp500", "us500": "sp500",
    "nq": "nasdaq100", "ndx": "nasdaq100", "nasdaq": "nasdaq100", "ym": "dow", "djia": "dow", "rty": "russell", "russell2000": "russell",
    "vx": "vix", "2y": "ust2y", "zt": "ust2y", "5y": "ust5y", "zf": "ust5y", "10y": "ust10y", "zn": "ust10y", "30y": "ust30y",
    "zb": "ust30y", "bond": "ust30y", "sr3": "sofr", "ed": "eurodollar", "6e": "eur", "eurusd": "eur", "euro": "eur",
    "6b": "gbp", "gbpusd": "gbp", "cable": "gbp", "pound": "gbp", "6j": "jpy", "usdjpy": "jpy", "yen": "jpy", "6s": "chf",
    "6c": "cad", "6a": "aud", "6n": "nzd", "6m": "mxn", "dxy": "usd", "dx": "usd", "btc": "bitcoin", "eth": "ether",
    "gc": "gold", "xau": "gold", "si": "silver", "xag": "silver", "hg": "copper", "pl": "platinum", "cl": "wti", "crude": "wti",
    "oil": "wti", "bz": "brent", "ng": "natgas", "gas": "natgas", "rb": "gasoline", "rbob": "gasoline", "ho": "heatingoil",
    "ulsd": "heatingoil", "zc": "corn", "zw": "wheat", "zs": "soybeans", "soy": "soybeans", "sb": "sugar", "kc": "coffee",
    "cc": "cocoa", "ct": "cotton", "le": "cattle", "livecattle": "cattle",
}


def resolve(market: str) -> str:
    k = market.lower().strip()
    k = re.sub(r"[\s_\-]+", "", k[:-2] if k.endswith("=f") else k)
    k = ALIASES.get(k, k)
    if k not in MARKETS:
        raise KeyError(f"unknown COT market '{market}'. Known: {', '.join(MARKETS)}")
    return k


def default_report(key: str) -> str:
    return "tff" if MARKETS[key]["kind"] == "fin" else "disagg"


# =============================================================================== fetch
def _socrata(dataset: str, where: str, limit: int = 5000, order: str = f"{DATE} DESC") -> list[dict]:
    """One Socrata query. Kept tiny so tests can monkeypatch it."""
    from .net import session
    headers = {"X-App-Token": env("CFTC_APP_TOKEN")} if env("CFTC_APP_TOKEN") else {}
    r = session().get(BASE.format(dataset), params={"$where": where, "$limit": limit, "$order": order}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_rows(key: str, report: str, since: pd.Timestamp) -> list[dict]:
    """Rows for one market by contract code, falling back to a market-name LIKE match."""
    m, ds = MARKETS[key], DATASETS[report]
    when = f"{DATE} >= '{since:%Y-%m-%d}T00:00:00'"
    rows = _socrata(ds, f"cftc_contract_market_code = '{m['code']}' AND {when}")
    if not rows:
        pat = m["pattern"].upper().replace("'", "''")
        rows = _socrata(ds, f"upper(market_and_exchange_names) like '%{pat}%' AND {when}")
    return rows


# =============================================================================== parse + analytics (pure)
def _col(df: pd.DataFrame, names: list[str], cls: str, side: str) -> pd.Series | None:
    for n in names:
        if n in df:
            return pd.to_numeric(df[n], errors="coerce")
    rx = re.compile(FUZZY[cls] + r"[a-z_]*?positions?_+" + side + r"(_all)?$")  # excludes *_old / *_other crop-year cuts
    hits = [c for c in df.columns if rx.search(c)]
    return pd.to_numeric(df[hits[0]], errors="coerce") if hits else None


def parse(rows: list[dict], report: str) -> pd.DataFrame:
    """Socrata rows -> weekly frame indexed by report date: open_interest, <class>_long/_short/_net."""
    if not rows:
        raise LookupError("no COT rows")
    raw = pd.DataFrame(rows)
    raw["date"] = pd.to_datetime(raw[DATE].str[:10])
    raw["open_interest"] = pd.to_numeric(raw.get("open_interest_all"), errors="coerce")
    if "cftc_contract_market_code" in raw and raw["cftc_contract_market_code"].nunique() > 1:
        # a name pattern matched several contracts: keep the one with the largest recent open interest
        last = raw.sort_values("date").groupby("cftc_contract_market_code")["open_interest"].last()
        raw = raw[raw["cftc_contract_market_code"] == last.idxmax()]
    out = pd.DataFrame({"open_interest": raw["open_interest"].values}, index=raw["date"].values)
    for cls, (longs, shorts) in CLASSES[report].items():
        lo, sh = _col(raw, longs, cls, "long"), _col(raw, shorts, cls, "short")
        if lo is None or sh is None:
            continue
        out[f"{cls}_long"], out[f"{cls}_short"] = lo.values, sh.values
        out[f"{cls}_net"] = out[f"{cls}_long"] - out[f"{cls}_short"]
    out.index.name = "date"
    if "market_and_exchange_names" in raw:
        out.attrs["market_name"] = str(raw["market_and_exchange_names"].iloc[0])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def cot_index(net: pd.Series, weeks: int = 156, min_periods: int = 26) -> pd.Series:
    """0-100: where this week's net sits between the trailing `weeks` min and max (inclusive)."""
    lo = net.rolling(weeks, min_periods=min_periods).min()
    hi = net.rolling(weeks, min_periods=min_periods).max()
    rng = (hi - lo).replace(0, np.nan)
    return ((net - lo) / rng * 100).clip(0, 100)


def analytics(df: pd.DataFrame, report: str, weeks: int = 156) -> pd.DataFrame:
    """Add per-class weekly change, % of OI, COT index, z-score and extreme flag."""
    out = df.copy()
    for cls in CLASSES[report]:
        n = f"{cls}_net"
        if n not in out:
            continue
        net = out[n]
        out[f"{cls}_chg"] = net.diff()
        out[f"{cls}_pct_oi"] = net / out["open_interest"] * 100
        out[f"{cls}_index"] = cot_index(net, weeks)
        r = net.rolling(weeks, min_periods=26)
        out[f"{cls}_z"] = (net - r.mean()) / r.std().replace(0, np.nan)
        idx = out[f"{cls}_index"]
        out[f"{cls}_flag"] = np.where(idx > 90, "extreme long", np.where(idx < 10, "extreme short", ""))
    out["oi_chg"] = out["open_interest"].diff()
    return out


def summary(df: pd.DataFrame, report: str) -> pd.DataFrame:
    """Latest week, one row per trader class."""
    last = df.iloc[-1]
    rows = []
    for cls in CLASSES[report]:
        if f"{cls}_net" not in df:
            continue
        g = lambda s: last.get(f"{cls}_{s}", np.nan)  # noqa: E731
        rows.append({"class": cls + (" (spec)" if cls == SPEC[report] else ""), "long": g("long"), "short": g("short"),
                     "net": g("net"), "wk_chg": g("chg"), "net_%oi": g("pct_oi"), "cot_index": g("index"), "z": g("z"),
                     "flag": g("flag") or ""})
    return pd.DataFrame(rows).set_index("class")


def divergence(df: pd.DataFrame, close: pd.Series, report: str, lookback: int = 4,
               price_thresh: float = 0.02, pos_thresh: float = 2.0, cls: str | None = None) -> pd.DataFrame:
    """Price trend vs speculator positioning over `lookback` weeks.
    price up & specs net down (by > pos_thresh pp of OI) -> bearish divergence (rally not backed by specs);
    price down & specs net up -> bullish divergence (specs accumulating into weakness); same direction -> confirming."""
    cls = cls or SPEC[report]
    c = close.copy()
    c.index = pd.DatetimeIndex(c.index).tz_localize(None) if getattr(c.index, "tz", None) is not None else pd.DatetimeIndex(c.index)
    c = c.sort_index()
    weekly = pd.merge_asof(pd.DataFrame(index=df.index.sort_values()).reset_index().rename(columns={"index": "date"}),
                           c.rename("close").reset_index().rename(columns={c.index.name or "index": "date"}),
                           on="date", direction="backward").set_index("date")["close"]
    pct = df[f"{cls}_net"] / df["open_interest"] * 100
    out = pd.DataFrame({"close": weekly, "price_chg": weekly.pct_change(lookback, fill_method=None),
                        "spec_net": df[f"{cls}_net"], "spec_chg_pp_oi": pct.diff(lookback)})
    up, dn = out["price_chg"] > price_thresh, out["price_chg"] < -price_thresh
    pu, pd_ = out["spec_chg_pp_oi"] > pos_thresh, out["spec_chg_pp_oi"] < -pos_thresh
    out["signal"] = np.select(
        [up & pd_, dn & pu, up & pu, dn & pd_],
        ["bearish divergence: price up, specs cutting longs", "bullish divergence: price down, specs adding longs",
         "confirming: price up, specs adding", "confirming: price down, specs selling"], "")
    return out


# =============================================================================== public API
def history(market: str, report: str | None = None, years: int = 3, refresh: bool = False, rows: list[dict] | None = None,
            store: bool = True) -> pd.DataFrame:
    """Weekly COT frame with analytics. Cache-first: only weeks after the cached tail are fetched."""
    from . import cache
    key = resolve(market)
    report = report or default_report(key)
    since = pd.Timestamp.now().normalize() - pd.DateOffset(years=years) - pd.Timedelta(weeks=160)  # room for the 3y index
    ck = f"{report}_{key}"
    if rows is None:
        cached = None if refresh else cache.load("cftc", ck, "1wk")
        fresh = cached is not None and len(cached) > 0 and cached.index[-1] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
        if fresh:  # CFTC publishes Friday for Tuesday's positions: a <10-day-old tail is current
            raw = cached
        else:
            full = cached is None or cached.empty or cached.index[0] > since + pd.Timedelta(days=14)
            start = since if full else cached.index[-1] - pd.Timedelta(days=14)  # overlap a couple of weeks for revisions
            new = parse(fetch_rows(key, report, start), report)
            raw = cache.save("cftc", ck, "1wk", new) if store else new
        src = f"CFTC {report} {DATASETS[report]} (publicreporting.cftc.gov)" + (" (cache)" if fresh else "")
    else:
        raw, src = parse(rows, report), f"CFTC {report} (fixture)"
    if rows is None:
        raw = raw[raw.index >= since]
    out = analytics(raw, report)
    out = out[out.index >= pd.Timestamp.now().normalize() - pd.DateOffset(years=years)] if rows is None else out
    out.attrs.update(source=src, market=MARKETS[key]["label"], report=report, key=key,
                     fetched_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                     last_bar=str(out.index[-1].date()) if len(out) else None)
    return out


def report_for(market: str, report: str | None = None, years: int = 3, lookback: int = 4, prices: pd.DataFrame | None = None) -> dict:
    """Everything the `cot` command prints: summary, recent weeks, divergence."""
    df = history(market, report, years)
    rep, key = df.attrs["report"], df.attrs["key"]
    res = {"market": df.attrs["market"], "report": rep, "as_of": df.attrs["last_bar"], "summary": summary(df, rep), "frame": df}
    ysym = MARKETS[key]["yahoo"]
    if ysym or prices is not None:
        try:
            if prices is None:
                from .data import get_prices
                prices = get_prices(ysym, "1d", period=f"{max(years, 1) + 1}y" if years < 4 else "5y")
            res["divergence"] = divergence(df, prices["close"], rep, lookback)
        except Exception as e:
            res["divergence_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return res


def scan_extremes(report: str | None = None, years: int = 3, markets=None, hi: float = 90, lo: float = 10) -> pd.DataFrame:
    """Spec-class COT index for every mapped market; attrs['errors'] holds per-market failures."""
    rows, errs = [], {}
    for k in markets or MARKETS:
        if k == "eurodollar":
            continue
        try:
            df = history(k, report, years)
            rep = df.attrs["report"]
            last, cls = df.iloc[-1], SPEC[rep]
            row = {"market": k, "label": MARKETS[k]["label"], "report": rep, "date": df.index[-1].date(),
                   "spec_net": last[f"{cls}_net"], "spec_wk_chg": last[f"{cls}_chg"], "spec_%oi": last[f"{cls}_pct_oi"],
                   "spec_index": last[f"{cls}_index"], "spec_z": last[f"{cls}_z"]}
            other = {"legacy": "comm", "disagg": "producer", "tff": "asset_mgr"}[rep]
            if f"{other}_index" in df:
                row[f"other_index ({other})"] = last[f"{other}_index"]
            row["flag"] = "extreme long" if row["spec_index"] > hi else "extreme short" if row["spec_index"] < lo else ""
            rows.append(row)
        except Exception as e:
            errs[k] = f"{type(e).__name__}: {str(e)[:100]}"
    out = pd.DataFrame(rows).set_index("market") if rows else pd.DataFrame()
    if len(out):
        out = out.sort_values("spec_index", ascending=False)
    out.attrs["errors"] = errs
    return out


# =============================================================================== CLI
def cmd_cot(a):
    from .cli import show, src_line
    if a.market.lower() in ("list", "markets"):
        show(pd.DataFrame(MARKETS).T[["label", "code", "yahoo", "kind"]], a.json, title="Mapped COT markets")
        return
    r = report_for(a.market, a.report, a.years, a.lookback)
    df, rep = r["frame"], r["report"]
    show(r["summary"], a.json, title=f"{r['market']} COT ({rep}) as of {r['as_of']}")
    cls = SPEC[rep]
    cols = ["open_interest", "oi_chg"] + [c for c in df if c.startswith(cls) and not c.endswith(("_long", "_short"))]
    show(df[cols].tail(a.tail).iloc[::-1], a.json, title=f"Last {a.tail} weeks: {cls}")
    if "divergence" in r:
        d = r["divergence"].tail(a.tail).iloc[::-1]
        show(d, a.json, title=f"Positioning vs price ({a.lookback}w change, {MARKETS[df.attrs['key']]['yahoo']})")
    elif "divergence_error" in r:
        print(f"\n(price divergence unavailable: {r['divergence_error']})")
    src_line(df)


def cmd_cot_extremes(a):
    from .cli import show
    t = scan_extremes(a.report, a.years)
    show(t if a.all else t[t["flag"] != ""] if len(t) else t, a.json,
         title=f"COT positioning {'(all mapped markets)' if a.all else 'extremes (spec index >90 or <10)'}")
    if t.attrs.get("errors"):
        show(pd.DataFrame({"error": t.attrs["errors"]}), a.json, title="Unavailable markets")
    print(f"\n_Source: CFTC public reporting (Socrata) · {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def register(add):
    q = add("cot", cmd_cot, "CFTC Commitments of Traders for a market (gold, wti, eur, sp500, ust10y...; 'list' for all)")
    q.add_argument("market"); q.add_argument("--report", choices=list(DATASETS)); q.add_argument("--years", type=int, default=3)
    q.add_argument("--lookback", type=int, default=4, help="weeks for the price/positioning divergence")
    q.add_argument("--tail", type=int, default=12)
    q = add("cot-extremes", cmd_cot_extremes, "scan mapped COT markets for positioning extremes")
    q.add_argument("--report", choices=list(DATASETS)); q.add_argument("--years", type=int, default=3)
    q.add_argument("--all", action="store_true", help="show every market, not just extremes")
