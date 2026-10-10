"""Frozen offline datasets: versioned, checksummed snapshots of open historical data, with fixed
train / validation / test splits, so strategies are tested the way ML models are.

    mlab datasets catalog                       # what can be built, source and licence of each
    mlab datasets build xasset-daily            # snapshot -> data/datasets/xasset-daily/v20261010/
    mlab datasets list | show NAME[@VER] | verify NAME[@VER]
    mlab algo backtest tsmom ^FTSE --dataset xasset-daily --split train

Layout (Hive-style, long format, one schema per kind, ready to register as an Iceberg table later):

    $MLAB_DATA_DIR/data/datasets/<name>/<version>/symbol=<SYM>/data.parquet
    $MLAB_DATA_DIR/data/datasets/<name>/<version>/manifest.json

A version is immutable once written: rebuilding creates a new version, `verify` re-hashes every file
against the manifest, and loading the test split needs allow_test=True and is logged next to the data.
"""
from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import re
import shutil
import struct
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

DATASETS_DIR = Path(os.environ.get("MLAB_DATASETS_DIR", DATA_DIR / "data" / "datasets"))
BAR_COLS = ["open", "high", "low", "close", "volume", "adj_close", "spread"]
SERIES_COLS = ["value"]
EVENT_COLS = ["eps_estimate", "eps_actual", "surprise_pct", "hour_et"]
SPLITS = ("train", "validation", "test")

# ---- sources --------------------------------------------------------------------------------
# Licence notes are what each provider's own terms page says (links in docs/wiki/Datasets.md).
# Snapshots stay in the private data dir; none of this data is ever committed to the public repo.
SOURCE_INFO = {
    "yahoo": {"url": "https://finance.yahoo.com (via yfinance)",
              "licence": "Yahoo terms of service: personal, non-commercial use; no redistribution",
              "adjustments": "close = raw close (repair off); adj_close = split- and dividend-adjusted (ETFs/stocks). "
                             "Indices are price-only; futures (=F) are unadjusted front-month rolls"},
    # FRED is deliberately absent: its terms (fred.stlouisfed.org/legal, section (p)) forbid storing,
    # caching or archiving FRED content, so each series is frozen from its original publisher instead.
    "fed_h15": {"url": "https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15",
                "licence": "Federal Reserve Board H.15 release, US government public data",
                "adjustments": "none (constant-maturity yields in percent, investment basis)"},
    "ecb": {"url": "https://data-api.ecb.europa.eu/service/data/EXR/D.<CCY>.EUR.SP00.A",
            "licence": "ECB statistics: free reuse, commercial or not, with the source quoted and data unmodified",
            "adjustments": "none (euro reference rates, 14:15 CET fixing, units of currency per EUR)"},
    "eia": {"url": "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm",
            "licence": "US EIA: US government publications are in the public domain",
            "adjustments": "none (spot USD per barrel, no futures roll)"},
    "boe": {"url": "https://www.bankofengland.co.uk/boeapps/database/",
            "licence": "Bank of England database: UK Open Government Licence; selected FX series excluded "
                       "(LSEG-sourced), fine for private backtests",
            "adjustments": "none (yields and rates in percent, FX spot at 4pm London)"},
    "binance": {"url": "https://data.binance.vision/?prefix=data/spot/monthly/klines/",
                "licence": "Binance public data T&C: CC BY-NC-SA 4.0; personal backtesting allowed, "
                           "no live trading execution on the datasets",
                "adjustments": "none; complete calendar months only"},
    "dukascopy": {"url": "https://datafeed.dukascopy.com/datafeed/<SYM>/...",
                  "licence": "Dukascopy free historical feed: no published licence, treated as personal use only",
                  "adjustments": "mid = (bid+ask)/2 OHLC, spread = ask_close - bid_close; complete months only; CFD cash-index "
                                 "and spot quotes, the closest free proxy for IG's own prices"},
    "yahoo_earnings": {"url": "https://finance.yahoo.com (yfinance get_earnings_dates)",
                       "licence": "Yahoo terms of service: personal, non-commercial use; no redistribution",
                       "adjustments": "event time = announcement timestamp (UTC); hour_et = hour in New York "
                                      "(>=16 after close, <10 before open); the next scheduled date has no actual yet"},
    "french": {"url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
               "licence": "Kenneth R. French Data Library, copyright Fama and French, free download for research",
               "adjustments": "returns in percent per day"},
}


@dataclass(frozen=True)
class Member:
    symbol: str          # name inside the dataset (also the partition key)
    ref: str             # provider symbol / series code
    note: str = ""
    point: float = 1.0   # dukascopy integer price -> price divisor


@dataclass(frozen=True)
class Spec:
    name: str
    source: str
    kind: str            # "bars", "series" or "events"
    interval: str
    members: tuple[Member, ...]
    splits: dict = field(default_factory=dict)   # {"train": end, "validation": end}; test = rest
    start: str = "1970-01-01"
    description: str = ""


def _m(*items) -> tuple[Member, ...]:
    return tuple(Member(*i) if isinstance(i, tuple) else Member(i, i) for i in items)


DEFAULT_SPLITS = {"train": "2014-12-31", "validation": "2019-12-31"}
MEGACAPS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY", "V", "MA",
            "XOM", "UNH", "COST", "NFLX", "AMD", "ORCL", "WMT", "PLTR", "CRM", "ADBE", "INTC", "BAC")

CATALOG: dict[str, Spec] = {s.name: s for s in [
    Spec("xasset-daily", "yahoo", "bars", "1d", _m(
        "^FTSE", "^GSPC", "^NDX", "^DJI", "^GDAXI", "^STOXX50E", "^FCHI", "^N225", "^HSI", "^VIX",
        "GBPUSD=X", "EURUSD=X", "USDJPY=X", "EURGBP=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "GBPJPY=X",
        "DX-Y.NYB", "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "^TNX", "^IRX",
        "SPY", "QQQ", "TLT", "IEF", "SHY", "GLD", "HYG", "LQD",
        "VWRP.L", "VWRL.L", "VUKG.L", "VUKE.L", "ISF.L", "IGLT.L", "VFEG.L", "VJPA.L", "VUAG.L", "SGLN.L",
        "BTC-USD", "ETH-USD"),
        DEFAULT_SPLITS, description="Cross-asset daily bars: indices, FX, futures, US and London ETFs, crypto"),
    Spec("us-stocks-daily", "yahoo", "bars", "1d", _m(*MEGACAPS, "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE",
                                                       "XLV", "XLY", "XLC", "SMH", "^VIX"),
         DEFAULT_SPLITS, description="US mega caps (today's leaders, so survivorship-biased) plus index and sector "
                                     "ETFs, split- and dividend-adjusted closes"),
    Spec("us-earnings", "yahoo_earnings", "events", "event", _m(*MEGACAPS),
         DEFAULT_SPLITS, start="2000-01-01", description="Earnings dates, EPS estimate, actual and surprise for the "
                                                          "US mega caps (about 25 years)"),
    Spec("ust-daily", "fed_h15", "series", "1d", _m(*[(f"UST_{k}", c, f"UST {k} constant maturity") for k, c in (
        ("3M", "RIFLGFCM03_N.B"), ("1Y", "RIFLGFCY01_N.B"), ("2Y", "RIFLGFCY02_N.B"), ("5Y", "RIFLGFCY05_N.B"),
        ("10Y", "RIFLGFCY10_N.B"), ("20Y", "RIFLGFCY20_N.B"), ("30Y", "RIFLGFCY30_N.B"))]),
        DEFAULT_SPLITS, start="1962-01-01", description="US Treasury constant-maturity yields since 1962 (Fed H.15)"),
    Spec("ecb-fx-daily", "ecb", "series", "1d", _m(*[(f"EUR{c}", c, f"{c} per EUR") for c in
                                                       ("USD", "GBP", "JPY", "CHF", "AUD", "CAD", "SEK", "NOK")]),
        DEFAULT_SPLITS, start="1999-01-01", description="ECB euro reference rates since 1999"),
    Spec("eia-oil-daily", "eia", "series", "1d", _m(("BRENT", "RBRTE", "Europe Brent spot FOB"),
                                                    ("WTI", "RWTC", "WTI Cushing spot")),
        DEFAULT_SPLITS, start="1986-01-01", description="Brent and WTI spot since 1986/87 (EIA)"),
    Spec("boe-daily", "boe", "series", "1d", _m(
        ("SONIA", "IUDSOIA", "SONIA overnight rate"), ("BANK_RATE", "IUDBEDR", "Bank Rate"),
        ("GILT_5Y", "IUDSNPY", "5y nominal par gilt yield"), ("GILT_10Y", "IUDMNPY", "10y nominal par gilt yield"),
        ("GILT_20Y", "IUDLNPY", "20y nominal par gilt yield"), ("GBPUSD", "XUDLUSS", "USD per GBP, 4pm London"),
        ("GBPEUR", "XUDLERS", "EUR per GBP"), ("GBPJPY", "XUDLJYS", "JPY per GBP")),
        DEFAULT_SPLITS, description="UK rates, gilt yields and sterling FX (Bank of England)"),
    Spec("crypto-daily", "binance", "bars", "1d", _m(("BTCUSDT", "BTCUSDT"), ("ETHUSDT", "ETHUSDT")),
         {"train": "2021-06-30", "validation": "2023-06-30"}, start="2017-08-01",
         description="BTC and ETH daily bars from Binance public data"),
    Spec("crypto-1h", "binance", "bars", "1h", _m(("BTCUSDT", "BTCUSDT"), ("ETHUSDT", "ETHUSDT")),
         {"train": "2021-06-30", "validation": "2023-06-30"}, start="2017-08-01",
         description="BTC and ETH hourly bars from Binance public data"),
    *[Spec(f"duka-{iv}", "dukascopy", "bars", iv, _m(
        ("EURUSD", "EURUSD", "", 1e5), ("GBPUSD", "GBPUSD", "", 1e5), ("USDJPY", "USDJPY", "", 1e3),
        ("EURGBP", "EURGBP", "", 1e5), ("AUDUSD", "AUDUSD", "", 1e5), ("USDCAD", "USDCAD", "", 1e5),
        ("USDCHF", "USDCHF", "", 1e5), ("XAUUSD", "XAUUSD", "spot gold", 1e3),
        ("BRENT", "BRENTCMDUSD", "Brent CFD", 1e3), ("WTI", "LIGHTCMDUSD", "WTI CFD", 1e3),
        ("UK100", "GBRIDXGBP", "FTSE 100 CFD", 1e3), ("US500", "USA500IDXUSD", "S&P 500 CFD", 1e3),
        ("USTEC", "USATECHIDXUSD", "Nasdaq 100 CFD", 1e3), ("DE40", "DEUIDXEUR", "DAX CFD", 1e3),
        ("EU50", "EUSIDXEUR", "Euro Stoxx 50 CFD", 1e3), ("JP225", "JPNIDXJPY", "Nikkei 225 CFD", 1e3)),
        {"train": "2018-12-31", "validation": "2021-12-31"} if iv == "1h" else DEFAULT_SPLITS,
        start="2012-01-01" if iv == "1h" else "2003-01-01",
        description=f"FX, gold, oil and index CFD {iv} mid bars with spread (Dukascopy)") for iv in ("1d", "1h")],
    Spec("ff-factors-daily", "french", "series", "1d", _m(
        ("MKT_RF", "F-F_Research_Data_Factors_daily:Mkt-RF"), ("SMB", "F-F_Research_Data_Factors_daily:SMB"),
        ("HML", "F-F_Research_Data_Factors_daily:HML"), ("RF", "F-F_Research_Data_Factors_daily:RF"),
        ("MOM", "F-F_Momentum_Factor_daily:Mom")),
        {"train": "1999-12-31", "validation": "2012-12-31"}, start="1926-01-01",
        description="Fama-French 3 factors + momentum, daily since 1926 (US)"),
]}


# ---- fetchers: each returns a frame indexed by UTC time -------------------------------------
def _utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index.name = "time"
    return df.sort_index()


def fetch_yahoo(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .providers import yahoo
    return yahoo.history(m.ref, start=spec.start, end=None, interval=spec.interval)


def fetch_yahoo_earnings(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .earnings import fetch_earnings_dates
    raw = fetch_earnings_dates(m.ref, limit=100)  # Yahoo caps at 100 (about 25 years)
    hour = raw.index.hour.astype(float)
    df = pd.DataFrame({"eps_estimate": raw["EPS Estimate"].values, "eps_actual": raw["Reported EPS"].values,
                       "surprise_pct": raw["Surprise(%)"].values, "hour_et": hour}, index=raw.index)
    df = _utc_index(df)
    # Yahoo repeats some events with a partial row; keep the most complete one per timestamp.
    df["_n"] = df.notna().sum(axis=1)
    df = df.sort_values("_n").groupby(level=0).last().drop(columns="_n")
    return df.sort_index()


def fetch_fed_h15(m: Member, spec: Spec, log=print, _memo={}) -> pd.DataFrame:
    from .net import session
    if "h15" not in _memo:  # one package download covers every maturity
        r = session("market-lab/0.1").get("https://www.federalreserve.gov/datadownload/Output.aspx", timeout=120, params={
            "rel": "H15", "series": "bf17364827e38702b42a58cf8eaa3f78", "lastobs": "", "from": "", "to": "",
            "filetype": "csv", "label": "include", "layout": "seriescolumn", "type": "package"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=5)
        df.index = pd.to_datetime(df.pop("Time Period"))
        _memo["h15"] = df.apply(pd.to_numeric, errors="coerce")
    return _utc_index(_memo["h15"][[m.ref]].rename(columns={m.ref: "value"}).dropna())


def fetch_ecb(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .net import session
    r = session("market-lab/0.1").get(f"https://data-api.ecb.europa.eu/service/data/EXR/D.{m.ref}.EUR.SP00.A",
                                      params={"format": "csvdata", "startPeriod": spec.start[:10]}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    out = pd.DataFrame({"value": pd.to_numeric(df["OBS_VALUE"], errors="coerce").values},
                       index=pd.to_datetime(df["TIME_PERIOD"]))
    return _utc_index(out.dropna())


def fetch_eia(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .net import session
    r = session("market-lab/0.1").get(f"https://www.eia.gov/dnav/pet/hist_xls/{m.ref}d.xls", timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=3, header=None)  # needs xlrd
    out = pd.DataFrame({"value": pd.to_numeric(df[1], errors="coerce").values}, index=pd.to_datetime(df[0]))
    return _utc_index(out.dropna())


def fetch_boe(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .net import session
    # The BoE database answers browser-like User-Agents with an error redirect and python-requests with 403;
    # a plain tool name is served.
    s = session("market-lab/0.1")
    start = pd.Timestamp(spec.start).strftime("%d/%b/%Y")
    r = s.get("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp", timeout=60, params={
        "csv.x": "yes", "Datefrom": start, "Dateto": "now", "SeriesCodes": m.ref, "CSVF": "TN", "UsingCodes": "Y"})
    r.raise_for_status()
    if not r.text.startswith("DATE"):
        raise LookupError(f"boe: unexpected response for {m.ref}: {r.text[:80]!r}")
    df = pd.read_csv(io.StringIO(r.text))
    out = pd.DataFrame({"value": pd.to_numeric(df.iloc[:, 1], errors="coerce").values},
                       index=pd.to_datetime(df.iloc[:, 0], format="%d %b %Y"))
    return _utc_index(out.dropna())


def _binance_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    rows = [ln.split(",") for ln in raw.strip().splitlines() if ln and ln[0].isdigit()]
    a = np.array([[float(x) for x in r[:6]] for r in rows])
    t = a[:, 0]
    # Binance switched spot files to microsecond timestamps from 2025; older files are milliseconds.
    unit = np.where(t > 1e14, 1e6, 1e3)
    idx = pd.to_datetime((t / unit * 1e9).astype("int64"), utc=True)
    return pd.DataFrame(a[:, 1:6], columns=["open", "high", "low", "close", "volume"], index=idx)


def fetch_binance(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .net import session
    s, frames = session(), []
    last_full = (pd.Timestamp.now(tz="UTC").normalize().replace(day=1) - pd.Timedelta(days=1))
    for mo in pd.period_range(pd.Timestamp(spec.start), last_full.tz_localize(None), freq="M"):
        url = (f"https://data.binance.vision/data/spot/monthly/klines/{m.ref}/{spec.interval}/"
               f"{m.ref}-{spec.interval}-{mo.year}-{mo.month:02d}.zip")
        r = s.get(url, timeout=60)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        frames.append(_binance_zip(r.content))
    if not frames:
        raise LookupError(f"binance: no monthly files for {m.ref} {spec.interval}")
    df = pd.concat(frames)
    df.index.name = "time"
    return df[~df.index.duplicated()].sort_index()


class _Throttle:
    """Dukascopy's free feed answers 429 to bursts; pace requests and back off on 429."""

    def __init__(self, gap: float = 1.2):
        self.gap, self.t = gap, 0.0

    def wait(self):
        dt = time.monotonic() - self.t
        if dt < self.gap:
            time.sleep(self.gap - dt)
        self.t = time.monotonic()


_DUKA = _Throttle(float(os.environ.get("MLAB_DUKA_GAP", "1.2")))


def _duka_get(s, url: str, log=print) -> bytes | None:
    for attempt in range(8):
        _DUKA.wait()
        r = s.get(url, timeout=60)
        if r.status_code in (429, 503):
            pause = min(15 * 2 ** attempt, 300)
            log(f"  dukascopy {r.status_code}, backing off {pause}s")
            time.sleep(pause)
            continue
        if r.status_code == 404 or not r.content:
            return None
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"dukascopy kept rate-limiting {url}")


def _duka_decode(blob: bytes, base: pd.Timestamp, point: float) -> pd.DataFrame:
    """bi5 candles: LZMA, 24-byte big-endian records (uint32 secs from base, int32 open, close, low, high,
    float32 volume)."""
    d = lzma.decompress(blob)
    n = len(d) // 24
    rec = np.array([struct.unpack(">I4if", d[i * 24:(i + 1) * 24]) for i in range(n)], dtype=float)
    if not n:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    idx = base + pd.to_timedelta(rec[:, 0], unit="s")
    df = pd.DataFrame({"open": rec[:, 1], "close": rec[:, 2], "low": rec[:, 3], "high": rec[:, 4]}, index=idx) / point
    df["volume"] = rec[:, 5]
    flat = (df["volume"] == 0) & (df["high"] == df["low"])  # weekend/holiday filler bars
    return df[~flat][["open", "high", "low", "close", "volume"]]


def fetch_dukascopy(m: Member, spec: Spec, log=print) -> pd.DataFrame:
    from .net import session
    s = session()
    s.mount("https://", __import__("requests").adapters.HTTPAdapter())  # own 429 handling, no urllib3 retry storm
    now = pd.Timestamp.now(tz="UTC")
    this_year = pd.Timestamp(year=now.year, month=1, day=1, tz="UTC")
    months = [(p.to_timestamp().tz_localize("UTC"), f"{p.year}/{p.month - 1:02d}/{{side}}_candles_hour_1.bi5", False)
              for p in pd.period_range(pd.Timestamp(spec.start), now.tz_localize(None), freq="M")]
    if spec.interval == "1d":
        # Yearly day-candle files exist for complete years only; the current year is rebuilt from
        # monthly hour candles (complete months), resampled to UTC days.
        periods = [(pd.Timestamp(year=y, month=1, day=1, tz="UTC"), f"{y}/{{side}}_candles_day_1.bi5", False)
                   for y in range(pd.Timestamp(spec.start).year, now.year)]
        periods += [(b, p, True) for b, p, _ in months if b >= max(this_year, pd.Timestamp(spec.start, tz="UTC"))]
    elif spec.interval == "1h":
        periods = months
    else:
        raise ValueError("dukascopy datasets support 1d and 1h")
    frames = []
    for base, path, to_daily in periods:
        sides = {}
        for side in ("BID", "ASK"):
            blob = _duka_get(s, f"https://datafeed.dukascopy.com/datafeed/{m.ref}/{path.format(side=side)}", log)
            if blob:
                sides[side] = _duka_decode(blob, base, m.point)
        if len(sides) < 2 or sides["BID"].empty:
            continue
        b, a = sides["BID"].align(sides["ASK"], join="inner", axis=0)
        mid = (b[["open", "high", "low", "close"]] + a[["open", "high", "low", "close"]]) / 2
        mid["volume"] = b["volume"]
        mid["spread"] = a["close"] - b["close"]
        if to_daily:
            mid = mid.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                          "volume": "sum", "spread": "last"}).dropna(subset=["close"])
        frames.append(mid)
    if not frames:
        raise LookupError(f"dukascopy: no data for {m.ref}")
    df = pd.concat(frames)
    df = df[df.index <= now]
    df.index.name = "time"
    return df[~df.index.duplicated()].sort_index()


def _french_csv(name: str, s) -> pd.DataFrame:
    r = s.get(f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip", timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith(",") or ln.lower().startswith("date,"))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows = []
    for ln in lines[head + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break  # daily block ends at the first blank / footer line
        rows.append([parts[0]] + parts[1:len(cols) + 1])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_french(m: Member, spec: Spec, log=print, _memo={}) -> pd.DataFrame:
    from .net import session
    file, col = m.ref.split(":")
    if file not in _memo:
        _memo[file] = _french_csv(file, session())
    return _utc_index(_memo[file][[col]].rename(columns={col: "value"}).dropna())


FETCHERS = {"yahoo": fetch_yahoo, "yahoo_earnings": fetch_yahoo_earnings, "fed_h15": fetch_fed_h15, "ecb": fetch_ecb, "eia": fetch_eia,
            "boe": fetch_boe, "binance": fetch_binance,
            "dukascopy": fetch_dukascopy, "french": fetch_french}


# ---- quality checks -------------------------------------------------------------------------
def quality(df: pd.DataFrame, kind: str, interval: str) -> dict:
    if kind == "events":
        return {"rows": int(len(df)), "start": str(df.index[0]), "end": str(df.index[-1]),
                "with_actual": int(df["eps_actual"].notna().sum()), "duplicate_times": int(df.index.duplicated().sum())}
    col = "close" if kind == "bars" else "value"
    s = df[col].astype(float)
    q = {"rows": int(len(df)), "start": str(df.index[0]), "end": str(df.index[-1]),
         "nan_" + col: int(s.isna().sum()), "duplicate_times": int(df.index.duplicated().sum())}
    gaps = df.index.to_series().diff().dropna()
    q["max_gap_days"] = round(gaps.max() / pd.Timedelta(days=1), 2) if len(gaps) else 0
    if kind == "bars":
        q["non_positive"] = int((s <= 0).sum())
        bad = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
        q["ohlc_inconsistent"] = int(bad.sum())
        r = np.log(s[s > 0]).diff().abs()
        thr = 0.25 if interval == "1d" else 0.10
        jumps = r[r > thr]
        q["jumps_over_%d%%" % int(thr * 100)] = int(len(jumps))
        q["largest_jumps"] = {str(k)[:19]: round(float(v), 4) for k, v in jumps.nlargest(5).items()}
        if "spread" in df and df["spread"].notna().any():
            q["median_spread"] = float(df["spread"].median())
    return q


# ---- build ----------------------------------------------------------------------------------
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def _long(df: pd.DataFrame, kind: str, symbol: str) -> pd.DataFrame:
    cols = {"bars": BAR_COLS, "series": SERIES_COLS, "events": EVENT_COLS}[kind]
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c].astype("float64") if c in df else np.nan
    out = out.reset_index().rename(columns={"index": "time"})
    out["time"] = pd.to_datetime(out["time"], utc=True).astype("datetime64[us, UTC]")
    out.insert(1, "symbol", symbol)
    return out


def new_version(name: str, when: pd.Timestamp | None = None) -> str:
    base = f"v{(when or pd.Timestamp.now(tz='UTC')):%Y%m%d}"
    v, n = base, 1
    while (DATASETS_DIR / name / v).exists():
        n += 1
        v = f"{base}.{n}"
    return v


def splits_for(spec: Spec, end: str) -> dict:
    tr, va = pd.Timestamp(spec.splits["train"]), pd.Timestamp(spec.splits["validation"])
    return {"train": [spec.start, str(tr.date())],
            "validation": [str((tr + pd.Timedelta(days=1)).date()), str(va.date())],
            "test": [str((va + pd.Timedelta(days=1)).date()), end[:10]]}


def build(name: str, members: list[str] | None = None, start: str | None = None, log=print,
          fetch=None) -> Path:
    """Fetch every member, write an immutable new version and its manifest. Returns the version dir."""
    spec = CATALOG[name]
    if start:
        spec = Spec(**{**spec.__dict__, "start": start})
    chosen = [m for m in spec.members if not members or m.symbol in members or m.ref in members]
    fetcher = fetch or FETCHERS[spec.source]
    version = new_version(name)
    final = DATASETS_DIR / name / version
    tmp = DATASETS_DIR / name / f".{version}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files, failed = {}, {}
    for m in chosen:
        try:
            df = fetcher(m, spec, log=log)
            df = df[df.index >= pd.Timestamp(spec.start, tz="UTC")]
            if df.empty:
                raise LookupError("no rows after start date")
            df = df[~df.index.duplicated(keep="last")].sort_index()
            p = tmp / f"symbol={_safe(m.symbol)}" / "data.parquet"
            p.parent.mkdir(parents=True)
            _long(df, spec.kind, m.symbol).to_parquet(p, index=False, compression="zstd")
            files[m.symbol] = {"path": str(p.relative_to(tmp)), "ref": m.ref, "note": m.note,
                               "sha256": _sha256(p), "bytes": p.stat().st_size,
                               "qc": quality(df, spec.kind, spec.interval)}
            log(f"  {m.symbol}: {len(df)} rows {str(df.index[0])[:10]} -> {str(df.index[-1])[:10]}")
        except Exception as e:
            failed[m.symbol] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"  {m.symbol}: FAILED {failed[m.symbol]}")
    if not files:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"{name}: every member failed: {failed}")
    built = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    end = max(f["qc"]["end"] for f in files.values())
    import pyarrow
    manifest = {
        "dataset": name, "version": version, "built_at": built, "kind": spec.kind, "interval": spec.interval,
        "source": spec.source, **SOURCE_INFO[spec.source], "description": spec.description,
        "columns": ["time", "symbol"] + {"bars": BAR_COLS, "series": SERIES_COLS, "events": EVENT_COLS}[spec.kind],
        "splits": splits_for(spec, end), "split_rule": "inclusive date ranges on the bar's UTC date",
        "members": files, "failed": failed,
        "software": {"mlab_commit": _git_commit(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    tmp.rename(final)
    for p in final.rglob("*"):
        if p.is_file():
            try:
                p.chmod(0o444)
            except OSError:
                pass
    return final


# ---- read -----------------------------------------------------------------------------------
def versions(name: str) -> list[str]:
    d = DATASETS_DIR / name
    vs = [p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")] if d.exists() else []
    return sorted(vs, key=lambda v: [int(x) for x in re.findall(r"\d+", v)])


def resolve(ref: str) -> tuple[str, str, Path]:
    name, _, ver = ref.partition("@")
    vs = versions(name)
    if not vs:
        raise LookupError(f"dataset '{name}' has no built versions in {DATASETS_DIR} (run `mlab datasets build {name}`)")
    ver = ver or vs[-1]
    if ver not in vs:
        raise LookupError(f"{name}@{ver} not found; have {', '.join(vs)}")
    return name, ver, DATASETS_DIR / name / ver


def manifest(ref: str) -> dict:
    return json.loads((resolve(ref)[2] / "manifest.json").read_text())


def load(ref: str, symbols: list[str] | None = None, split: str | None = None, allow_test: bool = False,
         note: str = "") -> dict[str, pd.DataFrame]:
    """{symbol: frame indexed by UTC time}. split in train | validation | test | train+validation | None (all)."""
    name, ver, root = resolve(ref)
    man = json.loads((root / "manifest.json").read_text())
    parts = split.split("+") if split else []
    if any(p not in SPLITS for p in parts):
        raise ValueError(f"split must be one of {SPLITS} or 'train+validation'")
    if (not split or "test" in parts) and not allow_test:
        raise PermissionError(f"{name}@{ver}: the test split is held out; pass allow_test=True (CLI --allow-test) "
                              "once, for the final grade")
    if (not split or "test" in parts):
        log_line = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "split": split or "all",
                    "symbols": symbols, "note": note}
        try:
            with open(DATASETS_DIR / name / f"test_access.{ver}.jsonl", "a") as f:
                f.write(json.dumps(log_line) + "\n")
        except OSError:
            pass
    out = {}
    want = symbols or list(man["members"])
    for s in want:
        meta = man["members"].get(s)
        if meta is None:
            raise LookupError(f"{name}@{ver} has no member '{s}'; members: {', '.join(man['members'])}")
        df = pd.read_parquet(root / meta["path"]).set_index("time").drop(columns="symbol")
        df = df.dropna(axis=1, how="all")
        if parts:
            mask = np.zeros(len(df), bool)
            d = df.index.tz_convert("UTC").normalize().tz_localize(None)
            for p in parts:
                a, b = man["splits"][p]
                mask |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
            df = df[mask]
        df.attrs.update({"source": f"dataset {name}@{ver} ({man['source']} {meta['ref']}) split={split or 'all'}",
                         "fetched_at": man["built_at"], "last_bar": str(df.index[-1]) if len(df) else None,
                         "sha256": meta["sha256"]})
        out[s] = df
    return out


def verify(ref: str) -> dict:
    name, ver, root = resolve(ref)
    man = json.loads((root / "manifest.json").read_text())
    bad = {s: "missing" if not (root / m["path"]).exists() else "sha256 mismatch"
           for s, m in man["members"].items()
           if not (root / m["path"]).exists() or _sha256(root / m["path"]) != m["sha256"]}
    return {"dataset": f"{name}@{ver}", "files": len(man["members"]), "ok": not bad, "bad": bad}


def listing() -> pd.DataFrame:
    rows = []
    for name in sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir()) if DATASETS_DIR.exists() else []:
        for v in versions(name):
            try:
                m = json.loads((DATASETS_DIR / name / v / "manifest.json").read_text())
            except Exception:
                continue
            q = [x["qc"] for x in m["members"].values()]
            rows.append({"dataset": f"{name}@{v}", "source": m["source"], "interval": m["interval"],
                         "members": len(m["members"]), "failed": len(m["failed"]),
                         "start": min(x["start"] for x in q)[:10], "end": max(x["end"] for x in q)[:10],
                         "mb": round(sum(x["bytes"] for x in m["members"].values()) / 1e6, 2),
                         "built": m["built_at"][:16]})
    return pd.DataFrame(rows)


def catalog_table() -> pd.DataFrame:
    return pd.DataFrame([{"dataset": s.name, "source": s.source, "interval": s.interval, "members": len(s.members),
                          "start": s.start, "train_to": s.splits["train"], "validation_to": s.splits["validation"],
                          "licence": SOURCE_INFO[s.source]["licence"][:70]} for s in CATALOG.values()])


# ---- CLI ------------------------------------------------------------------------------------
def cmd_datasets(a):
    from .cli import show
    act = a.action
    if act == "catalog":
        show(catalog_table(), a.json, title="Buildable datasets (mlab datasets build NAME)")
        return
    if act == "list":
        df = listing()
        show(df, a.json, title=f"Built datasets in {DATASETS_DIR}") if len(df) else print(f"none built yet in {DATASETS_DIR}")
        return
    if not a.name:
        raise SystemExit("dataset name required")
    if act == "build":
        names = list(CATALOG) if a.name == "all" else [a.name]
        for n in names:
            print(f"building {n} ...")
            path = build(n, a.members, a.start)
            m = json.loads((path / "manifest.json").read_text())
            print(f"wrote {path}  ({len(m['members'])} members, {len(m['failed'])} failed)")
        return
    if act == "show":
        m = manifest(a.name)
        rows = [{"symbol": s, "ref": x["ref"], **{k: v for k, v in x["qc"].items() if k != "largest_jumps"}}
                for s, x in m["members"].items()]
        print(f"{m['dataset']}@{m['version']} · {m['source']} · built {m['built_at']}\n"
              f"licence: {m['licence']}\nadjustments: {m['adjustments']}\nsplits: {m['splits']}")
        if m["failed"]:
            print(f"failed: {m['failed']}")
        show(pd.DataFrame(rows), a.json, title="Members and quality checks")
        return
    if act == "verify":
        r = verify(a.name)
        print(json.dumps(r, indent=2))
        if not r["ok"]:
            raise SystemExit(1)


def register(add):
    q = add("datasets", cmd_datasets,
            "frozen offline datasets: catalog | build NAME|all | list | show NAME[@VER] | verify NAME[@VER]")
    q.add_argument("action", choices=["catalog", "build", "list", "show", "verify"])
    q.add_argument("name", nargs="?")
    q.add_argument("--members", nargs="*", help="build only these members (symbol or provider code)")
    q.add_argument("--start", help="override the dataset's start date")
