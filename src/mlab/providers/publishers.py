"""Original publishers behind the economic dictionary, fetched direct so their data can be stored when the
licence allows it (FRED mirrors may not be stored):

- HMRC uktradeinfo OData API (Open Government Licence v3): UK overseas trade by commodity code, monthly.
  No key; 60 requests a minute; filters with `or` chains (the `in` operator is rejected).
- BLS Public Data API v2 (US public domain): any BLS series id. BLS_API_KEY raises the daily limit from 25
  to 500 queries; without a key a shared egress IP can hit the limit early.
- Federal Reserve Board G.17 (public domain): industrial production and capacity utilisation for every market
  and industry group, from the release's ipdisk text files. No key.
- Census Monthly Retail Trade Survey workbook (US federal work): retail and food services sales by kind of
  business since 1992, seasonally adjusted block. No key (the Census Data API now needs one).
"""
from __future__ import annotations

import io
import re
import time
from functools import lru_cache

import pandas as pd

from ..config import env
from ..net import session

HMRC = "https://api.uktradeinfo.com"
BLS = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
G17 = "https://www.federalreserve.gov/releases/g17/Current/ipdisk/{}.txt"
G17_TABLES = {"IP": "ip_sa", "CU": "utl_sa"}
MRTS = "https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx"
FLOWS = {"imports": (1, 3), "exports": (2, 4), "eu_imports": (1,), "noneu_imports": (3,),
         "eu_exports": (2,), "noneu_exports": (4,)}
_last_call = [0.0]


def _hmrc_get(path: str, params: dict) -> list[dict]:
    """GET with a 1.1 s spacing (60/min limit) and @odata.nextLink paging."""
    s, out, url = session(), [], f"{HMRC}/{path}"
    while url:
        wait = 1.1 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        r = s.get(url, params=params, timeout=60)
        _last_call[0] = time.time()
        r.raise_for_status()
        j = r.json()
        out += j.get("value", [])
        url, params = j.get("@odata.nextLink"), None
    return out


def hmrc_commodities(codes: list[str]) -> pd.DataFrame:
    """CN8 commodity ids under each HS prefix (2, 4, 6 or 8 digits)."""
    rows = []
    for c in codes:
        c = c.replace(".", "")
        field = {2: "Hs2Code", 4: "Hs4Code", 6: "Hs6Code"}.get(len(c), "Cn8Code")
        rows += _hmrc_get("Commodity", {"$filter": f"{field} eq '{c}'",
                                        "$select": "CommodityId,Cn8Code,Hs6Code,Hs6Description"})
    return pd.DataFrame(rows).drop_duplicates("CommodityId") if rows else pd.DataFrame()


def hmrc_trade(codes: str | list[str], flow: str = "imports", start: str = "2010-01") -> pd.DataFrame:
    """Monthly UK trade summed over the HS codes: value (GBP) and net_mass (kg), indexed by month start."""
    codes = [c.strip() for c in codes.split(",")] if isinstance(codes, str) else codes
    ids = hmrc_commodities(codes)["CommodityId"].tolist() if codes else []
    if not ids:
        raise KeyError(f"no HMRC commodity under {codes}")
    flows = " or ".join(f"FlowTypeId eq {f}" for f in FLOWS[flow])
    month = int(pd.Timestamp(start).strftime("%Y%m"))
    rows = []
    for i in range(0, len(ids), 40):  # keep the filter URL short
        chunk = " or ".join(f"CommodityId eq {c}" for c in ids[i:i + 40])
        rows += _hmrc_get("OTS", {"$apply": f"filter(({chunk}) and MonthId ge {month} and ({flows}))"
                                            "/groupby((MonthId),aggregate(Value with sum as Value,"
                                            "NetMass with sum as NetMass))"})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["value", "net_mass"])
    df = df.groupby("MonthId")[["Value", "NetMass"]].sum()
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
    df.columns = ["value", "net_mass"]
    df.attrs["source"] = f"HMRC uktradeinfo OTS, {flow}, HS {','.join(codes)} ({len(ids)} CN8 codes)"
    return df.sort_index()


def bls(series_id: str, start_year: int = 2000, end_year: int | None = None) -> pd.Series:
    """One BLS series (monthly M01-M12 or quarterly Q01-Q04 periods; annual M13 rows dropped)."""
    end_year = end_year or pd.Timestamp.now().year
    key = env("BLS_API_KEY")
    span = 20 if key else 10  # years per request
    obs = []
    for y0 in range(start_year, end_year + 1, span):
        body = {"seriesid": [series_id], "startyear": str(y0), "endyear": str(min(y0 + span - 1, end_year))}
        if key:
            body["registrationkey"] = key
        r = session().post(BLS, json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS {series_id}: {' '.join(j.get('message') or [j.get('status', '?')])[:160]}")
        obs += j["Results"]["series"][0]["data"]
    return _bls_series(series_id, obs)


def _bls_series(series_id: str, obs: list[dict]) -> pd.Series:
    vals = {}
    for o in obs:
        p = o["period"]
        if p.startswith("M") and p != "M13":
            vals[pd.Timestamp(f"{o['year']}-{p[1:]}-01")] = o["value"]
        elif p.startswith("Q") and p != "Q05":
            vals[pd.Period(f"{o['year']}Q{int(p[1:])}", "Q").to_timestamp()] = o["value"]
    s = pd.to_numeric(pd.Series(vals, name=series_id), errors="coerce").dropna().sort_index()
    s.attrs["source"] = f"BLS {series_id}"
    return s


# ---- Federal Reserve G.17 and Census MRTS -----------------------------------------------------------------------
def parse_g17(text: str, codes: set[str] | None = None) -> pd.DataFrame:
    """G.17 ip_sa.txt / utl_sa.txt: rows `"CODE"  YEAR  v1 .. v12`, blank cells where not yet published."""
    rows = {}
    for ln in text.splitlines():
        mt = re.match(r'^"([A-Z0-9.]+)"\s+(\d{4})\s+(.*)$', ln)
        if not mt or (codes and mt.group(1) not in codes):
            continue
        code, year = mt.group(1), int(mt.group(2))
        for i, v in enumerate(mt.group(3).split()[:12]):
            rows.setdefault(code, {})[pd.Timestamp(year, i + 1, 1)] = float(v)
    return pd.DataFrame({c: pd.Series(v) for c, v in rows.items()}).sort_index()


def _mrts_key(code, label) -> str:
    code = "" if pd.isna(code) else str(code).strip()
    return code if code and code[0].isdigit() else str(label).strip()


def parse_mrts_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """One year sheet of mrtssales92-present.xlsx (header=None) -> seasonally adjusted sales, one column per code."""
    lab = sheet[1].map(lambda v: "" if pd.isna(v) else str(v).strip())
    month = re.compile(r"^([A-Z][a-z]{2})\.? (\d{4})(\(p\))?$")  # "Jan. 2026", "May 2026", "Aug. 2026(p)"
    hdr = next(i for i in sheet.index if month.match(str(sheet.iat[i, 2])))
    months = {j: pd.to_datetime(" ".join(mt.group(1, 2)), format="%b %Y")
              for j in range(2, sheet.shape[1]) if (mt := month.match(str(sheet.iat[hdr, j]).strip()))}
    start = next(i for i in sheet.index if lab[i].startswith("ADJUSTED"))
    out = {}
    for i in sheet.index[sheet.index > start]:
        if not lab[i]:
            if out:
                break  # the adjusted block ends at the first blank row
            continue
        key = _mrts_key(sheet.iat[i, 0], lab[i])
        out[key] = {d: pd.to_numeric(sheet.iat[i, j], errors="coerce") for j, d in months.items()}
    return pd.DataFrame(out).sort_index()


def parse_mrts(book: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [parse_mrts_sheet(s) for name, s in book.items() if name.strip().isdigit()]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


@lru_cache(maxsize=4)
def g17_table(table: str = "ip_sa") -> pd.DataFrame:
    """Every series of one G.17 file (ip_sa: production, utl_sa: utilisation), fetched once per process."""
    r = session().get(G17.format(table), timeout=120)
    r.raise_for_status()
    df = parse_g17(r.text)
    df.attrs["source"] = f"Federal Reserve G.17 {table} {G17.format(table)}"
    return df


def fed_g17(sid: str) -> pd.Series:
    """One G.17 series by dictionary id: G17-IP-<code> (production) or G17-CU-<code> (utilisation)."""
    m = re.match(r"^G17-(IP|CU)-([A-Z0-9.]+)$", sid)
    if not m:
        raise ValueError(f"G.17 id {sid!r}: expected G17-IP-<code> or G17-CU-<code>")
    df = g17_table(G17_TABLES[m.group(1)])
    s = df[m.group(2)].dropna().rename(sid)
    s.attrs["source"] = f"Federal Reserve G.17 {m.group(2)}"
    return s


@lru_cache(maxsize=1)
def mrts_table() -> pd.DataFrame:
    """The whole seasonally adjusted MRTS block, 1992 to the latest (preliminary) month, fetched once per process."""
    r = session().get(MRTS, timeout=120)
    r.raise_for_status()
    df = parse_mrts(pd.read_excel(io.BytesIO(r.content), sheet_name=None, header=None))
    df.attrs["source"] = f"Census MRTS {MRTS}"
    return df


def census_mrts(code: str) -> pd.Series:
    """One MRTS kind of business by NAICS code ("443") or total-row label, seasonally adjusted, USD millions."""
    s = mrts_table()[code].dropna().rename(code)
    s.attrs["source"] = f"Census MRTS {code}"
    return s
