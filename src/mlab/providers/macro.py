"""Macro: FRED series (keyless CSV, or API with FRED_API_KEY) and ECB FX reference rates."""
from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import env
from ..net import session

# Curated dashboard: the series a macro (Soros-style) read starts from.
DASHBOARD = {
    "DGS2": "UST 2y yield", "DGS10": "UST 10y yield", "T10Y2Y": "10y-2y spread",
    "T10Y3M": "10y-3m spread", "DFF": "Fed funds effective", "CPIAUCSL": "CPI index",
    "CPILFESL": "Core CPI index", "PCEPILFE": "Core PCE index", "UNRATE": "Unemployment rate",
    "ICSA": "Initial jobless claims", "WALCL": "Fed balance sheet", "RRPONTSYD": "ON RRP usage",
    "WTREGEN": "Treasury General Account", "BAMLH0A0HYM2": "HY OAS", "BAMLC0A0CM": "IG OAS",
    "DTWEXBGS": "Broad USD index", "VIXCLS": "VIX", "T5YIE": "5y breakeven",
    "DCOILWTICO": "WTI crude", "M2SL": "M2 money stock", "NFCI": "Chicago Fed financial conditions",
}


def fred(series: str, start=None) -> pd.Series:
    key = env("FRED_API_KEY")
    # fred.stlouisfed.org silently stalls browser-like and custom User-Agents; the plain requests UA is served.
    s = session(requests.utils.default_user_agent())
    if key:
        r = s.get("https://api.stlouisfed.org/fred/series/observations", timeout=20, params={
            "series_id": series, "api_key": key, "file_type": "json",
            **({"observation_start": pd.Timestamp(start).date().isoformat()} if start else {})})
        r.raise_for_status()
        obs = r.json()["observations"]
        ser = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    else:
        r = s.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        ser = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))
    ser = pd.to_numeric(ser.replace(".", None), errors="coerce").dropna()
    ser.name = series
    if start is not None:
        ser = ser[ser.index >= pd.Timestamp(start)]
    return ser


def dashboard(start="2015-01-01", series=None) -> pd.DataFrame:
    """Latest value, 1m/3m/1y change for each dashboard series."""
    rows = []
    for sid in series or DASHBOARD:
        try:
            s = fred(sid, start=start)
        except Exception as e:  # keep going; report what failed
            rows.append({"series": sid, "name": DASHBOARD.get(sid, sid), "error": str(e)[:80]})
            continue
        last_date, last = s.index[-1], s.iloc[-1]

        def chg(days):
            past = s[s.index <= last_date - pd.Timedelta(days=days)]
            return None if past.empty else last - past.iloc[-1]
        rows.append({"series": sid, "name": DASHBOARD.get(sid, sid), "date": last_date.date(),
                     "last": round(last, 3), "chg_1m": chg(30), "chg_3m": chg(91), "chg_1y": chg(365)})
    return pd.DataFrame(rows)


def ecb_fx(base="EUR", symbols="USD,GBP,JPY,CHF", start=None, end=None) -> pd.DataFrame:
    s = session()
    if start:
        url = f"https://api.frankfurter.app/{pd.Timestamp(start).date()}..{'' if end is None else pd.Timestamp(end).date()}"
    else:
        url = "https://api.frankfurter.app/latest"
    r = s.get(url, params={"from": base, "to": symbols}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "rates" in j and isinstance(next(iter(j["rates"].values())), dict):
        return pd.DataFrame(j["rates"]).T.sort_index()
    return pd.DataFrame([j["rates"]], index=[j["date"]])
