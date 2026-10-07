"""Stooq keyless daily CSV. Symbols: aapl.us, ^spx, eurusd, gc.f, btcusd. Good Yahoo fallback."""
from __future__ import annotations

import io

import pandas as pd

from ..net import session

INTERVALS = {"1d": "d", "1wk": "w", "1mo": "m"}


def to_stooq(symbol: str) -> str:
    s = symbol.lower()
    if s.endswith("=x"):
        return s[:-2]
    if s.startswith("^"):
        return {"^gspc": "^spx", "^ixic": "^ndq", "^dji": "^dji", "^ftse": "^ftx"}.get(s, s)
    if "-usd" in s:
        return s.replace("-", "")
    if "." not in s and s.isalpha():
        return f"{s}.us"
    return s


def history(symbol: str, start=None, end=None, interval: str = "1d", **_) -> pd.DataFrame:
    if interval not in INTERVALS:
        raise ValueError("stooq supports 1d/1wk/1mo only")
    params = {"s": to_stooq(symbol), "i": INTERVALS[interval]}
    if start is not None:
        params["d1"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end is not None:
        params["d2"] = pd.Timestamp(end).strftime("%Y%m%d")
    r = session().get("https://stooq.com/q/d/l/", params=params, timeout=20)
    r.raise_for_status()
    if not r.text or r.text.startswith("No data"):
        raise LookupError(f"stooq returned no data for {symbol}")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df = df.set_index("time").drop(columns=["date"])
    if "volume" not in df:
        df["volume"] = float("nan")
    return df[["open", "high", "low", "close", "volume"]]
