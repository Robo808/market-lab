"""Massive (formerly Polygon.io) aggregates for US stocks and ETFs: a live backup and cross-checker.

Key from MASSIVE_API_KEY, or a network secret that injects 'Authorization: Bearer' for api.massive.com.
The free Basic plan keeps 2 years of minute bars at 5 calls a minute. Massive's market-data terms make the
data display-only (https://massive.com/legal/market-data-terms-of-service), so mlab never writes it to the
price cache or to a frozen dataset: it is fetched, compared or shown, then dropped.
"""
from __future__ import annotations

import os
import time

import pandas as pd

API = "https://api.massive.com/v2/aggs/ticker"
SPANS = {"1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"), "30m": (30, "minute"),
         "1h": (1, "hour"), "60m": (1, "hour"), "4h": (4, "hour"), "1d": (1, "day"), "1wk": (1, "week"),
         "1mo": (1, "month")}
HISTORY_DAYS = 730  # free plan


def _session():
    import requests

    from ..config import env
    from ..net import session
    s = session("market-lab/0.1")
    s.mount("https://", requests.adapters.HTTPAdapter())  # no urllib3 retries: 429s are handled below
    if env("MASSIVE_API_KEY"):
        s.headers["Authorization"] = f"Bearer {env('MASSIVE_API_KEY')}"
    return s


def history(symbol: str, start=None, end=None, interval: str = "1d", adjusted: bool = True,
            log=lambda *a: None, **_) -> pd.DataFrame:
    if interval not in SPANS:
        raise ValueError(f"massive: unsupported interval {interval}")
    if not symbol.replace(".", "").replace("-", "").isalpha() or symbol.startswith("^"):
        raise ValueError(f"massive: US stock/ETF tickers only, got {symbol}")
    mult, span = SPANS[interval]
    now = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(start or now - pd.Timedelta(days=365))
    start = start.tz_localize("UTC") if start.tz is None else start
    end = pd.Timestamp(end) if end is not None else now
    s, gap = _session(), float(os.environ.get("MLAB_MASSIVE_GAP", "12.5"))
    url = f"{API}/{symbol.upper().replace('-', '.')}/range/{mult}/{span}/{start.date()}/{end.date()}"
    params, rows, first = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}, [], True
    while url:
        for attempt in range(6):
            if not first:
                time.sleep(gap)
            first = False
            r = s.get(url, params=params, timeout=60)
            if r.status_code != 429:
                break
            log(f"  massive 429, backing off {gap * 2 ** attempt:.0f}s")
            time.sleep(gap * 2 ** attempt)
        if r.status_code in (401, 403):
            raise PermissionError(f"massive {r.status_code}: set MASSIVE_API_KEY or an api.massive.com network secret"
                                  f" ({str(r.json().get('message', ''))[:120] if r.content else ''})")
        r.raise_for_status()
        j = r.json()
        rows += j.get("results") or []
        url, params = j.get("next_url"), None  # next_url carries the cursor and the original query
    if not rows:
        raise LookupError(f"massive: no {interval} bars for {symbol}")
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.index.name = "time"
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap",
                            "n": "trades"})
    cols = ["open", "high", "low", "close", "volume"] + [c for c in ("vwap", "trades") if c in df]
    return df[cols].astype(float)
