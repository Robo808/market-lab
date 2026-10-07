"""Crypto: Binance klines (with public mirror), Kraken fallback, CoinGecko market overview, Fear & Greed."""
from __future__ import annotations

import pandas as pd

from ..net import session

BINANCE_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
BINANCE_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
KRAKEN_INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}


def to_pair(symbol: str) -> str:
    """BTC, BTC-USD, btcusdt -> BTCUSDT."""
    s = symbol.upper().replace("-", "").replace("/", "")
    if s.endswith("USD"):
        s += "T"
    if not any(s.endswith(q) and len(s) > len(q) for q in ("USDT", "USDC", "BTC", "ETH", "EUR")):
        s += "USDT"
    return s


def binance(symbol: str, interval: str = "1d", start=None, end=None, limit: int = 1000) -> pd.DataFrame:
    interval = {"1wk": "1w", "1mo": "1M"}.get(interval, interval)
    if interval not in BINANCE_INTERVALS:
        raise ValueError(f"binance interval must be one of {sorted(BINANCE_INTERVALS)}")
    params = {"symbol": to_pair(symbol), "interval": interval, "limit": limit}
    if start is not None:
        params["startTime"] = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    if end is not None:
        params["endTime"] = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    s, last_err, rows = session(), None, []
    for host in BINANCE_HOSTS:
        try:
            while True:  # page forward until end or no more data
                r = s.get(f"{host}/api/v3/klines", params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                rows += batch
                if len(batch) < limit or start is None:
                    break
                params["startTime"] = batch[-1][0] + 1
            break
        except Exception as e:
            last_err, rows = e, []
    if not rows:
        raise LookupError(f"binance: no data for {symbol} ({last_err})")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.set_index("time").astype(float)


def kraken(symbol: str, interval: str = "1d", start=None, **_) -> pd.DataFrame:
    pair = symbol.upper().replace("-", "").replace("/", "").replace("USDT", "USD")
    if not pair.endswith(("USD", "EUR")):
        pair += "USD"
    params = {"pair": pair, "interval": KRAKEN_INTERVALS[{"1wk": "1w"}.get(interval, interval)]}
    if start is not None:
        params["since"] = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = session().get("https://api.kraken.com/0/public/OHLC", params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise LookupError(f"kraken: {j['error']}")
    key = next(k for k in j["result"] if k != "last")
    df = pd.DataFrame(j["result"][key], columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)


def history(symbol: str, start=None, end=None, interval: str = "1d", **_) -> pd.DataFrame:
    try:
        return binance(symbol, interval, start, end)
    except Exception:
        return kraken(symbol, interval, start)


def market_overview(n: int = 25, vs: str = "usd") -> pd.DataFrame:
    r = session().get("https://api.coingecko.com/api/v3/coins/markets", timeout=20, params={
        "vs_currency": vs, "order": "market_cap_desc", "per_page": n, "page": 1,
        "price_change_percentage": "24h,7d,30d"})
    r.raise_for_status()
    cols = ["symbol", "name", "current_price", "market_cap", "total_volume",
            "price_change_percentage_24h_in_currency", "price_change_percentage_7d_in_currency",
            "price_change_percentage_30d_in_currency", "ath_change_percentage"]
    return pd.DataFrame(r.json())[cols]


def fear_greed(days: int = 30) -> pd.DataFrame:
    r = session().get("https://api.alternative.me/fng/", params={"limit": days}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    return df.set_index("time")[["value", "value_classification"]].sort_index()
