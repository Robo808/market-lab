"""One entry point for prices: get_prices("AAPL"), get_prices("ig:IX.D.FTSE.DAILY.IP"),
get_prices("crypto:BTC", interval="4h"). Routes to a provider, falls back on failure, caches to parquet.

Every returned frame carries .attrs["source"] and .attrs["fetched_at"] so analyses can cite them.
"""
from __future__ import annotations

import re

import pandas as pd

from . import cache

PERIODS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653, "max": 365 * 40}
IG_EPIC = re.compile(r"^[A-Z]{2}\.[A-Z]\.[A-Z0-9_]+\.[A-Z0-9_]+\.[A-Z0-9_]+$")
US_TICKER = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")
NO_CACHE = {"massive"}  # Massive's terms are display-only: fetched and shown, never written to the cache
YF_INTRADAY_LIMIT = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729, "90m": 59}


def route(symbol: str) -> tuple[str, str]:
    """Return (provider, provider_symbol)."""
    if ":" in symbol:
        p, s = symbol.split(":", 1)
        return p.lower(), s
    if IG_EPIC.match(symbol):
        return "ig", symbol
    return "yahoo", symbol


def _window(start, end, period):
    end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC")
    else:
        start_ts = (end_ts or pd.Timestamp.now(tz="UTC")) - pd.Timedelta(days=PERIODS.get(period or "1y", 366))
    return start_ts, end_ts


def _max_age(interval: str) -> pd.Timedelta:
    if interval.endswith("m") and not interval.endswith("mo"):
        return pd.Timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return pd.Timedelta(hours=int(interval[:-1]))
    return pd.Timedelta(days={"1d": 1, "1wk": 7, "1mo": 31}.get(interval, 1))


def get_prices(symbol: str, interval: str = "1d", start=None, end=None, period: str | None = "1y",
               refresh: bool = False, ig_env: str | None = None) -> pd.DataFrame:
    provider, sym = route(symbol)
    start_ts, end_ts = _window(start, end, period)

    if provider == "ig":
        from .config import ig_config
        from .providers.ig import IG
        with IG(ig_config(ig_env)) as ig:
            df = ig.prices(sym, interval, start_ts, end_ts, use_cache=not refresh)
        return _tag(df, f"IG {ig.cfg.acc_type} {sym} {interval}")

    chain = {"yahoo": ["yahoo", "stooq"], "stooq": ["stooq", "yahoo"], "crypto": ["crypto"],
             "binance": ["crypto"], "kraken": ["crypto"], "massive": ["massive"]}.get(provider)
    if chain is None:
        raise ValueError(f"unknown provider '{provider}'")
    if provider == "yahoo" and US_TICKER.match(sym):
        chain = ["yahoo", "massive", "stooq"]  # Massive backs Yahoo up on US stocks and ETFs

    errors = []
    for prov in chain:
        if not refresh and prov not in NO_CACHE:
            hit = cache.load(prov, sym, interval)
            if cache.covers(hit, start_ts, end_ts, _max_age(interval)):
                return _tag(_slice(hit, start_ts, end_ts), f"{prov} {sym} {interval} (cache)")
        try:
            df = _fetch(prov, sym, interval, start_ts, end_ts)
            if prov not in NO_CACHE:
                df = cache.save(prov, sym, interval, df)
            return _tag(_slice(df, start_ts, end_ts), f"{prov} {sym} {interval}")
        except Exception as e:
            errors.append(f"{prov}: {type(e).__name__}: {str(e)[:160]}")
    raise LookupError(f"no data for {symbol}. Tried -> " + " | ".join(errors))


def _fetch(prov, sym, interval, start_ts, end_ts) -> pd.DataFrame:
    if prov == "yahoo":
        from .providers import yahoo
        if interval in YF_INTRADAY_LIMIT:  # Yahoo caps intraday lookback
            start_ts = max(start_ts, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=YF_INTRADAY_LIMIT[interval]))
        return yahoo.history(sym, start=start_ts.date(), end=None if end_ts is None else (end_ts + pd.Timedelta(days=1)).date(),
                             interval=interval)
    if prov == "stooq":
        from .providers import stooq
        return stooq.history(sym, start_ts, end_ts, interval)
    if prov == "massive":
        from .providers import massive
        return massive.history(sym, start_ts, end_ts, interval)
    if prov == "crypto":
        from .providers import crypto
        return crypto.history(sym, start_ts, end_ts, interval)
    raise ValueError(prov)


def _slice(df, start_ts, end_ts):
    out = df[df.index >= start_ts]
    return out[out.index <= end_ts + pd.Timedelta(days=1)] if end_ts is not None else out


def _tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.attrs["source"] = source
    df.attrs["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    df.attrs["last_bar"] = str(df.index[-1]) if len(df) else None
    return df


def get_many(symbols: list[str], field: str = "close", **kw) -> pd.DataFrame:
    """Aligned panel of one field across symbols (for correlation, relative strength, pairs)."""
    out, errs = {}, {}
    for s in symbols:
        try:
            out[s] = get_prices(s, **kw)[field]
        except Exception as e:
            errs[s] = str(e)[:120]
    panel = pd.DataFrame(out)
    panel.attrs["errors"] = errs
    return panel
