"""Yahoo Finance via yfinance: equities, ETFs, indices (^GSPC), FX (EURUSD=X), futures (CL=F), crypto (BTC-USD)."""
from __future__ import annotations

import pandas as pd

from . import OHLCV


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index.name = "time"
    cols = [c for c in OHLCV + ["adj_close"] if c in df.columns]
    return df[cols].dropna(subset=["close"])


def history(symbol: str, start=None, end=None, interval: str = "1d", period: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    kw = {"interval": interval, "auto_adjust": False, "actions": False}
    if period and not start:
        kw["period"] = period
    else:
        kw["start"], kw["end"] = start, end
    df = yf.Ticker(symbol).history(**kw)
    if df is None or df.empty:
        raise LookupError(f"yahoo returned no data for {symbol}")
    return _norm(df)


def info(symbol: str) -> dict:
    import yfinance as yf

    return yf.Ticker(symbol).get_info()


def fundamentals(symbol: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    return {
        "income": t.income_stmt, "income_q": t.quarterly_income_stmt,
        "balance": t.balance_sheet, "cashflow": t.cashflow,
    }


def options_chain(symbol: str, expiry: str | None = None):
    import yfinance as yf

    t = yf.Ticker(symbol)
    exps = t.options
    if not exps:
        raise LookupError(f"no listed options for {symbol}")
    ch = t.option_chain(expiry or exps[0])
    return {"expiries": list(exps), "calls": ch.calls, "puts": ch.puts}


def holders(symbol: str) -> dict:
    import yfinance as yf

    t = yf.Ticker(symbol)
    return {"institutional": t.institutional_holders, "insider_tx": t.insider_transactions,
            "major": t.major_holders}


def news(symbol: str) -> list[dict]:
    import yfinance as yf

    return yf.Ticker(symbol).news or []
