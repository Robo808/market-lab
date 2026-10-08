"""Vectorized single-instrument backtester with IG-style costs (spread in points + overnight funding).

Strategies return a position series in {-1, 0, 1} (or fractional), decided on bar t and applied to
the return of bar t+1, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ta
from .stats import summary


def sma_cross(df, fast=50, slow=200, short=False):
    f, s = ta.sma(df["close"], fast), ta.sma(df["close"], slow)
    return pd.Series(np.where(f > s, 1, -1 if short else 0), index=df.index).where(s.notna(), 0)


def donchian_breakout(df, entry=55, exit=20, short=True):
    """Turtle-style: enter on N-bar breakout, exit on M-bar opposite break."""
    hi_e, lo_e = df["high"].rolling(entry).max().shift(), df["low"].rolling(entry).min().shift()
    hi_x, lo_x = df["high"].rolling(exit).max().shift(), df["low"].rolling(exit).min().shift()
    pos, cur = [], 0
    for c, he, le, hx, lx in zip(df["close"], hi_e, lo_e, hi_x, lo_x):
        if cur == 0:
            cur = 1 if c > he else (-1 if short and c < le else 0)
        elif cur == 1 and c < lx:
            cur = 0
        elif cur == -1 and c > hx:
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=df.index)


def rsi_reversion(df, low=30, high=70, n=14, trend_filter=200):
    r = ta.rsi(df["close"], n)
    up = df["close"] > ta.sma(df["close"], trend_filter) if trend_filter else True
    pos, cur = [], 0
    for rv, u in zip(r, up if not isinstance(up, bool) else [True] * len(r)):
        if cur == 0 and rv < low and u:
            cur = 1
        elif cur == 1 and rv > 50:
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=df.index)


def supertrend_follow(df, n=10, mult=3.0, short=False):
    t = ta.supertrend(df, n, mult)["trend"]
    return t.clip(lower=-1 if short else 0).fillna(0)


STRATEGIES = {"sma_cross": sma_cross, "donchian": donchian_breakout, "rsi_reversion": rsi_reversion,
              "supertrend": supertrend_follow}


def run(df: pd.DataFrame, position: pd.Series, spread: float = 0.0, funding_annual: float = 0.0,
        periods: int = 252) -> dict:
    """spread: round-trip cost in price units per unit position change (IG spread);
    funding_annual: overnight financing rate applied to |position| (e.g. 0.06 for ~SONIA+2.5%)."""
    c = df["close"]
    pos = position.reindex(c.index).fillna(0).shift().fillna(0)  # trade next bar
    gross = pos * c.pct_change().fillna(0)
    turns = pos.diff().abs().fillna(pos.abs())
    cost = turns * (spread / 2) / c  # half-spread per side
    fund = pos.abs() * funding_annual / periods
    net = gross - cost - fund
    equity = (1 + net).cumprod()
    trades = int((turns > 0).sum())
    bh = c / c.iloc[0]
    res = {"strategy": summary(equity, periods=periods), "buy_hold": summary(bh, periods=periods),
           "trades": trades, "exposure": float((pos != 0).mean()), "cost_drag_total": float((cost + fund).sum())}
    res["equity"] = pd.DataFrame({"strategy": equity, "buy_hold": bh})
    return res


def walk_forward(df, strat, grid: list[dict], train=504, test=126, **run_kw) -> pd.DataFrame:
    """Pick the best params on each train window (by Sharpe), evaluate out-of-sample on the next test window."""
    rows = []
    for start in range(0, len(df) - train - test + 1, test):
        tr, te = df.iloc[start:start + train], df.iloc[start + train - 250:start + train + test]
        best = max(grid, key=lambda p: np.nan_to_num(run(tr, strat(tr, **p), **run_kw)["strategy"].get("sharpe", np.nan), nan=-9))
        oos = run(te, strat(te, **best), **run_kw)
        eq = oos["equity"]["strategy"].loc[df.index[start + train]:]
        rows.append({"test_start": df.index[start + train].date(), "params": best,
                     "oos_return": eq.iloc[-1] / eq.iloc[0] - 1})
    return pd.DataFrame(rows)
