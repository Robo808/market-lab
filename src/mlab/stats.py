"""Return/risk statistics for a price series or a strategy equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = {"1d": 252, "1wk": 52, "1mo": 12, "1h": 252 * 7, "4h": 252 * 2}


def returns(close: pd.Series, log: bool = False) -> pd.Series:
    return (np.log(close).diff() if log else close.pct_change()).dropna()


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(close: pd.Series, bench: pd.Series | None = None, periods: int = 252, rf: float = 0.0) -> dict:
    r = returns(close)
    if r.empty:
        return {}
    years = len(r) / periods
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    down = r[r < 0].std() * np.sqrt(periods)
    dd = drawdown(close)
    out = {
        "start": str(close.index[0].date()), "end": str(close.index[-1].date()), "total_return": close.iloc[-1] / close.iloc[0] - 1,
        "cagr": cagr, "vol": vol, "sharpe": (r.mean() * periods - rf) / vol if vol else np.nan,
        "sortino": (r.mean() * periods - rf) / down if down else np.nan, "max_drawdown": dd.min(),
        "max_dd_date": str(dd.idxmin().date()), "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "hit_rate": (r > 0).mean(), "skew": r.skew(), "kurt": r.kurt(),
        "var_95_1d": r.quantile(0.05), "cvar_95_1d": r[r <= r.quantile(0.05)].mean(),
    }
    if bench is not None:
        b = returns(bench).reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 20:
            beta = np.cov(rr, b)[0, 1] / b.var()
            out.update({"beta": beta, "corr": rr.corr(b),
                        "alpha_ann": (rr.mean() - beta * b.mean()) * periods})
    return out


def correlation(panel: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    r = panel.pct_change().dropna(how="all")
    return r.tail(window).corr() if window else r.corr()


def relative_strength(panel: pd.DataFrame, lookbacks=(21, 63, 126, 252)) -> pd.DataFrame:
    """Momentum table: % change over each lookback plus an average rank (higher = stronger)."""
    out = pd.DataFrame({f"r{n}": panel.iloc[-1] / panel.shift(n).iloc[-1] - 1 for n in lookbacks if len(panel) > n})
    out["rank_avg"] = out.rank(pct=True).mean(axis=1)
    return out.sort_values("rank_avg", ascending=False)


def zscore(s: pd.Series, n: int = 60) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def pair_spread(a: pd.Series, b: pd.Series, n: int = 60) -> pd.DataFrame:
    """Hedge ratio (OLS on logs), spread z-score and Engle-Granger cointegration p-value."""
    from statsmodels.tsa.stattools import coint

    df = pd.concat([np.log(a), np.log(b)], axis=1).dropna()
    df.columns = ["a", "b"]
    beta = np.polyfit(df["b"], df["a"], 1)[0]
    spread = df["a"] - beta * df["b"]
    out = pd.DataFrame({"spread": spread, "z": zscore(spread, n)})
    out.attrs.update({"hedge_ratio": beta, "coint_pvalue": coint(df["a"], df["b"])[1]})
    return out
