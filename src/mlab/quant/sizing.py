"""Position sizing: volatility targeting, ATR risk sizing, inverse-vol and fractional Kelly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ta
from ..risk import kelly_fraction  # noqa: F401  (re-exported)


def vol_scalar(close: pd.Series, target: float = 0.15, n: int = 20, cap: float = 2.0, periods: int = 252) -> float:
    rv = close.pct_change().tail(n).std() * np.sqrt(periods)
    return float(min(target / rv, cap)) if rv and not np.isnan(rv) else 0.0


def atr_units(equity: float, risk_pct: float, df: pd.DataFrame, atr_mult: float = 2.0, n: int = 14,
              point_size: float = 1.0) -> dict:
    """Stake per point so that an `atr_mult` x ATR stop loses `risk_pct` of equity."""
    a = float(ta.atr(df, n).iloc[-1])
    stop_pts = atr_mult * a / point_size
    risk = equity * risk_pct / 100
    return {"atr": a, "stop_distance_pts": stop_pts, "risk_cash": risk, "stake_per_point": risk / stop_pts if stop_pts else 0.0}


def inverse_vol_weights(panel: pd.DataFrame, n: int = 60) -> pd.Series:
    vol = panel.pct_change().tail(n).std()
    w = 1 / vol
    return w / w.sum()


def expectancy(r_multiples: list[float]) -> dict:
    r = np.asarray([x for x in r_multiples if x is not None], float)
    if not len(r):
        return {"n": 0}
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / len(r)
    payoff = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() else np.nan
    return {"n": len(r), "win_rate": win_rate, "avg_win_R": wins.mean() if len(wins) else 0.0,
            "avg_loss_R": losses.mean() if len(losses) else 0.0, "expectancy_R": r.mean(), "payoff": payoff,
            "quarter_kelly": kelly_fraction(win_rate, payoff) if payoff == payoff else 0.0}
