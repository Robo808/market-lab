"""Energy axis: is the economy spending energy to make money (demand-driven oil) or being drained by it (supply
shock)? Market-wide features joined to every instrument's cube rows.

Kilian (2009, AER) separates oil price moves by cause: demand-driven rises go with rising equities, supply shocks
hit them. Ready (2018, Review of Finance) does it daily with market data; here, in its simplest form, the
demand component of the day's oil return is the part explained by the equity market (rolling 252-day beta,
estimated on past days only) and the supply component is the residual. Hamilton (2009) and Ayres & Warr (2009)
motivate the energy-output link.

Point-in-time: every feature is lagged one session, because on a shared date calendar the oil settlement and the
US close come after the Asian and European closes of the same date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENERGY_FEATURES = ["oil_ret_21", "oil_ret_63", "oil_demand_63", "oil_supply_63", "oil_vs_equity_63"]


def features(oil: pd.Series, equity: pd.Series, beta_window: int = 252) -> pd.DataFrame:
    """oil, equity: daily closes on a session-date index. Returns lagged energy features on the same index."""
    d = pd.DataFrame({"oil": oil, "eq": equity}).dropna()
    d = d[(d["oil"] > 0) & (d["eq"] > 0)]
    ro, re = np.log(d["oil"]).diff(), np.log(d["eq"]).diff()
    cov = ro.rolling(beta_window).cov(re)
    beta = (cov / re.rolling(beta_window).var()).shift(1)       # estimated through yesterday
    demand = beta * re
    supply = ro - demand
    vol = ro.rolling(63).std()
    f = pd.DataFrame(index=d.index)
    f["oil_ret_21"] = np.log(d["oil"]).diff(21)
    f["oil_ret_63"] = np.log(d["oil"]).diff(63)
    f["oil_demand_63"] = demand.rolling(63).sum() / (vol * np.sqrt(63))
    f["oil_supply_63"] = supply.rolling(63).sum() / (vol * np.sqrt(63))
    f["oil_vs_equity_63"] = (np.log(d["oil"]) - np.log(d["eq"])).diff(63)
    return f.replace([np.inf, -np.inf], np.nan).shift(1)


def load(log=print) -> pd.DataFrame:
    """WTI spot (EIA, from 1986) and the S&P 500 (Yahoo) from the frozen store, train + validation only."""
    from .. import datasets
    oil = datasets.load("eia-oil-daily", ["WTI"], "train+validation")["WTI"]["value"]
    eq = datasets.load("xasset-daily", ["^GSPC"], "train+validation")["^GSPC"]["close"]
    oil.index, eq.index = oil.index.normalize(), eq.index.normalize()
    oil = oil[~oil.index.duplicated(keep="last")]
    eq = eq[~eq.index.duplicated(keep="last")]
    log(f"_energy axis: EIA WTI spot {oil.index[0].date()}..{oil.index[-1].date()}, "
        f"S&P 500 {eq.index[0].date()}..{eq.index[-1].date()} (frozen store)_")
    return features(oil, eq)


def attach(cube: pd.DataFrame, energy: pd.DataFrame) -> pd.DataFrame:
    """Energy columns aligned to the cube's (symbol, time) rows, carrying the last known value forward."""
    t = cube.index.get_level_values("time")
    e = energy.reindex(energy.index.union(t.unique())).ffill(limit=5).reindex(t)
    return pd.DataFrame(e.to_numpy(), index=cube.index, columns=energy.columns)


def shock_table(energy: pd.DataFrame, equity: pd.Series, h: int = 21) -> pd.DataFrame:
    """Next-h-session equity return by demand/supply quadrant (descriptive; quadrant signs only, no fitting)."""
    lc = np.log(equity)
    fwd = (lc.shift(-h) - lc).reindex(energy.index)
    d = pd.DataFrame({"dem": energy["oil_demand_63"], "sup": energy["oil_supply_63"], "fwd": fwd}).dropna()
    d["quadrant"] = np.select(
        [(d.dem > 0) & (d.sup <= 0), (d.dem > 0) & (d.sup > 0), (d.dem <= 0) & (d.sup > 0)],
        ["demand up, supply calm", "demand up + supply shock", "supply shock, demand down"], "demand down, supply calm")
    g = d.groupby("quadrant")["fwd"]
    return pd.DataFrame({"days": g.size(), "share": g.size() / len(d), f"mean_fwd_{h}d": g.mean(),
                         "hit_up": g.apply(lambda x: (x > 0).mean())})
