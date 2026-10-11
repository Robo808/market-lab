"""Event studies on daily bars: abnormal returns, extreme-move days, forward CARs from the next open, and
group comparisons with a permutation null. The building block for every news-before-price test (#55).

Timing rules (no look-ahead):
- abnormal return on day t: AR_t = r_t - beta_{t-1} * r_mkt,t, beta and sigma estimated on bars up to t-1;
- an event on day t is traded at the open of t+1 and held to the close of t+h;
- forward CAR(+1..+h) = stock open(t+1)->close(t+h) minus beta * market over the same window; event CARs are
  signed by the day-t move (positive = continuation) and net of a round-trip cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _adj(df: pd.DataFrame) -> pd.DataFrame:
    """Open and close on the adjusted scale (adj_close / close applied to the open)."""
    f = (df["adj_close"] / df["close"]) if "adj_close" in df else 1.0
    out = pd.DataFrame({"open": df["open"] * f, "close": df["close"] * f}, index=df.index)
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC").normalize() if out.index.tz is not None \
        else pd.DatetimeIndex(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].dropna()


def abnormal(px: pd.DataFrame, mkt: pd.DataFrame, beta_window: int = 250, sigma_window: int = 60) -> pd.DataFrame:
    """Daily AR with beta and sigma from data up to t-1. Returns frame with r, rm, beta, ar, sigma, z."""
    a, m = _adj(px), _adj(mkt)
    j = a.join(m, rsuffix="_m", how="inner")
    r, rm = j["close"].pct_change(), j["close_m"].pct_change()
    minp = beta_window // 2
    beta = (r.rolling(beta_window, min_periods=minp).cov(rm) / rm.rolling(beta_window, min_periods=minp).var()).shift(1)
    ar = r - beta * rm
    sigma = ar.rolling(sigma_window, min_periods=sigma_window // 2).std().shift(1)
    return pd.DataFrame({"open": j["open"], "close": j["close"], "open_m": j["open_m"], "close_m": j["close_m"],
                         "r": r, "rm": rm, "beta": beta, "ar": ar, "sigma": sigma, "z": ar / sigma})


def forward_car(ab: pd.DataFrame, h: int) -> pd.Series:
    """Gross CAR from the open of t+1 to the close of t+h, aligned to day t (NaN where the window runs off)."""
    o1, oc = ab["open"].shift(-1), ab["close"].shift(-h)
    m1, mc = ab["open_m"].shift(-1), ab["close_m"].shift(-h)
    return (oc / o1 - 1) - ab["beta"] * (mc / m1 - 1)


def events(frames: dict[str, pd.DataFrame], mkt: pd.DataFrame, flags: dict[str, set] | None = None,
           k: float = 2.5, horizons=(1, 5, 10), cost_bp: float = 0.0) -> pd.DataFrame:
    """One row per extreme day (|z| > k): symbol, date, z, direction, flag (news or not), and forward CARs signed
    by the day-t move (positive = continuation) net of a round-trip cost."""
    rows = []
    for sym, px in frames.items():
        ab = abnormal(px, mkt)
        cars = {h: forward_car(ab, h) for h in horizons}
        hit = ab.index[(ab["z"].abs() > k)]
        f = (flags or {}).get(sym, set())
        for t in hit:
            d = float(np.sign(ab.at[t, "ar"]))
            row = {"symbol": sym, "date": t, "z": float(ab.at[t, "z"]), "dir": d, "flag": t in f}
            for h in horizons:
                row[f"car{h}"] = d * float(cars[h].get(t, np.nan)) - cost_bp / 1e4  # signed, net of costs
            rows.append(row)
    ev = pd.DataFrame(rows)
    return ev.sort_values(["date", "symbol"]).reset_index(drop=True) if len(ev) else ev


def _by_date_mean(ev: pd.DataFrame, col: str, mask: pd.Series) -> pd.Series:
    """Average same-date events first so one market-wide day doesn't count as many independent observations."""
    return ev[mask].groupby("date")[col].mean().dropna()


def compare(ev: pd.DataFrame, col: str = "car5", n_perm: int = 2000, seed: int = 7) -> dict:
    """Mean signed CAR for flagged vs unflagged events, difference, Welch t on date-averaged events, and a
    permutation p-value (two-sided) shuffling the flag within symbol-year."""
    ev = ev.dropna(subset=[col])
    a, b = _by_date_mean(ev, col, ev["flag"]), _by_date_mean(ev, col, ~ev["flag"])
    if len(a) < 5 or len(b) < 5:
        return {"n_flag": int(ev["flag"].sum()), "n_noflag": int((~ev["flag"]).sum()), "note": "too few events"}
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    rng = np.random.default_rng(seed)
    grp = ev["symbol"] + ev["date"].dt.year.astype(str)
    flags, vals = ev["flag"].to_numpy(), ev[col].to_numpy()
    idx = [np.flatnonzero(grp.to_numpy() == g) for g in grp.unique()]
    null = np.empty(n_perm)
    for i in range(n_perm):
        f = flags.copy()
        for ix in idx:
            f[ix] = rng.permutation(f[ix])
        null[i] = vals[f].mean() - vals[~f].mean() if f.any() and (~f).any() else 0.0
    raw = vals[flags].mean() - vals[~flags].mean()
    p = (np.sum(np.abs(null) >= abs(raw)) + 1) / (n_perm + 1)
    return {"n_flag": int(flags.sum()), "n_noflag": int((~flags).sum()), "dates_flag": len(a), "dates_noflag": len(b),
            "mean_flag_bp": round(a.mean() * 1e4, 1), "mean_noflag_bp": round(b.mean() * 1e4, 1),
            "diff_bp": round(diff * 1e4, 1), "t": round(float(diff / se), 2) if se > 0 else None,
            "t_flag": round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 2),
            "t_noflag": round(float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))), 2),
            "perm_p": round(float(p), 4)}


def trading_day_flags(dates: pd.Series, index: pd.DatetimeIndex, spill: int = 1) -> set:
    """Map calendar dates (filing dates) to the trading days whose move they may explain: the trading day on or
    after the date, plus `spill` following days (an after-close filing explains the next session)."""
    index = pd.DatetimeIndex(index).sort_values()
    out = set()
    for d in pd.to_datetime(dates):
        d = pd.Timestamp(d)
        d = d.tz_localize("UTC") if d.tz is None else d.tz_convert("UTC")
        i = index.searchsorted(d.normalize())
        out.update(index[i:i + 1 + spill])
    return out
