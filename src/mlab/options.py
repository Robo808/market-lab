"""Options desk: Black-Scholes-Merton pricing/greeks/IV, Yahoo chain normalisation, and pure chain
analytics (term structure, skew, expected move, max pain, PCR, dealer GEX, unusual activity, IV rank,
IV vs realized vol) plus an instrument-agnostic strategy payoff helper.

Units: greeks() returns raw derivatives (vega/vanna per 1.00 vol, theta/charm per year, rho per 1.00
rate). Tables scale vega to per vol point and theta to per calendar day. Times are in years (ACT/365).
Network access lives only in fetch_chain / fetch_spot / risk_free_rate / desk.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .config import DATA_DIR, env

HOSTS = {  # all already in net.SOURCES; listed so `mlab doctor` users see what this desk needs
    "query1.finance.yahoo.com": "Yahoo spot/history (options desk RV)",
    "query2.finance.yahoo.com": "Yahoo option chains (yfinance)",
}
IV_DIR = Path(env("MLAB_IV_DIR") or DATA_DIR / "data" / "iv_history")  # outside the LRU price cache: never pruned
YEAR = 365.0
GEX_CONVENTIONS = {  # (call sign, put sign) of dealer gamma per contract of open interest
    "long_calls_short_puts": (1, -1),  # customers sell calls (overwriting) and buy puts (hedging): SqueezeMetrics-style
    "short_calls_long_puts": (-1, 1),
    "short_all": (-1, -1),  # customers are net buyers of everything
    "long_all": (1, 1),
}


# ---- Black-Scholes-Merton ---------------------------------------------------------
def _is_call(kind) -> np.ndarray:
    a = np.asarray(kind)
    if a.dtype == bool:
        return a
    return np.char.startswith(np.char.upper(a.astype(str)), "C")


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _setup(S, K, T, r, sigma, q, kind):
    c = _is_call(kind)
    S, K, T, r, sigma, q, c = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q)), c)
    Tp = np.maximum(T, 0.0)
    vt = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), np.nan) * np.sqrt(Tp)
    ok = vt > 0
    vts = np.where(ok, vt, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * Tp) / vts
    return S, K, Tp, r, sigma, q, c.astype(bool), ok, vts, d1, d1 - vts


def bs_price(S, K, T, r, sigma, q=0.0, kind="C"):
    """BSM value of a European call/put (kind 'C'/'P', arrays broadcast). T<=0 or vol<=0 -> discounted intrinsic."""
    S, K, T, r, sigma, q, c, ok, _, d1, d2 = _setup(S, K, T, r, sigma, q, kind)
    dr, dq = np.exp(-r * T), np.exp(-q * T)
    with np.errstate(invalid="ignore"):
        call = S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
        put = K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
    fwd_c, fwd_p = np.maximum(S * dq - K * dr, 0.0), np.maximum(K * dr - S * dq, 0.0)
    return _out(np.where(ok, np.where(c, call, put), np.where(c, fwd_c, fwd_p)))


def greeks(S, K, T, r, sigma, q=0.0, kind="C") -> dict:
    """delta, gamma, vega, theta (-dV/dT, per year), rho, vanna (dDelta/dVol), charm (-dDelta/dT, per year)."""
    S, K, T, r, sigma, q, c, ok, vts, d1, d2 = _setup(S, K, T, r, sigma, q, kind)
    dr, dq = np.exp(-r * T), np.exp(-q * T)
    sqT = np.sqrt(np.where(T > 0, T, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = norm.pdf(d1)
        Nd1, Nd2, Nmd1, Nmd2 = norm.cdf(d1), norm.cdf(d2), norm.cdf(-d1), norm.cdf(-d2)
        delta = np.where(c, dq * Nd1, -dq * Nmd1)
        gamma = dq * pdf / (S * vts)
        vega = S * dq * pdf * sqT
        common = -S * dq * pdf * sigma / (2 * sqT)
        theta = np.where(c, common - r * K * dr * Nd2 + q * S * dq * Nd1, common + r * K * dr * Nmd2 - q * S * dq * Nmd1)
        rho = np.where(c, K * T * dr * Nd2, -K * T * dr * Nmd2)
        vanna = -dq * pdf * d2 / sigma
        core = dq * pdf * (2 * (r - q) * T - d2 * vts) / (2 * T * vts)
        charm = np.where(c, q * dq * Nd1 - core, -q * dq * Nmd1 - core)
    expired = np.where(c, (S > K).astype(float), -(S < K).astype(float))
    out = {"delta": np.where(ok, delta, expired)}
    for k, v in (("gamma", gamma), ("vega", vega), ("theta", theta), ("rho", rho), ("vanna", vanna), ("charm", charm)):
        out[k] = np.where(ok, v, 0.0)
    return {k: _out(v) for k, v in out.items()}


def implied_vol(price, S, K, T, r=0.0, q=0.0, kind="C", lo: float = 1e-4, hi: float = 5.0) -> float:
    """Brent solve for BSM vol. NaN for bad quotes: non-finite, outside no-arbitrage bounds, no time value."""
    try:
        price, S, K, T, r, q = (float(x) for x in (price, S, K, T, r, q))
    except (TypeError, ValueError):
        return math.nan
    if not all(map(math.isfinite, (price, S, K, T, r, q))) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return math.nan
    c = bool(_is_call(kind))
    fs, fk = S * math.exp(-q * T), K * math.exp(-r * T)
    lower, upper = (max(fs - fk, 0.0), fs) if c else (max(fk - fs, 0.0), fk)
    if price <= lower + 1e-10 * max(S, 1) or price >= upper:
        return math.nan

    def f(s):
        return bs_price(S, K, T, r, s, q, "C" if c else "P") - price

    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        return math.nan
    try:
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return math.nan


def implied_vols(price, S, K, T, r=0.0, q=0.0, kind="C") -> np.ndarray:
    """Vectorised implied_vol over broadcast inputs."""
    arrs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, S, K, T, r, q)), _is_call(kind))
    flat = [a.ravel() for a in arrs]
    out = [implied_vol(p, s, k, t, rr, qq, "C" if cc else "P") for p, s, k, t, rr, qq, cc in zip(*flat)]
    return np.asarray(out, dtype=float).reshape(arrs[0].shape)


# ---- chain fetch & normalisation ---------------------------------------------------
def fetch_spot(ticker: str) -> float:
    import yfinance as yf

    t = yf.Ticker(ticker)
    try:
        px = float(t.fast_info["last_price"])
        if px > 0:
            return px
    except Exception:
        pass
    h = t.history(period="5d")
    if h is None or h.empty:
        raise LookupError(f"no spot for {ticker}")
    return float(h["Close"].iloc[-1])


def fetch_chain(ticker: str, expiries: int | None = 6) -> tuple[float, pd.DataFrame]:
    """(spot, raw yfinance chain rows with 'expiry' and 'type' added). expiries=None/0 -> all listed."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = list(t.options or [])
    if not exps:
        raise LookupError(f"no listed options for {ticker}")
    frames = []
    for e in exps[:expiries] if expiries else exps:
        ch = t.option_chain(e)
        for typ, df in (("C", ch.calls), ("P", ch.puts)):
            if df is not None and len(df):
                frames.append(df.assign(expiry=e, type=typ))
    if not frames:
        raise LookupError(f"empty option chain for {ticker}")
    return fetch_spot(ticker), pd.concat(frames, ignore_index=True)


def risk_free_rate(default: float = 0.04) -> float:
    """13-week T-bill yield (^IRX, percent) as a decimal; default when unreachable."""
    try:
        from .data import get_prices
        v = float(get_prices("^IRX", period="1mo")["close"].dropna().iloc[-1]) / 100
        return v if 0 <= v < 0.25 else default
    except Exception:
        return default


def normalize_chain(raw: pd.DataFrame, spot: float, r: float = 0.04, q: float = 0.0, now=None) -> pd.DataFrame:
    """One tidy frame: expiry, dte, T, type, strike, bid, ask, mid, last, volume, oi, iv, iv_yahoo, iv_src,
    moneyness (K/S), delta, gamma. IV is solved from mid; Yahoo's iv is used only when that fails and it is sane."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    col = lambda *names: next((raw[n] for n in names if n in raw), pd.Series(np.nan, index=raw.index))  # noqa: E731
    num = lambda s: pd.to_numeric(s, errors="coerce").astype(float)  # noqa: E731
    exp = pd.to_datetime(raw["expiry"])
    exp = (exp.dt.tz_localize(None) if exp.dt.tz is not None else exp).dt.normalize()
    df = pd.DataFrame({
        "expiry": exp, "type": raw["type"].astype(str).str.upper().str[0], "strike": num(col("strike")),
        "bid": num(col("bid")), "ask": num(col("ask")), "last": num(col("lastPrice", "last")),
        "volume": num(col("volume")).fillna(0.0), "oi": num(col("openInterest", "oi")).fillna(0.0),
        "iv_yahoo": num(col("impliedVolatility", "iv")),
    })
    close_utc = (df["expiry"] + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    df["T"] = ((close_utc - now).dt.total_seconds() / (YEAR * 86400)).clip(lower=1 / (YEAR * 24))
    df["dte"] = (df["expiry"] - now.tz_localize(None).normalize()).dt.days
    two_sided = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = np.where(two_sided, (df["bid"] + df["ask"]) / 2, np.where(df["last"] > 0, df["last"], np.nan))
    iv_mid = implied_vols(df["mid"], spot, df["strike"], df["T"], r, q, df["type"].to_numpy())
    sane = df["iv_yahoo"].between(0.01, 5.0)
    df["iv"] = np.where(np.isfinite(iv_mid), iv_mid, np.where(sane, df["iv_yahoo"], np.nan))
    df["iv_src"] = np.where(np.isfinite(iv_mid), "mid", np.where(sane, "yahoo", "none"))
    df["moneyness"] = df["strike"] / spot
    g = greeks(spot, df["strike"], df["T"], r, df["iv"], q, df["type"].to_numpy())
    df["delta"] = np.where(df["iv"].notna(), g["delta"], np.nan)
    df["gamma"] = np.where(df["iv"].notna(), g["gamma"], np.nan)
    df = df[df["strike"] > 0].sort_values(["expiry", "type", "strike"]).reset_index(drop=True)
    df.attrs.update(spot=float(spot), r=float(r), q=float(q), asof=now.isoformat(timespec="seconds"))
    return df


# ---- chain analytics (pure) --------------------------------------------------------
def _spot(chain, spot=None) -> float:
    return float(spot if spot is not None else chain.attrs["spot"])


def _iv_at(g: pd.DataFrame, x: float, by: str = "strike") -> float:
    h = g[g["iv"].notna() & g[by].notna()].sort_values(by)
    if h.empty:
        return math.nan
    xs = h[by].to_numpy()
    if xs[0] <= x <= xs[-1]:
        return float(np.interp(x, xs, h["iv"].to_numpy()))
    return math.nan if by == "delta" else float(h["iv"].iloc[np.abs(xs - x).argmin()])


def atm_term_structure(chain: pd.DataFrame, spot=None) -> pd.DataFrame:
    """Per expiry: ATM IV (call/put IV interpolated at spot, averaged) and forward vol between expiries."""
    S = _spot(chain, spot)
    rows = []
    for e, g in chain.groupby("expiry"):
        ivs = [v for t in "CP" if np.isfinite(v := _iv_at(g[g["type"] == t], S))]
        rows.append({"expiry": e, "dte": int(g["dte"].iloc[0]), "T": float(g["T"].mean()),
                     "atm_iv": float(np.mean(ivs)) if ivs else math.nan})
    df = pd.DataFrame(rows).set_index("expiry")
    if df.empty:
        return df
    w = df["atm_iv"] ** 2 * df["T"]
    dw, dT = w.diff(), df["T"].diff()
    df["fwd_vol"] = np.sqrt((dw / dT).where(dw > 0))
    return df


def atm_iv_at(term: pd.DataFrame, days: float = 30) -> float:
    """Constant-maturity ATM IV by linear interpolation in total variance (flat extrapolation)."""
    t = term.dropna(subset=["atm_iv"])
    if t.empty:
        return math.nan
    T, x = t["T"].to_numpy(), days / YEAR
    if x <= T[0]:
        return float(t["atm_iv"].iloc[0])
    if x >= T[-1]:
        return float(t["atm_iv"].iloc[-1])
    return float(np.sqrt(np.interp(x, T, (t["atm_iv"] ** 2 * t["T"]).to_numpy()) / x))


def skew(chain: pd.DataFrame, spot=None, wing: float = 0.25) -> pd.DataFrame:
    """Per expiry: 25-delta call/put IV, risk reversal (call - put) and butterfly ((call+put)/2 - ATM)."""
    term = atm_term_structure(chain, spot)
    rows = []
    for e, g in chain.groupby("expiry"):
        c, p = _iv_at(g[g["type"] == "C"], wing, "delta"), _iv_at(g[g["type"] == "P"], -wing, "delta")
        atm = term.loc[e, "atm_iv"]
        rows.append({"expiry": e, "dte": int(g["dte"].iloc[0]), "atm_iv": atm, "iv_c25": c, "iv_p25": p,
                     "rr25": c - p, "bf25": (c + p) / 2 - atm})
    return pd.DataFrame(rows).set_index("expiry")


def expected_move(chain: pd.DataFrame, spot=None) -> pd.DataFrame:
    """Per expiry: ATM straddle (strike nearest spot with both mids) as the market's expected move,
    alongside the 1-sd move from ATM IV (S*iv*sqrt(T)); the straddle is ~0.8 of one sd."""
    S = _spot(chain, spot)
    term = atm_term_structure(chain, S)
    rows = []
    for e, g in chain.groupby("expiry"):
        piv = g.pivot_table(index="strike", columns="type", values="mid", aggfunc="first")
        if not {"C", "P"} <= set(piv.columns):
            continue
        piv = piv.dropna(subset=["C", "P"])
        if piv.empty:
            continue
        k = float(piv.index[np.abs(piv.index.to_numpy() - S).argmin()])
        st = float(piv.loc[k, "C"] + piv.loc[k, "P"])
        iv, T = term.loc[e, "atm_iv"], term.loc[e, "T"]
        rows.append({"expiry": e, "dte": int(g["dte"].iloc[0]), "atm_strike": k, "straddle": st, "move_pct": st / S,
                     "lower": S - st, "upper": S + st, "iv_1sd": S * iv * math.sqrt(T), "iv_1sd_pct": iv * math.sqrt(T)})
    return pd.DataFrame(rows).set_index("expiry") if rows else pd.DataFrame()


def max_pain(chain: pd.DataFrame, spot=None) -> pd.DataFrame:
    """Per expiry: settlement strike minimising total intrinsic value paid to option holders (OI-weighted)."""
    S = _spot(chain, spot)
    rows = []
    for e, g in chain.groupby("expiry"):
        ks = np.sort(g["strike"].unique())
        c, p = g[g["type"] == "C"], g[g["type"] == "P"]
        P = ks[:, None]
        pay = (np.maximum(P - c["strike"].to_numpy()[None, :], 0) * c["oi"].to_numpy()[None, :]).sum(1) + \
              (np.maximum(p["strike"].to_numpy()[None, :] - P, 0) * p["oi"].to_numpy()[None, :]).sum(1)
        mp = float(ks[pay.argmin()])
        rows.append({"expiry": e, "dte": int(g["dte"].iloc[0]), "max_pain": mp, "vs_spot": mp / S - 1,
                     "call_oi": float(c["oi"].sum()), "put_oi": float(p["oi"].sum())})
    return pd.DataFrame(rows).set_index("expiry")


def put_call_ratios(chain: pd.DataFrame) -> pd.DataFrame:
    """Put/call ratios by volume and open interest, per expiry and in total ('ALL')."""
    agg = chain.pivot_table(index="expiry", columns="type", values=["volume", "oi"], aggfunc="sum").fillna(0)
    out = pd.DataFrame({"call_vol": agg.get(("volume", "C"), 0), "put_vol": agg.get(("volume", "P"), 0),
                        "call_oi": agg.get(("oi", "C"), 0), "put_oi": agg.get(("oi", "P"), 0)}).astype(float)
    out.index = out.index.astype(str).str[:10]
    out.loc["ALL"] = out.sum()
    out["pcr_vol"] = out["put_vol"] / out["call_vol"].replace(0, np.nan)
    out["pcr_oi"] = out["put_oi"] / out["call_oi"].replace(0, np.nan)
    return out


def _gex_signs(chain, convention):
    cs, ps = GEX_CONVENTIONS[convention] if isinstance(convention, str) else convention
    return np.where(chain["type"].to_numpy() == "C", cs, ps)


def gex_profile(chain: pd.DataFrame, levels, convention="long_calls_short_puts", multiplier: float = 100) -> pd.Series:
    """Total dealer gamma exposure ($ per 1% move) if spot were at each level (IV, T held fixed)."""
    c = chain[chain["iv"].notna() & (chain["oi"] > 0)]
    sign, K, T, iv, oi = _gex_signs(c, convention), c["strike"].to_numpy(), c["T"].to_numpy(), c["iv"].to_numpy(), c["oi"].to_numpy()
    r, q = chain.attrs.get("r", 0.0), chain.attrs.get("q", 0.0)
    L = np.asarray(levels, dtype=float)
    gam = greeks(L[:, None], K[None, :], T[None, :], r, iv[None, :], q, "C")["gamma"]  # gamma is the same for C and P
    tot = (np.atleast_2d(gam) * (sign * oi)[None, :]).sum(1) * multiplier * L ** 2 * 0.01
    return pd.Series(tot, index=pd.Index(L, name="level"), name="gex")


def zero_gamma_level(profile: pd.Series, spot: float) -> float:
    """Level where the GEX profile crosses zero (linear interpolation), nearest to spot; NaN if none."""
    x, y = profile.index.to_numpy(dtype=float), profile.to_numpy(dtype=float)
    roots = [x[i] for i in range(len(y)) if y[i] == 0]
    for i in np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]:
        roots.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float(min(roots, key=lambda v: abs(v - spot))) if roots else math.nan


def gex(chain: pd.DataFrame, spot=None, convention="long_calls_short_puts", multiplier: float = 100,
        grid=(0.7, 1.3, 241)) -> dict:
    """Dealer gamma exposure, $ per 1% spot move: gamma * OI * multiplier * S^2 * 1%, signed by `convention`
    (default: dealers long calls / short puts). Returns by_strike, total, flip (zero-gamma level), profile."""
    S = _spot(chain, spot)
    c = chain[chain["gamma"].notna()].copy()
    c["gex"] = _gex_signs(c, convention) * c["gamma"] * c["oi"] * multiplier * S ** 2 * 0.01
    by = c.pivot_table(index="strike", columns="type", values="gex", aggfunc="sum").fillna(0.0)
    by = by.reindex(columns=["C", "P"], fill_value=0.0).rename(columns={"C": "call_gex", "P": "put_gex"})
    by["net_gex"] = by["call_gex"] + by["put_gex"]
    prof = gex_profile(chain, np.linspace(grid[0] * S, grid[1] * S, int(grid[2])), convention, multiplier)
    return {"convention": convention if isinstance(convention, str) else str(convention), "total": float(by["net_gex"].sum()),
            "flip": zero_gamma_level(prof, S), "by_strike": by, "profile": prof,
            "max_pos_strike": float(by["net_gex"].idxmax()) if len(by) else math.nan,
            "max_neg_strike": float(by["net_gex"].idxmin()) if len(by) else math.nan}


def unusual_activity(chain: pd.DataFrame, min_volume: float = 100, vol_oi: float = 1.5, vs_median: float = 5.0,
                     premium: float = 500_000, multiplier: float = 100, top: int = 15) -> pd.DataFrame:
    """Contracts with volume > open interest, volume far above the expiry/type median, or large premium traded."""
    d = chain.copy()
    d["premium"] = d["volume"] * d["mid"].fillna(d["last"]) * multiplier
    d["vol_oi"] = d["volume"] / d["oi"].where(d["oi"] > 0)
    med = d.groupby(["expiry", "type"])["volume"].transform(lambda s: s[s > 0].median())
    d["vol_vs_med"] = d["volume"] / med
    f_oi = (d["vol_oi"] >= vol_oi) | ((d["oi"] == 0) & (d["volume"] >= min_volume))
    f_med, f_prem = d["vol_vs_med"] >= vs_median, d["premium"] >= premium
    d["flags"] = [", ".join(n for n, b in (("vol>oi", a), ("vs median", b_), ("premium", c_)) if b)
                  for a, b_, c_ in zip(f_oi, f_med, f_prem)]
    d = d[(d["volume"] >= min_volume) & (f_oi | f_med | f_prem)]
    cols = ["expiry", "type", "strike", "volume", "oi", "vol_oi", "vol_vs_med", "mid", "premium", "iv", "delta", "flags"]
    return d.sort_values("premium", ascending=False)[cols].head(top).reset_index(drop=True)


# ---- IV history, rank, realized vol --------------------------------------------------
def _naive_day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return (ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts).normalize()


def _iv_path(ticker: str) -> Path:
    return IV_DIR / (re.sub(r"[^A-Za-z0-9._=-]+", "_", ticker.upper()) + ".csv")


def record_atm_iv(ticker: str, atm_iv30: float, spot: float, asof=None) -> pd.DataFrame:
    """Upsert today's 30d ATM IV snapshot into data/iv_history/<TICKER>.csv; returns the full history."""
    day = _naive_day(asof if asof is not None else pd.Timestamp.now(tz="UTC"))
    hist = load_iv_history(ticker)
    if np.isfinite(atm_iv30):
        hist.loc[day, ["atm_iv30", "spot"]] = [float(atm_iv30), float(spot)]
        hist = hist.sort_index()
        p = _iv_path(ticker)
        p.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(p, index_label="date")
    return hist


def load_iv_history(ticker: str) -> pd.DataFrame:
    p = _iv_path(ticker)
    if not p.exists():
        return pd.DataFrame(columns=["atm_iv30", "spot"], index=pd.DatetimeIndex([], name="date"), dtype=float)
    return pd.read_csv(p, index_col="date", parse_dates=["date"]).astype(float)


def iv_rank(history: pd.Series, current: float | None = None, lookback: int = 252) -> dict:
    """IV rank ((cur-min)/(max-min)) and IV percentile (share of days below cur) over the last `lookback` obs."""
    h = pd.Series(history, dtype=float).dropna().iloc[-lookback:]
    cur = float(h.iloc[-1]) if current is None and len(h) else current
    if len(h) < 2 or cur is None or not np.isfinite(cur):
        return {"iv_rank": math.nan, "iv_percentile": math.nan, "n_obs": len(h)}
    lo, hi = float(h.min()), float(h.max())
    return {"iv_rank": (cur - lo) / (hi - lo) if hi > lo else math.nan, "iv_percentile": float((h < cur).mean()),
            "n_obs": len(h), "iv_min": lo, "iv_max": hi}


def realized_vol(close: pd.Series, n: int = 20, periods: int = 252) -> pd.Series:
    """Rolling close-to-close realized vol (annualised, log returns)."""
    return np.log(close).diff().rolling(n).std() * math.sqrt(periods)


def yang_zhang(df: pd.DataFrame, n: int = 20, periods: int = 252) -> pd.Series:
    """Rolling Yang-Zhang OHLC volatility (annualised): overnight + k*open-close + (1-k)*Rogers-Satchell."""
    o = np.log(df["open"] / df["close"].shift())
    c = np.log(df["close"] / df["open"])
    u, d = np.log(df["high"] / df["open"]), np.log(df["low"] / df["open"])
    rs = u * (u - c) + d * (d - c)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = o.rolling(n).var() + k * c.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(var * periods)


def iv_vs_rv(iv: float, ohlc: pd.DataFrame) -> dict:
    """ATM IV vs 20d/60d close-close and Yang-Zhang realized vol; vrp = iv - rv (vol points as decimals)."""
    last = lambda s: float(s.dropna().iloc[-1]) if s.notna().any() else math.nan  # noqa: E731
    out = {"atm_iv30": iv, "rv20": last(realized_vol(ohlc["close"], 20)), "rv60": last(realized_vol(ohlc["close"], 60))}
    if {"open", "high", "low"} <= set(ohlc):
        out["yz20"], out["yz60"] = last(yang_zhang(ohlc, 20)), last(yang_zhang(ohlc, 60))
    out["vrp_20"] = iv - out["rv20"]
    out["vrp_60"] = iv - out["rv60"]
    out["iv_rv20_ratio"] = iv / out["rv20"] if out["rv20"] else math.nan
    return out


# ---- strategy payoff -------------------------------------------------------------------
@dataclass
class Leg:
    kind: str  # 'C', 'P' or 'U' (underlying / future / CFD)
    strike: float = math.nan
    days: float = 0.0
    qty: float = 1.0
    premium: float | None = None  # per unit; underlying: entry price
    iv: float | None = None  # per-leg vol override
    label: str = field(default="", compare=False)


_TENOR = re.compile(r"^(\d+(?:\.\d+)?)([dwmy])$", re.I)


def _days(tok: str, today=None) -> float:
    m = _TENOR.match(tok)
    if m:
        return float(m.group(1)) * {"d": 1, "w": 7, "m": 365 / 12, "y": 365}[m.group(2).lower()]
    return float((pd.Timestamp(tok).normalize() - _naive_day(today if today is not None else pd.Timestamp.now(tz="UTC"))).days)


def parse_leg(s: str, today=None) -> Leg:
    """'C 100 30d +1 @2.5', 'P 95 2026-12-18 -2 @1.1 iv=28%', 'U +1 @100'. Premium optional (BS-priced)."""
    toks, prem, iv = s.split(), None, None
    rest = []
    for t in toks:
        if t.startswith("@"):
            prem = float(t[1:])
        elif t.lower().startswith("iv="):
            v = t[3:]
            iv = float(v.rstrip("%")) / 100 if v.endswith("%") or float(v) > 3 else float(v)
        else:
            rest.append(t)
    kind = rest[0].upper()[0]
    if kind in "USF":
        return Leg("U", qty=float(rest[1]) if len(rest) > 1 else 1.0, premium=prem, label=s)
    if kind not in "CP" or len(rest) < 3:
        raise ValueError(f"bad leg '{s}': use 'C|P STRIKE EXPIRY(30d|2026-12-18) QTY [@PREMIUM] [iv=25%]'")
    return Leg(kind, float(rest[1]), _days(rest[2], today), float(rest[3]) if len(rest) > 3 else 1.0, prem, iv, s)


def _leg_value(leg: Leg, S, days_left, vol, r, q):
    if leg.kind == "U":
        return np.asarray(S, dtype=float)
    return np.asarray(bs_price(S, leg.strike, max(days_left, 0) / YEAR, r, leg.iv or vol, q, leg.kind), dtype=float)


def payoff(legs: list[Leg], spot: float, vol: float, r: float = 0.0, q: float = 0.0, multiplier: float = 1.0,
           range_pct: float = 0.3, steps: int = 13) -> dict:
    """Payoff at the first option expiry (later legs valued by BS with time left) and BS value today across a
    spot range; breakevens, max profit/loss (inf = unbounded), net premium (+debit) and net greeks."""
    legs = [Leg(**{**lg.__dict__}) for lg in legs]
    for lg in legs:
        if lg.premium is None:
            lg.premium = float(spot) if lg.kind == "U" else float(_leg_value(lg, spot, lg.days, vol, r, q))
    opt_days = [lg.days for lg in legs if lg.kind != "U"]
    t0 = min(opt_days) if opt_days else 0.0

    def at_expiry(S):
        return sum(lg.qty * (_leg_value(lg, S, lg.days - t0, vol, r, q) - lg.premium) for lg in legs) * multiplier

    def today(S):
        return sum(lg.qty * (_leg_value(lg, S, lg.days, vol, r, q) - lg.premium) for lg in legs) * multiplier

    ks = [lg.strike for lg in legs if lg.kind != "U"]
    hi = max([3 * spot] + [1.5 * k for k in ks])
    grid = np.unique(np.concatenate([np.linspace(0, hi, 6001), ks, [spot]]))
    pe = np.asarray(at_expiry(grid), dtype=float)
    bes = [float(grid[i]) for i in np.where(pe == 0)[0]]
    for i in np.where(np.sign(pe[:-1]) * np.sign(pe[1:]) < 0)[0]:
        bes.append(float(grid[i] - pe[i] * (grid[i + 1] - grid[i]) / (pe[i + 1] - pe[i])))
    slope = (pe[-1] - pe[-2]) / (grid[-1] - grid[-2])
    mx = math.inf if slope > 1e-9 else float(pe.max())
    mn = -math.inf if slope < -1e-9 else float(pe.min())

    net = {"delta": 0.0, "gamma": 0.0, "vega_1pt": 0.0, "theta_day": 0.0, "rho_1pt": 0.0, "vanna": 0.0, "charm_day": 0.0}
    for lg in legs:
        w = lg.qty * multiplier
        if lg.kind == "U":
            net["delta"] += w
            continue
        g = greeks(spot, lg.strike, lg.days / YEAR, r, lg.iv or vol, q, lg.kind)
        for k, src, sc in (("delta", "delta", 1), ("gamma", "gamma", 1), ("vega_1pt", "vega", 0.01), ("theta_day", "theta", 1 / YEAR),
                           ("rho_1pt", "rho", 0.01), ("vanna", "vanna", 0.01), ("charm_day", "charm", 1 / YEAR)):
            net[k] += w * g[src] * sc

    S = spot * (1 + np.linspace(-range_pct, range_pct, steps))
    table = pd.DataFrame({"spot": S, "pct": S / spot - 1, "pnl_expiry": at_expiry(S), "pnl_today": today(S)}).set_index("spot")
    legs_df = pd.DataFrame([{"leg": lg.label or lg.kind, "type": lg.kind, "strike": lg.strike, "days": lg.days, "qty": lg.qty,
                             "premium": lg.premium, "iv": lg.iv or (vol if lg.kind != "U" else math.nan)} for lg in legs])
    return {"net_premium": sum(lg.qty * lg.premium for lg in legs if lg.kind != "U") * multiplier,
            "breakevens": sorted(set(round(b, 6) for b in bes)), "max_profit": mx, "max_loss": mn,
            "payoff_horizon_days": t0, "greeks": net, "legs": legs_df, "table": table}


# ---- desk (network orchestration) ----------------------------------------------------
def desk(ticker: str, expiries: int | None = 6, r: float | None = None, q: float = 0.0,
         convention: str = "long_calls_short_puts", top: int = 15) -> dict:
    """Fetch chain + prices and run every analytic. Records today's 30d ATM IV so IV rank accumulates."""
    spot, raw = fetch_chain(ticker, expiries)
    r = risk_free_rate() if r is None else r
    chain = normalize_chain(raw, spot, r, q)
    term = atm_term_structure(chain)
    iv30 = atm_iv_at(term, 30)
    hist = record_atm_iv(ticker, iv30, spot)
    out = {"ticker": ticker, "spot": spot, "r": r, "q": q, "asof": chain.attrs["asof"], "chain": chain, "term": term,
           "skew": skew(chain), "expected_move": expected_move(chain), "max_pain": max_pain(chain),
           "pcr": put_call_ratios(chain), "gex": gex(chain, convention=convention), "unusual": unusual_activity(chain, top=top),
           "atm_iv30": iv30, "iv_rank": iv_rank(hist["atm_iv30"], iv30)}
    try:
        from .data import get_prices
        px = get_prices(ticker, period="1y")
        out["iv_vs_rv"], out["prices_source"] = iv_vs_rv(iv30, px), px.attrs.get("source")
    except Exception as e:
        out["iv_vs_rv"], out["prices_source"] = {}, f"unavailable: {str(e)[:80]}"
    return out


# ---- CLI --------------------------------------------------------------------------------
def jsonable(obj):
    """DataFrames -> records, Series -> {str(index): value}, recursively, so --json never trips on keys."""
    if isinstance(obj, pd.DataFrame):
        return json_records(obj)
    if isinstance(obj, pd.Series):
        return {str(k): v for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def json_records(df: pd.DataFrame) -> list[dict]:
    import json
    return json.loads(df.reset_index().to_json(orient="records", date_format="iso", default_handler=str))


def _ix(df):
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y-%m-%d")
    return df


def cmd_options(a):
    from .cli import show
    d = desk(a.ticker, a.expiries or None, a.rate, a.div, a.gex, a.top)
    if a.json:
        show(jsonable({k: v for k, v in d.items() if k != "chain"} | {"gex": {k: v for k, v in d["gex"].items() if k != "profile"}}), True)
        return
    print(f"## {a.ticker} options desk · spot {d['spot']:,.2f} · r {d['r']:.2%} · q {d['q']:.2%}")
    show(_ix(d["term"]).assign(T=lambda x: x["T"] * YEAR).rename(columns={"T": "days"}), title="ATM IV term structure (fwd_vol = forward vol between expiries)")
    show(_ix(d["skew"]), title="Skew: 25-delta risk reversal (call-put) and butterfly")
    show(_ix(d["expected_move"]), title="Expected move: ATM straddle vs 1-sd from ATM IV")
    show(_ix(d["max_pain"]), title="Max pain")
    show(d["pcr"], title="Put/call ratios")
    g = d["gex"]
    show({"convention": g["convention"], "total_gex_$per1%": g["total"], "zero_gamma_flip": g["flip"],
          "flip_vs_spot": g["flip"] / d["spot"] - 1 if np.isfinite(g["flip"]) else math.nan,
          "max_pos_strike": g["max_pos_strike"], "max_neg_strike": g["max_neg_strike"]}, title="Dealer gamma exposure (GEX, $ per 1% move)")
    by = g["by_strike"]
    near = by[(by.index > d["spot"] * 0.85) & (by.index < d["spot"] * 1.15)]
    show(near.loc[near["net_gex"].abs().sort_values(ascending=False).index[:12]].sort_index(), title="GEX by strike (largest within ±15%)")
    ua = d["unusual"].copy()
    if len(ua):
        ua["expiry"] = ua["expiry"].dt.strftime("%Y-%m-%d")
    show(ua, title="Unusual activity")
    show({"atm_iv30": d["atm_iv30"], **d["iv_rank"]}, title="IV rank / percentile (history from data/iv_history)")
    show(d["iv_vs_rv"], title="IV vs realized vol (variance risk premium)")
    print(f"\n_Source: Yahoo via yfinance ({len(d['chain'])} contracts, {d['chain']['expiry'].nunique()} expiries) · prices {d['prices_source']}"
          f" · {pd.Timestamp(d['asof']):%Y-%m-%d %H:%M} UTC_")


def cmd_payoff(a):
    from .cli import show
    legs = [parse_leg(s) for s in a.leg]
    res = payoff(legs, a.spot, a.vol, a.rate, a.div, a.mult, a.range, a.steps)
    if a.json:
        show(jsonable(res), True)
        return
    show(res["legs"].set_index("leg"), title="Legs (premium per unit; blank premiums priced by BS)")
    show({"net_premium (+debit)": res["net_premium"], "max_profit": res["max_profit"], "max_loss": res["max_loss"],
          "payoff_at_days": res["payoff_horizon_days"], "breakevens": ", ".join(f"{b:,.2f}" for b in res["breakevens"]) or "none"},
         title="Summary (at first expiry)")
    show(res["greeks"], title="Net greeks (vega/rho per 1 pt, theta/charm per day, x multiplier)")
    show(res["table"], title="P&L across spot")
    print(f"\n_Model: Black-Scholes-Merton, spot {a.spot}, vol {a.vol:.2%}, r {a.rate:.2%}, q {a.div:.2%} · "
          f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def register(add):
    q = add("options", cmd_options, "options desk: term structure, skew, expected move, max pain, PCR, GEX, unusual activity, IV vs RV")
    q.add_argument("ticker"); q.add_argument("--expiries", type=int, default=6, help="first N expiries (0 = all)")
    q.add_argument("--rate", type=float, help="risk-free rate (default: ^IRX, else 4%%)"); q.add_argument("--div", type=float, default=0.0)
    q.add_argument("--gex", default="long_calls_short_puts", choices=list(GEX_CONVENTIONS)); q.add_argument("--top", type=int, default=15)

    q = add("payoff", cmd_payoff, 'strategy payoff/greeks: --spot 100 --vol 0.25 --leg "C 100 30d +1 @2.5" --leg ...')
    q.add_argument("--spot", type=float, required=True); q.add_argument("--vol", type=float, required=True)
    q.add_argument("--rate", type=float, default=0.0); q.add_argument("--div", type=float, default=0.0)
    q.add_argument("--leg", action="append", required=True, help="'C|P STRIKE 30d|YYYY-MM-DD QTY [@PREM] [iv=25%%]' or 'U QTY [@PRICE]'")
    q.add_argument("--mult", type=float, default=1.0, help="contract multiplier / stake per point")
    q.add_argument("--range", type=float, default=0.3); q.add_argument("--steps", type=int, default=13)
