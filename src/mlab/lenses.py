"""Investor lenses as data: each returns the numbers that lens cares about plus a pass/fail checklist.
The verdict and the trade are made by the analyst (Claude + Cezar) on top of these facts.

  buffett(ticker)  quality compounder at a sensible price: ROIC, margins, FCF, debt, buybacks, owner-earnings yield, DCF
  burry(ticker)    contrarian deep value / balance-sheet: EV/EBIT, FCF yield, net cash, insiders, shorts, washout
  soros(...)       macro + reflexivity: liquidity, rates, credit, USD, trend/momentum stage of the boom-bust
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mcap(ticker: str, fin: pd.DataFrame | None = None) -> dict:
    from .providers import yahoo
    info = {}
    try:
        info = yahoo.info(ticker)
    except Exception:
        pass
    px = info.get("currentPrice") or info.get("regularMarketPrice")
    mc = info.get("marketCap")
    if mc is None and fin is not None and "diluted_shares" in fin and px:
        mc = px * fin["diluted_shares"].dropna().iloc[-1]
    return {"price": px, "market_cap": mc, "enterprise_value": info.get("enterpriseValue"), "info": info}


def _cagr(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def dcf_owner_earnings(oe: float, growth: float, years: int = 10, terminal_g: float = 0.025, discount: float = 0.10) -> float:
    """Two-stage DCF on owner earnings; growth fades linearly to terminal_g over `years`."""
    v, cf = 0.0, oe
    for t in range(1, years + 1):
        g = growth + (terminal_g - growth) * (t - 1) / max(1, years - 1)
        cf *= 1 + g
        v += cf / (1 + discount) ** t
    tv = cf * (1 + terminal_g) / (discount - terminal_g)
    return v + tv / (1 + discount) ** years


def buffett(ticker: str) -> dict:
    from .providers import sec
    fin = sec.annual_financials(ticker)
    m = _mcap(ticker, fin)
    last = fin.iloc[-1]
    oe = (last.get("fcf_ex_sbc") if pd.notna(last.get("fcf_ex_sbc", np.nan)) else last.get("fcf"))
    rev_cagr = _cagr(fin.get("revenue", pd.Series(dtype=float)))
    fcf_cagr = _cagr(fin.get("fcf", pd.Series(dtype=float)))
    growth = np.nanmin([x for x in [rev_cagr, fcf_cagr, 0.12] if pd.notna(x)]) if pd.notna(rev_cagr) else 0.05
    iv = {}
    if oe and oe > 0:
        for name, g in {"bear": growth * 0.5, "base": growth, "bull": min(growth * 1.4, 0.25)}.items():
            iv[name] = dcf_owner_earnings(oe, g)
    debt_fcf = last.get("total_debt", np.nan) / last.get("fcf", np.nan) if last.get("fcf", 0) > 0 else np.nan
    shares = fin.get("diluted_shares", pd.Series(dtype=float)).dropna()
    share_chg = (shares.iloc[-1] / shares.iloc[0]) ** (1 / max(1, len(shares) - 1)) - 1 if len(shares) > 1 else np.nan
    roic = fin.get("roic_pre_tax", pd.Series(dtype=float)).dropna()
    gm = fin.get("gross_margin", pd.Series(dtype=float)).dropna()
    mc = m["market_cap"]
    metrics = {
        "fiscal_years": f"{fin.index[0].year}-{fin.index[-1].year}", "price": m["price"], "market_cap": mc,
        "revenue_cagr": rev_cagr, "fcf_cagr": fcf_cagr, "roic_pre_tax_avg": roic.mean() if len(roic) else np.nan,
        "roic_pre_tax_min": roic.min() if len(roic) else np.nan, "gross_margin_last": gm.iloc[-1] if len(gm) else np.nan,
        "gross_margin_stdev": gm.std() if len(gm) > 2 else np.nan, "op_margin_last": last.get("op_margin"),
        "fcf_conversion": last.get("fcf", np.nan) / last.get("net_income", np.nan) if last.get("net_income", 0) > 0 else np.nan,
        "sbc_pct_revenue": last.get("sbc", np.nan) / last.get("revenue", np.nan),
        "debt_to_fcf": debt_fcf, "share_count_cagr": share_chg,
        "owner_earnings": oe, "owner_earnings_yield": oe / mc if oe and mc else np.nan,
        "intrinsic_value_mcap": iv, "margin_of_safety_base": (iv["base"] / mc - 1) if iv and mc else np.nan,
    }
    checks = {
        "ROIC >= 15% every year": bool(len(roic)) and roic.min() >= 0.15,
        "Gross margin stable (stdev < 5pts)": bool(len(gm) > 2) and gm.std() < 0.05,
        "FCF conversion >= 80%": pd.notna(metrics["fcf_conversion"]) and metrics["fcf_conversion"] >= 0.8,
        "Debt < 3x FCF": pd.notna(debt_fcf) and debt_fcf < 3,
        "Share count flat or shrinking": pd.notna(share_chg) and share_chg <= 0.005,
        "Owner-earnings yield >= 5%": pd.notna(metrics["owner_earnings_yield"]) and metrics["owner_earnings_yield"] >= 0.05,
        "Base DCF >= 25% above market cap": pd.notna(metrics["margin_of_safety_base"]) and metrics["margin_of_safety_base"] >= 0.25,
    }
    return {"lens": "Buffett", "ticker": ticker, "metrics": metrics, "checks": checks,
            "score": f"{sum(checks.values())}/{len(checks)}", "financials": fin,
            "sources": [f"SEC EDGAR XBRL companyfacts ({ticker}, FY{fin.index[-1].year})", "Yahoo Finance quote/market cap"]}


def burry(ticker: str) -> dict:
    from .data import get_prices
    from .providers import sec
    fin = sec.annual_financials(ticker)
    m = _mcap(ticker, fin)
    info, last = m["info"], fin.iloc[-1]
    mc = m["market_cap"]
    ev = m["enterprise_value"] or (mc + last.get("net_debt", 0) if mc else None)
    px = get_prices(ticker, period="5y")
    c = px["close"]
    ins = pd.DataFrame()
    try:
        ins = sec.insider_filings(ticker, 100)
    except Exception:
        pass
    ins_90 = int((pd.to_datetime(ins["filingDate"]) >= pd.Timestamp.now() - pd.Timedelta(days=90)).sum()) if len(ins) else None
    tang_bv = last.get("equity", np.nan) - last.get("goodwill", 0) if pd.notna(last.get("equity", np.nan)) else np.nan
    metrics = {
        "price": m["price"], "market_cap": mc, "enterprise_value": ev,
        "ev_ebit": ev / last["operating_income"] if ev and last.get("operating_income", 0) > 0 else np.nan,
        "fcf_yield": last.get("fcf", np.nan) / mc if mc else np.nan,
        "net_cash_pct_mcap": -last.get("net_debt", np.nan) / mc if mc else np.nan,
        "price_to_tangible_book": mc / tang_bv if mc and tang_bv and tang_bv > 0 else np.nan,
        "drawdown_from_5y_high": c.iloc[-1] / c.max() - 1, "pct_above_52w_low": c.iloc[-1] / c.tail(252).min() - 1,
        "current_ratio": last.get("current_ratio"), "interest_cover": last.get("interest_cover"),
        "short_pct_float": info.get("shortPercentOfFloat"), "short_ratio_days": info.get("shortRatio"),
        "insider_form4_last_90d": ins_90, "held_by_insiders": info.get("heldPercentInsiders"),
        "held_by_institutions": info.get("heldPercentInstitutions"),
    }
    checks = {
        "EV/EBIT <= 8": pd.notna(metrics["ev_ebit"]) and metrics["ev_ebit"] <= 8,
        "FCF yield >= 10%": pd.notna(metrics["fcf_yield"]) and metrics["fcf_yield"] >= 0.10,
        "Hated: >= 40% off 5y high": metrics["drawdown_from_5y_high"] <= -0.40,
        "Near lows: within 20% of 52w low": metrics["pct_above_52w_low"] <= 0.20,
        "Balance sheet survives: current ratio >= 1.2": pd.notna(metrics["current_ratio"]) and metrics["current_ratio"] >= 1.2,
        "Interest cover >= 3x (or no debt)": pd.isna(metrics["interest_cover"]) or metrics["interest_cover"] >= 3,
        "Insider Form 4 activity in last 90d": bool(ins_90),
    }
    return {"lens": "Burry", "ticker": ticker, "metrics": metrics, "checks": checks,
            "score": f"{sum(checks.values())}/{len(checks)}", "insider_filings": ins.head(15),
            "sources": [f"SEC EDGAR XBRL + Form 4 ({ticker})", px.attrs.get("source", "prices"), "Yahoo Finance quote/short data"]}


def net_liquidity(start="2020-01-01") -> pd.DataFrame:
    """Fed balance sheet - TGA - ON RRP (USD bn): the liquidity tide risk assets swim in."""
    from .providers.macro import fred
    w = fred("WALCL", start) / 1000          # millions -> billions
    tga = fred("WTREGEN", start) / 1000      # millions -> billions
    rrp = fred("RRPONTSYD", start)           # billions
    df = pd.concat({"fed_bs": w, "tga": tga, "rrp": rrp}, axis=1).ffill().dropna()
    df["net_liquidity"] = df["fed_bs"] - df["tga"] - df["rrp"]
    return df.resample("W-WED").last()


def soros(assets: list[str] | None = None) -> dict:
    """Macro regime + cross-asset trend board. Reflexivity read: is price feeding the fundamental
    story (boom), or has the trend broken while the narrative still holds (bust beginning)?"""
    from . import ta
    from .data import get_many
    from .providers.macro import dashboard
    assets = assets or ["^GSPC", "^NDX", "^STOXX50E", "^FTSE", "^N225", "DX-Y.NYB", "EURUSD=X", "GBPUSD=X",
                        "USDJPY=X", "GC=F", "CL=F", "HG=F", "^TNX", "BTC-USD", "^VIX"]
    panel = get_many(assets, period="2y")
    board = []
    for a in panel.columns:
        s = panel[a].dropna()
        if len(s) < 210:
            continue
        sma200 = s.rolling(200).mean()
        board.append({"asset": a, "last": s.iloc[-1], "r1m": s.iloc[-1] / s.iloc[-22] - 1, "r3m": s.iloc[-1] / s.iloc[-64] - 1,
                      "r12m": s.iloc[-1] / s.iloc[-253] - 1 if len(s) > 253 else np.nan,
                      "vs_200d": s.iloc[-1] / sma200.iloc[-1] - 1, "200d_slope_1m": sma200.iloc[-1] / sma200.iloc[-22] - 1,
                      "rsi14": ta.rsi(s).iloc[-1],
                      "stage": _stage(s, sma200)})
    out = {"lens": "Soros", "trend_board": pd.DataFrame(board), "errors": panel.attrs.get("errors"),
           "sources": ["Yahoo Finance (cross-asset closes)", "FRED (macro)"]}
    try:
        out["macro"] = dashboard()
    except Exception as e:
        out["macro_error"] = str(e)
    try:
        nl = net_liquidity()
        out["net_liquidity"] = {"last_bn": nl["net_liquidity"].iloc[-1], "chg_4w_bn": nl["net_liquidity"].diff(4).iloc[-1],
                                "chg_13w_bn": nl["net_liquidity"].diff(13).iloc[-1], "as_of": str(nl.index[-1].date())}
    except Exception as e:
        out["net_liquidity_error"] = str(e)
    return out


def _stage(s: pd.Series, sma200: pd.Series) -> str:
    """Crude boom-bust stage tag for the reflexivity read."""
    ext = s.iloc[-1] / sma200.iloc[-1] - 1
    slope = sma200.iloc[-1] / sma200.iloc[-22] - 1
    if slope > 0 and ext > 0.15:
        return "boom: extended (reflexive acceleration)"
    if slope > 0 and ext > 0:
        return "boom: trend intact"
    if slope > 0 and ext <= 0:
        return "test: below rising 200d (boom questioned)"
    if slope <= 0 and ext > 0:
        return "repair: above falling 200d"
    if slope <= 0 and ext < -0.15:
        return "bust: capitulation zone"
    return "bust: trend down"
