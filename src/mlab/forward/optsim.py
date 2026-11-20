"""Rule-based option books marked daily from the live chain, with full Greeks and P&L attribution.

Per underlying and run: fetch the expiries the books need (the one nearest each target DTE plus any held),
solve IV per quote, and record a volatility snapshot (ATM IV30, 25-delta skew, term slope, 20-day realised vol,
IV/RV) plus a near-the-money chain snapshot. Each structure then:
- when flat: checks its entry condition (some only sell rich vol or buy cheap vol) and logs the trigger either
  way; on entry it picks the expiry nearest its DTE target and strikes nearest its delta targets, buying at
  the ask and selling at the bid, plus commission;
- when open: marks every leg at the mid (IV interpolated on that expiry's smile when a strike has no quote),
  computes delta, gamma, theta (per day), vega (per vol point) per leg and in total, splits the day's P&L into
  delta, gamma, theta, vega and residual from yesterday's Greeks, and exits at 21 DTE, the profit target or the
  loss stop, crossing the spread again.
Returns are daily P&L over the trade's capital base (max loss for defined risk, strike for cash-secured puts,
spot for stock legs, 20% of spot for naked short strangles: assumption, broker margin unverified).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import options as opt
from . import costs

MULT = 100
NAKED_MARGIN = 0.20  # assumption: short strangle capital base, 20% of notional


def fetch_selected(ticker: str, dtes: list[int], held: list[str], now: pd.Timestamp | None = None):
    """(spot, normalized chain) for the expiries nearest each target DTE plus the held ones."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = list(t.options or [])
    if not exps:
        raise LookupError(f"no listed options for {ticker}")
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    by_dte = {e: (pd.Timestamp(e) - today).days for e in exps}
    want = set(held)
    for d in dtes:
        cands = [e for e, x in by_dte.items() if x >= 1]
        if cands:
            want.add(min(cands, key=lambda e: abs(by_dte[e] - d)))
    frames = []
    for e in sorted(want):
        if e not in by_dte:
            continue
        ch = t.option_chain(e)
        for typ, df in (("C", ch.calls), ("P", ch.puts)):
            if df is not None and len(df):
                frames.append(df.assign(expiry=e, type=typ))
    if not frames:
        raise LookupError(f"empty chain for {ticker}")
    spot = opt.fetch_spot(ticker)
    return spot, pd.concat(frames, ignore_index=True)


def vol_snapshot(chain: pd.DataFrame, spot: float, closes: pd.Series | None) -> dict:
    term = opt.atm_term_structure(chain, spot)
    iv30 = opt.atm_iv_at(term, 30)
    out = {"spot": spot, "atm_iv30": iv30}
    try:
        sk = opt.skew(chain, spot)
        near = sk.iloc[(sk["dte"] - 30).abs().argsort()[:1]]
        out.update({"rr25": float(near["rr25"].iloc[0]), "bf25": float(near["bf25"].iloc[0]),
                    "skew_dte": int(near["dte"].iloc[0])})
    except Exception:
        pass
    t = term.dropna(subset=["atm_iv"])
    if len(t) >= 2:
        out["term_slope"] = float(t["atm_iv"].iloc[-1] - t["atm_iv"].iloc[0])
        out["term_dte"] = [int(t["dte"].iloc[0]), int(t["dte"].iloc[-1])]
    if closes is not None and len(closes) > 21:
        rv = float(closes.pct_change().tail(20).std() * math.sqrt(252))
        out["rv20"] = rv
        out["iv_rv"] = iv30 / rv if rv else math.nan
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in out.items()}


def _quote(chain: pd.DataFrame, expiry: str, typ: str, strike: float) -> dict:
    g = chain[(chain["expiry"] == pd.Timestamp(expiry)) & (chain["type"] == typ)]
    hit = g[np.isclose(g["strike"], strike)]
    if len(hit) and np.isfinite(hit["iv"].iloc[0]):
        r = hit.iloc[0]
        return {"bid": float(r["bid"]), "ask": float(r["ask"]), "mid": float(r["mid"]), "iv": float(r["iv"]),
                "T": float(r["T"]), "dte": int(r["dte"]), "src": "quote"}
    h = g[g["iv"].notna()].sort_values("strike")
    if h.empty:
        return {}
    iv = float(np.interp(strike, h["strike"], h["iv"]))
    T = float(h["T"].iloc[0])
    return {"bid": math.nan, "ask": math.nan, "mid": math.nan, "iv": iv, "T": T, "dte": int(h["dte"].iloc[0]),
            "src": "smile_interp"}


def _leg_greeks(S, K, T, r, iv, typ) -> dict:
    g = opt.greeks(S, K, T, r, iv, 0.0, typ)
    return {"delta": float(g["delta"]), "gamma": float(g["gamma"]), "theta_day": float(g["theta"]) / 365.0,
            "vega_pt": float(g["vega"]) / 100.0}


def mark(pos: dict, chain: pd.DataFrame, spot: float, r: float) -> dict:
    """Value at mid and Greeks per leg and in total ($, per contract set)."""
    legs, tot = [], {"value": 0.0, "delta": 0.0, "gamma": 0.0, "theta_day": 0.0, "vega_pt": 0.0}
    for lg in pos["legs"]:
        if lg["type"] == "U":
            v = lg["qty"] * spot
            row = {**lg, "price": spot, "value": v, "delta": lg["qty"], "gamma": 0.0, "theta_day": 0.0, "vega_pt": 0.0}
        else:
            q = _quote(chain, lg["expiry"], lg["type"], lg["strike"])
            if not q:
                T = max((pd.Timestamp(lg["expiry"]) - pd.Timestamp.now().normalize()).days, 0) / 365.0
                q = {"iv": lg.get("iv_last", 0.3), "T": T, "mid": math.nan, "src": "stale"}
            price = q["mid"] if np.isfinite(q.get("mid", math.nan)) else float(
                opt.bs_price(spot, lg["strike"], q["T"], r, q["iv"], 0.0, lg["type"]))
            g = _leg_greeks(spot, lg["strike"], q["T"], r, q["iv"], lg["type"])
            n = lg["qty"] * MULT
            row = {**lg, "price": price, "iv": q["iv"], "iv_src": q["src"], "bid": q.get("bid"), "ask": q.get("ask"),
                   "dte": q.get("dte"), "value": n * price, "delta": n * g["delta"], "gamma": n * g["gamma"],
                   "theta_day": n * g["theta_day"], "vega_pt": n * g["vega_pt"]}
        legs.append(row)
        for k in tot:
            tot[k] += row[k]
    return {"legs": legs, **tot}


def _exit_value(m: dict) -> float:
    """Value received when closing: long legs at the bid, short legs at the ask (mid when one-sided)."""
    v = 0.0
    for lg in m["legs"]:
        if lg["type"] == "U":
            v += lg["value"]
            continue
        n = lg["qty"] * MULT
        px = lg.get("bid") if n > 0 else lg.get("ask")
        if px is None or not np.isfinite(px) or px <= 0:
            px = lg["price"]
        v += n * px
    return v


def pick_legs(st: dict, chain: pd.DataFrame, spot: float) -> list[dict] | None:
    exps = sorted(chain["expiry"].unique())
    dte = {e: int(chain.loc[chain["expiry"] == e, "dte"].iloc[0]) for e in exps}
    ok = [e for e in exps if 20 <= dte[e] <= 80]
    if not ok:
        return None
    e = min(ok, key=lambda x: abs(dte[x] - st["dte"]))
    g = chain[(chain["expiry"] == e) & chain["iv"].notna() & chain["mid"].notna()]
    legs = []
    atm_k = None
    for typ, tgt, qty in st["legs"]:
        if typ == "U":
            legs.append({"type": "U", "qty": qty})
            continue
        h = g[g["type"] == typ]
        if h.empty:
            return None
        if tgt == "atm":
            atm_k = atm_k or float(h.iloc[(h["strike"] - spot).abs().argsort().iloc[0]]["strike"])
            k = atm_k
        else:
            k = float(h.iloc[(h["delta"] - tgt).abs().argsort().iloc[0]]["strike"])
        legs.append({"type": typ, "strike": k, "expiry": str(pd.Timestamp(e).date()), "qty": qty})
    strikes = [(lg["type"], lg["strike"]) for lg in legs if lg["type"] != "U"]
    if len(set(strikes)) < len(strikes):
        return None   # two legs collapsed onto one strike: chain too sparse for this structure today
    return legs


def capital_base(name: str, legs: list[dict], spot: float, entry_cost: float) -> float:
    shorts = [lg for lg in legs if lg["type"] != "U" and lg["qty"] < 0]
    longs = [lg for lg in legs if lg["type"] != "U" and lg["qty"] > 0]
    if any(lg["type"] == "U" for lg in legs):
        return spot * MULT
    if name.startswith("cash_secured_put"):
        return shorts[0]["strike"] * MULT
    if shorts and longs:          # defined-risk spread / condor: widest wing minus credit
        width = 0.0
        for t in "CP":
            ks = sorted(lg["strike"] for lg in legs if lg["type"] == t)
            if len(ks) >= 2:
                width = max(width, ks[-1] - ks[0])
        return max(width * MULT + min(entry_cost, 0.0), abs(entry_cost), 1.0)
    if shorts:
        return NAKED_MARGIN * spot * MULT   # one side of a naked short margined (assumption)
    return max(abs(entry_cost), 1.0)


def step_option(cell: dict, st: dict | None, chain: pd.DataFrame, spot: float, r: float, volsnap: dict,
                today: str, prov: dict, earnings: list | None) -> tuple:
    """One daily step for an option cell. Returns (state, signal row, triggers, fills, greeks rows)."""
    spec, name = cell["spec"], cell["structure"]
    ex = cell["exits"]
    st = st or {"spec_sha": cell["spec_sha"], "first_bar": today, "last_bar": None, "pos": None, "pnl_cum": 0.0,
                "trades": 0}
    trig, fills, grk = [], [], []
    pnl_day, ret_day, attr = 0.0, 0.0, {}
    if st["last_bar"] == today:
        return st, None, [], [], []
    pos = st["pos"]
    if pos:
        m = mark(pos, chain, spot, r)
        prev = pos["last"]
        pnl_day = m["value"] - prev["value"]
        dS = spot - prev["spot"]
        dt = max((pd.Timestamp(today) - pd.Timestamp(prev["date"])).days, 0)
        div = sum(((lg.get("iv") or 0) - (pl.get("iv") or 0)) * 100 * (pl.get("vega_pt") or 0)
                  for lg, pl in zip(m["legs"], prev["legs"]) if lg["type"] != "U")
        attr = {"delta": prev["delta"] * dS, "gamma": 0.5 * prev["gamma"] * dS * dS, "theta": prev["theta_day"] * dt,
                "vega": div}
        attr["residual"] = pnl_day - sum(attr.values())
        trade_pnl = m["value"] - pos["entry_value"] - pos["commission"]
        basis = abs(pos["premium"]) if pos["premium"] else pos["capital"]
        reason = None
        if min((lg.get("dte") or 999) for lg in m["legs"] if lg["type"] != "U") <= ex["dte_exit"]:
            reason = "dte"
        elif spec["credit"] and trade_pnl >= ex["take_profit"] * basis:
            reason = "take_profit"
        elif spec["credit"] and -trade_pnl >= ex["stop_loss"] * basis:
            reason = "stop_loss"
        elif not spec["credit"] and trade_pnl >= ex["take_profit"] * basis:
            reason = "take_profit"
        elif not spec["credit"] and -trade_pnl >= ex["stop_loss"] * basis:
            reason = "stop_loss"
        for lg in m["legs"]:
            grk.append({"cell": cell["id"], "bar": today, "underlying": cell["symbols"][0], "spot": spot,
                        "leg": f"{lg['type']} {lg.get('strike', '')} {lg.get('expiry', '')} x{lg['qty']}",
                        **{k: lg.get(k) for k in ("price", "iv", "iv_src", "bid", "ask", "dte", "value", "delta",
                                                  "gamma", "theta_day", "vega_pt")}, **prov})
        grk.append({"cell": cell["id"], "bar": today, "underlying": cell["symbols"][0], "spot": spot, "leg": "TOTAL",
                    **{k: m[k] for k in ("value", "delta", "gamma", "theta_day", "vega_pt")},
                    "pnl_day": pnl_day, "attribution": attr, "trade_pnl": trade_pnl, **prov})
        if reason:
            out_v = _exit_value(m)
            slip = out_v - m["value"]
            pnl_day += slip
            fills.append({"cell": cell["id"], "bar": today, "action": "close", "reason": reason, "value": out_v,
                          "slippage": slip, "commission": costs.OPTION_COMMISSION_CLOSE, "legs": pos["legs"],
                          "cost_source": costs.OPTION_COST_SOURCE, **prov})
            trig.append({"cell": cell["id"], "bar": today, "symbol": cell["symbols"][0], "kind": "exit",
                         "reason": reason, "taken_raw": True, "taken_desk": True, "trade_pnl": trade_pnl + slip,
                         "context": volsnap, **prov, "spec_sha": cell["spec_sha"]})
            ret_day = pnl_day / pos["capital"]
            st["pnl_cum"] += pnl_day
            st["pos"] = None
            pos = None
        else:
            ret_day = pnl_day / pos["capital"]
            st["pnl_cum"] += pnl_day
            pos["last"] = {"date": today, "spot": spot, "value": m["value"], "delta": m["delta"], "gamma": m["gamma"],
                           "theta_day": m["theta_day"], "legs": [{k: lg.get(k) for k in ("type", "iv", "vega_pt")}
                                                                for lg in m["legs"]]}
            for lg, ml in zip(pos["legs"], m["legs"]):
                if lg["type"] != "U":
                    lg["iv_last"] = ml.get("iv")
    else:
        cond = spec.get("when", {})
        ivrv = volsnap.get("iv_rv")
        ok, why = True, None
        if "iv_rv_min" in cond and not (ivrv and ivrv >= cond["iv_rv_min"]):
            ok, why = False, f"iv_rv {ivrv} < {cond['iv_rv_min']}"
        if "iv_rv_max" in cond and not (ivrv and ivrv <= cond["iv_rv_max"]):
            ok, why = False, f"iv_rv {ivrv} > {cond['iv_rv_max']}"
        legs = pick_legs(spec, chain, spot) if ok else None
        if ok and legs is None:
            ok, why = False, "no expiry/strikes fit (chain sparse)"
        earn_in = None
        if legs and earnings:
            exp = max(pd.Timestamp(lg["expiry"]) for lg in legs if lg["type"] != "U")
            earn_in = any(pd.Timestamp(today) <= pd.Timestamp(d) <= exp for d in earnings)
        trig.append({"cell": cell["id"], "bar": today, "symbol": cell["symbols"][0], "kind": "entry",
                     "taken_raw": ok, "taken_desk": ok, "skip_reason": why, "earnings_in_trade": earn_in,
                     "legs": legs, "context": volsnap, **prov, "spec_sha": cell["spec_sha"]})
        if ok:
            pos = {"legs": legs, "opened": today}
            m = mark(pos, chain, spot, r)
            fill_v = 0.0
            for lg in m["legs"]:
                if lg["type"] == "U":
                    fill_v += lg["value"]
                    continue
                n = lg["qty"] * MULT
                px = lg.get("ask") if n > 0 else lg.get("bid")
                if px is None or not np.isfinite(px) or px <= 0:
                    px = lg["price"]
                fill_v += n * px
            n_contracts = sum(abs(lg["qty"]) for lg in legs if lg["type"] != "U")
            comm = n_contracts * costs.OPTION_COMMISSION_OPEN
            premium = sum(lg["value"] for lg in m["legs"] if lg["type"] != "U")
            cap = capital_base(name, legs, spot, premium)
            pos.update({"entry_value": fill_v, "commission": comm, "premium": premium, "capital": cap,
                        "entry_spot": spot,
                        "last": {"date": today, "spot": spot, "value": m["value"], "delta": m["delta"],
                                 "gamma": m["gamma"], "theta_day": m["theta_day"],
                                 "legs": [{k: lg.get(k) for k in ("type", "iv", "vega_pt")} for lg in m["legs"]]}})
            for lg, ml in zip(pos["legs"], m["legs"]):
                if lg["type"] != "U":
                    lg["iv_last"] = ml.get("iv")
            pnl_day = m["value"] - fill_v - comm
            ret_day = pnl_day / cap
            st["pnl_cum"] += pnl_day
            st["trades"] += 1
            st["pos"] = pos
            fills.append({"cell": cell["id"], "bar": today, "action": "open", "value": fill_v, "mid_value": m["value"],
                          "slippage": m["value"] - fill_v, "commission": comm, "premium_mid": premium, "capital": cap,
                          "legs": legs, "cost_source": costs.OPTION_COST_SOURCE, **prov})
            grk.append({"cell": cell["id"], "bar": today, "underlying": cell["symbols"][0], "spot": spot,
                        "leg": "TOTAL", **{k: m[k] for k in ("value", "delta", "gamma", "theta_day", "vega_pt")},
                        "pnl_day": pnl_day, "attribution": {"entry_cost": pnl_day}, "trade_pnl": pnl_day, **prov})
    sig = {"cell": cell["id"], "bar": today, "symbols": cell["symbols"], "role": cell["role"],
           "strategy": name, "group": "options", "in_trade": st["pos"] is not None, "pnl_day": pnl_day,
           "ret_raw": ret_day, "ret_desk": ret_day, "pnl_cum": st["pnl_cum"], "trades_raw": st["trades"],
           "trades_desk": st["trades"], "attribution": attr, "context": volsnap, "spec_sha": cell["spec_sha"], **prov}
    st["last_bar"] = today
    return st, sig, trig, fills, grk
