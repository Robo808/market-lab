import math

import numpy as np
import pandas as pd
import pytest

from mlab import options as op

S, K, T, R, Q, V = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25


def test_put_call_parity():
    c, p = op.bs_price(S, K, T, R, V, Q, "C"), op.bs_price(S, K, T, R, V, Q, "P")
    assert c - p == pytest.approx(S * math.exp(-Q * T) - K * math.exp(-R * T), abs=1e-10)


def test_known_value():  # Hull example: S=42 K=40 r=10% vol=20% T=0.5 -> call 4.76, put 0.81
    assert op.bs_price(42, 40, 0.5, 0.1, 0.2, 0, "C") == pytest.approx(4.759, abs=1e-3)
    assert op.bs_price(42, 40, 0.5, 0.1, 0.2, 0, "P") == pytest.approx(0.8086, abs=1e-3)


@pytest.mark.parametrize("kind", ["C", "P"])
def test_greeks_vs_finite_differences(kind):
    g = op.greeks(S, K, T, R, V, Q, kind)
    f = lambda s=S, t=T, r=R, v=V: op.bs_price(s, K, t, r, v, Q, kind)  # noqa: E731
    d = lambda s=S, t=T, v=V: op.greeks(s, K, t, R, v, Q, kind)["delta"]  # noqa: E731
    h = 1e-4
    assert g["delta"] == pytest.approx((f(s=S + h) - f(s=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((f(s=S + 1e-2) - 2 * f() + f(s=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(v=V + h) - f(v=V - h)) / (2 * h), rel=1e-5)
    assert g["theta"] == pytest.approx(-(f(t=T + h) - f(t=T - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-5)
    assert g["vanna"] == pytest.approx((d(v=V + h) - d(v=V - h)) / (2 * h), rel=1e-4)
    assert g["charm"] == pytest.approx(-(d(t=T + h) - d(t=T - h)) / (2 * h), rel=1e-4)


def test_vectorised_and_expired():
    px = op.bs_price(np.array([90, 100, 110.0]), 100, 0.0, 0.0, 0.2, 0, "C")
    assert list(px) == [0, 0, 10]
    g = op.greeks(110, 100, 0, 0, 0.2, 0, "P")
    assert g["delta"] == 0 and g["gamma"] == 0


@pytest.mark.parametrize("kind,k,t,v", [("C", 80, 0.1, 0.6), ("P", 120, 2.0, 0.15), ("C", 150, 0.05, 1.2), ("P", 100, 1 / 365, 0.3)])
def test_iv_round_trip(kind, k, t, v):
    px = op.bs_price(S, k, t, R, v, Q, kind)
    assert op.implied_vol(px, S, k, t, R, Q, kind) == pytest.approx(v, abs=1e-6)


def test_iv_bad_quotes_are_nan():
    assert math.isnan(op.implied_vol(0.0, S, K, T, R, Q, "C"))
    assert math.isnan(op.implied_vol(1000, S, K, T, R, Q, "C"))  # above the underlying
    assert math.isnan(op.implied_vol(0.5, 120, 100, T, 0, 0, "C"))  # below intrinsic
    assert math.isnan(op.implied_vol(float("nan"), S, K, T, R, Q, "C"))
    v = op.implied_vols([op.bs_price(S, K, T, R, 0.3, Q, "P"), -1], S, K, T, R, Q, ["P", "P"])
    assert v[0] == pytest.approx(0.3, abs=1e-6) and math.isnan(v[1])


# ---- hand-built chain -----------------------------------------------------------------
NOW = pd.Timestamp("2026-01-02 15:00", tz="UTC")


def make_chain(spot=100.0, vol=0.3, oi=None, vol_traded=None, expiries=("2026-01-30", "2026-03-20")):
    rows = []
    for e in expiries:
        Te = ((pd.Timestamp(e) + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC") - NOW).total_seconds() / (365 * 86400)
        for k in range(80, 125, 5):
            for typ in "CP":
                iv = vol + (0.002 * (100 - k) if typ == "P" else 0.0)  # mild put skew
                mid = op.bs_price(spot, k, Te, 0.0, iv, 0, typ)
                rows.append({"expiry": e, "type": typ, "strike": float(k), "bid": mid * 0.99, "ask": mid * 1.01, "lastPrice": mid,
                             "volume": (vol_traded or {}).get((typ, k), 10), "openInterest": (oi or {}).get((typ, k), 100),
                             "impliedVolatility": 1e-5})
    return op.normalize_chain(pd.DataFrame(rows), spot, 0.0, 0.0, now=NOW)


def test_normalize_recomputes_junk_iv():
    ch = make_chain()
    assert (ch["iv_src"] == "mid").all()
    atm = ch[(ch["strike"] == 100) & (ch["type"] == "C")]
    assert atm["iv"].iloc[0] == pytest.approx(0.3, abs=1e-6)
    assert {"expiry", "dte", "type", "strike", "bid", "ask", "mid", "last", "volume", "oi", "iv", "moneyness", "delta", "gamma"} <= set(ch)
    term = op.atm_term_structure(ch)
    assert term["atm_iv"].iloc[0] == pytest.approx(0.3, abs=1e-6)
    assert op.atm_iv_at(term, 30) == pytest.approx(0.3, abs=1e-6)


def test_skew_sign_with_put_skew():
    sk = op.skew(make_chain())
    assert (sk["rr25"] < 0).all()  # puts richer than calls


def test_max_pain_hand_built():
    oi = {(t, k): 0 for t in "CP" for k in range(80, 125, 5)}
    oi.update({("C", 110): 1000, ("P", 90): 1000, ("C", 95): 50, ("P", 105): 50})
    mp = op.max_pain(make_chain(oi=oi))
    # payout(P) = 1000*max(P-110,0)+1000*max(90-P,0)+50*max(P-95,0)+50*max(105-P,0): flat 1000 on [95,105], higher outside
    assert 95 <= mp["max_pain"].iloc[0] <= 105
    oi3 = {(t, k): 0 for t in "CP" for k in range(80, 125, 5)}
    oi3.update({("C", 85): 300, ("P", 100): 100, ("P", 120): 400})
    pay3 = {P: 300 * max(P - 85, 0) + 100 * max(100 - P, 0) + 400 * max(120 - P, 0) for P in range(80, 125, 5)}
    assert op.max_pain(make_chain(oi=oi3))["max_pain"].iloc[0] == min(pay3, key=pay3.get)


def test_gex_sign_conventions_and_flip():
    zero = {(t, k): 0 for t in "CP" for k in range(80, 125, 5)}
    calls = make_chain(oi={**zero, ("C", 100): 1000})
    assert op.gex(calls)["total"] > 0
    assert op.gex(calls, convention="short_all")["total"] < 0
    puts = make_chain(oi={**zero, ("P", 100): 1000})
    assert op.gex(puts)["total"] < 0
    assert op.gex(puts, convention="short_calls_long_puts")["total"] > 0
    # puts below, calls above: dealer gamma negative low, positive high -> flip between the walls
    both = make_chain(oi={**zero, ("P", 90): 1000, ("C", 110): 1000}, expiries=("2026-03-20",))
    g = op.gex(both)
    assert 90 < g["flip"] < 110
    assert g["by_strike"].loc[90, "put_gex"] < 0 < g["by_strike"].loc[110, "call_gex"]
    assert g["total"] == pytest.approx(g["by_strike"]["net_gex"].sum())
    # magnitude: gamma*oi*100*S^2*1%
    one = calls[(calls["type"] == "C") & (calls["strike"] == 100)]
    assert op.gex(calls)["total"] == pytest.approx((one["gamma"] * 1000 * 100 * 100 ** 2 * 0.01).sum())


def test_zero_gamma_level_interpolates():
    prof = pd.Series([-2.0, -1.0, 1.0, 3.0], index=[90.0, 95.0, 100.0, 105.0])
    assert op.zero_gamma_level(prof, 100) == pytest.approx(97.5)
    assert math.isnan(op.zero_gamma_level(pd.Series([1.0, 2.0], index=[1.0, 2.0]), 1.5))


def test_expected_move_from_straddle():
    ch = make_chain()
    em = op.expected_move(ch)
    row = ch[(ch["strike"] == 100) & (ch["expiry"] == pd.Timestamp("2026-01-30"))]
    straddle = row["mid"].sum()
    assert em["atm_strike"].iloc[0] == 100
    assert em["straddle"].iloc[0] == pytest.approx(straddle)
    assert em["move_pct"].iloc[0] == pytest.approx(straddle / 100)
    assert em["straddle"].iloc[0] == pytest.approx(0.8 * em["iv_1sd"].iloc[0], rel=0.03)  # sqrt(2/pi) rule


def test_pcr_and_unusual():
    ch = make_chain(vol_traded={("P", 90): 5000}, oi={("P", 90): 200})
    pcr = op.put_call_ratios(ch)
    assert pcr.loc["ALL", "pcr_vol"] > 1
    ua = op.unusual_activity(ch, premium=1e12)
    assert len(ua) == 2 and (ua["strike"] == 90).all() and ua["flags"].str.contains("vol>oi").all()


def test_iv_rank_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "IV_DIR", tmp_path)
    for i, v in enumerate([0.2, 0.3, 0.4, 0.25]):
        h = op.record_atm_iv("TEST", v, 100, asof=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i))
    op.record_atm_iv("TEST", 0.35, 100, asof="2026-01-04 20:00")  # same day overwrites
    h = op.load_iv_history("TEST")
    assert len(h) == 4 and h["atm_iv30"].iloc[-1] == 0.35
    r = op.iv_rank(h["atm_iv30"], 0.3)
    assert r["iv_rank"] == pytest.approx(0.5) and r["iv_percentile"] == pytest.approx(0.25)


def test_realized_vol_gbm():
    rng = np.random.default_rng(0)
    n, sig, steps = 3000, 0.2, 200
    sd = sig / math.sqrt(252)
    gap = rng.normal(0, sd * math.sqrt(0.2), n)  # 20% of variance overnight
    path = np.cumsum(rng.normal(0, sd * math.sqrt(0.8 / steps), (n, steps)), axis=1)
    lo_ = np.minimum(path.min(1), 0)
    hi_ = np.maximum(path.max(1), 0)
    lvl = np.cumsum(gap + path[:, -1]) - path[:, -1]  # log open of each day
    df = pd.DataFrame({"open": np.exp(lvl), "close": np.exp(lvl + path[:, -1]), "high": np.exp(lvl + hi_), "low": np.exp(lvl + lo_)}) * 100
    assert op.realized_vol(df["close"], 2000).iloc[-1] == pytest.approx(sig, rel=0.06)
    assert op.yang_zhang(df, 2000).iloc[-1] == pytest.approx(sig, rel=0.15)
    out = op.iv_vs_rv(0.3, df)
    assert out["vrp_20"] == pytest.approx(0.3 - out["rv20"])


# ---- payoff ------------------------------------------------------------------------------
def test_parse_leg():
    lg = op.parse_leg("C 100 30d +1 @2.5")
    assert (lg.kind, lg.strike, lg.days, lg.qty, lg.premium) == ("C", 100, 30, 1, 2.5)
    lg = op.parse_leg("P 95 2026-02-01 -2 iv=28%", today="2026-01-02")
    assert (lg.kind, lg.days, lg.qty, lg.premium, lg.iv) == ("P", 30, -2, None, pytest.approx(0.28))
    assert op.parse_leg("U -1 @101").kind == "U"


def test_payoff_breakevens_and_extremes():
    long_call = op.payoff([op.parse_leg("C 100 30d +1 @2.5")], 100, 0.2)
    assert long_call["breakevens"] == [pytest.approx(102.5)]
    assert long_call["max_profit"] == math.inf and long_call["max_loss"] == pytest.approx(-2.5)
    straddle = op.payoff([op.parse_leg("C 100 30d +1 @3"), op.parse_leg("P 100 30d +1 @2")], 100, 0.2)
    assert straddle["breakevens"] == [pytest.approx(95), pytest.approx(105)]
    spread = op.payoff([op.parse_leg("C 100 30d +1 @5"), op.parse_leg("C 110 30d -1 @1")], 100, 0.2, multiplier=10)
    assert spread["breakevens"] == [pytest.approx(104)]
    assert spread["max_profit"] == pytest.approx(60) and spread["max_loss"] == pytest.approx(-40)
    assert spread["net_premium"] == pytest.approx(40)
    short_put = op.payoff([op.parse_leg("P 100 30d -1 @4")], 100, 0.2)
    assert short_put["max_loss"] == pytest.approx(-96) and short_put["max_profit"] == pytest.approx(4)
    covered = op.payoff([op.parse_leg("U +1 @100"), op.parse_leg("C 105 30d -1 @2")], 100, 0.2)
    assert covered["breakevens"] == [pytest.approx(98)] and covered["max_profit"] == pytest.approx(7)
    assert covered["greeks"]["delta"] < 1


def test_payoff_greeks_and_bs_premium():
    res = op.payoff([op.parse_leg("C 100 30d +1"), op.parse_leg("P 100 30d +1")], 100, 0.2, r=0.0)
    g = op.greeks(100, 100, 30 / 365, 0, 0.2, 0, "C")
    assert res["greeks"]["gamma"] == pytest.approx(2 * g["gamma"])
    assert res["greeks"]["vega_1pt"] == pytest.approx(2 * g["vega"] / 100)
    assert abs(res["greeks"]["delta"]) < 0.05
    assert res["table"].loc[100.0, "pnl_today"] == pytest.approx(0, abs=1e-9)
