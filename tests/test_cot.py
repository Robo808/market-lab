"""Offline tests for mlab.cot: fixture Socrata rows -> nets, COT index bounds/extremes, divergence logic."""
import numpy as np
import pandas as pd
import pytest

from mlab import cache, cot


def _rows(report: str, n: int = 200, spec_net=None, code="088691", name="GOLD - COMMODITY EXCHANGE INC.", end="2026-09-29"):
    """Socrata-shaped rows (strings, ISO timestamps, CFTC field-name quirks), newest first like $order DESC."""
    dates = pd.date_range(end=end, periods=n, freq="W-TUE")
    spec_net = np.asarray(spec_net if spec_net is not None else np.linspace(-50_000, 150_000, n))
    rows = []
    for d, net in zip(dates, spec_net):
        oi = 500_000
        r = {"market_and_exchange_names": name, "report_date_as_yyyy_mm_dd": f"{d:%Y-%m-%d}T00:00:00.000",
             "cftc_contract_market_code": code, "open_interest_all": str(oi)}
        sl, ss = 200_000 + max(net, 0), 200_000 - min(net, 0)
        if report == "legacy":
            r.update(noncomm_positions_long_all=str(int(sl)), noncomm_positions_short_all=str(int(ss)), noncomm_postions_spread_all="1000",
                     comm_positions_long_all=str(int(ss)), comm_positions_short_all=str(int(sl)),
                     nonrept_positions_long_all="30000", nonrept_positions_short_all="32000")
        elif report == "disagg":
            r.update(m_money_positions_long_all=str(int(sl)), m_money_positions_short_all=str(int(ss)),
                     m_money_positions_long_old="1", m_money_positions_long_other="2",
                     prod_merc_positions_long="90000", prod_merc_positions_short="210000",
                     swap_positions_long_all="40000", swap__positions_short_all="120000", swap__positions_spread_all="5000",
                     other_rept_positions_long="20000", other_rept_positions_short="15000",
                     nonrept_positions_long_all="30000", nonrept_positions_short_all="25000")
        else:
            r.update(lev_money_positions_long=str(int(sl)), lev_money_positions_short=str(int(ss)),
                     asset_mgr_positions_long="250000", asset_mgr_positions_short="60000",
                     dealer_positions_long_all="40000", dealer_positions_short_all="180000",
                     other_rept_positions_long="10000", other_rept_positions_short="12000",
                     nonrept_positions_long_all="20000", nonrept_positions_short_all="18000")
        rows.append(r)
    return rows[::-1]


@pytest.mark.parametrize("report,cls", [("legacy", "noncomm"), ("disagg", "managed_money"), ("tff", "lev_funds")])
def test_parse_nets(report, cls):
    df = cot.parse(_rows(report, 10, spec_net=[1000 * i for i in range(10)]), report)
    assert df.index.is_monotonic_increasing and len(df) == 10
    assert list(df[f"{cls}_net"]) == [1000.0 * i for i in range(10)]
    assert (df["open_interest"] == 500_000).all()
    if report == "disagg":
        assert df["swap_net"].iloc[-1] == 40_000 - 120_000  # double-underscore field picked up
        assert df["producer_net"].iloc[-1] == -120_000
        assert df["managed_money_long"].iloc[-1] != 1  # *_old crop-year column not used
    if report == "tff":
        assert df["asset_mgr_net"].iloc[-1] == 190_000 and df["dealer_net"].iloc[-1] == -140_000
    if report == "legacy":
        assert (df["comm_net"] == -df["noncomm_net"]).all()


def test_fuzzy_field_fallback():
    rows = _rows("disagg", 5)
    for r in rows:
        r["m_money_positions_long_all_x"] = r.pop("m_money_positions_long_all")  # unknown spelling
        r["m_money_positions_long"] = r["m_money_positions_long_all_x"]          # alternate known spelling
    df = cot.parse(rows, "disagg")
    assert "managed_money_net" in df


def test_parse_picks_largest_contract_when_pattern_matches_many():
    a = _rows("legacy", 5, code="088691")
    b = _rows("legacy", 5, code="088695", name="GOLD MICRO")
    for r in b:
        r["open_interest_all"] = "1000"
    df = cot.parse(a + b, "legacy")
    assert len(df) == 5 and (df["open_interest"] == 500_000).all()


def test_parse_empty_raises():
    with pytest.raises(LookupError):
        cot.parse([], "legacy")


def test_cot_index_bounds_and_extremes():
    rng = np.random.default_rng(1)
    net = pd.Series(rng.normal(0, 50_000, 300).cumsum())
    idx = cot.cot_index(net, 156)
    v = idx.dropna()
    assert v.between(0, 100).all() and v.min() == 0 and v.max() > 50
    assert idx.iloc[:25].isna().all()
    # a new 3y high sits at 100, a new low at 0
    up = pd.Series(np.arange(200.0))
    assert cot.cot_index(up).iloc[-1] == 100
    assert cot.cot_index(-up).iloc[-1] == 0


def test_analytics_flags_and_changes():
    n = 200
    spec = np.r_[np.linspace(0, 100_000, n - 1), 160_000]  # final week: fresh 3y high -> extreme long
    df = cot.analytics(cot.parse(_rows("disagg", n, spec), "disagg"), "disagg")
    last = df.iloc[-1]
    assert last["managed_money_index"] == 100 and last["managed_money_flag"] == "extreme long"
    assert last["managed_money_chg"] == pytest.approx(160_000 - 100_000)
    assert last["managed_money_pct_oi"] == pytest.approx(160_000 / 500_000 * 100)
    assert last["managed_money_z"] > 2
    low = cot.analytics(cot.parse(_rows("tff", n, -spec), "tff"), "tff").iloc[-1]
    assert low["lev_funds_flag"] == "extreme short" and low["lev_funds_index"] == 0
    s = cot.summary(df, "disagg")
    assert "managed_money (spec)" in s.index and s.loc["managed_money (spec)", "flag"] == "extreme long"


def _weekly_close(dates, values):
    days = pd.bdate_range(dates[0] - pd.Timedelta(days=7), dates[-1] + pd.Timedelta(days=3))
    return pd.Series(np.interp(days.asi8, pd.DatetimeIndex(dates).asi8, values), index=days.tz_localize("UTC"))


def test_divergence_logic():
    n = 12
    df = cot.parse(_rows("legacy", n, spec_net=np.r_[np.full(6, 100_000), np.linspace(100_000, 40_000, 6)]), "legacy")
    close = _weekly_close(df.index, np.r_[np.full(6, 100.0), np.linspace(100, 115, 6)])  # price up, specs cut longs
    d = cot.divergence(df, close, "legacy", lookback=4)
    assert d["signal"].iloc[-1].startswith("bearish divergence")
    assert d["price_chg"].iloc[-1] > 0.02 and d["spec_chg_pp_oi"].iloc[-1] < -2
    # mirror image: price down while specs add -> bullish divergence
    df2 = cot.parse(_rows("legacy", n, spec_net=np.r_[np.full(6, 0), np.linspace(0, 60_000, 6)]), "legacy")
    d2 = cot.divergence(df2, _weekly_close(df2.index, np.r_[np.full(6, 100.0), np.linspace(100, 85, 6)]), "legacy")
    assert d2["signal"].iloc[-1].startswith("bullish divergence")
    # both up -> confirming; flat -> no signal
    d3 = cot.divergence(df2, _weekly_close(df2.index, np.r_[np.full(6, 100.0), np.linspace(100, 115, 6)]), "legacy")
    assert d3["signal"].iloc[-1].startswith("confirming: price up")
    d4 = cot.divergence(df2, _weekly_close(df2.index, np.full(12, 100.0)), "legacy")
    assert d4["signal"].iloc[-1] == ""


def test_resolve_and_markets():
    assert cot.resolve("GC=F") == "gold" and cot.resolve("6E") == "eur" and cot.resolve("es") == "sp500"
    assert cot.resolve("10y") == "ust10y" and cot.resolve("crude") == "wti" and cot.resolve("Live Cattle") == "cattle"
    assert cot.default_report("gold") == "disagg" and cot.default_report("eur") == "tff"
    with pytest.raises(KeyError):
        cot.resolve("unobtainium")
    need = {"sp500", "nasdaq100", "dow", "russell", "vix", "ust2y", "ust5y", "ust10y", "ust30y", "sofr", "eurodollar", "eur", "gbp",
            "jpy", "chf", "cad", "aud", "nzd", "mxn", "usd", "bitcoin", "ether", "gold", "silver", "copper", "platinum", "wti", "brent",
            "natgas", "gasoline", "heatingoil", "corn", "wheat", "soybeans", "sugar", "coffee", "cocoa", "cotton", "cattle"}
    assert need <= set(cot.MARKETS)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=2)

    def fake(dataset, where, limit=5000, order=None):
        calls.append((dataset, where))
        if "unknown" in where or "'067651'" in where:
            return []  # WTI code miss -> name fallback
        n = 200 if "upper(" not in where else 170
        spec = np.r_[np.linspace(0, 100_000, n - 1), 180_000]
        return _rows("disagg" if dataset == cot.DATASETS["disagg"] else "tff" if dataset == cot.DATASETS["tff"] else "legacy",
                     n, spec, end=f"{end:%Y-%m-%d}")
    monkeypatch.setattr(cot, "_socrata", fake)
    return calls


def test_history_caches_and_falls_back_to_name(offline):
    df = cot.history("gold", years=3)
    assert df.attrs["report"] == "disagg" and "managed_money_index" in df and len(df) > 100
    assert cache.load("cftc", "disagg_gold", "1wk") is not None
    n_calls = len(offline)
    cot.history("gold", years=3)  # fresh cache -> no new request
    assert len(offline) == n_calls
    w = cot.history("wti", years=3)
    assert any("upper(market_and_exchange_names)" in q for _, q in offline) and len(w)


def test_scan_extremes(offline):
    t = cot.scan_extremes(markets=["gold", "eur", "sp500"])
    assert set(t.index) == {"gold", "eur", "sp500"}
    assert (t["flag"] == "extreme long").all() and t["spec_index"].between(0, 100).all()
    assert t.loc["eur", "report"] == "tff" and not t.attrs["errors"]


def test_report_for_with_prices(offline):
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=900)
    prices = pd.DataFrame({"close": np.linspace(1500, 2600, len(dates))}, index=dates.tz_localize("UTC"))
    r = cot.report_for("gold", prices=prices)
    assert "divergence" in r and len(r["summary"]) == 5
    assert r["divergence"]["signal"].isin(["", "confirming: price up, specs adding", "bearish divergence: price up, specs cutting longs",
                                           "bullish divergence: price down, specs adding longs", "confirming: price down, specs selling"]).all()
