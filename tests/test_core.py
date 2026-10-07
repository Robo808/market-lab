"""Offline tests for the core: cache, TA, risk, journal, IG client (mocked HTTP), SEC parsing, routing."""
import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    df = pd.DataFrame({"open": c * (1 + rng.normal(0, 0.002, n)), "close": c}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.004
    df["low"] = df[["open", "close"]].min(axis=1) * 0.996
    df["volume"] = 1e6
    return df[["open", "high", "low", "close", "volume"]]


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLAB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MLAB_JOURNAL_DIR", str(tmp_path / "journal"))
    import mlab.cache, mlab.config, mlab.journal
    importlib.reload(mlab.config)
    importlib.reload(mlab.cache)
    importlib.reload(mlab.journal)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(mlab.config)
    importlib.reload(mlab.cache)
    importlib.reload(mlab.journal)


def test_cache_merge_and_covers(tmp_env, ohlcv):
    from mlab import cache
    cache.save("t", "X", "1d", ohlcv.iloc[:300])
    merged = cache.save("t", "X", "1d", ohlcv.iloc[250:])
    assert len(merged) == len(ohlcv) and merged.index.is_unique
    assert cache.covers(merged, ohlcv.index[0], ohlcv.index[-1], pd.Timedelta(days=1))
    assert not cache.covers(merged, ohlcv.index[0] - pd.Timedelta(days=30), ohlcv.index[-1], pd.Timedelta(days=1))
    assert cache.stats()["files"] == 1


def test_ta_core(ohlcv):
    from mlab import ta
    r = ta.rsi(ohlcv["close"]).dropna()
    assert r.between(0, 100).all()
    m = ta.macd(ohlcv["close"])
    assert np.allclose((m["macd"] - m["signal"]).dropna(), m["hist"].dropna())
    snap = ta.snapshot(ohlcv)
    assert snap["regime_trend"] in {"up", "down", "mixed"} and isinstance(snap["signals"], list)
    sr = ta.support_resistance(ohlcv)
    assert set(sr["kind"]) <= {"support", "resistance"}
    st = ta.supertrend(ohlcv).dropna()
    assert len(st) > 500 and set(st["trend"].unique()) == {1.0, -1.0}
    from mlab import ta_catalog
    ref = ta_catalog.supertrend(ohlcv)
    assert np.allclose(st["supertrend"], ref.loc[st.index, "supertrend"])
    assert (st["trend"] == ref.loc[st.index, "trend"]).all()


def test_risk_card():
    from mlab.risk import TradeCard, stake_per_point
    z = stake_per_point(20000, 1, 7410, 7330)
    assert z["risk_amount"] == 200 and z["stake_per_point"] == 2.5
    fx = stake_per_point(10000, 1, 1.1000, 1.0950, point_size=0.0001)
    assert fx["stop_distance_points"] == 50 and fx["stake_per_point"] == 2.0
    c = TradeCard("X", "SHORT", (110, 110), 115, [100, 95])
    assert c.rr == [2.0, 3.0]
    assert "SHORT X" in c.line() and "R:R 2/3" in c.line()


def test_journal_lifecycle(tmp_env):
    from mlab import journal
    from mlab.risk import TradeCard
    c = TradeCard("FTSE", "LONG", (100, 102), 96, [110], lens="Soros")
    r = journal.add(c.to_dict())
    journal.update(r["id"], status="open", fill=101)
    out = journal.update(r["id"], status="closed", exit_price=111)
    assert out["r_multiple"] == 2.0
    rv = journal.review()
    assert rv["closed_trades"] == 1 and rv["expectancy_R"] == 2.0
    assert (tmp_env / "journal" / "JOURNAL.md").exists()


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers = status, body or {}, headers or {}
        self.text = str(body)

    def json(self):
        return self._b


def _ig(monkeypatch, handler):
    monkeypatch.setenv("IG_API_KEY", "k")
    monkeypatch.setenv("IG_USERNAME", "u")
    monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("IG_ACC_TYPE", "DEMO")
    from mlab.providers.ig import IG
    ig = IG()
    calls = []

    def req(method, url, **kw):
        calls.append((method, url, kw))
        return handler(method, url, kw)
    monkeypatch.setattr(ig.s, "request", req)
    return ig, calls


def _login_ok(method, url, kw):
    if url.endswith("/session") and method == "POST" and "_method" not in kw["headers"]:
        return FakeResp(200, {"currentAccountId": "ABC", "lightstreamerEndpoint": "https://ls", "accounts": [{"accountId": "ABC"}]},
                        {"CST": "c", "X-SECURITY-TOKEN": "x"})
    return None


def test_ig_read_only_guard(monkeypatch):
    from mlab.providers.ig import ReadOnlyViolation
    ig, calls = _ig(monkeypatch, lambda m, u, k: _login_ok(m, u, k) or FakeResp(200, {}))
    for m, p in [("POST", "/positions/otc"), ("PUT", "/positions/otc/DEAL1"), ("DELETE", "/workingorders/otc/1"),
                 ("POST", "/workingorders/otc"), ("POST", "/watchlists")]:
        with pytest.raises(ReadOnlyViolation):
            ig._request(m, p)
    assert calls == []  # nothing left the process
    assert ig.cfg.base_url.startswith("https://demo-api.ig.com")


def test_ig_prices_parse_and_tail_cache(tmp_env, monkeypatch):
    def bar(t, px):
        p = {"bid": px - 0.5, "ask": px + 0.5, "lastTraded": None}
        return {"snapshotTime": t.replace("-", "/").replace("T", " "), "snapshotTimeUTC": t, "openPrice": p, "highPrice": p,
                "lowPrice": p, "closePrice": p, "lastTradedVolume": 10}
    days = pd.date_range("2026-01-05", periods=10, freq="B")
    served = []

    def handler(m, u, kw):
        lo = _login_ok(m, u, kw)
        if lo:
            return lo
        if "/prices/" in u:
            frm = pd.Timestamp(kw["params"]["from"])
            rows = [bar(d.strftime("%Y-%m-%dT%H:%M:%S"), 100 + i) for i, d in enumerate(days) if d >= frm]
            served.append(len(rows))
            return FakeResp(200, {"prices": rows, "metadata": {"allowance": {"remainingAllowance": 9990},
                                                               "pageData": {"pageNumber": 1, "totalPages": 1}}})
        return FakeResp(200, {})
    ig, calls = _ig(monkeypatch, handler)
    df = ig.prices("IX.D.FTSE.DAILY.IP", "1d", start="2026-01-05", end="2026-01-16")
    assert len(df) == 10 and df["close"].iloc[0] == 100 and df["spread"].iloc[0] == 1.0
    assert str(df.index.tz) == "UTC"
    ig.prices("IX.D.FTSE.DAILY.IP", "1d", start="2026-01-05", end="2026-01-19")
    assert served[1] < served[0]  # second call only asked for the tail
    assert ig.last_allowance["remainingAllowance"] == 9990


def test_ig_missing_creds(monkeypatch):
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_DEMO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from mlab.providers.ig import IG, IGError
    with pytest.raises(IGError, match="IG_API_KEY"):
        IG()


def test_sec_annual_financials(monkeypatch):
    from mlab.providers import sec

    def fy(val, y, start=True):
        d = {"val": val, "end": f"{y}-12-31", "form": "10-K", "filed": f"{y + 1}-02-01", "fp": "FY"}
        if start:
            d["start"] = f"{y}-01-01"
        return d
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fy(100 + 10 * i, 2015 + i) for i in range(10)]}},
        "GrossProfit": {"units": {"USD": [fy(60 + 6 * i, 2015 + i) for i in range(10)]}},
        "OperatingIncomeLoss": {"units": {"USD": [fy(30 + 3 * i, 2015 + i) for i in range(10)]}},
        "NetIncomeLoss": {"units": {"USD": [fy(20 + 2 * i, 2015 + i) for i in range(10)]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [fy(28 + 2 * i, 2015 + i) for i in range(10)]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [fy(5, 2015 + i) for i in range(10)]}},
        "StockholdersEquity": {"units": {"USD": [fy(100, 2015 + i, False) for i in range(10)]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [fy(50, 2015 + i, False) for i in range(10)]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [fy(20, 2015 + i, False) for i in range(10)]}},
    }}}
    monkeypatch.setattr(sec, "company_facts", lambda t: facts)
    fin = sec.annual_financials("X")
    last = fin.iloc[-1]
    assert last["revenue"] == 190 and last["fcf"] == 41 and abs(last["gross_margin"] - 114 / 190) < 1e-9
    assert last["net_debt"] == 30 and abs(last["roic_pre_tax"] - 57 / 130) < 1e-9


def test_routing():
    from mlab.data import route
    assert route("IX.D.FTSE.DAILY.IP") == ("ig", "IX.D.FTSE.DAILY.IP")
    assert route("crypto:BTC") == ("crypto", "BTC")
    assert route("AAPL") == ("yahoo", "AAPL")
    from mlab.providers.stooq import to_stooq
    assert to_stooq("AAPL") == "aapl.us" and to_stooq("EURUSD=X") == "eurusd" and to_stooq("^GSPC") == "^spx"
    from mlab.providers.crypto import to_pair
    assert to_pair("BTC") == "BTCUSDT" and to_pair("eth-usd") == "ETHUSDT"


def test_backtest_no_lookahead(ohlcv):
    from mlab import backtest
    pos = pd.Series(1, index=ohlcv.index)
    res = backtest.run(ohlcv, pos)
    bh = res["buy_hold"]["total_return"]
    # a position decided on bar t earns bar t+1's return, so always-long == buy & hold
    assert abs(res["strategy"]["total_return"] - bh) < 1e-9
    # a signal that is only on at the last bar must earn nothing (no future bar to trade)
    last_only = pd.Series(0, index=ohlcv.index); last_only.iloc[-1] = 1
    assert backtest.run(ohlcv, last_only)["strategy"]["total_return"] == 0


def test_data_dir_override(tmp_path, monkeypatch):
    import mlab.config, mlab.journal
    monkeypatch.setenv("MLAB_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MLAB_JOURNAL_DIR", raising=False)
    monkeypatch.delenv("MLAB_CACHE_DIR", raising=False)
    try:
        importlib.reload(mlab.config)
        importlib.reload(mlab.journal)
        assert mlab.config.DATA_DIR == tmp_path
        assert mlab.config.CACHE_DIR == tmp_path / "data" / "cache"
        assert mlab.journal.JOURNAL_DIR == tmp_path / "journal"
    finally:
        monkeypatch.undo()
        importlib.reload(mlab.config)
        importlib.reload(mlab.journal)
