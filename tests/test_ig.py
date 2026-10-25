"""IG client (mocked HTTP, no network): price routing, allowance-safe caching, paging, dealing rules."""
import importlib

import pandas as pd
import pytest


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers = status, body or {}, headers or {}
        self.text = str(body)

    def json(self):
        return self._b


@pytest.fixture
def ig_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLAB_CACHE_DIR", str(tmp_path / "cache"))
    for k, v in {"IG_API_KEY": "k", "IG_USERNAME": "u", "IG_PASSWORD": "p", "IG_ACC_TYPE": "DEMO"}.items():
        monkeypatch.setenv(k, v)
    import mlab.cache
    import mlab.config
    importlib.reload(mlab.config)
    importlib.reload(mlab.cache)
    yield tmp_path


def _wire(monkeypatch, handler):
    """Patch requests.Session.request so every IG instance (also ones built inside get_prices) is mocked."""
    import requests
    calls = []

    def req(self, method, url, **kw):
        calls.append((method, url, kw))
        if url.endswith("/session") and method == "POST" and "_method" not in kw["headers"]:
            return FakeResp(200, {"currentAccountId": "ABC", "accounts": [{"accountId": "ABC"}]},
                            {"CST": "c", "X-SECURITY-TOKEN": "x"})
        return handler(method, url, kw)
    monkeypatch.setattr(requests.Session, "request", req)
    return calls


DAYS = pd.date_range("2026-01-05", periods=30, freq="B")


def _prices_handler(served):
    def bar(t, px):
        p = {"bid": px - 0.5, "ask": px + 0.5, "lastTraded": None}
        return {"snapshotTimeUTC": t, "openPrice": p, "highPrice": p, "lowPrice": p, "closePrice": p, "lastTradedVolume": 1}

    def handler(m, u, kw):
        if "/prices/" in u:
            frm, to = pd.Timestamp(kw["params"]["from"]), pd.Timestamp(kw["params"]["to"])
            rows = [bar(d.strftime("%Y-%m-%dT%H:%M:%S"), 100 + i) for i, d in enumerate(DAYS) if frm <= d <= to]
            served.append(len(rows))
            return FakeResp(200, {"prices": rows, "metadata": {"pageData": {"totalPages": 1}}})
        return FakeResp(200, {})
    return handler


def test_get_prices_routes_ig_with_aware_timestamps(ig_env, monkeypatch):
    served = []
    _wire(monkeypatch, _prices_handler(served))
    from mlab.data import get_prices
    df = get_prices("ig:IX.D.FTSE.DAILY.IP", start="2026-01-05", end="2026-01-16")
    assert len(df) == 10 and df.attrs["source"].startswith("IG DEMO")


def test_ig_prices_fetch_only_head_gap(ig_env, monkeypatch):
    served = []
    _wire(monkeypatch, _prices_handler(served))
    from mlab.providers.ig import IG
    ig = IG()
    ig.prices("IX.D.FTSE.DAILY.IP", "1d", start="2026-01-26", end="2026-02-13")  # 15 bars cached
    df = ig.prices("IX.D.FTSE.DAILY.IP", "1d", start="2026-01-05", end="2026-02-13")
    assert served[0] == 15
    assert served[1] <= 16  # the 15 missing head bars (+ the boundary bar), not the whole range again
    assert len(served) == 2 and len(df) == 30


def test_ig_activity_follows_paging(ig_env, monkeypatch):
    def handler(m, u, kw):
        if u.endswith("/history/activity"):
            if kw["params"].get("to"):
                return FakeResp(200, {"activities": [{"dealId": "B"}], "metadata": {"paging": {"next": None}}})
            return FakeResp(200, {"activities": [{"dealId": "A"}], "metadata": {"paging": {
                "next": "/history/activity?version=3&from=2026-01-01T00:00:00&to=2026-01-02T00:00:00&detailed=true&pageSize=500"}}})
        return FakeResp(200, {})
    calls = _wire(monkeypatch, handler)
    from mlab.providers.ig import IG
    df = IG().activity(30)
    assert list(df["dealId"]) == ["A", "B"]
    assert all("version" not in c[2].get("params", {}) for c in calls if c[1].endswith("/history/activity"))


def test_ig_rules_flatten_market_details(ig_env, monkeypatch):
    details = {"instrument": {"epic": "IX.D.FTSE.DAILY.IP", "name": "FTSE 100", "type": "INDICES", "marketId": "FT100",
                              "valueOfOnePip": "1.00", "contractSize": "1", "marginFactor": 5, "marginFactorUnit": "PERCENTAGE",
                              "currencies": [{"code": "GBP", "isDefault": True}],
                              "openingHours": {"marketTimes": [{"openTime": "00:00", "closeTime": "21:00"}]}},
               "dealingRules": {"minDealSize": {"unit": "POINTS", "value": 0.5},
                                "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 6}},
               "snapshot": {"marketStatus": "TRADEABLE", "bid": 8000.0, "offer": 8001.0, "scalingFactor": 1}}
    _wire(monkeypatch, lambda m, u, kw: FakeResp(200, details) if "/markets/" in u else FakeResp(200, {}))
    from mlab.providers.ig import IG
    r = IG().rules("IX.D.FTSE.DAILY.IP")
    assert r["min_deal_size"] == "0.5 POINTS" and r["min_stop_or_limit_distance"] == "6 POINTS"
    assert r["currency"] == "GBP" and r["margin_factor"] == "5 PERCENTAGE" and r["hours_utc"] == "00:00-21:00"


def test_new_endpoints_stay_read_only(ig_env, monkeypatch):
    calls = _wire(monkeypatch, lambda m, u, kw: FakeResp(200, {"clientSentiments": [], "nodes": []}))
    from mlab.providers.ig import IG
    ig = IG()
    ig.related_sentiment("FT100"), ig.navigation(), ig.navigation("123"), ig.allowance()
    assert {c[0] for c in calls if not c[1].endswith("/session")} == {"GET"}
