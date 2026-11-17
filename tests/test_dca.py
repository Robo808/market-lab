"""Scheduled-buy planner: fill detection, due dates, weekend rolls, storage round trip (offline)."""
import datetime as dt

import pandas as pd
import pytest

from mlab import dca


def _bars(rows):
    """rows: (date, low, close) -> daily OHLCV frame indexed in UTC like the price cache."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, _, _ in rows])
    low = [lo for _, lo, _ in rows]
    close = [c for _, _, c in rows]
    return pd.DataFrame({"open": close, "high": close, "low": low, "close": close, "volume": 1.0}, index=idx)


def _plan():
    return dca.Plan(name="t", symbol="TEST", start="2026-11-17", tranches=[
        dca.parse_tranche("1000:2026-11-18", 1),
        dca.parse_tranche("1000@95.00:2027-03-01", 2),
        dca.parse_tranche("1000@90.00:2027-02-01", 3),
        dca.parse_tranche("1000@85.00:2027-01-02", 4),   # Saturday: rolls to Mon 4 Jan
    ])


PRICES = _bars([("2026-11-16", 80.0, 100.0),   # before plan start: must be ignored
                ("2026-11-17", 99.0, 100.0),
                ("2026-11-18", 95.00, 97.0),  # touches t2 exactly
                ("2026-11-19", 94.0, 96.0)])


def test_parse_tranche_forms():
    t = dca.parse_tranche("1000@95.00:2027-03-01", 2)
    assert (t.id, t.amount, t.limit, t.due) == ("t2", 1000.0, 95.00, "2027-03-01")
    assert dca.parse_tranche("500@99", 1).due is None
    assert dca.parse_tranche("500:2027-01-04", 1).limit is None
    for bad in ("500", "abc", "500@x", "500:2027-13-01"):
        with pytest.raises(ValueError):
            dca.parse_tranche(bad, 1)


def test_trading_day_rolls_weekends_and_holidays():
    assert dca.trading_day(dt.date(2027, 1, 2)) == dt.date(2027, 1, 4)
    assert dca.trading_day(dt.date(2027, 1, 1), ["2027-01-01"]) == dt.date(2027, 1, 4)
    assert dca.trading_day(dt.date(2026, 11, 18)) == dt.date(2026, 11, 18)


def test_evaluate_states_and_actions():
    tab = dca.evaluate(_plan(), PRICES, dt.date(2026, 11, 19))
    assert tab.loc["t1", "state"] == "overdue"               # due 18 Nov, today 19 Nov
    assert tab.loc["t1", "est_shares"] == 10                 # floor(1000 / 96.0)
    assert tab.loc["t2", "state"] == "limit_touched"
    assert tab.loc["t2", "evidence"] == "low 95.00 <= 95.00 on 2026-11-18"
    assert tab.loc["t2", "est_shares"] == 10                 # floor(1000 / 95.00)
    assert tab.loc["t3", "state"] == "waiting"               # 120.0 low on 16 Nov is before the start
    assert tab.loc["t3", "evidence"] == "limit -6.2% from last close 96.00; 74 days to 2027-02-01"
    assert tab.loc["t4", "evidence"].endswith("46 days to 2027-01-04")


def test_due_today_and_limit_fallback_to_market():
    p = _plan()
    p.tranche("t1").status, p.tranche("t1").fill_price, p.tranche("t1").fill_date = "filled", 100.0, "2026-11-18"
    tab = dca.evaluate(p, PRICES, dt.date(2027, 1, 4))
    assert tab.loc["t1", "evidence"] == "filled 2026-11-18 at 100.0"
    assert tab.loc["t4", "state"] == "due"
    assert tab.loc["t4", "action"] == "buy 1,000 at market and cancel the resting limit"
    s = dca.summary(p, tab)
    assert (s["total"], s["filled"], s["pending"], s["actions_today"]) == (4000, 1000, 3000, 2)  # t2 touched + t4 due


def test_storage_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(dca, "DCA_DIR", tmp_path)
    p = _plan()
    dca.save(p)
    q = dca.load("t")
    assert q.to_dict() == p.to_dict()
    with pytest.raises(ValueError):
        dca.plan_path("../escape")
