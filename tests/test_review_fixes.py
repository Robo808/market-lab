"""Regression tests for the code-review fixes: Sortino, Bollinger, SEC tag merge, dual momentum, walk-forward."""
import numpy as np
import pandas as pd
import pytest
from conftest import synth_ohlcv


def test_sortino_uses_downside_deviation():
    from mlab.stats import returns, summary
    close = synth_ohlcv(600)["close"]
    r = returns(close)
    down = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    assert summary(close)["sortino"] == pytest.approx(r.mean() * 252 / down)


def test_bollinger_population_std():
    from mlab.ta import bollinger
    close = synth_ohlcv(200)["close"]
    b = bollinger(close, 20, 2.0)
    w = close.iloc[-20:].to_numpy()
    assert b["upper"].iloc[-1] == pytest.approx(w.mean() + 2 * w.std(ddof=0))


def test_sec_later_tag_fills_missing_years(monkeypatch):
    from mlab.providers import sec

    def fy(val, y):
        return {"val": val, "end": f"{y}-12-31", "start": f"{y}-01-01", "form": "10-K", "filed": f"{y + 1}-02-01"}
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [fy(200 + i, 2018 + i) for i in range(6)]}},
        "Revenues": {"units": {"USD": [fy(100 + i, 2014 + i) for i in range(6)]}},  # old tag, overlaps 2018-2019
    }}}
    monkeypatch.setattr(sec, "company_facts", lambda t: facts)
    rev = sec.annual_financials("X")["revenue"]
    assert rev.index.min().year == 2014 and len(rev) == 10
    assert rev.loc["2018-12-31"] == 200  # the first tag keeps priority where both exist


def test_dual_momentum_rejects_unknown_safe_asset():
    from mlab.quant.strategies import dual_momentum
    panel = pd.DataFrame({"A": synth_ohlcv(400, 1)["close"], "B": synth_ohlcv(400, 2)["close"]})
    with pytest.raises(ValueError, match="safe asset"):
        dual_momentum(panel, safe="BIL")


def test_walk_forward_ignores_nan_sharpe(monkeypatch):
    from mlab import backtest
    df = synth_ohlcv(900)
    flat = lambda d, **p: pd.Series(0.0, index=d.index)  # noqa: E731  flat strategy -> NaN Sharpe
    wf = backtest.walk_forward(df, flat, [{"x": 1}, {"x": 2}], train=504, test=126)
    assert len(wf) > 0
