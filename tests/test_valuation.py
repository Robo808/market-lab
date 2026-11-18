"""Offline tests for mlab.valuation: exact values on hand-checkable series, parser on a trimmed real response."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlab import valuation as v

FIX = Path(__file__).parent / "fixtures" / "valuation"


def test_forward_annualised_constant_growth():
    idx = pd.date_range("2000-01-01", periods=25, freq="MS")
    level = pd.Series(100 * 1.1 ** (np.arange(25) / 12), index=idx)
    f = v.forward_annualised(level, 12)
    assert f.iloc[0] == pytest.approx(0.10, abs=1e-12)
    assert f.iloc[12] == pytest.approx(0.10, abs=1e-12)
    assert f.iloc[13:].isna().all()


def test_ols_recovers_line():
    x = np.arange(10.0)
    b, r2, se = v.ols(x, 2 + 3 * x)
    assert b == pytest.approx([2.0, 3.0])
    assert r2 == pytest.approx(1.0)
    assert se == pytest.approx(0.0, abs=1e-12)


def test_split_by_outcome_uses_window_end():
    idx = pd.date_range("2004-11-01", periods=4, freq="MS")  # windows end 2014-11, 2014-12, 2015-01, 2015-02
    df = pd.DataFrame({"x": range(4)}, index=idx)
    s = v.split_by_outcome(df, 120)
    assert list(s["train"].index.strftime("%Y-%m")) == ["2004-11", "2004-12"]
    assert list(s["val"].index.strftime("%Y-%m")) == ["2005-01", "2005-02"]


def test_split_never_returns_test_windows():
    idx = pd.date_range("2010-01-01", periods=3, freq="MS")  # windows end 2020+: locked test split
    s = v.split_by_outcome(pd.DataFrame({"x": range(3)}, index=idx), 120)
    assert s["train"].empty and s["val"].empty


def test_lump_vs_staged_flat_price_zero_cash():
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    t = v.lump_vs_staged(pd.Series(1.0, index=idx), pd.Series(0.0, index=idx), tranches=2, horizon=2)
    assert t["lump"].tolist() == pytest.approx([1.0] * 4)
    assert t["staged"].tolist() == pytest.approx([1.0] * 4)


def test_lump_vs_staged_price_jump():
    idx = pd.date_range("2000-01-01", periods=3, freq="MS")
    t = v.lump_vs_staged(pd.Series([1.0, 2.0, 2.0], index=idx), pd.Series(0.0, index=idx), tranches=2, horizon=2)
    # lump: 2.0; staged: 0.5 units at 1 + 0.25 units at 2 = 0.75 units * 2 = 1.5
    assert t.loc["2000-01-01", "lump"] == pytest.approx(2.0)
    assert t.loc["2000-01-01", "staged"] == pytest.approx(1.5)
    assert t.loc["2000-01-01", "gap"] == pytest.approx(0.5)


def test_lump_vs_staged_cash_interest():
    idx = pd.date_range("2000-01-01", periods=3, freq="MS")
    t = v.lump_vs_staged(pd.Series(1.0, index=idx), pd.Series(0.12, index=idx), tranches=2, horizon=2)
    # cash 0.5 after first buy, x1.01 -> 0.505, second buy leaves 0.005, x1.01 -> 0.00505
    assert t.loc["2000-01-01", "staged"] == pytest.approx(1.00505, abs=1e-12)
    assert t.loc["2000-01-01", "gap"] == pytest.approx(-0.00505, abs=1e-12)


def test_parse_shiller_october_quirk():
    raw = pd.DataFrame({"Date": [2023.09, 2023.1, 2023.11, "note"], "P": [1, 2, 3, None], "D": [0.1] * 4,
                        "E": [1.0] * 4, "CPI": [300.0] * 4, "Rate GS10": [4.0] * 4, "CAPE": [30.0, 31.0, 32.0, None],
                        "Price.1": [10.0, 11.0, 12.0, None]})
    d = v.parse_shiller(raw)
    assert list(d.index.strftime("%Y-%m")) == ["2023-09", "2023-10", "2023-11"]
    assert d["cape"].tolist() == [30.0, 31.0, 32.0]


def test_parse_multpl_real_fixture():
    s = v.parse_multpl((FIX / "multpl_shiller_pe_20261118.html").read_text())
    assert s.index[-1] == pd.Timestamp("2026-11-17")
    assert s.iloc[-1] == 38.76
    assert s.tolist() == [40.42, 40.60, 39.82, 38.76]


def test_models_and_buckets_on_synthetic_history():
    # Returns fall exactly with log(CAPE): the simple model must recover the slope.
    idx = pd.date_range("1891-01-01", "2024-12-01", freq="MS")
    rng = np.random.default_rng(0)
    cape = pd.Series(np.exp(rng.uniform(np.log(8), np.log(40), len(idx))), index=idx)
    f = pd.DataFrame({"cape": cape, "fwd": 0.30 - 0.08 * np.log(cape), "fwd12": 0.0,
                      "ry": rng.normal(0.02, 0.01, len(idx)), "infl": rng.normal(0.03, 0.01, len(idx)),
                      "vol": rng.uniform(0.1, 0.2, len(idx))})
    f["ey"] = 1 / f["cape"]
    m = v.fit_models(f)
    assert m.simple == pytest.approx([0.30, -0.08], abs=1e-10)
    assert m.r2["simple"] == pytest.approx(1.0)
    assert float(m.predict_simple(np.e ** 3)) == pytest.approx(0.30 - 0.24)
    out = v.validate(m, f)
    assert out["rmse_simple"] == pytest.approx(0.0, abs=1e-10)
    b = v.bucket_table(f, bins=(0, 20, 100))
    assert b["months"].sum() == len(f)
