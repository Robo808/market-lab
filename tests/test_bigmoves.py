"""Offline tests for mlab.bigmoves (issue #78): detector maths, point-in-time joins, real-response parsers, study blocks."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlab import bigmoves as bm
from mlab import bigmoves_study as st

FIX = Path(__file__).parent / "fixtures" / "bigmoves"


def _alternating(n=80, start="2019-01-01"):
    """Closes 100, 101, 100, 101 ...: every log return is exactly +/- log(1.01)."""
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([100.0 * (1.01 if i % 2 else 1.0) for i in range(n)], index=idx)


def test_detect_scores_with_ex_ante_sigma_and_groups_episodes():
    c = _alternating()
    nxt = pd.bdate_range(c.index[-1] + pd.Timedelta(days=1), periods=2)
    c = pd.concat([c, pd.Series([c.iloc[-1] * np.exp(0.05), c.iloc[-1] * np.exp(0.05) * np.exp(-0.04)], index=nxt)])
    mv = bm.detect(c, span=60, z_min=3.0, min_obs=60)
    L = np.log(1.01)
    a = 2 / 61
    assert list(mv.index) == list(nxt)
    assert mv["z"].iloc[0] == pytest.approx(0.05 / L, rel=1e-9)  # 5.02496...
    sig2 = np.sqrt((1 - a) * L**2 + a * 0.05**2)  # the first jump enters the second move's scale
    assert mv["z"].iloc[1] == pytest.approx(-0.04 / sig2, rel=1e-9)
    assert mv["direction"].tolist() == [1, -1]
    assert mv["episode"].nunique() == 1 and mv["episode_rank"].tolist() == [1, 2]


def test_price_features_use_only_prior_closes():
    c = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2019-01-01", periods=4))
    f = bm.price_features(c)
    assert np.isnan(f["pre_ret_1d"].iloc[1])
    assert f["pre_ret_1d"].iloc[2] == pytest.approx(np.log(110 / 100))
    assert f["pre_ret_1d"].iloc[3] == pytest.approx(np.log(99 / 110))  # never sees the 120 close of its own day


def test_asof_is_strictly_before_and_honours_release_lag():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]), name="x")
    d = pd.DatetimeIndex(pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-10"]))
    assert bm.asof(s, d)["x"].tolist() == [1.0, 2.0, 3.0]
    lagged = bm.asof(s, d, lag_days=3)["x"]  # 2019-01-01 data becomes usable after 2019-01-04
    assert np.isnan(lagged.iloc[0]) and np.isnan(lagged.iloc[1]) and lagged.iloc[2] == 3.0


def test_zscore_prior_excludes_the_value_itself():
    s = pd.Series([1.0, 2.0, 3.0, 10.0])
    z = bm.zscore_prior(s, window=3, min_periods=3)
    assert z.iloc[3] == pytest.approx((10 - 2) / 1.0)


def test_parse_fomc_real_pages():
    f19 = bm.parse_fomc((FIX / "fomchistorical2019_h5.htm").read_text(), 2019)
    assert len(f19) == 9
    assert pd.Timestamp("2019-05-01") in set(f19["date"])  # "April/May 30-1 Meeting"
    assert f19.loc[f19["date"] == "2019-10-04", "kind"].item() == "unscheduled"
    f08 = bm.parse_fomc((FIX / "fomchistorical2008_h5.htm").read_text(), 2008)
    assert len(f08) == 14 and (f08["kind"] == "meeting").sum() == 8
    assert f08.loc[f08["date"] == "2008-01-21", "kind"].item() == "call"


def test_parse_bbds_real_rows():
    b = bm.parse_bbds(FIX / "bbds_2018_2019.xlsx")
    assert len(b) == 13 and b.index.min() == pd.Timestamp("2018-02-05")
    assert b.loc["2018-02-05", "bbds_return"] == pytest.approx(-0.040869, abs=1e-6)
    assert b.loc["2019-08-23", "bbds_policy"] == pytest.approx(0.979167, abs=1e-6)
    shares = b[[f"bbds_{c}" for c in bm.BBDS_CATEGORIES]]
    assert (b["bbds_primary_share"] == shares.max(axis=1)).all()
    assert set(b["bbds_primary"]) <= set(bm.BBDS_CATEGORIES)


def test_calendar_features():
    td = pd.bdate_range("2019-07-01", "2019-08-30")
    fomc = pd.DataFrame({"date": pd.to_datetime(["2019-07-31", "2019-09-18"]), "kind": ["meeting", "meeting"], "label": ["", ""]})
    d = pd.DatetimeIndex(pd.to_datetime(["2019-07-31", "2019-08-02", "2019-08-16", "2019-08-30"]))
    f = bm.calendar_features(d, td, fomc)
    assert f["day_fomc"].tolist() == [True, False, False, False]
    assert f["day_first_friday"].tolist() == [False, True, False, False]
    assert f["day_opex"].tolist() == [False, False, True, False]
    assert f["day_month_end"].tolist() == [True, False, False, False]  # 08-30 is the last bar in `td`, next month unknown
    assert f["pre_days_to_fomc"].iloc[0] == 0 and f["pre_days_since_fomc"].iloc[1] == 2


def test_controls_keep_away_from_moves():
    c = _alternating(400)
    moves = c.index[[300, 320]]
    cd = bm.controls(c, moves, k=5, seed=1, buffer=5, min_obs=252)
    pos = pd.Series(np.arange(len(c)), index=c.index)[cd].to_numpy()
    assert len(cd) == 10 and (pos >= 252).all()
    assert all(min(abs(p - 300), abs(p - 320)) > 5 for p in pos)


def test_build_refuses_test_split():
    with pytest.raises(PermissionError):
        bm.build(end="2020-06-30", fetch=False)


def test_auc_and_bh_exact():
    assert st.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert st.bh([0.01, 0.04, 0.03, 0.5]).round(6).tolist() == [0.04, 0.053333, 0.053333, 0.5]


def test_blocked_folds_are_contiguous_in_time():
    dates = pd.Series(pd.bdate_range("2019-01-01", periods=12)[::-1])
    f = st.blocked_folds(dates, k=3)
    assert f.tolist() == [2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]


def test_classification_trial_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    dates = pd.Series(pd.bdate_range("2010-01-01", periods=n))
    y_sig = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(float)
    y_noise = (rng.random(n) < 0.5).astype(float)
    sig = st.classification_trial(X, y_sig, dates, 60, np.random.default_rng(1))
    noise = st.classification_trial(X, y_noise, dates, 60, np.random.default_rng(2))
    assert sig["z_vs_null"] > 3 and sig["stat"] > 0.8
    assert noise["z_vs_null"] < 3


def test_silhouette_two_clean_blobs():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    s = st.silhouette(X, np.array([0, 0, 1, 1]))
    b = (10 + np.sqrt(101)) / 2  # mean distance to the other blob
    assert s == pytest.approx((b - 1) / b)


def test_cluster_vs_controls_detects_blobs_against_diffuse_days():
    rng = np.random.default_rng(3)
    centres = np.array([[4.0, 0.0], [-4.0, 0.0], [0.0, 4.0], [0.0, -4.0]])
    Xm = centres[rng.integers(0, 4, 120)] + 0.3 * rng.normal(size=(120, 2))
    Xc = rng.normal(scale=2.5, size=(600, 2))
    r = st.cluster_vs_controls(Xm, Xc, 30, np.random.default_rng(4))
    assert r["stat"] > 0.7 and r["z_vs_null"] > 3
    same = st.cluster_vs_controls(Xc[:120], Xc[120:], 30, np.random.default_rng(5))
    assert same["z_vs_null"] < 3
