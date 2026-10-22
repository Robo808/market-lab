"""Supply/demand map and study: parsers on real saved responses (fixtures/econ_sd), exact statistics. Offline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlab.econ import supply_demand as sd
from mlab.providers import publishers as pub

FIX = Path(__file__).parent / "fixtures" / "econ_sd"


def test_map_is_consistent():
    m = sd.load_map()
    cats = m["categories"]
    assert len(cats) >= 20
    for k, c in cats.items():
        assert c["side"] in {"both", "supply", "demand"}, k
        assert bool(c.get("supply")) == (c["side"] in {"both", "supply"}), k
        assert bool(c.get("demand")) == (c["side"] in {"both", "demand"}), k
        assert c["industry"], k


def test_map_industries_exist_in_the_french_file():
    cols = set(sd.parse_french_monthly((FIX / "ff49_head.csv").read_text()).columns)
    for k, c in sd.load_map()["categories"].items():
        assert c["industry"] == "mkt" or c["industry"] in cols, k


def test_map_indicators_join_the_graph():
    from mlab.econ.graph import build
    g = build()
    for k, c in sd.load_map()["categories"].items():
        for i in c.get("indicators", []):
            assert f"sig:{i}" in g.nodes, (k, i)


def test_map_codes_are_registered_on_the_linked_sig_nodes():
    from mlab.econ.graph import build
    g = build()
    for k, c in sd.load_map()["categories"].items():
        ids = {(s["source"], s["id"]) for i in c.get("indicators", []) for s in g.nodes[f"sig:{i}"].data["series"]}
        if c.get("supply"):
            assert ("frb", f"G17-IP-{c['supply']}") in ids, k
        if c.get("util"):
            assert ("frb", f"G17-CU-{c['util']}") in ids, k
        for d in c.get("demand", []):
            assert ("census_mrts", d) in ids, k
        assert sd.release_lags()[k] == 1, k


def test_fetch_series_routes_g17_and_mrts(monkeypatch):
    from mlab.econ import fetch
    ip = pub.parse_g17((FIX / "ip_sa_excerpt.txt").read_text())
    rs = pub.parse_mrts_sheet(pd.read_csv(FIX / "mrts_2026_sheet.csv", header=None))
    monkeypatch.setattr(pub, "g17_table", lambda table="ip_sa": ip)
    monkeypatch.setattr(pub, "mrts_table", lambda: rs)
    s = fetch.fetch_series({"source": "frb", "id": "G17-IP-G3344"})
    assert s.loc["2026-09-01"] == 198.6335 and s.name == "G17-IP-G3344"
    assert fetch.fetch_series({"source": "census_mrts", "id": "444"}).loc["2026-08-01"] == 44323
    assert fetch.fetchable({"source": "frb", "id": "G17-IP-B50001"})
    assert not fetch.fetchable({"source": "frb", "id": "G19"})  # DDP release, not a G.17 series
    with pytest.raises(ValueError):
        pub.fed_g17("G17-XX-G3344")


def test_parse_g17_real_excerpt():
    df = pub.parse_g17((FIX / "ip_sa_excerpt.txt").read_text())
    assert set(df.columns) == {"B50001", "G3344"}
    assert df.loc["2026-09-01", "G3344"] == 198.6335
    assert df.loc["2024-01-01", "B50001"] == 99.2223
    assert df.index.max() == pd.Timestamp("2026-09-01")  # Oct-Dec 2026 not yet published: no rows, no zeros
    assert len(df) == 33 and df.notna().all().all()
    assert pub.parse_g17((FIX / "ip_sa_excerpt.txt").read_text(), {"G3344"}).columns.tolist() == ["G3344"]


def test_yoy_is_log_percent():
    s = pub.parse_g17((FIX / "ip_sa_excerpt.txt").read_text())["G3344"]
    assert sd.yoy(s).loc["2026-09-01"] == pytest.approx(18.148166865, abs=1e-6)


def test_parse_mrts_sheet_reads_the_adjusted_block_including_preliminary_month():
    sheet = pd.read_csv(FIX / "mrts_2026_sheet.csv", header=None)
    df = pub.parse_mrts_sheet(sheet)
    assert df.index.min() == pd.Timestamp("2026-01-01") and df.index.max() == pd.Timestamp("2026-08-01")
    assert df.loc["2026-01-01", "444"] == 43235  # adjusted, not the not-adjusted block
    assert df.loc["2026-08-01", "444"] == 44323  # "Aug. 2026(p)"
    assert df.loc["2026-01-01", "Retail and food services sales, total"] == 701860
    assert df.loc["2026-08-01", "Retail and food services sales, total"] == 739322
    assert pd.isna(df.loc["2026-01-01", "44811"])  # "(S)" suppressed


def test_parse_french_monthly_stops_at_block_end_and_maps_missing():
    df = sd.parse_french_monthly((FIX / "ff49_head.csv").read_text())
    assert len(df) == 3 and df.index[0] == pd.Timestamp("1926-07-01")
    assert df.loc["1926-07-01", "Agric"] == pytest.approx(0.0236)
    assert df.loc["1926-07-01", "Food"] == pytest.approx(0.0009)
    assert pd.isna(df.loc["1926-07-01", "Soda"])  # -99.99 = missing
    assert "Chips" in df.columns and len(df.columns) == 49


def test_fwd_sum_sums_the_next_h_months():
    r = pd.Series([1.0, 2, 3, 4, 5])
    out = sd.fwd_sum(r, 2)
    assert out.tolist()[:3] == [5.0, 7.0, 9.0] and out.iloc[3:].isna().all()


def test_nw_t_with_no_lags_is_the_hc0_t():
    x = np.array([1.0, 2, 3, 4, 5, 6])
    y = np.array([1.0, 3, 2, 5, 4, 6])
    b, t = sd.nw_t(y, x, 0)
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(6), z])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    V = np.linalg.inv(X.T @ X) @ (X.T * e**2) @ X @ np.linalg.inv(X.T @ X)
    assert b == pytest.approx(beta[1]) and t == pytest.approx(beta[1] / np.sqrt(V[1, 1]))


def test_nonoverlap_t_uses_every_hth_point():
    x = np.arange(24, dtype=float)
    y = np.where(np.arange(24) % 2 == 0, 1.0, -1.0) + 0.1 * x
    t3 = sd.nonoverlap_t(y, x, 3)
    ts = [sd.nw_t(y[o::3], x[o::3], 0)[1] for o in range(3)]
    assert t3 == pytest.approx(np.median(ts))
    assert sd.nonoverlap_t(y, x, 1) == pytest.approx(sd.nw_t(y, x, 0)[1])


def test_bh_adjusts_exactly():
    q = sd.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_shift_null_keeps_a_noise_feature_quiet():
    rng = np.random.default_rng(0)
    y, x = rng.normal(size=300), np.cumsum(rng.normal(size=300))
    null = sd.shift_null(y, x, 3, 200, np.random.default_rng(1))
    assert len(null) == 200 and (null >= 3).mean() < 0.05


def _toy_raw(n=360, seed=3):
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    m = sd.load_map()
    lvl = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n))), idx)  # noqa: E731
    ip = pd.DataFrame({c["supply"]: lvl() for c in m["categories"].values() if c.get("supply")})
    util = pd.DataFrame({c["util"]: 75 + rng.normal(0, 1, n).cumsum() * 0.1 for c in m["categories"].values()
                         if c.get("util")}, index=idx)
    codes = {x for c in m["categories"].values() for x in c.get("demand", [])}
    sales = pd.DataFrame({k: lvl() for k in codes})
    inds = {c["industry"] for c in m["categories"].values()} - {"mkt"}
    ind = pd.DataFrame({k: rng.normal(0.008, 0.05, n) for k in inds}, index=idx)
    return sd.Raw(ip, util, sales, ind, pd.Series(rng.normal(0.008, 0.04, n), idx), "toy")


def test_study_on_structureless_data_rarely_clears_the_bar():
    res = sd.study(_toy_raw(), "train", n_null=50)
    P = res["predict"]
    assert len(P) > 200
    assert (P["t"].abs() >= 3).mean() < 0.06  # coin-flip data: a few false positives at most
    assert (P["q_bh"] >= P["p_norm"] - 1e-12).all() and (P.loc[P["t"].abs() >= 3, "q_bh"] > 0.05).mean() > 0.5
    assert {"t_nonoverlap", "n_indep"} <= set(P.columns)
    assert set(res["xs"]["feature"]) >= {"gap", "s_yoy"} and (res["xs"]["t"].abs() < 3).all()


def test_test_split_is_locked():
    with pytest.raises(PermissionError):
        sd.study(_toy_raw(), "test")
