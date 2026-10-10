"""Event-study engine: timing (no look-ahead), signing, costs, permutation null. Synthetic data only."""
import numpy as np
import pandas as pd

from mlab.textlab import events as ev


def _bars(n=900, seed=0, shocks=()):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n, tz="UTC")
    r = rng.normal(0, 0.01, n)
    for i, size, drift in shocks:  # a shock on day i, then `drift` per day for 5 days from the next open
        r[i] += size
        r[i + 1:i + 6] += drift
    c = 100 * np.cumprod(1 + r)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": c, "low": c, "close": c, "adj_close": c, "volume": 1.0}, index=idx)


def test_abnormal_uses_only_past_bars():
    a, m = _bars(seed=1), _bars(seed=2)
    full = ev.abnormal(a, m)
    cut = ev.abnormal(a.iloc[:600], m.iloc[:600])
    pd.testing.assert_series_equal(full["z"].iloc[:600], cut["z"], check_names=False)


def test_continuation_vs_reversal_is_detected():
    news = [(300 + 40 * i, 0.08, 0.004) for i in range(12)]     # news shocks continue
    noise = [(320 + 40 * i, 0.08, -0.004) for i in range(12)]   # no-news shocks revert
    a, m = _bars(shocks=news + noise), _bars(seed=9)
    idx = a.index
    flags = {"A": {idx[i] for i, *_ in news}}
    e = ev.events({"A": a}, m, flags, k=2.5, horizons=(5,), cost_bp=10)
    r = ev.compare(e, "car5", n_perm=200)
    assert r["n_flag"] >= 10 and r["mean_flag_bp"] > 0 > r["mean_noflag_bp"] and r["perm_p"] < 0.05


def test_cost_is_charged_on_both_directions():
    a, m = _bars(shocks=[(300, 0.08, 0.0), (400, -0.08, 0.0)]), _bars(seed=9)
    gross = ev.events({"A": a}, m, k=2.5, horizons=(1,), cost_bp=0)
    net = ev.events({"A": a}, m, k=2.5, horizons=(1,), cost_bp=10)
    assert np.allclose(gross["car1"] - net["car1"], 0.001)


def test_filing_dates_map_to_next_session_with_spill():
    idx = pd.bdate_range("2024-01-01", periods=10, tz="UTC")
    f = ev.trading_day_flags(pd.Series(["2024-01-06"]), idx)  # Saturday -> Monday 8th and Tuesday 9th
    assert f == {pd.Timestamp("2024-01-08", tz="UTC"), pd.Timestamp("2024-01-09", tz="UTC")}


def _gkg_zip(rows):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.gkg.csv", "\n".join("\t".join(r) for r in rows))
    return buf.getvalue()


def _gkg_row(orgs, tone, themes="ECON_STOCKMARKET;EPU_POLICY", title="Headline"):
    r = [""] * 27
    r[3], r[4], r[7], r[13] = "example.com", f"https://example.com/{orgs[:5]}", themes, orgs
    r[15] = f"{tone},2,3,5,20,1,400"
    r[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return r


def test_gdelt_org_matching():
    from mlab.textlab.gdelt import match_symbols
    assert match_symbols("amazon web services;white house") == {"AMZN"}
    assert match_symbols("tesla inc;nvidia corporation") == {"TSLA", "NVDA"}
    assert match_symbols("amazonia;teslacoil;apple daily") == set()
    from mlab.textlab.gdelt import match
    assert match("instagram;meta platforms", "elon musk") == {"META": "org:meta platforms", "TSLA": "person:elon musk"}
    assert match("youtube") == {"GOOGL": "platform:youtube"}


def test_gdelt_ingest_is_point_in_time_and_resumable(tmp_path, monkeypatch):
    from mlab.textlab import gdelt
    monkeypatch.setattr(gdelt, "STORE", tmp_path)
    blob = _gkg_zip([_gkg_row("amazon;reuters", -3.5, title="Amazon cuts jobs"), _gkg_row("white house", 1.0)])
    calls = []

    class R:
        status_code = 200
        content = blob

        def raise_for_status(self):
            pass

    class S:
        def get(self, url, timeout=None):
            calls.append(url)
            return R()
    ts = pd.Timestamp("2026-10-09 14:15", tz="UTC")
    assert gdelt.ingest_one(ts, S()) == "ok 1"
    assert gdelt.ingest_one(ts, S()) == "skip" and len(calls) == 1
    f = gdelt.load("firm", ["AMZN"])
    assert f.iloc[0]["avail"] == ts and f.iloc[0]["title"] == "Amazon cuts jobs" and f.iloc[0]["tone"] == -3.5
    m = gdelt.load("macro")
    assert set(m["theme"]) == {"_ALL", "ECON_STOCKMARKET", "EPU_POLICY"} and m.loc[m.theme == "_ALL", "n"].item() == 2
    d = gdelt.daily("AMZN")
    assert d["articles"].item() == 1 and d["neg_share"].item() == 1.0
