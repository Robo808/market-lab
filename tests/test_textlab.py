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
        def __init__(self, code):
            self.status_code, self.content = code, blob

        def raise_for_status(self):
            pass

    class S:
        def get(self, url, timeout=None):
            calls.append(url)
            return R(404 if "1430" in url else 200)
    monkeypatch.setattr("mlab.net.session", lambda *a, **k: S())
    ts = pd.Timestamp("2026-10-09 14:15", tz="UTC")
    r = gdelt.ingest("2026-10-09 14:15", "2026-10-09 14:30", workers=1, log=lambda *a: None)
    assert r == {"ok": 1, "missing": 1, "failed": 0, "articles": 1} and len(calls) == 2
    assert gdelt.ingest("2026-10-09 14:15", "2026-10-09 14:30", log=lambda *a: None)["ok"] == 0 and len(calls) == 2
    assert sorted(p.name for p in tmp_path.rglob("*.parquet")) == ["2026-10-09.parquet", "2026-10-09.parquet"]
    f = gdelt.load("firm", ["AMZN"])
    assert f.iloc[0]["avail"] == ts and f.iloc[0]["title"] == "Amazon cuts jobs" and f.iloc[0]["tone"] == -3.5
    m = gdelt.load("macro")
    assert set(m["theme"]) == {"_ALL", "ECON_STOCKMARKET", "EPU_POLICY"} and m.loc[m.theme == "_ALL", "n"].item() == 2
    d = gdelt.daily("AMZN")
    assert d["articles"].item() == 1 and d["neg_share"].item() == 1.0


def test_novelty_and_daily_timing():
    from mlab.textlab import features as ft
    t0 = pd.Timestamp("2024-01-08 14:00", tz="UTC")
    a = pd.DataFrame({"avail": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=7), t0 + pd.Timedelta(days=9)],
                      "title": ["Amazon cuts 10,000 jobs in cloud unit", "Amazon cuts 10,000 jobs in cloud unit",
                                "Amazon wins Pentagon contract", "Amazon cuts 10,000 jobs in cloud unit"],
                      "tone": [-5.0, -5.0, 3.0, -5.0]})
    a["novelty"] = ft.novelty(a)
    assert a["novelty"].tolist()[0] == 1.0 and a["novelty"].tolist()[1] < 0.01 and a["novelty"].tolist()[3] == 1.0
    idx = pd.bdate_range("2024-01-08", periods=10, tz="UTC")
    d = ft.firm_daily(a, idx)
    # 14:00 and 15:00 count on Monday; 21:00 is after the cutoff and moves to Tuesday
    assert d.loc[idx[0], "n"] == 2 and d.loc[idx[1], "n"] == 1 and d.loc[idx[0], "n_fresh"] == 1


def test_slug_headline_from_url():
    from mlab.textlab import features as ft
    assert ft.slug("https://betanews.com/2017/06/01/xbox-game-pass-opens-up-to-everyone/") == \
        "xbox game pass opens up to everyone"
    assert ft.slug("http://www.sunherald.com/news/business/article153733974.html") == ""
    assert ft.slug(None) == ""


def test_novelty_blocks_match_row_by_row():
    from mlab.textlab import features as ft
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(40)]
    n = 600
    df = pd.DataFrame({"avail": pd.Timestamp("2020-01-01", tz="UTC")
                       + pd.to_timedelta(np.sort(rng.integers(0, 20 * 96, n)) * 15, unit="min"),
                       "title": [" ".join(rng.choice(vocab, 4)) for _ in range(n)]})
    a, b = ft.novelty(df, block=n), ft.novelty(df, block=37)
    assert np.allclose(a, b)
    x = ft.hash_matrix(df["title"].tolist())
    t = df["avail"]
    i = 400  # brute force one row: max cosine vs earlier rows within 5 days
    prev = [j for j in range(i) if t[j] >= t[i] - pd.Timedelta("5D")]
    want = 1 - max((x[i] @ x[j].T).toarray()[0, 0] for j in prev)
    assert abs(a[i] - want) < 1e-12


def test_attention_events_use_prior_days_only():
    from mlab.textlab import studies
    idx = pd.bdate_range("2021-01-01", periods=400, tz="UTC")
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"open": 100 * np.exp(np.cumsum(rng.normal(0, .01, 400))),
                       "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, 400)))}, index=idx)
    mk = pd.DataFrame({"open": 100 * np.exp(np.cumsum(rng.normal(0, .01, 400))),
                       "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, 400)))}, index=idx)
    n = rng.poisson(10, 400).astype(float)
    n[[150, 300, 350]] = 80
    fresh = np.floor(n * rng.uniform(.2, .8, 400))
    d = pd.DataFrame({"n": n, "tone": rng.normal(0, 1, 400), "n_fresh": fresh}, index=idx)
    spec = {"event": {"baseline_days": 60, "spike_sd": 2.0, "min_articles": 20, "fresh_share_days": 250},
            "horizons": [1, 5], "costs": {"round_trip_bp": 10}}
    ab = ev.abnormal(px, mk)
    full = studies.attention_events(ab, d, spec)
    assert {idx[150], idx[300], idx[350]} <= set(full["date"])  # the planted spikes are found
    cut = 320  # truncating the future changes nothing about the day-300 event except CARs running off the end
    part = studies.attention_events(ev.abnormal(px.iloc[:cut], mk.iloc[:cut]), d.iloc[:cut], spec)
    a, b = full.set_index("date").loc[idx[300]], part.set_index("date").loc[idx[300]]
    assert (a["dir"], a["flag"], a["fresh_share"], a["car5"]) == (b["dir"], b["flag"], b["fresh_share"], b["car5"])


def test_primary_needs_company_in_headline():
    from mlab.textlab import features as ft
    assert ft.primary("netflix raises prices in uk", "NFLX")
    assert not ft.primary("paramount heathers parkland", "NFLX")  # GKG tagged Netflix, the story isn't about it
    assert not ft.primary("intelligence agencies warn", "INTC") and ft.primary("intel cuts guidance", "INTC")
    assert ft.primary("facebook ending feed experiment", "META") and not ft.primary("", "META")


def test_gdelt_day_with_every_file_missing(tmp_path, monkeypatch):
    from mlab.textlab import gdelt
    monkeypatch.setattr(gdelt, "STORE", tmp_path)

    class S:
        def get(self, url, timeout=None):
            return type("R", (), {"status_code": 404, "content": b"", "raise_for_status": lambda self: None})()
    monkeypatch.setattr("mlab.net.session", lambda *a, **k: S())
    r = gdelt.ingest("2025-06-20 00:00", "2025-06-20 00:30", workers=1, log=lambda *a: None)
    assert r["missing"] == 3 and r["ok"] == 0
    assert [p.parent.name for p in tmp_path.rglob("*.parquet")] == ["macro"]  # marked done, no firm file
    assert gdelt.ingest("2025-06-20 00:00", "2025-06-20 00:30", log=lambda *a: None)["missing"] == 0


def _events_zip(rows):
    import io
    import zipfile
    lines = []
    for a1, a2, geo, root, quad, gold, arts, tone in rows:
        r = [""] * 61
        r[7], r[17], r[53], r[28], r[29], r[25] = a1, a2, geo, root, str(quad), "1"
        r[30], r[33], r[34] = str(gold), str(arts), str(tone)
        lines.append("\t".join(r))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.export.CSV", "\n".join(lines))
    return buf.getvalue()


def test_gdelt_events_aggregate_and_resume(tmp_path, monkeypatch):
    from mlab.textlab import gdelt, gdelt_events as ge
    monkeypatch.setattr(ge, "STORE", tmp_path / "events")
    blob = _events_zip([("USA", "CHN", "US", "16", 4, -4.0, 3, -2.0), ("USA", "CHN", "US", "16", 4, -8.0, 1, -6.0),
                        ("", "", "IR", "14", 3, -6.5, 2, -5.0)])
    calls = []

    class S:
        def get(self, url, timeout=None):
            calls.append(url)
            code = 404 if "0030" in url else 200
            return type("R", (), {"status_code": code, "content": blob, "raise_for_status": lambda self: None})()
    monkeypatch.setattr("mlab.net.session", lambda *a, **k: S())
    r = ge.ingest("2026-10-09 00:00", "2026-10-09 00:30", workers=1, log=lambda *a: None)
    assert r == {"ok": 2, "missing": 1, "failed": 0, "events": 6} and len(calls) == 3
    assert ge.ingest("2026-10-09 00:00", "2026-10-09 00:30", log=lambda *a: None)["ok"] == 0 and len(calls) == 3
    df = ge.load()
    sanc = df[(df.a1 == "USA") & (df.root == "16")].iloc[0]
    assert sanc["avail"] == pd.Timestamp("2026-10-09 00:00", tz="UTC")  # 00:00 and 00:15 files, same bucket
    assert ge.bucket(pd.Timestamp("2026-10-09 20:15", tz="UTC")).hour == 20
    assert sanc["events"] == 4 and sanc["articles"] == 8 and abs(sanc["goldstein"] - (-5.0)) < 1e-9
    assert gdelt.STORE != ge.STORE


def test_gdelt_themes_by_country_and_resume(tmp_path, monkeypatch):
    from mlab.textlab import gdelt_themes as gt
    monkeypatch.setattr(gt, "STORE", tmp_path / "themes")
    a = _gkg_row("x", -4.0, themes="SANCTIONS;TAX_FNCACT_LEADER;ECON_TRADE_DISPUTE")
    a[9] = "4#Moscow#RS#RS48#55#37#1;1#Russia#RS##60#100#2;1#United States#US##38#-97#3"
    b = _gkg_row("y", 2.0, themes="SANCTIONS")
    b[9] = "1#United States#US##38#-97#3"
    blob = _gkg_zip([a, b, _gkg_row("z", 0.0, themes="SPORTS")])
    calls = []

    class S:
        def get(self, url, timeout=None):
            calls.append(url)
            code = 404 if "2015" in url else 200
            return type("R", (), {"status_code": code, "content": blob, "raise_for_status": lambda self: None})()
    monkeypatch.setattr("mlab.net.session", lambda *x, **k: S())
    r = gt.ingest("2026-10-09 19:45", "2026-10-09 20:00", workers=1, log=lambda *x: None)
    assert r == {"ok": 2, "missing": 0, "failed": 0, "articles": 6} and len(calls) == 2
    assert gt.ingest("2026-10-09 19:45", "2026-10-09 20:00", log=lambda *x: None)["ok"] == 0 and len(calls) == 2
    d = gt.load()
    early = d[d["avail"] == pd.Timestamp("2026-10-09", tz="UTC")].set_index(["theme", "country"])
    assert early.loc[("SANCTIONS", "_ALL"), "n"] == 2 and early.loc[("SANCTIONS", "RS"), "n"] == 1
    assert early.loc[("SANCTIONS", "US"), "tone"] == 2.0 and ("TAX_FNCACT_LEADER", "_ALL") not in early.index
    assert set(d["avail"].dt.hour) == {0, 20}  # the 20:00 file lands in the late bucket


def test_gpr_country_portfolio_timing_and_sign():
    from mlab.textlab import studies
    days = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
    months = pd.date_range("2020-01-01", "2020-05-01", freq="MS", tz="UTC")
    gpr = {"A": pd.Series([1, 2, 2, 2, 2.0], months), "B": pd.Series([1, 1, 1, 1, 1.0], months),
           "C": pd.Series([1, 0.5, 0.5, 0.5, 0.5], months)}
    px = {"EA": pd.Series(np.linspace(100, 130, len(days)), days), "EB": pd.Series(100.0, days),
          "EC": pd.Series(np.linspace(100, 90, len(days)), days)}
    spec = {"etf": {"A": "EA", "B": "EB", "C": "EC"}, "rebalance_day": 7, "n_side": 1, "min_countries": 3,
            "costs": {"round_trip_bp": 0}}
    b = studies.gpr_country_portfolio(gpr, px, spec)
    first = b[b["longs"] == "A"].iloc[0]  # February's jump in A's GPR is known in early March
    assert first["start"] == pd.Timestamp("2020-03-09", tz="UTC") and first["shorts"] == "C" and first["gross"] > 0
    assert abs(studies.nw_t(pd.Series([1.0, 1.1, 0.9, 1.0, 1.05]))) > 10


def test_gpr_country_portfolio_costs_and_zero_index():
    from mlab.textlab import studies
    days = pd.bdate_range("2020-01-01", "2020-04-30", tz="UTC")
    months = pd.date_range("2020-01-01", "2020-02-01", freq="MS", tz="UTC")
    gpr = {"A": pd.Series([1, 2.0], months), "B": pd.Series([0, 1.0], months), "C": pd.Series([1, 0.5], months)}
    px = {e: pd.Series(100.0, days) for e in ("EA", "EB", "EC")}
    spec = {"etf": {"A": "EA", "B": "EB", "C": "EC"}, "rebalance_day": 7, "n_side": 1, "min_countries": 2,
            "costs": {"round_trip_bp": 20}}
    b = studies.gpr_country_portfolio(gpr, px, spec)
    assert b.iloc[0]["n"] == 2  # B's index was 0 in January: no log change, sits out
    assert abs(b.iloc[0]["net"] - (-0.0020)) < 1e-12  # opening long A and short C: one round trip each, 1/1 weight
