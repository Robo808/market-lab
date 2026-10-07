import math

import numpy as np
import pandas as pd
import pytest

from mlab import earnings as ea


def price_frame(jumps: dict, start="2025-01-01", n=160, drift=0.0):
    """Business-day OHLC stamped like Yahoo (UTC, 05:00), flat at 100 except planted moves.
    jumps: {date: (gap_at_open, close_vs_prev_close)} applied multiplicatively from that session on."""
    days = pd.bdate_range(start, periods=n)
    close = np.empty(n)
    opn = np.empty(n)
    lvl = 100.0
    rng = np.random.default_rng(1)
    for i, d in enumerate(days):
        g, c = jumps.get(d.strftime("%Y-%m-%d"), (0.0, None))
        opn[i] = lvl * (1 + g)
        noise = 0.001 * rng.standard_normal()
        lvl = lvl * (1 + c) if c is not None else lvl * (1 + drift + noise)
        close[i] = lvl
    idx = (days + pd.Timedelta(hours=5)).tz_localize("UTC")
    return pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.005, "low": np.minimum(opn, close) * 0.995,
                         "close": close, "volume": 1e6}, index=idx)


def raw_dates(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, *_ in rows], name="Earnings Date")
    return pd.DataFrame({"EPS Estimate": [r[1] for r in rows], "Reported EPS": [r[2] for r in rows],
                         "Surprise(%)": [r[3] for r in rows]}, index=idx)


def test_timing_and_quarter_labels():
    assert ea.infer_timing(pd.Timestamp("2025-03-04 07:00", tz="America/New_York")) == "BMO"
    assert ea.infer_timing(pd.Timestamp("2025-03-04 16:05", tz="America/New_York")) == "AMC"
    assert ea.infer_timing(pd.Timestamp("2025-03-04 12:00", tz="America/New_York")) == "DMH"
    assert ea.infer_timing(pd.Timestamp("2025-03-04", tz="America/New_York")) == "unknown"
    assert ea.infer_timing(pd.Timestamp("2025-03-04 21:05", tz="UTC")) == "AMC"
    assert ea.quarter_label("2025-01-28") == "2024Q4"
    assert ea.quarter_label("2025-04-01") == "2025Q1"
    assert ea.quarter_label("2025-10-30", fiscal_offset=1) == "2025Q4"


def test_normalize_earnings():
    ev = ea.normalize_earnings(raw_dates([("2025-04-30 16:05", 1.0, 1.1, np.nan), ("2025-01-29 07:00", 1.0, 0.9, -10.0),
                                          ("2025-07-30 16:05", 1.2, np.nan, np.nan)]))
    assert list(ev["timing"]) == ["BMO", "AMC", "AMC"]
    assert ev["surprise_pct"].iloc[1] == pytest.approx(10.0)  # computed when missing
    assert list(ev["reported"]) == [True, True, False]
    assert list(ev["quarter"]) == ["2024Q4", "2025Q1", "2025Q2"]


def test_reaction_windows_bmo_vs_amc():
    # BMO on Tue 2025-03-04: the event day itself gaps +5% and closes +8%.
    # AMC on Wed 2025-04-16: next day (Thu 04-17) gaps -6% and closes -10%.
    px = price_frame({"2025-03-04": (0.05, 0.08), "2025-04-17": (-0.06, -0.10)})
    ev = ea.normalize_earnings(raw_dates([("2025-03-04 07:00", 1.0, 1.2, 20.0), ("2025-04-16 16:10", 1.0, 1.1, 10.0)]))
    rx = ea.reaction_study(ev, px)
    bmo, amc = rx.iloc[0], rx.iloc[1]
    assert bmo["react_date"] == pd.Timestamp("2025-03-04") and bmo["pre_date"] == pd.Timestamp("2025-03-03")
    assert bmo["gap"] == pytest.approx(0.05) and bmo["day1"] == pytest.approx(0.08)
    assert amc["react_date"] == pd.Timestamp("2025-04-17") and amc["pre_date"] == pd.Timestamp("2025-04-16")
    assert amc["gap"] == pytest.approx(-0.06) and amc["day1"] == pytest.approx(-0.10)
    assert bmo["z"] > 5 and amc["move_atr"] < -1  # planted jumps dwarf the 0.1% noise
    # treating the AMC event as BMO would measure the quiet event day instead
    wrong = ea.reaction_study(ev.assign(timing="BMO"), px).iloc[1]
    assert abs(wrong["day1"]) < 0.01
    unknown = ea.reaction_study(ev.assign(timing="unknown"), px).iloc[1]
    assert unknown["pre_date"] == pd.Timestamp("2025-04-15") and unknown["react_date"] == pd.Timestamp("2025-04-17")


def test_drift_and_benchmark():
    # deterministic +1%/day for 20 sessions after the event day
    days = pd.bdate_range("2025-03-05", periods=20).strftime("%Y-%m-%d")
    px = price_frame({"2025-03-04": (0.0, 0.05), **{d: (0.0, 0.01) for d in days}})
    bench = price_frame({"2025-03-04": (0.0, 0.01)})
    ev = ea.normalize_earnings(raw_dates([("2025-03-04 07:00", 1.0, 1.1, 10.0)]))
    r = ea.reaction_study(ev, px, bench, implied={"2025-03-04": 0.04}).iloc[0]
    assert r["drift5"] == pytest.approx(1.01 ** 5 - 1)
    assert r["drift20"] == pytest.approx(1.01 ** 20 - 1)
    assert r["abn_day1"] == pytest.approx(0.05 - 0.01)
    assert r["realized_vs_implied"] == pytest.approx(0.05 / 0.04)


def test_summary_stats():
    rx = pd.DataFrame({"surprise_pct": [5, 3, -2, 4, -6], "day1": [0.04, -0.02, -0.05, 0.03, 0.01], "gap": [0.03, -0.01, -0.04, 0.02, 0.0],
                       "drift5": [0.01] * 5, "drift20": [0.02, 0.01, -0.03, 0.04, 0.0], "z": [2, -1, -2.5, 1.5, 0.5], "move_atr": [1] * 5})
    s = ea.summarize_reactions(rx)
    assert s["n"] == 5 and s["beat_but_sold_off"] == 1 and s["miss_but_rallied"] == 1
    assert s["up_rate"] == pytest.approx(0.6)
    assert s["avg_abs_day1"] == pytest.approx(0.03)
    assert s["drift20_after_beats"] == pytest.approx((0.02 + 0.01 + 0.04) / 3)
    assert s["corr_n"] == 5 and -1 <= s["corr_surprise_day1"] <= 1


def test_lexicon_tone():
    pos = ea.tone("We delivered record revenue and strong growth, an excellent quarter with robust demand.")
    neg = ea.tone("Results were disappointing; we saw weak demand, declining margins and a loss due to headwinds.")
    unc = ea.tone("Demand may fluctuate and we could see uncertain conditions; the outlook is unclear.")
    assert pos["net_tone"] == 1.0 and pos["positive"] >= 4
    assert neg["net_tone"] == -1.0 and neg["negative"] >= 5
    assert unc["uncertainty"] >= 4 and unc["uncertainty_share"] > pos["uncertainty_share"]
    negated = ea.tone("Results were not good and not strong.")
    assert negated["positive"] == 0 and negated["negative"] == 2
    lit = ea.tone("The plaintiff filed a lawsuit alleging patent infringement in federal court.")
    assert lit["litigious"] >= 4


def test_guidance_extraction():
    text = ("Revenue grew 12% year over year. We are raising our full-year revenue guidance to $10 billion. "
            "We reaffirm our outlook for operating margin. Given tariff uncertainty, we are lowering our fiscal 2026 EPS outlook. "
            "Our guidance assumes no further rate cuts.")
    g = ea.guidance_sentences(text)
    assert list(g["direction"]) == ["raise", "reaffirm", "lower", "mention"]
    s = ea.guidance_summary(g)
    assert (s["guidance"], s["g_raise"], s["g_lower"], s["g_reaffirm"]) == ("mixed", 1, 1, 1)
    w = ea.guidance_summary(ea.guidance_sentences("Due to the pandemic we are withdrawing our full-year guidance."))
    assert w["guidance"] == "withdraw" and w["guidance_score"] == -1
    r = ea.guidance_summary(ea.guidance_sentences("We raised our outlook. We are increasing our full-year forecast."))
    assert r["guidance"] == "raise" and r["guidance_score"] == 1
    assert ea.guidance_summary(ea.guidance_sentences("Nothing about the future here."))["guidance"] == "none"


TRANSCRIPT = """Operator: Good afternoon and welcome to the Example Corp fourth quarter earnings call.
Jane Doe (CEO): Thank you. We delivered an excellent quarter with record revenue and strong demand for our AI platform.
Margins improved and we are raising our full-year guidance. We will continue to invest in AI and expect pricing to remain robust.
We also repurchased $2 billion of stock under our buyback program.

John Roe (CFO): Gross margins expanded 200 basis points. We expect revenue growth to accelerate next quarter.

Operator: We will now begin the question-and-answer session. Our first question comes from an analyst.
Analyst: Can you talk about China and tariffs? Is there any risk to demand?
Jane Doe (CEO): There may be some uncertainty around tariffs and China could be volatile, but demand remains solid.
"""


def test_analyze_text_qa_topics_fls():
    a = ea.analyze_text(TRANSCRIPT)
    assert a["has_qa"]
    prep, qa = ea.split_qa(TRANSCRIPT)
    assert "question-and-answer" in qa and "buyback" in prep
    assert a["qa_net_tone"] < a["prepared_net_tone"]
    assert a["guidance"] == "raise"
    t = a["topics"]["mentions"]
    assert t["AI"] == 2 and t["China"] == 2 and t["tariffs"] == 2 and t["buyback"] == 2
    assert a["fls_per1k"] > 0 and 0 < a["fls_sentence_share"] <= 1
    assert "aid" not in ea.topic_counts("We said the paid plan was aimed well.").query("mentions > 0").index


def test_tone_history_and_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "TRANSCRIPTS_DIR", tmp_path)
    ea.save_transcript("TEST", "2024Q4", "Strong excellent record quarter. We are raising our full-year guidance.")
    ea.save_transcript("TEST", "2025Q1", "Weak quarter with losses and declining demand. We are lowering our full-year outlook.")
    ea.save_transcript("TEST", "2025Q2", "Solid quarter, good progress, some challenges. We reaffirm our full-year guidance.")
    docs = {k: ea.load_transcript_text(p) for k, p in ea.list_transcripts("TEST").items()}
    assert list(docs) == ["2024Q4", "2025Q1", "2025Q2"]
    th = ea.tone_history(docs)
    assert th.loc["2025Q1", "net_tone_chg"] < 0 < th.loc["2025Q2", "net_tone_chg"]
    assert list(th["guidance"]) == ["raise", "lower", "reaffirm"]

    rx = pd.DataFrame({"quarter": ["2024Q4", "2025Q1", "2025Q2", "2025Q3"], "surprise_pct": [5.0, -3.0, 1.0, 2.0],
                       "day1": [0.06, -0.08, 0.01, 0.02], "drift20": [0.05, -0.04, 0.0, 0.01]})
    tm = pd.DataFrame({k: ea.text_metrics(ea.analyze_text(v)) for k, v in docs.items()}).T
    j = ea.align_frames(rx, tm)
    assert list(j.index) == ["2024Q4", "2025Q1", "2025Q2", "2025Q3"]
    assert j.loc["2025Q1", "guidance"] == "lower" and pd.isna(j.loc["2025Q3", "net_tone"])
    assert j.loc["2025Q1", "net_tone_chg"] < 0
    st = ea.alignment_stats(j)
    assert st.loc[("guidance_score", "drift20"), "n"] == 3
    assert st.loc[("surprise_pct", "day1"), "n"] == 4 and st.loc[("surprise_pct", "day1"), "pearson"] > 0.9
    assert {"pearson", "p_value", "spearman", "ci95_lo", "ci95_hi"} <= set(st.columns)


def test_html_and_exhibit_pick():
    html = "<html><head><title>x</title></head><body><p>Revenue&nbsp;rose <b>10%</b>.</p><table><tr><td>EPS</td><td>$1.10</td></tr></table></body></html>"
    t = ea.html_to_text(html)
    assert "Revenue rose 10%" in t and "EPS" in t and "$1.10" in t and "<" not in t and "title" not in t.lower()
    names = ["0001-index.htm", "aapl-20250130.htm", "aapl-20250130xex991.htm", "aapl-20250130xex992.htm", "R1.htm"]
    assert ea.pick_exhibit_991(names) == "aapl-20250130xex991.htm"
    assert ea.pick_exhibit_991(["d8k.htm", "dex991.htm"]) == "dex991.htm"
    assert ea.pick_exhibit_991(["form8-k.htm", "exhibit99-1.htm"]) == "exhibit99-1.htm"
    assert ea.pick_exhibit_991(["form8-k.htm"]) is None


def test_attach_revenue():
    ev = ea.normalize_earnings(raw_dates([("2025-01-30 16:05", 1, 1.1, 10), ("2025-05-01 16:05", 1, 1.1, 10)]))
    rev = pd.Series([90.0, 95, 97, 99, 100, 110], index=pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30",
                                                                         "2024-12-31", "2025-03-31"]))
    out = ea.attach_revenue(ev, rev)
    assert out["revenue"].tolist() == [100, 110]
    assert out["revenue_yoy"].iloc[0] == pytest.approx(100 / 90 - 1)
    assert math.isclose(out["revenue_yoy"].iloc[1], 110 / 95 - 1)


def test_sec_press_release_path(tmp_path, monkeypatch):
    from mlab.providers import sec
    monkeypatch.setattr(ea, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(sec, "cik", lambda t: "0000000123")
    sub = {"filings": {"recent": {"form": ["8-K", "8-K", "10-Q"], "items": ["2.02,9.01", "5.02", ""],
                                  "filingDate": ["2025-04-17", "2025-03-01", "2025-05-01"], "accessionNumber": ["0001-25-000010", "0001-25-000009", "0001-25-000011"],
                                  "primaryDocument": ["x8k.htm", "y8k.htm", "q.htm"]}}}
    idx = {"directory": {"item": [{"name": "x8k.htm"}, {"name": "x-ex991.htm"}, {"name": "0001-index.htm"}]}}
    monkeypatch.setattr(sec, "_get", lambda url: sub if "submissions" in url else idx)
    seen = []
    monkeypatch.setattr(ea, "_sec_text", lambda url: seen.append(url) or "<p>We are raising our full-year guidance.</p>")
    rel = ea.fetch_earnings_releases("TEST")
    assert len(rel) == 1 and rel.loc[0, "folder"].endswith("/123/000125000010/")
    ev = ea.normalize_earnings(raw_dates([("2025-04-16 16:10", 1.0, 1.1, 10.0)]))
    texts, srcs = ea.quarter_documents("TEST", ev)
    assert srcs == {"2025Q1": "8-K ex99.1"} and "raising" in texts["2025Q1"]
    assert seen == ["https://www.sec.gov/Archives/edgar/data/123/000125000010/x-ex991.htm"]
    ea.quarter_documents("TEST", ev)  # second call served from the press cache
    assert len(seen) == 1
