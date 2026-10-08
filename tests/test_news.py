"""Offline tests for mlab.news: fixture payloads, scorer, tagging, dedupe, aggregation, brief() resilience."""
import numpy as np
import pandas as pd
import pytest

from mlab import cache, news

NOW = pd.Timestamp.now(tz="UTC").floor("min")
RFC = lambda t: t.strftime("%a, %d %b %Y %H:%M:%S GMT")  # noqa: E731

GOOGLE_RSS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<generator>NFE/5.0</generator><title>"NVDA" stock when:7d - Google News</title><link>https://news.google.com/search?q=NVDA</link>
<language>en-GB</language><lastBuildDate>{RFC(NOW)}</lastBuildDate>
<item><title>Nvidia beats estimates and raises guidance as AI demand soars - Reuters</title>
<link>https://news.google.com/rss/articles/CBMiAAA?oc=5</link><guid isPermaLink="false">CBMiAAA</guid>
<pubDate>{RFC(NOW - pd.Timedelta(hours=2))}</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMiAAA?oc=5" target="_blank"&gt;Nvidia beats estimates and raises guidance as AI demand soars&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Reuters&lt;/font&gt;</description>
<source url="https://www.reuters.com">Reuters</source></item>
<item><title>Nvidia faces DOJ antitrust probe over chip sales - Financial Times</title>
<link>https://news.google.com/rss/articles/CBMiBBB?oc=5</link><guid isPermaLink="false">CBMiBBB</guid>
<pubDate>{RFC(NOW - pd.Timedelta(hours=20))}</pubDate>
<description>&lt;a href="x"&gt;Nvidia faces DOJ antitrust probe over chip sales&lt;/a&gt;</description>
<source url="https://www.ft.com">Financial Times</source></item>
</channel></rss>"""

YAHOO_RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Yahoo! Finance: NVDA News</title>
<item><description>The chipmaker topped Wall Street expectations on data-center revenue.</description>
<guid isPermaLink="false">a1b2</guid><link>https://finance.yahoo.com/news/nvidia-beats-estimates-120000.html</link>
<pubDate>{RFC(NOW - pd.Timedelta(hours=3))}</pubDate><title>Nvidia beats estimates, raises guidance as AI demand soars</title></item>
<item><description>Analyst lifts target.</description><guid>c3d4</guid><link>https://finance.yahoo.com/news/ms-upgrade.html</link>
<pubDate>{RFC(NOW - pd.Timedelta(hours=30))}</pubDate><title>Morgan Stanley upgrades Nvidia to overweight, raises price target</title></item>
</channel></rss>"""

YF_NEWS = [
    {"id": "u1", "content": {"title": "Nvidia announces $50 billion share buyback program", "summary": "Board approves.",
                             "pubDate": (NOW - pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "canonicalUrl": {"url": "https://finance.yahoo.com/news/buyback.html"}, "provider": {"displayName": "Reuters"}}},
    {"uuid": "u2", "title": "Nvidia CFO sells shares", "link": "https://example.com/cfo",
     "providerPublishTime": int((NOW - pd.Timedelta(hours=8)).timestamp()), "publisher": "MarketWatch"},
]

GDELT_ART = {"articles": [
    {"url": "https://www.example.co.uk/nvda-record", "url_mobile": "", "title": "Nvidia posts record revenue on AI boom",
     "seendate": (NOW - pd.Timedelta(hours=6)).strftime("%Y%m%dT%H%M%SZ"), "socialimage": "", "domain": "example.co.uk",
     "language": "English", "sourcecountry": "United Kingdom"}]}


def _tl(name, values):
    days = pd.date_range(end=NOW.floor("D"), periods=len(values), freq="D")
    return {"query_details": {"title": "x", "date_resolution": "day"},
            "timeline": [{"series": name, "data": [{"date": d.strftime("%Y%m%dT%H%M%SZ"), "value": v} for d, v in zip(days, values)]}]}


STOCKTWITS = {"response": {"status": 200}, "symbol": {"symbol": "NVDA"}, "messages": [
    {"id": 101, "body": "$NVDA to the moon 🚀🚀 diamond hands", "created_at": (NOW - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "user": {"username": "bullguy", "followers": 10}, "entities": {"sentiment": {"basic": "Bullish"}}, "likes": {"total": 12}},
    {"id": 102, "body": "$NVDA overvalued, buying puts", "created_at": (NOW - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "user": {"username": "bearguy"}, "entities": {"sentiment": {"basic": "Bearish"}}, "likes": {"total": 3}},
    {"id": 103, "body": "$NVDA earnings on Wednesday", "created_at": (NOW - pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "user": {"username": "neutral"}, "entities": {"sentiment": None}},
]}

REDDIT = {"kind": "Listing", "data": {"after": None, "children": [
    {"kind": "t3", "data": {"title": "NVDA calls printing, short squeeze incoming", "selftext": "YOLO on weeklies",
                            "created_utc": (NOW - pd.Timedelta(hours=4)).timestamp(), "permalink": "/r/wallstreetbets/comments/abc/nvda/",
                            "author": "ape1", "score": 420, "num_comments": 69, "subreddit": "wallstreetbets"}},
    {"kind": "t3", "data": {"title": "Bagholder here, NVDA rug pull. guh", "selftext": "",
                            "created_utc": (NOW - pd.Timedelta(hours=9)).timestamp(), "permalink": "/r/stocks/comments/def/x/",
                            "author": "sad", "score": 15, "num_comments": 4, "subreddit": "stocks"}},
]}}

HN = {"hits": [{"title": "Nvidia unveils new Blackwell chip", "url": "https://blog.example.com/blackwell", "author": "pg",
                "points": 321, "num_comments": 120, "created_at_i": int((NOW - pd.Timedelta(hours=7)).timestamp()), "objectID": "1"}]}

AV = {"items": "1", "feed": [{"title": "Nvidia stock jumps", "url": "https://av.example/x", "time_published": NOW.strftime("%Y%m%dT%H%M%S"),
                             "summary": "s", "source": "Benzinga", "overall_sentiment_score": 0.2,
                             "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9", "ticker_sentiment_score": "0.41"}]}]}
FMP = [{"symbol": "NVDA", "publishedDate": NOW.strftime("%Y-%m-%d %H:%M:%S"), "title": "Nvidia slumps after downgrade", "site": "fool.com",
        "text": "t", "url": "https://fmp.example/x"}]
FINNHUB = [{"category": "company", "datetime": int(NOW.timestamp()), "headline": "Nvidia recall hits shares", "id": 1, "related": "NVDA",
            "source": "Yahoo", "summary": "s", "url": "https://fh.example/x"}]


# ------------------------------------------------------------------ parsing
def test_parse_google_rss_splits_publisher():
    df = news.parse_rss(GOOGLE_RSS, "google", "NVDA")
    assert list(df.columns) == news.COLUMNS
    assert len(df) == 2 and df["time"].dt.tz is not None
    assert df.iloc[0]["title"] == "Nvidia beats estimates and raises guidance as AI demand soars"
    assert df.iloc[0]["author"] == "Reuters" and df.iloc[0]["summary"] in ("", "Reuters")
    assert df["time"].is_monotonic_decreasing


def test_parse_yahoo_rss_and_yfinance_shapes():
    y = news.parse_rss(YAHOO_RSS, "yahoo", "NVDA")
    assert len(y) == 2 and "topped Wall Street" in y.iloc[0]["summary"]
    f = news.parse_yf_news(YF_NEWS, "NVDA")
    assert set(f["author"]) == {"Reuters", "MarketWatch"}
    assert f["url"].str.startswith("https://").all() and f["time"].notna().all()


def test_parse_gdelt_and_timeline():
    a = news.parse_gdelt_artlist(GDELT_ART, "NVDA")
    assert len(a) == 1 and a.iloc[0]["author"] == "example.co.uk"
    tone = news.parse_gdelt_timeline(_tl("Average Tone", [1.0, -2.0, 0.5]), "gdelt_tone")
    assert len(tone) == 3 and tone.iloc[1] == -2.0 and str(tone.index.tz) == "UTC"
    assert news.parse_gdelt_timeline({}, "gdelt_tone").empty


def test_parse_social_and_keyed():
    st = news.parse_stocktwits(STOCKTWITS, "NVDA")
    assert list(st["label"]) == ["Bullish", "Bearish", ""] and st.iloc[0]["engagement"] == 12
    rd = news.parse_reddit(REDDIT, "NVDA")
    assert rd.iloc[0]["engagement"] == 420 and rd.iloc[0]["url"].startswith("https://www.reddit.com/r/wallstreetbets")
    hn = news.parse_hn(HN, "NVDA")
    assert hn.iloc[0]["engagement"] == 321
    av = news.parse_alphavantage(AV, "NVDA")
    assert av.iloc[0]["vendor_score"] == pytest.approx(0.41)
    assert news.parse_fmp(FMP, "NVDA").iloc[0]["author"] == "fool.com"
    assert news.parse_finnhub(FINNHUB, "NVDA").iloc[0]["title"].startswith("Nvidia recall")
    with pytest.raises(LookupError):
        news.parse_alphavantage({"Information": "rate limit"}, "NVDA")
    sec = news.parse_sec_8k(pd.DataFrame([{"filingDate": "2026-10-01", "form": "8-K", "reportDate": "2026-09-30",
                                           "primaryDocDescription": "8-K", "url": "https://www.sec.gov/x"}]), "NVDA")
    assert sec.iloc[0]["source"] == "sec" and sec.iloc[0]["title"].startswith("8-K")


# ------------------------------------------------------------------ scorer
@pytest.mark.parametrize("text", [
    "Apple beats estimates and raises full-year guidance",
    "Record revenue as margins expand",
    "Board approves dividend hike and new share repurchase program",
    "Goldman upgrades Tesla to buy",
])
def test_positive_finance(text):
    assert news.score(text) > 0.3


@pytest.mark.parametrize("text", [
    "Apple misses estimates, cuts guidance",
    "SEC investigation into accounting irregularities",
    "Hindenburg short report alleges fraud",
    "Company files for bankruptcy",
    "Morgan Stanley downgrades Nvidia",
    "Automaker announces recall of 2 million vehicles",
])
def test_negative_finance(text):
    assert news.score(text) < -0.3


def test_negation_but_emphasis_boosters():
    assert news.score("Tesla beat estimates") > 0 > news.score("Tesla did not beat estimates")
    assert news.score("Tesla didn't beat estimates") < 0
    assert news.score("Revenue was strong but guidance was cut") < 0
    assert news.score("Guidance was cut but revenue was very strong") > 0
    assert news.score("GREAT quarter!!!") > news.score("great quarter") > 0
    assert news.score("extremely strong demand") > news.score("strong demand") > news.score("slightly strong demand")
    assert news.score("") == 0.0 and news.score(None) == 0.0
    assert -1 <= news.score("terrible awful worst crash collapse fraud!!!!") <= 1


def test_slang():
    assert news.score("TSLA to the moon 🚀🚀 diamond hands", social=True) > 0.6
    assert news.score("rug pull, bagholders everywhere, guh", social=True) < -0.6
    assert news.score("loading up on calls", social=True) > 0 > news.score("buying puts", social=True)
    # calls/puts are ambiguous in news prose and only count for social posts
    assert news.score("CEO calls meeting") == 0.0


def test_enrich_uses_stocktwits_labels():
    st = news.enrich(news.parse_stocktwits(STOCKTWITS, "NVDA"))
    assert st.iloc[0]["compound"] == 0.6 and st.iloc[1]["compound"] == -0.6
    assert st.iloc[2]["compound"] == st.iloc[2]["text_score"]


# ------------------------------------------------------------------ events
@pytest.mark.parametrize("title,tag", [
    ("Nvidia beats Q3 earnings estimates", "earnings"),
    ("Apple cuts full-year guidance", "guidance"),
    ("Microsoft agrees to buy Activision for $69bn", "m&a"),
    ("Goldman upgrades Nvidia to buy", "analyst_up"),
    ("Barclays downgrades Tesla to underweight, cuts price target", "analyst_down"),
    ("FTC sues Amazon over Prime", "regulatory_legal"),
    ("Boeing CEO steps down", "management"),
    ("Apple unveils new iPhone", "product"),
    ("Fed signals rate cuts as inflation cools", "macro"),
    ("Company raises quarterly dividend, adds buyback", "capital_return"),
    ("Hindenburg Research discloses short position in Super Micro", "short_report"),
    ("Nvidia CEO sells shares under 10b5-1 plan", "insider_trade"),
])
def test_tag_events(title, tag):
    assert tag in news.tag_events(title)


def test_upgrade_not_tagged_as_mna():
    assert "m&a" not in news.tag_events("Goldman upgrades Nvidia to Buy")


# ------------------------------------------------------------------ aggregation
def test_dedupe_merges_cross_source_near_duplicates():
    items = news.enrich(pd.concat([news.parse_rss(GOOGLE_RSS, "google", "NVDA"), news.parse_rss(YAHOO_RSS, "yahoo", "NVDA"),
                                   news.parse_stocktwits(STOCKTWITS, "NVDA")], ignore_index=True))
    dd = news.dedupe(items)
    beats = dd[dd["title"].str.contains("beats estimates")]
    assert len(beats) == 1 and beats.iloc[0]["source"] == "yahoo"  # higher weight wins
    assert beats.iloc[0]["n_sources"] == 2 and beats.iloc[0]["also_in"] == "google"
    assert (dd["source"] == "stocktwits").sum() == 3  # social posts never merged
    assert news.similar(news.norm_title("Fed holds rates steady - CNBC"), news.norm_title("Fed holds rates steady"))
    assert not news.similar(news.norm_title("Apple beats"), news.norm_title("Tesla recalls cars"))


def test_attention_zscore_flags_spike():
    vol = pd.Series([10.0, 12, 9, 11, 10, 13, 8, 10, 11, 12, 9, 10] * 3 + [60.0],
                    index=pd.date_range("2026-08-01", periods=37, freq="D", tz="UTC"))
    z = news.attention_z(vol)
    assert z.iloc[:10].isna().all()
    assert z.iloc[-1] > 5 and abs(z.iloc[-2]) < 2


def test_daily_index_with_timeline():
    items = news.enrich(pd.concat([news.parse_rss(GOOGLE_RSS, "google", "NVDA"), news.parse_stocktwits(STOCKTWITS, "NVDA"),
                                   news.parse_reddit(REDDIT, "NVDA")], ignore_index=True))
    tl = pd.concat([news.parse_gdelt_timeline(_tl("Average Tone", list(np.linspace(-1, 1, 40))), "gdelt_tone"),
                    news.parse_gdelt_timeline(_tl("Article Count", [100, 110, 90, 105, 95] * 8 + [400]), "gdelt_volume")], axis=1, sort=True)
    d = news.daily_index(items, tl)
    assert {"sentiment", "mentions", "bull_ratio", "gdelt_tone", "gdelt_volume", "attention_z"} <= set(d.columns)
    assert d["mentions"].sum() == len(items)
    assert 0 <= d["bull_ratio"].dropna().iloc[-1] <= 1
    assert d["attention_z"].dropna().iloc[-1] > 2


def test_sentiment_vs_price_lead_lag_on_synthetic():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-01-05", periods=160)
    sent = pd.Series(rng.normal(0, 1, len(days)), index=days)
    ret = sent.shift(1).fillna(0) * 0.01 + rng.normal(0, 0.002, len(days))  # sentiment leads returns by one day
    close = (100 * (1 + ret).cumprod()).values
    prices = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=days.tz_localize("UTC"))
    daily = pd.DataFrame({"sentiment": sent.values, "mentions": 5.0, "bull_ratio": 0.5 + 0.1 * np.tanh(sent.values),
                          "attention_z": rng.normal(0, 1, len(days))}, index=days.tz_localize("UTC"))
    r = news.sentiment_vs_price("SYN", daily=daily, prices=prices)
    ll = r["lead_lag"]
    assert ll.loc["sentiment", "t+1"] > 0.8
    assert abs(ll.loc["sentiment", "t+3"]) < 0.3
    assert {"t-1", "t+0", "t+5", "n"} <= set(ll.columns)
    assert "attention_z~|ret|" in ll.index


def test_contrarian_extremes():
    days = pd.bdate_range("2026-03-02", periods=60)
    close = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 160, 30)], index=days)  # strong rally -> RSI > 70
    prices = pd.DataFrame({"close": close.values}, index=days.tz_localize("UTC"))
    br = np.r_[np.full(55, 0.55), np.full(5, 0.95)]
    daily = pd.DataFrame({"sentiment": 0.1, "mentions": 3.0, "bull_ratio": br}, index=days.tz_localize("UTC"))
    ext = news.sentiment_vs_price("X", daily=daily, prices=prices)["extremes"]
    assert len(ext) >= 3 and ext["flag"].str.contains("contrarian bearish").all()


def test_weekend_news_rolls_to_monday():
    daily = pd.DataFrame({"sentiment": [0.5, -0.5], "mentions": [2.0, 4.0]},
                         index=pd.DatetimeIndex(["2026-10-03", "2026-10-04"], tz="UTC"))  # Sat, Sun
    s = news._sessions(daily, pd.bdate_range("2026-10-01", "2026-10-09"))
    assert list(s.index) == [pd.Timestamp("2026-10-05")] and s.iloc[0]["mentions"] == 6 and s.iloc[0]["sentiment"] == 0


def test_rank_prefers_recent_strong_weighted():
    df = pd.DataFrame({"time": [NOW - pd.Timedelta(hours=1), NOW - pd.Timedelta(days=4), NOW - pd.Timedelta(hours=1)],
                       "source": ["sec", "google", "reddit"], "compound": [0.8, 0.8, 0.8], "title": ["a", "b", "c"]})
    r = news.rank_headlines(df, NOW)
    assert list(r["title"]) == ["a", "c", "b"]


def test_make_query():
    q = news.make_query("NVDA", name="Nvidia")
    assert q.ticker == "NVDA" and "Nvidia" in news.gdelt_query(q)
    t = news.make_query("UK inflation", 3)
    assert t.ticker is None and news.gdelt_query(t).startswith('"UK inflation"')


# ------------------------------------------------------------------ brief end to end (offline)
@pytest.fixture
def tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("MLAB_NEWS_DIR", str(tmp_path / "news_history"))
    monkeypatch.setattr(news, "GDELT_PAUSE", 0)
    return tmp_path


def _boom(*a, **k):
    raise ConnectionError("blocked by network policy")


def test_brief_survives_all_sources_failing(monkeypatch, tmp_cache):
    for fn in ("_get", "_yf_news_raw", "_yf_name", "_sec_8k_raw"):
        monkeypatch.setattr(news, fn, _boom)
    monkeypatch.setattr(news, "gdelt_timeline", lambda qc, days=90, pause=0: _boom())
    r = news.brief("NVDA", 7)
    assert r["items"] == 0 and r["headlines"].empty
    assert {"google", "yahoo", "yfinance", "gdelt", "sec", "stocktwits", "reddit", "hn", "gdelt_timeline"} <= set(r["errors"])
    assert all("ConnectionError" in v for v in r["errors"].values())
    assert np.isnan(r["sentiment_7d"]) and r["attention_spike"] is False
    topic = news.brief("UK inflation", 3)
    assert topic["items"] == 0 and "sec" not in topic["errors"]


def test_brief_with_fixtures(monkeypatch, tmp_cache):
    def fake_get(url, params=None, kind="json", **k):
        if "news.google.com" in url:
            return GOOGLE_RSS
        if "feeds.finance.yahoo.com" in url:
            return YAHOO_RSS
        if "gdeltproject" in url:
            mode = params["mode"]
            return GDELT_ART if mode == "ArtList" else _tl("Average Tone", [0.5] * 40) if mode == "TimelineTone" \
                else _tl("Article Count", [50, 55, 45, 52, 48] * 8 + [300])
        if "stocktwits" in url:
            return STOCKTWITS
        if "reddit" in url:
            return REDDIT
        if "algolia" in url:
            return HN
        raise ConnectionError(url)
    monkeypatch.setattr(news, "_get", fake_get)
    monkeypatch.setattr(news, "_yf_news_raw", lambda t: YF_NEWS)
    monkeypatch.setattr(news, "_yf_name", lambda t: "NVIDIA Corporation")
    monkeypatch.setattr(news, "_sec_8k_raw", _boom)
    r = news.brief("NVDA", 7)
    assert r["items"] > 8 and "sec" in r["errors"] and "google" in r["sources_ok"]
    assert len(r["headlines"]) and r["headlines"].iloc[0]["compound"] != 0
    assert r["sentiment_7d"] > 0 and r["attention_spike"] is True
    assert 0 < r["social_bull_ratio"] < 1
    assert "earnings" in r["events"].index
    # history was stored (one row per unique (time, uid)), so a second run sees it
    h = news.history("NVDA")
    assert len(h) >= r["unique_headlines"] and not h.duplicated(["time", "uid"]).any()
    news.brief("NVDA", 7)  # a second fetch adds a file, never duplicates rows
    assert len(news.history("NVDA")) == len(h)


def test_rss_parser_rejects_entity_expansion():
    """Feeds are untrusted: a billion-laughs payload must be refused, not expanded."""
    import pytest
    from defusedxml import EntitiesForbidden

    from mlab.news import parse_rss
    bomb = ('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]>'
            '<rss><channel><item><title>&b;</title></item></channel></rss>')
    with pytest.raises(EntitiesForbidden):
        parse_rss(bomb, "google")
