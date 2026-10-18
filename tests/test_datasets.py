"""Frozen datasets: build, immutability, checksums, splits, decoders. Offline (synthetic fetchers)."""
import io
import lzma
import os
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import synth_ohlcv


@pytest.fixture()
def ds(tmp_path, monkeypatch):
    from mlab import datasets
    monkeypatch.setattr(datasets, "DATASETS_DIR", tmp_path)
    return datasets


def _fake(m, spec, log=print):
    return synth_ohlcv(n=6000, seed=hash(m.symbol) % 100, start="2000-01-03")


def test_build_load_splits_and_verify(ds):
    path = ds.build("xasset-daily", members=["^FTSE", "SPY"], fetch=_fake, log=lambda *a: None)
    man = ds.manifest("xasset-daily")
    assert man["version"] == path.name and set(man["members"]) == {"^FTSE", "SPY"}
    assert man["splits"]["train"][1] == "2014-12-31" and man["splits"]["validation"][0] == "2015-01-01"
    assert ds.verify("xasset-daily")["ok"]

    tr = ds.load("xasset-daily", ["^FTSE"], split="train")["^FTSE"]
    va = ds.load("xasset-daily", ["^FTSE"], split="validation")["^FTSE"]
    assert tr.index.max() < pd.Timestamp("2015-01-01", tz="UTC") <= va.index.min()
    assert va.index.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert list(tr.columns[:4]) == ["open", "high", "low", "close"]
    assert "dataset xasset-daily@" in tr.attrs["source"]

    with pytest.raises(PermissionError):
        ds.load("xasset-daily", split="test")
    te = ds.load("xasset-daily", ["SPY"], split="test", allow_test=True, note="t")["SPY"]
    assert te.index.min() >= pd.Timestamp("2020-01-01", tz="UTC")
    assert (ds.DATASETS_DIR / "xasset-daily" / f"test_access.{path.name}.jsonl").exists()


def test_versions_are_immutable_and_tamper_is_caught(ds):
    v1 = ds.build("crypto-daily", fetch=_fake, log=lambda *a: None)
    v2 = ds.build("crypto-daily", fetch=_fake, log=lambda *a: None)
    assert v1 != v2 and ds.versions("crypto-daily") == [v1.name, v2.name]
    assert ds.resolve("crypto-daily")[1] == v2.name
    f = next(v1.rglob("data.parquet"))
    os.chmod(f, 0o600)
    f.write_bytes(f.read_bytes() + b"x")
    r = ds.verify(f"crypto-daily@{v1.name}")
    assert not r["ok"] and r["bad"]


def test_failed_member_is_recorded(ds):
    def flaky(m, spec, log=print):
        if m.symbol == "SPY":
            raise LookupError("down")
        return _fake(m, spec)
    ds.build("xasset-daily", members=["^FTSE", "SPY"], fetch=flaky, log=lambda *a: None)
    man = ds.manifest("xasset-daily")
    assert list(man["members"]) == ["^FTSE"] and "SPY" in man["failed"]


def test_dukascopy_decoder():
    from mlab.datasets import _duka_decode
    recs = [(0, 745520, 745191, 744888, 746179, 0.3), (3600, 745191, 746339, 745188, 747171, 0.6),
            (7200, 746000, 746000, 746000, 746000, 0.0)]  # last one is a filler bar
    blob = lzma.compress(b"".join(struct.pack(">I4if", *r) for r in recs))
    df = _duka_decode(blob, pd.Timestamp("2023-06-01", tz="UTC"), 100.0)
    assert len(df) == 2
    assert df.iloc[0][["open", "high", "low", "close"]].tolist() == [7455.20, 7461.79, 7448.88, 7451.91]
    assert df.index[1] == pd.Timestamp("2023-06-01 01:00", tz="UTC")


def test_binance_zip_ms_and_us_timestamps():
    from mlab.datasets import _binance_zip
    ms = 1704067200000  # 2024-01-01 in ms
    rows = [f"{ms},1,2,0.5,1.5,10,0,0,0,0,0,0", f"{(ms + 86_400_000) * 1000},1,2,0.5,1.6,11,0,0,0,0,0,0"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "\n".join(rows))
    df = _binance_zip(buf.getvalue())
    assert list(df.index) == [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")]
    assert np.allclose(df["close"], [1.5, 1.6])


def test_algo_reads_dataset_split(ds):
    from types import SimpleNamespace

    from mlab.quant.cli import _frames
    ds.build("xasset-daily", members=["^FTSE"], fetch=_fake, log=lambda *a: None)
    a = SimpleNamespace(dataset="xasset-daily", symbols=["^FTSE"], split="validation", allow_test=False,
                        action="backtest", strategy="tsmom")
    f = _frames(a)["^FTSE"]
    assert f.index.min() >= pd.Timestamp("2015-01-01", tz="UTC") and f.index.max() < pd.Timestamp("2020-01-01", tz="UTC")


def test_events_dataset(ds):
    def fake(m, spec, log=print):
        idx = pd.to_datetime(["2012-01-30 21:00", "2016-04-25 21:00", "2021-07-26 20:00"], utc=True)
        return pd.DataFrame({"eps_estimate": [1.0, 1.1, 1.2], "eps_actual": [1.1, 1.0, np.nan],
                             "surprise_pct": [10.0, -9.1, np.nan], "hour_et": [16.0, 17.0, 16.0]}, index=idx)
    ds.build("us-earnings", members=["AMZN"], fetch=fake, log=lambda *a: None)
    man = ds.manifest("us-earnings")
    assert man["kind"] == "events" and man["members"]["AMZN"]["qc"]["with_actual"] == 2
    va = ds.load("us-earnings", ["AMZN"], split="validation")["AMZN"]
    assert len(va) == 1 and va["surprise_pct"].iloc[0] == -9.1


def test_accumulate_merges_previous_version(ds):
    calls = {"n": 0}

    def window(m, spec, log=print):  # each build sees a later, overlapping 10-day window
        calls["n"] += 1
        end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=10 * (2 - calls["n"]))
        idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="h")
        c = np.linspace(100, 110, len(idx)) + calls["n"]
        return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}, index=idx)
    first = ds.build("us-stocks-1h", members=["AMZN"], fetch=window, log=lambda *a: None)
    second = ds.build("us-stocks-1h", members=["AMZN"], fetch=window, log=lambda *a: None)
    a = ds.load(f"us-stocks-1h@{first.name}", allow_test=True)["AMZN"]
    b = ds.load(f"us-stocks-1h@{second.name}", allow_test=True)["AMZN"]
    assert b.index.min() == a.index.min() and b.index.max() > a.index.max()
    assert not b.index.duplicated().any()
    assert ds.start_ts(ds.CATALOG["us-stocks-1m"]) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)


class _Resp:
    status_code = 200
    content = b"{}"

    def __init__(self, j):
        self._j = j

    def json(self):
        return self._j

    def raise_for_status(self):
        pass


def _massive_session(pages, seen):
    class S:
        headers = {}

        def mount(self, *a):
            pass

        def get(self, url, params=None, timeout=None):
            seen.append((url, params))
            return _Resp(pages[min(len(seen), len(pages)) - 1])
    return S()


def test_massive_pages_through_next_url(monkeypatch):
    from mlab.providers import massive
    pages = [{"results": [{"t": 1704205800000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}],
              "next_url": "https://api.massive.com/next?cursor=abc"},
             {"results": [{"t": 1704205860000, "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 50}]}]
    seen = []
    monkeypatch.setattr("mlab.net.session", lambda *a, **k: _massive_session(pages, seen))
    monkeypatch.setenv("MLAB_MASSIVE_GAP", "0")
    df = massive.history("AMZN", "2024-01-02", "2024-01-02", "1m")
    assert len(df) == 2 and df["close"].tolist() == [1.5, 1.8]
    assert seen[0][1]["adjusted"] == "true" and seen[1] == ("https://api.massive.com/next?cursor=abc", None)
    assert df.index[0] == pd.Timestamp("2024-01-02 14:30", tz="UTC")
    with pytest.raises(ValueError):
        massive.history("^GSPC", interval="1d")


def test_massive_backup_is_never_cached(monkeypatch, tmp_path):
    from mlab import cache, data
    from mlab.providers import massive, yahoo
    saved = []
    monkeypatch.setattr(yahoo, "history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yahoo down")))
    idx = pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0], "volume": 1.0}, index=idx)
    monkeypatch.setattr(massive, "history", lambda *a, **k: bars)
    monkeypatch.setattr(cache, "load", lambda *a, **k: None)
    monkeypatch.setattr(cache, "save", lambda p, *a, **k: saved.append(p) or a[-1])
    df = data.get_prices("AMZN", start="2026-01-05", end="2026-01-07")
    assert df.attrs["source"].startswith("massive") and saved == []


def test_crosscheck_compares_without_storing(ds, monkeypatch):
    from mlab.providers import massive
    now = pd.Timestamp.now(tz="UTC").normalize()
    idx = pd.date_range(now - pd.Timedelta(days=9), now - pd.Timedelta(days=1), freq="D")

    def fake(m, spec, log=print):
        return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=idx)
    ds.build("us-stocks-daily", members=["AMZN"], fetch=fake, log=lambda *a: None)
    theirs = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.1, "volume": 20.0},
                          index=idx[-5:] + pd.Timedelta(hours=4))  # Massive daily bars stamp midnight New York
    monkeypatch.setattr(massive, "history", lambda *a, **k: theirs)
    before = sorted(p.name for p in ds.DATASETS_DIR.rglob("*"))
    r = ds.crosscheck("us-stocks-daily", days=5, log=lambda *a: None).iloc[0]
    assert r["matched"] == 5 and abs(r["close_bp_median"] - 9.99) < 0.05 and r["vol_ratio_median"] == 0.5
    assert sorted(p.name for p in ds.DATASETS_DIR.rglob("*")) == before


def test_listing_ignores_staging(ds):
    (ds.DATASETS_DIR / ".staging" / "duka-stocks-1m").mkdir(parents=True)
    assert ds.listing().empty
