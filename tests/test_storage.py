"""Shared-folder safety: atomic writes, write-once journal events, unprunable news history."""
import importlib
import multiprocessing as mp

import pandas as pd
import pytest


@pytest.fixture
def jr(tmp_path, monkeypatch):
    monkeypatch.setenv("MLAB_JOURNAL_DIR", str(tmp_path / "journal"))
    import mlab.journal
    importlib.reload(mlab.journal)
    yield mlab.journal
    monkeypatch.undo()
    importlib.reload(mlab.journal)


CARD = {"bias": "long", "instrument": "X", "entry_zone": [100, 102], "stop": 95, "targets": [110]}


def _add(n):
    import mlab.journal as j
    for i in range(n):
        j.add({**CARD, "instrument": f"X{i}"})


def test_parallel_writers_lose_nothing(jr):
    procs = [mp.get_context("spawn").Process(target=_add, args=(10,)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert len(jr.table()) == 40  # a read-modify-write journal loses entries here


def test_update_after_someone_else_added(jr):
    a = jr.add(CARD)
    jr.add({**CARD, "instrument": "Y"})  # another thread's entry lands in between
    jr.update(a["id"], status="open", fill=101)
    jr.update(a["id"], status="closed", exit_price=113)
    t = jr.table().set_index("id")
    assert len(t) == 2 and t.loc[a["id"], "status"] == "closed" and t.loc[a["id"], "R"] == 2.0


def test_legacy_jsonl_still_read(jr):
    jr.JOURNAL_DIR.mkdir(parents=True)
    row = {"id": "old1", "created": "2026-10-01T00:00:00+00:00", "status": "idea", "card": CARD,
           "events": [{"t": "2026-10-01T00:00:00+00:00", "status": "idea", "note": ""}]}
    jr.FILE.write_text(pd.Series([row]).apply(__import__("json").dumps).str.cat(sep="\n") + "\n")
    jr.update("old1", status="passed")
    assert jr.table().set_index("id").loc["old1", "status"] == "passed"
    assert "old1" in jr.FILE.read_text()  # the legacy file itself is never rewritten


def test_atomic_write_cleans_up_on_failure(tmp_path):
    from mlab.storage import atomic_write, atomic_write_text
    atomic_write_text(tmp_path / "a.txt", "v1")

    def boom(_):
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        atomic_write(tmp_path / "a.txt", boom)
    assert (tmp_path / "a.txt").read_text() == "v1" and len(list(tmp_path.iterdir())) == 1


def test_prune_never_touches_news(tmp_path, monkeypatch):
    from mlab import cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    df = pd.DataFrame({"close": range(50_000)}, index=pd.date_range("2000", periods=50_000, freq="h", tz="UTC"))
    for prov in ("yahoo", "news"):
        p = cache.path_for(prov, "X", "1d")
        p.parent.mkdir(parents=True)
        df.to_parquet(p)
    cache.prune(max_mb=0)
    assert not cache.path_for("yahoo", "X", "1d").exists() and cache.path_for("news", "X", "1d").exists()
