"""Forward lab: write-once store, registry freezing, honest fills, catch-up equivalence, option marks, scoring."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest
from conftest import synth_ohlcv

from mlab.forward import context, costs, linear, optsim, score, specs, store


@pytest.fixture()
def fwd_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MLAB_FORWARD_DIR", str(tmp_path / "fwd"))
    monkeypatch.setenv("MLAB_LEDGER_DIR", str(tmp_path / "ledger"))
    return tmp_path


def _day(df):
    return context._day_index(df)


# ---- store ---------------------------------------------------------------------------------
def test_write_once_refuses_overwrite_and_verifies(fwd_dir):
    p = fwd_dir / "x.bin"
    sha = store.write_once(p, b"abc")
    assert sha == store.sha256_bytes(b"abc")
    with pytest.raises(FileExistsError):
        store.write_once(p, b"other")
    assert p.read_bytes() == b"abc"


def test_events_are_stamped_with_schema_and_read_back(fwd_dir):
    store.append_events("signals", "2026-11-20T2040Z-aaaa", [{"cell": "a", "ret_raw": 0.01}])
    store.append_events("signals", "2026-11-21T2040Z-bbbb", [{"cell": "a", "ret_raw": -0.02}])
    df = store.read_events("signals")
    assert list(df["ret_raw"]) == [0.01, -0.02]
    assert set(df["schema"]) == {"fwd.signals/1"}
    with pytest.raises(ValueError):
        store.append_events("nonsense", "2026-11-20T2040Z-aaaa", [{"x": 1}])
    with pytest.raises(FileExistsError):
        store.append_events("signals", "2026-11-20T2040Z-aaaa", [{"cell": "b"}])


def test_run_lock_blocks_a_second_run(fwd_dir):
    with store.RunLock("r1"), pytest.raises(RuntimeError), store.RunLock("r2"):
        pass
    with store.RunLock("r3"):
        pass


# ---- costs ---------------------------------------------------------------------------------
def test_costs_route_by_asset_class_and_direction():
    assert costs.asset_class("crypto:BTC") == "crypto"
    assert costs.asset_class("EURUSD=X") == "fx"
    assert costs.asset_class("GC=F") == "commodity"
    assert costs.asset_class("^FTSE") == "index"
    assert costs.asset_class("VWRP.L") == "uk_etf"
    assert costs.asset_class("SHEL.L") == "uk_share"
    assert costs.asset_class("SPY") == "us_etf"
    assert costs.asset_class("NVDA") == "us_share"
    assert costs.venue_for("NVDA", 1).name == "isa_us_share"
    assert costs.venue_for("NVDA", -1).name == "spreadbet_share"


def test_fill_cost_exact_values():
    # buy 100% of 1.0 equity of a US share: 2bp spread + 70bp FX = 72bp
    c, v = costs.fill_cost("NVDA", 0.0, 1.0, 1.0)
    assert c == pytest.approx(0.0072) and v == ["isa_us_share"]
    # UK share buy pays stamp, sale does not
    assert costs.fill_cost("SHEL.L", 0.0, 1.0, 1.0)[0] == pytest.approx(0.0055)
    assert costs.fill_cost("SHEL.L", 1.0, 0.0, 1.0)[0] == pytest.approx(0.0005)
    # flip long -> short: close on the ISA (72bp) then open the short spread bet (10bp)
    c, v = costs.fill_cost("NVDA", 1.0, -1.0, 1.0)
    assert c == pytest.approx(0.0082) and v == ["isa_us_share", "spreadbet_share"]


def test_funding_long_pays_short_index_receives_net_of_admin():
    long_f = costs.funding_cost("^FTSE", 1.0, 1.0, 365)
    short_f = costs.funding_cost("^FTSE", -1.0, 1.0, 365)
    assert long_f == pytest.approx(costs.REF_RATE + 0.025)
    assert short_f == pytest.approx(0.025 - costs.REF_RATE)
    assert costs.funding_cost("NVDA", 1.0, 1.0, 365) == 0.0  # cash shares in an ISA: no funding


# ---- registry ------------------------------------------------------------------------------
def test_register_freezes_and_refuses_a_changed_spec(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    res = specs.register(path, now="2026-11-20T00:00:00+00:00")
    reg = specs.load(path)
    assert len(res["added"]) == reg["counts"]["cells"] > 1000
    assert specs.verify(reg) == []
    again = specs.register(path)
    assert again["added"] == [] and again["unchanged"] == reg["counts"]["cells"]
    # a cell whose spec is edited in place is refused; tampering with the file is caught by verify
    monkeypatch.setattr(specs, "SINGLE_MATRIX", [("sma_cross", "", {"fast": 20}, {})] + specs.SINGLE_MATRIX[1:])
    with pytest.raises(ValueError, match="bump the version"):
        specs.register(path)
    reg["cells"][0]["params"] = {"fast": 21}
    assert reg["cells"][0]["id"] in specs.verify(reg)


def test_every_instrument_and_basket_has_its_controls():
    cells = specs.generate()
    ids = {c["id"] for c in cells}
    for c in cells:
        if c["role"] == "candidate":
            assert score.null_of(c["id"], c["kind"]) in ids, c["id"]
    assert len(ids) == len(cells)


# ---- linear books ----------------------------------------------------------------------------
def _cell(strategy="sma_cross", params=None, sym="TEST", overlays=None):
    c = {"id": f"{strategy}.v1:{sym}", "kind": "linear", "role": "candidate", "strategy": strategy,
         "params": params or {}, "overlays": overlays or {}, "symbols": [sym], "group": "us_share"}
    c["spec_sha"] = "x"
    return c


def test_desk_target_rules():
    assert linear.desk_target(0.0, 1.0, 0.0, ok=True) == 1.0      # fresh trigger, filters pass
    assert linear.desk_target(0.0, 1.0, 0.0, ok=False) == 0.0     # skipped
    assert linear.desk_target(1.0, 1.0, 0.0, ok=True) == 0.0      # missed entry is not chased later
    assert linear.desk_target(1.0, 0.0, 1.0, ok=False) == 0.0     # exits always follow
    assert linear.desk_target(1.0, 0.5, 1.0, ok=False) == 0.5     # resizing on the same side follows
    assert linear.desk_target(1.0, -1.0, 1.0, ok=False) == 0.0    # blocked flip goes flat


def test_first_bar_only_then_fill_at_next_open_with_cost():
    df = _day(synth_ohlcv(400, seed=3))
    frames = {"TEST": df}
    cell = _cell("always_long")
    cut0 = df.index[300]
    st, sig, trig, fil = linear.step_cell(cell, None, frames, pd.DataFrame(), {}, {}, cut0, {"run_id": "r0"})
    assert len(sig) == 1 and sig[0]["bar"] == str(df.index[299].date()) and sig[0]["first"]
    assert fil == [] and trig[0]["kind"] == "entry"
    st, sig, trig, fil = linear.step_cell(cell, st, frames, pd.DataFrame(), {}, {}, df.index[301], {"run_id": "r1"})
    assert len(fil) == 2  # raw and desk both buy (no context: filters n/a)
    o, c = df["open"].iloc[300], df["close"].iloc[300]
    fee = costs.VENUES["isa_us_share"].side_bps(True) / 1e4
    assert fil[0]["price"] == pytest.approx(o)
    assert sig[0]["equity_raw"] == pytest.approx(1 - fee + (c / o - 1))


def test_incremental_runs_equal_one_catch_up_run():
    """Running day by day must give exactly what one catch-up run over the same bars gives (no look-ahead,
    no dependence on run timing)."""
    df = _day(synth_ohlcv(600, seed=11))
    frames = {"TEST": df}
    cell = _cell("donchian", {"entry": 20, "exit": 10}, overlays={"vol_target": {"target": 0.15}})
    st_a, _, _, _ = linear.step_cell(cell, None, frames, pd.DataFrame(), {}, {}, df.index[400], {})
    st_b = json.loads(json.dumps(st_a))
    for i in range(401, 460):
        st_a, *_ = linear.step_cell(cell, st_a, frames, pd.DataFrame(), {}, {}, df.index[i], {})
    st_b, sig_b, *_ = linear.step_cell(cell, st_b, frames, pd.DataFrame(), {}, {}, df.index[459], {})
    assert len(sig_b) == 59
    assert st_a["raw"]["equity"] == pytest.approx(st_b["raw"]["equity"], rel=1e-12)
    assert st_a["raw"]["trades"] == st_b["raw"]["trades"]


def test_filters_skip_a_gated_long_and_log_it():
    f = linear.filters("NVDA", 1, {"gate_sox": False, "gate_all_fail": False}, {"atr_pct": 0.03},
                       pd.Timestamp("2026-11-18"), {"NVDA": ["2026-11-19"]})
    assert f["L027_index_gate"] is False and f["L020_earnings"] is False and f["earnings_date"] == "2026-11-19"
    assert f["L019_cost_in_R"] is False and f["cost_R"] == pytest.approx(144 / 600, abs=1e-3)  # 0.24R > 0.2R
    assert not linear.passes(f)
    # shorts are not gated by the index regime
    s = linear.filters("NVDA", -1, {"gate_sox": False, "gate_all_fail": True}, {"atr_pct": 0.03},
                       pd.Timestamp("2026-11-01"), {"NVDA": ["2026-11-19"]})
    assert s["L027_index_gate"] is None and s["L028_all_gates_fail"] is None and linear.passes(s)


# ---- options ---------------------------------------------------------------------------------
def _chain(spot=100.0, vol=0.25, now=pd.Timestamp("2026-11-20 18:00", tz="UTC")):
    from mlab import options as opt
    rows = []
    for exp in ("2026-12-26", "2027-01-16", "2027-02-20"):
        T = (pd.Timestamp(exp) - now.tz_localize(None)).days / 365
        for k in np.arange(70, 131, 2.5):
            for t in "CP":
                iv = vol + 0.1 * (1 - k / spot)  # downside skew
                p = float(opt.bs_price(spot, k, T, 0.04, iv, 0.0, t))
                rows.append({"strike": k, "bid": max(p - 0.05, 0.01), "ask": p + 0.05, "lastPrice": p,
                             "volume": 10, "openInterest": 100, "impliedVolatility": iv, "expiry": exp, "type": t})
    return opt.normalize_chain(pd.DataFrame(rows), spot, 0.04, now=now)


def test_option_entry_crosses_spread_and_greeks_are_consistent():
    ch = _chain()
    cell = {"id": "iron_condor.v1:TEST", "role": "candidate", "symbols": ["TEST"], "structure": "iron_condor",
            "spec": json.loads(json.dumps(specs.OPTION_STRUCTURES["iron_condor"])), "spec_sha": "x",
            "exits": {"dte_exit": 21, "take_profit": 0.5, "stop_loss": 2.0}}
    vs = optsim.vol_snapshot(ch, 100.0, None)
    st, sig, trig, fil, grk = optsim.step_option(cell, None, ch, 100.0, 0.04, vs, "2026-11-20", {}, None)
    assert trig[0]["taken_raw"] and fil[0]["action"] == "open"
    assert fil[0]["slippage"] == pytest.approx(-4 * 0.05 * 100)     # 4 legs x half-spread x 100, a cost
    assert fil[0]["commission"] == 4.0
    assert sig["pnl_day"] == pytest.approx(fil[0]["slippage"] - 4.0)
    tot = grk[-1]
    assert abs(tot["delta"]) < 15 and tot["theta_day"] > 0 and tot["vega_pt"] < 0  # short premium profile
    # next day, same chain shifted up 1: P&L attribution sums back to the day's P&L
    ch2 = _chain(spot=101.0, now=pd.Timestamp("2026-11-21 18:00", tz="UTC"))
    _, sig2, _, _, _ = optsim.step_option(cell, st, ch2, 101.0, 0.04, vs, "2026-11-21", {}, None)
    a = sig2["attribution"]
    assert sum(a.values()) == pytest.approx(sig2["pnl_day"])
    assert a["delta"] == pytest.approx(tot["delta"] * 1.0)


def test_conditional_structure_logs_a_skip():
    ch = _chain()
    cell = {"id": "short_strangle_rich.v1:TEST", "role": "candidate", "symbols": ["TEST"],
            "structure": "short_strangle_rich", "spec": json.loads(json.dumps(specs.OPTION_STRUCTURES["short_strangle_rich"])),
            "spec_sha": "x", "exits": {"dte_exit": 21, "take_profit": 0.5, "stop_loss": 2.0}}
    st, sig, trig, fil, _ = optsim.step_option(cell, None, ch, 100.0, 0.04, {"iv_rv": 1.05}, "2026-11-20", {}, None)
    assert trig[0]["taken_raw"] is False and "1.05" in trig[0]["skip_reason"] and fil == []
    assert st["pos"] is None and sig["ret_raw"] == 0.0


# ---- scoring -------------------------------------------------------------------------------------
def test_bh_qvalues_exact():
    q = score.bh(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}))
    assert q["a"] == pytest.approx(0.04) and q["b"] == pytest.approx(0.16 / 3)
    assert q["c"] == pytest.approx(0.16 / 3) and q["d"] == pytest.approx(0.5)


def test_gate_refuses_nothing_from_nulls_on_noise():
    reg = {"bars": specs.BARS, "cells": [
        {"id": "always_long.v1:A", "role": "null", "kind": "linear", "strategy": "always_long", "params": {},
         "overlays": {}, "symbols": ["A"]},
        {"id": "coin_flip-h5.v1:A", "role": "null", "kind": "linear", "strategy": "coin_flip",
         "params": {"hold": 5, "seed": 1}, "overlays": {}, "symbols": ["A"]}]
        + [{"id": f"c{i}.v1:A", "role": "candidate", "kind": "linear"} for i in range(1000)]}
    frames = {"A": _day(synth_ohlcv(900, seed=5))}
    g = score.gate(reg, frames, days=504)
    assert g["status"] == "OK" and g["pass_full"] == 0 and g["trials"] == 1000


def test_verdict_pending_before_min_days():
    r = pd.Series({"role": "candidate", "n_days": 10, "strategy": "tsmom", "trades": 20, "dsr": 1.0, "q": 0.0,
                   "excess_psr": 1.0})
    assert score.verdict(r, specs.BARS) == "PENDING"
    r["n_days"] = 200
    assert score.verdict(r, specs.BARS) == "PASS"
    r["excess_psr"] = 0.5
    assert score.verdict(r, specs.BARS) == "FAIL"


# ---- ledger ---------------------------------------------------------------------------------------
def test_ledger_snapshots_dedupe_chain_and_detect_corruption(fwd_dir, monkeypatch):
    from mlab import ledger
    base = fwd_dir / "data"
    (base / "journal").mkdir(parents=True)
    (base / "LESSONS.md").write_text("v1")
    (base / "journal" / "trades.jsonl").write_text('{"id":1}\n')
    d1 = ledger.snapshot(base)
    assert set(d1["files"]) == {"LESSONS.md", "journal/trades.jsonl"}
    assert ledger.snapshot(base).get("unchanged")
    (base / "LESSONS.md").write_text("v2")
    ledger.snapshot(base)
    hist = ledger.log("LESSONS.md")
    assert len(hist) == 2 and ledger.show(hist[0]["sha"]) == b"v1"
    assert ledger.verify()["status"] == "ok"
    obj = next((fwd_dir / "ledger" / "objects").rglob("*.gz"))
    obj.write_bytes(gzip.compress(b"tampered"))
    assert ledger.verify()["status"] == "BROKEN"


def test_book_grade_pools_a_matrix_row_across_instruments():
    reg = {"bars": specs.BARS, "cells": [
        {"id": "tsmom.v1:A", "role": "candidate", "kind": "linear", "group": "x", "strategy": "tsmom"},
        {"id": "tsmom.v1:B", "role": "candidate", "kind": "linear", "group": "x", "strategy": "tsmom"},
        {"id": "always_long.v1:A", "role": "null", "kind": "linear", "group": "x", "strategy": "always_long"},
        {"id": "always_long.v1:B", "role": "null", "kind": "linear", "group": "x", "strategy": "always_long"},
        {"id": "xs_momentum-top2.v1:bk", "role": "candidate", "kind": "linear", "group": "basket",
         "strategy": "xs_momentum"}]}
    rows = []
    for i in range(5):
        for cid, r in (("tsmom.v1:A", 0.01), ("tsmom.v1:B", 0.03), ("always_long.v1:A", 0.0),
                       ("always_long.v1:B", 0.0), ("xs_momentum-top2.v1:bk", 0.001 * i)):
            rows.append({"cell": cid, "bar": f"2026-12-0{i + 1}", "ret_raw": r, "run_id": f"r{i}", "first": False,
                         "trades_raw": 1})
    b = score.grade_books(pd.DataFrame(rows), reg)
    assert set(b.index) == {"tsmom.v1:all", "xs_momentum-top2.v1:bk"} and b.attrs["trials"] == 2
    assert b.loc["tsmom.v1:all", "total"] == pytest.approx(1.02 ** 5 - 1)
    assert b.loc["tsmom.v1:all", "cells"] == 2 and b.loc["tsmom.v1:all", "trades"] == 2
