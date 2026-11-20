"""The forward registry: which cells run, frozen before their first signal, and the bars they are graded on.

A cell is one strategy with fixed parameters and overlays on one instrument (single strategies) or one basket
(portfolio strategies), or one rule-based option structure on one underlying. Its spec is hashed (SHA-256 of
the canonical JSON) and stamped with the registration time. The registry lives in the repo
(research/forward/registry.json) so every change is reviewed in a PR; each run also stores a write-once copy
under its hash in the data dir. A spec whose hash changes is refused: register a new version id instead.

Null stubs (always_long and a seeded coin_flip on every instrument and basket) are registered as controls.
They count toward nothing except the gate: the pass bars must refuse them.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

from ..config import ROOT
from .store import canonical, now_iso, sha256_bytes

REGISTRY_PATH = ROOT / "research" / "forward" / "registry.json"
REGISTRY_SCHEMA = "fwd.registry/1"

# ---- instruments -------------------------------------------------------------------------
UNIVERSE: dict[str, list[str]] = {
    "index": ["^GSPC", "^NDX", "^RUT", "^FTSE", "^GDAXI", "^STOXX50E", "^N225"],
    "us_etf": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "XLK", "XLE", "XLF", "SMH"],
    "uk_etf": ["VWRP.L", "ISF.L"],
    "us_share": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM", "XOM", "LLY"],
    "uk_share": ["SHEL.L", "AZN.L", "HSBA.L", "RR.L", "BARC.L", "ULVR.L"],
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "EURGBP=X", "AUDUSD=X"],
    "commodity": ["GC=F", "SI=F", "CL=F", "HG=F", "NG=F"],
    "crypto": ["crypto:BTC", "crypto:ETH", "crypto:SOL", "crypto:XRP"],
}
# Context series fetched every run (regime tags and the desk's index gate, LESSONS L-027).
CONTEXT_SYMBOLS = ["^GSPC", "^NDX", "^SOX", "^VIX", "^VIX3M", "crypto:BTC", "^TNX", "DX-Y.NYB"]

# ---- single-instrument matrix: (strategy, variant tag, params, overlays) --------------------
VT15 = {"vol_target": {"target": 0.15}}
SINGLE_MATRIX: list[tuple[str, str, dict, dict]] = [
    ("sma_cross", "", {}, {}),
    ("ema_cross", "", {}, {}),
    ("donchian", "", {}, {}),
    ("donchian", "vt15", {}, VT15),
    ("supertrend", "", {}, {}),
    ("tsmom", "", {}, {}),
    ("tsmom", "6m", {"lookback": 126, "skip": 0}, {}),
    ("tsmom", "vt15", {}, VT15),
    ("macd_trend", "", {}, {}),
    ("adx_trend", "", {}, {}),
    ("keltner_breakout", "", {}, {}),
    ("squeeze_breakout", "", {}, {}),
    ("nr7_breakout", "", {}, {}),
    ("rsi_reversion", "", {}, {}),
    ("rsi2", "", {}, {}),
    ("bollinger_reversion", "", {}, {}),
    ("zscore_reversion", "", {}, {}),
    ("ibs_reversion", "", {}, {}),
    ("turn_of_month", "", {}, {}),
    ("ema_cross", "rf", {"fast": 10, "slow": 50}, {"regime_filter": {}}),
]

# ---- basket matrix: (strategy, variant, basket name, symbols, params) -----------------------
CROSS_ASSET = ["SPY", "EFA", "EEM", "TLT", "GLD", "IEF"]
PORTFOLIO_MATRIX: list[tuple[str, str, str, list[str], dict]] = [
    ("xs_momentum", "top3", "us_mega", UNIVERSE["us_share"], {"top": 3}),
    ("xs_momentum", "top2", "cross_asset", CROSS_ASSET, {"top": 2}),
    ("xs_momentum", "top2", "sectors", ["XLK", "XLF", "XLE", "SMH", "XLV", "XLI", "XLY", "XLP", "XLU"], {"top": 2}),
    ("xs_momentum", "top1", "crypto", UNIVERSE["crypto"], {"top": 1}),
    ("xs_momentum", "top2-6m", "commodities", UNIVERSE["commodity"], {"top": 2, "lookback": 126, "skip": 0}),
    ("dual_momentum", "", "gem", ["SPY", "EFA", "BIL"], {"safe": "BIL"}),
    ("dual_momentum", "", "gem_bonds", ["SPY", "EFA", "IEF"], {"safe": "IEF"}),
    ("inverse_vol", "", "cross_asset", CROSS_ASSET, {}),
    ("risk_parity", "", "cross_asset", CROSS_ASSET, {}),
    ("pairs", "", "NVDA-AMD", ["NVDA", "AMD"], {}),
    ("pairs", "", "XOM-CVX", ["XOM", "CVX"], {}),
    ("pairs", "", "GLD-SLV", ["GLD", "SLV"], {}),
    ("pairs", "", "KO-PEP", ["KO", "PEP"], {}),
    ("pairs", "", "EWA-EWC", ["EWA", "EWC"], {}),
    ("pairs", "", "SHEL-BP", ["SHEL.L", "BP.L"], {}),
]

# ---- options matrix ------------------------------------------------------------------------
OPTION_UNDERLYINGS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "NVDA", "TSLA", "AMZN", "MSFT"]
# Legs: (type C/P/U, delta target or "atm", qty per contract set). Entry rules and exits live in optsim.
OPTION_STRUCTURES: dict[str, dict] = {
    "covered_call": {"legs": [("U", None, 100), ("C", 0.30, -1)], "credit": True, "dte": 35},
    "cash_secured_put": {"legs": [("P", -0.30, -1)], "credit": True, "dte": 35},
    "bull_put_spread": {"legs": [("P", -0.30, -1), ("P", -0.15, 1)], "credit": True, "dte": 35},
    "bear_call_spread": {"legs": [("C", 0.30, -1), ("C", 0.15, 1)], "credit": True, "dte": 35},
    "iron_condor": {"legs": [("P", -0.20, -1), ("P", -0.10, 1), ("C", 0.20, -1), ("C", 0.10, 1)], "credit": True,
                    "dte": 35},
    "short_strangle": {"legs": [("P", -0.16, -1), ("C", 0.16, -1)], "credit": True, "dte": 35},
    "short_strangle_rich": {"legs": [("P", -0.16, -1), ("C", 0.16, -1)], "credit": True, "dte": 35,
                            "when": {"iv_rv_min": 1.2}},
    "long_straddle": {"legs": [("P", "atm", 1), ("C", "atm", 1)], "credit": False, "dte": 35},
    "long_straddle_cheap": {"legs": [("P", "atm", 1), ("C", "atm", 1)], "credit": False, "dte": 35,
                            "when": {"iv_rv_max": 0.9}},
    "put_spread_hedge": {"legs": [("P", -0.30, 1), ("P", -0.10, -1)], "credit": False, "dte": 60},
    "collar": {"legs": [("U", None, 100), ("P", -0.25, 1), ("C", 0.25, -1)], "credit": False, "dte": 45},
}

# ---- pass bars (pre-registered; changing them is a new registry version) ---------------------
BARS = {
    "min_days": 126,
    "min_trades": 8,
    "min_trades_allocation": 6,
    "dsr_min": 0.95,
    "fdr_q": 0.10,
    "excess_psr_min": 0.90,
    "trials": "number of registered candidate cells (all versions, retired included) at grading time",
    "excess_over": "the instrument's or basket's always_long control, same venue and costs",
    "note": "reads before min_days are provisional; a cell is graded once, at min_days, then monitored",
    "primary_unit": "book: each single-instrument matrix row run equal-weight across all its instruments, each "
                    "option structure equal-weight across its underlyings, each basket strategy as is; trials = "
                    "number of books. Per-instrument cells are graded too, with trials = all candidate cells",
}


def _sha(spec: dict) -> str:
    return sha256_bytes(canonical(spec))


def _cid(strategy: str, variant: str, target: str, version: int = 1) -> str:
    return f"{strategy}{'-' + variant if variant else ''}.v{version}:{target}"


def _seed(s: str) -> int:
    return zlib.crc32(s.encode()) % 100_000


def generate() -> list[dict]:
    """Every cell the matrix defines, as bare specs (no hashes or times yet)."""
    out = []
    for group, symbols in UNIVERSE.items():
        for sym in symbols:
            for strat, var, params, ov in SINGLE_MATRIX:
                out.append({"id": _cid(strat, var, sym), "kind": "linear", "role": "candidate", "strategy": strat,
                            "params": params, "overlays": ov, "symbols": [sym], "group": group})
            out.append({"id": _cid("always_long", "", sym), "kind": "linear", "role": "null",
                        "strategy": "always_long", "params": {}, "overlays": {}, "symbols": [sym], "group": group})
            out.append({"id": _cid("coin_flip", "h5", sym), "kind": "linear", "role": "null", "strategy": "coin_flip",
                        "params": {"hold": 5, "seed": _seed(sym)}, "overlays": {}, "symbols": [sym], "group": group})
    seen_baskets = {}
    for strat, var, basket, syms, params in PORTFOLIO_MATRIX:
        out.append({"id": _cid(strat, var, basket), "kind": "linear", "role": "candidate", "strategy": strat,
                    "params": params, "overlays": {}, "symbols": syms, "group": "basket"})
        seen_baskets.setdefault(basket, syms)
    for basket, syms in seen_baskets.items():
        out.append({"id": _cid("always_long", "", basket), "kind": "linear", "role": "null", "strategy": "always_long",
                    "params": {}, "overlays": {}, "symbols": syms, "group": "basket"})
    for und in OPTION_UNDERLYINGS:
        for name, st in OPTION_STRUCTURES.items():
            out.append({"id": _cid(name, "", und), "kind": "option", "role": "candidate", "structure": name,
                        "spec": json.loads(json.dumps(st)), "symbols": [und], "group": "options",
                        "exits": {"dte_exit": 21, "take_profit": 0.5, "stop_loss": 2.0} if st["credit"]
                        else {"dte_exit": 21, "take_profit": 1.0, "stop_loss": 0.5}})
        out.append({"id": _cid("always_long", "", f"opt-{und}"), "kind": "linear", "role": "null",
                    "strategy": "always_long", "params": {}, "overlays": {}, "symbols": [und], "group": "options"})
    return out


def load(path: Path | None = None) -> dict:
    p = Path(path or REGISTRY_PATH)
    if not p.exists():
        return {"schema": REGISTRY_SCHEMA, "bars": BARS, "bars_sha": _sha(BARS), "cells": []}
    return json.loads(p.read_text())


def register(path: Path | None = None, priors: dict | None = None, now: str | None = None) -> dict:
    """Add new cells from the matrix to the registry; refuse any existing id whose spec changed.
    Returns {"added": [...], "unchanged": n, "refused": [...]} and writes the registry file."""
    p = Path(path or REGISTRY_PATH)
    reg = load(p)
    if reg.get("bars_sha") and reg["bars_sha"] != _sha(BARS):
        raise ValueError("pass bars changed since registration: start a new registry version instead")
    by_id = {c["id"]: c for c in reg["cells"]}
    added, refused, unchanged = [], [], 0
    stamp = now or now_iso()
    for spec in generate():
        sha = _sha(spec)
        old = by_id.get(spec["id"])
        if old is not None:
            if old["spec_sha"] != sha:
                refused.append(spec["id"])
            else:
                unchanged += 1
            continue
        cell = {**spec, "spec_sha": sha, "registered_at": stamp}
        if priors and spec["id"] in priors:
            cell["prior"] = priors[spec["id"]]
        reg["cells"].append(cell)
        added.append(spec["id"])
    if refused:
        raise ValueError(f"{len(refused)} registered specs changed (e.g. {refused[:3]}); bump the version (.v2) "
                         "instead of editing a frozen spec")
    reg["updated_at"] = stamp
    reg["counts"] = count(reg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=1, sort_keys=False) + "\n")
    return {"added": added, "unchanged": unchanged, "refused": refused}


def verify(reg: dict) -> list[str]:
    """Problems with a registry: a cell whose stored hash no longer matches its spec fields."""
    bad = []
    for c in reg["cells"]:
        spec = {k: v for k, v in c.items() if k not in ("spec_sha", "registered_at", "prior", "retired_at")}
        if _sha(spec) != c["spec_sha"]:
            bad.append(c["id"])
    if reg.get("bars_sha") != _sha(reg.get("bars", {})):
        bad.append("<bars>")
    return bad


def count(reg: dict) -> dict:
    cells = reg["cells"]
    return {"cells": len(cells), "candidates": sum(c["role"] == "candidate" for c in cells),
            "nulls": sum(c["role"] == "null" for c in cells),
            "linear": sum(c["kind"] == "linear" for c in cells), "option": sum(c["kind"] == "option" for c in cells)}


def symbols(reg: dict) -> list[str]:
    s = {x for c in reg["cells"] if not c.get("retired_at") for x in c["symbols"]}
    return sorted(s | set(CONTEXT_SYMBOLS))
