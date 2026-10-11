"""Pre-registered event studies. Each function reads its frozen spec from the hypothesis file and runs it; the
spec's "study" key picks the function (absent = the first study, news vs no-news extremes).

    python -m mlab.textlab.studies research/hypotheses/H-20261010-news-vs-nonews-extremes.md [--test]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import datasets
from . import events as ev
from . import features as ft


def read_spec(path: Path) -> dict:
    txt = Path(path).read_text()
    spec_txt = re.search(r"## Spec \(frozen at registration\)\n```json\n(.*?)\n```", txt, re.S).group(1)
    sha = re.search(r"Spec-SHA256: ([0-9a-f]{64})", txt).group(1)
    if hashlib.sha256(spec_txt.encode()).hexdigest() != sha:
        raise ValueError(f"{path}: spec was edited after registration (hash mismatch)")
    return json.loads(spec_txt)


def news_flags(symbols: list[str], index_for: dict[str, pd.DatetimeIndex], earnings_ref: str,
               log=print) -> dict[str, set]:
    """8-K / 8-K/A filing dates plus earnings dates, mapped to trading days t and t+1."""
    from ..providers import sec
    earn = datasets.load(earnings_ref, symbols, allow_test=True, _record=False)
    out = {}
    for s in symbols:
        dates = []
        try:
            dates += list(sec.filing_history(s)["filingDate"])
        except Exception as e:  # report and fall back to earnings dates only
            log(f"  {s}: EDGAR failed ({type(e).__name__}: {str(e)[:80]}); earnings dates only")
        if s in earn:
            dates += list(earn[s].index.tz_convert("America/New_York").tz_localize(None).normalize())
        out[s] = ev.trading_day_flags(pd.Series(dates), index_for[s]) if dates else set()
    return out


def _split(e: pd.DataFrame, lo, hi) -> pd.DataFrame:
    m = pd.Series(True, index=e.index)
    if lo not in (None, "start"):
        m &= e["date"] >= pd.Timestamp(lo, tz="UTC")
    if hi not in (None, "end"):
        m &= e["date"] <= pd.Timestamp(hi, tz="UTC")
    return e[m]


def run_news_extremes(path: Path, open_test: bool = False, log=print) -> dict:
    spec = read_spec(path)
    syms = spec["symbols"]
    frames = datasets.load(spec["dataset"], syms + [spec["market"]], None if open_test else "train+validation",
                           allow_test=open_test, note=f"{Path(path).name} test split")
    mkt = frames.pop(spec["market"])
    idx = {s: ev.abnormal(frames[s], mkt).index for s in syms if s in frames}
    flags = news_flags(list(idx), idx, spec["earnings"], log=log)
    e = ev.events({s: frames[s] for s in idx}, mkt, flags, k=spec["k_sigma"], horizons=tuple(spec["horizons"]),
                  cost_bp=spec["costs"]["round_trip_bp"])
    sp, col = spec["splits"], spec["primary"]
    res = {"train_plus_validation": ev.compare(_split(e, sp["train"][0], sp["validation"][1]), col,
                                                spec["permutation"]["n"], spec["permutation"]["seed"]),
           "train": ev.compare(_split(e, *sp["train"]), col, 500),
           "validation": ev.compare(_split(e, *sp["validation"]), col, 500),
           "by_horizon_tv": {f"car{h}": ev.compare(_split(e, sp["train"][0], sp["validation"][1]), f"car{h}", 200)
                             for h in spec["horizons"]}}
    if open_test:
        res["test"] = ev.compare(_split(e, *sp["test"]), col, spec["permutation"]["n"], spec["permutation"]["seed"])
    res["events"] = len(e)
    return res


def firm_text_daily(symbols: list[str], index_for: dict[str, pd.DatetimeIndex], since, until, window: str,
                    fresh: float, min_words: int, log=print) -> dict[str, pd.DataFrame]:
    """Daily GDELT features per symbol from company-name matches only: URL-slug headlines, deduplicated by URL,
    scored for novelty against the firm's prior `window`, bucketed to trading days at the 20:00 UTC cutoff."""
    from concurrent.futures import ThreadPoolExecutor
    from . import gdelt
    files = sorted((gdelt.STORE / "firm").glob("????-??-??.parquet"))
    files = [f for f in files if str(pd.Timestamp(since).date()) <= f.stem <= str(pd.Timestamp(until).date())]

    def read(f):
        d = pd.read_parquet(f, columns=["avail", "symbol", "via", "url", "tone"])
        return d[d["via"].str.startswith("org:") & d["symbol"].isin(symbols)].drop(columns="via")
    with ThreadPoolExecutor(4) as pool:
        art = pd.concat(list(pool.map(read, files)), ignore_index=True)
    log(f"  {len(art)} company-name articles from {len(files)} GDELT days ({gdelt.CITATION})")
    art = art.sort_values("avail", kind="stable").drop_duplicates(["symbol", "url"])
    art["text"] = art["url"].map(ft.slug)
    art = art[art["text"].str.split().str.len() >= min_words]
    out = {}
    for s, a in art.groupby("symbol"):
        if s not in index_for:
            continue
        a = a.copy()
        a["novelty"] = ft.novelty(a, window, text="text")
        out[s] = ft.firm_daily(a, index_for[s], fresh=fresh)
        log(f"  {s}: {len(a)} headlines, {a['novelty'].ge(fresh).mean():.0%} fresh")
    return out


def attention_events(ab: pd.DataFrame, d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Attention spikes for one symbol, each signed by its tone surprise and flagged fresh- or stale-led, with
    forward CARs from the next open. Every baseline (count, tone, fresh share) uses prior days only."""
    e = spec["event"]
    d = d.reindex(ab.index)
    n = d["n"].fillna(0)
    base = n.rolling(e["baseline_days"], min_periods=e["baseline_days"] // 2)
    spike = (n > base.mean().shift(1) + e["spike_sd"] * base.std().shift(1)) & (n >= e["min_articles"])
    tone_base = d["tone"].rolling(e["baseline_days"], min_periods=e["baseline_days"] // 3).mean().shift(1)
    direction = np.sign(d["tone"] - tone_base)
    share = d["n_fresh"] / n.where(n > 0)
    share_base = share.rolling(e["fresh_share_days"], min_periods=e["fresh_share_days"] // 4).median().shift(1)
    cars = {h: ev.forward_car(ab, h) for h in spec["horizons"]}
    hit = ab.index[spike & (direction != 0) & share_base.notna()]
    rows = []
    for t in hit:
        row = {"date": t, "n": int(n[t]), "dir": float(direction[t]), "fresh_share": float(share[t]),
               "flag": bool(share[t] > share_base[t])}
        for h in spec["horizons"]:
            row[f"car{h}"] = row["dir"] * float(cars[h].get(t, np.nan)) - spec["costs"]["round_trip_bp"] / 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def run_novelty_drift(path: Path, open_test: bool = False, log=print) -> dict:
    spec = read_spec(path)
    syms, sp = spec["symbols"], spec["splits"]
    last = pd.Timestamp("2100-01-01") if open_test else pd.Timestamp(sp["validation"][1])
    frames = datasets.load(spec["dataset"], syms + [spec["market"]], allow_test=True,
                           note=f"{Path(path).name}: own splits; events after {sp['validation'][1]} dropped"
                                + (" (TEST OPENED)" if open_test else ""))
    mkt = frames.pop(spec["market"])
    abn = {s: ev.abnormal(frames[s], mkt, spec["beta_window"], spec["sigma_window"]) for s in syms if s in frames}
    text_start = pd.Timestamp(sp["train"][0])
    feats = firm_text_daily(syms, {s: a.index for s, a in abn.items()}, text_start - pd.Timedelta(spec["novelty"]["window"]),
                            min(last, pd.Timestamp.now()), spec["novelty"]["window"], spec["novelty"]["fresh"],
                            spec["novelty"]["min_words"], log=log)
    rows = []
    for s, d in feats.items():
        e = attention_events(abn[s], d, spec)
        if len(e):
            rows.append(e.assign(symbol=s))
    e = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    e = e[(e["date"] >= pd.Timestamp(sp["train"][0], tz="UTC")) & (e["date"] <= pd.Timestamp(last, tz="UTC"))]
    col, perm = spec["primary"], spec["permutation"]
    res = {"train": ev.compare(_split(e, *sp["train"]), col, perm["n"], perm["seed"]),
           "validation": ev.compare(_split(e, *sp["validation"]), col, perm["n"], perm["seed"]),
           "by_horizon_train": {f"car{h}": ev.compare(_split(e, *sp["train"]), f"car{h}", 200)
                                for h in spec["horizons"]},
           "by_horizon_validation": {f"car{h}": ev.compare(_split(e, *sp["validation"]), f"car{h}", 200)
                                     for h in spec["horizons"]}}
    if open_test:
        res["test"] = ev.compare(_split(e, *sp["test"]), col, perm["n"], perm["seed"])
    res["events"] = len(e)
    return res


STUDIES = {"news_extremes": run_news_extremes, "novelty_drift": run_novelty_drift}

if __name__ == "__main__":
    study = STUDIES[read_spec(Path(sys.argv[1])).get("study", "news_extremes")]
    print(json.dumps(study(Path(sys.argv[1]), "--test" in sys.argv), indent=2, default=str))
