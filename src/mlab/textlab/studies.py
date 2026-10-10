"""Pre-registered event studies. Each function reads its frozen spec from the hypothesis file and runs it.

    python -m mlab.textlab.studies research/hypotheses/H-20261010-news-vs-nonews-extremes.md [--test]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from .. import datasets
from . import events as ev


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


if __name__ == "__main__":
    print(json.dumps(run_news_extremes(Path(sys.argv[1]), "--test" in sys.argv), indent=2, default=str))
