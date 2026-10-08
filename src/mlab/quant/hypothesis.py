"""Pre-registered hypotheses: write the claim, the test and the pass bars down BEFORE running it.

Files live in the repo at research/hypotheses/H-<yyyymmdd>-<slug>.md, each on its own
`hypo/<yyyymmdd>-<slug>` branch. The JSON spec block is hashed at registration; `test` refuses to
grade a spec that changed afterwards (a changed test is a new hypothesis). PASS and FAIL files are
both merged to main: the failures are the trial count that keeps the deflated Sharpe honest.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil

# Only used to run git with fixed arguments and a slugified branch name.
import subprocess  # nosec B404
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from . import validation
from .strategies import get

HYPO_DIR = ROOT / "research" / "hypotheses"

DEFAULT_SPEC = {
    "strategy": "tsmom",
    "symbols": ["^GSPC"],
    "params": {},
    "overlays": {},
    "interval": "1d",
    "period": "max",
    "start": None,
    "oos_start": None,
    "costs": {"spread_bps": 2.0, "funding_annual": 0.0},
    "trials": 1,
    "pass": {"oos_sharpe_min": 0.5, "dsr_min": 0.95, "perm_p_max": 0.05, "max_dd_max": 0.30, "min_trades": 30},
}

TEMPLATE = """# {hid}: {claim}

Status: PRE-REGISTERED | Verdict: PENDING
Branch: `{branch}` | Registered: {today} | Spec-SHA256: {sha}

## Claim
{claim}

## Mechanism
Why should this edge exist (behavioural, structural, risk premium), and why has it not been arbitraged away?

## Kill criteria
Any pass bar missed = FAIL. No re-running with new parameters on this file: a new test is a new hypothesis.

## Spec (frozen at registration)
```json
{spec}
```

## Results
_Filled by `bash mlab hypo test {rel}`. Do not edit by hand._
"""

PASS_CHECKS = {
    "oos_sharpe_min": ("out-of-sample Sharpe", lambda m: m["oos_sharpe"], ">="),
    "sharpe_min": ("full-period Sharpe", lambda m: m["sharpe"], ">="),
    "dsr_min": ("deflated Sharpe probability", lambda m: m["dsr"], ">="),
    "psr_min": ("probabilistic Sharpe vs 0", lambda m: m["psr"], ">="),
    "perm_p_max": ("permutation p-value", lambda m: m["perm_p"], "<="),
    "max_dd_max": ("max drawdown (abs)", lambda m: abs(m["max_dd"]), "<="),
    "min_trades": ("trades", lambda m: m["trades"], ">="),
}


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def _spec_text(spec: dict) -> str:
    return json.dumps(spec, indent=2)


def _sha(spec_text: str) -> str:
    return hashlib.sha256(json.dumps(json.loads(spec_text), sort_keys=True).encode()).hexdigest()[:16]


def new(slug: str, claim: str, spec: dict | None = None, branch: bool = False, directory: Path | None = None) -> Path:
    d = directory or HYPO_DIR
    today = date.today()
    hid = f"H-{today:%Y%m%d}-{_slugify(slug)}"
    path = d / f"{hid}.md"
    if path.exists():
        raise FileExistsError(path)
    spec = {**DEFAULT_SPEC, **(spec or {})}
    get(spec["strategy"])  # fail early on a typo
    text = _spec_text(spec)
    br = f"hypo/{today:%Y%m%d}-{_slugify(slug)}"
    if branch:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("git not found; create the branch yourself or drop --branch")
        subprocess.run([git, "-C", str(ROOT), "checkout", "-b", br], check=True)
    d.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    path.write_text(TEMPLATE.format(hid=hid, claim=claim, branch=br, today=today, sha=_sha(text), spec=text, rel=rel))
    return path


def parse(path: Path) -> dict:
    t = Path(path).read_text()
    m = re.search(r"## Spec[^\n]*\n```json\n(.*?)\n```", t, re.S)
    if not m:
        raise ValueError(f"{path}: no ```json spec block")
    sha = re.search(r"Spec-SHA256: (\w+)", t)
    verdict = re.search(r"Verdict: (\w+)", t)
    title = t.splitlines()[0].lstrip("# ").strip()
    return {"text": t, "spec_text": m.group(1), "spec": json.loads(m.group(1)), "sha": sha.group(1) if sha else None,
            "verdict": verdict.group(1) if verdict else "PENDING", "title": title}


def grade(metrics: dict, bars: dict) -> tuple[str, list[dict]]:
    rows, ok = [], True
    for key, bar in bars.items():
        if key not in PASS_CHECKS:
            rows.append({"check": key, "bar": bar, "value": None, "pass": None, "note": "unknown check"})
            continue
        label, fn, op = PASS_CHECKS[key]
        v = fn(metrics)
        good = (v >= bar) if op == ">=" else (v <= bar)
        good = bool(good) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False
        ok &= good
        rows.append({"check": label, "bar": f"{op} {bar}", "value": v, "pass": good})
    return ("PASS" if ok else "FAIL"), rows


def _frames(spec: dict) -> dict[str, pd.DataFrame]:
    from ..data import get_prices
    return {s: get_prices(s, spec.get("interval", "1d"), spec.get("start"), None, spec.get("period", "max"))
            for s in spec["symbols"]}


def test(path: Path, frames: dict[str, pd.DataFrame] | None = None, force: bool = False,
         n_perm: int = 500, n_boot: int = 500) -> dict:
    path = Path(path)
    h = parse(path)
    if h["sha"] and _sha(h["spec_text"]) != h["sha"] and not force:
        raise ValueError("spec changed after registration: register a new hypothesis (or --force to grade anyway, "
                         "which is recorded in the results)")
    if h["verdict"] != "PENDING" and not force:
        raise ValueError(f"already graded {h['verdict']}; a re-test is a new hypothesis")
    spec = h["spec"]
    frames = frames or _frames(spec)
    costs = spec.get("costs", {})
    rep = validation.report(spec["strategy"], frames, spec.get("params"), spec.get("overlays"),
                            trials=int(spec.get("trials", 1)), oos_start=spec.get("oos_start"), n_perm=n_perm,
                            n_boot=n_boot, spread_bps=costs.get("spread_bps", 0.0),
                            funding_annual=costs.get("funding_annual", 0.0))
    metrics = {"sharpe": rep["sharpe"], "oos_sharpe": rep["split"]["out_of_sample"]["sharpe"], "dsr": rep["dsr"],
               "psr": rep["psr_vs_0"], "perm_p": rep["permutation"]["perm_p"], "max_dd": rep["max_dd"],
               "trades": rep["trades"]}
    verdict, rows = grade(metrics, spec.get("pass", {}))
    sources = {s: f"{df.attrs.get('source', '?')} {str(df.index[0])[:10]}..{str(df.index[-1])[:10]}"
               for s, df in frames.items()}
    tbl = pd.DataFrame(rows).to_markdown(index=False)
    b = rep["bootstrap"]
    body = (f"\nGraded {date.today()} · verdict **{verdict}**" + (" · spec changed after registration (forced)"
                                                                   if force else "") + "\n\n"
            f"{tbl}\n\n"
            f"- Sharpe {rep['sharpe']:.2f} (90% bootstrap CI {b['ci_low']:.2f} to {b['ci_high']:.2f}), "
            f"PSR vs 0 {rep['psr_vs_0']:.3f}, DSR {rep['dsr']:.3f} over {rep['trials']} trial(s)\n"
            f"- In-sample Sharpe {rep['split']['in_sample']['sharpe']:.2f}, out-of-sample from "
            f"{rep['split']['oos_start']}: {rep['split']['out_of_sample']['sharpe']:.2f}\n"
            f"- CAGR {rep['cagr']:.2%}, max drawdown {rep['max_dd']:.2%}, trades {rep['trades']}, "
            f"exposure {rep['exposure']:.0%}, cost drag {rep['cost_drag']:.2%}\n"
            f"- Benchmark (equal-weight buy and hold) Sharpe {rep['benchmark'].get('sharpe', float('nan')):.2f}, "
            f"max drawdown {rep['benchmark'].get('max_drawdown', float('nan')):.2%}\n"
            f"- Data: {json.dumps(sources)}\n")
    t = h["text"]
    t = re.sub(r"Status: \w[\w-]*", "Status: GRADED", t, count=1)
    t = re.sub(r"Verdict: \w+", f"Verdict: {verdict}", t, count=1)
    t = t.rstrip() + "\n" + body
    path.write_text(t)
    return {"verdict": verdict, "checks": pd.DataFrame(rows), "report": rep, "path": str(path)}


def listing(directory: Path | None = None) -> pd.DataFrame:
    d = directory or HYPO_DIR
    rows = []
    for p in sorted(d.glob("H-*.md")) if d.exists() else []:
        try:
            h = parse(p)
            rows.append({"id": p.stem, "verdict": h["verdict"], "strategy": h["spec"].get("strategy"),
                         "symbols": ",".join(h["spec"].get("symbols", [])), "claim": h["title"].split(": ", 1)[-1][:60]})
        except Exception as e:
            rows.append({"id": p.stem, "verdict": "UNREADABLE", "claim": str(e)[:60]})
    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


def trial_count(directory: Path | None = None) -> int:
    """All graded hypotheses so far: feed this into `trials` so the DSR bar rises as you search more."""
    df = listing(directory)
    return int((df["verdict"].isin(["PASS", "FAIL"])).sum()) if len(df) else 0
