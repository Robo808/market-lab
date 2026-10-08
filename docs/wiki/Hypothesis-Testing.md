# Hypothesis Testing

The research loop: write the claim and the pass bars down, commit them, then test. Every test counts, pass or fail.

```
issue [H] -> branch hypo/<date>-<slug> -> mlab hypo new (commit) -> mlab hypo test -> verdict
   PASS -> PR to main -> paper book (3 months / 20 trades) -> can inform trade cards
   FAIL -> PR the FAIL file to main -> delete branch -> close issue with verdict:fail
```

## 1. Open an issue

Use the **Hypothesis** template (`[H] ` title prefix, labels `hypothesis`, `status:pre-registered`). It asks for the
claim, mechanism, universe, test, pass bars, costs, kill criteria and the number of variants you plan to try.

## 2. Branch

| Prefix | For |
|---|---|
| `hypo/<yyyymmdd>-<slug>` | a hypothesis, e.g. `hypo/20261007-ftse-rsi2` |
| `strat/<name>` | a new strategy in the library |
| `data/<source>` | a new or fixed data source |
| `feat/<x>` | other features |
| `fix/<x>` | bug fixes |

## 3. Pre-register

```bash
bash mlab hypo new ftse-rsi2 --claim "Long FTSE 100 when RSI(2) < 10 above the 200 SMA beats cash after IG costs"
```

This creates `research/hypotheses/H-<yyyymmdd>-<slug>.md` from the template, and optionally the `hypo/` branch.
Fill in the mechanism and the spec block, then **commit it before running anything**. The commit timestamp is the
proof that the bars came first.

The file holds:

- **Claim**: one falsifiable sentence.
- **Mechanism**: why the edge should exist (behavioural, structural, risk premium).
- **Spec**: a fenced `json` block that `hypo test` executes:

```json
{
  "strategy": "rsi2",
  "symbols": ["^FTSE"],
  "params": {"entry": 10, "trend": 200, "exit_sma": 5},
  "overlays": {},
  "interval": "1d",
  "period": "max",
  "start": "2000-01-01",
  "oos_start": "2016-01-01",
  "costs": {"spread_bps": 2.0, "funding_annual": 0.06},
  "trials": 12,
  "pass": {
    "oos_sharpe_min": 0.5,
    "dsr_min": 0.95,
    "perm_p_max": 0.05,
    "max_dd_max": 0.25,
    "min_trades": 30
  }
}
```

Available pass checks: `oos_sharpe_min`, `sharpe_min`, `dsr_min`, `psr_min`, `perm_p_max`, `max_dd_max`, `min_trades`.
`oos_start` defaults to the last 30% of the history. `hypo new` sets `trials` to the number of hypotheses already
graded plus one, so the deflated-Sharpe bar rises as the search widens. The spec is hashed at registration
(`Spec-SHA256` in the header) and `hypo test` refuses a spec that changed afterwards.

Field names follow the template that `hypo new` writes; use that file as the reference.

## 4. Test

```bash
bash mlab hypo test research/hypotheses/H-20261007-ftse-rsi2.md
```

Runs the spec: backtest net of costs, in-sample/out-of-sample split, PSR and DSR (using the trial
count), bootstrap CI, permutation test. It appends a **Results** section with every metric next to its bar and a
**PASS** or **FAIL** verdict. PASS needs every bar. See [[Validation and Overfitting]].

```bash
bash mlab hypo list      # every hypothesis with its date, strategy and verdict
```

## 5. Merge or kill

**PASS**

- PR to main with the hypothesis file and its results. Label the issue `verdict:pass`, then `status:paper`.
- Start a paper book: `bash mlab algo paper <strategy> <symbols...> --book <name>` ([[Paper Trading]]).
- At least **3 months and 20 trades** in paper, inside the kill criteria, before it can inform a real trade card.

**FAIL**

- Still PR the hypothesis file with its FAIL verdict to main. The graveyard is data: every failed variant is a trial
  in the deflated Sharpe of everything that comes after it. Hiding failures is p-hacking.
- Delete the branch. Close the issue with `verdict:fail`.

## Rules

- Never edit the pass bars, the period, the costs or the universe after seeing results. A changed test is a new
  hypothesis with a new file, and the old one keeps its verdict.
- Count every variant. If you planned 12 and ran 30, the trial count is 30.
- Costs are IG costs: the spread in points for the exact instrument (DFB or CFD) plus overnight funding for anything held past the close.
- Report the out-of-sample number, not the best in-sample number.
- "No edge" is a valid result and goes in the graveyard like any other.
