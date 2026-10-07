---
name: quant-research
description: Test a systematic trading idea properly - pick or add a strategy from the 21-pattern library, pre-register the hypothesis on a hypo/ branch, run backtest and validation (OOS, PSR/DSR, bootstrap, permutation), grade PASS/FAIL against bars set in advance, then paper trade the winners. Use for "does X work", "backtest X", "is this edge real", "build me a strategy".
---

# Quant research loop

1. **Library first.** `bash mlab algo list` (21 strategies + vol_target / regime_filter overlays). If the idea is new,
   add a function to `src/mlab/quant/strategies.py`, register it in `REGISTRY`, and add it to the no-look-ahead
   test (`tests/test_quant.py` picks up every single-instrument strategy automatically).
2. **Explore cheaply, but count it.** `bash mlab algo backtest <strategy> <symbols...> --spread-bps <cost>` and
   `bash mlab algo validate ... --grid '{...}'`. Every variant looked at is a trial.
3. **Pre-register before the real test.** Branch `hypo/<yyyymmdd>-<slug>`, then
   `bash mlab hypo new <slug> --claim "..." --strategy X --symbols A B --spec '{"pass": {...}, "costs": {...}, "trials": N}'`.
   Fill in Mechanism, commit the file, push. Open a `[H]` issue from the template.
4. **Grade once.** `bash mlab hypo test research/hypotheses/H-....md`. Never edit bars, period, costs or universe after
   that; a new test is a new file. The spec hash enforces it.
5. **Merge or kill.** PASS and FAIL files both merge to main (failures feed the deflated Sharpe trial count).
   PASS goes to paper: `bash mlab algo paper <strategy> <symbols...> --book <name>` once per bar; check with
   `bash mlab algo paper-status`. 3 months / 20 trades in paper before it backs a trade card.
6. **Report like the desk.** Lead with the verdict and the out-of-sample number, then PSR/DSR, permutation p,
   max drawdown, cost drag, data source and dates. A strategy that loses to buy-and-hold on a risk-adjusted basis
   is "no edge", say so.

Live execution is off: strategies run in backtest, signal and paper modes, and IG stays read-only.
Wiki: `docs/wiki/` (Strategy-Library, Hypothesis-Testing, Validation-and-Overfitting, Paper-Trading).
