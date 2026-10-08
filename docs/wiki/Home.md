# Market Lab

Market Lab is a trading research desk: a Python toolkit (CLI `mlab`) plus Claude Code desk agents and
playbooks. It pulls market data with caching, runs the full TA catalogue with signal hit rates, options,
earnings, sentiment, macro and fundamentals desks, writes trade cards with IG stake sizing, keeps a trade
journal, and runs a library of algorithmic strategies through backtest, validation and paper trading.

It is built around IG (spread bets and CFDs, GBP); the trader executes every trade by hand. The IG client is read-only.
Strategies run in backtest, signal and paper modes only. Live execution is off by design.

## Quickstart

```bash
bash bootstrap.sh        # venv at ~/.venvs/market-lab, installs mlab, runs mlab doctor
bash mlab doctor         # which data hosts are reachable, which credentials are present, where state lives
bash mlab --help         # every command
```

Then for example:

```bash
bash mlab ta NVDA --chart
bash mlab algo list
bash mlab algo validate tsmom ^GSPC
bash mlab hypo new ftse-rsi2 --claim "RSI(2) < 10 above the 200 SMA on the FTSE beats cash after IG costs"
```

## Code in the repo, state in the shared folder

| Lives in git (this repo) | Lives in `MLAB_DATA_DIR` (never in git) |
|---|---|
| `src/mlab/` code, `tests/` | `journal/` trade cards and `JOURNAL.md` |
| `.claude/agents/`, `.claude/skills/` | `reports/` charts and notes |
| `docs/`, `docs/wiki/` (this wiki) | `data/cache/` parquet price cache, `data/iv_history/`, `data/transcripts/` |
| `research/hypotheses/` pre-registrations and verdicts | `paper/<book>.json` paper books |

`MLAB_DATA_DIR` defaults to `/mnt/project-files/market-lab` (the project shared folder) when it exists, and to
the checkout otherwise. Every thread writes to the same journal, so logging a card never needs a commit.
Per-item overrides: `MLAB_JOURNAL_DIR`, `MLAB_REPORTS_DIR`, `MLAB_CACHE_DIR`, `MLAB_IV_DIR`, `MLAB_TRANSCRIPTS_DIR`.

## Pages

- [[Getting Started]]: install, configuration, credentials, first commands
- [[CLI Reference]]: every `mlab` command with an example
- [[Architecture]]: modules, the quant package, agents, skills, data flow
- [[Desk Workflow]]: PM, six desks, lenses, trade cards, journal
- [[Strategy Library]]: every registered strategy and overlay, and how to add one
- [[Hypothesis Testing]]: pre-registration, branches, verdicts, merge-or-kill
- [[Validation and Overfitting]]: OOS, walk-forward, PSR, DSR, bootstrap, permutation, costs
- [[Risk and Sizing]]: stake per point, ATR stops, vol targeting, risk parity, Kelly, heat
- [[Paper Trading]]: signal and paper books, why live is off
- [[Data Sources]]: providers, gaps, fallbacks
- [[Contributing]]: issues, labels, branches, PRs, CI
- [[Glossary]]

This wiki is generated from `docs/wiki/` in the repo. Edit it there; the wiki-sync workflow mirrors it on push to main.
