# Market Lab: agent guide

Market Lab is a market research toolkit (the `mlab` CLI, `src/mlab/`) driven by Claude Code agents. This file is what
an agent reads on opening the repo: how the system is laid out, how to run it, and the rules every change and every
analysis follows. Humans start with [`README.md`](README.md) and the [wiki](docs/wiki/Home.md).

## Architecture in one paragraph

A **PM session** (the main Claude Code session) takes a question, fans it out in parallel to six **desk agents**
(`.claude/agents/`), each of which runs `mlab` commands and returns numbers with their source and timestamp. The PM
weighs the desks through three investment **lenses** (macro/reflexivity, quality/value, contrarian/forensic), writes
the conclusion and, when there is a trade, a structured **trade card** that is logged to the journal. Systematic ideas
go through a separate **quant research loop** with pre-registered hypotheses and fixed pass bars. **Skills**
(`.claude/skills/`) are the playbooks for each workflow. Broker access is read-only.

```
question -> PM session -> desk agents (parallel) -> mlab CLI -> data providers (cached, source-tagged)
                 |              numbers + sources
                 v
           lenses -> conclusion -> trade card -> journal
           quant loop: hypo new -> gate -> test (PASS/FAIL, once) -> paper
```

## Setup

```bash
bash bootstrap.sh          # venv at ~/.venvs/market-lab, installs mlab, runs `mlab doctor`
bash mlab doctor           # data hosts reachable, credentials present, where code and data live
~/.venvs/market-lab/bin/pytest -q    # offline test suite
```

`mlab` self-bootstraps on first use, so `bash mlab <cmd>` always works (`./mlab` too, except on mounts without exec
rights). A SessionStart hook in `.claude/settings.json` runs the bootstrap quietly.

**Code vs state.** Code lives in git. State (journal, reports, price cache, IV history, transcripts, paper books,
datasets) lives in `MLAB_DATA_DIR`: a shared project folder when one is mounted, else the checkout. Per-item overrides:
`MLAB_JOURNAL_DIR`, `MLAB_REPORTS_DIR`, `MLAB_CACHE_DIR`, `MLAB_IV_DIR`, `MLAB_TRANSCRIPTS_DIR`. State is gitignored
and is never committed.

## Desk agents (`.claude/agents/`)

Fan out to these in parallel for anything non-trivial. Each one crunches; none makes the call.

| Agent | Crunches | Main commands |
|---|---|---|
| `technicals-desk` | 60+ indicators, candlesticks, chart patterns, levels, regime, multi-timeframe, plus the backtested hit rate of every signal on that instrument | `tafull`, `firing`, `signals`, `patterns`, `ta`, `scan`, `backtest` |
| `options-desk` | IV term structure, skew, expected move, max pain, GEX and gamma flip, unusual activity, IV vs RV, structure payoffs | `options`, `payoff` |
| `earnings-desk` | Earnings history vs price reactions (gap, day 1, drift, abnormal return), call and release tone, guidance | `earnings`, `earnings-text`, `earnings-align` |
| `news-sentiment-desk` | News (Google News, Yahoo, GDELT, SEC 8-K), social (StockTwits, Reddit, HN), sentiment index, attention spikes, lead/lag vs price | `news`, `sentiment`, `social` |
| `macro-desk` | Rates, curve, inflation, liquidity (Fed BS − TGA − RRP), credit, USD, cross-asset trend board, CFTC COT | `macro --liquidity`, `lens soros`, `cot`, `cot-extremes`, `rs`, `corr` |
| `fundamentals-desk` | 10 years of SEC financials, filings, insiders, quality and deep-value checklists, owner-earnings DCF | `dd`, `lens buffett`, `lens burry` |

## Skills (`.claude/skills/`)

| Skill | Workflow |
|---|---|
| `trade-idea` | The hub: pin the instrument, fan out the desks, frame with a lens, write the trade card |
| `morning-brief` | Pre-market brief: moves, regime, catalysts, open positions and journal ideas re-checked |
| `lens-soros`, `lens-buffett`, `lens-burry` | Thesis framing on top of desk numbers |
| `market-data` | Any data pull: symbols, caching, source tags, fallbacks |
| `ig-account` | Read-only broker account views |
| `trade-journal` | Log, update and review cards; score which lens and conviction level pays |
| `quant-research` | The pre-registered hypothesis loop |

## Rules for analysis

- **Crunch first.** Run the desks, then conclude. Technicals are for timing; the thesis comes from the other desks.
- **Every number carries its source and timestamp** (the CLI prints them). If data is stale or a source failed, say
  which and carry on with what is available.
- **Never invent prices, levels, figures or news.** Keep fact, inference and opinion visibly separate, and quantify
  (probabilities, ranges, expected value). "No trade" is a valid conclusion.
- **Every trade idea ends with the trade card**, built and journaled with `mlab card ... --journal`:
  `Bias, venue and instrument | Entry zone | Stop / invalidation | Targets | Reward:risk | Risk per trade as % of
  account | Horizon | Conviction 1-5 | Catalysts and dates | What kills it` (one line, the strongest counter-case).
- **Venue-agnostic.** Pick the instrument and venue that fit the trade; the IG integration is one option, not the default.

## Rules for research (the gates)

- Research runs on frozen datasets (`mlab datasets`): versioned, SHA-256 checksummed snapshots with fixed splits
  (train to 2014, validation 2015-19, test 2020+). The test split needs `--allow-test` and every look is logged.
- A systematic claim is pre-registered before any result is seen: `mlab hypo new` writes the claim, data, costs,
  pass bars and kill criteria to `research/hypotheses/`, and the spec is frozen by its SHA-256 hash.
- Pass bars must refuse null strategies first (`mlab hypo gate`: buy-and-hold and seeded coin flips) before grading.
- `mlab hypo test` grades once, PASS or FAIL. Never edit bars, period, costs or universe afterwards; a new test is a new
  file. FAILs merge to main too: they are the trial count behind every deflated Sharpe.
- Validation means out-of-sample, walk-forward, probabilistic and deflated Sharpe, block bootstrap and permutation
  tests. Decisions at the close of bar t earn bar t+1; tests enforce no look-ahead by truncation.
- Modes are backtest, signal and paper. There is no live execution mode.

## Broker access (read-only)

- `src/mlab/providers/ig.py` refuses every request except GETs and session login/switch/logout. No order or deal
  endpoints exist; do not add any.
- Credentials come only from environment variables or proxy-injected network secrets (see `docs/NETWORK.md` and
  `.env.example`). Never print, echo, log, write or commit them, and never ask for them in chat.
- IG historical prices are allowance-limited (10k points/week): go through `mlab price ig:...` (cache-first, fetches
  only the missing tail). Prefer Yahoo/Stooq for long history.

## Data sources and fallbacks

Free and keyless by default: Yahoo, Stooq, FRED, SEC EDGAR, Binance/Kraken/CoinGecko, ECB FX, GDELT, Google News RSS,
StockTwits, Reddit, CFTC. Optional keys: `FRED_API_KEY`, `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY`,
`SEC_USER_AGENT`. When `mlab doctor` shows a host blocked: use a connected data connector, then WebSearch/WebFetch
(cite URLs and times), then run the same command on a machine with open network. See `docs/NETWORK.md` and
`docs/SKILLS_MAP.md`.

## CLI cheat sheet (`bash mlab --help` for all)

```bash
./mlab price AAPL -p 2y                 # OHLCV (Yahoo -> Stooq fallback), cached
./mlab price crypto:BTC -i 4h -p 3mo    # Binance -> Kraken
./mlab ta NVDA --chart                  # snapshot, S/R, pivots, fibs, signals, HTML chart in reports/
./mlab tafull NVDA ; ./mlab firing NVDA ; ./mlab signals NVDA -p 10y
./mlab scan -u indices -u fx --sort 3m  # cross-asset scan
./mlab options NVDA ; ./mlab earnings NVDA ; ./mlab news NVDA ; ./mlab sentiment NVDA
./mlab macro --liquidity ; ./mlab lens soros ; ./mlab cot gold
./mlab dd AAPL ; ./mlab lens buffett AAPL ; ./mlab lens burry XYZ
./mlab size --equity 20000 --risk 1 --entry 7410 --stop 7330
./mlab card "Example index" --bias long --entry 7400 7420 --stop 7330 --targets 7600 7750 --journal
./mlab journal list ; ./mlab journal review
./mlab algo list ; ./mlab algo validate tsmom ^GSPC --grid '{"lookback":[126,252]}'
./mlab hypo new <slug> --claim "..." ; ./mlab hypo test research/hypotheses/H-....md ; ./mlab hypo list
./mlab datasets catalog ; ./mlab datasets build all
./mlab algo backtest tsmom UK100 US500 --dataset duka-1d --split train
```

Python API: `from mlab.data import get_prices`, `from mlab import ta, ta_catalog, signal_lab, stats, lenses, options,
earnings, news, cot, risk, journal`.

## Working on the code

- Work is tracked as GitHub issues under the Roadmap epic (#16) (`docs/wiki/Tracking.md`). Find or open the issue
  first; the PR body starts with `Closes #N` or `Part of #N`. Follow-ups become issues, not TODO comments.
- Every change keeps `pytest -q` green offline; every strategy gets a no-look-ahead test. Standards in
  `docs/wiki/Contributing.md`.
- The repo is public: no credentials, account numbers, journal entries, reports, positions or personal details in
  code, docs, issues or commit messages.

## Files

- Repo: `src/mlab/` code · `tests/` pytest · `.claude/` agents, skills, settings · `research/hypotheses/`
  pre-registrations and verdicts · `docs/` network, skills map, wiki source
- `$MLAB_DATA_DIR`: `data/cache/` parquet price cache (LRU-pruned at `MLAB_CACHE_MAX_MB`, default 750) ·
  `data/transcripts/` · `data/iv_history/` · `reports/` · `journal/` · `paper/`
