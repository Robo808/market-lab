# Market Lab

[![CI](https://github.com/Robo808/market-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Robo808/market-lab/actions/workflows/ci.yml)
[![Security](https://github.com/Robo808/market-lab/actions/workflows/security.yml/badge.svg)](https://github.com/Robo808/market-lab/actions/workflows/security.yml)
[![CodeQL](https://github.com/Robo808/market-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/Robo808/market-lab/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**A market research desk in a box.** Market Lab is a Python toolkit (the `mlab` CLI) plus a set of
[Claude Code](https://claude.com/claude-code) desk agents and playbooks. It pulls market data from free sources with
caching and fallbacks, crunches it across technicals, options, earnings, news and sentiment, macro and fundamentals,
turns the result into a sized trade card, journals every call, and runs systematic strategies through backtest,
statistical validation and paper trading.

It is a research and decision-support tool. Broker integrations are **read-only by construction**: nothing in this
repo can place, amend or close an order.

**Documentation:** the [wiki](https://github.com/Robo808/market-lab/wiki) (source in [`docs/wiki/`](docs/wiki/Home.md)).

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [The `mlab` CLI](#the-mlab-cli)
- [Desk agents and playbooks](#desk-agents-and-playbooks)
- [Quant research layer](#quant-research-layer)
- [Project layout](#project-layout)
- [Development](#development)
- [Security](#security)

## What it does

| Area | Highlights |
|---|---|
| **Data** | Prices for stocks, ETFs, indices, FX, futures and crypto (Yahoo, Stooq, Binance, Kraken, CoinGecko), macro (FRED, ECB), filings (SEC EDGAR), CFTC positioning, news and social feeds, IG instrument prices. Parquet cache that fetches only the missing tail. Every number is tagged with its source and timestamp. |
| **Technicals** | A catalogue of 60+ indicators, candlestick and chart patterns, support/resistance, pivots, Fibonacci, regime, plus a backtested hit rate for every signal on the instrument you are looking at (Benjamini-Hochberg adjusted). |
| **Options** | IV term structure, skew, expected move, max pain, put/call ratios, dealer gamma (GEX) and gamma flip, unusual activity, implied vs realised vol, payoff and greeks for multi-leg structures. |
| **Earnings** | Past reports lined up against the price reaction (gap, day 1, drift, abnormal return), transcript and press-release tone and guidance, and the alignment between them. |
| **News and sentiment** | Google News, Yahoo, GDELT tone, SEC 8-Ks, StockTwits, Reddit, Hacker News; a daily sentiment index, attention spikes and lead/lag against price. |
| **Macro** | Rates, curve, inflation, liquidity (Fed balance sheet minus TGA minus RRP), credit, USD, a cross-asset trend board, COT positioning extremes. |
| **Fundamentals** | Ten years of SEC financials, filings, insiders, valuation, quality and deep-value checklists, an owner-earnings DCF. |
| **Trade cards and journal** | A standard card (bias, entry, stop, targets, reward:risk, stake per point, horizon, conviction, catalysts, counter-case) and a journal with review stats. |
| **Quant** | 21 systematic strategies and two overlays in backtest, signal and paper modes; out-of-sample tests, probabilistic and deflated Sharpe, bootstrap and permutation tests; pre-registered hypotheses graded PASS or FAIL. |

## Quick start

Requires Python 3.11+ and git. Works on Linux, macOS and in cloud containers.

```bash
git clone https://github.com/Robo808/market-lab.git
cd market-lab
bash bootstrap.sh        # venv at ~/.venvs/market-lab, installs mlab, runs `mlab doctor`
```

`bootstrap.sh` uses `uv` when present and falls back to `python -m venv`. After that, `./mlab <command>` (or
`bash mlab <command>`) runs the CLI; the wrapper bootstraps itself on first use.

```bash
./mlab doctor                       # which data hosts are reachable, which keys are set, where state lives
./mlab price AAPL -p 2y             # OHLCV with source and timestamp
./mlab ta NVDA --chart              # technical snapshot + interactive HTML chart in reports/
./mlab firing ^GSPC                 # signals firing now and how they performed historically on this instrument
./mlab options NVDA                 # options desk summary
./mlab macro --liquidity            # macro dashboard with the liquidity line
./mlab algo backtest tsmom ^GSPC -p 20y
```

No API keys are needed for the core features: the default providers are free and keyless.

## Configuration

All configuration is through environment variables. Copy [`.env.example`](.env.example) into your shell profile or
your environment's secret store. `.env` is gitignored; never commit real values.

### Where state lives

Code lives in git. Desk state lives in a data directory that is **never** committed:

| Variable | Default | Holds |
|---|---|---|
| `MLAB_DATA_DIR` | a shared project folder if present, else the checkout | everything below |
| `MLAB_JOURNAL_DIR` | `$MLAB_DATA_DIR/journal` | `events/`, legacy `trades.jsonl`, `JOURNAL.md` |
| `MLAB_REPORTS_DIR` | `$MLAB_DATA_DIR/reports` | charts and notes |
| `MLAB_CACHE_DIR` | `$MLAB_DATA_DIR/data/cache` | parquet price cache |
| `MLAB_IV_DIR` | `$MLAB_DATA_DIR/data/iv_history` | implied-vol history |
| `MLAB_TRANSCRIPTS_DIR` | `$MLAB_DATA_DIR/data/transcripts` | earnings call texts |
| `MLAB_CACHE_MAX_MB` | `750` | cache size before LRU pruning |

Pointing several machines or sessions at one `MLAB_DATA_DIR` gives them one shared journal and cache.

### Optional data keys

| Variable | Unlocks |
|---|---|
| `SEC_USER_AGENT` | identifies SEC EDGAR requests (`"Your Name you@example.com"`) |
| `FRED_API_KEY` | FRED JSON API (a keyless CSV route works without it) |
| `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY` | extra fundamentals, transcripts, news and calendars |

### Broker integrations (read-only)

**IG** (spread bets and CFDs). `src/mlab/providers/ig.py` only issues GET requests plus session login, account
switch and logout; it refuses every other method and path, and no order or deal endpoint exists in the code. It
reads accounts, positions, working orders, watchlists, market search, client sentiment, historical prices (cache-first,
to protect IG's weekly data allowance) and streaming quotes.

Credentials reach the client through proxy-injected network secrets or environment variables, and are never stored
in the repo:

| Credential | How it is supplied |
|---|---|
| API key | a network secret that adds the `X-IG-API-KEY` header on the IG host, or `IG_API_KEY` / `IG_DEMO_API_KEY` / `IG_LIVE_API_KEY` |
| Username and password | a body-parameter network secret on the IG session endpoint that fills the login body, or `IG_USERNAME` / `IG_PASSWORD` (and `IG_DEMO_*` / `IG_LIVE_*` overrides) |
| `IG_ACC_TYPE` | always set: `DEMO` or `LIVE` |
| `IG_ACC_NUMBER` | optional, account to switch to after login |

See `docs/NETWORK.md` for the secret setup.

**Kraken.** Public market data (prices, OHLC) is used today with no key. Account access, when added, follows the same
rule: a query-only API key from environment variables, read-only client, no trading endpoints.

Credentials are read from the environment at runtime. The CLI never prints, logs or writes them.

## The `mlab` CLI

`./mlab --help` lists every command; `./mlab <command> --help` shows its options. Add `--json` to any command for
machine-readable output. The full reference is on the wiki ([CLI Reference](docs/wiki/CLI-Reference.md)).

| Group | Commands |
|---|---|
| Setup and data | `doctor`, `price`, `cache`, `universes`, `crypto` |
| Technicals | `ta`, `chart`, `tafull`, `catalog`, `patterns`, `signals`, `firing`, `scan`, `rs`, `corr` |
| Options | `options`, `payoff` |
| Earnings | `earnings`, `earnings-text`, `earnings-align` |
| News and sentiment | `news`, `sentiment`, `social` |
| Macro and positioning | `macro`, `cot`, `cot-extremes`, `lens soros` |
| Fundamentals | `dd`, `lens buffett`, `lens burry` |
| Risk and journal | `size`, `card`, `journal` |
| Broker (read-only) | `ig login|accounts|positions|orders|watchlists|search|market|sentiment|prices|stream|...` |
| Quant | `backtest`, `algo list|backtest|validate|signal|paper|paper-status`, `hypo new|test|list` |

Symbols follow Yahoo conventions (`AAPL`, `^GSPC`, `EURUSD=X`, `GC=F`, `VOD.L`), with prefixes for other routes:
`crypto:BTC`, `ig:<EPIC>`.

Sizing example: risk 1% of a 20,000 account on a long from 7,410 with a stop at 7,330.

```bash
./mlab size --equity 20000 --risk 1 --entry 7410 --stop 7330
./mlab card "Example index" --bias long --entry 7400 7420 --stop 7330 --targets 7600 7750
```

## Desk agents and playbooks

Opening the repo in Claude Code loads [`CLAUDE.md`](CLAUDE.md), six desk agents and a set of playbooks. A "PM"
session fans the desks out in parallel, weighs their output and writes the call plus trade card.

| Agent (`.claude/agents/`) | Crunches |
|---|---|
| `technicals-desk` | the TA catalogue, patterns, levels, regime, signal hit rates |
| `options-desk` | vol surface, positioning, expected move, structures |
| `earnings-desk` | reports vs price reactions, call tone, next-print pricing |
| `news-sentiment-desk` | headlines, filings, social, sentiment vs price |
| `macro-desk` | rates, liquidity, credit, USD, cross-asset trends, COT |
| `fundamentals-desk` | SEC financials, insiders, quality and value checklists, DCF |

Playbooks (`.claude/skills/`): `trade-idea`, `morning-brief`, `lens-soros`, `lens-buffett`, `lens-burry`,
`ig-account`, `trade-journal`, `market-data`, `quant-research`. See the wiki's
[Desk Workflow](docs/wiki/Desk-Workflow.md).

## Quant research layer

`src/mlab/quant/` holds a strategy library, a portfolio backtest engine with IG-style costs (spread on turnover,
overnight funding on gross exposure), validation, sizing and paper books.

- **Strategies (21):** trend (SMA/EMA cross, Donchian, Supertrend, time-series momentum, MACD, ADX), breakout
  (Keltner, squeeze, NR7), mean reversion (RSI, RSI(2), Bollinger, z-score, IBS), seasonality (turn of month),
  cross-sectional momentum, dual momentum, inverse vol, risk parity, pairs. Overlays: `vol_target`, `regime_filter`.
- **No look-ahead:** a decision at the close of bar *t* earns bar *t+1*; tests enforce this by truncation.
- **Validation:** in-sample/out-of-sample split, walk-forward, probabilistic and deflated Sharpe, block bootstrap
  confidence intervals, permutation tests, parameter-sensitivity grids.
- **Modes:** `backtest`, `signal` (today's target), `paper` (books in `$MLAB_DATA_DIR/paper/`). There is no live mode.

```bash
./mlab algo list
./mlab algo validate tsmom ^GSPC --grid '{"lookback":[126,252]}'
./mlab algo backtest xs_momentum SPY EFA EEM GLD TLT -p 10y --params '{"top":2}'
./mlab algo paper rsi2 SPY QQQ --book rsi2-us
```

**Hypotheses** are pre-registered before any result is seen: `mlab hypo new` writes a file under
`research/hypotheses/` with the claim, data, costs and pass bars on a `hypo/<yyyymmdd>-<slug>` branch;
`mlab hypo test` grades it once, PASS or FAIL, and it merges either way so failures stay on the record. See
[Hypothesis Testing](docs/wiki/Hypothesis-Testing.md) and
[Validation and Overfitting](docs/wiki/Validation-and-Overfitting.md).

## Project layout

```
src/mlab/              Python package; CLI entry point mlab.cli:main
  providers/           yahoo, stooq, macro (FRED, ECB), sec, crypto (Binance, Kraken, CoinGecko), ig (read-only)
  quant/               strategies, engine, validation, sizing, paper, hypothesis
  ta.py, ta_catalog.py, signal_lab.py, options.py, earnings.py, news.py, cot.py, lenses.py, risk.py, journal.py, ...
tests/                 pytest, offline only
.claude/agents/        desk agents
.claude/skills/        playbooks
docs/                  NETWORK.md (hosts to allowlist), SKILLS_MAP.md, wiki/ (mirrored to the GitHub wiki)
research/hypotheses/   pre-registrations and verdicts
.github/               CI, security scans, CodeQL, wiki sync, issue and PR templates, Dependabot
bootstrap.sh, mlab     one-command setup and the self-bootstrapping CLI wrapper
```

## Development

```bash
bash bootstrap.sh
~/.venvs/market-lab/bin/pytest -q
```

Tests run offline. Issues use the templates in `.github/ISSUE_TEMPLATE/` (hypothesis, strategy, bug, data source);
PRs follow `.github/pull_request_template.md`. CI runs pytest, the security workflow and CodeQL on every PR. See
[Contributing](docs/wiki/Contributing.md).

The wiki is generated from `docs/wiki/`: edit pages there, and the wiki-sync workflow mirrors them to the GitHub wiki
on every push to `main`.

## Security

No credentials, journal entries, reports, caches or paper books are ever committed; gitleaks scans the full history
on every PR. Report vulnerabilities privately through **Security > Report a vulnerability**. Details in
[SECURITY.md](SECURITY.md).
