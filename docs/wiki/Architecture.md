# Architecture

## Layout

```
src/mlab/            Python package, CLI entry point mlab.cli:main
  quant/             strategy library, portfolio engine, validation, sizing, paper, hypotheses
.claude/agents/      the six desk agents
.claude/skills/      playbooks (trade-idea, morning-brief, lenses, ig-account, trade-journal, market-data)
tests/               pytest, offline only
docs/                network, skills map, this wiki (docs/wiki)
research/hypotheses/ pre-registered hypotheses and their verdicts
bootstrap.sh, mlab   one-command setup and the self-bootstrapping CLI wrapper
```

## Core modules (`src/mlab`)

| Module | Does |
|---|---|
| `config` | Resolves `MLAB_DATA_DIR` and the per-item dirs, reads env vars |
| `net` | HTTP session, retries, timeouts, proxy handling |
| `data` | `get_prices(symbol, interval, period)`: routes `crypto:`, `ig:` and plain symbols to providers, with fallbacks |
| `cache` | Parquet price cache in `data/cache/`, fetches only the missing head and tail, LRU-pruned at `MLAB_CACHE_MAX_MB` |
| `storage` | Atomic writes, in-container file locks, write-once event files for state shared across threads (see [[Design Standards]]) |
| `providers/` | `yahoo`, `stooq`, `macro` (FRED, ECB FX), `sec` (EDGAR), `crypto` (Binance, Kraken, CoinGecko), `ig` (read-only) |
| `universes` | Named symbol lists for scans (`indices`, `fx`, ...) |
| `ta` | Indicator primitives (SMA, EMA, RSI, ATR, MACD, Bollinger, Keltner, ADX, Supertrend, ...) |
| `ta_catalog` | The full TA catalogue: 60+ indicators, candlesticks, chart patterns, levels, regime |
| `signal_lab` | Every catalogue signal scored against the base rate on one instrument, BH-adjusted |
| `backtest` | Vectorised single-instrument backtester with IG costs (spread in points, overnight funding), walk-forward |
| `stats` | Returns, drawdown, summary (CAGR, Sharpe, Sortino, Calmar, max DD), correlation, relative strength, z-score, pair spread |
| `risk` | Stake per point, ATR stop, fractional Kelly, reward:risk, `TradeCard` |
| `options` | IV term structure, skew, expected move, max pain, GEX and gamma flip, unusual activity, payoffs |
| `earnings` | Earnings reactions (gap, day 1, drift, abnormal return), transcript tone, alignment |
| `news` | Google News, Yahoo, GDELT tone, SEC 8-K, StockTwits, Reddit, HN, sentiment index |
| `cot` | CFTC Commitments of Traders, positioning extremes |
| `lenses` | Soros, Buffett, Burry checklists and scoring |
| `journal` | Event-sourced trade journal: `events/` (one file per change) folded over legacy `trades.jsonl`, rendered to `JOURNAL.md` |
| `charts` | Plotly HTML charts into `reports/` |
| `cli` | Every `mlab` command |

## The quant package (`src/mlab/quant`)

| Module | Does |
|---|---|
| `strategies.py` | `REGISTRY`: name to strategy function plus default params and family. Single-instrument strategies take an OHLCV frame and return a position in [-1, 1]; portfolio strategies take a panel of closes and return weights per asset. Overlays wrap either. See [[Strategy Library]] |
| `engine.py` | Portfolio backtest: applies weights decided at close t to returns of bar t+1, charges IG spread on turnover and funding on gross exposure, returns equity, trades, exposure and cost drag |
| `validation.py` | IS/OOS split, walk-forward, PSR, DSR, block bootstrap CI on Sharpe, permutation test, parameter-sensitivity grid. See [[Validation and Overfitting]] |
| `sizing.py` | Fixed-fractional, vol targeting, inverse-vol, risk parity, fractional Kelly. See [[Risk and Sizing]] |
| `paper.py` | Paper books in `$MLAB_DATA_DIR/paper/<book>.json`: rebalance to target, mark to market, log fills. See [[Paper Trading]] |
| `hypothesis.py` | `mlab hypo new/test/list`: pre-registration files, spec parsing, verdict against pre-set bars. See [[Hypothesis Testing]] |

## Agents and skills

- `.claude/agents/`: `technicals-desk`, `options-desk`, `earnings-desk`, `news-sentiment-desk`, `macro-desk`,
  `fundamentals-desk`. Each crunches its area with `mlab` and returns numbers with sources. See [[Desk Workflow]].
- `.claude/skills/`: `trade-idea`, `morning-brief`, `lens-soros`, `lens-buffett`, `lens-burry`, `ig-account`,
  `trade-journal`, `market-data`. Playbooks the PM session follows.

## Data flow

```
providers (Yahoo, Stooq, FRED, SEC, crypto, CFTC, news, IG read-only)
   -> data.get_prices / provider calls
   -> cache (parquet in $MLAB_DATA_DIR/data/cache)
   -> ta, ta_catalog, signal_lab, options, earnings, news, cot, lenses     (desk analysis)
   -> quant.strategies -> quant.engine -> quant.validation                 (systematic research)
   -> risk.TradeCard -> journal ($MLAB_DATA_DIR/journal)                   (discretionary calls)
   -> quant.paper ($MLAB_DATA_DIR/paper)                                   (systematic forward test)
```

Every number carries its source and timestamp. Nothing writes to IG.
