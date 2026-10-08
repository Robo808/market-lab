# CLI Reference

`bash mlab --help` lists every command and `bash mlab <command> --help` its options. If this page and `--help`
disagree, `--help` wins.

Common options on data-driven commands: `--interval/-i` (`1d`, `1h`, `4h`, ...), `--period/-p` (`6mo`, `2y`, `10y`,
`max`), `--start YYYY-MM-DD`, `--refresh` (bypass the cache), `--json` (machine-readable output).

Symbols use Yahoo conventions (`AAPL`, `^GSPC`, `^FTSE`, `EURUSD=X`, `GC=F`, `VOD.L`) plus route prefixes:
`crypto:BTC` (Binance, Kraken fallback) and `ig:<EPIC>` (IG, cache-first).

## Setup and data

| Command | Does | Example |
|---|---|---|
| `doctor` | Network reachability, keys present, code and data dirs, cache | `mlab doctor` |
| `price` | OHLCV history with source and timestamp (`--tail`, `--csv`) | `mlab price ^GSPC -p 5y --tail 10` |
| `cache` | `stats`, `prune`, `clear [--provider]` | `mlab cache stats` |
| `universes` | Named symbol universes and common IG epics | `mlab universes` |
| `crypto` | Crypto overview plus fear and greed | `mlab crypto` |

## Technicals

| Command | Does | Example |
|---|---|---|
| `ta` | Snapshot, S/R, pivots, Fibonacci, signals; `--chart` writes HTML to `reports/` | `mlab ta NVDA --chart` |
| `chart` | Interactive candlestick chart | `mlab chart AAPL -p 1y` |
| `tafull` | Latest readings across the whole catalogue, recent candles, active chart patterns | `mlab tafull NVDA` |
| `catalog` | Lists the catalogue: indicators, candles, chart patterns | `mlab catalog` |
| `patterns` | Candlestick and chart-pattern history over the window | `mlab patterns ^FTSE -p 2y` |
| `signals` | Every signal scored against the base rate, BH-adjusted p-values (`--horizons`, `--entry`) | `mlab signals NVDA -p 10y` |
| `firing` | Signals firing now plus their hit rate and edge on this instrument | `mlab firing NVDA` |
| `scan` | Cross-asset scan by universe or symbol list (`--sort 3m`) | `mlab scan -u indices -u fx --sort 3m` |
| `rs` | Relative strength ranking | `mlab rs SPY QQQ IWM EFA EEM` |
| `corr` | Return correlation matrix | `mlab corr SPY TLT GLD DX-Y.NYB` |

## Options

| Command | Does | Example |
|---|---|---|
| `options` | Term structure, skew, expected move, max pain, PCR, GEX and gamma flip, unusual activity, IV vs RV | `mlab options NVDA` |
| `payoff` | Payoff and greeks for a structure; legs as `"C 100 30d +1 @2.5"` | `mlab payoff --spot 100 --vol 0.25 --leg "C 100 30d +1 @2.5" --leg "C 110 30d -1 @0.8"` |

## Earnings

| Command | Does | Example |
|---|---|---|
| `earnings` | History vs reactions: gap, day 1, drift, ATR z-score, abnormal return (`--bench`) | `mlab earnings NVDA --quarters 12` |
| `earnings-text` | Tone, guidance and topics of the latest transcript or 8-K release | `mlab earnings-text NVDA` |
| `earnings-align` | Per-quarter join of surprise, tone, guidance and reaction, with correlations | `mlab earnings-align NVDA` |

## News and sentiment

| Command | Does | Example |
|---|---|---|
| `news` | Headlines, sentiment, event types, attention for a ticker or topic (`--days`, `--sources`) | `mlab news NVDA --days 7` |
| `sentiment` | Daily sentiment index, attention z-score, lead/lag vs price | `mlab sentiment TSLA` |
| `social` | StockTwits and Reddit crowd: bull ratio, top posts | `mlab social GME` |

## Macro and positioning

| Command | Does | Example |
|---|---|---|
| `macro` | FRED dashboard; `--liquidity` adds Fed balance sheet minus TGA minus RRP | `mlab macro --liquidity` |
| `cot` | CFTC Commitments of Traders (`--report legacy|disagg|tff`); `list` for all markets | `mlab cot gold` |
| `cot-extremes` | Positioning extremes across mapped markets | `mlab cot-extremes` |
| `lens soros` | Macro and reflexivity screen across assets | `mlab lens soros` |

## Fundamentals

| Command | Does | Example |
|---|---|---|
| `dd` | 10y financials, filings, insiders, valuation snapshot (`--years`) | `mlab dd AAPL` |
| `lens buffett` | Quality, owner earnings, margin of safety checklist | `mlab lens buffett AAPL` |
| `lens burry` | Deep value and balance-sheet forensics checklist | `mlab lens burry XYZ` |

## Risk, trade cards and journal

| Command | Does | Example |
|---|---|---|
| `size` | Stake per point for an account, risk %, entry and stop | `mlab size --equity 20000 --risk 1 --entry 7410 --stop 7330` |
| `card` | Builds the trade card (R:R, stake per point); `--journal` logs it | `mlab card "Example index" --bias long --entry 7400 7420 --stop 7330 --targets 7600 7750 --journal` |
| `journal` | `list`, `update ID --status/--fill/--exit/--note`, `review`, `render` | `mlab journal review` |

`card` options: `--risk`, `--horizon`, `--conviction`, `--catalyst`, `--kills`, `--epic`, `--lens`, `--thesis`,
`--source`, `--equity`, `--point-size`, `--status idea|open`.

## IG (read-only)

`mlab ig <sub>` with `login`, `accounts`, `positions`, `orders`, `watchlists`, `watchlist`, `search`, `market`,
`snapshot`, `sentiment`, `prices`, `stream`, `activity`, `transactions`. Options: `--env DEMO|LIVE`, `--interval`,
`--start`, `--tail`, `--refresh`, `--seconds` (stream), `--days` (history). See [[Getting Started]] for credentials.

## Quant

| Command | Does | Example |
|---|---|---|
| `backtest` | Single-instrument backtest with IG costs (`--strategy`, `--spread` in points, `--funding`) | `mlab backtest ^GSPC --strategy donchian --spread 0.6` |
| `algo list` | The strategy library and overlays | `mlab algo list` |
| `algo backtest` | Backtest any library strategy (`--params`, `--spread-bps`, `--funding`, `--vol-target`, `--regime-filter`) | `mlab algo backtest xs_momentum SPY EFA EEM GLD TLT -p 10y --params '{"top":2}'` |
| `algo validate` | OOS split, PSR/DSR, bootstrap, permutation, sensitivity (`--grid`, `--trials`, `--oos-start`, `--n-perm`) | `mlab algo validate tsmom ^GSPC --grid '{"lookback":[126,252]}'` |
| `algo signal` | Today's target position or weights | `mlab algo signal dual_momentum SPY EFA BIL` |
| `algo paper` | Rebalance a paper book to today's target (`--book`, `--capital`) | `mlab algo paper rsi2 SPY QQQ --book rsi2-us` |
| `algo paper-status` | Equity, drawdown, holdings and fills of one or all books | `mlab algo paper-status` |
| `hypo new` | Pre-register a hypothesis (`--claim`, `--strategy`, `--symbols`, `--spec`, `--branch`) | `mlab hypo new ftse-rsi2 --claim "..." --branch` |
| `hypo test` | Grade a pre-registered file once: PASS or FAIL | `mlab hypo test research/hypotheses/H-....md` |
| `hypo list` | All hypotheses and their verdicts | `mlab hypo list` |

See [[Strategy Library]], [[Validation and Overfitting]], [[Hypothesis Testing]] and [[Paper Trading]].
