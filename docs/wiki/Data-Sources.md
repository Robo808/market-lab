# Data Sources

`bash mlab doctor` shows which hosts are reachable from where you run it and which keys are present. Full host list
for allowlisting: `docs/NETWORK.md`.

For backtests use frozen snapshots, not live pulls: see [[Datasets]].

## Providers

| Source | Powers | Key | Notes |
|---|---|---|---|
| Yahoo (yfinance) | Prices (stocks, ETFs, indices, FX, futures), options chains, earnings dates | none | Primary price source. Blocks robots on web pages |
| Stooq | Price fallback, long daily history | none | Used when Yahoo fails |
| FRED | Rates, curve, inflation, liquidity (Fed balance sheet, TGA, RRP), credit spreads | optional `FRED_API_KEY` | Keyless CSV route works without a key |
| ECB / Frankfurter | FX reference rates | none | |
| SEC EDGAR | 10y financials, filings, 8-Ks, insiders | `SEC_USER_AGENT` ("Name email") | US issuers only |
| Binance, Kraken, CoinGecko | Crypto prices (`crypto:BTC`), market data | none | Binance then Kraken |
| alternative.me | Crypto fear and greed | none | |
| CFTC | Commitments of Traders | none | Weekly, Friday release for Tuesday data |
| Google News RSS, GDELT | News, tone | none | |
| StockTwits, Reddit, HN (Algolia) | Social sentiment, attention | none | |
| Alpha Vantage, FMP, Finnhub | Fundamentals, transcripts, news, calendars | `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY` | Optional |
| IG (read-only) | Exact traded instrument prices, positions, sentiment, market search | proxy-injected network secrets or env vars; `IG_ACC_TYPE` (see [[Getting Started]]) | 10k historical points per week: always cache-first via `mlab price ig:...` |

Credentials come from proxy-injected secrets or env vars, never from the repo. Never print, log or commit them.

## Known gaps

- **Cloud container**: the default environment allows package indexes and GitHub only; market data hosts are refused
  until a cloud environment with them allowlisted is added (`docs/NETWORK.md`).
- **Stooq**: intermittent timeouts; treat as a fallback, not a primary.
- **Reddit**: blocked from cloud IPs. Use StockTwits and HN, or run on a local machine.
- **GDELT**: the DOC API throttles bursts (429s); space requests out and rely on the cache. For history use
  `mlab newsstore gdelt`, which reads the 15-minute GKG raw files from data.gdeltproject.org (no quota, about
  15 s per day of files with 6 workers) into `data/news/gdelt/` with the file time as the availability time.
  Matches carry `via` (`org:`, `platform:` for "posted on Facebook"-type mentions, `person:` for CEO stories).
  GDELT requires a citation and a link to https://www.gdeltproject.org.
- **LSE prices**: Yahoo quotes most `.L` shares in pence (GBp) and some series flip to pounds; check the units before
  computing returns or stakes. A 100x jump in a series is a units error, not a move.
- **IG allowance**: historical price requests burn the weekly 10k-point allowance. Use Yahoo or Stooq for long history
  of the underlying, IG for recent bars of the exact instrument.
- **Epics**: DFB (spread bet) and CFD epics differ; confirm with `bash mlab ig search`.
- **Survivorship**: free sources have no delisted names. See [[Validation and Overfitting]].

## Fallbacks, in order

1. Built-in provider fallbacks (Yahoo to Stooq, Binance to Kraken) and the parquet cache.
2. Connected claude.ai connectors (Alpha Vantage, FMP, Financial Datasets, Bigdata.com, TEXT TO QUANT, Meltwater,
   CoinMarketCap): they route outside the container's network policy.
3. WebSearch / WebFetch for news, transcripts and quotes, citing URL and time.
4. Run the same `bash mlab` command on a local machine (for example via Claude Code Remote Control), where the network is open.

Always say which source a number came from and when.

## Adding a source

Open a `[data-source]` issue (template asks for access, host and reliability), branch `data/<source>`, add the provider
under `src/mlab/providers/`, add the host to `docs/NETWORK.md` and the doctor probe, and test it offline with recorded fixtures.
