# Datasets (frozen, offline)

Strategies are tested the way ML models are: on a **frozen snapshot** with fixed **train / validation / test**
dates, never on whatever a live API returns today. A snapshot is versioned, checksummed and immutable, so a
result can always be re-run on the exact bytes it was produced from.

```bash
bash mlab datasets catalog                 # what can be built, source, splits and licence of each
bash mlab datasets build xasset-daily      # or: build all
bash mlab datasets list                    # built versions, date ranges, size
bash mlab datasets show duka-1h            # members, quality checks, licence, splits
bash mlab datasets verify xasset-daily@v20261010   # re-hash every file against the manifest

# research loop: tune on train, choose on validation, grade once on test
bash mlab algo backtest tsmom ^FTSE ^GSPC --dataset xasset-daily --split train
bash mlab algo validate tsmom UK100 US500 --dataset duka-1d --split train+validation
bash mlab algo backtest tsmom UK100 --dataset duka-1d --split test --allow-test   # logged
```

```python
from mlab.datasets import load
frames = load("duka-1h", ["UK100", "US500"], split="train")   # {symbol: DataFrame}, no network
```

## Rules

- **Immutable versions.** `build` always writes a new version (`v<YYYYMMDD>`, then `.2`, `.3`); files are
  written read-only and hashed (SHA-256) in `manifest.json`.
- **Held-out test split.** Loading the test split needs `allow_test=True` / `--allow-test` and appends a line to
  `test_access.<version>.jsonl` next to the data, so the number of looks at the test set is on record (it feeds
  the trial count that the deflated Sharpe in [[Validation and Overfitting]] needs). Pre-registered hypotheses
  ([[Hypothesis Testing]]) should name the dataset version and split they are graded on.
- **Quality checks at build time**, stored per member: rows, date range, NaNs, duplicate stamps, largest gap,
  non-positive prices, OHLC consistency, jumps over 25% (daily) or 10% (hourly), median spread.
- **Data stays private.** Snapshots live in `$MLAB_DATA_DIR/data/datasets/` (override with
  `MLAB_DATASETS_DIR`), never in this repository: several sources allow personal use only.

## Layout

```
data/datasets/<name>/<version>/manifest.json
data/datasets/<name>/<version>/symbol=<SYMBOL>/data.parquet     # zstd, long format
```

One schema per kind: bars are `time, symbol, open, high, low, close, volume, adj_close, spread`; series are
`time, symbol, value`; events are `time, symbol, eps_estimate, eps_actual, surprise_pct, hour_et`. Times are UTC. The Hive-style `symbol=` partitions register as an Iceberg table without
rewriting files, so snapshots move into the planned lake (issue #34) as they are; until then the shared folder
is the store, which is a stopgap, not the platform.

## Catalogue

Licences and coverage were checked on the providers' own pages and GitHub repos on 2026-10-10.

| Dataset | Source | Members | From | Licence (provider's terms) |
|---|---|---|---|---|
| `xasset-daily` | Yahoo (yfinance) | 47: world indices, FX majors, gold/oil/metals front-month futures, UST yields, US ETFs, London UCITS ETFs (VWRP, VUKG, ISF, IGLT, VFEG, VJPA, VUAG, SGLN...), BTC, ETH | max available | Personal use only ([yfinance README](https://github.com/ranaroussi/yfinance), [Yahoo API terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html)) |
| `us-stocks-daily` | Yahoo (yfinance) | 25 US mega caps (AAPL, MSFT, NVDA, AMZN, TSLA, META...) plus SPY, QQQ, IWM, DIA, sector ETFs, VIX; split- and dividend-adjusted | max available | Personal use only (as above) |
| `us-earnings` | Yahoo (yfinance) | Earnings dates, EPS estimate, actual, surprise and announcement hour for the same mega caps (Yahoo returns up to 100 events, about 25 years) | 2000 | Personal use only (as above) |
| `us-stocks-1h`, `us-stocks-5m`, `us-stocks-1m` | Yahoo (yfinance) | The 25 mega caps plus SPY, QQQ, IWM, DIA | rolling: Yahoo keeps 730 days of 1h, 60 days of 5m, 30 days of 1m | Personal use only (as above). **Accumulating**: each build merges the previous version, so a weekly build grows the history past Yahoo's window |
| `duka-stocks-1h`, `duka-stocks-1m` | Dukascopy datafeed | 13 US mega cap CFDs (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, NFLX, AMD, JPM, LLY, PLTR) plus SPY, QQQ, IWM, DIA; mid OHLC plus spread | 2017-02 (1h); last 120 days, accumulating (1m) | As Dukascopy above. Start dates from [dukascopy-node's instrument metadata](https://github.com/Leo4815162342/dukascopy-node) |
| `duka-1d`, `duka-1h` | [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) datafeed | 16: 7 FX majors, XAU/USD, Brent, WTI, UK100, US500, USTEC, DE40, EU50, JP225 (CFD and spot quotes, mid OHLC plus spread) | 2003 (1d), 2012 (1h) | No published licence; treat as personal use |
| `ust-daily` | [Fed H.15](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15) | 3m to 30y constant-maturity yields | 1962 | US government public data |
| `boe-daily` | [Bank of England database](https://www.bankofengland.co.uk/boeapps/database/) | SONIA, Bank Rate, 5/10/20y gilt par yields, GBP/USD, GBP/EUR, GBP/JPY | 1975 | [UK Open Government Licence](https://www.bankofengland.co.uk/legal); some FX series excluded (LSEG-sourced) |
| `ecb-fx-daily` | [ECB data portal](https://data.ecb.europa.eu/) | EUR vs USD, GBP, JPY, CHF, AUD, CAD, SEK, NOK | 1999 | [Free reuse, source quoted](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html) |
| `eia-oil-daily` | [EIA](https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm) | Brent, WTI spot | 1986 | [Public domain](https://www.eia.gov/about/copyrights_reuse.php) |
| `crypto-daily`, `crypto-1h` | [Binance public data](https://github.com/binance/binance-public-data) | BTCUSDT, ETHUSDT | 2017-08 | [CC BY-NC-SA 4.0 + T&C](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md): personal backtesting allowed, no live execution on the data |
| `ff-factors-daily` | [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Mkt-RF, SMB, HML, RF, Mom | 1926 | Copyright Fama and French, free for research |

Default splits: train to 2014-12-31, validation 2015-2019, test 2020 to the snapshot end. Shorter histories
shift them: crypto train to 2021-06, validation to 2023-06; `duka-1h` train to 2018, validation 2019-2021;
factors train to 1999, validation 2000-2012.

**Why not FRED?** FRED's terms forbid storing, caching or archiving FRED content and its use in building
machine-learning systems ([FRED terms, sections II and (p)](https://fred.stlouisfed.org/legal/)). The same series
are frozen from their publishers instead (Fed H.15, ECB, EIA, BoE). `mlab macro` still reads FRED live.

**Why Dukascopy for IG instruments?** Its index and commodity quotes are CFDs on the cash index, the same kind
of price IG's daily spread bets track, with bid/ask so the spread is in the data. IG's own history costs
API allowance (10k points a week); Dukascopy covers years of it for free. The free feed rate-limits hard
(HTTP 429/503): `build` paces requests (`MLAB_DUKA_GAP`, default 1.2 s) and backs off, so `duka-1h` takes hours.
Dukascopy also offers a [requester-pays S3 bucket](https://www.dukascopy.com/wiki/en/development/data-export/)
for bulk tick history.

## Intraday for swing trading

Read a swing name on 1m, 5m, 1h and 1d together from these sets; indicators (RSI, volume, Heikin-Ashi, VWAP and
the rest of the catalogue) are computed from the stored OHLCV by `mlab.ta` / `mlab.ta_catalog`, so they are not
stored separately.

| Bars | Free depth now | Source |
|---|---|---|
| 1h | 2017 onwards | `duka-stocks-1h` (CFD quotes with spread); `us-stocks-1h` for exchange prints over the last 2 years |
| 1m | last 30 days (Yahoo), last 120 days (Dukascopy), growing with every weekly build | `us-stocks-1m`, `duka-stocks-1m` |
| 5m | last 60 days, growing | `us-stocks-5m` |

Free intraday windows roll off, so the intraday sets accumulate: build them at least weekly or the gap is lost
for good. Longer exchange-grade intraday history is a paid buy:

| Vendor | What it adds | Price (checked 2026-10-10) |
|---|---|---|
| [Massive (formerly Polygon.io)](https://massive.com/pricing) | Minute aggregates and flat files for all US stocks | Basic $0 (2 years, 5 calls/min), Starter $29/mo (5 years), Developer $79/mo (10 years), Advanced $199/mo (20+ years) |
| [FirstRate Data](https://firstratedata.com/b/22/stock-complete) | 1m/5m/30m/1h bars for 16k US tickers from 2000, one-off download | Bundle price on the buy page; updates $59.95/mo |
| [Databento](https://databento.com/pricing) | Exchange feeds incl. OHLCV-1s/1m, priced per GB | $125 starting credit, then pay as you go |

## Known source issues

- Yahoo: London listings mix pence and pounds (issue #7); jump checks in `show` flag them. Index levels are
  price-only; `=F` futures are unadjusted front-month rolls.
- Binance: spot files use microsecond timestamps from 2025 (handled); open issues on missing or relabelled bars
  ([issues](https://github.com/binance/binance-public-data/issues)). Monthly files only, so a snapshot ends at
  the last complete month.
- Dukascopy: complete months only; open issues on timeouts and gaps in the main downloader
  ([dukascopy-node issues](https://github.com/Leo4815162342/dukascopy-node/issues)). Index CFDs carry no
  dividends; oil CFDs have no roll in the series.
- `us-stocks-daily` holds today's mega caps, so tests on it carry survivorship bias (names that fell out of the
  top are missing). Fine for swing-timing rules on these names; not for "buy the biggest stocks" claims.
- Stooq: bulk downloads now need an API key ([pandas-datareader #1012](https://github.com/pydata/pandas-datareader/issues/1012)), so it is not a dataset source.

## What free data does not cover, and what fills it

| Gap | Paid filler |
|---|---|
| Back-adjusted continuous futures (ES, FTSE Z, FDAX, Brent, Long Gilt) | [Norgate Futures](https://norgatedata.com/futurespackage.php), US$270 a year; [Pinnacle CLC](https://pinnacledata2.com/clc.html); [FirstRate Data](https://firstratedata.com/) |
| Total-return daily indices (FTSE 100, S&P 500) and total-return gilts/Treasuries | Index vendor licences; model from yields plus [Shiller](https://shillerdata.com/) (monthly) meanwhile |
| Survivorship-free stock prices including delisted names | [Norgate US stocks](https://norgatedata.com/stockmarketpackages.php) (no UK package) |
| LBMA gold fix history | IBA licence ([LBMA](https://www.lbma.org.uk/prices-and-data/precious-metal-prices)); spot XAU/USD from Dukascopy instead |
| Options chains and implied-volatility history | [ORATS](https://orats.com/near-eod-data) (US$599 backfill), [Cboe DataShop](https://datashop.cboe.com/option-eod-summary), OptionMetrics |
| IG's own prices, spreads and funding | Only from IG (API allowance or account statements) |
| Exchange-grade intraday futures | [Databento](https://databento.com/pricing), pay as you go, US$125 starting credit |

Research-only sets worth adding later: AQR data library (time-series momentum, century of factor premia,
private use only), Jordà-Schularick-Taylor macrohistory (CC BY-NC-SA), Open Source Asset Pricing, Kraken
OHLCVT bulk files, BoE fitted yield curves.
