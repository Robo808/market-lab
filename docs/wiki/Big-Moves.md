# Big Moves

A small research table of the largest daily moves in a few markets, each joined to the state of the world at the
prior close, plus a matched sample of ordinary days with the same columns. It exists so clustering and classification
can ask two questions: what does the world look like before a large move, and what caused it? Issue #78.

```bash
bash mlab bigmoves build                 # fetch raw sources once, build 1990-2019, write data/bigmoves/<version>/
bash mlab bigmoves build --no-fetch      # rebuild from the saved raw responses only
bash mlab bigmoves show --top 25         # largest moves by |z|
bash mlab bigmoves study --perm 500      # exploratory trials against permutation nulls and coin-flip stubs
```

## Prior work this reuses

Attributing large index moves to news is an established line of research (event attribution / jump attribution), sitting
between event studies and macro regime work. It is neither company due diligence nor technical analysis.

| Work | What it found | How it is used here |
|---|---|---|
| Niederhoffer (1971), "The analysis of world events and stock prices", J. Business 44(2) | Large world events cluster with large index moves | Motivation |
| Cutler, Poterba and Summers (1989), "What moves stock prices?", J. Portfolio Management 15(3) | Most of the 50 largest S&P 500 moves 1941-87 had no identifiable news | Expect a large "unknown" share |
| Fair (2002), "Events that shook the market", J. Business 75(4) | Most large intraday S&P futures moves map to US macro and monetary news, many do not | Same |
| Andersen, Bollerslev, Diebold and Vega (2003), AER | Scheduled macro releases move prices at known times | Calendar columns |
| Lee and Mykland (2008), RFS | Nonparametric jump test scaled by local volatility | Detector is vol-scaled (z), not a fixed % threshold |
| Baker, Bloom, Davis and Sammon, "What triggers stock market jumps?", NBER w28687 (2021, rev. 2025) | Human-coded next-day newspaper causes of every US daily move above 2.5% since 1900; policy news dominates up-jumps after crashes; clearer causes, lower later volatility | Cause labels (`bbds_*`), from the public data at stockmarketjumps.com |
| Caldara and Iacoviello (2022), AER; Baker, Bloom and Davis (2016), QJE | Daily geopolitical risk and economic policy uncertainty indices | `pre_gprd_*`, `pre_epu_us_*`, `pre_tpu_epu_*` |

## Layout

`$MLAB_DATA_DIR/data/bigmoves/`

| Path | What |
|---|---|
| `raw/bbds_jumps.xlsx` | Baker-Bloom-Davis-Sammon workbook as downloaded (Google Sheets xlsx export) |
| `raw/fomc/fomchistoricalYYYY.htm` | federalreserve.gov FOMC historical pages, one per year |
| `raw/cot_tff_<market>.json` | CFTC Traders in Financial Futures rows as returned by the Socrata API |
| `<version>/moves.parquet` | one row per (market, move day) |
| `<version>/controls.parquet` | same columns for random ordinary days (no `bbds_*`), `kind = control` |
| `<version>/bbds.parquet` | all 1,179 BBDS jump days 1900-2022 with category shares |
| `<version>/fomc.parquet` | FOMC meetings, calls and unscheduled actions |
| `<version>/news_daily.parquet` | GDELT macro news per UTC day: count, tone, theme shares |
| `<version>/manifest.json` | parameters, sources, licences, per-item status, sha256 of every file |

Versions are dated (`vYYYYMMDD`) and never edited in place. The default build stops at 2019-12-31 because 2020+ is the
locked test split of the frozen datasets; `--allow-test` lifts it and the access is logged by `mlab datasets`.

## Detector

`z_t = r_t / sigma_{t-1}`, with `r_t` the daily log return and `sigma` the zero-mean EWMA of squared returns (span 60)
through the previous bar. A move is any day with `|z| >= 3` (`--z`). Moves within 5 bars of each other share an
`episode`; both are kept because a crash and its rebound are different events. `episode_rank` orders them by `|z|`.

Vol scaling means a 3% fall in a calm year counts and a 3% fall in October 2008 may not: the table is about surprises
relative to the regime, which is what a trading desk feels. Plain percentage thresholds (the BBDS 2.5% rule) are
available through `day_ret`.

Controls: 5 per move (`--controls`), drawn with a fixed seed from bars at least 5 bars away from any move, inside the
same date window, after the first 252 bars of history.

## Columns

Point-in-time rule: `pre_*` columns use only data dated strictly before the row's date. Other calendars are joined as
of their last earlier date; CFTC positions become usable 3 days after the Tuesday report date (Friday release).
`day_*` columns describe the row's own day: context and labels, never prediction inputs.

| Column | Meaning | Source, coverage |
|---|---|---|
| `date`, `market`, `kind` | Trading day, SPX / NDX / FTSE / BTC, move or control | |
| `day_close`, `day_ret`, `day_z`, `day_direction` | Close, log return, vol-scaled z, sign | xasset-daily (Yahoo), crypto-daily (Binance BTCUSDT from 2017-08) |
| `pre_sigma` | Ex-ante daily sigma used for `day_z` | same |
| `episode`, `episode_rank` | Move cluster id and rank (controls: -1, 0) | |
| `pre_ret_{1,5,20,60}d` | Log return over the previous n bars | own market |
| `pre_rv_20d`, `pre_rv_ratio_5_60` | Annualised 20-day realised vol; 5-day over 60-day vol | own market |
| `pre_dd_252d`, `pre_dist_sma50`, `pre_dist_sma200`, `pre_up_days_10` | Drawdown from 1y high, distance to averages, up days in last 10 | own market |
| `pre_vix`, `pre_vix_chg5`, `pre_vix_z252` | VIX level, 5-day change, z vs prior year | ^VIX from 1990 |
| `pre_tnx`, `pre_tnx_chg20`, `pre_curve_10y_3m` | US 10y yield, 20-day change, 10y minus 3m | ^TNX, ^IRX |
| `pre_{dxy,oil,gold,hyg,spx}_ret20`, `pre_spx_ret1` | 20-day log returns of USD index, WTI, gold, high yield credit, S&P (non-SPX rows) | oil and gold from 2000, HYG from 2007 |
| `pre_{gprd,gprd_act,gprd_threat,epu_us,tpu_epu}_{7d,z}` | 7-day mean of the daily index and its z vs the prior year | policy-daily dataset |
| `pre_cot_lev_net_oi`, `pre_cot_am_net_oi`, `pre_cot_lev_index` | Leveraged funds and asset manager net as share of open interest; 3-year COT index of leveraged net | CFTC TFF from 2006 (bitcoin from 2017-12); none for FTSE |
| `pre_news_n`, `pre_news_tone`, `pre_z_news_*` | Previous UTC day's GDELT macro article count, tone, theme shares; `pre_z_` = z vs its prior 60 days | GDELT from 2015-02-19 |
| `day_z_news_*` | Same for the move day itself | same |
| `day_weekday`, `day_month_end`, `day_opex`, `day_first_friday` | Calendar flags (`day_first_friday` is a payrolls proxy, not the BLS schedule) | |
| `day_fomc`, `day_fomc_kind` | FOMC decision, call or unscheduled action dated that day | federalreserve.gov |
| `pre_days_to_fomc`, `pre_days_since_fomc` | Trading days to the next and since the last scheduled FOMC decision | same |
| `bbds_*` | BBDS cause shares (17 categories), `bbds_policy`, clarity, ease, journalist confidence, coders, `bbds_primary` (largest share), `bbds_match` | US rows (SPX, NDX) on BBDS jump days |

Known caveats:
- FOMC conference calls are dated by the call; some announcements came the next morning (for example the call of
  2008-01-21 and the cut announced 2008-01-22).
- GPR and EPU are compiled from each day's newspapers and published later; they describe the world state, a live
  forecasting use needs their real release lag.
- GDELT macro coverage has gaps (about 13% of Sep-Nov 2020 source files are missing upstream); missing days are NaN.
- BBDS codes CRSP value-weighted index days; on SPX and NDX rows the label is joined by date.

## Research rules

The `study` action runs four counted trials with one bar fixed in advance (z >= 3 against a permutation null and
BH-adjusted p <= 0.01), after checking that seeded coin-flip labels through the same pipeline all fail it (L-012). It
reads only 1990-2019. A PASS is a candidate to pre-register with `mlab hypo`, not a finding.

## Growing it

The demo is under 1.5 MB. A full build changes only parameters: more markets in `MARKETS`, `--start` back to the
first bar, more GDELT themes, more controls. The slow step is reading the GDELT macro day files (about 10 minutes for
five years from the shared folder); later builds reuse `news_daily` when it already covers the window.
