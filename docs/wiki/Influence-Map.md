# Influence map: politics, industries and markets (#58)

Goal: find how geopolitics, policy and industries lead markets and how markets feed back into them, with the same
discipline as every other test here (point-in-time data, discovery on train years only, pre-registered families).

## Data map

| Layer | Series | Source and store | From | Timing rule |
|---|---|---|---|---|
| Politics: events | Country-pair counts by CAMEO root (01-20) and quad class (verbal/material cooperation/conflict), Goldstein, tone, articles; where it happened | GDELT 2.0 Events, `mlab newsstore events` -> `data/news/gdelt/events/` | 2015-02-19 | Two buckets per UTC day: published before 20:00 UTC counts for day t, later counts for t+1 |
| Politics: themes | Daily intensity of GKG themes (tariffs, sanctions, elections, military, regulation, energy, labour) by country | GDELT GKG second pass, `mlab newsstore themes` -> `data/news/gdelt/themes/` | 2015-02-19 | Same cutoff |
| Economy and policy news | Article counts and tone per ECON_/EPU_ theme, every 15 minutes | GDELT GKG, `macro/` table | 2015-02-19 | Same cutoff |
| Companies | Headline-named articles, tone, novelty for 25 mega caps | GDELT GKG, `firm/` table, `features.primary` | 2015-02-19 | Same cutoff |
| Markets: industries | US sectors and politically exposed industries (defence ITA/XAR, banks KRE, oil and gas XOP, metals XME, semis SOXX/SMH, China internet KWEB, transport IYT, airlines JETS, solar TAN, uranium URA, lithium LIT, homebuilders XHB) | `influence-daily`, `us-stocks-daily` | 2006-2015 by ETF | Close of t |
| Markets: countries | 22 US-listed country ETFs (one close time) and 11 EM currencies (CNY, MXN, TRY, BRL, INR, KRW, TWD, ZAR, RUB, PLN, ILS) | `influence-daily` | 2000-2015 | Close of t; FX quotes are Yahoo's daily stamp |
| Markets: macro | Indices, G10 FX, oil, gas, gold, copper, grains, softs, Treasuries, gilts, credit, VIX, MOVE | `xasset-daily`, `influence-daily`, `ust-daily`, `boe-daily`, `duka-1d` | 1990s-2003 | Close of t |
| Ready-made risk indices | Geopolitical risk daily (total, threats, acts) and monthly (world plus 44 countries); US economic policy uncertainty daily; trade policy uncertainty daily (two versions) | `policy-daily`, `policy-monthly` (Caldara-Iacoviello, Baker-Bloom-Davis; CC BY) | 1985 (TPU_D 2015) | Monthly GPR is stamped at the month start but published in the first days of the next month; GPR daily updates on Mondays; EPU daily the next day. Lag before use |
| Positioning | CFTC COT net positioning by market | `mlab cot` | 2006 | Tuesday positions, published Friday: usable from the Friday |

Prior work first: the published indices above are the baseline. Our GDELT series only earn a place if they add
out-of-sample information beyond them (literature sweep: `reports/influence_literature_20261011.md` in the data
dir). GDELT's own daily event totals roughly halved between 2016 and 2026, so raw counts drift: use shares of
the day's total, z-scores against a trailing window, and deduplicated counts, never raw levels.

Known gaps: GDELT has no files for about 13% of slots in late Sep to mid Nov 2020 and for mid June to early July
2025 (404 at the source). Country ETFs are survivors (Russia's RSX is gone), so Russia enters through RUB and events.

## Splits

GDELT starts 2015-02-19, so this work uses its own splits: discovery and train 2015-02-19 to 2019-12-31,
validation 2020-2022, test 2023 onward (opened once per hypothesis family, logged).

## Discovery (train years only)

1. Daily panel: political and theme series as standardised changes (z-scores vs the prior 60 days), markets as
   abnormal returns vs SPY or vs the region.
2. Lead-lag network: for every political series x market pair, does x at t predict the market at t+1..t+5 beyond
   the market's own lags (Granger, plus transfer entropy for non-linear links); Benjamini-Hochberg across all pairs.
3. Spillovers: Diebold-Yilmaz connectedness between industries and countries, rolling, to see which sectors
   transmit shocks and whether political intensity raises connectedness.
4. Regimes: cluster days by their mix of themes and event types; compare market behaviour across clusters.

## From discovery to tests

Links that survive the correction become one pre-registered family in `research/hypotheses/`, each with a claim,
mechanism, sign and pass bars, graded once on validation and then once on test. Examples of the shape:
- tariff and trade-dispute intensity (US-CHN events, root 16 and 17) leads semis and China internet
  underperforming SPY, and CNY weakening;
- Middle East material conflict (quad 4) leads oil up and defence outperforming;
- election and policy-uncertainty intensity leads small caps underperforming large caps and VIX up.
