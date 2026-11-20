# Forward testing

Backtests shortlist. The forward record decides. `mlab fwd` runs the whole strategy library forward on paper,
in parallel, across many instruments and market contexts, and logs every signal, so the verdict on a rule comes
from bars it had never seen. There is no execution path.

## What runs

The registry (`research/forward/registry.json`) is generated from the matrix in `src/mlab/forward/specs.py`:

| Unit | Count | What it is |
|---|---:|---|
| Single-instrument cells | 1,100 | 20 rule variants (trend, breakout, mean reversion, seasonality, with and without vol targeting or a regime filter) x 55 instruments: indices, US and UK ETFs, US and UK shares, FX, commodity futures, crypto |
| Basket books | 15 | cross-sectional momentum (mega caps, sectors, cross-asset, crypto, commodities), dual momentum, inverse vol, risk parity, six pairs |
| Option books | 110 | 11 rule-based structures x 10 underlyings: covered call, cash-secured put, bull put and bear call spreads, iron condor, short strangle (always, and only when IV/RV >= 1.2), long straddle (always, and only when IV/RV <= 0.9), put-spread hedge, collar |
| Null controls | 133 | `always_long` and a seeded `coin_flip` on every instrument, `always_long` on every basket and option underlying |

## Pre-registration

Each cell's spec (strategy, parameters, overlays, instruments, exits) is hashed with SHA-256 and stamped with
its registration time before its first signal. The registry file goes through a PR, so the PR is the
pre-registration record; each run also stores a write-once copy of the registry under its own hash in the data
dir, and every row carries the registry hash, the spec hash and the code commit.

- `mlab fwd register` adds new matrix rows and refuses any existing id whose spec changed: edit the matrix by
  adding a `.v2` cell, never by changing a frozen one.
- The run refuses a cell whose spec hash differs from the one frozen at its first signal.
- Each new cell records a trailing three-year backtest as its `prior`, so backtest-to-forward decay can be measured.

## Pass bars that scale with breadth

Testing more rules makes a lucky winner more likely, so the bar rises with the count. A candidate passes only if,
net of costs and funding, after at least 126 forward bars:

| Bar | Threshold |
|---|---|
| Deflated Sharpe | >= 0.95, with trials = every registered candidate |
| False discovery rate | Benjamini-Hochberg q <= 0.10 across all candidates (one-sided t on mean daily return) |
| Not just beta | PSR of the daily excess over the same instrument's `always_long` control >= 0.90 |
| Enough trades | >= 8 entries (allocation books >= 6) |

The primary unit is the **book**: each matrix row run equal-weight across all its instruments (so "trend rule X
across 55 markets" is one test), each option structure across its underlyings, each basket as is. Per-instrument
cells are graded as well, against the full cell count. Before 126 bars every verdict is `PENDING`.

`mlab fwd gate` applies the bars to trailing backtests of all 133 null controls and refuses the bar set if any
null passes (LESSONS L-012). At registration: 0 of 133 passed; no null cleared the deflated-Sharpe or FDR bar.

## Honest fills and costs

- Decide at the close of bar t, fill at the open of bar t+1 (processed by the next run), mark at the close.
- Only complete bars are used: a run never reads today's partial bar. A missed day is caught up bar by bar, and a
  test checks that day-by-day runs and one catch-up run give the same equity.
- Costs per venue and direction (`src/mlab/forward/costs.py`), each with its source; unverified numbers are
  labelled `assumption`. Shorts on cash-share venues route to the spread-bet venue. Spread bets pay daily funding.
- Options cross the bid/ask on entry and exit, pay commission to open, and are marked at the mid. They run only
  during US hours (09:45-16:00 New York), when quotes are two-sided.

## Every trigger, taken or skipped

Each cell keeps two books on one signal stream. The **raw** book takes every target. The **desk** book takes an
exposure-increasing trigger only if the desk filters pass: index regime gate, all-gates-fail rule, cost under
0.2R (1R = 2 x ATR(14)), no scheduled earnings within two sessions. Every trigger is logged with each filter's
verdict, so `mlab fwd score --filters` grades the filters themselves (desk minus raw).

## Context

Every row carries market tags (S&P 500 vs its 200-day average, VIX bucket and term structure, index gates, 10-year
yield direction, dollar trend) and instrument tags (trend vs 50/200-day averages, realised-vol percentile, RSI
bucket, ATR). `mlab fwd score --by vix_bucket` pools each rule's daily returns by context.

## Options and Greeks

Per underlying and run: the chain for the expiries the books need, IV solved per quote, and a volatility row
(ATM IV30, 25-delta risk reversal and butterfly, term slope, 20-day realised vol, IV/RV). ATM IV30 is also
appended to `data/iv_history/`. A near-the-money chain snapshot (moneyness 0.75-1.25, up to 100 days) is saved as
parquet. Each open book logs per-leg and total value, delta, gamma, theta per day and vega per vol point, and the
day's P&L split into delta, gamma, theta, vega and residual from the previous day's Greeks.

## Storage

Under `$MLAB_DATA_DIR/forward` (override `MLAB_FORWARD_DIR`), never modified after writing, each write read back
and hash-checked:

| Path | Schema | Content |
|---|---|---|
| `events/signals/<YYYY-MM>/<run_id>.jsonl.gz` | `fwd.signals/1` | one row per cell per bar: target, held weights, equity and return for both books, cost, funding, context |
| `events/triggers/...` | `fwd.triggers/1` | every entry, exit or flip with filter verdicts and whether each book took it |
| `events/fills/...` | `fwd.fills/1` | fills with price source, venue, cost; option fills with slippage and commission |
| `events/greeks/...` | `fwd.greeks/1` | option legs and totals with Greeks and attribution |
| `events/vol/...` | `fwd.vol/1` | per-underlying volatility snapshot |
| `events/runs/...` | `fwd.runs/1` | per run: code commit, registry hash, per-item status, data sources |
| `chains/<YYYY-MM>/<run_id>.parquet` | | near-the-money option chain |
| `state/<run_id>.json.gz` | | book state after the run (the newest is live) |
| `registry/<sha>.json` | `fwd.registry/1` | frozen registry copies |

A run holds a lock (`RUNNING`), so two runs cannot fork the state. Expected size: well under 100 MB a year.

## Desk-state history

`mlab ledger snapshot | verify | log PATH | show SHA` keeps a content-addressed, hash-chained history of desk files
that live outside git (journal, lessons, assumption register, paper log). Every forward run takes a snapshot.

## Commands

```bash
bash mlab fwd register          # new matrix rows into the registry (commit it)
bash mlab fwd gate              # bars must refuse every null control
bash mlab fwd run               # daily, after the US close
bash mlab fwd status            # runs, store size
bash mlab fwd score --books     # primary verdicts
bash mlab fwd score             # per-instrument cells
bash mlab fwd score --by vix_bucket ; bash mlab fwd score --filters
bash mlab fwd audit             # registry hashes, freeze drift, schemas, ledger chain
```
