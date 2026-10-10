# H-20261010-news-vs-nonews-extremes: After an extreme abnormal day, moves with identified firm news continue and moves without news revert

Status: PRE-REGISTERED | Verdict: PENDING
Branch: `claude/project-thread-zph1h2` (with the engine, PR #54) | Registered: 2026-10-10 | Spec-SHA256: 6458221c095ef6490cb6cd0d1536939b4f63024a41620a985bba249b08b4b12f

## Claim
On the 25 US mega caps, after a day with |abnormal return| > 2.5 sigma (vs SPY, beta and sigma from prior bars),
the signed CAR from the next open to the close of day +5 is higher when the day carried identified firm news
(an 8-K or an earnings release) than when it did not, by at least 50 bp net of a 10 bp round trip.

## Mechanism
Boudoukh, Feldman, Kogan & Richardson (NBER w18725): identified-news days are information shocks that are
under-reacted to and continue; no-news days are liquidity or noise shocks that revert. Chan (2003) finds the same
split with headlines. It should survive in mega caps only weakly: Ke, Kelly & Xiu find large-cap news is priced in
about a day, which is why the bar is a 5-day CAR from the next open.

## Known weaknesses (fixed before the test)
- 8-Ks and earnings miss most news (analyst calls, product, macro), so the no-news group is contaminated with news.
  This biases the difference towards zero, not away from it. GDELT (#52) gives a cleaner flag in a later test.
- EDGAR JSON acceptance stamps are unreliable, so timing uses filingDate with a one-day spill: a filing dated t
  flags days t and t+1. Entry is the open after the extreme day, by which time every filing dated t is public.
- 10 bp round trip is an assumed IG US share CFD cost, not reconciled from statements.

## Kill criteria
Any pass bar missed = FAIL. No re-running with new parameters on this file: a new test is a new hypothesis.
Train and validation are evaluated together once; the test split is opened once, logged, after that verdict.

## Spec (frozen at registration)
```json
{
  "beta_window": 250,
  "costs": {
    "round_trip_bp": 10
  },
  "dataset": "us-stocks-daily@v20261010",
  "earnings": "us-earnings@v20261010",
  "engine": "mlab.textlab.events",
  "entry": "open t+1",
  "exit": "close t+h",
  "horizons": [
    1,
    5,
    10
  ],
  "k_sigma": 2.5,
  "kind": "event_study",
  "market": "SPY",
  "news_flag": "8-K or 8-K/A filingDate (EDGAR) or earnings date (Yahoo) on trading day t or t-1 (filing date mapped to the next trading day on or after it; spill 1)",
  "pass": {
    "test": {
      "diff_bp_sign": "positive",
      "t_min": 2.0
    },
    "train_plus_validation": {
      "diff_bp_min": 50,
      "perm_p_max": 0.01,
      "t_min": 3.0
    },
    "validation": {
      "diff_bp_sign": "positive"
    }
  },
  "permutation": {
    "n": 2000,
    "seed": 7,
    "shuffle": "flag within symbol-year"
  },
  "primary": "car5",
  "sigma_window": 60,
  "splits": {
    "test": [
      "2020-01-01",
      "end"
    ],
    "train": [
      "start",
      "2014-12-31"
    ],
    "validation": [
      "2015-01-01",
      "2019-12-31"
    ]
  },
  "symbols": [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "BRK-B",
    "JPM",
    "LLY",
    "V",
    "MA",
    "XOM",
    "UNH",
    "COST",
    "NFLX",
    "AMD",
    "ORCL",
    "WMT",
    "PLTR",
    "CRM",
    "ADBE",
    "INTC",
    "BAC"
  ],
  "trials": 1
}
```

## Results
_Filled after the run. Do not edit the spec._
