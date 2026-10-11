# H-20261011-fresh-vs-stale-attention: Tone surprises on fresh-news attention spikes drift; on stale-news spikes they don't

Status: REGISTERED | Verdict: pending
Branch: `claude/project-thread-zph1h2` (with the engine, PR #56) | Registered: 2026-10-11 | Spec-SHA256: 0da7a8896fffe4f6a2fe5e220c0ee103f05b64a7cc11abcd7cf85021aee76e24

## Claim
On the 25 US mega caps, on days when a firm's GDELT article count spikes (> 2 sd above its prior 60-day mean,
at least 20 articles), the CAR from the next open to the close of day +5, signed by the day's tone surprise, is
higher when the spike is led by fresh stories (novel headlines) than when it is led by stale ones (rewrites of
the last 5 days' stories), by at least 30 bp net of a 10 bp round trip.

## Mechanism
Tetlock (2011, "All the news that's fit to reprint", RFS): investors overreact to stale news, so returns after
stale-news days partly reverse, while genuinely new information is absorbed slowly and drifts. Ke, Kelly & Xiu
(2019) find the return-predictive part of news text is concentrated in fresh articles. Mega caps price news
fast, so the bar is a 5-day CAR from the next open, after the day-t close has absorbed the first reaction.

## Known weaknesses (fixed before the test)
- GKG page titles only exist from late September 2019, so the headline text is the URL slug for every year
  (85-93% of company articles carry a 3+ word slug in 2015-2022, checked before registration). Slugs are
  noisier than titles; that biases the fresh/stale split towards no difference, not away from it.
- Company matching is by GKG organisation names; some articles mention the firm in passing. Same bias.
- GDELT has no file for about 13% of its 15-minute slots in May to Nov 2020, mostly late Sep to mid Nov 2020
  (confirmed 404 at the source). Spikes are relative to the firm's own trailing baseline, which absorbs it.
- 10 bp round trip is an assumed IG US share CFD cost, not reconciled from statements.
- GDELT 2.0 starts 2015-02-19, so the default dataset splits (train to 2014) cannot be used. This test has
  its own splits; prices from the dataset's held-out 2020+ range are read for the validation years, logged in
  the dataset access log, and events after 2022-12-31 are dropped until the test is opened.
- Before registration I looked only at feature coverage (title and slug coverage, fresh share for AMZN Q1 2026),
  never at returns on these events.
- Trial 2 of the news-before-price family (#55); trial 1 failed.

## Kill criteria
Any pass bar missed = FAIL. No re-running with new parameters on this file: a new test is a new hypothesis.
Train is graded, then validation, once each; the test split (2023+) is opened once, logged, only after both pass.

## Spec (frozen at registration)
```json
{
  "beta_window": 250,
  "costs": {
    "round_trip_bp": 10
  },
  "dataset": "us-stocks-daily@v20261010",
  "engine": "mlab.textlab.studies.run_novelty_drift",
  "event": {
    "baseline_days": 60,
    "direction": "sign of day-t mean tone minus its prior 60-day mean",
    "flag": "fresh-led: fresh share on t above its prior 250-day median (else stale-led)",
    "fresh_share_days": 250,
    "min_articles": 20,
    "spike": "n_t > mean + 2 sd of n over the prior 60 trading days, and n_t >= 20",
    "spike_sd": 2.0
  },
  "horizons": [
    1,
    5,
    10
  ],
  "kind": "event_study",
  "market": "SPY",
  "news": "GDELT GKG 2.0 raw-file store (mlab newsstore), company-name matches (via org:) only, deduplicated by (symbol, url)",
  "novelty": {
    "def": "1 - max cosine (hashed unigrams+bigrams) vs the firm's earlier slugs in the prior 5 days",
    "fresh": 0.75,
    "min_words": 3,
    "window": "5D"
  },
  "pass": {
    "test": {
      "diff_bp_sign": "positive",
      "t_min": 1.5
    },
    "train": {
      "diff_bp_min": 30,
      "perm_p_max": 0.01,
      "t_min": 2.5
    },
    "validation": {
      "diff_bp_sign": "positive",
      "t_min": 1.5
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
      "2023-01-01",
      "end"
    ],
    "train": [
      "2015-02-19",
      "2019-12-31"
    ],
    "validation": [
      "2020-01-01",
      "2022-12-31"
    ]
  },
  "study": "novelty_drift",
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
  "text": "URL slug headline (features.slug), at least min_words words",
  "timing": "articles available before 20:00 UTC on trading day t count for t; entry open t+1, exit close t+h",
  "trials": 2
}
```

## Results
Not run yet.
