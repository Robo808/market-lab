# H-20261011-stale-spike-reversal: Tone surprises on stale-news attention spikes reverse

Status: REGISTERED | Verdict: pending
Branch: `claude/project-thread-zph1h2` (with the engine, PR #56) | Registered: 2026-10-11 | Spec-SHA256: 90aab34cea1954b3ced03effe65f2334330fbf9493aa3ab1b8010b63d3596a7a

## Claim
On the 25 US mega caps, on attention-spike days led by stale stories (H2's exact definitions: GDELT company-name
matches, URL-slug headlines, novelty vs the prior 5 days, spike > 2 sd over the prior 60 days with 20+ articles,
fresh share at or below its prior 250-day median), the CAR from the next open to the close of day +5, signed by
the day's tone surprise, is negative: at most -20 bp net of a 10 bp round trip, with t <= -2.0, and the +10 day
CAR is negative too. Trading against the tone of a stale news flood earns.

## Mechanism
Tetlock (2011, RFS): investors treat recycled news as new and overreact; prices partly reverse within a week.
A stale flood (syndicated rewrites) inflates attention without information, so the tone-driven move fades.

## Where it came from (why only one window)
Found in H2's descriptive output after grading H2 on 2015-2022 (stale-led signed CAR +5: -21.9 bp, t -1.78 in
2015-19; -44.1 bp, t -2.49 in 2020-22; +10: -79.9 bp, t -3.31 in 2020-22). Those years are spent, so the only
honest test is data H2 never touched: events from 2023-01-01 to 2026-09-30, graded once. Features are unchanged
from H2 on purpose (org-only matching, not the headline-named filter of #57), so this tests the finding as found.

## Known weaknesses (fixed before the test)
- Inherits H2's noisy company matching (#57); same bias towards zero.
- News for 2023-01 to 2025-09 is being backfilled now; prices for the window were never looked at for these
  events. The dataset's held-out 2020+ range was already read by H2 for 2020-2022 (logged).
- One look: a miss is a FAIL, no reruns with other parameters.
- Trial 3 of the news-before-price family (#55); trials 1 and 2 failed.

## Kill criteria
Any pass bar missed = FAIL. The study refuses to run without --test and is run once.

## Spec (frozen at registration)
```json
{
  "beta_window": 250,
  "costs": {
    "round_trip_bp": 10
  },
  "dataset": "us-stocks-daily@v20261010",
  "derived_from": "H-20261011-fresh-vs-stale-attention (descriptive finding on 2015-2022)",
  "engine": "mlab.textlab.studies.run_stale_reversal",
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
  "news_from": "2022-01-01",
  "novelty": {
    "def": "1 - max cosine (hashed unigrams+bigrams) vs the firm's earlier slugs in the prior 5 days",
    "fresh": 0.75,
    "min_words": 3,
    "window": "5D"
  },
  "pass": {
    "test": {
      "stale_car10_mean_bp_sign": "negative",
      "stale_car5_mean_bp_max": -20,
      "stale_car5_t_max": -2.0
    }
  },
  "primary": "stale-led car5",
  "sigma_window": 60,
  "splits": {
    "test": [
      "2023-01-01",
      "2026-09-30"
    ]
  },
  "study": "stale_reversal",
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
  "trials": 3
}
```

## Results
Not run yet.
