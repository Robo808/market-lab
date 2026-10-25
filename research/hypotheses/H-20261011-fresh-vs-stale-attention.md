# H-20261011-fresh-vs-stale-attention: Tone surprises on fresh-news attention spikes drift; on stale-news spikes they don't

Status: GRADED | Verdict: FAIL
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
Run 2026-10-11 with `python -m mlab.textlab.studies research/hypotheses/H-20261011-fresh-vs-stale-attention.md` on
us-stocks-daily@v20261010 (Yahoo) and the GDELT store (every day 2015-02-19 to 2022-12-31; 12,985,620
company-name articles with a 3+ word slug). 2216 attention-spike events, 2015-2022. A first attempt ran
out of memory while loading articles (before any event or return was computed); it left one extra line in the
dataset access log.

| Bar (train 2015-19, CAR +1..+5, net 10 bp) | Required | Got | |
|---|---|---|---|
| fresh-led minus stale-led | >= 30 bp | -18.3 bp | miss |
| t (date-averaged, Welch) | >= 2.5 | -0.46 | miss |
| permutation p | <= 0.01 | 0.5727 | miss |

Verdict **FAIL** on the train bars. Validation is reported but cannot rescue it (diff -101.1 bp,
t -0.94). The test split (2023+) stays unopened.

What the run showed (descriptive only, not a new claim):
- Fresh-led spikes are rare: 72 of 1348 train events. Spikes are
  mostly floods of near-duplicate syndicated stories, so the flag splits off a thin tail.
- Company matching by GKG organisation name is noisy: NFLX (3.46M), MSFT (2.44M) and WMT (1.70M) collect far more
  articles than AMZN (0.31M), while META (15k) and BAC (2k) collect almost none (META's "facebook" matches are
  classed as platform mentions). Fixing entity matching comes before any further GDELT test.
- Both groups drift against the tone surprise: stale-led spikes -21.9 bp (t -1.78) in train and
  -44.1 bp (t -2.49) in validation at +5; at +10, -79.9 bp (t -3.31) in validation. That is
  the reversal Tetlock (2011) predicts for stale news. It was found after looking at 2015-2022, so it can only
  be tested as a new hypothesis on data not yet seen (2023+).

```json
{
  "validation": {
    "n_flag": 27,
    "n_noflag": 837,
    "dates_flag": 21,
    "dates_noflag": 437,
    "mean_flag_bp": -145.1,
    "mean_noflag_bp": -44.1,
    "diff_bp": -101.1,
    "t": -0.94,
    "t_flag": -1.37,
    "t_noflag": -2.49,
    "perm_p": 0.1694
  },
  "by_horizon_train": {
    "car1": {
      "n_flag": 72,
      "n_noflag": 1276,
      "dates_flag": 55,
      "dates_noflag": 674,
      "mean_flag_bp": -22.5,
      "mean_noflag_bp": -16.2,
      "diff_bp": -6.3,
      "t": -0.4,
      "t_flag": -1.56,
      "t_noflag": -2.87,
      "perm_p": 0.4428
    },
    "car5": {
      "n_flag": 72,
      "n_noflag": 1276,
      "dates_flag": 55,
      "dates_noflag": 674,
      "mean_flag_bp": -40.2,
      "mean_noflag_bp": -21.9,
      "diff_bp": -18.3,
      "t": -0.46,
      "t_flag": -1.06,
      "t_noflag": -1.78,
      "perm_p": 0.5622
    },
    "car10": {
      "n_flag": 72,
      "n_noflag": 1276,
      "dates_flag": 55,
      "dates_noflag": 674,
      "mean_flag_bp": -89.3,
      "mean_noflag_bp": -42.9,
      "diff_bp": -46.4,
      "t": -0.76,
      "t_flag": -1.52,
      "t_noflag": -2.73,
      "perm_p": 0.393
    }
  },
  "by_horizon_validation": {
    "car1": {
      "n_flag": 27,
      "n_noflag": 837,
      "dates_flag": 21,
      "dates_noflag": 437,
      "mean_flag_bp": -2.4,
      "mean_noflag_bp": -20.1,
      "diff_bp": 17.8,
      "t": 0.66,
      "t_flag": -0.09,
      "t_noflag": -2.57,
      "perm_p": 0.4975
    },
    "car5": {
      "n_flag": 27,
      "n_noflag": 837,
      "dates_flag": 21,
      "dates_noflag": 437,
      "mean_flag_bp": -145.1,
      "mean_noflag_bp": -44.1,
      "diff_bp": -101.1,
      "t": -0.94,
      "t_flag": -1.37,
      "t_noflag": -2.49,
      "perm_p": 0.1891
    },
    "car10": {
      "n_flag": 27,
      "n_noflag": 837,
      "dates_flag": 21,
      "dates_noflag": 437,
      "mean_flag_bp": -53.6,
      "mean_noflag_bp": -79.9,
      "diff_bp": 26.3,
      "t": 0.21,
      "t_flag": -0.43,
      "t_noflag": -3.31,
      "perm_p": 0.8806
    }
  }
}
```
