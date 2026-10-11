# H-20261011-gpr-country-reversal: EM countries whose geopolitical risk jumped outperform the next month

Status: REGISTERED | Verdict: pending
Branch: `claude/project-thread-zph1h2` (PR #56) | Registered: 2026-10-11 | Spec-SHA256: 0de5f3ef0a1bb9f7a1071a85b7a84c6125204383052b53ab1babf0ed2cbfcc4d

## Claim
Among 11 emerging-market country ETFs, going long the 3 whose country geopolitical risk index (Caldara-Iacoviello
GPRC) rose most last month and short the 3 where it rose least earns at least 0.5% a month net of a 20 bp round trip,
with Newey-West t >= 3 over 2010-2026, and stays positive after the result was published (2022 on).

## Mechanism and prior evidence
Investors overreact to vivid geopolitical news (availability bias), so prices of the countries hit drop too far and
recover the next month. Zaremba, Cakici, Demir and Long (Journal of Financial Stability 2022,
https://ideas.repec.org/a/eee/finsta/v58y2022ics1572308921001236.html) find up to ~1% a month across EMs: a single
study. Consistent with the short-lived equity drop after GPR shocks in Caldara-Iacoviello. Published effects shrink
26-58% after publication (McLean-Pontiff), hence a 0.5% bar and a separate post-publication check.

## Known weaknesses (fixed before the test)
- The GPR history is the current vintage; the publisher revises past values, so this is not fully point-in-time.
  Revisions are small relative to month-on-month swings, but they bias towards finding an effect.
- US-listed ETFs in USD, so currency is inside the return (as an IG trader would get on the ETF or index CFD).
- 11 countries and 3 per side is a thin cross-section; Saudi Arabia only from late 2015, India from 2012.
- The original paper's sample overlaps ours before about 2020; the post-2022 leg is the clean out-of-sample check.
- First trial of the influence-map family (#58). Graded once.

## Kill criteria
Any pass bar missed = FAIL. No re-runs with other parameters.

## Spec (frozen at registration)
```json
{
  "costs": {
    "round_trip_bp": 20
  },
  "dataset": "influence-daily@v20261011",
  "engine": "mlab.textlab.studies.run_gpr_country_reversal",
  "etf": {
    "BRA": "EWZ",
    "CHN": "FXI",
    "IDN": "EIDO",
    "IND": "INDA",
    "KOR": "EWY",
    "MEX": "EWW",
    "POL": "EPOL",
    "SAU": "KSA",
    "TUR": "TUR",
    "TWN": "EWT",
    "ZAF": "EZA"
  },
  "gpr_dataset": "policy-monthly@v20261011",
  "kind": "cross_sectional",
  "min_countries": 8,
  "n_side": 3,
  "pass": {
    "full": {
      "mean_net_pct_min": 0.5,
      "t_nw_min": 3.0
    },
    "post_publication": {
      "mean_net_pct_sign": "positive"
    }
  },
  "post_publication_from": "2022-01-01",
  "rebalance_day": 7,
  "signal": "log change in country GPR (GPRC_xxx) over calendar month m",
  "study": "gpr_country_reversal",
  "timing": "rebalance at the close of the first trading day on or after day 7 of month m+1 (GPR for m is published in the first days of m+1); hold one month; adjusted closes in USD",
  "trials": 1,
  "window": [
    "2010-01-01",
    "2026-09-30"
  ]
}
```

## Results
Not run yet.
