# H-20261014-hawkes-branching-vol: A Bayesian Hawkes branching ratio on extreme moves forecasts next-period realised volatility beyond a HAR baseline

Status: PRE-REGISTERED | Verdict: PENDING
Branch: `claude/project-thread-zsn18q` | Registered: 2026-10-14 | Spec-SHA256: 8fbe72006f23c18b76e629b8151028717548fc405a6c38b052f1fa25c5de655c

## Claim
On BTC and ETH hourly bars and on US500, USTEC and UK100 daily bars, the posterior-mean branching ratio n of an
exponential-kernel Hawkes process, fitted by MCMC to the times of extreme moves in a trailing window, carries
information about next-period realised volatility that last-period and last-4-period realised volatility do not:
its coefficient is positive in train (t >= 2.5) and validation, and adding it cuts validation forecast MSE by
at least 1% against the same baseline fitted on train.

## Mechanism
Seismology's ETAS aftershock model (Ogata) is a marked Hawkes process: each event raises the rate of the next.
Filimonov & Sornette (2012, arXiv:1201.3572) read the branching ratio n as the endogenous share of activity
(reflexivity) and argue that n near 1 marks a market close to criticality, where moves feed on themselves
(liquidation cascades, stop runs, forced de-risking). If that is real, a high n should precede more volatility than
the current volatility level alone implies, because the self-excited part of activity has not yet played out.
Hardiman, Bercot & Bouchaud (arXiv:1302.1405) dispute that n carries a trend, which is why this is a test and not
an assumption. Events are normalised by trailing sigma, so n measures clustering beyond the current vol level.

## Known weaknesses (fixed before the test)
- About 43 events per 90-day crypto window and about 60 per 1000-day index window (counted on train before
  registration, no outcomes looked at), so each posterior is wide; a noisy n biases the test towards FAIL.
- The decay time is capped (48 h crypto, 10 days indices) because with long decays n and the baseline rate are
  not separately identified (seen on synthetic Poisson data while writing the estimator).
- Index data are daily: duka-1h never finished building, so intraday index clustering is not tested here.
- Crypto validation and test are short (2 and 3 years); one regime can dominate.

## Kill criteria
Any pass bar missed = FAIL for that leg. No re-running with new parameters on this file: a new test is a new
hypothesis. Train and validation are run once together; a leg's test split is opened once, logged, only if that
leg passed.

## Spec (frozen at registration)
```json
{
  "kind": "hawkes_vol_forecast",
  "engine": "mlab.quant.hawkes",
  "legs": {
    "crypto": {
      "dataset": "crypto-1h@v20261010",
      "symbols": [
        "BTCUSDT",
        "ETHUSDT"
      ],
      "sigma_window": 720,
      "k_sigma": 3.0,
      "fit_window": 2160,
      "horizon": 168,
      "every": 168,
      "decay_bounds": [
        1,
        48
      ],
      "mcmc": {
        "iters": 4000,
        "burn": 1000,
        "step": 0.3
      },
      "hac_lags": 4
    },
    "indices": {
      "dataset": "duka-1d@v20261010",
      "symbols": [
        "US500",
        "USTEC",
        "UK100"
      ],
      "sigma_window": 60,
      "k_sigma": 2.0,
      "fit_window": 1000,
      "horizon": 20,
      "every": 20,
      "decay_bounds": [
        1,
        10
      ],
      "mcmc": {
        "iters": 4000,
        "burn": 1000,
        "step": 0.3
      },
      "hac_lags": 3
    }
  },
  "target": "log realised vol over the next `horizon` bars (sqrt of summed squared log returns)",
  "baseline": "pooled OLS: symbol intercepts + log RV of the last horizon + log RV of the last 4 horizons (per-horizon scale)",
  "model": "baseline + z-scored posterior-mean branching ratio n (z uses train mean and sd)",
  "errors": "Driscoll-Kraay (statsmodels hac-groupsum) by evaluation date",
  "pass": {
    "train": {
      "t_min": 2.5
    },
    "validation": {
      "coef_sign": "positive",
      "oos_mse_gain_min": 0.01
    },
    "test": {
      "coef_sign": "positive",
      "oos_mse_gain_min": 0.0
    }
  },
  "grading": "each leg graded on its own; a leg that passes train+validation opens its test split once",
  "splits": "dataset manifests (crypto-1h: train 2017-08..2021-06, validation 2021-07..2023-06; duka-1d: train 2003..2014, validation 2015..2019); a row counts in a split only if its forward window ends inside it",
  "trials": 1
}
```

## Results
_Filled after `python -m mlab.quant.hawkes research/hypotheses/H-20261014-hawkes-branching-vol.md`._
