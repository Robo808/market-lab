# H-20261014-hawkes-branching-vol: A Bayesian Hawkes branching ratio on extreme moves forecasts next-period realised volatility beyond a HAR baseline

Status: GRADED | Verdict: FAIL
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
Run 2026-10-14 with `python -m mlab.quant.hawkes research/hypotheses/H-20261014-hawkes-branching-vol.md` on
crypto-1h@v20261010 (Binance, 2017-08-17..2023-06-30) and duka-1d@v20261010 (Dukascopy, 2003-01-02..2019-12-31).

**Crypto leg (BTCUSDT, ETHUSDT): FAIL**

| Bar | Required | Got | |
|---|---|---|---|
| train t(n_z) | >= 2.5 | 0.590 | miss |
| validation coef(n_z) | > 0 | 0.011 | pass |
| validation OOS MSE gain | >= 0.01 | 0.002 | miss |

**Index leg (US500, USTEC, UK100): FAIL**

| Bar | Required | Got | |
|---|---|---|---|
| train t(n_z) | >= 2.5 | -1.442 | miss |
| validation coef(n_z) | > 0 | -0.106 | miss |
| validation OOS MSE gain | >= 0.01 | -0.020 | miss |

Verdict **FAIL** on both legs. Neither test split is opened: a failed hypothesis has nothing to confirm, and keeping
them locked preserves them for the next one.

Detail (descriptive only, not a new claim). On indices the sign is the opposite of the claim: in validation the
high-n tercile had lower forward vol and shallower drawdowns.

```json
{
  "crypto": {
    "rows": {
      "train": 368,
      "validation": 206
    },
    "n_train_mean": 0.5110896249510379,
    "n_train_sd": 0.17385416219101218,
    "mean_events_per_window": 45.22554347826087,
    "mean_accept": 0.5197663043478261,
    "train": {
      "coef_n_z": 0.016692178015119057,
      "t_n_z": 0.5903384674141435,
      "r2": 0.42158545483827337,
      "nobs": 368,
      "by_n_tercile": {
        "low": {
          "n": 0.298,
          "fwd_rv_mean": 0.09391,
          "fwd_dd_mean": -0.07999,
          "fwd_dd_p10": -0.17671
        },
        "mid": {
          "n": 0.552,
          "fwd_rv_mean": 0.13205,
          "fwd_dd_mean": -0.11053,
          "fwd_dd_p10": -0.28607
        },
        "high": {
          "n": 0.684,
          "fwd_rv_mean": 0.12563,
          "fwd_dd_mean": -0.08186,
          "fwd_dd_p10": -0.20294
        }
      }
    },
    "validation": {
      "coef_n_z": 0.011105005628718318,
      "t_n_z": 0.36621733846777516,
      "r2": 0.43573491156905375,
      "nobs": 206,
      "by_n_tercile": {
        "low": {
          "n": 0.249,
          "fwd_rv_mean": 0.08236,
          "fwd_dd_mean": -0.07102,
          "fwd_dd_p10": -0.15661
        },
        "mid": {
          "n": 0.482,
          "fwd_rv_mean": 0.08624,
          "fwd_dd_mean": -0.0734,
          "fwd_dd_p10": -0.1736
        },
        "high": {
          "n": 0.658,
          "fwd_rv_mean": 0.09866,
          "fwd_dd_mean": -0.07273,
          "fwd_dd_p10": -0.15376
        }
      },
      "oos_mse_gain": 0.0017347014195364663
    }
  },
  "indices": {
    "rows": {
      "train": 304,
      "validation": 205
    },
    "n_train_mean": 0.3886329618898741,
    "n_train_sd": 0.12048336590781542,
    "mean_events_per_window": 65.20723684210526,
    "mean_accept": 0.36239473684210527,
    "train": {
      "coef_n_z": -0.043201301253362176,
      "t_n_z": -1.4422611666143752,
      "r2": 0.5340955225155997,
      "nobs": 304,
      "by_n_tercile": {
        "low": {
          "n": 0.25,
          "fwd_rv_mean": 0.0546,
          "fwd_dd_mean": -0.03767,
          "fwd_dd_p10": -0.09122
        },
        "mid": {
          "n": 0.412,
          "fwd_rv_mean": 0.04901,
          "fwd_dd_mean": -0.03213,
          "fwd_dd_p10": -0.07772
        },
        "high": {
          "n": 0.506,
          "fwd_rv_mean": 0.05185,
          "fwd_dd_mean": -0.03115,
          "fwd_dd_p10": -0.07505
        }
      }
    },
    "validation": {
      "coef_n_z": -0.10611866562092545,
      "t_n_z": -1.8604996819015571,
      "r2": 0.2755057908143608,
      "nobs": 205,
      "by_n_tercile": {
        "low": {
          "n": 0.238,
          "fwd_rv_mean": 0.04264,
          "fwd_dd_mean": -0.03527,
          "fwd_dd_p10": -0.07965
        },
        "mid": {
          "n": 0.35,
          "fwd_rv_mean": 0.03447,
          "fwd_dd_mean": -0.02187,
          "fwd_dd_p10": -0.05024
        },
        "high": {
          "n": 0.431,
          "fwd_rv_mean": 0.03241,
          "fwd_dd_mean": -0.01749,
          "fwd_dd_p10": -0.03759
        }
      },
      "oos_mse_gain": -0.0202865214331438
    }
  }
}
```
