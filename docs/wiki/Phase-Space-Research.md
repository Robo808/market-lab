# Phase Space Research

`mlab phase` treats every instrument at every bar as a point in indicator space, finds the regimes ("weather
systems") in that space, and asks whether any forecast built on them survives strict validation. It runs only
on frozen datasets ([[Datasets]]), on the train and validation splits. The test split is graded once, through
[[Hypothesis Testing]].

## The pieces

| Idea | Component | Where |
|---|---|---|
| Phase space | 23 scale-free, point-in-time features per (symbol, bar): returns 1/5/21/63, RSI, MACD in ATRs, ATR %, realised vol and its 5/63 ratio, volume z, distance from SMA 20/50/200 in ATRs, Bollinger %b, ADX, efficiency ratio, Donchian position, skew, lag-1 autocorrelation, permutation entropy, and three higher-timeframe features from completed bars only | `phase/features.py` |
| Weather systems | Gaussian HMM (full covariance, Baum-Welch over all symbols at once). Signals use forward-filtered probabilities only; Viterbi and smoothed probabilities see the future and stay inside the fit | `phase/regimes.py` |
| Light cones | Truncation gate: the cube is recomputed on cut histories and every earlier value must be unchanged. A test plants three leaks (next close, centred mean, full-sample z-score) and the gate must catch all three. Walk-forward folds are re-run on data that ends at the fold, and the predictions must match. `--delay` adds execution latency | `tests/test_phase.py` |
| Honest evaluator | Anchored walk-forward with purge (label horizon) and embargo; scaler, HMM and regression refitted per fold; costs = half the quoted spread plus slippage (floored at asset-class levels), funding on gross; Newey-West t; deflated Sharpe; CSCV probability of backtest overfitting | `phase/evaluate.py` |
| Counting trials | Every configuration is written to `$MLAB_DATA_DIR/research/phase/experiments/`. The deflated Sharpe uses that count, failed configs included. Re-running an identical config does not add a trial | `evaluate.log_experiment` |
| Null stubs | Always long, plus five seeded coin flips matched to the candidate's exposure and holding period, run through the same costs and bars. If a stub clears the bars, the bars are refused | `evaluate.null_stubs` |
| Chaos limits | Out-of-sample rank IC by horizon (1 to 63 bars) with 90% block-bootstrap intervals | `evaluate.skill_by_horizon` |
| Vibration mechanics | Per regime: correlation of past and next q-bar returns (inertia above 0, restoring force below 0), and an OU half-life for the deviation from the 63-bar mean, measured against shuffled-return paths. A deviation from a moving average mean-reverts mechanically, so the null matters | `physics.forces_by_regime` |
| Entropy | Permutation entropy (Bandt & Pompe 2002), with out-of-sample IC split by entropy tercile. The tercile edges are set on pre-test data | `physics.entropy_skill` |
| Fourier | Fisher's g test on raw log prices and on returns. Raw prices are red noise and show spurious long cycles | `physics.spectral_check` |
| Simulation | Stationary block bootstrap of the out-of-sample returns (Sharpe, drawdown, worst year), plus synthetic markets rebuilt from resampled bars and run through the full walk-forward | `physics.stress`, `physics.null_world` |
| Reflexivity | Out-of-sample Sharpe by year, with a Theil-Sen slope for decay | `evaluate.evaluate` |

## Pass bars

A configuration passes only if it clears all of these:

- Newey-West t of 3 or more on the mean daily return after costs
- Deflated Sharpe probability of 0.95 or more, over all logged trials
- PBO below 0.5
- A Sharpe above the always-long stub

## Known limits

- The Yahoo stock universes keep only today's survivors. Prefer the index, FX and commodity sets.
- Price-only index CFDs leave out dividends while funding is charged, so long books are penalised. Each run
  reports the Sharpe before costs and before funding next to the net Sharpe.
- With 14 to 24 instruments, the pooled IC is noisy. Read it with its bootstrap interval.

## References

López de Prado (2018) *Advances in Financial Machine Learning*, ch. 7 and 11; Bailey & López de Prado (2014) "The
Deflated Sharpe Ratio"; Bailey, Borwein, López de Prado & Zhu (2017) "The Probability of Backtest Overfitting";
Hamilton (1989); Ang & Timmermann (2012) "Regime Changes and Financial Markets"; Moskowitz, Ooi & Pedersen (2012);
Jegadeesh & Titman (1993); Lo, Mamaysky & Wang (2000); Lo & MacKinlay (1988); Bandt & Pompe (2002); Politis &
Romano (1994).
