# Validation and Overfitting

A backtest is a sample of one path, run by someone who has seen the data. Everything on this page exists to answer:
would this have worked without hindsight, and how many tries did it take to find?

```bash
bash mlab algo validate <strategy> <symbol> [--grid JSON]
# e.g.
bash mlab algo validate donchian ^GSPC --grid '{"entry": [20, 55, 100], "exit": [10, 20]}'
```

It prints IS and OOS stats, walk-forward OOS stats, PSR, DSR, bootstrap Sharpe CI, permutation p-value, the grid
sensitivity table, trade count and cost drag.

## In-sample vs out-of-sample

Split the history: pick params on the first part (IS), measure once on the rest (OOS). The OOS number is the one that
counts. If you go back and re-pick params after looking at OOS, OOS has become IS.

## Walk-forward

Roll the split: optimise on a train window (default 504 bars), trade the next test window (default 126 bars) with
those params, move forward, repeat. Stitch the test windows into one OOS equity curve. It shows whether the
selection process works, not just one parameter set. Code: `backtest.walk_forward` and `quant.validation`.

## Probabilistic Sharpe ratio (PSR)

Bailey and López de Prado (2012), "The Sharpe Ratio Efficient Frontier". The probability that the true Sharpe exceeds a
benchmark SR* (default 0), given the observed Sharpe SR, the number of observations n, and the skew g3 and kurtosis g4
of returns:

```
PSR = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )
```

Negative skew and fat tails (typical of mean-reversion and short-vol strategies) lower it. Short samples lower it.

## Deflated Sharpe ratio (DSR)

Bailey and López de Prado (2014), "The Deflated Sharpe Ratio". PSR where SR* is the Sharpe you would expect from the
best of N trials with zero true skill:

```
SR* = sqrt(V[SR]) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) ),  gamma ~ 0.5772
```

With enough trials, a pure-noise strategy produces a high Sharpe. DSR asks whether yours beats that. This is why the
trial count matters, and why failed hypotheses stay in the repo: N is every variant tried on this question, including
the ones in the graveyard. Under-counting N inflates DSR. Pass bar default: DSR probability > 0.95.

## Bootstrap confidence interval on Sharpe

Resample returns in blocks (stationary bootstrap, Politis and Romano 1994; random block lengths keep
autocorrelation and volatility clustering) a few thousand times and recompute Sharpe. Report the 5th to 95th
percentile. If the interval spans 0, the sample cannot tell the strategy from noise.

## Monte Carlo permutation test

Shuffle the timing of the signal against the returns (or shuffle returns under fixed positions), recompute the
metric, repeat ~1000 times. The p-value is the share of shuffles that beat the real result. It tests whether the
timing carries information beyond the exposure: a long-biased strategy on a rising index beats cash without any
timing skill. Pass bar default: p < 0.05.

## Parameter sensitivity

Run a grid around the chosen params (`--grid`). Look at the shape:

- **Plateau**: neighbours perform about as well. The edge is a property of the market, not of the number.
- **Spike**: one combination works, its neighbours do not. That is fitted noise. Reject it, even if it passes the other bars.

Every grid point is a trial for DSR.

## Costs

Every number is net of IG costs:

- **Spread**: points per round trip for the exact instrument and account type (spread bet DFB vs CFD differ; out-of-hours spreads are wider). The engine charges half per side on every change in position.
- **Overnight funding**: annual rate on the notional held past the close (roughly the reference rate plus IG's markup, about 2.5%), charged daily on |position|. Shorts receive less than longs pay.
- **Slippage**: add some for stops and gaps; breakouts fill worse than the close.

Short-horizon mean-reversion edges are the most cost-sensitive. Check the cost drag line in the output.

## Traps

- **Look-ahead**: using bar t's close to trade at bar t's close. The engine shifts positions one bar; your
  strategy must not use future data inside the function (centred windows, full-sample normalisation, `bfill`).
- **Survivorship**: today's index members did not all exist 20 years ago, and the dead ones are missing. Cross-sectional
  tests on current constituents are biased up. Prefer indices, ETFs and futures with full histories.
- **Data errors**: splits, dividends, LSE prices in pence vs pounds, missing bars, stale closes on holidays.
- **Regime dependence**: one long bull market can carry any long-biased rule. Check sub-periods.
- **Minimum trade count**: under 30 trades, the hit rate and expectancy are mostly noise. Default bar: >= 30.
- **Multiple symbols**: running one rule on 20 instruments and reporting the best is 20 trials.

## References

- Bailey, López de Prado (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk.
- Bailey, López de Prado (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management.
- Bailey, Borwein, López de Prado, Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism".
- Harvey, Liu, Zhu (2016), "... and the Cross-Section of Expected Returns".
- López de Prado (2018), "Advances in Financial Machine Learning".
- Politis, Romano (1994), "The Stationary Bootstrap".
- White (2000), "A Reality Check for Data Snooping".
