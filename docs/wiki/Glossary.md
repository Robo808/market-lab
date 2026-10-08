# Glossary

**ADX**: Average Directional Index (Wilder). Trend strength 0 to 100, direction-agnostic. +DI and -DI give direction.

**ATR**: Average True Range. Average of the true range (max of high-low, |high-prev close|, |low-prev close|) over n bars. Unit of noise for stops and sizing.

**Block bootstrap**: resampling a return series in contiguous blocks to keep autocorrelation, used for confidence intervals on Sharpe.

**CAGR**: compound annual growth rate. (end / start)^(1 / years) - 1.

**Calmar ratio**: CAGR / |max drawdown|.

**CFD**: contract for difference. Sized in contracts; profits subject to capital gains tax in the UK.

**Cointegration**: two non-stationary series with a stationary linear combination. The basis of `pairs`. Tested with Engle-Granger.

**COT**: CFTC Commitments of Traders. Weekly futures positioning by commercials, managed money and others.

**Cost drag**: total spread plus funding as a share of returns.

**DFB**: "daily funded bet", IG's rolling spread bet with no expiry; funding charged nightly. Epics usually end in `.DAILY.IP` or contain `DFB`.

**Drawdown / max drawdown**: fall from the running equity peak; the largest such fall over the sample.

**DSR**: deflated Sharpe ratio (Bailey and López de Prado 2014). PSR against the Sharpe expected from the best of N skill-less trials. Needs the trial count.

**Epic**: IG's instrument identifier, e.g. `IX.D.FTSE.DAILY.IP`.

**Expectancy**: average P&L per trade, usually in R. hit rate x avg win R - (1 - hit rate) x avg loss R.

**Funding (overnight)**: IG's daily financing charge on positions held past the close, about the reference rate plus a markup, on notional.

**GEX**: gamma exposure. Dealer net gamma by strike; positive GEX dampens moves, negative amplifies them. The gamma flip is the price where it changes sign.

**Heat**: total GBP at risk if every open stop is hit, as % of equity.

**Hit rate**: share of trades that make money.

**IBS**: internal bar strength, (close - low) / (high - low).

**IS / OOS**: in-sample (used to choose) vs out-of-sample (used only to measure).

**IV / RV**: implied volatility (from option prices) vs realised volatility (from past returns). IV > RV is the usual variance risk premium.

**Kelly fraction**: bet size that maximises long-run log growth: p - (1 - p) / b. Use a fraction of it.

**Look-ahead bias**: using information at time t that was not available until later.

**Max pain**: the expiry price where option holders lose the most in aggregate.

**NR7**: the narrowest high-low range of the last 7 bars.

**Permutation test**: shuffling signal timing against returns to build a null distribution of a metric.

**Point**: IG's unit of price movement for a market (1 index point, 1p on UK shares, 0.0001 on EURUSD).

**PSR**: probabilistic Sharpe ratio (Bailey and López de Prado 2012). Probability the true Sharpe exceeds a benchmark, adjusting for sample length, skew and kurtosis.

**R-multiple**: trade P&L divided by the initial risk (entry to stop). A 2R win made twice what was risked.

**Risk parity**: weights with equal risk contribution from each asset.

**Sharpe ratio**: annualised mean excess return / annualised volatility. Daily: mean / std x sqrt(252).

**Skew (options)**: difference in IV across strikes; put skew = downside puts priced richer than calls.

**Skew (returns)**: asymmetry of the return distribution; negative skew = rare large losses.

**Sortino ratio**: like Sharpe but divides by downside deviation only.

**Spread bet**: IG product sized in GBP per point; UK gains currently free of capital gains tax and stamp duty.

**Squeeze**: Bollinger bands inside Keltner channels; low-volatility compression.

**Stake per point**: GBP gained or lost per point of movement on a spread bet.

**Survivorship bias**: testing only on instruments that still exist today.

**Trial count (N)**: every strategy variant, parameter set and symbol tried on a question. Input to DSR.

**TSMOM**: time-series momentum: trade an asset in the direction of its own past return.

**Vol targeting**: scaling exposure to hold a constant target volatility.

**Walk-forward**: rolling optimise-then-test windows, stitched into one out-of-sample record.

**XS momentum**: cross-sectional momentum: long recent winners vs losers within a universe.
