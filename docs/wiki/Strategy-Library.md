# Strategy Library

Every strategy lives in `src/mlab/quant/strategies.py` under its registry name. `bash mlab algo list` prints them
with family and default params.

Conventions:

- **Single-instrument** strategies take an OHLCV frame and return a position in [-1, 1]. The position is decided
  at the close of bar t and applied to the return of bar t+1. No look-ahead.
- **Portfolio** strategies take a panel of closes and return a weight per asset per bar, same timing rule.
- Some strategies trade both sides by default (`short=true`); set `--params '{"short": false}'` for long-only.
- Defaults are textbook values, not optimised values. If a default here and `bash mlab algo list` disagree, the code wins. Changing them is a new trial (see [[Validation and Overfitting]]).

```bash
bash mlab algo list
bash mlab algo validate <strategy> <symbol> [--grid JSON]
bash mlab algo signal <strategy> <symbols...>
```

Contents: [Trend](#trend) · [Breakout](#breakout) · [Mean reversion](#mean-reversion) · [Seasonality](#seasonality) ·
[Portfolio](#portfolio) · [Overlays](#overlays) · [How to add a strategy](#how-to-add-a-strategy)

---

## Trend

### sma_cross

- **Family**: trend
- **Rules**: long when SMA(fast) > SMA(slow), flat otherwise; with `short=true`, short when below.
- **Defaults**: `fast=50, slow=200, short=false`
- **Why it might work**: returns are autocorrelated over months because information diffuses slowly and
  investors under-react, then herd. The 200-day filter also keeps you out of most of the deep bear-market
  drawdowns, which is where most of its value on indices comes from.
- **When it fails**: choppy, range-bound markets (repeated whipsaws around the cross); V-shaped reversals, where
  it exits late and re-enters late.
- **Reference**: Faber (2007), "A Quantitative Approach to Tactical Asset Allocation"; Brock, Lakonishok, LeBaron (1992).

### ema_cross

- **Family**: trend
- **Rules**: long when EMA(fast) > EMA(slow), optional short mirror.
- **Defaults**: `fast=20, slow=100, short=false`
- **Why it might work**: same mechanism as `sma_cross`, reacts faster because EMAs weight recent bars more.
- **When it fails**: more trades, so more spread cost and more whipsaws in ranges.
- **Reference**: Covel, "Trend Following"; Hurst, Ooi, Pedersen (2017), "A Century of Evidence on Trend-Following Investing".

### supertrend

- **Family**: trend
- **Rules**: ATR trailing band; long while price is above the Supertrend line, flat (or short with `short=true`) when it flips.
- **Defaults**: `n=10, mult=3.0, short=false`
- **Why it might work**: volatility-scaled trailing stop: gives the trend room in proportion to current noise.
- **When it fails**: volatility spikes inside a trend (stopped out then re-entered higher); sideways markets.
- **Reference**: Olivier Seban; ATR from Wilder (1978), "New Concepts in Technical Trading Systems".

### tsmom

- **Family**: trend (time-series momentum)
- **Rules**: sign of the return from t-lookback to t-skip. Long if positive, flat (or short with `short=true`) if negative.
- **Defaults**: `lookback=252, skip=21, short=true`
- **Why it might work**: documented across 58 futures markets over decades. Under-reaction to news followed by
  delayed over-reaction; hedging demand from producers and risk managers. Skipping the last month avoids the
  short-term reversal effect.
- **When it fails**: sharp trend reversals (2009 rebound, 2020 March rebound); long sideways regimes.
- **Reference**: Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum", Journal of Financial Economics.

### macd_trend

- **Family**: trend
- **Rules**: long when MACD > signal line and MACD > 0; flat otherwise.
- **Defaults**: `fast=12, slow=26, signal=9, short=false`
- **Why it might work**: requires both positive momentum (MACD > 0) and accelerating momentum (above signal),
  filtering out counter-trend bounces.
- **When it fails**: the double condition is slow to enter; flat markets produce many short trades.
- **Reference**: Appel (2005), "Technical Analysis: Power Tools for Active Investors".

### adx_trend

- **Family**: trend
- **Rules**: long when +DI > -DI and ADX > threshold; with `short=true`, short when -DI > +DI and ADX > threshold; flat otherwise.
- **Defaults**: `n=14, threshold=25, short=false`
- **Why it might work**: only trades when directional movement is strong, so it sits out ranges.
- **When it fails**: ADX lags; it often confirms a trend near its end, and exits after a large giveback.
- **Reference**: Wilder (1978).

## Breakout

### donchian

- **Family**: trend / breakout
- **Rules**: Turtle system 2. Enter long on a close above the prior 55-bar high, exit on a close below the prior
  20-bar low; short mirror with `short=true`.
- **Defaults**: `entry=55, exit=20, short=true`
- **Why it might work**: new highs attract trend followers and force short covering; the asymmetric exit lets
  winners run and cuts losers.
- **When it fails**: false breakouts in ranges; low hit rate (often 30-40%) means long losing streaks that test discipline.
- **Reference**: Faith (2007), "Way of the Turtle"; Donchian (1960s).

### keltner_breakout

- **Family**: breakout
- **Rules**: long when close > upper Keltner band (EMA(n) + mult x ATR(n)); exit when close < middle line (EMA(n)).
- **Defaults**: `n=20, k=2.0, short=false`
- **Why it might work**: a close beyond 2 ATR from the mean is a volatility expansion that tends to persist for a few bars.
- **When it fails**: exhaustion spikes and news gaps that reverse the next day.
- **Reference**: Keltner (1960), "How to Make Money in Commodities"; ATR version by Linda Raschke.

### squeeze_breakout

- **Family**: breakout (volatility compression)
- **Rules**: squeeze = Bollinger bands (20, 2) inside Keltner channels (20, 1.5 ATR). After a squeeze ends, take the
  direction of the first close outside the Bollinger bands; exit when close crosses the 20 SMA.
- **Defaults**: `n=20, bb_k=2.0, kc_k=1.5, short=true`
- **Why it might work**: volatility clusters and mean-reverts; low-vol periods are followed by expansions, and the
  first break carries direction.
- **When it fails**: fake-outs where the first break reverses; squeezes that resolve with a gap through the stop.
- **Reference**: Carter (2005), "Mastering the Trade" (TTM Squeeze); Bollinger (2001), "Bollinger on Bollinger Bands".

### nr7_breakout

- **Family**: breakout
- **Rules**: an NR7 bar is the bar with the narrowest high-low range of the last 7. On the following bars, go long on a
  close above the NR7 high or short (with `short=true`) on a close below its low; hold `hold` bars then exit.
- **Defaults**: `lookback=7, hold=5, short=true`
- **Why it might work**: range contraction precedes range expansion; same volatility-clustering logic as the squeeze, on a shorter clock.
- **When it fails**: holiday or low-liquidity bars that are narrow for no reason; one-bar head fakes.
- **Reference**: Crabel (1990), "Day Trading with Short Term Price Patterns and Opening Range Breakout".

## Mean reversion

### rsi_reversion

- **Family**: mean-reversion
- **Rules**: long when RSI(14) < 30 and close > SMA(200); exit when RSI > 50.
- **Defaults**: `low=30, n=14, exit_level=50, trend_filter=200`
- **Why it might work**: in an uptrend, short-term selloffs are liquidity events (forced selling, stop runs) that
  get bought. The trend filter avoids catching falling knives.
- **When it fails**: the first leg of a bear market, while price is still above the 200 SMA; low trade count on slow instruments.
- **Reference**: Wilder (1978); Connors and Alvarez (2008), "Short Term Trading Strategies That Work".

### rsi2

- **Family**: mean-reversion
- **Rules**: long when RSI(2) < 10 and close > SMA(200); exit when close > SMA(5).
- **Defaults**: `entry=10, trend=200, exit_sma=5`
- **Why it might work**: equity indices show strong short-horizon reversal after 1-3 down days, especially in
  uptrends. Holding periods are short, so funding cost is small.
- **When it fails**: crash regimes and gap-down sequences; the edge per trade is small, so IG spread matters a lot.
  Has decayed on US large caps since publication.
- **Reference**: Connors and Alvarez (2008).

### bollinger_reversion

- **Family**: mean-reversion
- **Rules**: long when close < lower Bollinger band (20, 2); exit at the middle band. With `short=true`, short above the upper band, exit at the middle.
- **Defaults**: `n=20, k=2.0, short=false`
- **Why it might work**: overextension from a moving average reverts in range-bound instruments (FX crosses, some indices).
- **When it fails**: trending markets, where price walks the band. Pair with `regime_filter` or test only on ranging instruments.
- **Reference**: Bollinger (2001).

### zscore_reversion

- **Family**: mean-reversion
- **Rules**: z = (close - mean(20)) / std(20). Enter against the move at |z| > 2 (long when z < -2, short when z > 2 if
  `short=true`); exit when |z| < 0.5.
- **Defaults**: `n=20, entry=2.0, exit=0.5, short=true`
- **Why it might work**: same as Bollinger reversion, with an explicit exit band that avoids round-tripping at the mean.
- **When it fails**: regime shifts where the mean moves; fat tails mean |z| > 3 events keep going.
- **Reference**: Chan (2013), "Algorithmic Trading: Winning Strategies and Their Rationale".

### ibs_reversion

- **Family**: mean-reversion
- **Rules**: IBS = (close - low) / (high - low). Long when IBS < 0.2 (closed near the low); exit when IBS > 0.8.
- **Defaults**: `low=0.2, high=0.8`
- **Why it might work**: closes near the day's low on indices and index ETFs tend to be followed by up days
  (end-of-day liquidity pressure that reverses overnight).
- **When it fails**: individual stocks with news; regimes of persistent selling. Needs daily OHLC that matches the
  instrument's session (IG out-of-hours prices change the high/low).
- **Reference**: Pagonidis (2013), "The IBS Effect: Mean Reversion in Equity ETFs".

## Seasonality

### turn_of_month

- **Family**: seasonality
- **Rules**: long from the close of the second-to-last trading day of the month (so the position is held over the
  last trading day) through the close of the 3rd trading day of the next month; flat otherwise.
- **Defaults**: `days_before=1, days_after=3`
- **Why it might work**: month-end and month-start flows: salaries into pension and retirement plans, fund
  rebalancing, window dressing. Exposure is about 20% of days, so funding drag is low.
- **When it fails**: when the flows are front-run and the effect is arbitraged; crash months.
- **Reference**: Ariel (1987), "A Monthly Effect in Stock Returns"; Lakonishok and Smidt (1988); McConnell and Xu (2008).

## Portfolio

### xs_momentum

- **Family**: cross-sectional
- **Rules**: each month, rank assets by return from t-12 months to t-1 month. Long the top N equal weight; with
  `short=true`, short the bottom N equal weight. Rebalance monthly.
- **Defaults**: `lookback=252, skip=21, top=3, short_bottom=0`
- **Why it might work**: winners keep winning for 3-12 months across equities, countries, sectors and asset classes:
  under-reaction and herding.
- **When it fails**: momentum crashes at sharp market rebounds (2009), when the short leg (beaten-down junk) rallies hardest.
- **Reference**: Jegadeesh and Titman (1993), "Returns to Buying Winners and Selling Losers"; Daniel and Moskowitz (2016), "Momentum Crashes".

### dual_momentum

- **Family**: allocation
- **Rules**: each month, pick the risky asset with the best 12-month return (relative momentum). If its 12-month
  return beats the safe asset's (or 0 when no safe asset is given), hold it 100%; otherwise hold the safe asset (or cash).
- **Defaults**: `safe=null, lookback=252`
- **Why it might work**: combines cross-sectional and time-series momentum; the absolute filter cuts exposure in bear markets.
- **When it fails**: whipsaw years where the leader changes every month; one-asset concentration.
- **Reference**: Antonacci (2014), "Dual Momentum Investing".

### inverse_vol

- **Family**: allocation
- **Rules**: weight_i proportional to 1 / sigma_i, sigma = 60-day realised vol, normalised to sum to 1. Rebalance monthly.
- **Defaults**: `n=60`
- **Why it might work**: equalises each asset's standalone risk so one volatile asset does not dominate; low-vol
  assets have historically delivered better risk-adjusted returns.
- **When it fails**: ignores correlation; correlations go to 1 in crises; leans into bonds just before a rate shock.
- **Reference**: Maillard, Roncalli, Teiletche (2010); Frazzini and Pedersen (2014), "Betting Against Beta".

### risk_parity

- **Family**: allocation
- **Rules**: equal risk contribution using the full covariance matrix (trailing window). Each asset's w_i x (Sigma w)_i
  is equal. Solved numerically. Rebalance monthly.
- **Defaults**: `n=126, iters=200`
- **Why it might work**: diversifies risk, not capital; no return forecasts needed, so less estimation error.
- **When it fails**: joint stock-bond selloffs (2022); unlevered it carries a large bond weight.
- **Reference**: Maillard, Roncalli, Teiletche (2010), "The Properties of Equally Weighted Risk Contribution Portfolios"; Qian (2005).

### pairs

- **Family**: pairs / stat-arb
- **Rules**: two symbols A, B. On a rolling window, regress log(A) on log(B) by OLS to get the hedge ratio beta.
  Spread = log(A) - beta x log(B); z = rolling z-score of the spread. Enter when |z| > 2 (short the spread when
  z > 2, long when z < -2), exit when |z| < 0.5. The Engle-Granger cointegration p-value over the sample is reported.
- **Defaults**: `n=60, entry=2.0, exit=0.5`
- **Why it might work**: two assets tied by the same fundamentals (share classes, same-sector peers, index vs
  future) drift apart on liquidity shocks and come back.
- **When it fails**: the relationship breaks (takeover, rerating, index change): the spread trends and never comes
  back. A low cointegration p-value in-sample is no guarantee out of sample. Two legs means two spreads on IG.
- **Reference**: Gatev, Goetzmann, Rouwenhorst (2006), "Pairs Trading"; Vidyamurthy (2004); Engle and Granger (1987).

---

## Overlays

Overlays wrap any strategy's position or weights.

### vol_target

- **Rules**: scale the position by target_vol / realised_vol (annualised, trailing window), capped at `max_leverage`.
- **Defaults**: `target=0.15, n=20, cap=2.0, periods=252`
- **Why**: volatility is persistent, returns are not. Scaling down in high vol cuts the tails and usually improves
  Sharpe for trend strategies. See [[Risk and Sizing]].
- **Reference**: Moreira and Muir (2017), "Volatility-Managed Portfolios"; Harvey et al. (2018), "The Impact of Volatility Targeting".

### regime_filter

- **Rules**: allow longs only when close > SMA(n), shorts only when close < SMA(n); otherwise set the position to 0.
- **Defaults**: `n=200`
- **Why**: most mean-reversion edges on indices only exist inside an uptrend; most short edges only in downtrends.

`bash mlab algo list` shows how each overlay is passed on the command line.

---

## How to add a strategy

1. **Open a `[strategy]` issue** with the Strategy template: family, exact rules, params, references, expected regime.
2. **Branch** `strat/<name>`.
3. **Write a function** in `src/mlab/quant/strategies.py`: single-instrument `f(df, **params) -> pd.Series` of
   positions in [-1, 1], or portfolio `f(closes, **params) -> pd.DataFrame` of weights. Use only data up to and
   including bar t for the position at t; the engine shifts it.
4. **Register it** in `REGISTRY` with its family and defaults.
5. **Add a test** in `tests/` on synthetic data (offline): correct shape, range, no look-ahead (appending future
   bars must not change past positions), and one hand-checked case.
6. **PR** to main with the template. A strategy in the library is not evidence of an edge.
7. **Run it through a hypothesis** ([[Hypothesis Testing]]) before it can inform any trade card.
