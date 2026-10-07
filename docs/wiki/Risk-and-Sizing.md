# Risk and Sizing

Edge decides whether to trade. Sizing decides whether you are still trading next year. Code: `src/mlab/risk.py`
(discretionary cards) and `src/mlab/quant/sizing.py` (systematic books). All amounts in GBP.

## Fixed-fractional risk per trade

Risk a fixed % of equity between entry and stop. Default range 0.5% to 1% per trade; conviction 5 does not mean 5%.

```
risk amount  = equity x risk%
stop points  = |entry - stop| / point size
stake/point  = risk amount / stop points
```

```bash
bash mlab size --equity 20000 --risk 1 --entry 7410 --stop 7330
# risk 200.00 GBP, stop 80 points, stake 2.50 GBP per point
```

## Stake per point on IG

A spread bet is a stake in GBP per point of movement. What a "point" is depends on the market:

| Market | 1 point = | Example |
|---|---|---|
| Indices (FTSE, DAX, US 500) | 1 index point | FTSE 7400 to 7401 |
| UK shares | 1p | 412.0p to 413.0p |
| EURUSD, GBPUSD | 0.0001 (1 pip) | 1.0850 to 1.0851 |
| USDJPY | 0.01 | 151.20 to 151.21 |
| Gold | 0.1 or 1, check the market details | |

Pass `point_size` accordingly. Check the minimum stake and the IG margin factor with `bash mlab ig search`. For CFDs
the same maths gives contracts once you divide by the contract's value per point. Gaps go through stops: size so that a
2x stop-distance gap is survivable, or use a guaranteed stop and pay its premium.

## ATR stops

Stop distance = mult x ATR(14), default mult 2. It scales the stop with current noise, so the stake shrinks when the
market is wild and grows when it is quiet, keeping the GBP risk constant. `risk.atr_stop(entry, atr, direction, mult)`.

## Volatility targeting

Scale exposure so the position runs at a target annualised volatility:

```
scale = target vol / realised vol (trailing 20 days, annualised)    capped at max leverage
```

Default target 15%, cap 2x. Applied as the `vol_target` overlay in the [[Strategy Library]]. Volatility is persistent,
so this cuts exposure going into turbulent periods and usually improves Sharpe and drawdown for trend strategies.

## Inverse-vol and risk parity

For a book of several instruments:

- **Inverse-vol**: w_i proportional to 1 / sigma_i. Each asset has the same standalone vol contribution. Ignores correlation.
- **Risk parity (equal risk contribution)**: solve for w so that w_i x (Sigma w)_i is equal for every asset. Accounts for correlation.

Both are strategies (`inverse_vol`, `risk_parity`) and sizing functions in `quant/sizing.py`.

## Fractional Kelly

Kelly fraction f* = p - (1 - p) / b, with p = win rate and b = average win / average loss. Full Kelly maximises long-run
growth but assumes the inputs are exact; with estimated p and b it over-bets and produces brutal drawdowns. Use
quarter Kelly (the default in `risk.kelly_fraction`) as a ceiling, never as a target, and only with 30+ trades of
history behind p and b.

## Correlation and portfolio heat

- **Heat** = total GBP at risk if every open stop is hit, as % of equity. Cap it at about 5% to 6%.
- Correlated positions are one position. Long FTSE, long DAX and long US 500 is one equity-beta bet with three
  stops. Use `bash mlab corr` on the instruments; above 0.7 correlation, count them as one for heat.
- Same for FX: long EURUSD and short USDCHF are both short USD.

## Drawdown rules

| Drawdown from equity peak | Action |
|---|---|
| 10% | Halve risk per trade |
| 15% | Quarter risk per trade; only conviction 4 and 5 setups |
| 20% | Stop new trades; review the journal (`bash mlab journal review`) before restarting |

For a systematic book in paper: a drawdown deeper than 1.5x the backtest max drawdown means the backtest was wrong
or the regime changed. Kill it per its pre-registered criteria.
