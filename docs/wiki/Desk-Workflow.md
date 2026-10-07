# Desk Workflow

The main Claude session is the **PM**. For anything beyond a quick quote it fans out to the six desk agents in
parallel, weighs their output through the three lenses, and writes the call and the trade card.

## The desks

| Desk | Crunches | Main commands |
|---|---|---|
| `technicals-desk` | Full TA catalogue, candlesticks, chart patterns, levels, regime, backtested hit rate of every signal on that instrument | `tafull`, `firing`, `signals`, `patterns`, `ta`, `scan`, `backtest` |
| `options-desk` | IV term structure, skew, expected move, max pain, GEX and gamma flip, unusual activity, IV vs RV, structure payoffs | `options`, `payoff` |
| `earnings-desk` | Earnings history vs price reactions, transcript and press release tone, guidance | `earnings`, `earnings-text`, `earnings-align` |
| `news-sentiment-desk` | News, GDELT tone, 8-Ks, StockTwits, Reddit, HN, sentiment index, attention spikes | `news`, `sentiment`, `social` |
| `macro-desk` | Rates, curve, inflation, liquidity (Fed balance sheet minus TGA minus RRP), credit, USD, cross-asset trend, COT | `macro --liquidity`, `lens soros`, `cot`, `cot-extremes`, `rs`, `corr` |
| `fundamentals-desk` | 10 years of SEC financials, filings, insiders, Buffett and Burry checklists, owner-earnings DCF | `dd`, `lens buffett`, `lens burry` |

If the desks are not registered as agent types in a session, spawn one worker per desk with that desk's agent
file as its brief.

## The lenses

- **Soros**: macro and reflexivity. Where is the boom or bust, what is the market getting wrong, how does price feed back into fundamentals.
- **Buffett**: quality, owner earnings, margin of safety.
- **Burry**: contrarian, balance-sheet forensics, hated assets with asymmetric payoff.

Technicals set the timing, not the thesis. Systematic strategies from the [[Strategy Library]] that have passed a
hypothesis and survived [[Paper Trading]] can be cited as supporting evidence on a card.

## The trade card

Every idea ends with a card:

| Field | Content |
|---|---|
| Bias and instrument | Long, short or no trade; IG epic when known |
| Entry zone | Price range |
| Stop / invalidation | Level where the idea is wrong |
| Targets | One or more levels |
| Reward:risk | Per target, computed |
| Risk per trade | % of account |
| Horizon | Days, weeks, months |
| Conviction | 1 to 5 |
| Catalysts and dates | Earnings, data, meetings |
| What kills it | One line, the strongest counter-case |

```bash
bash mlab card "FTSE 100" --bias long --entry 7400 7420 --stop 7330 --targets 7600 7750 \
  --epic IX.D.FTSE.DAILY.IP --journal
```

`card` computes reward:risk and stake per point. `--journal` writes it to `$MLAB_DATA_DIR/journal/trades.jsonl`
and `JOURNAL.md`.

## Journal

```bash
bash mlab journal list
bash mlab journal update <id> ...
bash mlab journal review     # hit rate, average R, expectancy by desk, conviction and horizon
```

Review the journal monthly. Conviction that does not predict R is a calibration problem to fix.
