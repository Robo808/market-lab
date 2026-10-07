# Paper Trading

A strategy that passed its hypothesis runs in a paper book before it can inform a real trade card: at least
**3 months and 20 trades**, inside its pre-registered kill criteria. See [[Hypothesis Testing]].

## Today's signal

```bash
bash mlab algo signal <strategy> <symbols...>
# e.g.
bash mlab algo signal tsmom ^GSPC ^FTSE GC=F
bash mlab algo signal xs_momentum SPY EFA EEM TLT GLD DBC
```

Prints the target position or weights as of the latest close, with the data source and timestamp. Nothing is placed.

## Paper books

```bash
bash mlab algo paper <strategy> <symbols...> --book <name> --capital 10000
bash mlab algo paper-status [--book <name>]
```

`algo paper`:

1. Loads the book from `$MLAB_DATA_DIR/paper/<book>.json` (creates it with `--capital` GBP on first run).
2. Computes today's target with the same function as the backtest.
3. Rebalances the book to that target at the latest close, charging the spread (`--spread-bps`, default 5 bps per unit traded).
4. Marks every position to market and logs the fills.

Run it once per bar (daily for daily strategies), after the close. Missing a day means the book rebalances late,
the same as a real account would.

`algo paper-status` shows equity, return, drawdown, holdings and recent fills for one book, or a summary table of
all books.

Books live in the data dir, shared by every thread, never in git.

## Promoting a strategy

After 3 months and 20 trades, compare paper with the backtest: hit rate, average R, Sharpe, drawdown, cost drag. If
paper is inside the backtest's range, the strategy's signals can be cited on trade cards as supporting evidence.
Cezar still sizes and places every trade himself.

## Live execution is off

- The IG client in `src/mlab/providers/ig.py` is read-only: it allows GETs and session login, switch and logout only.
  No order or deal endpoints exist in the code.
- No strategy, book or agent places orders.
- Turning execution on requires Cezar to ask for it explicitly. Until then, systematic output is a signal or a paper fill.
