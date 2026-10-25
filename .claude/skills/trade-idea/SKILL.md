---
name: trade-idea
description: Turn a ticker, market or theme into a crunched, sourced trade call with the standard trade card, by fanning out the desk agents and framing the thesis through the Soros/Buffett/Burry lenses. Use whenever Cezar asks "what do you think of X", "should I long/short X", or wants a setup.
---
# Trade idea

Read `$MLAB_DATA_DIR/LESSONS.md` first: it carries the desk's standing corrections.

1. **Pin the instrument.** Yahoo symbol for analysis; IG epic if he trades it there (`./mlab ig search "<name>"`,
   or `src/mlab/universes.py` IG_COMMON_EPICS). Horizon: default swing (2-8 weeks) unless he says otherwise.
2. **Fan out in parallel** (one message, several Agent calls):
   - always: `technicals-desk`, `news-sentiment-desk`
   - single stock: + `fundamentals-desk`, `earnings-desk` (if a print is within ~6 weeks or reactions matter), `options-desk`
   - index / FX / commodity / crypto: + `macro-desk`, `options-desk` on the listed proxy
   Brief each with symbol, epic, horizon and what you need back. Brief `technicals-desk` for the
   multi-timeframe composite (1wk/1d/4h/1h/1m: Heikin-Ashi, RSI regime, relative volume, open/close
   structure), never a daily-only scan.
3. **Frame the thesis** with the lens that fits (`lens-soros`, `lens-buffett`, `lens-burry` skills). One lens leads;
   the others are checks. Technicals decide timing, not the thesis.
4. **Make the call.** LONG / SHORT / NO TRADE. If the desks conflict, say which one wins and why. "Single
   signals show no edge" is not a call: give the composite or trend-following setup with its tested numbers.
5. **Trade card** (always last):
   `./mlab card "<name>" --bias <long|short> --entry <lo> <hi> --stop <x> --targets <t1> <t2> --risk <pct> --horizon "<h>" --conviction <1-5> --catalyst "<event date>" --kills "<invalidation>" --epic <EPIC> --lens <Soros|Buffett|Burry|TA> --thesis "<one line>" --source "<src+time>" [--equity <acct> --point-size <p>] --journal`
   - Stop at structure (S/R cluster, swing) or 1.5-2.5x ATR; targets at levels/measured moves; R:R >= 2 or say why not.
   - Risk %: 0.5% conviction 1-2, 0.75% conviction 3, 1% conviction 4-5 (the cap) unless the owner sets otherwise.
   - Point size: 1 for indices/shares, 0.0001 for most FX, 0.01 for JPY pairs; confirm via `./mlab ig market EPIC`.
   - Costs in R next to R:R (spread, FX conversion, funding, commission), and R:R net of costs. Above 0.2R, use a
     cheaper expression of the same view or pass.
   - Scheduled binary events (earnings, central banks, budgets): no fresh entry in the two sessions before unless the
     event is the trade; then size to the gap. Cancel or halve resting entries ahead of the event.
   - Entries: keep a confirmation trigger on counter-trend and breakout ideas. On conviction 3+ quality longs in a
     multi-timeframe uptrend, split the entry: half about 0.5 ATR below last, half at the deep limit.
   - Trend-pullback longs need the index regime on side: NDX for US tech, SOX for chips, BTC for alts, above its
     20-day low and 50-day average; otherwise wait or halve. Oversold-quality cards at support are exempt.
   - Size on risk, capped at 1% of the account; on instruments that gap, budget 1.3R of loss. Move the stop only after
     T1 fills, and to breakeven at the earliest.
   - Tag the setup on every card (oversold-quality, breakout, trend-pullback, event, macro FX). A card is paper until
     the owner places it; read the journal back after every `--journal` write (`./mlab journal list`).
6. **Output shape:** call in one line, the timeframes and inputs the read used, the 3-6 numbers that drive it (each with source + UTC time), the lens read in a
   short paragraph, the trade card, then what to watch. No disclaimers.
