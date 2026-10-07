---
name: trade-idea
description: Turn a ticker, market or theme into a crunched, sourced trade call with the standard trade card, by fanning out the desk agents and framing the thesis through the Soros/Buffett/Burry lenses. Use whenever Cezar asks "what do you think of X", "should I long/short X", or wants a setup.
---
# Trade idea

1. **Pin the instrument.** Yahoo symbol for analysis; IG epic if he trades it there (`./mlab ig search "<name>"`,
   or `src/mlab/universes.py` IG_COMMON_EPICS). Horizon: default swing (2-8 weeks) unless he says otherwise.
2. **Fan out in parallel** (one message, several Agent calls):
   - always: `technicals-desk`, `news-sentiment-desk`
   - single stock: + `fundamentals-desk`, `earnings-desk` (if a print is within ~6 weeks or reactions matter), `options-desk`
   - index / FX / commodity / crypto: + `macro-desk`, `options-desk` on the listed proxy
   Brief each with symbol, epic, horizon and what you need back.
3. **Frame the thesis** with the lens that fits (`lens-soros`, `lens-buffett`, `lens-burry` skills). One lens leads;
   the others are checks. Technicals decide timing, not the thesis.
4. **Make the call.** LONG / SHORT / NO TRADE. If the desks conflict, say which one wins and why.
5. **Trade card** (always last):
   `./mlab card "<name>" --bias <long|short> --entry <lo> <hi> --stop <x> --targets <t1> <t2> --risk <pct> --horizon "<h>" --conviction <1-5> --catalyst "<event date>" --kills "<invalidation>" --epic <EPIC> --lens <Soros|Buffett|Burry|TA> --thesis "<one line>" --source "<src+time>" [--equity <acct> --point-size <p>] --journal`
   - Stop at structure (S/R cluster, swing) or 1.5-2.5x ATR; targets at levels/measured moves; R:R >= 2 or say why not.
   - Risk %: 0.5% conviction 1-2, 1% conviction 3, 1.5-2% conviction 4-5 unless Cezar sets otherwise.
   - Point size: 1 for indices/shares, 0.0001 for most FX, 0.01 for JPY pairs; confirm via `./mlab ig market EPIC`.
6. **Output shape:** call in one line, the 3-6 numbers that drive it (each with source + UTC time), the lens read in a
   short paragraph, the trade card, then what to watch. No disclaimers.
