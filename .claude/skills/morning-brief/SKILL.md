---
name: morning-brief
description: Produce the pre-market desk brief - overnight moves, macro regime, positioning, catalysts today/this week, Cezar's open IG positions and journal ideas re-checked, and the top 3 setups. Use for "morning brief", "what's happening", "game plan for today".
---
# Morning brief

Run in parallel (Agent fan-out or background shells):
- `./mlab scan -u indices -u fx -u commodities -u rates --sort 1d` and `./mlab scan -u crypto -i 1d`
- `macro-desk` agent: regime, liquidity, COT extremes, this week's calendar (dated)
- `./mlab ig positions` and `./mlab ig orders` (skip quietly if IG creds are absent), then `./mlab firing <each held instrument>`
- `./mlab journal list --status open` and `--status idea`: re-check each against today's levels (stop proximity in ATR, target hit, invalidation)
- `news-sentiment-desk` on "stock market" / the top movers

Write (<= 40 lines): regime line; overnight moves table (source + time); what changed; each open position with
status (hold / tighten stop to X / take profit at Y / thesis broken) and why; today's and this week's catalysts with
times (UK time); top 3 setups each with a trade card line. Save to `reports/brief_YYYYMMDD.md`.
