---
name: ig-account
description: Read Cezar's IG account (read-only) - positions, working orders, P&L, watchlists, market search, prices, client sentiment, streaming quotes. Use for anything touching his IG book or IG epics.
---
# IG account (read-only)

Commands (`--env DEMO|LIVE` overrides `IG_ACC_TYPE`):
`./mlab ig login | accounts | positions | orders | watchlists | watchlist ID | search TERM | market EPIC | snapshot EPIC... | sentiment EPIC | prices EPIC -i 1h --start 2026-09-01 | stream EPIC... --seconds 30 | activity --days 30 | transactions --days 90`

Rules:
- Read-only by construction: the client blocks every non-GET call except session login/switch/logout. Never add deal endpoints.
  Cezar places and manages orders himself; give him exact levels and stake per point.
- Credentials: `IG_USERNAME`, `IG_PASSWORD`, `IG_ACC_TYPE`, optional `IG_ACC_NUMBER` in env vars; the API key as a
  network secret on demo-api.ig.com / api.ig.com (header `X-IG-API-KEY`) or `IG_API_KEY`.
  Never print, write or ask for them. If missing, `./mlab doctor` says which; point Cezar to the environment settings.
- Price history is allowance-limited (10k points/week): use `./mlab price ig:EPIC` (cache-first, missing tail only).
  `ig prices` prints the remaining allowance.
- Position review: for each position, `./mlab firing <underlying>` + distance to stop/target in ATR, then hold / trail / cut.
