---
name: trade-journal
description: Log, update and review trades in the workspace journal (journal/trades.jsonl -> JOURNAL.md) and score which lens, desk and conviction level actually make money. Use when Cezar takes, skips, adjusts or closes a trade, or asks how his trading is going.
---
# Trade journal

- New idea: `./mlab card ... --journal` (status idea) or `--status open` when he's in.
- Filled: `./mlab journal update ID --status open --fill <price> --size <£/pt> --note "<why>"`
- Closed: `./mlab journal update ID --status closed --exit <price> --note "<what happened>"` (R multiple computed)
- Skipped: `./mlab journal update ID --status passed --note "<why>"`
- Review: `./mlab journal review` -> win rate, expectancy in R, by lens, by conviction. Use it to say bluntly what works
  and what to stop doing. Also reconcile with `./mlab ig activity --days 30` so the journal matches the real book.
The journal lives in the shared project folder: `/mnt/project-files/market-lab/journal/`. Never hand-edit the JSONL.
