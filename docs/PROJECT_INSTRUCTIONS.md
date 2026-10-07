# Proposed "Workspace" section for the project instructions

## Workspace
- The desk lives in `/mnt/project-files/market-lab` (a Claude Code project: read its `CLAUDE.md` first).
- New thread: `cd /mnt/project-files/market-lab && bash bootstrap.sh` (one command: venv, install, `mlab doctor`).
  After that, `./mlab <command>`; `./mlab --help` lists everything.
- Desk agents in `.claude/agents/`: technicals-desk, options-desk, earnings-desk, news-sentiment-desk, macro-desk,
  fundamentals-desk. Fan them out in parallel, then make the call as PM.
- Playbooks in `.claude/skills/`: trade-idea, morning-brief, lens-soros, lens-buffett, lens-burry, ig-account, trade-journal, market-data.
- Trade journal: `./mlab card ... --journal`, `./mlab journal list|update|review` (`journal/JOURNAL.md`).
- IG is read-only, credentials from env vars only. Prices are cached in `data/cache/`.
- If `mlab doctor` shows hosts blocked: use connected data connectors, then WebSearch/WebFetch, and say which source you used.
