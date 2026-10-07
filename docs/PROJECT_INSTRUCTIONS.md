# Proposed "Workspace" section for the project instructions

## Workspace
- The desk code lives in the GitHub repo `Robo808/market-lab` (a Claude Code project), which every thread clones
  fresh, so its `CLAUDE.md`, desk agents and skills load automatically. Read `CLAUDE.md` first.
- New thread: from the repo clone, `bash bootstrap.sh` (one command: venv, install, `mlab doctor`). After that,
  `bash mlab <command>`; `bash mlab --help` lists everything.
- Desk state (journal, reports, price cache, IV history, transcripts) lives in the shared folder
  `/mnt/project-files/market-lab`, which `mlab` uses by default (`MLAB_DATA_DIR` overrides it). Journal entries and
  reports never need a commit or a PR; only code, agent and skill changes go through the repo.
- Desk agents in `.claude/agents/`: technicals-desk, options-desk, earnings-desk, news-sentiment-desk, macro-desk,
  fundamentals-desk. Fan them out in parallel, then make the call as PM. If they are not registered as agent types in
  your session, spawn a worker per desk with that desk's agent file as its brief.
- Playbooks in `.claude/skills/`: trade-idea, morning-brief, lens-soros, lens-buffett, lens-burry, ig-account,
  trade-journal, market-data.
- Trade journal: `bash mlab card ... --journal`, `bash mlab journal list|update|review`
  (`/mnt/project-files/market-lab/journal/JOURNAL.md`). Reports go in `/mnt/project-files/market-lab/reports/`.
- IG is read-only, credentials from env vars only. Prices are cached in the shared folder's `data/cache/`.
- If `bash mlab doctor` shows hosts blocked: use connected data connectors, then WebSearch/WebFetch, and say which
  source you used.
