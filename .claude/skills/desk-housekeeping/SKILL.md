---
name: desk-housekeeping
description: Weekly sweep of the market-lab repo and project threads (stale branches, open PRs, issues, idle threads) plus the lessons-learned roll-up into LESSONS.md, project memory, team memory, skills and agents. Use for "clean up", "too many branches/threads", "what did we learn", or when the weekly routine fires.
---
# Desk housekeeping

Two jobs, one report. Nothing is deleted or closed without Cezar's go; recommendations are cheap, deletions are not.

## 1. Sprawl sweep
1. **Branches:** `git fetch origin` then, per remote branch, ahead/behind `origin/main` and last commit date.
   Ahead 0 = fully merged: recommend delete. Ahead > 0 with no open PR: find its thread, recommend PR or delete.
2. **PRs** (GitHub MCP `list_pull_requests`, open): age, draft, which thread owns it, and a test merge into main
   (`git merge --no-commit` on a detached main) plus pairwise merges between open PRs touching the same files.
   Recommend merge order; overlapping PRs fold into the older owner's branch.
3. **Issues** (`list_issues` OPEN): duplicates, issues done by a merged PR but still open, owner-action items
   waiting on Cezar, code work with no issue.
4. **Threads** (`list_thread_sessions`): completed and idle > 48 h with nothing owed = resolve; several threads on
   one workstream = consolidate into the owner thread; blocked = say what it waits on.
5. **Shared folder:** stray code copies, temp files, reports with no index entry.

## 2. Lessons roll-up
1. Read `$MLAB_DATA_DIR/LESSONS.md`, the project MEMORY.md, the team memory dir, and the last week of the
   project timeline and threads (`fetch_project_timeline`, `fetch_thread`): look for Cezar's corrections,
   frustrations, retractions, and findings that changed how the desk works.
2. Add each new lesson to LESSONS.md as `L-NNN title (date, open|folded)` with Learned / Rule / Folded into.
3. Fold it where it belongs: repo agents/skills via a PR (issue first, `Part of #N`), team memory files,
   project memory (via the coordinator), project instructions (send the proposed wording to the coordinator),
   Cezar's account skill "trade-call" via `propose_skills`. Mark it folded only when every place carries it.

## Output
`$MLAB_DATA_DIR/reports/housekeeping_<yyyymmdd>.md` with one table per category (item, state, recommendation,
owner), the lessons added, and the short list of actions that need Cezar's yes. Reply in the thread with that
list only; send the coordinator the threads to resolve.
