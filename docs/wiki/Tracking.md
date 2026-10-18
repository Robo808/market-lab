# Tracking

Work is tracked in GitHub issues, grouped into epics and milestones, and shown on a project board.

## The tree

```
[Epic] Roadmap (#16)
├── [Epic] IG live (read-only)       milestone: IG live (read-only)
├── [Epic] Honest backtests          milestone: Honest backtests
├── [Epic] IG streaming beyond quotes milestone: Streaming
├── [Epic] Data platform v1          milestone: Data platform v1
└── [Epic] Repo and code hygiene     milestone: Repo and code hygiene
    └── tasks, hypotheses, bugs and data-source issues as sub-issues
```

Every issue sits under one epic as a sub-issue and carries that epic's milestone, so progress shows on the issue, on the
epic and on the milestone page.

## Labels

| Kind | Labels | Use |
|---|---|---|
| Type | `epic`, `task`, `bug`, `strategy`, `hypothesis`, `data-source`, `research`, `infra`, `tech-debt` | What the issue is |
| Area | `area:ig`, `area:quant`, `area:data`, `area:repo`, `desk:*` | Where the code lives |
| Priority | `priority:high`, `priority:medium`, `priority:low` | Order inside a milestone |
| State | `blocked`, `owner-action`, `status:*`, `verdict:*` | Why it is not moving, or where a hypothesis stands |

`owner-action` issues are assigned to the repo owner: they are clicks in settings or decisions that no automation can make.
The full list with colours is `.github/labels.md`; the Labels workflow syncs it on push to main.

## Branches and PRs

- One PR per task. Branches are named by whoever creates them (`claude/...` for agent threads, `hypo/<yyyymmdd>-<slug>` for hypotheses).
- The PR body starts with `Closes #N` (finishes the task) or `Part of #N` (advances it). That links the branch and PR in the
  issue's Development panel, and merging closes the issue and moves it to Done on the board.
- An issue with no PR yet but work in flight gets a comment naming the branch.

## The board

The board is a user-level GitHub Project linked to this repo. To create it once:

1. Profile > **Projects** > **New project** > **Board** (under "Start from scratch"), name it `market-lab`, **Create project**.
2. In the project: menu (three dots) > **Workflows** > **Auto-add to project** > **Edit** > repository `market-lab`,
   filter `is:issue,pr is:open` > **Save and turn on workflow**. GitHub Free allows one auto-add workflow.
3. The built-in workflows that set closed issues and merged PRs to **Done** are on by default.
4. Repo > **Projects** tab > **Link a project** > pick `market-lab`.

Auto-add only picks up items created or updated after it is turned on. To pull in the issues that already exist, update
each one once (a label toggle is enough) or add them from the board with **Add item**.

Suggested board columns: Todo, In progress, In review, Done. Group or filter by milestone, label `owner-action` or
`priority:high` for focused views.

## Source

GitHub Docs, checked 2026-10-08: [adding items automatically](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/adding-items-automatically),
[built-in automations](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-built-in-automations),
[creating a project](https://docs.github.com/en/issues/planning-and-tracking-with-projects/creating-projects/creating-a-project),
[linking a project to a repository](https://docs.github.com/en/issues/planning-and-tracking-with-projects/managing-your-project/adding-your-project-to-a-repository).
