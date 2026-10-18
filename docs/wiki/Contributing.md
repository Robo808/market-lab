# Contributing

## Issues

Use a template (New issue in GitHub):

| Template | Title prefix | Labels | For |
|---|---|---|---|
| Hypothesis | `[H] ` | `hypothesis`, `status:pre-registered` | A falsifiable claim with pass bars set before testing |
| Strategy | `[S] ` | `strategy` | A new strategy or overlay for the library |
| Bug | `[Bug] ` | `bug` | Something broken; include `bash mlab doctor` output |
| Data source | `[Data] ` | `data-source` | A new or broken data source |

Blank issues are allowed for anything else; label them `infra` or `research`. The full label set with colours is in
`.github/labels.md`. Add a `desk:*` label when an issue belongs to one desk.

Hypothesis status moves `status:pre-registered` -> `status:testing` -> `status:paper` (on PASS), and the issue gets
`verdict:pass` or `verdict:fail`.

## Branches

| Prefix | For |
|---|---|
| `hypo/<yyyymmdd>-<slug>` | hypotheses |
| `strat/<name>` | new strategies |
| `data/<source>` | data sources |
| `feat/<x>` | features |
| `fix/<x>` | bug fixes |

Branch from main, PR back to main. Delete the branch after merge.

## Pull requests

The template asks for **Before / After / How** and a checklist:

- `ruff check src tests` and `pytest -q` pass.
- Hypothesis PRs: the pre-registration commit is linked and predates the results, the verdict is recorded, the trial count is updated.
- No credentials, no journal data.

## CI

`.github/workflows/ci.yml` runs on every push to main and every PR: Python 3.12, `pip install -e ".[dev]"`, `ruff check src tests`,
`pytest -q`, with `MLAB_DATA_DIR` pointed at a temp folder.

## Design standards

Read [[Design Standards]] before writing code: layering, functional core, data contracts, caching tiers, storage, concurrency
rules for the shared folder, streaming, logging, and the review checklist.

`.github/workflows/wiki-sync.yml` mirrors `docs/wiki/` to the GitHub wiki on push to main. Edit wiki pages in the repo.

## Tests

- Offline only. No network calls in tests: use synthetic series or recorded fixtures under `tests/`.
- Every strategy gets a no-look-ahead test: appending bars must not change earlier positions.
- Run locally: `~/.venvs/market-lab/bin/pytest -q`.

## Never commit

- Credentials of any kind (`.env`, IG keys, API keys, account numbers).
- Journal entries, reports, price cache, IV history, transcripts, paper books. They live in `MLAB_DATA_DIR` and are gitignored.
- Notebooks with outputs containing account data.

## Style

Plain, direct English. Numbers carry source and timestamp. No disclaimers.
