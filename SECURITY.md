# Security

## Reporting
Open a private report under the repo's **Security > Report a vulnerability** (private vulnerability reporting),
not a public issue.

## What this repo never holds
- Credentials: IG and data-provider keys come from environment variables only (`.env` is gitignored,
  `.env.example` has empty values). The IG client is read-only and has no order or deal endpoints.
- Desk state: the trade journal, reports, paper books and price caches live in `$MLAB_DATA_DIR`, outside git.

## Automated checks
- `Security` workflow on every PR, on main and weekly: pip-audit over the installed dependency tree, bandit
  (medium and high findings fail), gitleaks over the full git history.
- `CodeQL` workflow: code scanning for Python and the workflow files.
- Dependabot: weekly grouped updates for pip and GitHub Actions.

Run the same checks locally:

```bash
pip install pip-audit bandit
pip-audit --skip-editable
bandit -r src -ll
gitleaks git --log-opts=--all --redact .     # https://github.com/gitleaks/gitleaks
```
