# Getting Started

## Install

Requires Python 3.11+ and git.

```bash
git clone https://github.com/Robo808/market-lab.git
cd market-lab
bash bootstrap.sh        # venv at ~/.venvs/market-lab, installs mlab with the stream and dev extras, runs mlab doctor
```

`bootstrap.sh` uses `uv` when it is installed and `python -m venv` otherwise. Override the venv location with
`MLAB_VENV`. After that:

```bash
./mlab --help            # or: bash mlab --help (works on mounts without exec rights)
```

The `mlab` wrapper bootstraps itself on first use, so a fresh clone can go straight to `bash mlab <command>`.

## Check the environment

```bash
bash mlab doctor
```

`doctor` probes every data host, reports which optional keys are present (never their values), and prints where code
and data live. Hosts that fail are listed with the fallback to use; see [[Data Sources]] and `docs/NETWORK.md`.

## First commands

```bash
bash mlab price AAPL -p 2y               # daily OHLCV, cached, with source and timestamp
bash mlab price crypto:BTC -i 4h -p 3mo  # crypto via Binance, Kraken as fallback
bash mlab ta NVDA --chart                # snapshot, levels, pivots, fibs, signals, HTML chart in reports/
bash mlab firing ^GSPC                   # signals firing now and their hit rate on this instrument
bash mlab scan -u indices -u fx --sort 3m
bash mlab macro --liquidity
bash mlab algo list
```

## Configuration

Everything is configured through environment variables. `.env.example` lists them with empty values; `.env` is
gitignored.

### State directory

| Variable | Default |
|---|---|
| `MLAB_DATA_DIR` | `/mnt/project-files/market-lab` when that shared folder exists, else the checkout |
| `MLAB_JOURNAL_DIR` | `$MLAB_DATA_DIR/journal` |
| `MLAB_REPORTS_DIR` | `$MLAB_DATA_DIR/reports` |
| `MLAB_CACHE_DIR` | `$MLAB_DATA_DIR/data/cache` |
| `MLAB_IV_DIR` | `$MLAB_DATA_DIR/data/iv_history` |
| `MLAB_TRANSCRIPTS_DIR` | `$MLAB_DATA_DIR/data/transcripts` |
| `MLAB_CACHE_MAX_MB` | `750` (LRU pruning threshold for the price cache) |

Journal, reports, caches and paper books are gitignored and never committed.

### Optional data keys

`SEC_USER_AGENT` (`"Name email"`, SEC asks every client to identify itself), `FRED_API_KEY`, `ALPHAVANTAGE_API_KEY`,
`FMP_API_KEY`, `FINNHUB_API_KEY`. Core features work without any of them.

### IG (read-only)

Credentials come through proxy-injected network secrets or environment variables, and are never stored in the repo.

| Credential | How it is supplied |
|---|---|
| API key | network secret adding the `X-IG-API-KEY` header on `demo-api.ig.com` / `api.ig.com`, or `IG_API_KEY` (`IG_DEMO_API_KEY` / `IG_LIVE_API_KEY` per environment) |
| Username and password | **Body parameter** network secret on the same host, path `/gateway/deal/session`, keys `identifier` and `password`; or `IG_USERNAME` / `IG_PASSWORD` (`IG_DEMO_*` / `IG_LIVE_*`) |
| `IG_ACC_TYPE` | always set: `DEMO` or `LIVE`. With it set and no username or password in the environment, the client sends the login body for the proxy to fill |
| `IG_ACC_NUMBER` | optional: account to switch to after login |

Full setup, including what to do if the secrets list marks one as **Not sent**, is in `docs/NETWORK.md`.

The client allows GET requests plus session login, account switch and logout, and refuses everything else. There are
no order or deal endpoints in the code. Historical prices count against IG's weekly allowance (10,000 points), so
`mlab price ig:<EPIC>` is cache-first and fetches only the missing tail. Spread bet (DFB) and CFD epics differ:
confirm with `bash mlab ig search "<name>"`.

```bash
bash mlab ig login
bash mlab ig positions
bash mlab ig search "FTSE"
bash mlab ig sentiment IX.D.FTSE.DAILY.IP
```

### Kraken

Public market data needs no key. Account access, when added, uses a query-only key from environment variables
(`KRAKEN_API_KEY`, `KRAKEN_API_SECRET`) through a read-only client.

### Credentials rule

Credentials come from the environment at runtime. Nothing prints, logs, caches or writes them, and the Security
workflow runs gitleaks over the whole history on every PR.

## With Claude Code

Open the checkout in Claude Code. `CLAUDE.md`, the desk agents in `.claude/agents/` and the playbooks in
`.claude/skills/` load automatically. Ask for a trade idea, a morning brief or a backtest and the PM session fans out to
the desks; see [[Desk Workflow]].

## Next

- [[CLI Reference]] for every command
- [[Strategy Library]] and [[Hypothesis Testing]] for systematic research
- [[Risk and Sizing]] for stakes and trade cards
