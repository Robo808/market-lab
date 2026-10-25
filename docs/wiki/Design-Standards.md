# Design Standards

How market-lab code is written. Every change follows these rules, and reviews check them. When a rule here and the code disagree, the
code is the debt: fix it or log it under [Known debt](#known-debt).

## 1. Shape of the system

```
providers/  (I/O: HTTP, IG, files)        -> adapters, one per source, raise on failure
data, cache, storage                      -> routing, fallbacks, caching, safe writes
ta, ta_catalog, signal_lab, options, ...  -> pure analytics on DataFrames, no I/O
quant/                                    -> strategies, engine, validation, paper books
cli, quant/cli                            -> argument parsing, printing, logging setup
```

- **Dependencies point down only.** Analytics never import `cli`. Providers never import analytics.
- **Functional core, imperative shell.** Analytics are pure functions: `DataFrame in -> DataFrame/dict out`, with no network,
  disk, clock or globals. All I/O sits at the edges (providers, `data`, `storage`, `cli`). This keeps every number reproducible
  and every test offline.

## 2. Paradigm: functions first, objects where state lives

| Use | When | Examples here |
|---|---|---|
| Plain function | Transforms and calculations (the default) | `ta.rsi`, `stats.summary`, `options.bs_price` |
| `@dataclass(frozen=True)` | A value with several fields passed around together (make it frozen unless it must change) | `IGConfig`, `signal_lab.Signal`, `quant.strategies.Strategy`; `TradeCard`, `news.Query`, `options.Leg` are plain `@dataclass` |
| Class with a lifecycle | A resource that holds a connection, a session or a subscription; it is always a context manager | `IG` (session tokens, logout on exit), streaming client |
| Registry `dict[str, callable]` | A choice made by name at runtime (strategy pattern) | `quant.strategies.REGISTRY`, `ta_catalog.CATALOG`, `backtest.STRATEGIES` |
| Composable transform | Layering behaviour on another strategy's output (decorator pattern) | overlays `vol_target(position, df)`, `regime_filter(position, df)` |
| Ordered fallback chain | Try sources in order and record why each failed (chain of responsibility) | `data.get_prices` (Yahoo -> Stooq) |
| `typing.Protocol` | Several interchangeable providers | price providers: `history(symbol, start, end, interval) -> DataFrame` |

Avoid inheritance hierarchies, singletons, and module-level mutable state; prefer composition. A module-level registry filled at import
time is fine. A cache or a session kept in a module global is not: put it in an object or pass it in.

## 3. Data structures and contracts

- **OHLCV frame:**
  - Index: `DatetimeIndex`, tz-aware **UTC**, sorted and unique.
  - Columns: lowercase float `open high low close volume`, plus optional extras such as `adj_close`, `bid_close`, `ask_close`, `spread`.
  - `df.attrs` carries `source`, `fetched_at` and `last_bar`; every result cites them.
- **Panel:** a wide frame, one column per symbol, on the same index. Do not use long or melted frames in analytics.
- **Signals and positions:** a `Series` aligned to the input index. A position in `[-1, 1]` or weights per column. The decision taken at
  the close of bar *t* earns the return of bar *t+1*.
- **Results:** a `dict` of scalars and small frames. It must be JSON-serialisable through `options.jsonable`, so `--json` always works.
- **Vectorise** with numpy/pandas. Write a Python loop over bars only when the logic is path-dependent (trailing stops, PSAR, zigzag),
  and say so in a comment.
- Pick the structure for the access pattern:
  - `dict`/`set` for membership and lookups.
  - `collections.deque(maxlen=n)` for rolling buffers of ticks.
  - `heapq` for top-N.
  - `bisect` on sorted timestamps.
  - Never `list.index` or `x in list` inside a loop.

## 4. Numerics and algorithms

- **Use the textbook definition and cite it in the docstring:**
  - Wilder smoothing (`alpha = 1/n`) for RSI, ATR and ADX.
  - `ewm(adjust=False)` for EMA.
  - Population std (`ddof=0`) for Bollinger.
  - Downside deviation for Sortino.
  - Bailey & López de Prado for PSR/DSR.
- **Annualise by bar interval** (`stats.ANN`), never with a hard-coded 252. 24-hour markets (FX, crypto) have more bars per year.
- **No look-ahead:**
  - No `shift(-k)`, no `rolling(center=True)`, no `resample(label="right")` on the signal side.
  - Every strategy and signal has a truncation test: appending bars must not change earlier outputs.
- **Statistics:**
  - Overlapping-horizon returns need HAC (Newey-West) or block methods; never plain t-tests.
  - Correct for multiple testing with `statsmodels.stats.multitest`.
  - Count every trial for DSR.
  - Seed every random draw (`np.random.default_rng(seed)`).
- **Use total return** for anything held through a dividend. Price-only series are fine for intraday timing.
- Compare floats with `np.isclose`. Test NaN with `np.isnan`/`pd.isna`, never with `x == x` or the truthiness of `x or default`.

## 5. Time

- UTC everywhere: `pd.Timestamp.now(tz="UTC")`. A naive `Timestamp.now()` or `datetime.now()` is a bug.
- Normalise inputs once, at the edge:
  ```python
  t = pd.Timestamp(x)
  t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
  ```
  Never `pd.Timestamp(aware, tz=...)`, which raises.
- Convert to exchange-local time only for display and session logic. The bar still in progress is not a completed bar.

## 6. Caching (three tiers)

| Tier | What | Where | Rule |
|---|---|---|---|
| In-process | Repeated pure calls in one run (quotes, search, SEC facts) | `functools.lru_cache`, `cache.TTLMemo` | Pure inputs only; no cross-run staleness |
| Series cache | Price bars and other **re-fetchable** data | `$MLAB_DATA_DIR/data/cache/<provider>/<interval>/<symbol>.parquet` | Merge with new rows winning; fetch only the missing head and tail; staleness depends on interval; LRU-pruned at `MLAB_CACHE_MAX_MB` |
| History | **Irreplaceable** data: IV history, news history, journal, paper books | Their own directories under `$MLAB_DATA_DIR` | Never in the pruned cache; never deleted by code |

- **Allowance-limited sources** (IG: 10k points a week) are cache-first. Re-downloading bars already on disk is a bug.
- **Cache keys** are sanitised (`cache._safe`) and include provider, interval and environment (`ig-demo` vs `ig-live`).

## 7. Storage and databases

**Decision: files, not a database server.**

- **Formats:** Parquet for time series. Write-once JSON event files for the journal. CSV only for small, human-read tables.

**Why:**
- The shared folder is an object store mounted through FUSE into many containers.
- File locks do not cross containers.
- SQLite needs working locks and, in WAL mode, shared memory. Both break on this mount, so **never put SQLite in the shared folder**.
- Event files plus Parquet need no server and survive any container.

**Querying:** DuckDB over the Parquet files (`duckdb.sql("select ... from 'data/cache/yahoo/1d/*.parquet'")`) is the approved way to run
ad-hoc SQL. It only reads.

**Revisit when:**
- several people write the same records at once,
- or queries over more than about 1 GB become routine.

Then move to hosted Postgres behind a connector, keeping the same event model.

## 8. Concurrency

**The model:** many threads, each in its own container, write one shared folder. Nothing coordinates between containers. Inside one
process, threads are used only for I/O fan-out.

1. **Atomic writes, always.** Use `storage.atomic_write` / `atomic_write_text` / `atomic_to_parquet`: a unique temp file in the same
   directory, then `os.replace`. Never write the target in place, and never use a fixed `.tmp` name (two writers collide on it).
2. **Irreplaceable shared state is write-once events folded on read.**
   - The journal (`journal/events/`) and news history (`data/news_history/<KEY>/`) add one uniquely named file per change and never
     rewrite one.
   - Readers fold the files in name order (UTC time, then a random suffix).
   - A read-modify-write of a shared file loses updates: `tests/test_storage.py` shows a lock-free rewrite keeping 29 of 40 entries.
3. **Re-fetchable shared state may be last-writer-wins** (the price cache), with `storage.file_lock` to serialise writers inside one
   container.
4. **Rendered views are derived** (`JOURNAL.md`). They are regenerated, never edited, and never a source of truth.
5. **Threads:**
   - Use `concurrent.futures.ThreadPoolExecutor` for network fan-out.
   - Workers return results; they do not append to shared lists.
   - Put CPU-bound work in vectorised numpy, not threads.
   - When a callback must hand data across threads, use `queue.Queue` (bounded) or a `threading.Lock` around the append.
6. **Semaphores and rate limits:**
   - Bound concurrency per host with the executor's `max_workers` or a `threading.BoundedSemaphore`.
   - Pace requests to the published limits: SEC 10/s, GDELT about 1/s, IG non-trading requests per minute and the weekly price allowance.
   - On HTTP 429, back off; never hammer.
7. **Idempotency:**
   - Retry only idempotent requests (`net.session` retries GET, never POST/PUT; a repeated IG login deepens a lockout).
   - Every command is safe to re-run: a paper rebalance on the same bar only re-marks, and a journal add makes a new id instead of
     overwriting.

## 9. Streaming and events

- **Event shape:** an event is a plain `dict` with `type`, `t` (UTC ISO), `source` and a payload. Consumers de-duplicate by a key,
  because Lightstreamer and HTTP retries can deliver twice.
- **Listener callbacks** (Lightstreamer `onItemUpdate`) run on the client's thread.
  - Do the minimum there: parse, then `queue.put_nowait`.
  - Never do I/O or heavy work in a callback.
  - When the queue is full, drop the oldest tick and count the drop.
- **Every subscription** registers a subscription-error listener and a connection-status listener.
  - A bad epic or a lost session must be logged, not shown as silence.
  - Reconnect with exponential backoff and re-login when the session has expired.
- **Persist streams** as write-once partitions (one Parquet file per symbol per minute or hour), the same rule as §8.2. Rolling windows
  in memory use `deque(maxlen=n)`.
- **No event bus yet.** Add an in-process publish/subscribe layer only when two or more consumers need the same stream. Until then,
  the queue is the bus.

## 10. Errors and logging

- **Exceptions:**
  - Raise specific exceptions: `ValueError` for bad input, `LookupError` for no data, `IGError` / `ReadOnlyViolation` for IG.
  - Messages say what failed and what fixes it (`ig.FIX_HINTS`).
- **Catching:**
  - Catch broad `Exception` only at boundaries: a fallback chain, a per-source fan-out, or the CLI `main`. Record what was caught in
    `attrs["errors"]` or the result's `errors` dict, or log it.
  - **`except: pass` is banned** (ruff S110/S112 in CI).
  - An optional input that fails is logged at INFO with the reason, then work carries on.
- **Logging:**
  - Library modules call `log = logging.getLogger(__name__)` and never `print`.
  - The CLI configures logging (`cli.setup_logging`): WARNING by default, `MLAB_LOG_LEVEL=INFO|DEBUG`, and `MLAB_DEBUG=1` for DEBUG plus
    full tracebacks.
  - Results go to stdout and logs to stderr, so `--json` output stays parseable.
  - **Never log secrets:** no request headers, CST/X-SECURITY-TOKEN, env values, login bodies or account numbers.
  - Log the epic and status code, not the session.

## 11. Network

- One `net.session()` per client. Every request has a timeout. Retries cover idempotent methods only, and only on 429/5xx.
- Each new host goes in `net.SOURCES`, so `mlab doctor` probes it.
- A blocked host is a fallback, not a crash: try the next source and say which one served the data.

## 12. Security

- The IG client is read-only by construction (`_ALLOWED_WRITES`), and a test enforces it. Never widen it without an explicit decision.
- Parse untrusted XML/HTML with `defusedxml`. Run subprocesses with fixed argument lists. Never use `shell=True` or `eval`.
- Credentials come only from network secrets or env vars. They are never written to files, commits, logs or test fixtures.

## 13. Testing

- Tests run offline, on synthetic data or recorded fixtures under `tests/`. `MLAB_DATA_DIR` points at a temp dir (`conftest.py`).
- **Every bug fix ships with a regression test** that fails on `main`.
- Look-ahead truncation tests for every strategy, signal and overlay.
- Multi-process tests for anything that writes shared state.
- New indicators are checked against a reference implementation (TA-Lib, py_vollib, statsmodels) when one exists. Note the tolerance
  and any deliberate difference.

## 14. Style and tooling

- **Lint:** `ruff check src tests` runs in CI. The rule set in `pyproject.toml` is a floor: raise it, never lower it, and justify
  every per-file ignore.
- **Typing:** type hints on every public function. Docstrings state units and conventions (points or price, decimal or %, which bar
  a value refers to).
- **Names:** say what a thing is. Single letters only for math or OHLC conventions (`h l c o`, `r` for returns, `S K T` in options).
- **Dependencies:**
  - Prefer an established library when it is correct, maintained and light (statsmodels, rapidfuzz, vaderSentiment).
  - Keep hand-rolled code when it is small, tested and verified against a reference (the TA catalogue, Black-Scholes).
  - Pin lower bounds; Dependabot raises them.

## 15. Pitfalls we have actually hit

| Pitfall | Symptom | Rule |
|---|---|---|
| `pd.Timestamp(aware, tz="UTC")` | ValueError on every `ig:` price call | §5 |
| Re-downloading a range the cache already covers | IG weekly allowance burned | §6 |
| Read-modify-write of a shared file | Journal entries lost between threads | §8.2 |
| Fixed `.tmp` file name | Two writers corrupt each other | §8.1 |
| Hard-coded 252 periods | Hourly Sharpe and funding wrong | §4 |
| `std(ddof=1)` in Bollinger, std of negatives in Sortino | Bands and ratios off by 3-15% | §4 |
| `x or -9` when `x` is NaN | NaN wins `max()` | §4 |
| A `break` that made a merge dead code | SEC history silently truncated | §13, regression test |
| `except Exception: pass` | Failures invisible | §10 |
| Auto-retrying POST | Login replayed into an IG lockout | §8.7 |
| News history in the LRU cache | Irreplaceable data pruned | §6 |
| Price-only Yahoo series | Dividends missing from backtests | §4 (open) |

## 16. Review checklist

Before asking for review:

- [ ] `ruff check src tests` and `pytest -q` pass.
- [ ] No I/O in analytics. Logging, not print. No `except: pass`.
- [ ] UTC timestamps. No look-ahead. Annualised by interval.
- [ ] Shared-folder writes are atomic, and irreplaceable state is write-once.
- [ ] Retries only on idempotent calls. Every request has a timeout. The allowance is respected.
- [ ] A regression test for every bug fixed.
- [ ] No credentials, journal data or account data in the diff.

## Known debt

Tracked here until fixed:

- Yahoo series are price-only (`auto_adjust=False`, `adj_close` unused).
- `backtest.py` duplicates `quant/engine.py`.
- Supertrend, Yang-Zhang volatility and drawdown are implemented more than once.
- `signal_lab` t-stats ignore overlapping returns.
- `news.score` re-implements VADER, and `news.dedupe` is O(n²) difflib.
- Module-level path constants are read at import.
- `mlab algo` annualises at 252 regardless of interval.
- The IG stream has no error or status listener and no reconnect.
