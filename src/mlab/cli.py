"""mlab: the Market Lab command line. `mlab --help` lists everything; every command prints
markdown tables (or JSON with --json) and states its data source and timestamp."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


# ---- output helpers -------------------------------------------------------------
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "n/a"
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return v


def show(obj, as_json=False, title: str | None = None):
    if title:
        print(f"\n### {title}\n")
    if as_json:
        if isinstance(obj, pd.DataFrame):
            print(obj.reset_index().to_json(orient="records", date_format="iso", default_handler=str))
        else:
            print(json.dumps(obj, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o), indent=2))
        return
    if isinstance(obj, pd.DataFrame):
        print(obj.map(_fmt).to_markdown() if not obj.empty else "(empty)")
    elif isinstance(obj, dict):
        flat = {k: v for k, v in obj.items() if not isinstance(v, (pd.DataFrame, dict, list))}
        if flat:
            print(pd.DataFrame({"value": {k: _fmt(v) for k, v in flat.items()}}).to_markdown())
        for k, v in obj.items():
            if isinstance(v, (pd.DataFrame, dict)) and len(v):
                show(v, title=k)
            elif isinstance(v, list) and v:
                print(f"\n**{k}:** " + "; ".join(map(str, v)))
    else:
        print(obj)


def src_line(df: pd.DataFrame):
    print(f"\n_Source: {df.attrs.get('source', '?')} · fetched {df.attrs.get('fetched_at', '?')} · last bar {df.attrs.get('last_bar', '?')}_")


# ---- commands -----------------------------------------------------------------------
def cmd_doctor(a):
    from . import cache
    from .config import DATA_DIR, OPTIONAL_KEYS, ROOT, env, ig_config
    from .net import SOURCES, probe_all
    print("## Market Lab doctor\n")
    print(f"python {sys.version.split()[0]} · pandas {pd.__version__}")
    print(f"code: {ROOT} · data (journal, reports, cache): {DATA_DIR}")
    if not a.offline:
        import importlib
        hosts = dict(SOURCES)
        for mod in EXTENSIONS:
            try:
                hosts.update(getattr(importlib.import_module(mod), "HOSTS", {}))
            except ImportError:
                pass
        res = probe_all(hosts)
        rows = [{"host": h, "powers": hosts[h], "status": s} for h, s in res.items()]
        show(pd.DataFrame(rows).set_index("host"), title="Data sources reachability")
        blocked = [h for h, s in res.items() if "BLOCKED" in s or "unreachable" in s]
        if blocked:
            print("\nBlocked hosts need adding to this environment's allowed domains (see docs/NETWORK.md):\n  " + ", ".join(blocked))
    for t in ("DEMO", "LIVE"):
        c = ig_config(t)
        state = ("login from a Body parameter network secret (not visible here; `mlab ig login` tests it)"
                 if c.login_via_secret else "credentials present" if c.configured else "missing " + ", ".join(c.missing()))
        print(f"\nIG {t}: {state}")
    if not any(ig_config(t).configured for t in ("DEMO", "LIVE")):
        print("  IG login needs IG_ACC_TYPE (DEMO|LIVE) plus a username and password: best as a network secret of type\n"
              "  Body parameter on demo-api.ig.com / api.ig.com (path prefix /gateway/deal/session, parameters identifier\n"
              "  and password), or as IG_USERNAME / IG_PASSWORD (or IG_DEMO_* / IG_LIVE_*). The API key goes in a separate\n"
              "  network secret (header X-IG-API-KEY, no prefix) or IG_API_KEY. Then `mlab ig login` tests it.")
    print("\nOptional keys: " + ", ".join(f"{k}={'set' if env(k) else 'unset'}" for k in OPTIONAL_KEYS))
    show(cache.stats(), title="Cache")


def cmd_price(a):
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, a.end, a.period, refresh=a.refresh)
    if a.csv:
        df.to_csv(a.csv)
        print(f"wrote {len(df)} rows to {a.csv}")
    show(df.tail(a.tail), a.json)
    src_line(df)


def cmd_ta(a):
    from . import ta
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period, refresh=a.refresh)
    snap = ta.snapshot(df)
    show(snap, a.json, title=f"{a.symbol} {a.interval} technical snapshot")
    if not a.json:
        show(ta.support_resistance(df).set_index("level"), title="Support / resistance (swing clusters)")
        show(ta.pivots(df), title="Floor pivots (from last bar)")
        show(ta.fib_retracements(df), title="Fibonacci retracements (last 120 bars)")
    if a.chart:
        from .charts import candle_html
        p = candle_html(df.tail(a.chart_bars), f"{a.symbol} {a.interval}", levels=list(ta.support_resistance(df)["level"]))
        print(f"\nchart: {p}")
    src_line(df)


def cmd_chart(a):
    from . import ta
    from .charts import candle_html
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period)
    print(candle_html(df, f"{a.symbol} {a.interval}", levels=list(ta.support_resistance(df)["level"])))


def _symbols(a):
    from .universes import UNIVERSES
    syms = list(a.symbols or [])
    for u in a.universe or []:
        syms += UNIVERSES[u]
    if not syms:
        sys.exit("give symbols or --universe (see `mlab universes`)")
    return syms


def cmd_scan(a):
    from . import ta
    from .data import get_prices
    rows, errs = [], {}
    for s in _symbols(a):
        try:
            df = get_prices(s, a.interval, None, None, a.period)
            sn = ta.snapshot(df)
            rows.append({"symbol": s, "last": sn["last"], "1d": sn["chg_1d"], "1m": sn["chg_1m"], "3m": sn["chg_3m"],
                         "rsi": sn["rsi14"], "vs_sma200": sn["last"] / sn["sma200"] - 1 if sn["sma200"] == sn["sma200"] else np.nan,
                         "from_52w_hi": sn["pct_from_52w_high"], "atr%": sn["atr_pct"], "trend": sn["regime_trend"],
                         "adx": sn["regime_adx"], "signals": "; ".join(sn["signals"])})
        except Exception as e:
            errs[s] = str(e)[:100]
    df = pd.DataFrame(rows).set_index("symbol")
    if a.sort and a.sort in df:
        df = df.sort_values(a.sort, ascending=a.asc)
    show(df, a.json)
    if errs:
        print("\nerrors: " + json.dumps(errs))
    print(f"\n_Scan at {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC_")


def cmd_rs(a):
    from .data import get_many
    from .stats import relative_strength
    show(relative_strength(get_many(_symbols(a), period="2y")), a.json, title="Relative strength (higher rank_avg = stronger)")


def cmd_corr(a):
    from .data import get_many
    from .stats import correlation
    show(correlation(get_many(_symbols(a), period=a.period), a.window).round(2), a.json, title=f"Return correlation ({a.window or 'full'} bars)")


def cmd_macro(a):
    from .providers.macro import dashboard
    show(dashboard().set_index("series"), a.json, title="Macro dashboard (FRED)")
    if a.liquidity:
        from .lenses import net_liquidity
        show(net_liquidity().tail(8), a.json, title="US net liquidity (Fed BS - TGA - RRP, $bn)")


def cmd_dd(a):
    from .providers import sec
    fin = sec.annual_financials(a.ticker, a.years)
    cols = [c for c in ["revenue", "gross_margin", "op_margin", "net_income", "fcf", "fcf_ex_sbc", "sbc", "roic_pre_tax", "roe",
                        "total_debt", "net_debt", "cash", "diluted_shares", "buybacks", "current_ratio", "interest_cover"] if c in fin]
    t = fin[cols].T
    t.columns = [d.strftime("FY%Y") for d in t.columns]
    show(t, a.json, title=f"{a.ticker} annual financials (SEC XBRL, USD)")
    show(sec.filings(a.ticker, ("10-K", "10-Q", "8-K", "DEF 14A", "S-1", "20-F", "6-K"), 12).set_index("filingDate"), a.json, title="Recent filings")
    show(sec.insider_filings(a.ticker, 10).set_index("filingDate"), a.json, title="Insider (Form 4) filings")
    try:
        from .providers import yahoo
        info = yahoo.info(a.ticker)
        keys = ["longName", "sector", "industry", "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook",
                "enterpriseToEbitda", "profitMargins", "returnOnEquity", "debtToEquity", "shortPercentOfFloat", "heldPercentInsiders",
                "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions", "earningsTimestamp"]
        show({k: info.get(k) for k in keys}, a.json, title="Market snapshot (Yahoo)")
    except Exception as e:
        print(f"\n(yahoo snapshot unavailable: {str(e)[:80]})")


def cmd_lens(a):
    from . import lenses
    if a.which == "soros":
        r = lenses.soros(a.tickers or None)
    else:
        if not a.tickers:
            sys.exit("give a ticker")
        r = getattr(lenses, a.which)(a.tickers[0])
        r.pop("financials", None)
    if a.json:
        show(r, True)
        return
    for k in ("metrics", "checks", "trend_board", "net_liquidity", "macro", "insider_filings"):
        if k in r and len(r[k]) if hasattr(r.get(k), "__len__") else k in r:
            show(r[k] if not isinstance(r[k], pd.DataFrame) else r[k].set_index(r[k].columns[0]), title=f"{r['lens']} · {k}")
    if "score" in r:
        print(f"\n**{r['lens']} checklist score: {r['score']}**")
    print("\nSources: " + "; ".join(r.get("sources", [])) + f" · {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")


def cmd_crypto(a):
    from .providers import crypto
    show(crypto.market_overview(a.n).set_index("symbol"), a.json, title="Top crypto by market cap (CoinGecko)")
    try:
        show(crypto.fear_greed(7), a.json, title="Crypto Fear & Greed")
    except Exception as e:
        print(f"(fear & greed unavailable: {e})")


def cmd_backtest(a):
    from . import backtest
    from .data import get_prices
    df = get_prices(a.symbol, a.interval, a.start, None, a.period)
    params = json.loads(a.params) if a.params else {}
    res = backtest.run(df, backtest.STRATEGIES[a.strategy](df, **params), spread=a.spread, funding_annual=a.funding)
    show(pd.DataFrame({"strategy": res["strategy"], "buy_hold": res["buy_hold"]}), a.json, title=f"{a.strategy} on {a.symbol}")
    print(f"\ntrades {res['trades']} · exposure {res['exposure']:.0%} · cost drag {res['cost_drag_total']:.2%}")
    src_line(df)


def cmd_size(a):
    from .risk import stake_per_point
    show(stake_per_point(a.equity, a.risk, a.entry, a.stop, a.point_size), a.json, title="Position size")


def _card_from_args(a):
    from .risk import TradeCard
    lo, hi = (a.entry + [a.entry[0]])[:2]
    return TradeCard(instrument=a.instrument, bias=a.bias.upper(), entry_zone=(lo, hi), stop=a.stop, targets=a.targets,
                     risk_pct=a.risk, horizon=a.horizon, conviction=a.conviction, catalysts=a.catalyst or [],
                     kills_it=a.kills or "", epic=a.epic, lens=a.lens or "", thesis=a.thesis or "", sources=a.source or [])


def cmd_card(a):
    card = _card_from_args(a)
    print(card.markdown(a.equity, a.point_size))
    print("\n" + card.line())
    if a.journal:
        from . import journal
        r = journal.add(card.to_dict(), a.status)
        print(f"\njournaled as {r['id']} ({a.status})")


def cmd_journal(a):
    from . import journal
    if a.action == "list":
        show(journal.table(a.status), a.json)
    elif a.action == "update":
        show(journal.update(a.id, a.status, a.fill, a.exit, a.size, a.note or ""), a.json)
    elif a.action == "review":
        show(journal.review(), a.json, title="Journal review")
    elif a.action == "render":
        print(journal.render())


def cmd_ig(a):
    from .config import ig_config
    from .providers.ig import IG
    with IG(ig_config(a.env)) as ig:
        info = ig.login()
        act = a.action
        if act == "login":
            show(info, a.json, title=f"IG {a.env or ig.cfg.acc_type} session")
        elif act == "accounts":
            show(ig.accounts().set_index("accountId"), a.json)
        elif act == "positions":
            show(ig.positions(), a.json, title="Open positions")
        elif act == "orders":
            show(ig.working_orders(), a.json, title="Working orders")
        elif act == "watchlists":
            show(ig.watchlists(), a.json)
        elif act == "watchlist":
            show(ig.watchlist(a.arg[0]), a.json)
        elif act == "search":
            show(ig.search(" ".join(a.arg)), a.json)
        elif act == "market":
            show(ig.market(a.arg[0]), True)
        elif act == "snapshot":
            show(ig.snapshot(a.arg), a.json)
        elif act == "sentiment":
            show(ig.sentiment(a.arg[0]), a.json, title="IG client sentiment")
        elif act == "prices":
            df = ig.prices(a.arg[0], a.interval, a.start, None, use_cache=not a.refresh)
            show(df.tail(a.tail), a.json)
            print(f"\nallowance: {ig.last_allowance or 'served from cache'}")
        elif act == "stream":
            ig.stream(a.arg, a.seconds, on_update=lambda t: print(json.dumps(t), flush=True))
        elif act == "activity":
            show(ig.activity(a.days), a.json)
        elif act == "transactions":
            show(ig.transactions(a.days), a.json)


def cmd_cache(a):
    from . import cache
    if a.action == "stats":
        show(cache.stats())
    elif a.action == "prune":
        print(f"removed {cache.prune()} files")
    elif a.action == "clear":
        print(f"removed {cache.clear(a.provider)} files")


def cmd_universes(a):
    from .universes import IG_COMMON_EPICS, UNIVERSES
    for k, v in UNIVERSES.items():
        print(f"{k}: {' '.join(v)}")
    print("\nIG common epics (verify with `mlab ig search`):")
    for k, v in IG_COMMON_EPICS.items():
        print(f"  {k}: {v}")


# ---- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlab", description="Market Lab: data, TA, due diligence, lenses, IG (read-only), journal")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        q = sp.add_parser(name, help=help_)
        q.set_defaults(fn=fn)
        q.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        return q

    def win(q, interval="1d", period="1y"):
        q.add_argument("--interval", "-i", default=interval)
        q.add_argument("--period", "-p", default=period, help="1mo 3mo 6mo 1y 2y 5y 10y max")
        q.add_argument("--start")
        q.add_argument("--refresh", action="store_true", help="bypass cache")

    q = add("doctor", cmd_doctor, "check network reachability, credentials, cache")
    q.add_argument("--offline", action="store_true")

    q = add("price", cmd_price, "OHLCV history: AAPL, ^GSPC, EURUSD=X, crypto:BTC, ig:<EPIC>")
    q.add_argument("symbol"); win(q); q.add_argument("--end"); q.add_argument("--tail", type=int, default=15); q.add_argument("--csv")

    q = add("ta", cmd_ta, "technical snapshot, levels, pivots, fibs, signals")
    q.add_argument("symbol"); win(q); q.add_argument("--chart", action="store_true"); q.add_argument("--chart-bars", type=int, default=260)

    q = add("chart", cmd_chart, "interactive candlestick chart (HTML in reports/)")
    q.add_argument("symbol"); win(q)

    for name, fn, h in (("scan", cmd_scan, "technical scan across symbols/universes"),
                        ("rs", cmd_rs, "relative strength ranking"), ("corr", cmd_corr, "return correlation matrix")):
        q = add(name, fn, h)
        q.add_argument("symbols", nargs="*"); q.add_argument("--universe", "-u", action="append")
        if name == "scan":
            win(q); q.add_argument("--sort", default="3m"); q.add_argument("--asc", action="store_true")
        if name == "corr":
            q.add_argument("--period", default="1y"); q.add_argument("--window", type=int)

    q = add("macro", cmd_macro, "FRED macro dashboard (+ --liquidity)")
    q.add_argument("--liquidity", action="store_true")

    q = add("dd", cmd_dd, "due diligence pack: 10y financials, filings, insiders, valuation snapshot")
    q.add_argument("ticker"); q.add_argument("--years", type=int, default=10)

    q = add("lens", cmd_lens, "investor lens screens: buffett TICKER | burry TICKER | soros [ASSETS...]")
    q.add_argument("which", choices=["buffett", "burry", "soros"]); q.add_argument("tickers", nargs="*")

    q = add("crypto", cmd_crypto, "crypto market overview + fear & greed")
    q.add_argument("-n", type=int, default=20)

    q = add("backtest", cmd_backtest, "backtest a built-in strategy with IG-style costs")
    q.add_argument("symbol"); q.add_argument("--strategy", "-s", default="sma_cross", choices=["sma_cross", "donchian", "rsi_reversion", "supertrend"])
    q.add_argument("--params", help='JSON, e.g. \'{"fast":20,"slow":100}\''); q.add_argument("--spread", type=float, default=0.0)
    q.add_argument("--funding", type=float, default=0.0); win(q, period="10y")

    q = add("size", cmd_size, "stake per point for a given account, risk %%, entry and stop")
    q.add_argument("--equity", type=float, required=True); q.add_argument("--risk", type=float, default=1.0)
    q.add_argument("--entry", type=float, required=True); q.add_argument("--stop", type=float, required=True)
    q.add_argument("--point-size", type=float, default=1.0)

    q = add("card", cmd_card, "build the standard trade card (optionally journal it)")
    q.add_argument("instrument"); q.add_argument("--bias", required=True)
    q.add_argument("--entry", type=float, nargs="+", required=True, help="one price or low high")
    q.add_argument("--stop", type=float, required=True); q.add_argument("--targets", type=float, nargs="+", required=True)
    q.add_argument("--risk", type=float, default=1.0); q.add_argument("--horizon", default="swing (2-8 weeks)")
    q.add_argument("--conviction", type=int, default=3); q.add_argument("--catalyst", action="append")
    q.add_argument("--kills"); q.add_argument("--epic"); q.add_argument("--lens"); q.add_argument("--thesis")
    q.add_argument("--source", action="append"); q.add_argument("--equity", type=float); q.add_argument("--point-size", type=float, default=1.0)
    q.add_argument("--journal", action="store_true"); q.add_argument("--status", default="idea", choices=["idea", "open"])

    q = add("journal", cmd_journal, "trade journal: list | update ID | review | render")
    q.add_argument("action", choices=["list", "update", "review", "render"]); q.add_argument("id", nargs="?")
    q.add_argument("--status", choices=["idea", "open", "closed", "passed"]); q.add_argument("--fill", type=float)
    q.add_argument("--exit", type=float); q.add_argument("--size", type=float); q.add_argument("--note")

    q = add("ig", cmd_ig, "IG read-only: login accounts positions orders watchlists watchlist search market snapshot sentiment prices stream activity transactions")
    q.add_argument("action", choices=["login", "accounts", "positions", "orders", "watchlists", "watchlist", "search", "market",
                                      "snapshot", "sentiment", "prices", "stream", "activity", "transactions"])
    q.add_argument("arg", nargs="*"); q.add_argument("--env", choices=["DEMO", "LIVE"]); q.add_argument("--interval", "-i", default="1d")
    q.add_argument("--start"); q.add_argument("--tail", type=int, default=15); q.add_argument("--refresh", action="store_true")
    q.add_argument("--seconds", type=int, default=20); q.add_argument("--days", type=int, default=30)

    q = add("cache", cmd_cache, "cache stats | prune | clear [--provider]")
    q.add_argument("action", choices=["stats", "prune", "clear"]); q.add_argument("--provider")

    add("universes", cmd_universes, "list named symbol universes and common IG epics")

    # Desk modules register their own subcommands: each exposes register(add).
    import importlib
    for mod in EXTENSIONS:
        try:
            importlib.import_module(mod).register(add)
        except ModuleNotFoundError as e:
            if e.name != mod:
                raise
    return p


EXTENSIONS = ["mlab.ta_catalog", "mlab.signal_lab", "mlab.options", "mlab.earnings", "mlab.news", "mlab.cot",
              "mlab.quant.cli", "mlab.datasets", "mlab.textlab.gdelt"]


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # concise errors; full trace with MLAB_DEBUG=1
        import os
        if os.environ.get("MLAB_DEBUG"):
            raise
        sys.exit(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
