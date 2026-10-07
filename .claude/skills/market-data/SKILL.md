---
name: market-data
description: Pull live and historical market data (stocks, ETFs, indices, FX, futures, crypto, IG instruments, macro, filings) through the mlab CLI or Python API with caching, source tags and fallbacks. Use for any data pull before analysis.
---
# Market data

- Prices: `./mlab price SYMBOL [-i 1m|5m|15m|1h|4h|1d|1wk] [-p 1mo..max | --start] [--csv out.csv]`
  Symbols: `AAPL`, `VOD.L`, `^GSPC`, `EURUSD=X`, `GC=F`, `BTC-USD`, `crypto:BTC` (Binance/Kraken), `stooq:aapl.us`, `ig:EPIC`.
- Python: `from mlab.data import get_prices, get_many`; frames carry `attrs["source"]`, `attrs["fetched_at"]`, `attrs["last_bar"]`. Quote them.
- Macro: `from mlab.providers.macro import fred, dashboard, ecb_fx`. Filings: `from mlab.providers import sec`.
- Cache: parquet in `data/cache/` (`./mlab cache stats|prune|clear`). Use `--refresh` for intraday freshness.
- Universes: `./mlab universes` (indices, fx, commodities, rates, us_megacaps, sectors, uk_ftse, crypto, macro_board).
- Blocked host? `./mlab doctor` lists it. Then: connected data connectors (Alpha Vantage, FMP, Financial Datasets,
  Bigdata.com, CoinMarketCap) -> WebSearch/WebFetch -> run on Cezar's machine. Say which source you used.
