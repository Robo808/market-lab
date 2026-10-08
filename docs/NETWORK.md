# Network access

`./mlab doctor` probes every host below and shows which are reachable from wherever you run it.

The project's cloud sessions currently run on the built-in environment, whose network policy allows package
indexes and GitHub only, so the market data hosts are refused there. To open them, add a cloud environment
for the project (Project settings → Cloud environment → Add cloud environment) and either pick a broader
network access level or add these under **Allowed domains** (keep "Allow package managers" ticked).
Docs: https://code.claude.com/docs/en/cloud-environments#network-access

```
demo-api.ig.com
api.ig.com
demo-apd.marketdatasystems.com
apd.marketdatasystems.com
query1.finance.yahoo.com
query2.finance.yahoo.com
fc.yahoo.com
guce.yahoo.com
feeds.finance.yahoo.com
stooq.com
fred.stlouisfed.org
api.stlouisfed.org
www.sec.gov
data.sec.gov
efts.sec.gov
api.binance.com
data-api.binance.vision
api.kraken.com
api.coingecko.com
api.alternative.me
api.frankfurter.app
publicreporting.cftc.gov
news.google.com
api.gdeltproject.org
api.stocktwits.com
www.reddit.com
hn.algolia.com
www.alphavantage.co
financialmodelingprep.com
finnhub.io
```

IG: keep the API key as a network secret on `demo-api.ig.com` (DEMO) or `api.ig.com` (LIVE), custom header
`X-IG-API-KEY` with no prefix, so the key never enters the container (`IG_API_KEY` as a variable also works).
IG wants the username and password inside the login body, which a header secret cannot fill, so those go in as
environment variables: `IG_USERNAME`, `IG_PASSWORD` (or `IG_DEMO_*` / `IG_LIVE_*`), `IG_ACC_TYPE` (DEMO or LIVE),
optional `IG_ACC_NUMBER`. Other environment variables: optional `FRED_API_KEY`, `ALPHAVANTAGE_API_KEY`,
`FMP_API_KEY`, `FINNHUB_API_KEY`, `SEC_USER_AGENT` ("Name email@example.com").

Routes that work regardless of the container policy:
1. claude.ai connectors (Alpha Vantage, FMP, Financial Datasets, Bigdata.com, TEXT TO QUANT, Meltwater, CoinMarketCap).
2. WebSearch / WebFetch tools (Yahoo pages block robots; most news sites and IR pages work).
3. Running `./mlab` on your own machine (Remote Control), where your network is open and IG keys can live in your shell.
