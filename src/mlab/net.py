"""HTTP session with retries, plus a reachability probe for every data source we use."""
from __future__ import annotations

import concurrent.futures as cf

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = "Mozilla/5.0 (X11; Linux x86_64) market-lab/0.1"


def session(user_agent: str = UA) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=Retry.DEFAULT_ALLOWED_METHODS)  # never replay POST/PUT (e.g. a login)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = user_agent
    return s


# host -> what it powers. Kept in one place so `mlab doctor` and the docs agree.
SOURCES: dict[str, str] = {
    "demo-api.ig.com": "IG REST (DEMO)",
    "api.ig.com": "IG REST (LIVE)",
    "demo-apd.marketdatasystems.com": "IG streaming (DEMO, Lightstreamer)",
    "apd.marketdatasystems.com": "IG streaming (LIVE, Lightstreamer)",
    "query1.finance.yahoo.com": "Yahoo prices (yfinance)",
    "query2.finance.yahoo.com": "Yahoo fundamentals/options (yfinance)",
    "fc.yahoo.com": "Yahoo auth cookie (yfinance)",
    "guce.yahoo.com": "Yahoo consent (yfinance)",
    "stooq.com": "Stooq daily prices (fallback, keyless)",
    "fred.stlouisfed.org": "FRED macro CSV (keyless)",
    "api.stlouisfed.org": "FRED API (FRED_API_KEY)",
    "data.sec.gov": "SEC EDGAR company facts & filings",
    "www.sec.gov": "SEC EDGAR ticker map & documents",
    "efts.sec.gov": "SEC full-text search",
    "api.binance.com": "Binance crypto klines",
    "data-api.binance.vision": "Binance public market data mirror",
    "api.kraken.com": "Kraken crypto OHLC",
    "api.coingecko.com": "CoinGecko crypto market data",
    "api.frankfurter.app": "ECB FX reference rates",
    "api.alternative.me": "Crypto Fear & Greed index",
    "www.alphavantage.co": "Alpha Vantage (ALPHAVANTAGE_API_KEY)",
    "financialmodelingprep.com": "FMP (FMP_API_KEY)",
}


def probe(host: str, timeout: float = 8) -> tuple[str, str]:
    try:
        r = requests.get(f"https://{host}/", timeout=timeout, allow_redirects=False)
        return host, f"ok ({r.status_code})"
    except requests.exceptions.ProxyError as e:
        return host, "BLOCKED by network policy" if "403" in str(e) else f"proxy error: {e}"
    except requests.exceptions.RequestException as e:
        return host, f"unreachable: {type(e).__name__}"


def probe_all(hosts=None) -> dict[str, str]:
    hosts = list(hosts or SOURCES)
    with cf.ThreadPoolExecutor(12) as ex:
        return dict(ex.map(probe, hosts))
