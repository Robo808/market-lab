"""IG REST + Lightstreamer client, READ-ONLY by construction.

Only GET requests and session management (login, account switch, logout) can leave this module;
any other method/path raises before a request is sent. Cezar executes trades himself on IG.

Credentials: env vars only (IG_USERNAME, IG_PASSWORD, IG_ACC_TYPE=DEMO|LIVE, optional IG_ACC_NUMBER,
or per-environment IG_DEMO_* / IG_LIVE_* overrides). The API key comes from IG_API_KEY or, better, from a
cloud network secret that adds the X-IG-API-KEY header at the proxy so the key never enters the container. Session tokens
are kept in memory for the life of the process and never written to disk.

Historical prices count against IG's weekly allowance (10,000 points/week on the standard
plan), so price history is cached and only the missing tail is requested.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pandas as pd

from .. import cache
from ..config import IGConfig, ig_config
from ..net import session

RESOLUTIONS = {
    "1s": "SECOND", "1m": "MINUTE", "2m": "MINUTE_2", "3m": "MINUTE_3", "5m": "MINUTE_5",
    "10m": "MINUTE_10", "15m": "MINUTE_15", "30m": "MINUTE_30", "1h": "HOUR", "2h": "HOUR_2",
    "3h": "HOUR_3", "4h": "HOUR_4", "1d": "DAY", "1wk": "WEEK", "1mo": "MONTH",
}
BAR = {"1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min", "10m": "10min", "15m": "15min",
       "30m": "30min", "1h": "1h", "2h": "2h", "3h": "3h", "4h": "4h", "1d": "1D", "1wk": "7D", "1mo": "31D"}

_ALLOWED_WRITES = {("POST", "/session"), ("PUT", "/session"), ("DELETE", "/session")}


class ReadOnlyViolation(RuntimeError):
    pass


class IGError(RuntimeError):
    pass


# IG error codes that mean "change a setting", mapped to the one thing to change.
FIX_HINTS = {
    "error.security.api-key-missing": "no API key reached IG: add a network secret for demo-api.ig.com (DEMO) or api.ig.com (LIVE) with header X-IG-API-KEY and no prefix, or set IG_API_KEY",
    "error.security.api-key-invalid": "API key not recognised for this environment: a DEMO key only works with IG_ACC_TYPE=DEMO, a LIVE key with LIVE",
    "error.security.api-key-disabled": "API key is disabled: re-enable it on the IG web platform under My Account > Settings > API Keys",
    "error.security.api-key-revoked": "API key was revoked: generate a new one on the IG web platform under My Account > Settings > API Keys",
    "error.security.invalid-details": "username or password rejected: check IG_USERNAME / IG_PASSWORD, and that IG_ACC_TYPE matches the account (DEMO logins differ from LIVE)",
    "error.security.account-suspended": "IG account is suspended: email webapisupport@ig.com",
    "error.security.client-suspended": "IG has suspended this client login (not the API key, and not a timed lock): test the same login in IG's API companion (labs.ig.com/sample-apps/api-rest-companion-release/index.html), then email webapisupport@ig.com to lift it; a new password or API key does not clear it",
    "error.security.too-many-failed-attempts": "IG locked logins after failed attempts: wait about a minute, fix the password, then retry",
    "error.public-api.exceeded-api-key-allowance": "API key request allowance used up: wait for it to reset",
    "error.public-api.exceeded-account-historical-data-allowance": "weekly historical price allowance used up: use cached bars or Yahoo for long history",
    "error.security.oauth-token-invalid": "session expired: log in again",
}


def fix_hint(code: str) -> str | None:
    return next((h for c, h in FIX_HINTS.items() if c in str(code)), None)


class IG:
    def __init__(self, cfg: IGConfig | None = None):
        self.cfg = cfg or ig_config()
        if not self.cfg.configured:
            raise IGError("IG credentials missing from environment: " + ", ".join(self.cfg.missing()))
        self.s = session()
        self.s.headers.update({"Accept": "application/json; charset=UTF-8",
                               "Content-Type": "application/json; charset=UTF-8"})
        if self.cfg.api_key:  # otherwise a network secret adds X-IG-API-KEY at the proxy
            self.s.headers["X-IG-API-KEY"] = self.cfg.api_key
        self.account_id: str | None = None
        self.ls_endpoint: str | None = None
        self.last_allowance: dict | None = None
        self._logged_in = False

    # ---- transport -------------------------------------------------------
    def _request(self, method: str, path: str, version: int = 1, params=None, json=None, auth=True, _retry=True):
        method = method.upper()
        if method != "GET" and (method, path) not in _ALLOWED_WRITES:
            raise ReadOnlyViolation(f"{method} {path} blocked: market-lab's IG client is read-only")
        if auth and not self._logged_in:
            self.login()
        headers = {"Version": str(version)}
        if method == "DELETE":  # IG wants DELETE tunnelled through POST with _method header
            method, headers["_method"] = "POST", "DELETE"
        r = self.s.request(method, self.cfg.base_url + path, params=params, json=json, headers=headers, timeout=30)
        if r.status_code >= 400:
            try:
                code = r.json().get("errorCode", r.text)
            except Exception:
                code = r.text[:200]
            if auth and _retry and r.status_code == 401 and "token" in str(code).lower():
                self._logged_in = False  # expired session: log in once more and retry
                return self._request(headers.get("_method", method), path, version, params, json, auth, _retry=False)
            hint = fix_hint(code)
            raise IGError(f"IG {method} {path} -> {r.status_code}: {code}" + (f" (fix: {hint})" if hint else ""))
        return r

    def _get(self, path, version=1, **params):
        return self._request("GET", path, version, params={k: v for k, v in params.items() if v is not None}).json()

    # ---- session ---------------------------------------------------------
    def login(self) -> dict:
        r = self._request("POST", "/session", 2, auth=False, json={
            "identifier": self.cfg.username, "password": self.cfg.password, "encryptedPassword": False})
        self.s.headers["CST"] = r.headers["CST"]
        self.s.headers["X-SECURITY-TOKEN"] = r.headers["X-SECURITY-TOKEN"]
        body = r.json()
        self.account_id = body.get("currentAccountId")
        self.ls_endpoint = body.get("lightstreamerEndpoint")
        self._logged_in = True
        want = self.cfg.acc_number
        if want and want != self.account_id:
            self._request("PUT", "/session", 1, json={"accountId": want, "defaultAccount": False})
            self.account_id = want
        return {"account_id": self.account_id, "env": self.cfg.acc_type, "currency": body.get("currencyIsoCode"),
                "accounts": [a.get("accountId") for a in body.get("accounts", [])]}

    def logout(self):
        if self._logged_in:
            try:
                self._request("DELETE", "/session", 1)
            finally:
                self._logged_in = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.logout()

    # ---- account (read) --------------------------------------------------
    def accounts(self) -> pd.DataFrame:
        rows = []
        for a in self._get("/accounts")["accounts"]:
            b = a.get("balance") or {}
            rows.append({"accountId": a["accountId"], "name": a.get("accountName"), "type": a.get("accountType"),
                         "currency": a.get("currency"), "preferred": a.get("preferred"), **b})
        return pd.DataFrame(rows)

    def positions(self) -> pd.DataFrame:
        rows = []
        for p in self._get("/positions", 2)["positions"]:
            pos, m = p["position"], p["market"]
            sign = 1 if pos["direction"] == "BUY" else -1
            px_now = m.get("bid") if sign > 0 else m.get("offer")
            upl = None
            if px_now is not None and pos.get("level") is not None:
                upl = (px_now - pos["level"]) * sign * pos["size"] * (pos.get("contractSize") or 1)
            rows.append({"epic": m["epic"], "name": m["instrumentName"], "direction": pos["direction"],
                         "size": pos["size"], "level": pos["level"], "bid": m.get("bid"), "offer": m.get("offer"),
                         "stop": pos.get("stopLevel"), "limit": pos.get("limitLevel"), "currency": pos.get("currency"),
                         "upl_points_x_size": upl, "opened": pos.get("createdDateUTC"), "dealId": pos["dealId"],
                         "status": m.get("marketStatus")})
        return pd.DataFrame(rows)

    def working_orders(self) -> pd.DataFrame:
        rows = []
        for o in self._get("/workingorders", 2)["workingOrders"]:
            d, m = o["workingOrderData"], o["marketData"]
            rows.append({"epic": m["epic"], "name": m["instrumentName"], "direction": d["direction"],
                         "size": d.get("orderSize"), "level": d.get("orderLevel"), "type": d.get("orderType"),
                         "stop_distance": d.get("stopDistance"), "limit_distance": d.get("limitDistance"),
                         "good_till": d.get("goodTillDateISO"), "created": d.get("createdDateUTC")})
        return pd.DataFrame(rows)

    def activity(self, days: int = 30) -> pd.DataFrame:
        frm = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        return pd.DataFrame(self._get("/history/activity", 3, **{"from": frm, "detailed": "true", "pageSize": 500})["activities"])

    def transactions(self, days: int = 90, kind: str = "ALL") -> pd.DataFrame:
        frm = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
        return pd.DataFrame(self._get("/history/transactions", 2, type=kind, **{"from": frm, "pageSize": 0})["transactions"])

    # ---- markets (read) --------------------------------------------------
    def watchlists(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("/watchlists")["watchlists"])

    def watchlist(self, watchlist_id: str) -> pd.DataFrame:
        return pd.DataFrame(self._get(f"/watchlists/{watchlist_id}")["markets"])

    def search(self, term: str) -> pd.DataFrame:
        df = pd.DataFrame(self._get("/markets", searchTerm=term)["markets"])
        keep = [c for c in ["epic", "instrumentName", "instrumentType", "expiry", "bid", "offer",
                            "percentageChange", "marketStatus"] if c in df]
        return df[keep] if not df.empty else df

    def market(self, epic: str) -> dict:
        return self._get(f"/markets/{epic}", 3)

    def snapshot(self, epics: list[str]) -> pd.DataFrame:
        j = self._get("/markets", 2, epics=",".join(epics), filter="SNAPSHOT_ONLY")
        rows = [{"epic": d["instrument"]["epic"], "name": d["instrument"]["name"], **d["snapshot"]}
                for d in j.get("marketDetails", [])]
        return pd.DataFrame(rows)

    def sentiment(self, epic_or_market_id: str) -> dict:
        """IG client sentiment (% of IG clients long/short): a crowd-positioning, contrarian input."""
        mid = epic_or_market_id
        if "." in mid:  # an epic: resolve its marketId
            mid = self.market(mid)["instrument"]["marketId"]
        return self._get(f"/clientsentiment/{mid}")

    # ---- prices (read, cached) -------------------------------------------
    def _fetch_prices(self, epic: str, resolution: str, start: datetime, end: datetime) -> pd.DataFrame:
        rows, page = [], 1
        while True:
            j = self._get(f"/prices/{epic}", 3, resolution=resolution, pageSize=1000, pageNumber=page,
                          **{"from": start.strftime("%Y-%m-%dT%H:%M:%S"), "to": end.strftime("%Y-%m-%dT%H:%M:%S")})
            rows += j.get("prices", [])
            md = j.get("metadata", {})
            self.last_allowance = md.get("allowance")
            pd_ = md.get("pageData") or {}
            if page >= (pd_.get("totalPages") or 1):
                break
            page += 1
        return prices_to_frame(rows)

    def prices(self, epic: str, interval: str = "1d", start=None, end=None, use_cache: bool = True) -> pd.DataFrame:
        if interval not in RESOLUTIONS:
            raise ValueError(f"interval must be one of {list(RESOLUTIONS)}")
        end_ts = pd.Timestamp(end, tz="UTC") if end is not None else pd.Timestamp.now(tz="UTC")
        start_ts = pd.Timestamp(start, tz="UTC") if start is not None else end_ts - pd.Timedelta(days=365)
        prov = f"ig-{self.cfg.acc_type.lower()}"
        cached = cache.load(prov, epic, interval) if use_cache else None
        fetch_from = start_ts
        if cached is not None and not cached.empty and cached.index[0] <= start_ts + pd.Timedelta(days=4):
            fetch_from = max(start_ts, cached.index[-1])  # only the missing tail
        if fetch_from < end_ts - pd.Timedelta(BAR.get(interval, "1D")):
            fresh = self._fetch_prices(epic, RESOLUTIONS[interval], fetch_from.to_pydatetime(), end_ts.to_pydatetime())
            cached = cache.save(prov, epic, interval, fresh) if use_cache else fresh
        if cached is None or cached.empty:
            raise LookupError(f"IG returned no prices for {epic}")
        return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]

    # ---- streaming (read) ------------------------------------------------
    def price_items(self, epics: list[str]) -> list[str]:
        """Lightstreamer PRICE items. IG retired the old MARKET:{epic} items on 8 May 2026."""
        return [f"PRICE:{self.account_id}:{e}" for e in epics]

    def stream(self, epics: list[str], seconds: int = 30,
               fields=("BIDPRICE1", "ASKPRICE1", "HIGH", "LOW", "NET_CHG_PCT", "DLG_FLAG", "DELAY", "TIMESTAMP"),
               on_update=None) -> list[dict]:
        """Subscribe to live Lightstreamer quotes for `seconds`; returns collected ticks."""
        from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener

        if not self._logged_in:
            self.login()
        ticks, lock = [], threading.Lock()

        class L(SubscriptionListener):
            def onItemUpdate(self, u):
                t = {"epic": u.getItemName().split(":", 2)[2], **{f: u.getValue(f) for f in fields},
                     "recv": datetime.now(timezone.utc).isoformat()}
                with lock:
                    ticks.append(t)
                if on_update:
                    on_update(t)

        client = LightstreamerClient(self.ls_endpoint, None)
        client.connectionDetails.setUser(self.account_id)
        client.connectionDetails.setPassword(f"CST-{self.s.headers['CST']}|XST-{self.s.headers['X-SECURITY-TOKEN']}")
        client.connect()
        sub = Subscription(mode="MERGE", items=self.price_items(epics), fields=list(fields))
        sub.addListener(L())
        client.subscribe(sub)
        try:
            time.sleep(seconds)
        finally:
            client.unsubscribe(sub)
            client.disconnect()
        return ticks


def prices_to_frame(rows: list[dict]) -> pd.DataFrame:
    """IG v3 price rows -> OHLCV on mid prices, plus bid/ask close and spread."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def mid(p):
        b, a = p.get("bid"), p.get("ask")
        if b is not None and a is not None:
            return (b + a) / 2
        return p.get("lastTraded") if p.get("lastTraded") is not None else (b if b is not None else a)

    recs = []
    for r in rows:
        t = (r.get("snapshotTimeUTC") or r["snapshotTime"]).replace("/", "-")
        recs.append({"time": pd.Timestamp(t).tz_localize("UTC"),
                     "open": mid(r["openPrice"]), "high": mid(r["highPrice"]), "low": mid(r["lowPrice"]),
                     "close": mid(r["closePrice"]), "volume": r.get("lastTradedVolume"),
                     "bid_close": r["closePrice"].get("bid"), "ask_close": r["closePrice"].get("ask")})
    df = pd.DataFrame(recs).set_index("time").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["spread"] = df["ask_close"] - df["bid_close"]
    return df.astype(float)
