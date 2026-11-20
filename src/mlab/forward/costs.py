"""Per-venue cost models for paper fills. Every number carries its source; `assumption` means unverified.

A fill's cost is notional x (spread + commission + FX + stamp on buys) in basis points, charged per side.
Holding a spread bet or CFD costs overnight funding on the notional per calendar day. Shorts on cash-share
venues (an ISA cannot short) route to the matching spread-bet venue, so a long/short cell pays both.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

VERIFIED_FX = "LESSONS L-019: broker costs page read 2026-10-21, 0.7% FX each way on US shares in a UK ISA"
VERIFIED_STAMP = "LESSONS L-019: UK stamp duty 0.5% on share purchases"
FUNDING_ASSUMED = "assumption: broker admin fee over the reference rate, unverified until issue #24 lands"
REF_RATE = 0.04  # assumption: sterling/dollar reference rate for funding, refresh when #24 lands


@dataclass(frozen=True)
class Venue:
    name: str
    spread_bps: float          # half the quoted spread, paid per side
    commission_bps: float = 0.0
    fx_bps: float = 0.0        # per side
    stamp_buy_bps: float = 0.0
    funding_long: float = 0.0  # annual, on notional, long side
    funding_short: float = 0.0  # annual, on notional, short side (negative = receives)
    source: str = "assumption"

    def side_bps(self, buying: bool) -> float:
        return self.spread_bps + self.commission_bps + self.fx_bps + (self.stamp_buy_bps if buying else 0.0)


def _sb(name, spread_bps, admin, source_spread):
    return Venue(name, spread_bps, funding_long=REF_RATE + admin, funding_short=admin - REF_RATE,
                 source=f"spread: {source_spread}; funding: {FUNDING_ASSUMED}")


VENUES: dict[str, Venue] = {v.name: v for v in [
    Venue("isa_us_share", 2.0, fx_bps=70.0, source=f"FX: {VERIFIED_FX}; spread: assumption 2bp"),
    Venue("isa_us_etf", 1.0, fx_bps=70.0, source=f"FX: {VERIFIED_FX}; spread: assumption 1bp"),
    Venue("isa_uk_share", 5.0, stamp_buy_bps=50.0, source=f"stamp: {VERIFIED_STAMP}; spread: assumption 5bp"),
    Venue("isa_uk_etf", 5.0, source="no stamp on ETFs; spread: assumption 5bp"),
    _sb("spreadbet_index", 1.0, 0.025, "assumption 1bp on major index DFBs"),
    _sb("spreadbet_share", 10.0, 0.025, "assumption 10bp on single-share DFBs"),
    _sb("spreadbet_fx", 0.6, 0.01, "assumption 0.6bp on majors"),
    Venue("spreadbet_commodity", 3.0, source="assumption 3bp; futures-based bets carry roll in price, no funding"),
    Venue("crypto_spot", 25.0, source="assumption 25bp taker fee + spread on a UK spot exchange"),
    _sb("spreadbet_crypto", 50.0, 0.05, "assumption 50bp on crypto CFDs/bets"),
]}

# Options: per-contract commission to open, none to close (IG US options account via tastytrade,
# broker costs page dated 2026-09-08, read 2026-10-10). Fills cross the bid/ask.
OPTION_COMMISSION_OPEN = 1.0
OPTION_COMMISSION_CLOSE = 0.0
OPTION_COST_SOURCE = "IG US options account $1/contract open, $0 close (costs page dated 2026-09-08); fills at bid/ask"

US_ETFS = {"SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "SLV", "USO", "BIL", "HYG", "LQD", "XLK",
           "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "XBI", "EWA", "EWC",
           "VNQ", "DBC", "UUP", "VGK", "EWJ", "FXI", "IBIT"}
UK_ETFS = {"VWRP.L", "ISF.L", "SGLN.L", "VUSA.L", "CSP1.L", "IITU.L", "VUKE.L", "IGLT.L", "CNDX.L"}


def asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s.startswith("CRYPTO:") or s.endswith("-USD"):
        return "crypto"
    if s.endswith("=X") or s == "DX-Y.NYB":
        return "fx"
    if s.endswith("=F"):
        return "commodity"
    if s.startswith("^"):
        return "index"
    if s in UK_ETFS:
        return "uk_etf"
    if re.search(r"\.L$", s):
        return "uk_share"
    if s in US_ETFS:
        return "us_etf"
    return "us_share"


_LONG = {"crypto": "crypto_spot", "fx": "spreadbet_fx", "commodity": "spreadbet_commodity", "index": "spreadbet_index",
         "uk_etf": "isa_uk_etf", "uk_share": "isa_uk_share", "us_etf": "isa_us_etf", "us_share": "isa_us_share"}
_SHORT = {"crypto": "spreadbet_crypto", "fx": "spreadbet_fx", "commodity": "spreadbet_commodity",
          "index": "spreadbet_index", "uk_etf": "spreadbet_share", "uk_share": "spreadbet_share",
          "us_etf": "spreadbet_share", "us_share": "spreadbet_share"}


def venue_for(symbol: str, side: int) -> Venue:
    """Cheapest default UK venue per asset class and direction (side +1 long, -1 short)."""
    ac = asset_class(symbol)
    return VENUES[(_LONG if side >= 0 else _SHORT)[ac]]


def fill_cost(symbol: str, w_from: float, w_to: float, equity: float) -> tuple[float, list[str]]:
    """Cost of moving a position from weight w_from to w_to (fractions of equity). A flip through zero pays
    the close on one venue and the open on the other. Returns (cost in equity units, venue names used)."""
    cost, used = 0.0, []
    legs = []
    if w_from * w_to < 0:                      # flip: close then open
        legs = [(w_from, 0.0), (0.0, w_to)]
    else:
        legs = [(w_from, w_to)]
    for a, b in legs:
        if a == b:
            continue
        side = 1 if (b if b != 0 else a) > 0 else -1
        v = venue_for(symbol, side)
        buying = b > a
        cost += abs(b - a) * equity * v.side_bps(buying) / 1e4
        used.append(v.name)
    return cost, used


def funding_cost(symbol: str, w: float, equity: float, days: float) -> float:
    if w == 0 or days <= 0:
        return 0.0
    v = venue_for(symbol, 1 if w > 0 else -1)
    rate = v.funding_long if w > 0 else v.funding_short
    return abs(w) * equity * rate * days / 365.0


def round_trip_bps(symbol: str, side: int = 1) -> float:
    v = venue_for(symbol, side)
    return v.side_bps(True) + v.side_bps(False)


def table() -> list[dict]:
    return [asdict(v) for v in VENUES.values()]
