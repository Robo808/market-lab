"""Paths and settings. Everything secret comes from environment variables only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("MLAB_HOME", Path(__file__).resolve().parents[2]))
CACHE_DIR = Path(os.environ.get("MLAB_CACHE_DIR", ROOT / "data" / "cache"))
REPORTS_DIR = Path(os.environ.get("MLAB_REPORTS_DIR", ROOT / "reports"))
# Soft cap for the on-disk price cache; oldest-read files are pruned beyond it.
CACHE_MAX_MB = int(os.environ.get("MLAB_CACHE_MAX_MB", "750"))

# SEC asks every client to identify itself with a contact address.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "market-lab research (set SEC_USER_AGENT)")


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class IGConfig:
    api_key: str | None
    username: str | None
    password: str | None
    acc_type: str  # DEMO or LIVE
    acc_number: str | None

    @property
    def base_url(self) -> str:
        return (
            "https://api.ig.com/gateway/deal"
            if self.acc_type == "LIVE"
            else "https://demo-api.ig.com/gateway/deal"
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.username and self.password)

    def missing(self) -> list[str]:
        return [n for n, v in (("IG_API_KEY", self.api_key), ("IG_USERNAME", self.username),
                               ("IG_PASSWORD", self.password)) if not v]


def ig_config(acc_type: str | None = None) -> IGConfig:
    t = (acc_type or env("IG_ACC_TYPE", "DEMO")).upper()
    if t not in ("DEMO", "LIVE"):
        raise ValueError("IG_ACC_TYPE must be DEMO or LIVE")
    # Separate DEMO/LIVE credentials are supported via IG_DEMO_* / IG_LIVE_* overrides.
    def pick(key: str) -> str | None:
        return env(f"IG_{t}_{key}") or env(f"IG_{key}")
    return IGConfig(pick("API_KEY"), pick("USERNAME"), pick("PASSWORD"), t, pick("ACC_NUMBER"))


OPTIONAL_KEYS = {
    "FRED_API_KEY": "FRED macro series via the official API (falls back to the keyless CSV endpoint)",
    "ALPHAVANTAGE_API_KEY": "Alpha Vantage fallback prices/fundamentals",
    "FMP_API_KEY": "Financial Modeling Prep fundamentals",
    "SEC_USER_AGENT": "Contact string SEC requires, e.g. 'Your Name you@example.com'",
}
