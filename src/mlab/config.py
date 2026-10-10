"""Paths and settings. Everything secret comes from environment variables only."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("MLAB_HOME", Path(__file__).resolve().parents[2]))

# Where persistent desk state lives (journal, reports, caches, IV history, transcripts).
# The code can run from an ephemeral repo clone; this state must outlive the container, so it
# defaults to the project's shared folder when one exists, else to the checkout itself.
SHARED_DATA_DIR = Path("/mnt/project-files/market-lab")


def _default_data_dir() -> Path:
    return SHARED_DATA_DIR if SHARED_DATA_DIR.is_dir() else ROOT


DATA_DIR = Path(os.environ.get("MLAB_DATA_DIR") or _default_data_dir())
CACHE_DIR = Path(os.environ.get("MLAB_CACHE_DIR", DATA_DIR / "data" / "cache"))
REPORTS_DIR = Path(os.environ.get("MLAB_REPORTS_DIR", DATA_DIR / "reports"))
# Soft cap for the on-disk price cache; oldest-read files are pruned beyond it.
CACHE_MAX_MB = int(os.environ.get("MLAB_CACHE_MAX_MB", "750"))

# SEC asks every client to identify itself with a contact address.
# SEC 403s a User-Agent without an email address; set SEC_USER_AGENT to your own "Name email".
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "market-lab research ops@market-lab.invalid")


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
    # True when username and password are absent for the active environment (IG_ACC_TYPE set): a
    # "Body parameter" network secret on the IG host fills `identifier` and `password` into the login
    # body at the proxy, so neither enters the container.
    login_via_secret: bool = False

    @property
    def base_url(self) -> str:
        return (
            "https://api.ig.com/gateway/deal"
            if self.acc_type == "LIVE"
            else "https://demo-api.ig.com/gateway/deal"
        )

    @property
    def configured(self) -> bool:
        # IG_API_KEY is optional: a network secret can attach the X-IG-API-KEY header at the proxy instead.
        # Username and password come from the environment, or from a body-parameter network secret.
        return bool(self.username and self.password) or self.login_via_secret

    def missing(self) -> list[str]:
        if self.login_via_secret:
            return []
        return [n for n, v in (("IG_USERNAME", self.username), ("IG_PASSWORD", self.password)) if not v]


def ig_config(acc_type: str | None = None) -> IGConfig:
    t = (acc_type or env("IG_ACC_TYPE", "DEMO")).upper()
    if t not in ("DEMO", "LIVE"):
        raise ValueError("IG_ACC_TYPE must be DEMO or LIVE")
    # Separate DEMO/LIVE credentials are supported via IG_DEMO_* / IG_LIVE_* overrides.
    def pick(key: str) -> str | None:
        return env(f"IG_{t}_{key}") or env(f"IG_{key}")
    user, pw = pick("USERNAME"), pick("PASSWORD")
    via_secret = not user and not pw and (env("IG_ACC_TYPE") or "").upper() == t
    return IGConfig(pick("API_KEY"), user, pw, t, pick("ACC_NUMBER"), via_secret)


OPTIONAL_KEYS = {
    "FRED_API_KEY": "FRED macro series via the official API (falls back to the keyless CSV endpoint)",
    "ALPHAVANTAGE_API_KEY": "Alpha Vantage fallback prices/fundamentals",
    "FMP_API_KEY": "Financial Modeling Prep fundamentals",
    "MASSIVE_API_KEY": "Massive (Polygon) free key: live backup for US stocks and `mlab datasets crosscheck` (display-only, never stored)",
    "SEC_USER_AGENT": "Contact string SEC requires, e.g. 'Your Name you@example.com'",
}
