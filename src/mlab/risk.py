"""Position sizing and the trade card. Sizing works for IG spread bets (stake per point) and CFDs
(contracts): you give the account size, % risk and stop; it returns the size that loses exactly
that amount if the stop is hit (before slippage/gaps)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def stake_per_point(equity: float, risk_pct: float, entry: float, stop: float, point_size: float = 1.0) -> dict:
    """point_size: price move that equals 1 IG point (1 for indices/shares in pence, 0.0001 for EURUSD, 0.01 for USDJPY)."""
    risk_amt = equity * risk_pct / 100
    dist_pts = abs(entry - stop) / point_size
    if dist_pts == 0:
        raise ValueError("stop equals entry")
    return {"risk_amount": round(risk_amt, 2), "stop_distance_points": round(dist_pts, 2),
            "stake_per_point": round(risk_amt / dist_pts, 2)}


def atr_stop(entry: float, atr: float, direction: str = "long", mult: float = 2.0) -> float:
    return entry - mult * atr if direction == "long" else entry + mult * atr


def kelly_fraction(win_rate: float, payoff: float, fraction: float = 0.25) -> float:
    """Fractional Kelly (default quarter Kelly). payoff = avg win / avg loss."""
    k = win_rate - (1 - win_rate) / payoff
    return max(0.0, k * fraction)


def reward_risk(entry: float, stop: float, targets: list[float]) -> list[float]:
    risk = abs(entry - stop)
    return [round(abs(t - entry) / risk, 2) for t in targets] if risk else []


@dataclass
class TradeCard:
    """The standard card every trade idea in this project ends with."""
    instrument: str
    bias: str                      # LONG / SHORT / NO TRADE
    entry_zone: tuple[float, float]
    stop: float
    targets: list[float]
    risk_pct: float = 1.0
    horizon: str = "swing (2-8 weeks)"
    conviction: int = 3            # 1-5
    catalysts: list[str] = field(default_factory=list)
    kills_it: str = ""
    epic: str | None = None
    lens: str = ""                 # Soros / Buffett / Burry / TA
    thesis: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def entry_mid(self) -> float:
        return float(np.mean(self.entry_zone))

    @property
    def rr(self) -> list[float]:
        return reward_risk(self.entry_mid, self.stop, self.targets)

    def sizing(self, equity: float, point_size: float = 1.0) -> dict:
        return stake_per_point(equity, self.risk_pct, self.entry_mid, self.stop, point_size)

    def line(self) -> str:
        ez = f"{self.entry_zone[0]:g}-{self.entry_zone[1]:g}" if self.entry_zone[0] != self.entry_zone[1] else f"{self.entry_zone[0]:g}"
        inst = f"{self.instrument} ({self.epic})" if self.epic else self.instrument
        rr = "/".join(f"{x:g}" for x in self.rr)
        return (f"{self.bias} {inst} | Entry {ez} | Stop {self.stop:g} | Targets {', '.join(f'{t:g}' for t in self.targets)} | "
                f"R:R {rr} | Risk {self.risk_pct:g}% | {self.horizon} | Conviction {self.conviction}/5 | "
                f"Catalysts: {'; '.join(self.catalysts) or 'none scheduled'} | Kills it: {self.kills_it or 'stop hit'}")

    def markdown(self, equity: float | None = None, point_size: float = 1.0) -> str:
        rows = [
            ("Bias / instrument", f"**{self.bias}** {self.instrument}" + (f" (`{self.epic}`)" if self.epic else "")),
            ("Entry zone", f"{self.entry_zone[0]:g} – {self.entry_zone[1]:g}"),
            ("Stop / invalidation", f"{self.stop:g}"),
            ("Targets", ", ".join(f"{t:g}" for t in self.targets)),
            ("Reward:risk", " / ".join(f"{x:g}R" for x in self.rr)),
            ("Risk per trade", f"{self.risk_pct:g}% of account"),
            ("Horizon", self.horizon), ("Conviction", f"{self.conviction}/5"),
            ("Catalysts", "; ".join(self.catalysts) or "none scheduled"),
            ("What kills it", self.kills_it or "stop hit"),
        ]
        if equity:
            z = self.sizing(equity, point_size)
            rows.append(("Size", f"£{z['stake_per_point']}/pt over {z['stop_distance_points']} pts = £{z['risk_amount']} at risk"))
        md = "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
        if self.thesis:
            md = f"**Thesis ({self.lens or 'mixed'}):** {self.thesis}\n\n" + md
        if self.sources:
            md += "\n\nSources: " + "; ".join(self.sources)
        return md

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rr"] = self.rr
        return d
