"""Quant research layer: algorithmic strategy library, portfolio backtester, validation
(walk-forward, PSR/DSR, bootstrap, permutation), sizing, paper books and pre-registered hypotheses.

Strategies run in three modes only: backtest, signal (today's target) and paper. There is no live
execution path: the IG client stays read-only.
"""
from .strategies import REGISTRY, Strategy, get  # noqa: F401
