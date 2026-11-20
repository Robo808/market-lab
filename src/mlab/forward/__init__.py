"""Forward paper-testing lab: pre-registered strategy cells run forward on real bars, paper only.

Backtests shortlist; the forward record decides. Every cell (strategy x params x instrument or basket x venue)
is frozen with a SHA-256 before its first signal, every bar's signal is logged (including triggers the desk
filters would skip), fills happen at the next bar's open with per-venue costs, and option structures are
marked daily from the live chain with their Greeks. Nothing here can place an order.

Modules: costs (venue cost models), store (append-only write-once event files), specs (registry and
freezing), context (regime tags), linear (strategy books), optsim (option books), score (pass bars that
scale with the number of cells), cli (`mlab fwd ...`).
"""
SCHEMA_VERSION = 1
