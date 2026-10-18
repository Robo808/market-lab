"""Phase-space research: every instrument at every bar is a point in indicator space.

features  point-in-time feature cube (symbol x time x feature), schema contract, look-ahead checker
regimes   Gaussian HMM regimes with forward-filtered (never smoothed) probabilities, 3D phase-space view
evaluate  purged + embargoed walk-forward, costs, deflated Sharpe, PBO, null stubs, skill vs horizon,
          append-only experiment log that counts every trial
physics   OU restoring force vs momentum per regime, permutation entropy, spectral check, Monte Carlo
"""
