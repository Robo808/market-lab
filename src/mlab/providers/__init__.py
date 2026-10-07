"""Data providers. Every price provider returns a DataFrame indexed by UTC timestamp with
columns open, high, low, close, volume (volume may be NaN for FX/CFD)."""
OHLCV = ["open", "high", "low", "close", "volume"]
