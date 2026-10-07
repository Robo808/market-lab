"""Named symbol lists for scans. Yahoo symbols unless prefixed (crypto:, ig:)."""
UNIVERSES = {
    "indices": ["^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI", "^VIX"],
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X", "GBPJPY=X", "DX-Y.NYB"],
    "commodities": ["GC=F", "SI=F", "HG=F", "PL=F", "CL=F", "BZ=F", "NG=F", "ZW=F", "ZC=F", "ZS=F"],
    "rates": ["^IRX", "^FVX", "^TNX", "^TYX", "TLT", "IEF", "SHY", "HYG", "LQD"],
    "us_megacaps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY", "V", "XOM", "UNH", "COST"],
    "sectors": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "XBI"],
    "uk_ftse": ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L", "GSK.L", "BATS.L", "LSEG.L", "BARC.L", "LLOY.L", "RR.L", "GLEN.L", "VOD.L", "NWG.L"],
    "crypto": ["crypto:BTC", "crypto:ETH", "crypto:SOL", "crypto:XRP", "crypto:BNB", "crypto:DOGE", "crypto:ADA", "crypto:LINK"],
    "macro_board": ["^GSPC", "^NDX", "^FTSE", "^STOXX50E", "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F", "HG=F", "^TNX", "BTC-USD", "^VIX"],
}

# Handy IG epics (verify with `mlab ig search`; epics differ between spread bet and CFD accounts).
IG_COMMON_EPICS = {
    "FTSE 100 (DFB)": "IX.D.FTSE.DAILY.IP", "US 500 (DFB)": "IX.D.SPTRD.DAILY.IP", "US Tech 100 (DFB)": "IX.D.NASDAQ.IFD.IP",
    "Wall Street (DFB)": "IX.D.DOW.DAILY.IP", "Germany 40 (DFB)": "IX.D.DAX.DAILY.IP", "EUR/USD (CFD)": "CS.D.EURUSD.CFD.IP",
    "GBP/USD (CFD)": "CS.D.GBPUSD.CFD.IP", "USD/JPY (CFD)": "CS.D.USDJPY.CFD.IP", "Spot Gold (CFD)": "CS.D.CFDGOLD.CFDGC.IP",
    "Bitcoin (CFD)": "CS.D.BITCOIN.CFD.IP",
}
