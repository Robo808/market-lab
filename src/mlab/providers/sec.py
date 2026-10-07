"""SEC EDGAR: ticker->CIK, XBRL company facts (10 years of audited annual numbers), filings list,
insider (Form 4) filings. Free and keyless; SEC asks for a descriptive User-Agent (SEC_USER_AGENT)."""
from __future__ import annotations

import pandas as pd

from ..cache import TTLMemo
from ..config import SEC_USER_AGENT
from ..net import session

_memo = TTLMemo(ttl=3600)

# First tag present wins. Order matters: newer ASC 606 revenue tag before legacy ones.
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "dna": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "st_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "goodwill": ["Goodwill"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
FLOW = {"revenue", "gross_profit", "operating_income", "net_income", "cfo", "capex", "sbc", "dna",
        "interest_expense", "diluted_shares", "buybacks", "dividends"}


def _get(url: str) -> dict:
    hit = _memo.get(url)
    if hit is not None:
        return hit
    r = session(SEC_USER_AGENT).get(url, timeout=30)
    r.raise_for_status()
    return _memo.put(url, r.json())


def cik(ticker: str) -> str:
    data = _get("https://www.sec.gov/files/company_tickers.json")
    t = ticker.upper().replace("-", ".")
    for row in data.values():
        if row["ticker"].upper() in (t, t.replace(".", "-")):
            return f"{int(row['cik_str']):010d}"
    raise LookupError(f"{ticker} not found in SEC ticker map (US filers only)")


def company_facts(ticker: str) -> dict:
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik(ticker)}.json")


def annual_financials(ticker: str, years: int = 10) -> pd.DataFrame:
    """Rows = fiscal year end, columns = standardized line items (USD, shares as reported)."""
    facts = company_facts(ticker).get("facts", {}).get("us-gaap", {})
    out = {}
    for item, tags in TAGS.items():
        for tag in tags:
            if tag not in facts:
                continue
            units = facts[tag]["units"]
            vals = units.get("USD") or units.get("shares") or next(iter(units.values()))
            df = pd.DataFrame(vals)
            df = df[df["form"].isin(["10-K", "10-K/A", "20-F", "40-F"])]
            if df.empty:
                continue
            df["end"] = pd.to_datetime(df["end"])
            if item in FLOW and "start" in df:
                dur = (df["end"] - pd.to_datetime(df["start"])).dt.days
                df = df[(dur > 330) & (dur < 400)]
            df = df.sort_values("filed").drop_duplicates("end", keep="last")
            ser = df.set_index("end")["val"].astype(float)
            out[item] = ser if item not in out else out[item].combine_first(ser)
            break
    if not out:
        raise LookupError(f"no XBRL annual facts for {ticker}")
    fin = pd.DataFrame(out).sort_index()
    fin = fin[fin.index >= fin.index.max() - pd.DateOffset(years=years)]
    # Derived lines every lens uses.
    if {"cfo", "capex"} <= set(fin):
        fin["fcf"] = fin["cfo"] - fin["capex"]
        if "sbc" in fin:
            fin["fcf_ex_sbc"] = fin["fcf"] - fin["sbc"]
    if {"gross_profit", "revenue"} <= set(fin):
        fin["gross_margin"] = fin["gross_profit"] / fin["revenue"]
    if {"operating_income", "revenue"} <= set(fin):
        fin["op_margin"] = fin["operating_income"] / fin["revenue"]
    if {"net_income", "revenue"} <= set(fin):
        fin["net_margin"] = fin["net_income"] / fin["revenue"]
    if "lt_debt" in fin or "st_debt" in fin:
        zero = pd.Series(0.0, index=fin.index)
        fin["total_debt"] = fin.get("lt_debt", zero).fillna(0) + fin.get("st_debt", zero).fillna(0)
        if "cash" in fin:
            fin["net_debt"] = fin["total_debt"] - fin["cash"]
    if {"operating_income", "equity"} <= set(fin) and "total_debt" in fin:
        invested = fin["equity"] + fin["total_debt"] - fin.get("cash", 0)
        fin["roic_pre_tax"] = fin["operating_income"] / invested.where(invested > 0)
    if {"net_income", "equity"} <= set(fin):
        fin["roe"] = fin["net_income"] / fin["equity"].where(fin["equity"] > 0)
    if {"current_assets", "current_liabilities"} <= set(fin):
        fin["current_ratio"] = fin["current_assets"] / fin["current_liabilities"]
    if {"operating_income", "interest_expense"} <= set(fin):
        fin["interest_cover"] = fin["operating_income"] / fin["interest_expense"].where(fin["interest_expense"] > 0)
    return fin


def filings(ticker: str, forms: tuple[str, ...] | None = None, limit: int = 40) -> pd.DataFrame:
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik(ticker)}.json")
    rec = pd.DataFrame(sub["filings"]["recent"])
    if forms:
        rec = rec[rec["form"].isin(forms)]
    c = int(cik(ticker))
    rec["url"] = rec.apply(lambda r: f"https://www.sec.gov/Archives/edgar/data/{c}/"
                           f"{r['accessionNumber'].replace('-', '')}/{r['primaryDocument']}", axis=1)
    return rec[["filingDate", "form", "reportDate", "primaryDocDescription", "url"]].head(limit)


def insider_filings(ticker: str, limit: int = 40) -> pd.DataFrame:
    return filings(ticker, forms=("4", "4/A"), limit=limit)
