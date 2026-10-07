---
name: lens-burry
description: Frame a contrarian thesis Burry-style - hated or mispriced assets, balance-sheet forensics, asymmetric payoff, and the short side of bubbles. Use on top of fundamentals-desk, news-sentiment-desk and options-desk output.
---
# Burry lens

Inputs: `./mlab lens burry TICKER`, `./mlab dd TICKER`, `./mlab sentiment TICKER`, `./mlab options TICKER`, IG client sentiment.
Long side: deep value the crowd hates (EV/EBIT <= 8, FCF yield >= 10%, net cash, >= 40% off highs, insiders buying),
balance sheet that survives, a catalyst (buyback, asset sale, activist, cycle turn).
Short side: bubbles and fragility (valuation vs cash flow, SBC-inflated FCF, receivables growing faster than sales,
debt walls, euphoric sentiment, crowded longs), with a catalyst and a defined-risk structure (puts / stop above structure).
Always: what does the 10-K say that the narrative ignores? Read the filings (URLs from `./mlab dd`).
Output: the mispricing, the forensic evidence, the asymmetry (downside vs upside in numbers), the catalyst, then the trade card.
