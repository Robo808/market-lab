# Supply and demand vs industry returns

`src/mlab/econ/data/supply_demand.yaml` maps 22 product categories to a **supply** series (Federal Reserve G.17
industrial production, plus capacity utilisation where published), a **consumer-demand** series (Census monthly
retail sales by kind of business) and the **industry** that earns from them (Ken French 49 industry portfolios for
history, a US ETF for live tracking). Categories link to `sig:` nodes in the [[Econ Graph]].

```bash
mlab econ supply-demand              # the map, and where each category stands on its latest published month
mlab econ sd-study --split train     # explain / predict / rotation tables (1993-2014)
mlab econ sd-study --split validation
mlab econ sd-study --split test --allow-test   # locked 2020+ split: only for a pre-registered hypothesis
```

## Features (monthly)

| name | meaning |
|---|---|
| `s_yoy` | supply volume growth, 12-month log change, % |
| `d_yoy` | demand growth (nominal retail sales), % |
| `gap` | `d_yoy - s_yoy`: demand outrunning supply (includes price, on purpose) |
| `*_acc` | change in the growth rate over 3 months |
| `util_chg` | 12-month change in capacity utilisation, points |

Publication lag per category: the longest `release_lag_months` of its linked `sig:` nodes (1 month for every G.17
and MRTS series: both land mid-month for the previous month). Latest vintage only, so revisions flatter in-sample
fits. The series are fetched through `providers/publishers.py` (`fed_g17`, `census_mrts`), so `mlab econ panel`
and `mlab econ check` use them before the FRED mirrors.

## Three questions, three tests

1. **Explain**: 12-month industry excess return against same-window growth, plus the reverse, past 12-month return
   against the next 12 months of growth (does price move first?). Newey-West t, 12 lags.
2. **Predict**: feature known at month end against the excess return over the next 1, 3, 6, 12 months.
   Newey-West t, plus `t_nonoverlap` (median over phases of the OLS t on non-overlapping h-month steps, which
   cannot be inflated by overlap) and `n_indep` (independent observations). Null stubs are the same feature
   circularly shifted by random offsets (>= 24 months), which keeps its persistence: `p_null` = share of shifts
   with a |t| at least as large (3,000 shifts where |t| >= 3, so the 1/(n+1) floor does not bind); `null_t3` = how
   often a shifted feature clears t >= 3 (the bar's false-positive rate). `q_bh` = Benjamini-Hochberg over the
   normal two-sided p of every test in the split. In the 60-month validation window the circular shift has only
   about a dozen distinct offsets, so read `t_nonoverlap` there, not `p_null`.
3. **Rotation**: each month long the top third / short the bottom third of categories by a feature, one month,
   against random rankings with the same book shape.

## First results (fetched 22 Oct 2026; train 1993-2014, validation 2015-2019; test 2020+ untouched)

- **Explain: strong at the level of the whole economy.** The market's 12-month return correlates 0.62 (train) and
  0.68 (validation) with same-window retail sales growth, and 0.67 / 0.50 with industrial production growth.
  70% of the category links keep their sign out of sample; median |corr| 0.21.
- **Price tends to move first.** The past 12-month market return correlates 0.33 with the next 12 months of
  industrial production growth (t 2.25, train), in line with Fama (1990).
- **Predict: close to chance.** 13 of 400 tests clear |t| >= 3 in train against about 9 expected from the null
  rate (2.2%). On non-overlapping steps none of the 13 clears 3 (largest 2.96, fuel gap at 3 months), and none
  keeps its size in validation.
- **Machinery, the one that looked like it held:** output growth against the machinery industry's next 12-month
  excess return has a negative sign at every horizon in both windows, but it is weak. Train t is -0.8 / -1.0 / -1.7
  / -3.2 at 1 / 3 / 6 / 12 months, and only -1.6 on non-overlapping 12-month steps. Validation's -5.9 is -3.1 on
  non-overlapping steps over 5 independent years. That is too little power to spend the locked test split on as a
  single series; a cross-sectional version (all supply-only categories, hot output growth against 12-month excess
  return) would carry more observations.
- **Rotation: nothing.** No feature ranks categories with |t| above 1.6 in either window.

Reading: published monthly supply and demand data line up with returns but arrive after prices have moved. The
next step is faster proxies of the same quantities (company guidance, freight, card spending, chip billings) and
the influence-map lead-lag work.

Sources: Board of Governors of the Federal Reserve System, G.17 (public domain); U.S. Census Bureau, Monthly Retail
Trade Survey; Kenneth R. French Data Library.
