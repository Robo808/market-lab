# Econ Graph

One machine-readable dictionary of the numbers recorded about the economy, from GDP down to garden
ornament imports and an online bathroom retailer's average order value, wired as a typed graph. Use it to
find every source for a number, walk from a macro driver to the industries and companies it moves, and
assemble ML feature matrices for spending, saving and boom / bust / recession models. The full generated
listing is [[Econ Dictionary]].

## What is in it

| File (`src/mlab/econ/data/`) | Node kind | What |
|---|---|---|
| `classifications.yaml` | `cls:` | NAICS 2022 sectors, ISIC Rev.4 and UK SIC 2007 sections, COICOP, HS codes, sector crosswalk |
| `industries.yaml` | `ind:` | Industry tree from whole economy to niche lines (tiles, sanitaryware, garden ornaments, pallet networks), with NAICS / ISIC / UK SIC / COICOP / HS codes |
| `indicators.yaml` | `sig:` | The dictionary: layer (macro, meso, micro), geography, frequency, unit, cycle role (leading, coincident, lagging, target), series ids per source, ML transforms |
| `unit_economics.yaml` | `kpi:` | Company KPIs with formulas: SaaS (ARR, NRR, CAC, LTV, payback), retail (LFL, traffic, ticket, AOV), freight (tons per day, operating ratio, revenue per pallet), airlines, hotels, lenders |
| `sources.yaml` | `src:` | Every publisher: access (free / key / paid / members), cost, licence and its URL, commercial use, storage permission, rate limit, date checked |
| `instruments.yaml` | `mkt:` | Tradable proxies (sector ETFs, listed companies) with the industries they track and the KPIs they disclose |
| `edges.yaml` | | Economic links: `drives` and `leads` (sign, lag range in months, evidence) and `input_to` supply chains |

The loader generates the rest of the edges: `child_of` (tree), `measures` (indicator to industry),
`published_by` and `originates_from` (indicator to source and to the original publisher behind a mirror),
`applies_to` (KPI to industry), `proxies` and `discloses` (instrument to industry and KPI),
`classified_as` and `maps_to` (codes and crosswalks).

## Traversal

```bash
bash mlab econ summary                                   # counts and validation problems
bash mlab econ search gnome                              # find nodes
bash mlab econ show ind:pallet_networks                  # one node and all its edges
bash mlab econ walk ind:garden_ornaments -d 2            # neighbourhood (--types drives,leads --direction out --kind sig)
bash mlab econ path sig:uk_mortgage_approvals ind:retail_tiles_flooring
bash mlab econ sideways src:ons                          # alternative publishers for the same numbers
bash mlab econ sideways sig:uk_imports_garden_ornaments  # sibling indicators (US imports, plants, mowers, ...)
bash mlab econ sources ind:retail_bathroom_online -d 2   # who publishes what, at what cost, under which licence
```

`sideways` ranks same-kind peers by random-walk weight: each step splits its weight over the node's
edges, so a shared niche industry or a shared indicator counts for a lot and the whole-economy root counts
for nothing (it is never walked through). Sources search 4 hops (source, indicator, industry, indicator,
source) so they find substitutes that publish a different number on the same industry.

Python:

```python
from mlab.econ import load
g = load()
g.walk("ind:ltl_freight", depth=2, types={"drives", "leads", "input_to"})
g.path("sig:us_mortgage_rate", "mkt:hd")
g.sideways("src:fred", hops=2)
nodes, edges = g.frames()
```

## ML features

```bash
bash mlab econ panel ind:retail_home_improvement -d 2 --geo US        # FRED-backed, in memory, NBER label
bash mlab econ panel ind:retail_home_improvement -d 1 --geo UK --no-target --out uk.csv   # ONS (OGL), may be saved
```

The panel takes every indicator within `--depth` of a node that has a fetchable series (FRED or ONS today),
resamples it to month end, applies its `transform` list (yoy, mom, diff, log, z) and adds the target label
(`sig:us_recession_nber` by default). Two rules:

- **Storage follows the licence.** `--out` refuses when any column comes from a source whose
  `storage_allowed` is not `yes`. FRED's terms forbid storing or caching its content, so FRED columns stay
  in memory; freeze those numbers from the original publisher (BLS, BEA, Census, Fed Board: public domain)
  instead.
- **Reference period, not release date.** Rows are indexed by the period the number describes. Shift each
  column by its publication lag before training, or the model sees data that was not yet out.

`bash mlab econ check` hits every FRED and ONS series id once and reports any that fail; run it after editing
`indicators.yaml`. ONS ids were found with the ONS search API (`api.beta.ons.gov.uk/v1/search`).

## Influence map

`bash mlab econ export --out DIR` writes `nodes.csv` and `edges.csv`. Explicit `drives` and `leads` edges
with `evidence: prior` are desk priors, not findings: they are the candidate links for the lead-lag
network in the influence-map work (issue #58), which tests them on train years and keeps only what
survives. `literature` edges carry the paper that established them.

## Adding to it

1. Add or edit entries in the YAML (ids are the keys; references use `kind:key`).
2. A new source needs `access`, `licence`, `licence_url`, `commercial_use`, `storage_allowed`, read from the
   publisher's own terms page, with `verified: false` if that page could not be read.
3. `bash mlab econ summary` must show no validation problems; `bash mlab econ check` for new FRED / ONS ids.
4. Regenerate the listing: `bash mlab econ docs --out docs/wiki/Econ-Dictionary.md`.
