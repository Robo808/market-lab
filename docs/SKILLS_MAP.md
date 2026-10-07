# Installed skills: what to reach for

Checked 2026-10-07 in the cloud container. "Works" = methodology/output skills that run on data we feed them from `mlab`.

| Task | Reach for | Status here |
|---|---|---|
| Valuation of a single name | `financial-analysis:dcf-model`, `financial-analysis:comps-analysis`, `financial-analysis:3-statement-model` | Works with `./mlab dd` / `lens buffett` data |
| LBO / take-private math | `financial-analysis:lbo-model`, `private-equity:returns-analysis` | Works |
| Initiation / thesis / catalysts | `equity-research:initiating-coverage`, `equity-research:thesis-tracker`, `equity-research:catalyst-calendar` | Works |
| Earnings preview / review | `equity-research:earnings-preview`, `equity-research:earnings-analysis` (+ `earnings-desk`) | Works with `mlab earnings` data |
| Idea generation / screens | `equity-research:idea-generation`, `equity-research:screen`, `market-researcher:idea-generation` | Works with `mlab scan` / `signals` |
| Sector primers | `equity-research:sector-overview`, `market-researcher:sector-overview`, `financial-analysis:competitive-analysis` | Works (web + mlab data) |
| Due diligence checklist | `private-equity:dd-checklist`, `private-equity:dd-meeting-prep` | Works |
| Morning note | `equity-research:morning-note` (+ `morning-brief` workspace skill) | Works |
| Stats / backtest review | `data:statistical-analysis`, `anthropic-skills:applied-statistics`, `anthropic-skills:data-science` | Works |
| Charts / dashboards | `dataviz`, `data:create-viz`, `data:build-dashboard` | Works |
| Excel / PDF / slides out | `anthropic-skills:xlsx`, `financial-analysis:xlsx-author`, `anthropic-skills:pdf`, `anthropic-skills:pptx` | Works |
| Deep web research | `anthropic-skills:deep-research` | Works (WebSearch/WebFetch) |
| LSEG rates/FX/vol/bond analytics | `lseg:*` | Needs the LSEG connector (its MCP server is refused by the current network policy) |
| S&P tear sheets / funding digests | `sp-global:*` | Needs the S&P Capital IQ connector |
| End-to-end earnings model update | `earnings-reviewer` agent | Needs FactSet / Daloopa connectors |
| Sector research agent | `market-researcher` agent | Needs CapIQ / FactSet connectors; use its skills instead |

Connectors that plug straight into the desks once connected: Alpha Vantage (prices, options, transcripts, news sentiment),
FMP (fundamentals, transcripts, calendar), Financial Datasets (statements, insider, filings), Bigdata.com (news, transcripts,
events), TEXT TO QUANT (backtests), Meltwater (media/social), CoinMarketCap (crypto).
