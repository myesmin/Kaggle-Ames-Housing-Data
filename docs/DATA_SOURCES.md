# Data Sources

Every dataset used in this project, with its origin, retrieval method, verified extent,
and licence. Nothing here requires an API key: `make data` fetches all of it from a clean
clone.

Retrieval code: [`src/ames/data.py`](../src/ames/data.py).
Contract tests that assert these shapes and date ranges still hold:
[`tests/test_data.py`](../tests/test_data.py).

---

## 1. Property-level data

### Ames Housing (De Cock, 2011) — the primary source

| | |
|---|---|
| **URL** | `https://jse.amstat.org/v19n3/decock/AmesHousing.txt` |
| **Citation** | De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education*, 19(3). |
| **Documentation** | `https://jse.amstat.org/v19n3/decock/DataDocumentation.txt` |
| **Format** | Tab-delimited, 2,930 rows × 82 columns |
| **Coverage** | Residential sales in Ames, Iowa, January 2006 – July 2010 |
| **Compiled from** | Ames City Assessor's records |

**Why the publication and not the pre-split CSVs.** This project's first pass worked from
a split copy of the same data: `train.csv` (2,051 rows, target included) and `test.csv`
(878 rows, target withheld). With no labels on the holdout there is no way to measure test
error inside the project; the only feedback is an externally computed score, which is how
a corrupted test frame (Model Risk Log defect #1) survived undetected. De Cock's publication
carries all 2,930 rows *with* `SalePrice`. That makes holdout scoring possible
inside the project, and with it the re-scoring of those first-pass predictions in notebook 03.

Verified annual sale counts: **625 / 694 / 622 / 648 / 341** for 2006–2010. The 2010 count
is lower because the extract ends in July.

### Property coordinates

| | |
|---|---|
| **URL** | `https://github.com/topepo/AmesHousing/raw/master/data/ames_geo.rda` |
| **Source** | Max Kuhn's `AmesHousing` R package (CRAN), geocoded from parcel records |
| **Format** | R `.rda`, read with `pyreadr`; 2,932 rows × 3 columns |
| **Join key** | `PID`, zero-padded to 10 characters |
| **Verified join rate** | **99.59%** (2,918 of 2,930 properties) |
| **Extent** | Latitude 41.986 – 42.073, Longitude −93.693 – −93.577 |
| **Licence** | GPL-2 (the R package) |

The join key needs care: De Cock ships `PID` as a 9-digit integer, the geo table as a
zero-padded 10-character string. Joining without padding matches 1 row in 2,930.

### First-pass modelling split (committed)

`data/raw/baseline/{train.csv, test.csv, sample_sub_reg.csv, MY_submission.csv}`

The split this project used in its first pass, plus the predictions it produced. These are
committed rather than downloaded because they are local to this project and exist nowhere
public. They serve two purposes only: defining the `baseline_split` membership flag, and
re-scoring those predictions in notebook 03 against the now-known true prices.

---

## 2. Macroeconomic series (FRED)

All nine are fetched key-free through the `fredgraph.csv?id=` endpoint. **Do not set a
browser `User-Agent`** on these requests. FRED's edge black-holes requests that claim to
be Chrome but do not match at the TLS layer, and the connection hangs until timeout. The
stock `requests` User-Agent is accepted. (This is documented in `data.py` because it cost
an afternoon.)

| Series ID | Name in the panel | Freq | Verified extent | What it is |
|---|---|---|---|---|
| `ATNHPIUS11180Q` | `hpi_ames_msa` | Quarterly | 1986-Q4 → 2026-Q2 | FHFA All-Transactions House Price Index, Ames MSA |
| `ATNHPIUS19169A` | `hpi_story_county` | Annual | 1978 → 2025 | FHFA All-Transactions HPI, Story County |
| `CSUSHPINSA` | `hpi_case_shiller_us` | Monthly | 1987-01 → 2026-06 | S&P CoreLogic Case-Shiller U.S. National HPI, NSA |
| `MORTGAGE30US` | `mortgage30us` | Weekly | 1971-04 → 2026-09 | Freddie Mac 30-year fixed rate (PMMS) |
| `CPIAUCSL` | `cpi` | Monthly | 1947-01 → 2026-08 | CPI-U, All Items, seasonally adjusted |
| `DGS10` | `treasury10y` | Daily | 1962-01 → 2026-09 | 10-year Treasury constant maturity |
| `PCPI19169` | `income_pc_story` | Annual | 1969 → 2024 | Per-capita personal income, Story County |
| `MEHOINUSIAA672N` | `income_median_hh_iowa` | Annual | 1984 → 2025 | Real median household income, Iowa |
| `LAUMT191118000000003A` | `unemployment_ames` | Annual | 1990 → 2025 | Unemployment rate, Ames MSA |

**Frequency alignment.** Series are placed at their period start and **forward-filled only**
to a monthly grid; daily and weekly series are averaged to month. Nothing is back-filled,
so no observation is ever informed by a value published after it.
`tests/test_data.py::test_no_series_is_back_filled_into_the_past` asserts this.

**FHFA vs Case-Shiller.** These are not the same construction. Both are repeat-sales
indices, but FHFA All-Transactions is built from conforming mortgage records (purchases
*and* appraisals) while Case-Shiller covers all arm's-length purchases regardless of loan
type. They are used here to compare *shapes*, both indexed to 2006-Q1 = 100, never
subtracted from one another.

---

## 3. Zillow Research (public CSVs)

| | ZHVI | ZORI |
|---|---|---|
| **URL** | `.../public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` | `.../public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv` |
| **Region** | `Ames, IA` (RegionID 394325) | `Ames, IA` |
| **Verified extent** | 210 monthly obs, 2009-03 → 2026-08 | 107 monthly obs, 2017-10 → 2026-08 |
| **Verified levels** | $151,194 → $279,794 | $864 → $1,122 |
| **What it measures** | Smoothed, seasonally-adjusted typical value for homes in the 35th–65th percentile of the stock | Smoothed observed market-rate rent, repeat-listing weighted |
| **Licence** | Zillow permits use with attribution |

Base URL for both: `https://files.zillowstatic.com/research/public_csvs/`

**The ZORI limitation.** ZORI begins in October 2017. The sales in this
dataset ended in July 2010. There is no overlap. Notebook 07 is therefore framed as
*"underwrite these properties at today's values and today's rents"*. The data cannot
support a historical backtest. Any analysis claiming to backtest 2006–2010
acquisitions against realised rents would be fabricating the rent series.

**ZHVI as a control on FHFA.** Anchoring the FHFA Ames index to observed 2006–2010 sale
prices implies a 2026 median around **$310k**; ZHVI reports **$280k**, an 11% gap.
The two measure different things (repeat-sales of transacting homes vs. a hedonic estimate
of the whole stock), and notebook 02 carries the gap forward as a live uncertainty instead
of averaging it away.

**All 410 metros, not just Ames.** `hpi_at_metro.csv` carries the FHFA
All-Transactions index for every US metropolitan area, quarterly back to 1975, in one
4 MB key-free file.

| | |
|---|---|
| **URL** | `https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_metro.csv` |
| **Format** | CSV with **no header row**: `metro, cbsa, year, quarter, index_nsa, pct_change`; `-` for quarters an index does not cover |
| **Verified extent** | 410 metros with a measurable 2005–2013 window |
| **Verified values** | Ames (CBSA 11180) peak-to-trough **−4.26%**; median metro **−13.6%**; worst **Merced, CA −65.0%** |

This file allows Ames to be ranked within the full distribution of metros. "Ames fell
4.3% against 27.4% nationally" compares Ames with an average that blends Merced with
Ithaca; "82% of 410 metros fell further than Ames" locates it in the distribution. The
same file showed that the −30% scenario notebook 08 borrows from the supervisory exercise
was **exceeded by 19% of metros**, so the scenario should be read as a supervisory
convention and not as a calibrated tail.

---

## 4. Mortgage applications (HMDA)

| | |
|---|---|
| **URL** | `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years={year}&msamds=11180` |
| **Source** | CFPB HMDA Data Browser (Home Mortgage Disclosure Act public filings) |
| **Geography** | Ames, IA MSA (11180), the same market as the valuation model |
| **Verified extent** | **32,926 filings, 2018–2024**; 25,203 carry a credit decision |
| **Verified outcomes** | 21,752 originated · 2,943 denied · 11.7% denial rate on decisioned |
| **Key fields** | `action_taken`, `denial_reason-1..4`, `loan_to_value_ratio`, `property_value`, `income`, `debt_to_income_ratio`, `derived_race`, `derived_ethnicity`, `derived_sex` |
| **Licence** | Public domain (US federal government) |

**Follow the redirect.** The endpoint does not return the CSV. It pre-computes each
filtered query and answers **301** with a `Location` on `files.ffiec.cfpb.gov`.
`requests` follows redirects by default and gets the data; `curl` without `-L` gets a
182-byte HTML stub that decompresses to *"This and all future requests should be directed
to…"* and reads, to a script counting rows, as **an empty result set**. Three years
looked like zero applications before this was noticed.

**Why `action_taken == 6` is excluded.** Code 6 is "purchased loan": an origination by
another institution, bought on the secondary market. No credit decision was made by the
filer. Counting those as approvals drags every denial rate toward zero; 3,475 of the
32,926 filings are code 6.

**What this file does not contain.** No credit score, no assets, no reserves, no
employment history, no property condition. Credit score is the strongest single predictor
of mortgage denial. Every adjusted disparity in notebook 09 is therefore an **upper bound
on unexplained disparity**, and the notebook quantifies how strong that missing variable
would have to be to explain those disparities away.

---

## 5. Property tax (cited, committed)

Not downloadable as a machine-readable series, so the figures are hard-coded in
[`src/ames/config.py`](../src/ames/config.py) with their sources:

| Input | Value | Source |
|---|---|---|
| Consolidated levy, AMES/AMES district | **$30.58245** per $1,000 taxable value | Story County Auditor, *"LEVIES 2024 — Payable 2025-2026"*, consolidated rate page. Components: county 5.43987 + City of Ames 10.30432 + state/ag-extension 0.78046 + Ames CSD 14.05780 |
| Iowa residential rollback, AY2024 | **47.4316%** | Iowa Legislative Services Agency, *Fiscal FactBook*, "Rollback Percentages" table, certified by the Iowa Department of Revenue |
| **Effective rate on market value** | **1.4505%** | 30.58245 / 1,000 × 0.474316 |

Levy documents: `storycountyiowa.gov/643/Tax-Levy-Rates`.
Rollback orders: `storycountyiowa.gov/644/Rollback-Information`.

`tests/test_finance.py::test_property_tax_uses_the_cited_story_county_rate` asserts the
arithmetic, so the constant cannot drift from its citation unnoticed.

---

## 6. Deliberately excluded

| Source | Why excluded |
|---|---|
| **Census ACS** (detailed tables) | Requires a free API key. Including it would mean `make data` fails on a clean clone without a secret, which breaks the reproducibility claim. |
| **HUD Fair Market Rents** | Same reason: API key required. |
| **MLS / listing data** | Not publicly redistributable. |
| **Loan performance data** | No default outcomes exist for Ames properties in any public source. HMDA records the *decision*, never what happened afterwards. This is why notebook 08's PD curve is presented as a **shape** taken from the published literature, not calibrated, and never claimed as an estimate. |
| **Credit scores** | Not in any public file, by design. Their absence is the central limitation of notebook 09 and is handled explicitly there. |

Where an excluded source would have supplied a number (rental vacancy, insurance cost,
property tax), the substitute is named and cited in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) with a sensitivity range.

---

## Reproducing

```bash
make data          # ~2 minutes; writes data/{raw,external,processed}/
make test          # 304 tests, including the data contracts above
```

`data/raw/` and `data/external/` are gitignored; they are fully reproducible from this
document. Only `data/raw/baseline/` is committed.
