# Underwriting Assumptions

Every input to the investment DCF (notebook 07) and the credit model (notebook 08), with
its source, its default value, and the range it is stressed over.

Any figure that appears in a DCF cash-flow statement is listed here, either with a
citation or marked as a judgement call.

Defaults live in `UnderwritingAssumptions` in
[`src/ames/finance.py`](../src/ames/finance.py) and in
[`src/ames/config.py`](../src/ames/config.py). Anything marked **judgement** is not
sourced from data and is the first thing to argue with.

---

## 1. Operating assumptions

| Input | Default | Source | Sensitivity range | Notes |
|---|---:|---|---|---|
| **Vacancy rate** | 7.0% | Reported Ames rental vacancy rate (ACS-derived, via public rental-market aggregators) | 3% – 12% | Ames is 58% renter-occupied and university-driven, so turnover is high and concentrated in August. A 7% annual figure is a market average; a specific property near campus may be effectively 0% for nine months and 100% for three. |
| **Property management** | 8.0% of EGI | Standard third-party single-family management fee | 6% – 10% | Charged on collected rent, not scheduled rent, hence applied to EGI. Set to 0% if self-managing, which is worth roughly 80 bps of cap rate. |
| **Maintenance & repairs** | 8.0% of EGI | **Judgement**: mid-point of the 5–12% range commonly underwritten for detached SFR | 5% – 12% | Should scale with `house_age`; the median Ames home in this dataset was 35 years old at sale. Held flat here because no repair-cost data exists for these properties. |
| **CapEx reserve** | 5.0% of EGI | **Judgement**: sinking fund for roof, HVAC, appliances | 3% – 8% | Not an expense in year one; it is a reserve. Excluding it (as many brokers' pro formas do) inflates NOI and therefore the quoted cap rate. |
| **Insurance** | $2,800 / year | Iowa average annual homeowners premium, 2025 | $1,800 – $4,000 | A landlord (DP-3) policy typically runs above an owner-occupied HO-3, so this is a floor. Iowa premiums rose ~28% in 2025; the upper bound reflects that trend continuing. |
| **Property tax** | **1.4505%** of market value | Story County Auditor levy × Iowa DoR rollback; see [DATA_SOURCES.md §5](DATA_SOURCES.md#5-property-tax-cited-committed) | 1.2% – 1.7% | The only fully-cited operating input. Assessed value is assumed to track market value, which understates tax in a rising market (Iowa reassesses on a two-year cycle) and overstates it in a falling one. |

**Why percentages of EGI rather than of gross rent.** Vacancy is applied first to get
effective gross income; management and maintenance are then taken on what is actually
collected. Applying them to scheduled rent double-counts the vacancy loss.

**Why insurance and tax are not percentages.** Insurance is a fixed dollar amount per
property and does not scale with rent. Property tax scales with *assessed value*, not with
income. Modelling either as a share of rent would break the link between a property's
price and its expense load, and the cap rate depends on that link.

---

## 2. Capital structure

| Input | Default | Source | Sensitivity range | Notes |
|---|---:|---|---|---|
| **LTV** | 75% | Conventional investor (non-owner-occupied) maximum | 60% – 80% | Fannie/Freddie cap investor purchase LTV at 80% for a 1-unit; 75% is where pricing stops deteriorating sharply. |
| **Mortgage rate** | 7.25% | Freddie Mac PMMS 30-year + ~40 bp investor spread | 5.5% – 9.0% | PMMS is an owner-occupied survey rate; investment property carries a loan-level price adjustment worth roughly 0.25–0.75% in rate. |
| **Amortisation** | 30 years | Standard | 15 / 30 | |
| **Closing costs** | 2.0% of price | **Judgement**: title, origination, inspection, recording | 1.5% – 3.0% | Buyer side only. Added to the equity outlay at t=0, so it drags IRR from the first dollar. |

**Credit-model LTV is different and deliberately so.** Notebook 08 originates at **80%**
LTV rather than 75%, because it is modelling an owner-occupied loan book (the 2,930 homes
as originated mortgages), not investor acquisitions. The distinction matters: 80% is the
threshold above which private mortgage insurance is required, which is why it is the
modal origination LTV in the real world.

---

## 3. Exit

| Input | Default | Source | Sensitivity range | Notes |
|---|---:|---|---|---|
| **Hold period** | 10 years | **Judgement**: conventional institutional hold | 5 – 15 | |
| **Exit cap rate** | 6.5% | **Judgement**: going-in cap plus expansion for a 10-year-older asset | 5.0% – 8.0% | The single most consequential assumption in the model; notebook 07's tornado chart usually ranks it first or second. Setting exit cap = going-in cap is a common way to make a deal look good on paper. |
| **Selling costs** | 6.0% of exit value | Brokerage plus transfer and title at disposition | 4% – 8% | |

**Exit value convention.** Notebook 07 takes the exit value as the **greater** of (a)
direct capitalisation of forward NOI at the exit cap and (b) the HPA-grown price. Using
only (a) makes the return an artefact of the exit-cap assumption; using only (b) ignores
that a buyer prices off income. Neither is obviously right, and the tornado stresses both
drivers. This is a judgement call, stated because it materially moves the IRR.

---

## 4. Growth

| Input | Default | Source | Sensitivity range | Notes |
|---|---:|---|---|---|
| **Rent growth** | 3.0% / yr | ZORI Ames MSA, 2017-10 → 2026-08: $864 → $1,122, a 3.0% CAGR | 1.0% – 5.0% | Measured, not assumed. |
| **House price appreciation** | 3.0% / yr | FHFA Ames MSA long-run trend | 0% – 6% | **This is roughly the rate of inflation.** Notebook 02 shows real Ames prices *fell* 11% between 2006 and 2012 while nominal prices held. A 3% nominal HPA is a flat real assumption, not a growth assumption. |
| **Expense growth** | 3.0% / yr | Matched to CPI trend | 2% – 5% | Held equal to rent growth by default, so the expense ratio is constant. Setting expense growth above rent growth is the standard bear case and is inside the stated range. |

---

## 5. Monte Carlo correlation structure

The four drivers are simulated as correlated normal shocks. The correlation matrix in
`finance.DEFAULT_CORRELATION` is judgement, not estimation. Five years of one city's
data cannot identify a 4×4 correlation matrix.

| | rent growth | HPA | exit cap | mortgage rate |
|---|---:|---:|---:|---:|
| **rent growth** | 1.00 | 0.60 | −0.20 | 0.10 |
| **HPA** | 0.60 | 1.00 | −0.50 | −0.10 |
| **exit cap** | −0.20 | −0.50 | 1.00 | 0.55 |
| **mortgage rate** | 0.10 | −0.10 | 0.55 | 1.00 |

Three economic facts are encoded. Each is spelled out so that a reader can disagree with
a specific number:

1. **Rent growth and HPA move together (+0.60).** Both are driven by local income and
   household formation.
2. **Higher mortgage rates push cap rates up (+0.55).** Notebook 02 measures the
   mechanism directly: at a fixed monthly payment, +100 bp costs **9.4%** of supportable
   price. Debt cost is the alternative to equity yield, so they track.
3. **Cap-rate expansion accompanies weak appreciation (−0.50).** These are two
   descriptions of the same deteriorating market.

Default volatilities (standard deviations of the additive shock):

| Driver | σ | Rationale |
|---|---:|---|
| rent growth | 1.5% | ZORI Ames year-over-year dispersion |
| HPA | 2.5% | FHFA Ames quarterly index volatility, annualised |
| exit cap rate | 1.0% | Judgement; spans a full cycle of cap-rate movement |
| mortgage rate | 1.25% | PMMS dispersion over 2019–2026 |

Exit cap and mortgage rate are floored at 1% so a tail draw cannot produce an infinite
exit value.

---

## 6. Credit-risk assumptions (notebook 08)

| Input | Default | Basis | Notes |
|---|---:|---|---|
| **Origination LTV** | 80% | Modal conventional conforming LTV | Applied to the **AVM value**, not the sale price. Notebook 08 exists to measure the effect of that choice. |
| **PD curve** | logistic in MTM LTV: floor 0.5%, ceiling 45%, midpoint at LTV = 1.00, steepness 9 | **Shape, not estimate** | There are no default outcomes in the Ames data. What is encoded is the well-established empirical *shape*: PD near a floor while equity is positive, steep rise as MTM LTV crosses 100% and the default option moves into the money, saturating well below 100% because many underwater borrowers keep paying. It is not calibrated. |
| **Foreclosure cost haircut** | 25% | Within the observed range for single-family REO disposition | Notebook 01 provides supporting evidence from the data itself: `Abnorml` sales (foreclosure, short sale) transact **15% below** `Normal` sales per square foot. The remaining ~10% covers legal, carrying and maintenance costs. |
| **Stress scenario** | −30% house prices | Federal Reserve supervisory severely adverse scenario, house-price decline over the nine-quarter planning horizon | Applied as an instantaneous shock to collateral value with loan balances unchanged, which is the standard convention: a stress hits the asset side and leaves the borrower's amortisation schedule alone. |

---

## 7. What would change these

The assumptions most worth replacing with data, in order of how much they move the answer:

1. **Exit cap rate.** Usually the top of the tornado. Would be replaced by actual Ames
   SFR transaction cap rates, which are not publicly available at metro level.
2. **The PD curve.** Replacing the shape with a calibrated curve would need loan
   performance data (e.g. Fannie Mae's public single-family loan performance dataset,
   which has no Ames geography but could supply a national curve conditioned on MTM LTV).
3. **Rent by property.** ZORI is one metro-wide number. Notebook 07 scales it by bedroom
   count and size, which is a crude hedonic. Actual listing-level rents would replace the
   largest single source of error in the NOI.
4. **Maintenance and CapEx.** Currently judgement, and jointly worth ~13% of EGI.

All four are repeated in the README's limitations section.
