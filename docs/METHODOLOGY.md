# Methodology

Why each method was chosen, and what it would take to falsify the choice. Implementation
lives in [`src/ames/`](../src/ames); this document is the reasoning.

---

## 1. Target and retransformation

### Why `log(SalePrice)`

Sale prices are right-skewed (skew 1.74) and the residuals of a raw-dollar regression fan
out with the fitted value; Levene's test across fitted quartiles rejects equal variance
decisively. Three consequences follow, and only the first is cosmetic:

1. Squared-error loss weights a $600k home ~16× a $150k home, so the fit is driven by the
   tail rather than by the bulk of the book.
2. Coefficient standard errors are wrong under heteroskedasticity, so any inference drawn
   from the hedonic coefficients is invalid.
3. A multiplicative error structure is what the *market* has: appraisal error scales with
   price. Modelling it additively mis-states risk at both ends.

Taking logs converts the multiplicative structure into an additive one, which is what OLS
and its penalised variants assume.

### Why Duan smearing rather than naive `exp()`

A model fit on `log(y)` estimates `E[log y | x]`. Exponentiating gives the conditional
**median** of `y`, not its mean. For a right-skewed target this biases every dollar
valuation low, and by more at the top.

The Duan (1983) smearing estimate corrects it non-parametrically:

```
E[y | x] ≈ exp(x'β) · (1/n) Σᵢ exp(eᵢ)
```

The lognormal correction `exp(σ²/2)` is not used because it assumes the residuals are
Gaussian, and they are not. Duan's estimator assumes only that the residual distribution
is the same for the prediction as for the calibration sample.

In MdAPE terms the correction is nearly invisible, because MdAPE is a median and a
uniform multiplicative shift barely moves it. It does change the **mean signed error**, which is the quantity that propagates into a loan book: a 2% systematic
under-valuation across 2,400 properties is 2% of over-stated LTV on every loan, in the
same direction. That error does not diversify away.

Calibration uses **training** residuals only.

---

## 2. Feature encoding

### Ordinal ladders instead of one-hot

Ten features share the `Ex > Gd > TA > Fa > Po` scale, and another twelve have their own
orderings (`Functional`, `Bsmt Exposure`, `BsmtFin Type 1` and `2`, `Garage Finish`,
`Fence`, `Lot Shape`, `Land Slope`, `Paved Drive`, `Utilities`, `Electrical`,
`Central Air`).

One-hot encoding an ordered ladder spends five columns to express information worth one,
and also throws the ordering away. The model must relearn from data that `Ex > Gd`,
which it can only do where there are enough observations of each level.

Integer encoding preserves the order at the cost of imposing equal spacing between grades.
That is a real assumption. It is defensible here because notebook 03's hedonic
coefficients show the price effect per grade step is roughly constant across the ladder,
and because the alternative, one-hot encoding, imposes *no* structure at the cost of ~40 extra
columns on a training set of under 2,000 rows.

### `0` means "absent", and it sits below the worst grade

`NA` in `Bsmt Qual` means there is no basement, not that the basement's quality is
unknown. Encoding absent as `0` keeps the ladder monotone: *no basement < poor basement <
excellent basement*. Imputing the mode instead would place "no basement" in the middle of
the scale, which puts the levels in the wrong order.

### Geometry

Distance and bearing to ISU campus and to downtown, with bearing decomposed into
`sin`/`cos` so that 1° and 359° are adjacent rather than maximally distant.

Notebook 02 shows why these are supplied *separately* rather than as a composite
"location score": raw distance to campus carries a highly significant **+10.4% per mile**
on price per square foot, but once house age, quality, lot size and sale year enter the
regression the coefficient collapses to **−0.04% per mile** (*t* = −0.06). The apparent
campus gradient is entirely a housing-vintage effect. A hand-built composite would have
fused the two permanently and the model would have learned a premium that does not exist.

---

## 3. The pipeline, and why leakage is structural rather than procedural

Four of the ten Model Risk Log defects were leakage, and all four had the same root cause:
train and test were separate dataframes, each preprocessed by its own code path.

The rewrite does not depend on remembering to fit on training rows only. Every statistic
that must be *learned* (imputation medians, one-hot vocabulary, scaler moments) lives inside a single
`ColumnTransformer` fit on training rows. Everything applied outside it
(`ames.features.prepare`: ordinal ladders, absence encoding, geometry, arithmetic
aggregates) is a pure function of a single row and reads nothing from the sample.

Each part of that design is tested:

| Claim | Test |
|---|---|
| Fitted statistics are identical with and without test rows present | `test_fitted_statistics_are_identical_with_and_without_test_rows` |
| …and that check is not vacuous: a full-data fit really does differ | `test_fitting_on_the_full_dataset_really_does_change_the_statistics` |
| Transforming a row does not depend on its companions | `test_transform_of_a_row_is_independent_of_its_companions` |
| `prepare` is row-local | `test_prepare_is_row_local` |
| Design-matrix width is fixed by the training vocabulary | `test_column_count_is_fixed_by_the_training_vocabulary` |
| A level seen only at predict time is handled, not dropped | `test_a_category_seen_only_at_predict_time_is_handled_not_dropped` |

---

## 4. Validation design

### Two regimes, reported side by side

| Regime | Split | What it measures |
|---|---|---|
| **Random** | 75/25 on shuffled rows | Interpolation within a known market. Comparable to this project's first pass (which used 95/5; see defect #6). |
| **Temporal** | Train 2006–2008, test 2009–Jul 2010 | Forward prediction, which is the only thing an AVM is ever asked to do. |

Both are reported, together with the gap between them. A single random-split accuracy
number is the industry's most common way of overstating an AVM.

One caveat applies. Notebook 02 measures Ames's peak-to-trough decline at **4.3%**
against **27.4%** nationally; Iowa State's employment base made Ames nearly acyclical.
So this temporal split is a *mild* regime shift, and the degradation it shows is a **lower
bound** on what a cyclical market would produce. That is also why the credit stress in
notebook 08 imposes the supervisory −30% shock explicitly instead of using Ames's own
history, which contains no severe downturn to learn from.

### `GroupKFold` on `Neighborhood` for hyperparameter selection

Each fold holds out entire neighborhoods, so a model cannot tune its penalty by
memorising local price levels it will also see at validation.

This is deliberately conservative. Holding out a whole
neighborhood means its one-hot column is unseen at validation and routes to the
"infrequent" bucket, so the CV score is *pessimistic* relative to deployment, where an AVM
values a home in a neighborhood it has priced a thousand times. It is still the right
instrument for *selecting a penalty strength*, because it favours models that generalise
across space over models that overfit 28 location dummies. The reported accuracy comes
from the test sets, not from CV.

### The valuation universe

Two exclusions, both recommended by De Cock:

- **Arm's-length only** (`Sale Condition == Normal`, 82.4% of rows). Notebook 01 measures
  why: `Abnorml` transacts 15% below `Normal` per square foot, `Partial` 26% above. These
  are different price-formation processes; one model fit across all of them learns their
  average. The excluded rows return in notebook 08 as the empirical basis for the
  foreclosure haircut in LGD.
- **Five homes above 4,000 sq ft**, three of which are `Partial` sales of unfinished
  construction. They are **flagged, not dropped**, so the exclusion is visible in the data
  and reversible.

---

## 5. AVM accuracy metrics

R² measures how much variance the model explains. A lender wants to know how far a
valuation of its collateral is likely to be off. Those are different questions, and only the second has an
accepted industry vocabulary.

| Metric | Definition | Why |
|---|---|---|
| **MdAPE** | median \|(pred − sale)/sale\| | The headline. A median, so one catastrophic miss cannot flatter or ruin it. This matters because AVM error distributions have fat tails. |
| **PPE_N** | share of valuations within ±N% | What an underwriting policy is actually written against: "accept the AVM when it is within 10%". |
| **FSD** | sd of log(pred/sale) | Forecast Standard Deviation, the per-valuation confidence measure AVM vendors publish. Scale-free and symmetric in over/under-valuation. |
| **Mean signed error** | mean (pred − sale)/sale | Bias. Distinct from MdAPE: a model can be accurate on average and systematically wrong in a segment. This is the number that becomes LTV error. |

**Thresholds.** MdAPE < 5% and PPE10 > 75% are the levels institutional AVMs are commonly
held to by secondary-market purchasers. They are quoted here as benchmarks to be measured
against, pass or fail, not as a claim about any specific vendor's published criteria.

---

## 6. Split-conformal prediction intervals

### Why not the textbook prediction interval

The OLS prediction interval is valid only if residuals are homoskedastic and Gaussian.
They are neither (see §1). An interval that assumes what the data contradicts gives a
coverage guarantee that does not hold.

### The method

Lei et al. (2018), *Distribution-Free Predictive Inference for Regression*:

1. Fit on a proper training subset.
2. On a held-out **calibration** subset, compute `|residual|` in log space.
3. Take `q` = the `⌈(n+1)(1−α)⌉/n` empirical quantile of those residuals.
4. The interval for a new property is `exp(pred ± q)`.

Three properties make this the right tool:

- **Distribution-free.** It assumes only that calibration rows are exchangeable with
  prediction rows. No normality, no homoskedasticity.
- **Finite-sample valid.** The `(n+1)` correction in step 3 makes coverage ≥ 1−α exactly,
  not asymptotically. Dropping it under-covers on small calibration sets.
- **Multiplicative in dollars.** A constant `q` in log space is a constant *ratio* in
  dollars: a $500k home gets a wider dollar band than a $150k home, which is the correct
  shape for valuation error.

**Validation.** `tests/test_avm.py` checks empirical coverage against nominal at α ∈
{0.10, 0.20, 0.32}, including a deliberately heteroskedastic case designed to break the
Gaussian interval. Notebook 04 repeats the check on the real holdout.

### The exchangeability caveat

Conformal guarantees require exchangeability. Under the **temporal** split that assumption
is violated by construction: 2009–2010 rows are not exchangeable with 2006–2008 rows.
Notebook 04 measures the resulting coverage gap instead of asserting that the guarantee
holds. The measurement shows how much the guarantee degrades under the distribution shift
an AVM actually faces.

---

## 7. Spatial diagnostics

**Moran's I** on model residuals, over a row-standardised 8-nearest-neighbour weights
matrix built from the property coordinates.

A hedonic model is supposed to have absorbed location through `Neighborhood` and the
distance features. If residuals still cluster in space, there is location value the model
has not captured. This matters in practice: spatially
correlated errors are correlated where a lender's collateral is correlated, so
they do not diversify within a portfolio.

Significance is assessed with the Cliff–Ord normal approximation. The implementation is
validated in `tests/test_avm.py` against a spatially random field (I ≈ 0) and a pure
spatial gradient (I ≈ 0.99).

---

## 8. Fairness / error-parity testing

The June 2024 interagency AVM quality-control rule (OCC, Fed, FDIC, NCUA, CFPB, FHFA)
requires institutions to adopt policies addressing five factors. The fifth,
nondiscrimination, was added by the agencies beyond the four in the statute.

There is no protected-class attribute anywhere in the Ames data, so this is **not** a
fair-lending test.

It can do the structural half: measure whether accuracy is uniform across
neighborhoods and price segments. An AVM that is systematically pessimistic in particular
geographies transmits that pessimism into every LTV and every credit decision downstream,
which is the mechanism the rule is concerned with. Notebook 04 reports MdAPE,
PPE10 and mean signed error by neighborhood and by price decile, and flags where the AVM
systematically under-values.

Groups below 20 observations are dropped. A MdAPE computed on eight houses is noise, and
reporting it as a disparity would repeat the error the first pass made with
a 103-row test set.

---

## 9. Investment underwriting

### Framing, and the limitation that forces it

ZORI rents begin in October 2017. The sales in this dataset end in July 2010. There is
no overlap. Notebook 07 is therefore framed as *"underwrite these properties at today's
values and today's rents"*, not as a historical backtest. Claiming to backtest 2006–2010
acquisitions against realised rents would require inventing the rent series.

### Roll-forward

Each 2006–2010 sale price is indexed forward with the FHFA Ames MSA HPI to a 2026 value,
cross-checked against Zillow ZHVI. The two land **11% apart** ($310k vs $280k); notebook 02
explains why (repeat-sales of transacting homes vs. a hedonic estimate of the whole stock)
and notebook 07 carries the gap as a stated uncertainty rather than averaging it away.

### Returns

- **NOI** excludes debt service by definition. Mixing debt into NOI is the most common
  way a cap rate gets quoted wrong. Enforced by
  `test_noi_excludes_debt_service`.
- **IRR** by bisection, not Newton. Levered real-estate cash flows are not always
  well-behaved near the root, and a failed Newton iteration returns a plausible-looking
  wrong number where bisection returns `NaN`. `NaN` is the correct result when a stream has
  no sign change. Validated against `numpy_financial.irr` and against the defining
  property `NPV(IRR) = 0`.
- **Monte Carlo** with correlated draws on rent growth, HPA, exit cap and mortgage rate.
  The correlation structure is judgement, documented and justified in
  [`ASSUMPTIONS.md §5`](ASSUMPTIONS.md#5-monte-carlo-correlation-structure). Five years of
  one city's data cannot identify a 4×4 correlation matrix.

---

## 10. Credit risk

### Expected loss

`EL = PD × LGD × EAD`, per loan.

- **EAD** is the outstanding balance. A fully-drawn amortising mortgage has no undrawn
  commitment, so there is no credit-conversion factor.
- **PD** is a logistic function of mark-to-market LTV. This is a **shape, not an
  estimate**: there are no default outcomes in the Ames data and none are invented. The
  shape encodes the well-established empirical pattern: near a floor while the borrower
  has equity, steep as MTM LTV crosses 100% and the default option moves into the money,
  saturating well below 100% because many underwater borrowers keep paying.
- **LGD** from a foreclosure-cost haircut: `LGD = 1 − (1−h)/MTM_LTV`, zero when the loan
  is sufficiently over-collateralised. The 25% haircut has partial support from the data
  itself: notebook 01 measures `Abnorml` sales transacting 15% below `Normal` per square
  foot, with the remainder covering legal, carrying and maintenance costs.

### Valuation error as credit risk

A lender originates against a valuation of the property, which can differ from the price
paid.

So notebook 08 writes loans at 80% of the **AVM value**, not 80% of the price paid. AVM
error becomes origination LTV error, mechanically and immediately. An AVM that is 8% too
generous on a segment writes 80% LTV loans that are really 87% LTV loans, and the difference
shows up only when prices fall.

The final step makes it a dollar figure: recompute LGD marking collateral at the
**lower bound of the AVM's own conformal interval** instead of at the point estimate. The
difference is the price of valuation uncertainty, in basis points of exposure. A risk
function that takes collateral at the point estimate is implicitly assuming the AVM is
exactly right.

---

## 11. Fair lending: disparity, adjustment, and the limit of public data

Notebook 04 tests the AVM for error parity across neighbourhoods and says explicitly that
this is **not** a fair-lending test: the De Cock file has no protected-class attribute, so
what it measures is the structural half of the mechanism the interagency rule cares about,
and no more. Notebook 09 runs the fair-lending test on a source that does carry those
attributes.

### Why a logistic model, and which controls

Denial is binary and the quantity of interest is a *ratio* of odds between groups holding
the file constant, which is what a logit coefficient estimates. Controls are the factors
a lender could defend in an examination: ability to repay (log income, DTI band), exposure
(log loan amount, LTV), product (purpose, lien status, occupancy), and year, because
credit conditions moved sharply across 2018–2024 and the applicant mix moved with them.

Three choices change the answer:

- **Rows missing a control are dropped, not imputed.** Imputing a borrower's LTV inside a
  discrimination model would be inventing the evidence that the model then weighs.
- **"Race not available" is excluded.** It is a missing-value code stored as a category.
  Including it as a group dilutes every estimate toward the population mean.
- **Groups under 100 are pooled, fitted, and not interpreted.** They still inform the
  controls. They cannot carry an inference, and the table marks them `reliable=False`
  instead of dropping them, because removing the smallest groups from a disparity
  analysis would hide results from the reader.

The model is fitted by **IRLS** (`GLM`, binomial) rather than Newton–Raphson. With ~25
dummy columns and a protected group of 150, the Newton Hessian is near-singular: the fit
fails to converge and still returns coefficients, so a failed fit can pass for a result.
The first version of this analysis did that.

### The Wilson interval, not the normal approximation

Denial rates are reported with Wilson score intervals. At 24.7% on 174 observations the
normal approximation is already distorted, and several groups here are far smaller. The
interval turns "Black applicants were denied at 24.7%" into "somewhere between 19% and
32%, and more data is needed to say where."

### What to do about the missing credit score

HMDA records no credit score, no assets, no reserves. Credit score is the strongest single
predictor of mortgage denial and correlates with race in the United States for reasons that
predate any individual lender. So the obvious objection to any adjusted disparity computed
from public HMDA is that the variable which actually explains it has been omitted.

Acknowledging the limitation does not say whether the objection is *plausible*. The
**E-value** does: the minimum association an unmeasured confounder would need (with both
the group and the outcome, beyond every control already fitted) to reduce the observed
association to nothing.

> VanderWeele, T.J. & Ding, P. (2017). "Sensitivity Analysis in Observational Research:
> Introducing the E-Value." *Annals of Internal Medicine* 167(4), 268–274.

Two details matter, and both are tested. The E-value for an interval uses the limit
**nearest the null**, which for a protective estimate is the upper one. And if the interval
contains 1, the answer is exactly 1: no confounding at all is needed to explain away a
result already consistent with no effect. Mapping the function over the lower bound, which
is the natural thing to write, asserts robustness for null findings.

What this establishes is that the disparity is not fragile: it would not be
erased by a weak confounder, and an explanation resting on credit score must argue for
roughly a 1.8-fold association with both race and denial beyond income, loan size, LTV,
DTI, purpose and year. That is a claim a reader can evaluate. It is **not** a finding of
discrimination. Disparate treatment is a legal conclusion requiring evidence of conduct,
and a lender's file contains information this one does not.

---

## 12. Known methodological limitations

Repeated in the README, stated here with their mechanism:

1. **No repeat sales in Ames.** Every property appears once, so a residual-based
   mispricing screen cannot be validated against realised returns. The screen is presented
   as a *screen*, never as a backtest.
2. **Rents are not contemporaneous with sales.** ZORI starts seven years after the last
   sale. See §9.
3. **The PD curve is uncalibrated.** See §10.
4. **28 neighborhoods × 5 years is a thin panel.** Notebook 08's neighborhood risk/return
   scatter computes a volatility from four annual growth rates. That is a very noisy
   estimate and the notebook says so at the point of use. It is a hypothesis generator,
   not a risk measurement.
5. **Ames is nearly acyclical.** That makes it a poor laboratory for the
   downside scenarios the credit work cares about, and is why the stress is imposed rather
   than observed.
6. **The fair-lending analysis cannot see the credit file.** See §11. The estimates are
   upper bounds on unexplained disparity, quantified by their E-values, and the Black
   applicant group is 174 applications over seven years.
7. **HMDA and the AVM do not overlap in time.** The sales the model is fitted on end in
   July 2010; the applications begin in 2018. Notebook 09 is evidence about the *market*
   an AVM like this would serve, not a validation of this AVM's decisions.
