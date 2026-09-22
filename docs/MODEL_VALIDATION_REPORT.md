# AVM Model Validation Report

**Model:** Ames Residential Automated Valuation Model (AVM) v1.0
**Model owner:** M. Yesmin · **Validation performed by:** M. Yesmin *(self-assessment; see §7.1)*
**Report date:** 20 September 2026 · **Model tier:** illustrative (not in production use)

---

## 0. Purpose and how to read this

This report is structured the way a bank's independent model validation function would
structure one, following the Federal Reserve / OCC *Supervisory Guidance on Model Risk
Management* (**SR 11-7**), which defines validation as three activities:

| SR 11-7 activity | Question | Where |
|---|---|---|
| **Conceptual soundness** | Is the design defensible? | §2 |
| **Ongoing monitoring** | Is it still performing? | §4 |
| **Outcomes analysis** | Do outputs match reality? | §3 |

§5 covers the June 2024 **Interagency AVM Rule**, §6 records findings with severity
ratings, and §7 states limitations.

Every figure cited is reproducible with `make all`. Supporting analysis is in the notebook
named against each section.

---

## 1. Model overview

### 1.1 Intended use

Estimate the market value of a single-family residential property in Ames, Iowa, from
property attributes, for three downstream uses: investment underwriting (notebook 07),
mortgage collateral valuation (notebook 08), and a client-facing valuation report
(notebook 10).

### 1.2 Approved use and limitations on use

The model is validated **only** for owner-occupied and small residential properties in
Ames, Iowa, sold in arm's-length transactions. It is **not** validated for:

- properties outside Ames (no data supports transfer);
- non-arm's-length transactions (explicitly excluded from the development sample; the
  screen in notebook 06 exists to detect them);
- properties above 4,000 sq ft of living area (five observations, excluded per De Cock
  2011 §4);
- any regulated use requiring an appraisal.

### 1.3 Model description

| | |
|---|---|
| **Type** | Gradient-boosted regression trees (LightGBM), benchmarked against Ridge, Lasso and ElasticNet |
| **Target** | `log(SalePrice)`, retransformed to dollars with the Duan (1983) smearing estimator |
| **Development sample** | 1,550 arm's-length Ames sales, January 2006 – December 2008 |
| **Test sample** | 862 arm's-length sales, January 2009 – July 2010 (out of time) |
| **Features** | 81 numeric + 20 categorical; 10 quality ladders ordinal-encoded; distance and bearing to ISU campus and downtown |
| **Uncertainty** | Split-conformal prediction interval, α = 0.20 |

### 1.4 Data

Primary source: De Cock (2011), *Journal of Statistics Education* 19(3), compiled from
Ames City Assessor records. Coordinates from the `AmesHousing` R package (99.59% join).
Eleven macroeconomic series from FRED and Zillow Research. Full provenance and verified
extents in [`DATA_SOURCES.md`](DATA_SOURCES.md); shape and range assertions in
`tests/test_data.py`.

---

## 2. Conceptual soundness

*Reference: [`METHODOLOGY.md`](METHODOLOGY.md), notebooks 01 and 03.*

### 2.1 Theory and specification — **Satisfactory**

Hedonic pricing is the standard approach for residential valuation and is appropriate
here. The log target is supported by evidence: skew falls from 1.74
to −0.01, and Levene's test across fitted quartiles rejects equal residual variance on the
raw-dollar specification decisively.

Duan smearing is applied on retransformation. The distribution-free form was chosen over
the lognormal correction `exp(σ²/2)` because the residuals are demonstrably non-Gaussian.

### 2.2 Variable selection and treatment — **Satisfactory**

Ordinal encoding of the 10 Ex/Gd/TA/Fa/Po quality ladders preserves ordering at the cost
of assuming equal spacing between grades. The assumption is checked in notebook 03: the
estimated price effect per grade step is approximately constant across the ladder. The
alternative (one-hot) imposes no structure but adds ~40 columns to a 1,550-row sample.

`NA` is treated as an informative level in 15 columns, per the source data dictionary.
Notebook 01 confirms the levels are price-relevant and that their effects run in *opposite*
directions (fireplace +43%, fence −18%), so pooling them as "missing" would destroy
information.

Transaction attributes (`Sale Type`, `Sale Condition`) are excluded from the feature set.
These do not exist at the moment an AVM is asked for a value; including them would use the
outcome to predict itself.

### 2.3 Sample design — **Satisfactory with observation**

Both a random and an out-of-time split are reported. The out-of-time split is the
governing one.

**Observation (see finding F-3).** The out-of-time test period is a weak stress. Ames
declined 4.3% peak-to-trough against 27.4% nationally, so the measured degradation is a
lower bound on performance in a cyclical market.

### 2.4 Implementation controls — **Satisfactory**

All learned statistics reside in a single `ColumnTransformer` fitted on development rows
only. Thirteen regression tests in `tests/test_leakage.py` check that no learned statistic can
depend on test rows. They include a test that fitted statistics are bit-identical with and
without test rows present, and a companion test confirming that a full-data fit does change
them.

---

## 3. Outcomes analysis

*Reference: notebooks 03 and 04.*

### 3.1 Accuracy against benchmark — **Below threshold on MdAPE**

| Metric | Out-of-time result | Institutional benchmark | Assessment |
|---|---:|---|---|
| MdAPE | **5.49%** | < 5% | **Does not meet** |
| PPE10 | **76.6%** | > 75% | Meets |
| PPE25 | 98.0% | — | — |
| FSD | 0.103 | — | — |
| Mean signed error | +0.45% | ≈ 0 | Meets |

Accuracy on the random split (MdAPE 5.15%) is 6.2% better in relative terms, consistent
with the mild regime shift noted in §2.3.

### 3.2 Benchmarking against the predecessor model — **Material improvement**

The predecessor (this project's first-pass submission) was scored against the true prices of
its 878 withheld observations, which had never previously been available:

| | Predecessor | Current model |
|---|---:|---:|
| MdAPE | 36.9% | **5.1%** |
| PPE10 | 13.9% | **76.9%** |
| RMSE | $112,154 | **$24,996** |
| R² | −0.90 | **0.91** |

An R² of −0.90 indicates the predecessor performed worse than a constant-mean predictor.
Root causes are documented as findings F-1 and F-2.

### 3.3 Uncertainty quantification — **Satisfactory with observation**

The 80% conformal interval is multiplicative, calibrated separately for five valuation
bands (Mondrian conformal), and re-fitted each quarter on the trailing twelve months of
closed sales. As served: 30% of the point estimate for the cheapest band, 21–26% through
the middle, 23% for the top, 25% on average.

Coverage was validated by resampling rather than asserted from a single holdout. Under
full resampling of the random design, mean coverage is **80.1%** against a nominal 80%, so
the guarantee holds. On a single holdout, coverage varies by ±3 percentage points, which
is larger than the binomial standard error alone and is attributable to the estimated
interval half-width.

**Observation (finding F-4).** Under the out-of-time split, coverage is **76.3%** for the
constant-width interval and **76.7%** for the per-band interval calibrated once (77.2%
averaged over 25 calibration splits). Recalibrated each quarter on the trailing twelve
months, the per-band interval backtests at **78.8%** (paired 95% interval for the gain
+0.8 to +3.4 points). The served bands are fitted on the latest window, so their coverage
rests on that backtest; no later sales exist to score them on. Conformal validity requires exchangeability between
calibration and scoring populations, which a temporal split violates by construction.

### 3.4 Error distribution and parity — **Satisfactory with observation**

Error is not uniform:

- **By price:** grouped by sale price, the bottom decile shows +10.2% and the top −3.8%.
  An unbiased-by-construction model shows +7.6% and −4.2% under the same grouping, and
  grouped by valuation the error stays within ±2.8%. The over-valuation that remains is about
  2–3% at the cheap end and nil at the top (finding F-9).
- **By neighbourhood:** mean signed error ranges from −4.0% (BrkSide) to +6.1% (Edwards),
  a spread of 10.1 percentage points.

**Observation (finding F-5).** Residuals exhibit statistically significant spatial
autocorrelation (Moran's I = 0.114, p < 1e-12, k = 8), indicating location value the model
has not captured.

---

## 4. Ongoing monitoring

*Reference: notebook 05.*

### 4.1 Framework — **Satisfactory**

Six controls with limits registered in code (`monitoring.MonitoringThresholds`) **before**
any monitoring output was produced:

| Control | Limit | Latest observation | Status |
|---|---|---:|---|
| MdAPE | > 6.5% | 5.49% | Pass |
| PPE10 | < 72% | 76.6% | Pass |
| \|Mean signed error\| | > 2% | 0.11% | Pass |
| Score PSI | > 0.25 | 0.005 | Pass |
| Worst feature PSI | > 0.25 | 0.094 | Pass |
| Interval coverage | < 72% | 77.5% | Pass |

### 4.2 Control effectiveness testing — **Satisfactory**

Passing controls do not show that the controls work, so the suite was tested against a
population it should reject: the 517 non-arm's-length sales excluded from development.
**Five of six controls breach**, confirming the controls are connected and sensitive.

Score stability is the one control that does not fire, because non-market properties are
structurally similar to market ones; only the transacted price differs. This is a
documented blind spot and the reason the pack monitors inputs, scores *and* outcomes.

### 4.3 Metric design issues identified and remediated

Two defects in the monitoring metric itself were found during construction and fixed:

1. **Deterministic features produced permanent false alarms.** `sale_time` scores PSI 11.3
   every period, because under a temporal split the sale date cannot overlap by
   construction. Such features are now excluded by name.
2. **PSI is not scale-free in sample size.** With ten bins and eight observations
   (2010Q3), PSI reported 7.55, all of it sampling noise. A minimum of ten expected
   observations per bin is now enforced, returning "not measurable" below it.

Both would have raised alarms unrelated to model performance and reduced the credibility
of the dashboard.

### 4.4 Champion / challenger — **Satisfactory**

A quarterly-retrained challenger was compared against the frozen champion under a
pre-registered promotion rule (win in ≥ 60% of periods **and** improve the metric on
average). The challenger won 50% of periods with a mean improvement of 0.08pp. **Not
promoted.**

---

## 5. Interagency AVM Rule compliance (June 2024)

*Reference: notebook 04 §3. Rule issued by OCC, Federal Reserve, FDIC, NCUA, CFPB, FHFA.*

| # | Factor | Implementation | Gap |
|---|---|---|---|
| 1 | **Confidence in estimates** | Per-property split-conformal interval; coverage validated by resampling; interval width propagated into LGD (nb 08) | Coverage 78.8% out of time with quarterly recalibration, against 80% nominal (F-4) |
| 2 | **Data manipulation protection** | Data contracts assert row counts, column sets, join rates and source date ranges; all inputs from named public endpoints; known errors corrected in code with tests | No cryptographic checksums on downloads |
| 3 | **Conflicts of interest** | No party holds a financial interest in outcomes. Structurally, transaction attributes are excluded from features | Organisational control, not a code control |
| 4 | **Random sample testing** | Both regimes score on held-out samples; `GroupKFold` holds out whole neighbourhoods in tuning; 304 automated tests | Sample sizes modest (603 / 862) |
| 5 | **Nondiscrimination** | Error-parity testing by neighbourhood and valuation band; neighbourhood-level bias disclosed on client reports | **Not a fair-lending test.** No protected-class attribute exists in the data (F-6) |

---

## 6. Findings

| ID | Finding | Severity | Status |
|---|---|---|---|
| **F-1** | Predecessor wrote training values into the test frame and fitted a second scaler on test data, invalidating all reported results | **High** | Remediated: structurally prevented; 13 regression tests |
| **F-2** | Predecessor selected hyperparameters on the full dataset and encoded categoricals separately per frame | **High** | Remediated: single pipeline, unified encoder |
| **F-3** | Out-of-time test period is a weak stress (Ames −4.3% vs US −27.4%) | **Medium** | Accepted and disclosed; credit stress imposed rather than observed |
| **F-4** | Conformal coverage degrades to 76.3% out of time versus 80% nominal, and the constant-width interval covers the cheapest decile at 51% | **Medium** | Largely remediated: Mondrian by valuation (cheapest decile 51% → 63%), recalibrated quarterly on the trailing twelve months (coverage 76.7% → 78.8% in backtest), in serving, the model card, the client report and the credit analysis; about one point of marginal gap remains |
| **F-5** | Residuals are spatially autocorrelated (Moran's I = 0.114) | **Medium** | Open: a time-cutoff spatial lag of prior prices was built and measured (nb 03 §9): Moran's I −18%, no accuracy gain. Structural features (school attendance boundaries) recommended next |
| **F-6** | Nondiscrimination testing cannot be completed without protected-class data | **Medium** | Accepted: structural testing performed; disclosed as a limitation |
| **F-7** | MdAPE of 5.49% does not meet the 5% institutional benchmark | **Low** | Accepted and disclosed; ~1,550 training observations in one market. A Ridge / LightGBM blend was measured (MdAPE 5.38%, gain inside noise) and not promoted |
| **F-8** | PD curve is an assumed functional form, not calibrated | **Low** | Accepted: no default outcomes exist in the data; figures presented as indicative |
| **F-9** | Price-band bias was measured by sale price and applied to valuations in client reports, overstating it roughly fourfold at the cheap end and inventing it at the top | **Medium** | Remediated: flag withdrawn, bias re-measured conditional on valuation, regression test added (Model Risk Log #11) |
| **F-10** | The valuation API imputed derived features (quality × area, total square feet, age) from neighbourhood medians instead of computing them from the caller's fields, so served valuations barely responded to size or quality | **High** | Remediated: derivation after imputation using the training code; omitted fields imputed rather than zeroed; regression tests including exact reproduction of training features (Model Risk Log #12) |

---

## 7. Limitations and conclusion

### 7.1 Independence

**This is a self-assessment.** The model developer and the validator are the same person.
In a regulated environment SR 11-7 requires validation by staff independent of model
development, with separate reporting lines. Nothing here substitutes for that.

### 7.2 Other limitations

- No repeat sales exist in the data, so no repeat-sales index can be constructed and the
  residual-based screen cannot be backtested against realised returns.
- Rent data (ZORI, 2017+) does not overlap the sales period (2006–2010). Notebook 07 is
  framed as present-day underwriting, not a historical backtest.
- The neighbourhood concentration analysis rests on four annual growth observations per
  neighbourhood, and is presented as a hypothesis generator rather than a measurement.

### 7.3 Conclusion

The model is **conceptually sound** and **materially outperforms its predecessor**, with
uncertainty quantification and a tested monitoring framework in place. It does **not** meet
the 5% MdAPE benchmark. Of its five medium-severity findings, F-3 and F-6 are accepted
limitations of the data, F-9 is remediated, F-4 is largely remediated, and F-5 is open.

**Recommendation:** approved for the illustrative and analytical uses in §1.2. F-5 should
be remediated before any production deployment, and F-4's remaining one-point coverage
gap monitored quarterly.

---

*Prepared following SR 11-7 (Federal Reserve SR Letter 11-7 / OCC Bulletin 2011-12) and the
Interagency Quality Control Standards for Automated Valuation Models, June 2024.*
