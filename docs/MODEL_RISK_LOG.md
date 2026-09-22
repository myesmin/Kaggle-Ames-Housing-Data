# Model Risk Log

Ten defects in this project's first-pass notebooks, found by reading them line by line
before building anything on top. Each is recorded here with its location, its mechanism,
its impact, and how the current pipeline addresses it.

A model-validation review checks more than whether a notebook runs. It asks what would
have to be true for a number to be wrong, and then checks whether it is. Seven of the ten
below are invisible in the notebook's own output: the model ran, produced plausible
figures and scored, and nothing raised an error.

Those first-pass notebooks are kept in [`archive/notebooks/`](../archive/notebooks) with
their code untouched, so every cell referenced here can be checked against the source it
came from.

---

## Severity 1 — invalidates the reported results

### #1 — Train values written into the test frame

**Where:** `2. Data Cleaning.ipynb`, cell 27

```python
ames_train[features_num] = ames_train[features_num].fillna(0)
kaggle_test[features_num] = ames_train[features_num].fillna(0)   # <- ames_train
ames_train[features_cat] = ames_train[features_cat].fillna('null')
kaggle_test[features_cat] = ames_train[features_cat].fillna('null')   # <- ames_train
```

Both assignments to `kaggle_test` read from `ames_train`. Pandas aligns on index, so the
878 test rows were overwritten with the first 878 **training** rows for every numeric and
every categorical column. The frame was then written to `datasets/kaggle_test.csv` and
used for the submission.

**Impact.** The test frame no longer held test data. Every downstream number computed on
it, including the submitted predictions, describes training data under a test label.
Because the code raised no error and the shapes matched, nothing in the notebook's output
revealed it.

**Fix.** Train and test are never separate dataframes. One `ColumnTransformer` is fit on
training rows and `transform`-ed onto test rows; there is no assignment across frames to
get wrong. `tests/test_leakage.py::test_transform_of_a_row_is_independent_of_its_companions`
asserts that a row transforms identically alone and in company, which fails immediately
under any row-aligned contamination.

---

### #2 — A second `StandardScaler` fit on the test set

**Where:** `3. Feature selection, Feature engineering and Model Selection.ipynb`, cell 29

```python
ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc  = ss.transform(X_test)      # correct

ss = StandardScaler()                   # a new scaler
kaggle_test_sc = ss.fit_transform(kaggle_test)   # fit on the test set
```

**Impact.** Submission features were centred on the *test set's own* means and scaled by
its own standard deviations. The model's coefficients were estimated against the training
moments, so predictions were computed in the wrong units. The result was an affine distortion
of every submitted value, largest for features whose distribution differs most between the
two sets.

**Fix.** Exactly one scaler exists, inside the pipeline.
`tests/test_leakage.py::test_only_one_scaler_exists_in_the_pipeline` asserts that calling
`transform` on the test set leaves the fitted moments untouched, and
`test_test_rows_are_standardised_to_train_moments_not_their_own` asserts the observable
consequence: training rows centre on zero, test rows do not.

---

### #3 — Hyperparameter selection on the full dataset

**Where:** nb 3, cell 31: `lasso.fit(X, y)`, where `X`/`y` are the *undivided* dataset,
before the train/test split is applied at cell 27's variables.

**Impact.** `LassoCV` selects `alpha` by cross-validation. Run on all rows, the selected
penalty is informed by the held-out rows, so the reported test score is optimistic by an
unknown amount. This is subtler than a direct leak, since only one scalar crosses the
boundary, but it is enough to contaminate the test score.

**Fix.** All tuning happens inside `GridSearchCV` on a `Pipeline`, fit only on training
rows. Preprocessing is re-fit within every CV fold, so no fold's statistics leak into its
own validation split.

---

### #4 — `get_dummies` run separately on train and test

**Where:** nb 3, cells 16–23

```python
ames_train  = pd.get_dummies(ames_train,  columns=features_cat, drop_first=True)
kaggle_test = pd.get_dummies(kaggle_test, columns=features_cat, drop_first=True)
missing_cols = set(ames_train.columns) - set(kaggle_test.columns)
for c in missing_cols:
    kaggle_test[c] = 0
kaggle_test = kaggle_test[features_model]
```

**Impact.** Two separate encodings produce two different vocabularies. Levels present only
in test are dropped (the back-fill loop only adds columns test *lacks*, never
handles columns test has *extra*). And `drop_first=True` applied independently can drop a
*different* baseline level in each frame, so surviving columns no longer mean the same
thing in the two matrices.

**Fix.** A single `OneHotEncoder` with `handle_unknown="infrequent_if_exist"`, fit on
training rows. Unseen levels route to an infrequent bucket instead of vanishing, and
column order is fixed by the training fit. Three tests cover it:
`test_column_count_is_fixed_by_the_training_vocabulary`,
`test_a_category_seen_only_at_predict_time_is_handled_not_dropped`, and
`test_column_order_is_stable_across_frames`.

---

## Severity 2 — invalidates the model design

### #5 — EDA established the log target; the models used raw dollars

**Where:** nb 1 cell 11 concludes *"The Log-Transformed Sale Price Distribution exhibits a
more normal distribution… a beneficial preprocessing step"*. nb 3 cell 27 then sets
`y = ames_train['SalePrice']`.

**Impact.** Squared-error loss on raw dollars weights a $600k home roughly 16× a $150k
home. The residuals fan out with the fitted value (visible in the project's own
`Graph/Homoscedacity.png`), so RMSE is dominated by the expensive tail, coefficient
standard errors are wrong, and the model is systematically mis-specified in the price
range where most of the book actually sits.

**Fix.** The target is `log(SalePrice)` throughout. Notebook 01 quantifies the difference
(skew 1.74 → −0.01) and Levene's test across fitted quartiles confirms the
heteroskedasticity. Retransformation to dollars uses the Duan (1983) smearing estimate,
because naively exponentiating a log prediction returns the conditional *median*, biasing
every dollar figure low.

---

### #6 — `test_size=0.05`

**Where:** nb 3, cell 27: a 103-row test set.

**Impact.** The standard error on an RMSE from 103 observations is large enough that the
difference between two candidate models is usually indistinguishable from noise. The test
score could not function as a model-selection signal, which is what it was used for.

**Fix.** 25% for the random regime (603 rows) and the full 2009–2010 period for the
temporal regime (862 arm's-length rows).
`tests/test_leakage.py::test_the_test_set_is_large_enough_to_be_a_model_selection_signal`
enforces a floor.

---

### #7 — A random split over data spanning 2006–2010

**Where:** nb 3, cell 27: `train_test_split(X, y, random_state=42, test_size=0.05)`.

**Impact.** Sales run from January 2006 to July 2010. A random split trains on 2010 to
predict 2006, so the model sees the future. For a valuation model this matters because an
AVM in production is *always* asked about a date later than its training data, so a random
split measures the wrong quantity.

**Fix.** Both regimes are reported side by side, and the gap between them is presented as
the project's headline finding. Notebook 02 adds the necessary caveat:
Ames fell only 4.3% peak-to-trough against 27.4% nationally, so this temporal split is a
mild regime shift and the measured degradation is a **lower bound** on what a cyclical
market would show.

---

## Severity 3 — data quality

### #8 — Known data errors left in, corrections commented out

**Where:** nb 2, cells 12 and 15

```python
#train_data.drop(train_data[train_data['Gr Liv Area'] > 4000].index, inplace=True);
#train_data['Garage Yr Blt'].replace({2207: 2007}, inplace=True)
```

Both lines were written, then commented out. Neither ran.

**Impact.** A garage recorded as built in **2207** stayed in the training data, 197 years
after the house sold. Five homes above 4,000 sq ft stayed in, three of which are `Partial`
sales of unfinished construction that De Cock's paper explicitly recommends removing
because they are not market prices.

**Fix.** `ames.data.clean_known_errors` applies the typo correction and *flags* the
five outliers instead of dropping them, so the exclusion is visible in the data and
reversible. `tests/test_data.py::test_known_data_errors_are_corrected` asserts both.

---

### #9 — Five informative columns dropped for high nullity

**Where:** nb 2, cell 26

```python
drop_cols = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu']
```

**Impact.** In De Cock's data dictionary, `NA` in these columns does not mean "missing".
It means **"No Pool"**, **"None"**, **"No Alley Access"**, **"No Fence"**, **"No
Fireplace"**. The nulls are fully-observed information. Dropping `Fireplace Qu` alone
discarded a quality ladder on the 1,508 homes (51% of the dataset) that have a fireplace,
and those homes sell for **43% more** than homes without one.

Notebook 01 also shows the effects run in *different directions*: a fence is associated
with **18% lower** prices in Ames. It acts as a proxy for smaller, denser lots.
Treating all five as "missing, impute the mode" would have erased that.

**Fix.** 11 ordinal and 4 nominal columns get an explicit "absent" level, encoded as `0`
at the bottom of the quality ladder so the scale stays monotone: no basement < poor
basement < excellent basement. Nothing is dropped for nullity.

---

### #10 — In-place mutation makes the notebook non-re-runnable

**Where:** nb 1, cells 28–32

```python
train_data[numeric_columns] = np.log1p(train_data[numeric_columns])   # cell 28
...
train_data = pd.read_csv("datasets/train.csv")                        # cell 32, re-read
```

**Impact.** Cell 28 log-transforms every numeric column in place, including the target.
Cell 32 re-reads the CSV to undo it. Running any cell out of order, or the whole notebook
twice, double-log-transforms the data without raising an error. The notebook is not reproducible from a
clean kernel, which is the minimum bar for a result to be checkable. Cell 29 also
references an undefined name `train`, so the notebook as committed cannot run top to
bottom at all.

**Fix.** Every transformation returns a new frame; `ames.features.prepare` is pure.
`make run` executes all ten notebooks from a clean kernel in order and fails the build on
any error, so "it runs top to bottom" is checked on every build.

---

## Found in the rebuilt pipeline

Defects 1–10 were in the first-pass notebooks. These two were introduced by the rebuild
itself and caught by its own later diagnostics. They are logged on the same terms.

### #11 — Price-band bias measured on the outcome, applied on the valuation

**What it was.** Notebooks 03 and 04 reported that the model over-values the cheapest
price decile by 10.2% and under-values the most expensive by 3.8%, and treated this as a
model defect with credit and fair-lending consequences. `ames.report.build_flags` turned it
into client advice: a valuation under $110,000 carried the note *"this model reads about
10% high — consider the lower end of the range"*, and one over $320,000 the reverse.

**Why it was wrong.** The deciles grouped errors by **sale price**, the quantity being
estimated. A sale lands in the cheapest decile partly because its price came in below its
valuation, so grouping on the outcome selects on the error. A predictor that is unbiased
by construction, carrying this model's measured error spread, reproduces most of the
pattern: +7.6% at the bottom and −4.2% at the top. Grouped by the **valuation**, the only
grouping observable when a valuation is issued, the mean error stays within ±2.8% across
all ten deciles with no trend.

| | Observed, by sale price | Unbiased model, by sale price | Unexplained |
|---|---:|---:|---:|
| Cheapest decile | +10.2% | +5.6% to +7.6% | +2.7% |
| Most expensive decile | −3.8% | −4.2% | +0.4% |

**Impact.** The client flag keyed on the valuation but quoted the sale-price figure, so it
advised shading cheap valuations down by about 10% and expensive ones up by about 4%, to
correct a bias that does not exist conditional on the valuation. It also overstated the
credit channel: a lender observes the valuation, not the eventual price, so there is no
systematic price-band collateral mis-statement to transmit.

**Fix.** The flag is removed, with a comment in `report.py` recording why. A regression test
(`test_price_extremes_carry_no_directional_warning`) now asserts its absence. The 10.2% /
−3.8% figures are withdrawn across the notebooks, the validation report and the README;
the figure carried forward is the unexplained over-valuation of roughly 2–3% at the cheap
end. The neighbourhood flag is retained, because it conditions on and quotes the same
variable. A bias estimate can be used for correction only if it is conditioned on
information available at the point of use. Grouping by the outcome is a legitimate way to
ask *how often a true value lands in its range*, but it does not measure what to correct.

### #12 — The service imputed derived features instead of computing them

**What it was.** `POST /value` takes ten fields and fills the other ninety-odd model inputs
from the neighbourhood's median (numeric) or mode (categorical). Twenty-two of those inputs
are *derived*: `qual_x_sf` (quality × living area), `total_sf`, `house_age`, the
has-garage/has-basement indicators, distances to campus and downtown. They were filled
with the neighbourhood median like everything else, so they never reflected what the
caller supplied. The component areas had the same problem: a 2,600 sq ft house kept the
neighbourhood's median first-floor area, and a house with no garage kept a 418 sq ft one.

Separately, the request schema defaulted omitted fields to **zero**, so a caller who did
not mention a garage was describing a house without one, and `fields_supplied` reported
ten whatever was sent.

**Impact.** Within a neighbourhood the valuation barely responded to size or quality. In
NAmes a 900 sq ft quality-4 house and a 2,600 sq ft quality-9 house were valued at $132k
and $143k. The notebooks and the client reports were unaffected (they score complete rows
from the property frame), which is why no accuracy metric showed it: the defect lived only
in the path from ten caller-supplied fields to a full model row.

**Fix.** The service now reconciles imputed fields with supplied ones (component areas
rescaled to the supplied totals; a supplied zero removes the garage or basement) and then
recomputes every derived feature with the same `add_engineered` / `add_geo_features`
functions training uses. Omitted fields default to *unknown* and are imputed; only fields
actually sent count as supplied. The same two houses now value at $101k and $269k.
Regression tests in `tests/test_service.py` assert the derived features follow the supplied
fields, that the service's derivation reproduces the training features exactly for all
2,412 sold properties, and that an omitted field is imputed rather than zeroed.
Anything computed from other fields is now recomputed after imputation, from the completed row, by the code that
computed it at training time.

---

## Summary

| # | Defect | Severity | Structural fix |
|---|---|---|---|
| 1 | Train values written into the test frame | Invalidates results | Single `ColumnTransformer`; row-locality test |
| 2 | Second `StandardScaler` fit on test | Invalidates results | One scaler, inside the pipeline |
| 3 | `alpha` selected on the full dataset | Invalidates results | `GridSearchCV` on a `Pipeline`, train rows only |
| 4 | `get_dummies` run separately per frame | Invalidates results | One `OneHotEncoder`, `handle_unknown` set |
| 5 | Raw-dollar target despite log-normal EDA | Invalidates design | `log(SalePrice)` + Duan smearing |
| 6 | 103-row test set | Invalidates design | 25% / full 2009–10 period |
| 7 | Random split across 2006–2010 | Invalidates design | Temporal split reported beside random |
| 8 | Data-error fixes commented out | Data quality | `clean_known_errors`, asserted in tests |
| 9 | Informative nulls dropped | Data quality | Explicit "absent" level on 15 columns |
| 10 | In-place mutation mid-notebook | Reproducibility | Pure functions; `make run` from a clean kernel |
| 11 | Price-band bias measured on the outcome, applied on the valuation | Misleading client output | Flag removed; bias re-measured by valuation; regression test |
| 12 | Service imputed derived features instead of computing them | Wrong served valuations | Derive after imputation with the training code; omitted = unknown; regression tests |

Defects 1–4 are covered by 13 regression tests in `tests/test_leakage.py`; 8 and 9 by
`tests/test_data.py`; 11 by `tests/test_report.py`; 12 by `tests/test_service.py`.
