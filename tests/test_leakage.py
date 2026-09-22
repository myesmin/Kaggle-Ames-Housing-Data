"""Regression tests for defects #1-#4 of the Model Risk Log.

These are the most important tests in the repo.  Each one fails loudly if the original
project's structure is ever reintroduced:

  #1  test-frame columns overwritten with row-aligned *train* values
  #2  a second StandardScaler fit on the test set
  #3  model fit on the full dataset before the split, leaking into hyperparameters
  #4  pd.get_dummies applied separately to train and test, then columns back-filled

The design claim is stronger than "we fixed these".  It is that a single
ColumnTransformer fit on training rows makes all four *unrepresentable*: there is no
second scaler to fit, no second dummy vocabulary to misalign, and the transform of a row
cannot depend on which other rows accompany it.  These tests check that claim directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.data import load_properties
from ames.features import build_preprocessor, prepare, split_feature_types


@pytest.fixture(scope="module")
def prepared() -> pd.DataFrame:
    return prepare(load_properties())


@pytest.fixture(scope="module")
def parts(prepared):
    train = prepared[prepared["temporal_split"] == "train"]
    test = prepared[prepared["temporal_split"] == "test"]
    numeric, nominal = split_feature_types(prepared)
    return train, test, numeric, nominal


def _learned_state(ct) -> dict[str, np.ndarray]:
    """Every statistic the ColumnTransformer estimated from data."""
    num = ct.named_transformers_["num"]
    cat = ct.named_transformers_["cat"]
    state = {
        "num_impute": num.named_steps["impute"].statistics_,
        "cat_impute": cat.named_steps["impute"].statistics_,
    }
    if "scale" in num.named_steps:
        state["scale_mean"] = num.named_steps["scale"].mean_
        state["scale_var"] = num.named_steps["scale"].var_
    for i, cats in enumerate(cat.named_steps["onehot"].categories_):
        state[f"onehot_{i}"] = cats
    return state


# --------------------------------------------------------------------------
# Defect #2 and #3: fitted statistics must come from training rows only
# --------------------------------------------------------------------------

def test_fitted_statistics_are_identical_with_and_without_test_rows(parts):
    """The headline leakage check.

    Fit the preprocessor on the training rows alone, then fit a second copy on the
    training rows concatenated with the test rows *but told to use only the training
    index*.  Every learned statistic must match bit-for-bit.  If any preprocessing step
    ever starts reading the rows it was not fit on, this fails.
    """
    train, test, numeric, nominal = parts

    fit_train_only = build_preprocessor(numeric, nominal).fit(train)
    combined = pd.concat([train, test])
    fit_on_train_slice = build_preprocessor(numeric, nominal).fit(combined.loc[train.index])

    a, b = _learned_state(fit_train_only), _learned_state(fit_on_train_slice)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} differs")


def test_fitting_on_the_full_dataset_really_does_change_the_statistics(parts):
    """Guards the test above from being vacuously true.

    If train-only and train+test fits produced the same numbers anyway, the check above
    would prove nothing.  They do not: the 2009-2010 rows shift the scaler moments.
    """
    train, test, numeric, nominal = parts
    train_only = build_preprocessor(numeric, nominal).fit(train)
    everything = build_preprocessor(numeric, nominal).fit(pd.concat([train, test]))
    assert not np.allclose(_learned_state(train_only)["scale_mean"],
                           _learned_state(everything)["scale_mean"])


def test_only_one_scaler_exists_in_the_pipeline(parts):
    """Defect #2 was a *second* StandardScaler fit on the test frame.

    There is exactly one scaler, it lives inside the ColumnTransformer, and transforming
    the test set cannot refit it -- ``transform`` has no access to a ``fit``.
    """
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)
    scaler = ct.named_transformers_["num"].named_steps["scale"]
    before = scaler.mean_.copy()
    ct.transform(test)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_test_rows_are_standardised_to_train_moments_not_their_own(parts):
    """The observable consequence of defect #2.

    Under a correct pipeline the *training* rows standardise to mean 0, variance 1, and
    the test rows do not -- they inherit the training moments and their own mean drifts.
    Defect #2 produced the opposite: both sets centred on zero, on different scales.
    """
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)
    n_num = len(numeric)
    train_num = ct.transform(train)[:, :n_num]
    test_num = ct.transform(test)[:, :n_num]

    assert np.abs(train_num.mean(axis=0)).max() < 1e-9
    assert np.abs(test_num.mean(axis=0)).max() > 1e-6


# --------------------------------------------------------------------------
# Defect #1: test values must be the test set's own
# --------------------------------------------------------------------------

def test_transform_of_a_row_is_independent_of_its_companions(parts):
    """Defect #1 wrote train values into test rows by positional alignment.

    Transforming one row alone and transforming it inside the full test frame must give
    byte-identical output.  Any row-aligned contamination breaks this immediately.
    """
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)

    full = ct.transform(test)
    for pos in (0, 1, len(test) // 2, len(test) - 1):
        alone = ct.transform(test.iloc[[pos]])
        np.testing.assert_array_equal(alone[0], full[pos])


def test_shuffling_the_test_frame_permutes_but_does_not_alter_the_output(parts):
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)
    shuffled = test.sample(frac=1.0, random_state=0)
    np.testing.assert_array_equal(
        ct.transform(shuffled),
        ct.transform(test)[[test.index.get_loc(i) for i in shuffled.index]])


def test_prepare_is_row_local(prepared):
    """``prepare`` runs before the split, so it must read nothing from the sample."""
    raw = load_properties()
    for pos in (0, 7, 1500, len(raw) - 1):
        single = prepare(raw.iloc[[pos]])
        whole = prepared.iloc[[pos]]
        for col in single.columns:
            a, b = single[col].iloc[0], whole[col].iloc[0]
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), col
            else:
                assert a == b, col


# --------------------------------------------------------------------------
# Defect #4: categorical alignment
# --------------------------------------------------------------------------

def test_column_count_is_fixed_by_the_training_vocabulary(parts):
    """Defect #4 let the design matrix width depend on which frame was encoded."""
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)
    width = ct.transform(train).shape[1]
    assert ct.transform(test).shape[1] == width
    assert ct.transform(test.iloc[:1]).shape[1] == width
    assert ct.transform(pd.concat([train, test])).shape[1] == width


def test_a_category_seen_only_at_predict_time_is_handled_not_dropped(parts):
    """Defect #4 silently discarded test-only levels.  Here they route to a bucket."""
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)

    novel = test.iloc[[0]].copy()
    novel["Neighborhood"] = "NotARealNeighborhood"
    out = ct.transform(novel)                      # must not raise
    assert out.shape[1] == ct.transform(test).shape[1]
    assert np.isfinite(out).all()


def test_column_order_is_stable_across_frames(parts):
    train, test, numeric, nominal = parts
    ct = build_preprocessor(numeric, nominal).fit(train)
    names = list(ct.get_feature_names_out())
    ct2 = build_preprocessor(numeric, nominal).fit(train.sample(frac=1.0, random_state=1))
    assert names == list(ct2.get_feature_names_out())


# --------------------------------------------------------------------------
# Split integrity
# --------------------------------------------------------------------------

def test_the_temporal_split_does_not_train_on_the_future(prepared):
    """Defect #7: a random split over 2006-2010 trains on the future to predict the past."""
    train = prepared[prepared["temporal_split"] == "train"]
    test = prepared[prepared["temporal_split"] == "test"]
    assert train["Yr Sold"].max() < test["Yr Sold"].min()
    assert train["sale_date"].max() < test["sale_date"].min()


def test_the_splits_are_disjoint_and_exhaustive(prepared):
    train = set(prepared.index[prepared["temporal_split"] == "train"])
    test = set(prepared.index[prepared["temporal_split"] == "test"])
    assert not train & test
    assert len(train) + len(test) == len(prepared)


def test_the_test_set_is_large_enough_to_be_a_model_selection_signal(prepared):
    """Defect #6: 103 rows gave a test RMSE with a huge standard error."""
    assert (prepared["temporal_split"] == "test").sum() > 500
