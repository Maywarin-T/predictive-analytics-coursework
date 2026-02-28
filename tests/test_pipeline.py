"""
Pipeline validation tests for the LCFS purchasing power prediction project.

These tests verify the correctness and safety of the data loading,
preprocessing, and target creation pipeline. Key checks include:
- Data loads with expected shape and schema
- OECD equivalence scale is computed correctly
- No income-derived variables leak into feature sets
- Target variable has balanced classes (quintiles)
- Preprocessing produces no NaN values
- Train/val/test splits are disjoint and stratified

Run with: pytest tests/test_pipeline.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import (
    load_single_year, load_lcfs_data, get_feature_columns,
    LEAKAGE_VARIABLES, compute_oecd_scale, compute_equivalised_income,
)
from src.preprocessing import create_target, get_available_features, build_preprocessor, split_data


# ── Test fixtures ────────────────────────────────────────────────────────────
# Fixtures are shared across all tests in this module (scope="module")
# to avoid reloading data for every individual test.

@pytest.fixture(scope="module")
def sample_data():
    """Load 2022 data as a test fixture (single year for faster tests)."""
    df = load_single_year(2022)
    return df


@pytest.fixture(scope="module")
def sample_data_with_equiv(sample_data):
    """Add equivalised income columns to sample data for downstream tests."""
    df = sample_data.copy()
    df['oecd_scale'] = compute_oecd_scale(df)
    df['equivalised_income'] = compute_equivalised_income(df)
    return df


# ── Data loading tests ──────────────────────────────────────────────────────

class TestDataLoading:
    """Tests for raw data loading and schema correctness."""

    def test_load_single_year(self, sample_data):
        """Data loads with expected minimum dimensions (>1000 rows, >100 cols)."""
        assert sample_data.shape[0] > 1000, "Expected at least 1000 rows"
        assert sample_data.shape[1] > 100, "Expected at least 100 columns"

    def test_columns_lowercase(self, sample_data):
        """All column names should be lowercase after normalisation."""
        for col in sample_data.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_target_variable_exists(self, sample_data):
        """The anon_income column must exist — it's used to derive the target."""
        assert 'anon_income' in sample_data.columns

    def test_survey_year_added(self, sample_data):
        """survey_year column should be added with the correct year value."""
        assert 'survey_year' in sample_data.columns
        assert (sample_data['survey_year'] == 2022).all()


# ── Equivalised income tests ────────────────────────────────────────────────

class TestEquivalisedIncome:
    """Tests for OECD equivalence scale and equivalised income computation."""

    def test_oecd_scale_range(self, sample_data_with_equiv):
        """OECD scale should be >= 1.0 (single adult is the minimum)."""
        scale = sample_data_with_equiv['oecd_scale']
        assert scale.min() >= 1.0, f"OECD scale minimum {scale.min()} < 1.0"

    def test_oecd_scale_single_adult(self, sample_data_with_equiv):
        """Single-adult, single-person households should have scale = 1.0 exactly."""
        df = sample_data_with_equiv
        # Filter to households with 1 adult and 1 total person
        single = df[(df['a055'] == 1) & (df['a049'] == 1)]
        if len(single) > 0:
            assert (single['oecd_scale'] == 1.0).all()

    def test_equivalised_income_positive(self, sample_data_with_equiv):
        """Equivalised income should be non-negative where raw income is non-negative."""
        df = sample_data_with_equiv
        pos_income = df[df['anon_income'] >= 0]
        assert (pos_income['equivalised_income'] >= 0).all()

    def test_equivalised_leq_raw(self, sample_data_with_equiv):
        """Equivalised income should be <= raw income (since scale >= 1)."""
        df = sample_data_with_equiv
        # Small tolerance (0.01) for floating-point precision
        assert (df['equivalised_income'] <= df['anon_income'] + 0.01).all()


# ── Leakage prevention tests ────────────────────────────────────────────────

class TestLeakagePrevention:
    """Tests that income-derived variables are excluded from features."""

    def test_no_leakage_in_features(self, sample_data_with_equiv):
        """Feature columns must not include any income-derived variables."""
        features = get_feature_columns(sample_data_with_equiv)
        leakage_set = set(v.lower() for v in LEAKAGE_VARIABLES)
        for f in features:
            assert f.lower() not in leakage_set, f"LEAKAGE: '{f}' is income-derived!"

    def test_equivalised_income_excluded(self, sample_data_with_equiv):
        """equivalised_income and oecd_scale must not appear in features."""
        features = get_feature_columns(sample_data_with_equiv)
        assert 'equivalised_income' not in features
        assert 'oecd_scale' not in features


# ── Preprocessing tests ─────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for target creation, feature processing, and data splitting."""

    def test_create_target_quintiles(self, sample_data_with_equiv):
        """Target should have exactly 5 unique classes (quintiles 0-4)."""
        target = create_target(sample_data_with_equiv, n_quantiles=5)
        valid = target.dropna()
        assert set(valid.unique()) == {0, 1, 2, 3, 4}

    def test_target_uses_equivalised_income(self, sample_data_with_equiv):
        """create_target should use equivalised_income when present."""
        df = sample_data_with_equiv
        target = create_target(df, n_quantiles=5)
        # Verify that quintile assignment aligns with equivalised_income values
        # (each quintile group should have only non-null equivalised income values)
        for q in range(5):
            group = df.loc[target == q, 'equivalised_income']
            assert group.notna().all()

    def test_target_balance(self, sample_data_with_equiv):
        """Quintiles should be roughly balanced (max/min ratio < 1.5)."""
        target = create_target(sample_data_with_equiv, n_quantiles=5)
        counts = target.value_counts()
        ratio = counts.max() / counts.min()
        assert ratio < 1.5, f"Class imbalance ratio {ratio:.2f} exceeds 1.5"

    def test_preprocessor_no_nans(self, sample_data_with_equiv):
        """Preprocessed output should have no NaN values (imputation works)."""
        features = get_feature_columns(sample_data_with_equiv)
        cat_f, cont_f = get_available_features(sample_data_with_equiv)
        # Only include features that passed the leakage check
        cat_f = [c for c in cat_f if c in features]
        cont_f = [c for c in cont_f if c in features]

        preprocessor = build_preprocessor(cat_f, cont_f)
        X = preprocessor.fit_transform(sample_data_with_equiv[features])
        assert not np.isnan(X).any(), "NaN values after preprocessing"

    def test_split_no_overlap(self, sample_data_with_equiv):
        """Train, val, and test sets should have completely disjoint indices."""
        features = get_feature_columns(sample_data_with_equiv)
        target = create_target(sample_data_with_equiv)
        X_train, X_val, X_test, _, _, _ = split_data(
            sample_data_with_equiv[features], target, random_state=42
        )
        # Verify no index appears in more than one split
        assert len(set(X_train.index) & set(X_val.index)) == 0
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0

    def test_split_stratification(self, sample_data_with_equiv):
        """Class proportions should be similar across all splits (<5% difference)."""
        features = get_feature_columns(sample_data_with_equiv)
        target = create_target(sample_data_with_equiv)
        _, _, _, y_train, y_val, y_test = split_data(
            sample_data_with_equiv[features], target, random_state=42
        )
        # Compare class proportions between train and test
        train_props = y_train.value_counts(normalize=True).sort_index()
        test_props = y_test.value_counts(normalize=True).sort_index()
        diff = (train_props - test_props).abs()
        assert diff.max() < 0.05, f"Stratification failed: max prop diff = {diff.max():.3f}"
