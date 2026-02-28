"""
Preprocessing pipeline for LCFS income prediction task.

This module provides the full data preparation pipeline:
1. Target creation — bin equivalised income into quantile-based classes
2. Feature identification — separate categorical vs continuous columns
3. Preprocessing — imputation, scaling, and encoding via sklearn pipelines
4. Train/val/test splitting — stratified to preserve class balance
5. PCA — optional dimensionality reduction on continuous features
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# ── Custom transformer ──────────────────────────────────────────────────────

class CoerceNumeric(BaseEstimator, TransformerMixin):
    """
    Convert columns to numeric, replacing non-numeric values with NaN.

    This handles the case where survey data contains mixed types
    (e.g. empty strings or coded text values in otherwise numeric columns).
    It is used as the first step in the continuous feature pipeline,
    before imputation fills in the resulting NaN values.
    """
    def fit(self, X, y=None):
        # No fitting needed — this is a stateless transformer
        return self

    def transform(self, X):
        # Convert input to DataFrame, apply pd.to_numeric column-wise,
        # coercing any non-numeric values to NaN for downstream imputation
        df = pd.DataFrame(X)
        return df.apply(pd.to_numeric, errors='coerce').values


# ── Feature type definitions ────────────────────────────────────────────────
# These lists define which features should receive categorical vs continuous
# preprocessing. Categorical features are one-hot encoded; continuous features
# are scaled to zero mean and unit variance.

# Categorical features — treated as discrete groups, one-hot encoded
CATEGORICAL_FEATURES = ['gorx', 'sexhrp', 'a121', 'a099', 'a103']

# Continuous features — treated as numeric values, standardised
# Includes all expenditure categories (p-codes) plus numeric demographics
CONTINUOUS_FEATURES = [
    'p600', 'p530', 'p531', 'p532', 'p533', 'p534',
    'p535p', 'p536p', 'p537', 'p538', 'p539', 'p540',
    'p541', 'p542', 'p543', 'p544',
    'a049', 'a055', 'a116',  # household size, num adults, age of HRP
]


# ── Target variable creation ────────────────────────────────────────────────

def create_target(df: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """
    Bin equivalised income (purchasing power) into quantile-based classes.

    Uses pd.qcut to create equal-frequency bins, ensuring roughly balanced
    class sizes. This is preferable to equal-width bins because income
    distributions are typically right-skewed.

    Uses 'equivalised_income' column if present; otherwise falls back to
    'anon_income' for backward compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'equivalised_income' or 'anon_income' column.
    n_quantiles : int
        Number of income bins (default 5 = quintiles).

    Returns
    -------
    pd.Series
        Integer labels 0 to n_quantiles-1, where 0 = lowest purchasing power.
    """
    # Prefer equivalised income (adjusted for household size) over raw income
    if 'equivalised_income' in df.columns:
        income_col = 'equivalised_income'
    else:
        income_col = 'anon_income'

    # qcut creates quantile-based bins with approximately equal sample counts
    # duplicates='drop' handles cases where quantile boundaries coincide
    labels = pd.qcut(df[income_col], q=n_quantiles, labels=False, duplicates='drop')
    return labels


# ── Feature identification ──────────────────────────────────────────────────

def get_available_features(df: pd.DataFrame):
    """
    Identify which categorical and continuous features exist in the dataframe.

    Not all survey years contain all variables. This function filters the
    pre-defined feature lists to only include columns present in the data,
    preventing KeyError during preprocessing.

    Returns
    -------
    cat_features : list
        Categorical feature column names present in df.
    cont_features : list
        Continuous feature column names present in df.
    """
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    cont = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    return cat, cont


# ── Preprocessing pipeline ──────────────────────────────────────────────────

def build_preprocessor(cat_features: list, cont_features: list) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for the preprocessing pipeline.

    The pipeline applies different transformations to continuous vs categorical
    features:

    Continuous pipeline (3 steps):
        1. CoerceNumeric — convert any non-numeric values to NaN
        2. SimpleImputer(median) — fill NaN with column median
        3. StandardScaler — standardise to zero mean, unit variance

    Categorical pipeline (2 steps):
        1. SimpleImputer(most_frequent) — fill missing with mode
        2. OneHotEncoder — create binary dummy variables

    Columns not in either list are dropped (remainder='drop') to prevent
    leakage variables or irrelevant columns from entering the model.
    """
    # Pipeline for continuous/numeric features
    continuous_pipeline = Pipeline([
        ('coerce', CoerceNumeric()),         # handle mixed types (e.g. empty strings)
        ('imputer', SimpleImputer(strategy='median')),  # robust to outliers
        ('scaler', StandardScaler()),         # normalise for distance-based models
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # fill with mode
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # Combine into a ColumnTransformer that routes each feature type
    # to its appropriate pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_pipeline, cont_features),   # continuous → scale
            ('cat', categorical_pipeline, cat_features),   # categorical → one-hot
        ],
        remainder='drop',  # drop any columns not explicitly listed
    )
    return preprocessor


# ── Train / Validation / Test splitting ─────────────────────────────────────

def split_data(df: pd.DataFrame, target: pd.Series,
               test_size: float = 0.15, val_size: float = 0.15,
               random_state: int = 42):
    """
    Stratified train/validation/test split.

    Performs a two-stage split to create three disjoint sets:
        1. First split: separate test set (15%) from the rest
        2. Second split: separate validation set (15%) from train

    Stratification ensures each split has approximately the same class
    distribution as the full dataset, preventing imbalanced folds.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    target : pd.Series
        Target labels (must be aligned with df index).
    test_size : float
        Proportion for test set (default 0.15).
    val_size : float
        Proportion for validation set (default 0.15).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Remove rows where target is NaN (e.g. from qcut failing on edge cases)
    valid_mask = target.notna()
    df_clean = df.loc[valid_mask].copy()
    target_clean = target.loc[valid_mask].copy()

    # Stage 1: Split off the held-out test set (stratified by target class)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df_clean, target_clean,
        test_size=test_size,
        stratify=target_clean,
        random_state=random_state,
    )

    # Stage 2: Split remaining data into train and validation sets
    # val_fraction is adjusted because we're splitting from the reduced pool
    # e.g. if val_size=0.15 and test_size=0.15, val_fraction = 0.15/0.85 ≈ 0.176
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── PCA dimensionality reduction ────────────────────────────────────────────

def apply_pca(X_train_scaled: np.ndarray, X_val_scaled: np.ndarray,
              X_test_scaled: np.ndarray, n_components: float = 0.95):
    """
    Apply PCA to pre-scaled data, retaining components that explain
    `n_components` fraction of variance (default 95%).

    PCA is fitted ONLY on training data to prevent data leakage.
    The same transformation is then applied to validation and test sets.

    Returns
    -------
    X_train_pca, X_val_pca, X_test_pca, pca_model
        Transformed arrays and the fitted PCA model for inspection.
    """
    pca = PCA(n_components=n_components, random_state=42)

    # Fit on training data only, then transform all splits
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA: {X_train_scaled.shape[1]} features → {pca.n_components_} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance explained)")
    return X_train_pca, X_val_pca, X_test_pca, pca
