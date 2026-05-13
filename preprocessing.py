# =============================================================================
# preprocessing.py — Core ML Preprocessing Pipeline
# =============================================================================
# This module does all the heavy lifting:
#   1. Data cleaning  (duplicates, missing values, type fixes, text normalisation)
#   2. Outlier detection & handling  (IQR and Z-score)
#   3. Feature/target split, encoding, scaling, train-val-test split
#   4. Overfitting-prevention helpers  (correlation pruning, PCA, feature selection)
#
# Every function returns BOTH the transformed DataFrame and a human-readable
# "report" dict so the Streamlit UI can display exactly what changed.
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from utils import (
    get_numeric_columns,
    get_categorical_columns,
    try_cast_to_numeric,
    get_high_correlation_pairs,
)


# =============================================================================
# SECTION 1 — DATA CLEANING
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop exact duplicate rows (all column values identical).

    Returns the cleaned DataFrame and a report containing how many rows
    were removed.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df_clean)

    report = {
        "rows_before": n_before,
        "rows_after": len(df_clean),
        "duplicates_removed": n_removed,
    }
    return df_clean, report


def fix_data_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Try to cast object (string) columns to numeric where sensible.

    Strategy:
      - For each non-numeric column, attempt pd.to_numeric(errors='coerce').
      - Accept the cast only if ≥ 80 % of non-null values survive (see utils).
      - Also attempt to parse date-like strings to datetime.

    Returns the updated DataFrame and a list of columns that were changed.
    """
    df = df.copy()
    changed = []

    for col in df.select_dtypes(include="object").columns:
        converted = try_cast_to_numeric(df[col])
        if converted.dtype != df[col].dtype:
            df[col] = converted
            changed.append(f"{col}: object → numeric")
            continue

        # Try datetime parsing (only on columns with < 100 unique values)
        if df[col].nunique() < 100:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                # Accept if < 20 % became NaT
                nat_rate = parsed.isna().mean()
                original_na_rate = df[col].isna().mean()
                if nat_rate <= max(original_na_rate + 0.05, 0.20):
                    df[col] = parsed
                    changed.append(f"{col}: object → datetime")
            except Exception:
                pass  # Not a date column — move on

    report = {"columns_retyped": changed}
    return df, report


def remove_useless_columns(
    df: pd.DataFrame,
    unique_threshold: float = 0.95,
    null_threshold: float = 0.60,
) -> tuple[pd.DataFrame, dict]:
    """
    Drop columns that are unlikely to help a model learn:

    1. Near-constant columns  — too few unique values to be informative
       (not removed; instead flagged — constants are actually handled elsewhere)
    2. High-missing columns   — > null_threshold fraction of values are NaN
    3. ID-like columns        — unique ratio > unique_threshold AND dtype is
       int or object (e.g. row IDs, UUIDs)

    Parameters
    ----------
    unique_threshold : float
        If a column's unique-value ratio exceeds this, treat it as an ID.
    null_threshold : float
        If more than this fraction of values are missing, drop the column.
    """
    df = df.copy()
    dropped = []

    for col in df.columns:
        n_rows = len(df)
        n_null = df[col].isna().sum()
        n_unique = df[col].nunique(dropna=True)

        null_rate = n_null / n_rows if n_rows else 0
        unique_rate = n_unique / n_rows if n_rows else 0

        # Drop high-missing columns
        if null_rate > null_threshold:
            df.drop(columns=[col], inplace=True)
            dropped.append(f"{col} (missing={null_rate:.0%})")
            continue

        # Drop obvious ID columns (very high cardinality, not float)
        if unique_rate > unique_threshold and df[col].dtype in [object, "int64", "int32"]:
            df.drop(columns=[col], inplace=True)
            dropped.append(f"{col} (ID-like, unique={unique_rate:.0%})")

    report = {"columns_dropped": dropped}
    return df, report


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> tuple[pd.DataFrame, dict]:
    """
    Fill missing values column by column:

    - Numeric columns   → mean (strategy='mean') or median (strategy='median')
                          'auto' picks median (more robust to outliers)
    - Categorical cols  → mode (most frequent value)
    - If > 60 % missing the column should already have been removed by
      remove_useless_columns, so we skip columns with > 60 % still missing.

    Parameters
    ----------
    strategy : 'auto' | 'mean' | 'median'
    """
    df = df.copy()
    filled = {}

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        if strategy == "mean":
            fill_val = df[col].mean()
            method = "mean"
        else:  # 'auto' or 'median' — median is safer for skewed distributions
            fill_val = df[col].median()
            method = "median"

        df[col] = df[col].fillna(fill_val)
        filled[col] = {"missing": n_missing, "fill_value": round(float(fill_val), 4), "method": method}

    for col in categorical_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        mode_vals = df[col].mode()
        if mode_vals.empty:
            continue

        fill_val = mode_vals.iloc[0]
        df[col] = df[col].fillna(fill_val)
        filled[col] = {"missing": n_missing, "fill_value": str(fill_val), "method": "mode"}

    report = {"filled_columns": filled}
    return df, report


def normalise_text_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Standardise string / object columns so that e.g. 'Yes', 'YES', ' yes '
    all become 'yes'. This prevents the encoder from treating them as
    separate categories.

    Steps applied to every object column:
      1. Strip leading / trailing whitespace
      2. Lowercase everything
      3. Collapse multiple internal spaces to one
    """
    df = df.copy()
    normalised = []

    for col in df.select_dtypes(include="object").columns:
        original = df[col].copy()
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        # Mark 'nan' strings (from NaN → str) back to actual NaN
        df[col] = df[col].replace("nan", np.nan)

        if not df[col].equals(original.astype(str).str.strip().str.lower()):
            normalised.append(col)

    report = {"normalised_columns": normalised}
    return df, report


# =============================================================================
# SECTION 2 — OUTLIER DETECTION & HANDLING
# =============================================================================

def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> dict:
    """
    IQR (Interquartile Range) method.

    An observation is an outlier if it lies below Q1 − multiplier*IQR
    or above Q3 + multiplier*IQR.

    Returns a dict mapping column name → number of outliers detected.
    """
    outlier_counts = {}
    for col in get_numeric_columns(df):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out > 0:
            outlier_counts[col] = n_out

    return outlier_counts


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """
    Z-score method.

    A value is an outlier if |z| > threshold (default 3 — more than
    3 standard deviations from the mean).

    Returns a dict mapping column name → number of outliers detected.
    """
    outlier_counts = {}
    for col in get_numeric_columns(df):
        col_data = df[col].dropna()
        if col_data.std() == 0:
            continue  # Constant column — no meaningful z-score
        z_scores = np.abs(stats.zscore(col_data))
        n_out = int((z_scores > threshold).sum())
        if n_out > 0:
            outlier_counts[col] = n_out

    return outlier_counts


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    action: str = "cap",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect and handle outliers in every numeric column.

    Parameters
    ----------
    method : 'iqr' | 'zscore'
        Detection algorithm to use.
    action : 'cap' | 'remove'
        'cap'    → Replace outlier values with the boundary value (Winsorisation).
                   This preserves row count and is usually safer.
        'remove' → Drop rows that contain at least one outlier.
    iqr_multiplier : float
        Sensitivity for IQR method (lower = stricter).
    z_threshold : float
        Sensitivity for Z-score method (lower = stricter).

    Returns the cleaned DataFrame and a summary report.
    """
    df = df.copy()
    total_outliers = 0
    outlier_detail = {}
    rows_before = len(df)

    if action == "remove":
        # Collect a boolean mask for rows that should be kept
        keep_mask = pd.Series([True] * len(df), index=df.index)

    for col in get_numeric_columns(df):
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
        else:  # zscore
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(outlier_mask.sum())

        if n_out == 0:
            continue

        total_outliers += n_out
        outlier_detail[col] = n_out

        if action == "cap":
            # Winsorise: clip values to [lower, upper]
            df[col] = df[col].clip(lower=lower, upper=upper)
        else:
            # Mark outlier rows for removal
            keep_mask = keep_mask & ~outlier_mask

    if action == "remove" and total_outliers > 0:
        df = df[keep_mask].reset_index(drop=True)

    report = {
        "method": method,
        "action": action,
        "total_outliers_found": total_outliers,
        "column_detail": outlier_detail,
        "rows_removed": rows_before - len(df) if action == "remove" else 0,
    }
    return df, report


# =============================================================================
# SECTION 3 — ENCODING  (turning categories into numbers)
# =============================================================================

def encode_categorical(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "label",
    max_onehot_cardinality: int = 15,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical (non-numeric) columns so models can use them.

    Parameters
    ----------
    target_col : str | None
        The target/label column — skip encoding it here (it's handled
        separately in the pipeline).
    method : 'label' | 'onehot'
        'label'  → LabelEncoder  — turns each category to an integer.
                   Good for ordinal data or tree-based models.
        'onehot' → OneHotEncoder — creates one binary column per category.
                   Good for linear models / neural networks.
                   Skipped for high-cardinality columns (> max_onehot_cardinality).
    max_onehot_cardinality : int
        If a column has more unique values than this, fall back to label
        encoding even in 'onehot' mode (avoids dimension explosion).
    """
    df = df.copy()
    encoded = {}

    cat_cols = [
        c for c in get_categorical_columns(df)
        if c != target_col
    ]

    if method == "label":
        le = LabelEncoder()
        for col in cat_cols:
            # Fill any remaining NaN with placeholder before encoding
            df[col] = df[col].fillna("__missing__")
            df[col] = le.fit_transform(df[col].astype(str))
            encoded[col] = "label"

    elif method == "onehot":
        cols_to_onehot = []
        cols_to_label = []

        for col in cat_cols:
            if df[col].nunique() <= max_onehot_cardinality:
                cols_to_onehot.append(col)
            else:
                cols_to_label.append(col)

        # Label encode high-cardinality columns
        if cols_to_label:
            le = LabelEncoder()
            for col in cols_to_label:
                df[col] = df[col].fillna("__missing__")
                df[col] = le.fit_transform(df[col].astype(str))
                encoded[col] = "label (high cardinality fallback)"

        # One-hot encode low-cardinality columns
        if cols_to_onehot:
            df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)
            for col in cols_to_onehot:
                encoded[col] = "one-hot"

    report = {"method": method, "encoded_columns": encoded}
    return df, report


# =============================================================================
# SECTION 4 — SCALING  (normalising numeric feature magnitudes)
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, dict, object]:
    """
    Scale numeric feature columns so that no single feature dominates
    just because of its unit of measurement.

    Parameters
    ----------
    target_col : str | None
        Column to exclude from scaling (it is the prediction target).
    method : 'standard' | 'minmax'
        'standard' → StandardScaler  — zero mean, unit variance.
                     Best for linear models and neural networks.
        'minmax'   → MinMaxScaler    — scales to [0, 1].
                     Good when you need bounded values.

    Returns (scaled_df, report, fitted_scaler)
    The fitted scaler is returned so it can be reused on test data later.
    """
    df = df.copy()

    num_cols = [
        c for c in get_numeric_columns(df)
        if c != target_col
    ]

    if not num_cols:
        return df, {"scaled_columns": [], "method": method}, None

    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[num_cols] = scaler.fit_transform(df[num_cols])

    report = {"method": method, "scaled_columns": num_cols}
    return df, report, scaler


# =============================================================================
# SECTION 5 — OVERFITTING PREVENTION
# =============================================================================

def remove_correlated_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """
    Drop one of each pair of numeric features whose Pearson correlation
    exceeds `threshold`.

    Why this helps: Highly correlated features carry redundant information.
    Removing them reduces model complexity → less overfitting.

    Strategy: keep the column that appears first alphabetically (arbitrary
    but reproducible). Never drop the target column.
    """
    df = df.copy()
    pairs = get_high_correlation_pairs(df, threshold=threshold)

    # Decide which columns to drop (always drop the second in each pair)
    to_drop = set()
    for col_a, col_b, corr_val in pairs:
        if col_b != target_col:
            to_drop.add(col_b)
        elif col_a != target_col:
            to_drop.add(col_a)

    df.drop(columns=list(to_drop), inplace=True, errors="ignore")

    report = {
        "threshold": threshold,
        "correlated_pairs_found": len(pairs),
        "columns_dropped": list(to_drop),
        "pairs": pairs,
    }
    return df, report


def remove_low_variance_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    """
    Drop numeric features whose variance is below `threshold`.

    Near-constant features add noise without conveying signal, which
    can cause models to overfit to measurement artefacts.

    Note: The DataFrame should already be scaled before calling this,
    otherwise columns with very small absolute values may be dropped unfairly.
    """
    df = df.copy()

    feature_cols = [
        c for c in get_numeric_columns(df)
        if c != target_col
    ]

    if not feature_cols:
        return df, {"dropped": []}

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[feature_cols])

    # Columns that survive
    kept_mask = selector.get_support()
    kept = [c for c, k in zip(feature_cols, kept_mask) if k]
    dropped = [c for c, k in zip(feature_cols, kept_mask) if not k]

    # Keep only surviving feature columns + any non-numeric cols + target
    non_feature_cols = [c for c in df.columns if c not in feature_cols]
    df = df[kept + non_feature_cols]

    report = {"variance_threshold": threshold, "columns_dropped": dropped}
    return df, report


def apply_pca(
    df: pd.DataFrame,
    target_col: str | None = None,
    n_components: int | float = 0.95,
) -> tuple[pd.DataFrame, dict, PCA | None]:
    """
    Apply Principal Component Analysis (PCA) to numeric feature columns.

    PCA projects features into a lower-dimensional space that captures most
    of the variance, which:
      - Reduces dimensionality (faster training)
      - Removes noise (less overfitting)
      - Removes remaining multicollinearity

    Parameters
    ----------
    n_components : int or float
        int   → exact number of principal components to keep.
        float → keep enough components to explain this fraction of variance
                (e.g. 0.95 = 95 %).

    Returns (transformed_df, report, fitted_pca_object)
    """
    df = df.copy()

    feature_cols = [
        c for c in get_numeric_columns(df)
        if c != target_col
    ]

    if len(feature_cols) < 2:
        return df, {"applied": False, "reason": "Too few numeric features"}, None

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(df[feature_cols])

    n_comp = components.shape[1]
    comp_cols = [f"PC{i + 1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(components, columns=comp_cols, index=df.index)

    # Remove original feature columns and add PCA components
    non_feature_cols = [c for c in df.columns if c not in feature_cols]
    df = pd.concat([pca_df, df[non_feature_cols]], axis=1)

    explained = float(np.sum(pca.explained_variance_ratio_))
    report = {
        "applied": True,
        "original_features": len(feature_cols),
        "components_kept": n_comp,
        "variance_explained": round(explained * 100, 2),
    }
    return df, report, pca


# =============================================================================
# SECTION 6 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """
    Split the preprocessed DataFrame into three non-overlapping sets:

      ┌──────────────────────────────────────────────┐
      │   Full dataset  (100 %)                       │
      │  ┌──────────────────────┬────────┬─────────┐  │
      │  │   Training (70 %)   │ Val    │  Test   │  │
      │  │                      │(15 %)  │ (15 %)  │  │
      │  └──────────────────────┴────────┴─────────┘  │
      └──────────────────────────────────────────────┘

    Why three splits?
      - Training set  : the model learns from this data.
      - Validation set: used to tune hyperparameters and detect overfitting
                        during training (without touching test data).
      - Test set      : a completely unseen holdout for final evaluation.

    Returns
    -------
    splits : dict with keys X_train, X_val, X_test, y_train, y_val, y_test
    report : summary of set sizes
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: carve out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: from the remaining data, carve out validation
    # Adjust val_size relative to the remaining data size
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val, random_state=random_state
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    total = len(df)
    report = {
        "total_rows": total,
        "train_rows": len(X_train),
        "val_rows":   len(X_val),
        "test_rows":  len(X_test),
        "train_pct":  round(len(X_train) / total * 100, 1),
        "val_pct":    round(len(X_val)   / total * 100, 1),
        "test_pct":   round(len(X_test)  / total * 100, 1),
        "n_features": X_train.shape[1],
    }
    return splits, report


# =============================================================================
# SECTION 7 — FULL PIPELINE ORCHESTRATOR
# =============================================================================

def run_full_pipeline(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Run every preprocessing step in the correct order.

    Parameters
    ----------
    df     : Raw input DataFrame from the user's file.
    config : A dictionary of user-chosen settings from the Streamlit UI:
        {
          "missing_strategy"   : "auto" | "mean" | "median",
          "outlier_method"     : "iqr"  | "zscore",
          "outlier_action"     : "cap"  | "remove",
          "encoding_method"    : "label" | "onehot",
          "scaling_method"     : "standard" | "minmax",
          "target_col"         : str | None,
          "remove_correlated"  : bool,
          "corr_threshold"     : float,
          "remove_low_var"     : bool,
          "var_threshold"      : float,
          "apply_pca"          : bool,
          "pca_variance"       : float,
          "test_size"          : float,
          "val_size"           : float,
        }

    Returns
    -------
    (cleaned_df, full_report)
      cleaned_df  : The fully preprocessed DataFrame (features + target together).
      full_report : Nested dict with sub-reports for every step — shown in UI.
    """
    full_report = {}
    target = config.get("target_col")

    # ── Step 1: Remove duplicates ──────────────────────────────────────────
    df, rpt = remove_duplicates(df)
    full_report["duplicates"] = rpt

    # ── Step 2: Auto-fix data types ───────────────────────────────────────
    df, rpt = fix_data_types(df)
    full_report["type_fixes"] = rpt

    # ── Step 3: Drop useless columns ──────────────────────────────────────
    df, rpt = remove_useless_columns(df)
    full_report["useless_cols"] = rpt

    # ── Step 4: Normalise text values ─────────────────────────────────────
    df, rpt = normalise_text_values(df)
    full_report["text_normalisation"] = rpt

    # ── Step 5: Handle missing values ─────────────────────────────────────
    df, rpt = handle_missing_values(df, strategy=config.get("missing_strategy", "auto"))
    full_report["missing_values"] = rpt

    # ── Step 6: Outlier handling ──────────────────────────────────────────
    df, rpt = handle_outliers(
        df,
        method=config.get("outlier_method", "iqr"),
        action=config.get("outlier_action", "cap"),
    )
    full_report["outliers"] = rpt

    # ── Step 7: Encode categorical columns ────────────────────────────────
    df, rpt = encode_categorical(
        df,
        target_col=target,
        method=config.get("encoding_method", "label"),
    )
    full_report["encoding"] = rpt

    # ── Step 8: Remove highly correlated features ─────────────────────────
    if config.get("remove_correlated", True):
        df, rpt = remove_correlated_features(
            df,
            target_col=target,
            threshold=config.get("corr_threshold", 0.90),
        )
        full_report["correlation"] = rpt

    # ── Step 9: Scale features ────────────────────────────────────────────
    df, rpt, scaler = scale_features(
        df,
        target_col=target,
        method=config.get("scaling_method", "standard"),
    )
    full_report["scaling"] = rpt

    # ── Step 10: Remove low-variance features ─────────────────────────────
    if config.get("remove_low_var", False):
        df, rpt = remove_low_variance_features(
            df,
            target_col=target,
            threshold=config.get("var_threshold", 0.01),
        )
        full_report["low_variance"] = rpt

    # ── Step 11: Optional PCA ─────────────────────────────────────────────
    if config.get("apply_pca", False):
        df, rpt, pca_obj = apply_pca(
            df,
            target_col=target,
            n_components=config.get("pca_variance", 0.95),
        )
        full_report["pca"] = rpt

    full_report["final_shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    return df, full_report
