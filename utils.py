# =============================================================================
# utils.py — Helper / utility functions
# =============================================================================
# This module contains small, reusable helper functions used across the app.
# Keeping them here prevents repetition and makes the codebase easier to read.
# =============================================================================

import pandas as pd
import numpy as np
import io
import json


# ---------------------------------------------------------------------------
# 1. FILE LOADING
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB = 20  # Hard limit defined once here


def load_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Load a user-uploaded file into a Pandas DataFrame.

    Supports:
      - CSV  (.csv)
      - Excel (.xlsx / .xls)
      - JSON  (.json)

    Returns
    -------
    (DataFrame, error_message)
      On success : (df, "")
      On failure : (None, "<reason>")
    """

    # ── 1a. Check file size ────────────────────────────────────────────────
    file_bytes = uploaded_file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"File is {size_mb:.1f} MB — exceeds the {MAX_FILE_SIZE_MB} MB limit."

    # Put the bytes back so Pandas can read them
    uploaded_file.seek(0)
    buffer = io.BytesIO(file_bytes)

    # ── 1b. Parse based on extension ──────────────────────────────────────
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(buffer)

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer, engine="openpyxl")

        elif name.endswith(".json"):
            data = json.loads(file_bytes.decode("utf-8"))
            # JSON can be a list-of-dicts or a dict-of-lists — handle both
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                return None, "JSON structure not recognised. Expected a list or object."

        else:
            return None, "Unsupported file type. Please upload CSV, Excel, or JSON."

    except Exception as exc:
        return None, f"Could not parse file: {exc}"

    if df.empty:
        return None, "The uploaded file appears to be empty."

    return df, ""


# ---------------------------------------------------------------------------
# 2. QUICK DATASET SUMMARY
# ---------------------------------------------------------------------------

def summarise_dataset(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with high-level facts about a DataFrame.
    Displayed in the 'Dataset Preview' section of the UI.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_cols": int(df.select_dtypes(exclude=np.number).shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


# ---------------------------------------------------------------------------
# 3. COLUMN TYPE HELPERS
# ---------------------------------------------------------------------------

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return names of columns with numeric (int / float) dtype."""
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return names of non-numeric columns (object, category, bool, datetime)."""
    return df.select_dtypes(exclude=np.number).columns.tolist()


# ---------------------------------------------------------------------------
# 4. DATAFRAME → CSV BYTES  (for download button)
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes, ready for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# 5. SAFE NUMERIC COERCION
# ---------------------------------------------------------------------------

def try_cast_to_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to coerce a non-numeric column to numeric.
    Values that cannot be converted become NaN (errors='coerce').
    Returns the original series unchanged if conversion fails entirely.
    """
    converted = pd.to_numeric(series, errors="coerce")
    # Accept the conversion only if at least 80 % of non-null values survive
    original_non_null = series.dropna().shape[0]
    converted_non_null = converted.dropna().shape[0]

    if original_non_null == 0:
        return series

    survival_rate = converted_non_null / original_non_null
    if survival_rate >= 0.80:
        return converted
    return series


# ---------------------------------------------------------------------------
# 6. CORRELATION MATRIX HELPER
# ---------------------------------------------------------------------------

def get_high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.90) -> list[tuple]:
    """
    Find pairs of numeric columns whose absolute Pearson correlation
    exceeds `threshold`. Returns a list of (col_a, col_b, correlation) tuples.
    Used for the 'remove correlated features' step.
    """
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return []

    corr = num_df.corr().abs()
    # Upper triangle only (avoid duplicates like A-B and B-A)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and val >= threshold:
                pairs.append((row, col, round(float(val), 4)))

    return pairs


# ---------------------------------------------------------------------------
# 7. PRETTY-PRINT LARGE NUMBERS
# ---------------------------------------------------------------------------

def fmt_number(n: int | float) -> str:
    """Format a number with thousands separators for display."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{int(n):,}"
