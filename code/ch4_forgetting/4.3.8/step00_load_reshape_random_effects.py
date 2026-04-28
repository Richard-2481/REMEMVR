#!/usr/bin/env python3
"""Load and Reshape Random Effects: Load paradigm-specific random effects from RQ 5.3.7 and reshape from long to"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.8
LOG_FILE = RQ_DIR / "logs" / "step00_load_reshape_random_effects.log"

# Input from RQ 5.3.7
INPUT_FILE = PROJECT_ROOT / "results" / "ch5" / "5.3.7" / "data" / "step04_random_effects.csv"

# Output
OUTPUT_FILE = RQ_DIR / "data" / "step00_random_effects_wide.csv"

# Paradigm mapping (codes in data → column names)
PARADIGM_MAP = {
    'IFR': 'Free',
    'ICR': 'Cued',
    'IRE': 'Recognition'
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load and Reshape Random Effects")
        # Load Long-Format Random Effects from RQ 5.3.7

        log(f"Loading random effects from RQ 5.3.7...")
        log(f"File: {INPUT_FILE}")

        df_long = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{len(df_long)} rows, {len(df_long.columns)} columns")

        # Verify expected structure
        expected_cols = ['UID', 'paradigm', 'Total_Intercept', 'Total_Slope']
        if list(df_long.columns) != expected_cols:
            raise ValueError(f"Unexpected columns. Expected {expected_cols}, got {list(df_long.columns)}")

        # Verify row count
        if len(df_long) != 300:
            raise ValueError(f"Expected 300 rows (100 UIDs × 3 paradigms), got {len(df_long)}")

        # Check paradigm values
        paradigms = sorted(df_long['paradigm'].unique())
        log(f"Paradigms in data: {paradigms}")
        if paradigms != ['ICR', 'IFR', 'IRE']:
            raise ValueError(f"Expected paradigms [ICR, IFR, IRE], got {paradigms}")

        # Check for missing values
        if df_long.isnull().any().any():
            raise ValueError("Found missing values in input data")

        log(f"Verified: 300 rows, 3 paradigms, no missing values")
        # Pivot to Wide Format

        log("Pivoting from long to wide format...")

        # Map paradigm codes to column names
        df_long['paradigm_name'] = df_long['paradigm'].map(PARADIGM_MAP)

        # Pivot: create separate columns for each paradigm × feature combination
        df_wide = df_long.pivot(
            index='UID',
            columns='paradigm_name',
            values=['Total_Intercept', 'Total_Slope']
        )

        # Flatten multi-level columns: (feature, paradigm) → feature_paradigm
        df_wide.columns = [f'{feature}_{paradigm}' for feature, paradigm in df_wide.columns]

        # Reset index to make UID a column
        df_wide = df_wide.reset_index()

        log(f"Pivot complete: {len(df_wide)} rows, {len(df_wide.columns)} columns")
        log(f"Columns: {list(df_wide.columns)}")

        # Verify column names
        expected_wide_cols = [
            'UID',
            'Total_Intercept_Cued', 'Total_Intercept_Free', 'Total_Intercept_Recognition',
            'Total_Slope_Cued', 'Total_Slope_Free', 'Total_Slope_Recognition'
        ]
        # Sort both lists for comparison (pivot column order may vary)
        if sorted(df_wide.columns) != sorted(expected_wide_cols):
            raise ValueError(f"Column name mismatch after pivot. Expected {expected_wide_cols}, got {list(df_wide.columns)}")
        # Save Wide-Format Output
        # Output: 100 rows × 7 columns (UID + 6 features)
        # Downstream: step01 will standardize these features

        log(f"Saving wide-format random effects...")
        df_wide.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_wide)} rows, {len(df_wide.columns)} cols)")
        # Run Validation Tool
        # Validates: Row count = 100, all 7 columns present, no missing values

        log("Running validate_dataframe_structure...")

        validation_result = validate_dataframe_structure(
            df=df_wide,
            expected_rows=100,
            expected_columns=expected_wide_cols,
            column_types=None  # No type checking needed
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

        # Check validation passed
        if not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
