#!/usr/bin/env python3
"""
Step 00: Extract and Reshape Random Effects from RQ 5.4.6 (MODEL-AVERAGED UPDATE)

Load MODEL-AVERAGED random effects from RQ 5.4.6 (long format: 300 rows = 100 UID × 3 congruence levels)
and reshape to wide format (100 rows × 6 features) for clustering.

CRITICAL UPDATE (2025-12-09): Now uses model-averaged random effects (step02_averaged_random_effects.csv)
instead of Log-only random effects (step04_random_effects.csv). Model averaging reveals meaningful
slope variance (0.016-0.083) that was invisible in Log-only analysis (slopes ≈ 0).

Input: results/ch5/5.4.6/data/step02_averaged_random_effects.csv (300 rows)
Output: data/step00_random_effects_from_rq546.csv (100 rows × 7 columns)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
# UPDATED: Use model-averaged random effects (step02) instead of Log-only (step04)
INPUT_FILE = PROJECT_ROOT / "results/ch5/5.4.6/data/step02_averaged_random_effects.csv"
OUTPUT_FILE = RQ_DIR / "data/step00_random_effects_from_rq546.csv"
LOG_FILE = RQ_DIR / "logs/step00_extract_random_effects.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 00: Extract and Reshape Random Effects")

        # Load long-format random effects
        log(f"[LOAD] Reading {INPUT_FILE}")
        df_long = pd.read_csv(INPUT_FILE)
        log(f"[LOADED] {len(df_long)} rows, {len(df_long.columns)} columns")
        log(f"[INFO] Columns: {list(df_long.columns)}")

        # Verify expected structure (300 rows = 100 UID × 3 congruence)
        if len(df_long) != 300:
            raise ValueError(f"Expected 300 rows, got {len(df_long)}")

        unique_uids = df_long['UID'].nunique()
        unique_congruence = df_long['congruence'].nunique()
        log(f"[INFO] Unique UIDs: {unique_uids}, Unique congruence levels: {unique_congruence}")

        if unique_uids != 100:
            raise ValueError(f"Expected 100 unique UIDs, got {unique_uids}")
        if unique_congruence != 3:
            raise ValueError(f"Expected 3 congruence levels, got {unique_congruence}")

        # Pivot from long to wide format
        # UPDATED: Use 'intercept_avg' and 'slope_avg' (model-averaged columns)
        log("[RESHAPE] Pivoting from long to wide format")
        df_wide = df_long.pivot(index='UID', columns='congruence', values=['intercept_avg', 'slope_avg'])

        # Flatten multi-level column names
        # Pivot produces: (metric, congruence) tuples -> flatten to "congruence_metric"
        # UPDATED: intercept_avg -> Intercept, slope_avg -> Slope
        df_wide.columns = [f"{col[1].capitalize()}_{col[0].replace('_avg', '').capitalize()}" for col in df_wide.columns]
        df_wide = df_wide.reset_index()

        # Reorder columns to match expected format
        expected_cols = [
            'UID',
            'Common_Intercept', 'Common_Slope',
            'Congruent_Intercept', 'Congruent_Slope',
            'Incongruent_Intercept', 'Incongruent_Slope'
        ]

        # Check if all expected columns exist
        actual_cols = list(df_wide.columns)
        log(f"[INFO] Actual columns after pivot: {actual_cols}")

        # Verify expected columns are present
        missing_cols = [col for col in expected_cols if col not in actual_cols]
        if missing_cols:
            raise ValueError(f"Missing expected columns after pivot: {missing_cols}")

        # Reorder to match expected format
        df_wide = df_wide[expected_cols]

        log(f"[RESHAPED] {len(df_wide)} rows, {len(df_wide.columns)} columns")
        log(f"[INFO] Final columns: {list(df_wide.columns)}")

        # Verify no NaN values
        nan_count = df_wide.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values after reshaping")

        # Verify exactly 100 participants
        if len(df_wide) != 100:
            raise ValueError(f"Expected 100 participants, got {len(df_wide)}")

        # Save to CSV
        log(f"[SAVE] Writing to {OUTPUT_FILE}")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_wide.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_FILE}")

        # Validate output structure (column_types expects tuples)
        log("[VALIDATION] Validating output structure")
        validation_result = validate_dataframe_structure(
            df=df_wide,
            expected_rows=100,
            expected_columns=expected_cols,
            column_types={
                'UID': (object,),
                'Common_Intercept': (np.float64, float),
                'Common_Slope': (np.float64, float),
                'Congruent_Intercept': (np.float64, float),
                'Congruent_Slope': (np.float64, float),
                'Incongruent_Intercept': (np.float64, float),
                'Incongruent_Slope': (np.float64, float)
            }
        )

        if not validation_result['valid']:
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log(f"[VALIDATION PASS] {validation_result['message']}")
        log("[SUCCESS] Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK]")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
