#!/usr/bin/env python3
"""load_random_effects: Load MODEL-AVERAGED random effects (intercept_avg, slope_avg) from RQ 5.1.4 Step 06."""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_load_random_effects.log"

# Cross-RQ dependency path (RQ 5.1.4 Step 6 output - MODEL-AVERAGED)
SOURCE_FILE = PROJECT_ROOT / "results/ch5/5.1.4/data/step06_averaged_random_effects.csv"

# Local output path
OUTPUT_FILE = RQ_DIR / "data/step00_random_effects_from_rq514.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Random Effects from RQ 5.1.4")
        # Check Cross-RQ Dependency Exists

        log("Validating cross-RQ dependency...")

        if not SOURCE_FILE.exists():
            error_msg = (
                "EXPECTATIONS ERROR: RQ 5.1.4 Step 06 must complete before RQ 5.1.5\n"
                f"\n"
                f"Missing file: {SOURCE_FILE}\n"
                f"\n"
                f"Action: Run RQ 5.1.4 through Step 06 (model-averaged variance decomposition) before running RQ 5.1.5\n"
                f"\n"
                f"Dependency chain: RQ 5.1.1 (Steps 1-6) -> RQ 5.1.4 (Steps 1-6) -> RQ 5.1.5"
            )
            log(f"{error_msg}")
            sys.exit(1)

        log(f"Dependency file exists: {SOURCE_FILE}")
        # Load Source CSV from RQ 5.1.4
        # Columns: UID, intercept_avg, slope_avg (MODEL-AVERAGED across 10 competitive models)
        # We need: UID, intercept_avg, slope_avg

        log("Loading MODEL-AVERAGED random effects from RQ 5.1.4 Step 06...")

        try:
            df_source = pd.read_csv(SOURCE_FILE, encoding='utf-8')
            log(f"{SOURCE_FILE.name} ({len(df_source)} rows, {len(df_source.columns)} cols)")
        except Exception as e:
            log(f"Failed to load CSV: {e}")
            raise

        # Check expected columns exist in source file
        expected_source_cols = ['UID', 'intercept_avg', 'slope_avg']
        missing_cols = [col for col in expected_source_cols if col not in df_source.columns]

        if missing_cols:
            error_msg = (
                f"Source CSV missing expected columns\n"
                f"Expected columns: {expected_source_cols}\n"
                f"Actual columns: {list(df_source.columns)}\n"
                f"Missing: {missing_cols}"
            )
            log(error_msg)
            sys.exit(1)

        log(f"Source CSV has expected columns: {expected_source_cols}")
        # Extract Required Columns and Standardize Names
        # Extract: UID, intercept_avg, slope_avg
        # Rename to: UID, Total_Intercept, Total_Slope (standardized capitalization)

        log("Extracting and standardizing column names...")

        df_output = df_source[['UID', 'intercept_avg', 'slope_avg']].copy()

        # Standardize column names (capitalize for consistency across RQs)
        df_output.rename(columns={
            'intercept_avg': 'Total_Intercept',
            'slope_avg': 'Total_Slope'
        }, inplace=True)

        log(f"Standardized columns: {list(df_output.columns)}")
        # Validate Structure Before Saving

        log("Validating DataFrame structure...")

        # Check for NaN values in clustering variables
        nan_intercept = df_output['Total_Intercept'].isna().sum()
        nan_slope = df_output['Total_Slope'].isna().sum()

        if nan_intercept > 0 or nan_slope > 0:
            error_msg = (
                f"NaN values detected in clustering variables\n"
                f"Total_Intercept NaN count: {nan_intercept}\n"
                f"Total_Slope NaN count: {nan_slope}\n"
                f"No NaN values tolerated for clustering analysis"
            )
            log(error_msg)
            sys.exit(1)

        log("No NaN values in clustering variables")

        # Validate using tools.validation.validate_dataframe_structure
        validation_result = validate_dataframe_structure(
            df=df_output,
            expected_rows=100,
            expected_columns=['UID', 'Total_Intercept', 'Total_Slope'],
            column_types=None  # Skip type checking (pd.read_csv infers correctly)
        )

        if not validation_result['valid']:
            log(f"{validation_result['message']}")
            raise ValueError(validation_result['message'])

        log(f"{validation_result['message']}")
        # Save Local Copy for Lineage Tracking
        # Lineage: RQ 5.1.4 Step 4 -> RQ 5.1.5 Step 0 -> RQ 5.1.5 Step 1

        log(f"Saving local copy to {OUTPUT_FILE.name}...")

        df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        log(f"{OUTPUT_FILE.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Final Summary

        log("Step 00 complete:")
        log(f"  - Source: {SOURCE_FILE.relative_to(PROJECT_ROOT)}")
        log(f"  - Output: {OUTPUT_FILE.relative_to(RQ_DIR)}")
        log(f"  - Rows: {len(df_output)}")
        log(f"  - Columns: {list(df_output.columns)}")
        log(f"  - Total_Intercept range: [{df_output['Total_Intercept'].min():.4f}, {df_output['Total_Intercept'].max():.4f}]")
        log(f"  - Total_Slope range: [{df_output['Total_Slope'].min():.4f}, {df_output['Total_Slope'].max():.4f}]")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
