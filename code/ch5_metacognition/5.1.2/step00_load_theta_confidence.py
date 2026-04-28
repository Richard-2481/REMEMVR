#!/usr/bin/env python3
"""Load Theta Confidence Scores: Load IRT-derived confidence ability estimates from RQ 6.1.1 and merge with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.2
LOG_FILE = RQ_DIR / "logs" / "step00_load_theta_confidence.log"

# Input paths from RQ 6.1.1
THETA_INPUT = PROJECT_ROOT / "results/ch6/6.1.1/data/step03_theta_confidence.csv"
TSVR_INPUT = PROJECT_ROOT / "results/ch6/6.1.1/data/step00_tsvr_mapping.csv"

# Output path
OUTPUT_FILE = RQ_DIR / "data" / "step00_lmm_input.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Theta Confidence Scores")
        # Load Input Data

        log("Loading theta confidence scores from RQ 6.1.1...")
        df_theta = pd.read_csv(THETA_INPUT, encoding='utf-8')
        log(f"{THETA_INPUT.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"Theta columns: {df_theta.columns.tolist()}")

        log("Loading TSVR time mapping from RQ 6.1.1...")
        df_tsvr = pd.read_csv(TSVR_INPUT, encoding='utf-8')
        log(f"{TSVR_INPUT.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"TSVR columns: {df_tsvr.columns.tolist()}")

        # Fix composite_ID format mismatch: TSVR has "A010_1" but theta has "A010_T1"
        log("Converting TSVR composite_ID format (A010_1 -> A010_T1)...")
        df_tsvr['composite_ID'] = df_tsvr['composite_ID'].str.replace('_', '_T', regex=False)
        log(f"Converted {len(df_tsvr)} composite_IDs")
        # Merge DataFrames

        log("Merging theta confidence with TSVR mapping on composite_ID...")
        lmm_input = pd.merge(
            df_theta,
            df_tsvr,
            on='composite_ID',
            how='left',
            validate='one_to_one'
        )
        log(f"Merge complete ({len(lmm_input)} rows)")

        # Parse UID from composite_ID (format: "A010_T1" -> "A010")
        log("Parsing UID from composite_ID...")
        lmm_input['UID'] = lmm_input['composite_ID'].str.split('_').str[0]
        log(f"Parsed UID for {len(lmm_input)} rows")

        # Rename theta columns to match expected names
        log("Renaming theta columns...")
        if 'theta_All' in lmm_input.columns:
            lmm_input = lmm_input.rename(columns={'theta_All': 'theta_confidence', 'se_All': 'se_confidence'})
            log("Renamed theta_All -> theta_confidence, se_All -> se_confidence")

        # Reorder columns to match expected output
        column_order = ['composite_ID', 'UID', 'test', 'theta_confidence', 'se_confidence', 'TSVR_hours']
        lmm_input = lmm_input[column_order]
        log(f"Final columns: {lmm_input.columns.tolist()}")

        # Check for missing TSVR matches
        missing_tsvr = lmm_input['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            log(f"{missing_tsvr} theta scores missing TSVR match")
            missing_ids = lmm_input[lmm_input['TSVR_hours'].isna()]['composite_ID'].tolist()
            log(f"Missing composite_IDs: {missing_ids[:10]}...")  # Show first 10
        else:
            log("All theta scores have TSVR match (no missing values)")

        # Check for duplicates
        n_duplicates = lmm_input['composite_ID'].duplicated().sum()
        if n_duplicates > 0:
            log(f"{n_duplicates} duplicate composite_IDs found")
        else:
            log("No duplicate composite_IDs")
        # Save Merged Output
        # Output: data/step00_lmm_input.csv
        # Contains: Merged theta confidence + TSVR data
        # Columns: composite_ID, UID, test, theta_confidence, se_confidence, TSVR_hours

        log(f"Saving merged data to {OUTPUT_FILE.name}...")
        lmm_input.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Summary statistics
        log("Data summary:")
        log(f"  Unique UIDs: {lmm_input['UID'].nunique()}")
        log(f"  Unique tests: {lmm_input['test'].unique().tolist()}")
        log(f"  Theta range: [{lmm_input['theta_confidence'].min():.3f}, {lmm_input['theta_confidence'].max():.3f}]")
        log(f"  TSVR range: [{lmm_input['TSVR_hours'].min():.1f}, {lmm_input['TSVR_hours'].max():.1f}] hours")
        # Run Validation
        # Validates: Row count, column presence, data types, value ranges
        # Threshold: Row count in [390, 410] for expected ~400 rows

        log("Skipping validate_dataframe_structure (tool bug)...")
        log("Running manual validation checks instead...")

        # Manual validation checks
        if len(lmm_input) < 390 or len(lmm_input) > 410:
            raise ValueError(f"Row count out of range: {len(lmm_input)} (expected 390-410)")
        log(f"PASS - Row count: {len(lmm_input)} in range [390, 410]")

        expected_cols = ['composite_ID', 'UID', 'test', 'theta_confidence', 'se_confidence', 'TSVR_hours']
        if list(lmm_input.columns) != expected_cols:
            raise ValueError(f"Column mismatch: {list(lmm_input.columns)} != {expected_cols}")
        log(f"PASS - All expected columns present")

        if lmm_input.isna().any().any():
            raise ValueError(f"NaN values detected in merged data")
        log(f"PASS - No NaN values")

        # Additional custom validations
        log("Running custom range checks...")

        # Check theta_confidence range
        theta_out_of_range = ((lmm_input['theta_confidence'] < -3) | (lmm_input['theta_confidence'] > 3)).sum()
        if theta_out_of_range > 0:
            log(f"WARNING - {theta_out_of_range} theta values outside [-3, 3] range")
        else:
            log("PASS - All theta values in [-3, 3] range")

        # Check TSVR_hours range
        tsvr_out_of_range = ((lmm_input['TSVR_hours'] < 0) | (lmm_input['TSVR_hours'] > 200)).sum()
        if tsvr_out_of_range > 0:
            log(f"WARNING - {tsvr_out_of_range} TSVR values outside [0, 200] range")
        else:
            log("PASS - All TSVR values in [0, 200] range")

        # Check all UIDs have 4 observations
        uid_counts = lmm_input['UID'].value_counts()
        incomplete_uids = uid_counts[uid_counts != 4]
        if len(incomplete_uids) > 0:
            log(f"WARNING - {len(incomplete_uids)} UIDs do not have exactly 4 observations")
            log(f"Incomplete UIDs: {incomplete_uids.index.tolist()[:10]}...")
        else:
            log("PASS - All UIDs have exactly 4 observations")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
