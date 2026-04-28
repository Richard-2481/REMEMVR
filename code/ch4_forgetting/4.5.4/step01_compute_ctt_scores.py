#!/usr/bin/env python3
"""Compute CTT Mean Scores per Location Type: Compute Classical Test Theory (CTT) mean scores (proportion correct) for each"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_ctt_mean_scores_by_factor

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_compute_ctt_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Compute CTT Mean Scores per Location Type")
        # Load Input Data

        log("Loading input data...")

        # Load raw binary responses (wide format)
        raw_responses = pd.read_csv(RQ_DIR / "data/step00_raw_responses_filtered.csv")
        log(f"step00_raw_responses_filtered.csv ({len(raw_responses)} rows, {len(raw_responses.columns)} cols)")

        # Rename 'test' to 'TEST' for tool compatibility (tool expects uppercase)
        if 'test' in raw_responses.columns and 'TEST' not in raw_responses.columns:
            raw_responses = raw_responses.rename(columns={'test': 'TEST'})
            log(f"Renamed 'test' -> 'TEST' for tool compatibility")

        # Load purified items mapping (item_code -> location_type)
        purified_items = pd.read_csv(RQ_DIR / "data/step00_purified_items_from_rq551.csv")
        log(f"step00_purified_items_from_rq551.csv ({len(purified_items)} rows, {len(purified_items.columns)} cols)")
        log(f"Purified items: {len(purified_items)} items across {purified_items['location_type'].nunique()} location types")

        # Load IRT theta scores (for TSVR_hours merge)
        theta_long = pd.read_csv(RQ_DIR / "data/step00_irt_theta_from_rq551.csv")
        log(f"step00_irt_theta_from_rq551.csv ({len(theta_long)} rows, {len(theta_long.columns)} cols)")
        # Run Analysis Tool
        #               composite_ID × location_type combination

        log("Running compute_ctt_mean_scores_by_factor...")
        log("Computing CTT mean scores per location type...")
        log(f"Factor column: 'location_type'")
        log(f"Item column: 'item_code'")

        ctt_scores = compute_ctt_mean_scores_by_factor(
            df_wide=raw_responses,           # Wide-format binary responses (400 rows)
            item_factor_df=purified_items,   # Item-to-location mapping (32 rows)
            factor_col='location_type',      # Factor column name in purified_items
            item_col='item_code',            # Item column name in purified_items
            include_factors=None             # Include all location types (source, destination)
        )
        log("CTT mean score computation complete")
        log(f"CTT scores computed: {len(ctt_scores)} rows (expected: 800 = 400 participant-tests × 2 location types)")
        log(f"CTT output columns: {list(ctt_scores.columns)}")

        # Rename 'factor' column to 'location_type' if needed (tool outputs 'factor')
        if 'factor' in ctt_scores.columns and 'location_type' not in ctt_scores.columns:
            ctt_scores = ctt_scores.rename(columns={'factor': 'location_type'})
            log("Renamed 'factor' -> 'location_type'")

        # Rename 'CTT_score' to 'ctt_mean_score' for consistency with spec
        if 'CTT_score' in ctt_scores.columns and 'ctt_mean_score' not in ctt_scores.columns:
            ctt_scores = ctt_scores.rename(columns={'CTT_score': 'ctt_mean_score'})
            log("Renamed 'CTT_score' -> 'ctt_mean_score'")
        # STEP 2.5: Merge TSVR_hours
        # Merge strategy: Match on composite_ID + location_type

        log("Merging TSVR_hours from IRT theta scores...")

        # Select TSVR columns from theta_long
        tsvr_data = theta_long[['composite_ID', 'location_type', 'TSVR_hours']].copy()

        # Merge TSVR_hours into ctt_scores
        ctt_scores_with_tsvr = ctt_scores.merge(
            tsvr_data,
            on=['composite_ID', 'location_type'],
            how='left'
        )

        # Check for missing TSVR values
        n_missing_tsvr = ctt_scores_with_tsvr['TSVR_hours'].isna().sum()
        if n_missing_tsvr > 0:
            log(f"{n_missing_tsvr} rows missing TSVR_hours after merge")
        else:
            log(f"All {len(ctt_scores_with_tsvr)} rows have TSVR_hours")

        # Update ctt_scores variable
        ctt_scores = ctt_scores_with_tsvr
        log(f"TSVR merge complete")
        # Save Analysis Outputs
        # Output: data/step01_ctt_scores.csv
        # Contains: CTT mean scores per participant × test × location type (800 rows)
        # Columns: composite_ID, UID, test, location_type, ctt_mean_score, n_items, TSVR_hours
        # Downstream usage: Step 2 (correlations), Step 3 (LMM fitting), Step 7 (scatterplot)

        log(f"Saving data/step01_ctt_scores.csv...")

        ctt_scores.to_csv(RQ_DIR / "data/step01_ctt_scores.csv", index=False, encoding='utf-8')
        log(f"data/step01_ctt_scores.csv ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        log(f"Columns: {list(ctt_scores.columns)}")
        # Run Validation Tool
        # Validates: CTT mean scores in [0, 1] range (proportion correct)
        # Threshold: min=0.0, max=1.0 (inclusive)

        log("Running validate_numeric_range...")
        log("Checking ctt_mean_score in [0, 1] range...")

        validation_result = validate_numeric_range(
            data=ctt_scores['ctt_mean_score'],  # CTT mean scores to validate
            min_val=0.0,                        # Minimum allowed value (proportion 0%)
            max_val=1.0,                        # Maximum allowed value (proportion 100%)
            column_name='ctt_mean_score'        # Column name for error messages
        )

        # Report validation results
        if validation_result['valid']:
            log(f"{validation_result['message']}")
        else:
            log(f"{validation_result['message']}")
            if validation_result.get('violations'):
                log(f"Violations (first 10): {validation_result['violations'][:10]}")
            raise ValueError(f"CTT score validation failed: {validation_result['message']}")

        # Additional validation checks (inline)
        log("Running additional checks...")

        # Check expected row count (800 = 100 participants × 4 tests × 2 location types)
        expected_rows = 800
        if len(ctt_scores) == expected_rows:
            log(f"Expected N: {expected_rows} rows")
        else:
            log(f"Row count mismatch: expected {expected_rows}, got {len(ctt_scores)}")
            raise ValueError(f"Expected {expected_rows} rows, got {len(ctt_scores)}")

        # Check location type balance (400 rows per location type)
        location_counts = ctt_scores['location_type'].value_counts()
        log(f"Location type distribution:")
        for loc_type, count in location_counts.items():
            log(f"{loc_type}: {count} rows")
            if count != 400:
                log(f"Expected 400 rows for '{loc_type}', got {count}")
                raise ValueError(f"Location type balance violation: {loc_type} has {count} rows (expected 400)")
        log(f"Location balance: 400 rows per location type")

        # Check n_items > 0 for all rows
        n_zero_items = (ctt_scores['n_items'] == 0).sum()
        if n_zero_items > 0:
            log(f"{n_zero_items} rows have n_items = 0")
            raise ValueError(f"{n_zero_items} rows have zero items")
        else:
            log(f"All rows have n_items > 0")

        # Check for NaN values in ctt_mean_score
        n_nan = ctt_scores['ctt_mean_score'].isna().sum()
        if n_nan > 0:
            log(f"{n_nan} NaN values in ctt_mean_score")
            raise ValueError(f"{n_nan} NaN values in ctt_mean_score")
        else:
            log(f"No NaN values in ctt_mean_score")

        # Check TSVR_hours range [0, 360] hours (extended range for participants tested beyond 1 week)
        tsvr_min = ctt_scores['TSVR_hours'].min()
        tsvr_max = ctt_scores['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 360:
            log(f"TSVR_hours out of range: [{tsvr_min}, {tsvr_max}] (expected [0, 360])")
            raise ValueError(f"TSVR_hours out of range: [{tsvr_min}, {tsvr_max}]")
        else:
            log(f"TSVR_hours in valid range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
