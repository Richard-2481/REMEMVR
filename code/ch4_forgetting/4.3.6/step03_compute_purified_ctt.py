#!/usr/bin/env python3
"""Compute Purified CTT Scores (Retained Items Only): Extract raw response data for RETAINED items only and compute Purified CTT scores"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.6
LOG_FILE = RQ_DIR / "logs" / "step03_compute_purified_ctt.log"

# Paradigms to process
PARADIGMS = ["IFR", "ICR", "IRE"]


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Compute Purified CTT Scores (Retained Items Only)")
        # Load Input Data

        log("Loading input data...")

        # Load item mapping (output from Step 1)
        item_mapping_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        item_mapping = pd.read_csv(item_mapping_path, encoding='utf-8')
        log(f"step01_item_mapping.csv ({len(item_mapping)} rows, {len(item_mapping.columns)} cols)")

        # Filter to retained items ONLY
        retained_items = item_mapping[item_mapping['retained'] == True].copy()
        log(f"Retained items: {len(retained_items)} total")

        # Report retained counts per paradigm
        for paradigm in PARADIGMS:
            n_retained = len(retained_items[retained_items['paradigm'] == paradigm])
            log(f"         - {paradigm}: {n_retained} retained items")

        # Load raw response data (project-level cache)
        raw_data_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        raw_data = pd.read_csv(raw_data_path, encoding='utf-8')
        log(f"dfData.csv ({len(raw_data)} rows, {len(raw_data.columns)} cols)")

        # Note: dfData uses 'TEST' (all caps), we need 'test' (lowercase) for consistency
        if 'TEST' in raw_data.columns:
            raw_data['test'] = raw_data['TEST']
            log("Renamed TEST -> test for consistency")
        # Compute Purified CTT Scores per Paradigm
        #               compute mean proportion correct per UID × test

        log("Computing Purified CTT scores by paradigm...")

        # Initialize output DataFrame with UID and test
        ctt_purified = raw_data[['UID', 'test']].drop_duplicates().sort_values(['UID', 'test']).reset_index(drop=True)
        log(f"Output DataFrame initialized: {len(ctt_purified)} rows (UID × test combinations)")

        # Process each paradigm
        for paradigm in PARADIGMS:
            log(f"Processing {paradigm}...")

            # Get retained items for this paradigm
            paradigm_items = retained_items[retained_items['paradigm'] == paradigm]['item_name'].tolist()
            log(f"           Retained items: {len(paradigm_items)}")

            # Verify all item columns exist in raw_data
            missing_cols = [item for item in paradigm_items if item not in raw_data.columns]
            if len(missing_cols) > 0:
                log(f"Missing {len(missing_cols)} item columns in dfData: {missing_cols[:5]}...")
                # Filter to available columns only
                paradigm_items = [item for item in paradigm_items if item in raw_data.columns]
                log(f"           Using {len(paradigm_items)} available items")

            # Extract responses for retained items
            item_responses = raw_data[['UID', 'test'] + paradigm_items].copy()

            # Dichotomize: TQ >= 1 → 1, TQ < 1 → 0 (per REMEMVR coding)
            # Note: dfData may already be dichotomized, but apply logic to be safe
            for item in paradigm_items:
                item_responses[item] = (item_responses[item] >= 1).astype(int)

            # Compute mean across items per UID × test (proportion correct)
            ctt_col = f"CTT_purified_{paradigm}"
            item_responses[ctt_col] = item_responses[paradigm_items].mean(axis=1)

            # Merge into output DataFrame
            ctt_purified = ctt_purified.merge(
                item_responses[['UID', 'test', ctt_col]],
                on=['UID', 'test'],
                how='left'
            )

            # Report statistics
            mean_score = item_responses[ctt_col].mean()
            sd_score = item_responses[ctt_col].std()
            min_score = item_responses[ctt_col].min()
            max_score = item_responses[ctt_col].max()
            log(f"           Mean: {mean_score:.3f}, SD: {sd_score:.3f}, Range: [{min_score:.3f}, {max_score:.3f}]")

        log("Purified CTT computation complete")
        # Save Analysis Output
        # These outputs will be used by: Step 4 (reliability assessment), Step 5 (correlation analysis)

        output_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        log(f"Saving {output_path.name}...")

        # Verify column order
        expected_cols = ['UID', 'test', 'CTT_purified_IFR', 'CTT_purified_ICR', 'CTT_purified_IRE']
        ctt_purified = ctt_purified[expected_cols]

        ctt_purified.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(ctt_purified)} rows, {len(ctt_purified.columns)} cols)")
        # Run Validation Tool
        # Validates: CTT scores in [0, 1], no NaN values, correct row count
        # Threshold: min=0.0, max=1.0 (proportion correct range)

        log("Running validate_numeric_range...")

        # Validate each paradigm CTT score column
        validation_passed = True
        for paradigm in PARADIGMS:
            col_name = f"CTT_purified_{paradigm}"
            validation_result = validate_numeric_range(
                data=ctt_purified[col_name],
                min_val=0.0,
                max_val=1.0,
                column_name=col_name
            )

            if validation_result['valid']:
                log(f"{col_name}: PASS (range [0, 1], {len(ctt_purified)} values)")
            else:
                log(f"{col_name}: FAIL - {validation_result['message']}")
                validation_passed = False

        # Additional validation checks
        # Check for NaN values
        nan_counts = ctt_purified[['CTT_purified_IFR', 'CTT_purified_ICR', 'CTT_purified_IRE']].isna().sum()
        if nan_counts.sum() > 0:
            log(f"FAIL - NaN values detected: {nan_counts.to_dict()}")
            validation_passed = False
        else:
            log(f"No NaN values detected: PASS")

        # Check row count (expect 400 rows: 100 UIDs × 4 tests)
        expected_rows = 400
        if len(ctt_purified) == expected_rows:
            log(f"Row count: PASS ({expected_rows} rows)")
        else:
            log(f"Row count: FAIL (expected {expected_rows}, got {len(ctt_purified)})")
            validation_passed = False

        # Check for duplicate UID × test combinations
        duplicates = ctt_purified.duplicated(subset=['UID', 'test']).sum()
        if duplicates == 0:
            log(f"No duplicate UID × test combinations: PASS")
        else:
            log(f"FAIL - {duplicates} duplicate UID × test combinations detected")
            validation_passed = False

        if validation_passed:
            log("Step 03 complete - All validations passed")
            sys.exit(0)
        else:
            log("Step 03 failed validation checks")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
