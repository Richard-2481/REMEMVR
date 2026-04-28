#!/usr/bin/env python3
"""Extract Source and Destination Location Data: Extract source (-U-) and destination (-D-) location items from interactive VR"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_vr_data.log"

# Input file (relative to project root)
INPUT_FILE = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Output files (relative to RQ_DIR)
OUTPUT_IRT_INPUT = RQ_DIR / "data" / "step00_irt_input.csv"
OUTPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Extract Source and Destination Location Data")
        # Load Input Data

        log("Loading dfData.csv...")
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"dfData.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required base columns exist
        required_base_cols = ['UID', 'TEST', 'TSVR']
        missing_base_cols = [col for col in required_base_cols if col not in df.columns]
        if missing_base_cols:
            raise ValueError(f"Missing required columns: {missing_base_cols}")

        log(f"Base columns present: {required_base_cols}")
        # Filter to Source/Destination Items from Interactive Paradigms

        log("Identifying source/destination items from IFR/ICR/IRE...")

        # Pattern: TQ_{IFR|ICR|IRE}-{U|D}-i{1-6}
        # Example: TQ_IFR-U-i1, TQ_ICR-D-i3, TQ_IRE-U-i5
        item_pattern = re.compile(r'^TQ_(IFR|ICR|IRE)-(U|D)-i\d+$')

        item_cols = [col for col in df.columns if item_pattern.match(col)]
        item_cols.sort()  # Alphabetical order for consistency

        log(f"{len(item_cols)} source/destination items")

        if len(item_cols) == 0:
            raise ValueError("No items matching pattern TQ_{IFR|ICR|IRE}-(U|D)-iX found")
        if len(item_cols) != 36:
            log(f"Expected 36 items, found {len(item_cols)}")

        # Log breakdown by paradigm and location type
        for paradigm in ['IFR', 'ICR', 'IRE']:
            for location in ['U', 'D']:
                pattern_specific = re.compile(f'^TQ_{paradigm}-{location}-i\\d+$')
                count = sum(1 for col in item_cols if pattern_specific.match(col))
                log(f"{paradigm}-{location}: {count} items")
        # Dichotomize Item Responses
        # Threshold: TQ < 1.0 -> 0 (incorrect), TQ >= 1.0 -> 1 (correct)

        log("Converting TQ values to binary (threshold=1.0)...")

        # Create copy for IRT input
        df_irt = df[['UID', 'TEST'] + item_cols].copy()

        # Dichotomize: TQ < 1 -> 0, TQ >= 1 -> 1
        # NaN values remain NaN (missing responses)
        for col in item_cols:
            df_irt[col] = df_irt[col].apply(
                lambda x: np.nan if pd.isna(x) else (1 if x >= 1.0 else 0)
            )

        log(f"{len(item_cols)} items converted to binary")

        # Check value distribution
        total_values = df_irt[item_cols].count().sum()
        total_zeros = (df_irt[item_cols] == 0).sum().sum()
        total_ones = (df_irt[item_cols] == 1).sum().sum()
        total_nan = df_irt[item_cols].isna().sum().sum()

        log(f"Total responses: {total_values}")
        log(f"Zeros (incorrect): {total_zeros} ({100*total_zeros/total_values:.1f}%)")
        log(f"Ones (correct): {total_ones} ({100*total_ones/total_values:.1f}%)")
        log(f"Missing (NaN): {total_nan}")
        # Create composite_ID and Reshape to Wide Format

        log("Creating composite_ID and reshaping to wide format...")

        # Create composite_ID: UID + '_' + TEST
        # Example: A010_1, A010_2, B023_3, B023_4
        df_irt['composite_ID'] = df_irt['UID'].astype(str) + '_' + df_irt['TEST'].astype(str)

        # Reorder columns: composite_ID first, then items
        cols_order = ['composite_ID'] + item_cols
        df_irt = df_irt[cols_order]

        log(f"Wide-format IRT input created: {len(df_irt)} rows x {len(df_irt.columns)} cols")

        # Verify no duplicate composite_IDs
        if df_irt['composite_ID'].duplicated().any():
            duplicates = df_irt['composite_ID'][df_irt['composite_ID'].duplicated()].tolist()
            raise ValueError(f"Duplicate composite_IDs found: {duplicates[:10]}")

        log(f"No duplicate composite_IDs")
        # Create Q-Matrix for 2-Factor IRT

        log("[Q-MATRIX] Creating Q-matrix for 2-factor IRT...")

        q_matrix_data = []

        for item in item_cols:
            # Extract location type from tag (U=source, D=destination)
            if '-U-' in item:
                # Source item (loads on source factor only)
                q_matrix_data.append({
                    'item_tag': item,
                    'source': 1,
                    'destination': 0
                })
            elif '-D-' in item:
                # Destination item (loads on destination factor only)
                q_matrix_data.append({
                    'item_tag': item,
                    'source': 0,
                    'destination': 1
                })
            else:
                raise ValueError(f"Item {item} does not match expected pattern (missing -U- or -D-)")

        df_q_matrix = pd.DataFrame(q_matrix_data)

        log(f"[Q-MATRIX] Created Q-matrix: {len(df_q_matrix)} rows x {len(df_q_matrix.columns)} cols")

        # Validate Q-matrix structure
        source_items = (df_q_matrix['source'] == 1).sum()
        destination_items = (df_q_matrix['destination'] == 1).sum()

        log(f"[Q-MATRIX] Source items: {source_items}")
        log(f"[Q-MATRIX] Destination items: {destination_items}")

        # Check that each item loads on exactly one factor (row sum = 1)
        row_sums = df_q_matrix[['source', 'destination']].sum(axis=1)
        if not (row_sums == 1).all():
            invalid_items = df_q_matrix.loc[row_sums != 1, 'item_tag'].tolist()
            raise ValueError(f"Q-matrix validation failed: Items not loading on exactly one factor: {invalid_items}")

        log(f"Q-matrix validation passed: Each item loads on exactly one factor")
        # Extract TSVR Time Variable (Decision D070)

        log("Extracting TSVR time variable...")

        df_tsvr = df[['UID', 'TEST', 'TSVR']].copy()
        df_tsvr['composite_ID'] = df_tsvr['UID'].astype(str) + '_' + df_tsvr['TEST'].astype(str)

        # Rename TSVR to TSVR_hours for clarity (Decision D070)
        df_tsvr = df_tsvr.rename(columns={'TSVR': 'TSVR_hours', 'TEST': 'test'})

        # Reorder columns
        df_tsvr = df_tsvr[['composite_ID', 'UID', 'test', 'TSVR_hours']]

        log(f"TSVR mapping created: {len(df_tsvr)} rows x {len(df_tsvr.columns)} cols")

        # Validate TSVR range (0-168 hours for 7-day study)
        tsvr_min = df_tsvr['TSVR_hours'].min()
        tsvr_max = df_tsvr['TSVR_hours'].max()
        tsvr_mean = df_tsvr['TSVR_hours'].mean()

        log(f"Range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")
        log(f"Mean: {tsvr_mean:.2f} hours")

        if tsvr_min < 0 or tsvr_max > 168:
            log(f"TSVR values outside expected [0, 168] range")
        # Save Output Files
        # These outputs will be used by: Step 1 (IRT Pass 1 calibration)

        log(f"Saving output files...")

        # Save IRT input (wide format)
        OUTPUT_IRT_INPUT.parent.mkdir(parents=True, exist_ok=True)
        df_irt.to_csv(OUTPUT_IRT_INPUT, index=False, encoding='utf-8')
        log(f"{OUTPUT_IRT_INPUT.name} ({len(df_irt)} rows, {len(df_irt.columns)} cols)")

        # Save Q-matrix
        df_q_matrix.to_csv(OUTPUT_Q_MATRIX, index=False, encoding='utf-8')
        log(f"{OUTPUT_Q_MATRIX.name} ({len(df_q_matrix)} rows, {len(df_q_matrix.columns)} cols)")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"{OUTPUT_TSVR_MAPPING.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        # Validation Checks
        # Validates: File existence, row counts, value ranges, data consistency

        log("Running validation checks...")

        validation_passed = True

        # Check 1: IRT input has 400 rows
        if len(df_irt) != 400:
            log(f"IRT input has {len(df_irt)} rows, expected 400")
            validation_passed = False
        else:
            log(f"IRT input has 400 rows")

        # Check 2: All item values in {0, 1, NaN}
        valid_values = {0, 1}
        for col in item_cols:
            unique_vals = set(df_irt[col].dropna().unique())
            if not unique_vals.issubset(valid_values):
                log(f"Column {col} has invalid values: {unique_vals - valid_values}")
                validation_passed = False
        log(f"All item values in {{0, 1, NaN}}")

        # Check 3: Q-matrix has 36 rows, 18 source, 18 destination
        if len(df_q_matrix) != 36:
            log(f"Q-matrix has {len(df_q_matrix)} rows, expected 36")
            validation_passed = False
        elif source_items != 18 or destination_items != 18:
            log(f"Q-matrix has {source_items} source items, {destination_items} destination items (expected 18 each)")
            validation_passed = False
        else:
            log(f"Q-matrix has 36 rows (18 source, 18 destination)")

        # Check 4: TSVR mapping has 400 rows, TSVR_hours in [0, 168]
        if len(df_tsvr) != 400:
            log(f"TSVR mapping has {len(df_tsvr)} rows, expected 400")
            validation_passed = False
        else:
            log(f"TSVR mapping has 400 rows")

        tsvr_out_of_range = ((df_tsvr['TSVR_hours'] < 0) | (df_tsvr['TSVR_hours'] > 168)).sum()
        if tsvr_out_of_range > 0:
            log(f"{tsvr_out_of_range} TSVR values outside [0, 168] range")
        else:
            log(f"All TSVR_hours in [0, 168]")

        # Check 5: All 400 composite_IDs present in all 3 files
        ids_irt = set(df_irt['composite_ID'])
        ids_tsvr = set(df_tsvr['composite_ID'])

        if ids_irt != ids_tsvr:
            missing_in_irt = ids_tsvr - ids_irt
            missing_in_tsvr = ids_irt - ids_tsvr
            log(f"Composite_ID mismatch:")
            if missing_in_irt:
                log(f"  Missing in IRT input: {len(missing_in_irt)} IDs")
            if missing_in_tsvr:
                log(f"  Missing in TSVR mapping: {len(missing_in_tsvr)} IDs")
            validation_passed = False
        else:
            log(f"All 400 composite_IDs present in IRT input and TSVR mapping")

        # Check 6: Missing data report (<20% per item acceptable, >50% error)
        missing_pct_per_item = df_irt[item_cols].isna().sum() / len(df_irt) * 100
        items_high_missing = missing_pct_per_item[missing_pct_per_item > 50]

        if len(items_high_missing) > 0:
            log(f"{len(items_high_missing)} items with >50% missing data:")
            for item, pct in items_high_missing.items():
                log(f"  {item}: {pct:.1f}% missing")
            validation_passed = False
        else:
            max_missing = missing_pct_per_item.max()
            log(f"No items with >50% missing data (max: {max_missing:.1f}%)")

        # Final validation result
        if validation_passed:
            log("All validation checks passed")
        else:
            raise ValueError("Validation failed - see log for details")

        log("Step 0 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
