#!/usr/bin/env python3
"""Extract VR Data: Extract 5-category ordinal confidence ratings (TC_* items) from dfData.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_vr_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract VR Data")
        # Load Input Data

        log("Loading dfData.csv...")
        dfdata_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        df = pd.read_csv(dfdata_path, encoding='utf-8')
        log(f"dfData.csv ({len(df)} rows, {len(df.columns)} cols)")
        # Get ALL TC_* Columns (All Paradigms)
        # dfData.csv is ALREADY in wide format (UID × items)
        # Paradigm is EMBEDDED in column names: TC_{PARADIGM}-{DOMAIN}-{ITEM}
        # Include ALL paradigms: IFR, ICR, IRE, RFR, TCR, RRE
        # This matches Ch5 5.1.1 methodology which uses all TQ_* items

        log("Identifying TC_* columns (confidence ratings)...")

        # Get all TC_* columns - NO filtering by paradigm
        tc_columns = [col for col in df.columns if col.startswith('TC_')]
        log(f"{len(tc_columns)} total TC_* columns")

        if len(tc_columns) == 0:
            raise ValueError("No TC_* columns found in dfData.csv. Check column naming convention")

        # Count by paradigm for logging
        paradigm_counts = {}
        for col in tc_columns:
            parts = col.replace('TC_', '').split('-')
            if len(parts) >= 1:
                paradigm = parts[0]
                paradigm_counts[paradigm] = paradigm_counts.get(paradigm, 0) + 1

        log(f"All paradigms included:")
        for paradigm, count in sorted(paradigm_counts.items()):
            log(f"  {paradigm}: {count} items")

        # Verify required columns exist
        required_cols = ['UID', 'TEST', 'TSVR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dfData.csv: {missing_cols}")

        # Use df directly (already filtered by column selection)
        df_interactive = df.copy()
        # Create composite_ID (UID_TEST)
        # composite_ID = unique identifier for each UID × TEST combination
        # Format: P001_T1, P001_T2, ..., P100_T4
        # Total: 100 participants × 4 tests = 400 composite_IDs

        log("Creating composite_ID column...")
        df_interactive['composite_ID'] = df_interactive['UID'].astype(str) + '_' + df_interactive['TEST'].astype(str)
        log(f"composite_ID with {df_interactive['composite_ID'].nunique()} unique values")
        # Pivot to Wide Format (composite_ID × TC_* items)
        # IRT expects wide format: one row per person-test, one column per item
        # Result: 400 rows × ~103 columns (composite_ID + ~102 TC_* items)

        log("Pivoting to wide format (composite_ID x items)...")

        # For each composite_ID, aggregate TC_* values
        # Strategy: For each TC_* column, take the value for that composite_ID
        # (assumes each composite_ID × item combination appears once; if duplicates, take first)

        # Select columns for pivot
        pivot_cols = ['composite_ID', 'UID', 'TEST', 'TSVR'] + tc_columns
        df_pivot = df_interactive[pivot_cols].copy()

        # Group by composite_ID and aggregate (take first non-NaN value for each item)
        df_wide = df_pivot.groupby('composite_ID', as_index=False).first()

        log(f"Wide format: {len(df_wide)} rows × {len(df_wide.columns)} cols")

        # Verify expected row count (400 = 100 participants × 4 tests)
        if len(df_wide) != 400:
            log(f"Expected 400 rows (100 participants × 4 tests), got {len(df_wide)}")
        # Create Output 1 - IRT Input (composite_ID × TC_* items)
        # Output: step00_irt_input.csv
        # Contains: composite_ID + all TC_* item columns

        log("Creating IRT input file (composite_ID + TC_* items)...")

        # Select composite_ID + TC_* columns only
        irt_input_cols = ['composite_ID'] + tc_columns
        df_irt_input = df_wide[irt_input_cols].copy()
        # CRITICAL FIX: Convert fractional values to integers for IRT
        # IRT GRM expects integer categories 0, 1, 2, 3, 4 for 5-category ordinal data
        # But dfData.csv contains fractional values: 0.2, 0.4, 0.6, 0.8, 1.0
        # This mapping preserves the ordinal structure while providing correct input for IRT

        log("Converting fractional confidence values to integers for IRT...")
        value_mapping = {
            0.2: 0,  # Lowest confidence -> 0
            0.4: 1,
            0.6: 2,  # Middle confidence -> 2
            0.8: 3,
            1.0: 4   # Highest confidence -> 4
        }

        # Apply conversion to all TC_* columns
        for col in tc_columns:
            df_irt_input[col] = df_irt_input[col].map(lambda x: value_mapping.get(x, x) if pd.notna(x) else x)

        log(f"Values mapped: 0.2→0, 0.4→1, 0.6→2, 0.8→3, 1.0→4")

        # Save to CSV
        irt_input_path = RQ_DIR / "data" / "step00_irt_input.csv"
        df_irt_input.to_csv(irt_input_path, index=False, encoding='utf-8')
        log(f"{irt_input_path.name} ({len(df_irt_input)} rows, {len(df_irt_input.columns)} cols)")
        # Create Output 2 - TSVR Mapping (composite_ID → TSVR_hours)
        # Output: step00_tsvr_mapping.csv
        # Contains: composite_ID, TSVR_hours, test

        log("Creating TSVR mapping file...")

        # Extract composite_ID, TSVR, TEST
        df_tsvr = df_wide[['composite_ID', 'UID', 'TEST', 'TSVR']].copy()

        # Rename columns for clarity
        df_tsvr = df_tsvr.rename(columns={
            'TSVR': 'TSVR_hours',
            'TEST': 'test'
        })

        # Drop UID (not needed in mapping; composite_ID already contains it)
        df_tsvr = df_tsvr[['composite_ID', 'TSVR_hours', 'test']]

        # Save to CSV
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        df_tsvr.to_csv(tsvr_path, index=False, encoding='utf-8')
        log(f"{tsvr_path.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        # Create Output 3 - Q-matrix (Single "All" Factor)
        # Output: step00_q_matrix.csv
        # Contains: item_name, All (1 for all items)

        log("Creating Q-matrix (omnibus 'All' factor)...")

        # Create Q-matrix: all TC_* items load on single "All" factor
        df_qmatrix = pd.DataFrame({
            'item_name': tc_columns,
            'All': 1  # All items load on the "All" factor
        })

        # Save to CSV
        qmatrix_path = RQ_DIR / "data" / "step00_q_matrix.csv"
        df_qmatrix.to_csv(qmatrix_path, index=False, encoding='utf-8')
        log(f"{qmatrix_path.name} ({len(df_qmatrix)} rows, {len(df_qmatrix.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns exist in all 3 output files
        # Threshold: All required columns must be present

        log("Validating output files...")

        # Validate irt_input.csv
        validation_result_irt = validate_data_columns(
            df=df_irt_input,
            required_columns=['composite_ID']
        )

        if not validation_result_irt['valid']:
            raise ValueError(f"IRT input validation failed: {validation_result_irt}")

        log(f"irt_input.csv: {validation_result_irt['n_required']} required columns present")

        # Validate tsvr_mapping.csv
        validation_result_tsvr = validate_data_columns(
            df=df_tsvr,
            required_columns=['composite_ID', 'TSVR_hours', 'test']
        )

        if not validation_result_tsvr['valid']:
            raise ValueError(f"TSVR mapping validation failed: {validation_result_tsvr}")

        log(f"tsvr_mapping.csv: {validation_result_tsvr['n_required']} required columns present")

        # Validate q_matrix.csv
        validation_result_qmatrix = validate_data_columns(
            df=df_qmatrix,
            required_columns=['item_name', 'All']
        )

        if not validation_result_qmatrix['valid']:
            raise ValueError(f"Q-matrix validation failed: {validation_result_qmatrix}")

        log(f"q_matrix.csv: {validation_result_qmatrix['n_required']} required columns present")

        # Additional validation checks
        log("Running additional data quality checks...")

        # Check TC_* values are in expected range (0, 1, 2, 3, 4, or NaN) after conversion
        expected_values = {0, 1, 2, 3, 4}
        for col in tc_columns:
            unique_vals = set(df_irt_input[col].dropna().unique())
            unexpected = unique_vals - expected_values
            if unexpected:
                log(f"Column {col} has unexpected values: {unexpected}")

        # Check TSVR_hours range (0-168 hours = 1 week)
        tsvr_min = df_tsvr['TSVR_hours'].min()
        tsvr_max = df_tsvr['TSVR_hours'].max()
        log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")

        if tsvr_min < 0 or tsvr_max > 168:
            log(f"TSVR_hours outside expected range [0, 168]")

        # Check for fully missing items or participants
        n_missing_per_item = df_irt_input[tc_columns].isna().sum()
        fully_missing_items = n_missing_per_item[n_missing_per_item == len(df_irt_input)]
        if len(fully_missing_items) > 0:
            log(f"{len(fully_missing_items)} items have all missing values: {list(fully_missing_items.index)}")

        n_missing_per_participant = df_irt_input[tc_columns].isna().sum(axis=1)
        fully_missing_participants = df_irt_input[n_missing_per_participant == len(tc_columns)]
        if len(fully_missing_participants) > 0:
            log(f"{len(fully_missing_participants)} participants have all missing TC_* values")

        # Check composite_ID format (UID_TEST pattern)
        invalid_ids = df_irt_input[~df_irt_input['composite_ID'].str.match(r'^P\d{3}_T[1-4]$')]
        if len(invalid_ids) > 0:
            log(f"{len(invalid_ids)} composite_IDs don't match expected format (P###_T#)")
            log(f"Examples: {list(invalid_ids['composite_ID'].head())}")

        log("All validation checks complete")

        # Summary statistics
        log("Extraction complete:")
        log(f"  - Input: {len(df)} rows from dfData.csv")
        log(f"  - Filtered: {len(df_interactive)} rows (interactive paradigms)")
        log(f"  - TC_* items: {len(tc_columns)}")
        log(f"  - Composite IDs: {len(df_irt_input)}")
        log(f"  - Output files: 3 (irt_input, tsvr_mapping, q_matrix)")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
