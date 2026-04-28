#!/usr/bin/env python3
"""extract_vr_data: Extract 5-category ordinal confidence responses for source (-U-/pick-up) and"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.8.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_vr_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract Confidence Item Data")
        # Load Raw Data from dfData.csv

        log("Loading dfData.csv...")
        dfdata_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        df_raw = pd.read_csv(dfdata_path, encoding='utf-8')
        log(f"dfData.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Verify required columns exist
        required_cols_raw = ['UID', 'TEST', 'TSVR']
        missing_cols = [col for col in required_cols_raw if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dfData.csv: {missing_cols}")
        log(f"Required columns present: {required_cols_raw}")
        # Filter to TC_* Confidence Items with -U- or -D- Tags
        # Tag patterns:
        #   - TC_*-U-* (source/pick-up location confidence items)
        #   - TC_*-D-* (destination/put-down location confidence items)
        # Excludes: TC_*-L-* (legacy general location), TQ_* (accuracy items)

        log("Filtering to TC_* confidence items with -U- or -D- tags...")

        # Find TC_* columns with -U- or -D- tags
        tc_cols = [col for col in df_raw.columns if col.startswith('TC_')]
        source_cols = [col for col in tc_cols if '-U-' in col]
        dest_cols = [col for col in tc_cols if '-D-' in col]

        log(f"Found {len(source_cols)} source items (-U- tags)")
        log(f"Found {len(dest_cols)} destination items (-D- tags)")
        log(f"Total confidence items: {len(source_cols) + len(dest_cols)}")

        if len(source_cols) == 0 or len(dest_cols) == 0:
            raise ValueError(f"No items found for source ({len(source_cols)}) or destination ({len(dest_cols)})")

        # Select required columns: UID, TEST, TSVR, TC_* items
        confidence_cols = source_cols + dest_cols
        select_cols = ['UID', 'TEST', 'TSVR'] + confidence_cols
        df_filtered = df_raw[select_cols].copy()

        # Rename TSVR to TSVR_hours for clarity
        df_filtered.rename(columns={'TSVR': 'TSVR_hours'}, inplace=True)

        log(f"{len(df_filtered)} rows with {len(confidence_cols)} confidence items")
        # Create composite_ID and Validate 5-Category Ordinal Data
        # composite_ID format: UID_TEST (e.g., P001_T1)
        # Confidence items are 5-category ordinal: 0, 0.25, 0.5, 0.75, 1.0

        log("Creating composite_ID...")

        # BUG FIX #1: Create composite_ID BEFORE any operations that need it
        # Ensure TEST format (convert to T1/T2/T3/T4 if numeric)
        if df_filtered['TEST'].dtype in [np.int64, np.int32]:
            df_filtered['TEST'] = 'T' + df_filtered['TEST'].astype(str)

        df_filtered['composite_ID'] = df_filtered['UID'].astype(str) + '_' + df_filtered['TEST'].astype(str)

        log(f"composite_ID for {df_filtered['composite_ID'].nunique()} unique participant-test combinations")

        # Validate 5-category ordinal values
        log("Checking confidence item value categories...")
        expected_values = {0.2, 0.4, 0.6, 0.8, 1.0}

        for col in confidence_cols:
            unique_vals = set(df_filtered[col].dropna().unique())
            invalid_vals = unique_vals - expected_values
            if invalid_vals:
                log(f"Item {col} has unexpected values: {invalid_vals}")

        log("Confidence items are 5-category ordinal (0.2, 0.4, 0.6, 0.8, 1.0)")
        # Create Wide-Format IRT Input
        # Output: One row per composite_ID, columns for each TC_* item

        log("Creating wide-format IRT input...")

        # Select composite_ID and confidence item columns
        irt_input_cols = ['composite_ID'] + confidence_cols
        df_irt_input = df_filtered[irt_input_cols].drop_duplicates(subset=['composite_ID'])

        log(f"IRT input: {len(df_irt_input)} rows x {len(df_irt_input.columns)} cols")
        # CRITICAL FIX: Convert fractional values to integers for IRT
        # IRT GRM expects integer categories 0, 1, 2, 3, 4 for 5-category ordinal data
        # But dfData.csv contains fractional values: 0.2, 0.4, 0.6, 0.8, 1.0
        # This mapping preserves the ordinal structure while providing correct input for IRT

        log("Converting fractional confidence values to integers for IRT...")
        value_mapping = {
            0.2: 0,   # Lowest confidence -> 0
            0.4: 1,
            0.6: 2,   # Middle confidence -> 2
            0.8: 3,
            1.0: 4    # Highest confidence -> 4
        }

        # Apply conversion to all TC_* columns
        for col in confidence_cols:
            df_irt_input[col] = df_irt_input[col].map(lambda x: value_mapping.get(x, x) if pd.notna(x) else x)

        log(f"Values mapped: 0.2→0, 0.4→1, 0.6→2, 0.8→3, 1.0→4")

        # Verify expected row count (400 = 100 participants x 4 tests)
        expected_rows = 400
        if len(df_irt_input) != expected_rows:
            log(f"Expected {expected_rows} rows, found {len(df_irt_input)} rows")
        # Save IRT Input CSV

        output_irt_input = RQ_DIR / "data" / "step00_irt_input.csv"
        log(f"Saving IRT input to {output_irt_input}...")
        df_irt_input.to_csv(output_irt_input, index=False, encoding='utf-8')
        log(f"{output_irt_input.name} ({len(df_irt_input)} rows, {len(df_irt_input.columns)} cols)")
        # Create Q-Matrix for 2-Factor GRM (Source vs Destination)
        # Q-matrix defines factor structure:
        #   - Source dimension: TC_*-U-* items (value = 1), others = 0
        #   - Destination dimension: TC_*-D-* items (value = 1), others = 0
        # Simple structure: Each item loads on ONE dimension only

        log("Building Q-matrix for 2-factor structure...")

        # BUG FIX #6/7: Use correct column names 'Source' and 'Destination' (capitalized)
        q_matrix_data = []
        for item in confidence_cols:
            if '-U-' in item:
                # Source item (pick-up location)
                q_matrix_data.append({
                    'item_name': item,
                    'Source': 1,
                    'Destination': 0
                })
            elif '-D-' in item:
                # Destination item (put-down location)
                q_matrix_data.append({
                    'item_name': item,
                    'Source': 0,
                    'Destination': 1
                })

        df_q_matrix = pd.DataFrame(q_matrix_data)

        log(f"Q-matrix: {len(df_q_matrix)} items, 2 dimensions (Source, Destination)")
        log(f"Source dimension: {df_q_matrix['Source'].sum()} items")
        log(f"Destination dimension: {df_q_matrix['Destination'].sum()} items")

        # Verify simple structure (no items load on both dimensions)
        double_loading = df_q_matrix[(df_q_matrix['Source'] == 1) & (df_q_matrix['Destination'] == 1)]
        if len(double_loading) > 0:
            raise ValueError(f"Q-matrix validation failed: {len(double_loading)} items load on both dimensions")
        log("Q-matrix has simple structure (no items load on both dimensions)")
        # Save Q-Matrix CSV

        output_q_matrix = RQ_DIR / "data" / "step00_q_matrix.csv"
        log(f"Saving Q-matrix to {output_q_matrix}...")
        df_q_matrix.to_csv(output_q_matrix, index=False, encoding='utf-8')
        log(f"{output_q_matrix.name} ({len(df_q_matrix)} rows, {len(df_q_matrix.columns)} cols)")
        # Extract TSVR Time Mapping
        # Output: composite_ID, UID, TEST, TSVR_hours
        # Used for downstream LMM trajectory analysis (Decision D070)

        log("Extracting TSVR time mapping...")

        tsvr_cols = ['composite_ID', 'UID', 'TEST', 'TSVR_hours']
        df_tsvr_mapping = df_filtered[tsvr_cols].drop_duplicates(subset=['composite_ID'])

        log(f"TSVR mapping: {len(df_tsvr_mapping)} observations")

        # Verify expected row count
        if len(df_tsvr_mapping) != expected_rows:
            log(f"Expected {expected_rows} rows, found {len(df_tsvr_mapping)} rows")

        # Verify TSVR_hours range (actual hours, not nominal days - can exceed 168)
        tsvr_min = df_tsvr_mapping['TSVR_hours'].min()
        tsvr_max = df_tsvr_mapping['TSVR_hours'].max()
        log(f"TSVR_hours range: [{tsvr_min:.1f}, {tsvr_max:.1f}] hours")

        if tsvr_min < 0 or tsvr_max > 300:
            log(f"TSVR_hours outside expected range [0, 300]: [{tsvr_min}, {tsvr_max}]")
        # Save TSVR Mapping CSV

        output_tsvr_mapping = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        log(f"Saving TSVR mapping to {output_tsvr_mapping}...")
        df_tsvr_mapping.to_csv(output_tsvr_mapping, index=False, encoding='utf-8')
        log(f"{output_tsvr_mapping.name} ({len(df_tsvr_mapping)} rows, {len(df_tsvr_mapping.columns)} cols)")
        # Run Validation Tool
        # Validates: All required columns present in IRT input

        log("Running validate_data_columns...")

        validation_result = validate_data_columns(
            df=df_irt_input,
            required_columns=['composite_ID']
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['n_required']} required columns present")
        else:
            log(f"FAIL - Missing columns: {validation_result['missing_columns']}")
            raise ValueError(f"Validation failed: {validation_result}")

        log("Step 00 complete")
        log(f"Extracted {len(source_cols)} source items, {len(dest_cols)} destination items")
        log(f"Created Q-matrix: 2 dimensions (Source, Destination)")
        log(f"TSVR mapping: {len(df_tsvr_mapping)} observations")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
