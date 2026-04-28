#!/usr/bin/env python3
"""Create Composite ID: Create composite_ID identifier (UID_Test format) for cross-classified LMM modeling."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.9
LOG_FILE = RQ_DIR / "logs" / "step01_create_composite_id.log"

# Input files
INPUT_RESPONSE_DATA = RQ_DIR / "data" / "step00_response_level_data.csv"

# Output files
OUTPUT_ANALYSIS_READY = RQ_DIR / "data" / "step01_analysis_ready.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Create Composite ID")
        # Load Response-Level Data

        log("Loading response-level data from Step 00...")
        df = pd.read_csv(INPUT_RESPONSE_DATA)
        log(f"{INPUT_RESPONSE_DATA.name} ({len(df)} rows, {len(df.columns)} cols)")
        log(f"Columns: {df.columns.tolist()}")
        # Create Composite ID
        # Format: {UID}_{Test} (e.g., P001_T1, A010_1)

        log("Creating composite_ID (format: UID_Test)...")

        # Convert Test to string to handle numeric test values (1, 2, 3, 4)
        df['Test'] = df['Test'].astype(str)

        # Create composite_ID
        df['composite_ID'] = df['UID'] + '_' + df['Test']

        log(f"composite_ID column added")
        log(f"Example composite_IDs: {df['composite_ID'].head(3).tolist()}")

        # Verify expected count
        n_unique_composite = df['composite_ID'].nunique()
        expected_composite = 400  # 100 participants × 4 tests
        log(f"Unique composite_IDs: {n_unique_composite} (expected: {expected_composite})")

        if n_unique_composite != expected_composite:
            log(f"Unique composite_IDs ({n_unique_composite}) differs from expected ({expected_composite})")
        # Verify Paradigm Assignment Consistency
        # Check: Each Item should belong to exactly ONE paradigm across all observations

        log("Checking paradigm assignment consistency...")

        item_paradigm_check = df.groupby('Item')['paradigm'].nunique()
        items_multiple_paradigms = item_paradigm_check[item_paradigm_check > 1]

        if len(items_multiple_paradigms) > 0:
            log(f"{len(items_multiple_paradigms)} items belong to multiple paradigms:")
            for item, n_paradigms in items_multiple_paradigms.items():
                paradigms = df[df['Item'] == item]['paradigm'].unique()
                log(f"  {item}: {n_paradigms} paradigms ({', '.join(paradigms)})")
            raise ValueError(f"Data integrity violation: Items with multiple paradigm assignments")
        else:
            log(f"All items belong to exactly ONE paradigm")
        # Check for Duplicate UID × Test × Item Observations
        # Check: Each combination of (UID, Test, Item) should appear at most once

        log("Checking for duplicate UID × Test × Item observations...")

        # Identify duplicates
        duplicate_check = df.groupby(['UID', 'Test', 'Item']).size()
        duplicates = duplicate_check[duplicate_check > 1]

        if len(duplicates) > 0:
            log(f"{len(duplicates)} duplicate UID × Test × Item combinations found:")
            for (uid, test, item), count in duplicates.head(10).items():
                log(f"  {uid} × {test} × {item}: {count} observations")
            raise ValueError(f"Data integrity violation: Duplicate observations detected")
        else:
            log(f"No duplicate UID × Test × Item observations")
        # Sort and Save Analysis-Ready Data
        # Output: Long format with composite_ID added

        log("Saving analysis-ready data...")

        # Reorder columns: composite_ID first
        output_cols = ['composite_ID', 'UID', 'Test', 'Item', 'Response', 'paradigm', 'Difficulty']
        df_output = df[output_cols].copy()

        # Sort for reproducibility
        df_output = df_output.sort_values(['UID', 'Test', 'Item']).reset_index(drop=True)

        df_output.to_csv(OUTPUT_ANALYSIS_READY, index=False, encoding='utf-8')
        log(f"{OUTPUT_ANALYSIS_READY.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Run Validation - Data Format
        # Validate: All required columns present including composite_ID

        log("Running validate_data_format...")
        required_cols = ['composite_ID', 'UID', 'Test', 'Item', 'Response', 'paradigm', 'Difficulty']
        validation_result = validate_data_format(df_output, required_cols)

        if not validation_result['valid']:
            raise ValueError(f"Data format validation failed: {validation_result['message']}")

        log(f"Data format: {validation_result['message']}")
        # Verify Composite ID Format
        # Check: composite_ID follows expected format (contains "_")

        log("Verifying composite_ID format...")

        # Check format: should contain exactly one "_"
        format_check = df_output['composite_ID'].str.contains('_', na=False)
        invalid_format = (~format_check).sum()

        if invalid_format > 0:
            log(f"{invalid_format} composite_IDs with invalid format (missing '_')")
            raise ValueError(f"composite_ID format validation failed")
        else:
            log(f"All composite_IDs follow format: UID_Test")

        # Report summary
        log("Step 01 data structure verification complete:")
        log(f"  Total rows: {len(df_output)}")
        log(f"  Unique composite_IDs: {df_output['composite_ID'].nunique()}")
        log(f"  Unique participants: {df_output['UID'].nunique()}")
        log(f"  Unique tests: {df_output['Test'].nunique()}")
        log(f"  Unique items: {df_output['Item'].nunique()}")
        log(f"  Paradigms: {df_output['paradigm'].nunique()} ({', '.join(sorted(df_output['paradigm'].unique()))})")
        log(f"  Data integrity: No duplicates, paradigm assignment consistent")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
