#!/usr/bin/env python3
"""Extract Item-Level Response Data: Extract raw item-level binary responses (0/1) from dfData.csv (wide format) and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_data_format, check_missing_data

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.9
LOG_FILE = RQ_DIR / "logs" / "step00_extract_response_data.log"

# Input files
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
RQ531_ITEM_PARAMS_PATH = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step03_item_parameters.csv"
RQ531_TSVR_MAPPING_PATH = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step00_tsvr_mapping.csv"

# Output files
OUTPUT_RESPONSE_DATA = RQ_DIR / "data" / "step00_response_level_data.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"

# Paradigm filter
PARADIGMS_INCLUDE = {"IFR", "ICR", "IRE"}  # Item-level paradigms
PARADIGMS_EXCLUDE = {"RFR", "TCR", "RRE"}  # Room-level paradigms

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract Item-Level Response Data")
        # Load RQ 5.3.1 Purified Items and Difficulty Parameters

        log("Loading RQ 5.3.1 item parameters...")
        df_item_params = pd.read_csv(RQ531_ITEM_PARAMS_PATH)
        log(f"{RQ531_ITEM_PARAMS_PATH.name} ({len(df_item_params)} rows, {len(df_item_params.columns)} cols)")
        log(f"Columns: {df_item_params.columns.tolist()}")

        # Extract purified item names and difficulty (b parameter)
        # Note: RQ 5.3.1 uses 'item' column, difficulty is 'Difficulty_1' (b parameter)
        purified_items = set(df_item_params['item'].values)
        log(f"Purified items from RQ 5.3.1: {len(purified_items)} items")

        # Create item -> difficulty mapping (use Difficulty_1 as b parameter)
        item_difficulty_map = dict(zip(df_item_params['item'], df_item_params['Difficulty_1']))
        # Load dfData.csv (Wide Format)

        log("Loading dfData.csv (wide format)...")
        df_wide = pd.read_csv(DFDATA_PATH)
        log(f"{DFDATA_PATH.name} ({len(df_wide)} rows, {len(df_wide.columns)} cols)")

        # Standardize column names (TEST -> Test for consistency with analysis spec)
        df_wide = df_wide.rename(columns={'TEST': 'Test'})
        # Identify Item-Level Response Columns (TQ_*)
        # Pattern: TQ_{paradigm}-{domain}-{item_id}
        # Example: TQ_IFR-N-i1, TQ_ICR-U-i2, TQ_IRE-D-i3
        # Exclude: Room-level paradigms (TQ_RFR-*, TQ_TCR-*, TQ_RRE-*)

        log("Identifying item-level response columns (TQ_IFR, TQ_ICR, TQ_IRE)...")
        tq_cols = [col for col in df_wide.columns if col.startswith('TQ_')]

        # Filter to item-level paradigms only (IFR, ICR, IRE)
        item_level_cols = [col for col in tq_cols
                           if any(f"TQ_{p}" in col for p in PARADIGMS_INCLUDE)]

        log(f"Total TQ columns: {len(tq_cols)}")
        log(f"Item-level paradigm columns (IFR/ICR/IRE): {len(item_level_cols)}")
        # Reshape dfData from Wide to Long Format
        # Transform: 400 rows × ~100 TQ columns -> ~40,000 rows (UID × Test × Item)

        log("Converting dfData from wide to long format...")

        # Identify ID columns to keep
        id_cols = ['UID', 'Test']

        # Melt: Wide -> Long format
        df_long = df_wide.melt(
            id_vars=id_cols,
            value_vars=item_level_cols,
            var_name='Item',
            value_name='Response'
        )

        log(f"Wide format ({len(df_wide)} rows) -> Long format ({len(df_long)} rows)")
        # Extract Paradigm Code from Item Names
        # Pattern: TQ_{paradigm}-{domain}-{item_id}
        # Extract paradigm: TQ_IFR-N-i1 -> IFR, TQ_ICR-U-i2 -> ICR

        log("Extracting paradigm codes from item names...")

        def extract_paradigm(item_name):
            """Extract paradigm code from TQ_* item name."""
            # TQ_IFR-N-i1 -> IFR
            parts = item_name.split('_')
            if len(parts) >= 2:
                paradigm_domain = parts[1]  # IFR-N-i1
                paradigm = paradigm_domain.split('-')[0]  # IFR
                return paradigm
            return None

        df_long['paradigm'] = df_long['Item'].apply(extract_paradigm)

        # Verify paradigm extraction
        paradigm_counts = df_long['paradigm'].value_counts()
        log(f"Paradigm distribution:")
        for paradigm, count in paradigm_counts.items():
            log(f"  {paradigm}: {count} observations")

        # Filter to item-level paradigms only (should already be filtered by column selection)
        df_long = df_long[df_long['paradigm'].isin(PARADIGMS_INCLUDE)]
        log(f"After paradigm filter (IFR/ICR/IRE): {len(df_long)} rows")
        # Filter to Purified Items from RQ 5.3.1
        # Keep ONLY items present in RQ 5.3.1 item_parameters.csv (purified items)

        log("Filtering to purified items from RQ 5.3.1...")
        log(f"Items before purification filter: {df_long['Item'].nunique()} unique items")

        df_long = df_long[df_long['Item'].isin(purified_items)]

        log(f"After purification filter: {len(df_long)} rows, {df_long['Item'].nunique()} unique items")
        # Merge Item Difficulty from RQ 5.3.1
        # Add difficulty (b parameter) from item_parameters.csv

        log("Merging item difficulty (b parameter) from RQ 5.3.1...")
        df_long['Difficulty'] = df_long['Item'].map(item_difficulty_map)

        # Check for missing difficulty values (should be none if purified_items filter worked)
        missing_difficulty = df_long['Difficulty'].isna().sum()
        if missing_difficulty > 0:
            log(f"{missing_difficulty} rows with missing Difficulty (item not in RQ 5.3.1 params)")
        else:
            log(f"All items successfully merged with difficulty parameters")
        # Save Response-Level Data
        # Output: Long format (UID × Test × Item observations) with difficulty

        log("Saving response-level data...")

        # Select and order columns per analysis spec
        output_cols = ['UID', 'Test', 'Item', 'Response', 'paradigm', 'Difficulty']
        df_output = df_long[output_cols].copy()

        # Sort for reproducibility
        df_output = df_output.sort_values(['UID', 'Test', 'Item']).reset_index(drop=True)

        df_output.to_csv(OUTPUT_RESPONSE_DATA, index=False, encoding='utf-8')
        log(f"{OUTPUT_RESPONSE_DATA.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Copy TSVR Mapping for Reference
        # Copy RQ 5.3.1 TSVR mapping to this RQ's data folder

        log("Copying TSVR mapping from RQ 5.3.1...")
        df_tsvr = pd.read_csv(RQ531_TSVR_MAPPING_PATH)
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"{OUTPUT_TSVR_MAPPING.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        # Run Validation - Data Format
        # Validate: All required columns present

        log("Running validate_data_format...")
        required_cols = ['UID', 'Test', 'Item', 'Response', 'paradigm', 'Difficulty']
        validation_result_format = validate_data_format(df_output, required_cols)

        if not validation_result_format['valid']:
            raise ValueError(f"Data format validation failed: {validation_result_format['message']}")

        log(f"Data format: {validation_result_format['message']}")
        # Run Validation - Missing Data Check
        # Validate: Response may have NaN, but Difficulty/paradigm must not

        log("Running check_missing_data...")
        validation_result_missing = check_missing_data(df_output)

        log(f"Missing data summary:")
        log(f"  Total missing: {validation_result_missing['total_missing']} / {validation_result_missing['total_cells']} cells ({validation_result_missing['percent_missing']:.2f}%)")

        if 'missing_by_column' in validation_result_missing:
            log(f"  Missing by column:")
            for col, count in validation_result_missing['missing_by_column'].items():
                log(f"    {col}: {count} missing values")

        # Custom validation: Check forbidden missing columns
        forbidden_cols = ['Difficulty', 'paradigm']
        for col in forbidden_cols:
            missing_count = df_output[col].isna().sum()
            if missing_count > 0:
                raise ValueError(f"Forbidden missing data: Column '{col}' has {missing_count} NaN values (must be zero)")

        log(f"Forbidden columns (Difficulty, paradigm) have zero NaN - PASS")

        # Report summary
        log("Step 00 extraction complete:")
        log(f"  Input: {len(df_wide)} rows (wide format)")
        log(f"  Output: {len(df_output)} rows (long format)")
        log(f"  Participants: {df_output['UID'].nunique()}")
        log(f"  Tests: {df_output['Test'].nunique()}")
        log(f"  Items: {df_output['Item'].nunique()}")
        log(f"  Paradigms: {df_output['paradigm'].nunique()} ({', '.join(sorted(df_output['paradigm'].unique()))})")
        log(f"  Response rate: {(~df_output['Response'].isna()).sum() / len(df_output) * 100:.1f}%")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
