#!/usr/bin/env python3
"""Load Data from RQ 5.2.1 and Master Dataset: Load IRT theta scores from RQ 5.2.1 and extract raw VR item data for CTT computation."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_load_data.log"

# Input paths (from RQ 5.1)
RQ1_DIR = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data"
THETA_INPUT = RQ1_DIR / "step03_theta_scores.csv"
TSVR_INPUT = RQ1_DIR / "step00_tsvr_mapping.csv"
PURIFIED_ITEMS_INPUT = RQ1_DIR / "step02_purified_items.csv"
RAW_DATA_INPUT = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Output paths
DATA_DIR = RQ_DIR / "data"
THETA_OUTPUT = DATA_DIR / "step00_irt_theta_loaded.csv"
TSVR_OUTPUT = DATA_DIR / "step00_tsvr_loaded.csv"
PURIFIED_ITEMS_OUTPUT = DATA_DIR / "step00_purified_items.csv"
RAW_FILTERED_OUTPUT = DATA_DIR / "step00_raw_data_filtered.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Data from RQ 5.1 and Master Dataset")
        # Load IRT Theta Scores from RQ 5.1

        log("Loading IRT theta scores from RQ 5.2.1...")
        irt_theta_full = pd.read_csv(THETA_INPUT)
        log(f"{THETA_INPUT.name} ({len(irt_theta_full)} rows, {len(irt_theta_full.columns)} cols)")
        log(f"  Columns (original): {irt_theta_full.columns.tolist()}")

        # EXCLUDE theta_when (floor effects discovered in RQ 5.2.1)
        log("Excluding theta_when (floor effects - 6-9% probability at encoding)")
        irt_theta = irt_theta_full[['composite_ID', 'theta_what', 'theta_where']].copy()
        log(f"  Columns (after exclusion): {irt_theta.columns.tolist()}")
        log(f"  Theta range - What: [{irt_theta['theta_what'].min():.2f}, {irt_theta['theta_what'].max():.2f}]")
        log(f"  Theta range - Where: [{irt_theta['theta_where'].min():.2f}, {irt_theta['theta_where'].max():.2f}]")
        # Load TSVR Mapping from RQ 5.2.1

        log("Loading TSVR mapping from RQ 5.2.1...")
        tsvr_data = pd.read_csv(TSVR_INPUT)
        log(f"{TSVR_INPUT.name} ({len(tsvr_data)} rows, {len(tsvr_data.columns)} cols)")
        log(f"  Columns: {tsvr_data.columns.tolist()}")
        log(f"  TSVR range: [{tsvr_data['TSVR_hours'].min():.2f}, {tsvr_data['TSVR_hours'].max():.2f}] hours")
        # Load Purified Items List from RQ 5.2.1 (Exclude When domain)
        # CRITICAL: Exclude factor='when' due to floor effects

        log("Loading purified items list from RQ 5.2.1...")
        purified_items_full = pd.read_csv(PURIFIED_ITEMS_INPUT)
        log(f"{PURIFIED_ITEMS_INPUT.name} ({len(purified_items_full)} rows, {len(purified_items_full.columns)} cols)")
        log(f"  Columns: {purified_items_full.columns.tolist()}")

        # Count items per factor BEFORE filtering
        factor_counts_before = purified_items_full['factor'].value_counts().to_dict()
        log(f"  Items per factor (BEFORE filter): {factor_counts_before}")

        # EXCLUDE When domain items (floor effects)
        log("Excluding When domain items (factor='when') due to floor effects")
        purified_items = purified_items_full[purified_items_full['factor'].isin(['what', 'where'])].copy()

        # Count items per factor AFTER filtering
        factor_counts = purified_items['factor'].value_counts().to_dict()
        log(f"  Items per factor (AFTER filter): {factor_counts}")
        log(f"  Total items after exclusion: {len(purified_items)} (expected ~64)")
        # Load Raw Master Data

        log("Loading raw master data...")
        raw_data = pd.read_csv(RAW_DATA_INPUT)
        log(f"{RAW_DATA_INPUT.name} ({len(raw_data)} rows, {len(raw_data.columns)} cols)")
        # Filter Raw Data to Purified Items Only
        # Critical: CTT must use SAME items as IRT for valid comparison
        # Strategy: Keep UID, TEST columns + only item columns in purified_items list

        log("Filtering raw data to purified items only...")

        # Get purified item names
        purified_item_names = purified_items['item_name'].tolist()
        log(f"  Purified item count: {len(purified_item_names)}")

        # Check which purified items exist in raw data
        available_items = [col for col in purified_item_names if col in raw_data.columns]
        missing_items = [col for col in purified_item_names if col not in raw_data.columns]

        log(f"  Items available in raw data: {len(available_items)}")
        if missing_items:
            log(f"  Items in purified list but NOT in raw data: {len(missing_items)}")
            log(f"    Missing items: {missing_items}")

        # Select columns: UID, TEST, + purified items
        base_cols = ['UID', 'TEST']
        cols_to_keep = base_cols + available_items

        raw_data_filtered = raw_data[cols_to_keep].copy()
        log(f"Raw data filtered ({len(raw_data_filtered)} rows, {len(raw_data_filtered.columns)} cols)")
        log(f"  Base columns: {base_cols}")
        log(f"  Item columns: {len(available_items)}")
        # STEP 5b: DICHOTOMIZE Item Scores (CRITICAL FOR CTT-IRT COMPARISON)
        # Rule: 1 stays 1, all other values (<1) become 0
        # Reason: RQ 5.1 IRT used dichotomized data, CTT must use same for fair comparison
        # Reference: 1_concept.md line 135

        log("Applying dichotomization rule: 1=1, <1=0...")

        # Get item column names (exclude UID, TEST)
        item_cols = [col for col in raw_data_filtered.columns if col not in base_cols]

        # Check values BEFORE dichotomization
        unique_values_before = set()
        for col in item_cols:
            unique_values_before.update(raw_data_filtered[col].dropna().unique())
        log(f"  Unique values BEFORE dichotomization: {sorted(unique_values_before)}")

        # Apply dichotomization: 1 stays 1, everything else becomes 0
        for col in item_cols:
            raw_data_filtered[col] = (raw_data_filtered[col] == 1).astype(int)

        # Check values AFTER dichotomization
        unique_values_after = set()
        for col in item_cols:
            unique_values_after.update(raw_data_filtered[col].dropna().unique())
        log(f"  Unique values AFTER dichotomization: {sorted(unique_values_after)}")

        # Validate dichotomization success
        if unique_values_after != {0, 1}:
            raise ValueError(f"Dichotomization FAILED: Expected {{0, 1}}, got {unique_values_after}")

        log("SUCCESS: All item scores are now binary (0 or 1)")
        # Save All Outputs
        # Save local copies for RQ 5.11 analysis pipeline

        log("Saving output files...")

        # Save IRT theta scores
        irt_theta.to_csv(THETA_OUTPUT, index=False, encoding='utf-8')
        log(f"{THETA_OUTPUT.name} ({len(irt_theta)} rows, {len(irt_theta.columns)} cols)")

        # Save TSVR mapping
        tsvr_data.to_csv(TSVR_OUTPUT, index=False, encoding='utf-8')
        log(f"{TSVR_OUTPUT.name} ({len(tsvr_data)} rows, {len(tsvr_data.columns)} cols)")

        # Save purified items
        purified_items.to_csv(PURIFIED_ITEMS_OUTPUT, index=False, encoding='utf-8')
        log(f"{PURIFIED_ITEMS_OUTPUT.name} ({len(purified_items)} rows, {len(purified_items.columns)} cols)")

        # Save filtered raw data
        raw_data_filtered.to_csv(RAW_FILTERED_OUTPUT, index=False, encoding='utf-8')
        log(f"{RAW_FILTERED_OUTPUT.name} ({len(raw_data_filtered)} rows, {len(raw_data_filtered.columns)} cols)")
        # Run Validation
        # Validation: Check data formats, ranges, completeness

        log("Running data format validation...")

        # Validation 1: IRT theta format and ranges (NO theta_when - excluded)
        log("  IRT theta format (What, Where only - When excluded)...")
        theta_validation = validate_data_format(
            df=irt_theta,
            required_cols=['composite_ID', 'theta_what', 'theta_where']  # NO theta_when
        )

        if not theta_validation['valid']:
            raise ValueError(f"IRT theta validation failed: {theta_validation['message']}")

        # Check theta ranges (What and Where only)
        theta_min = min(irt_theta['theta_what'].min(), irt_theta['theta_where'].min())
        theta_max = max(irt_theta['theta_what'].max(), irt_theta['theta_where'].max())

        if theta_min < -3 or theta_max > 3:
            log(f"  Theta values outside typical range [-3, 3]: [{theta_min:.2f}, {theta_max:.2f}]")
        else:
            log(f"  Theta values in expected range: [{theta_min:.2f}, {theta_max:.2f}]")

        # Validation 2: TSVR format and ranges
        log("  TSVR format...")
        tsvr_validation = validate_data_format(
            df=tsvr_data,
            required_cols=['composite_ID', 'UID', 'test', 'TSVR_hours']
        )

        if not tsvr_validation['valid']:
            raise ValueError(f"TSVR validation failed: {tsvr_validation['message']}")

        # Check TSVR ranges
        tsvr_min = tsvr_data['TSVR_hours'].min()
        tsvr_max = tsvr_data['TSVR_hours'].max()

        if tsvr_min < 0 or tsvr_max > 300:
            log(f"  TSVR_hours outside expected range [0, 300]: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
        else:
            log(f"  TSVR_hours in expected range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        # Validation 3: Purified items format
        log("  Purified items format...")
        items_validation = validate_data_format(
            df=purified_items,
            required_cols=['item_name', 'factor', 'a', 'b']
        )

        if not items_validation['valid']:
            raise ValueError(f"Purified items validation failed: {items_validation['message']}")

        # Check item count (expected ~64 after When exclusion)
        item_count = len(purified_items)
        if item_count < 50 or item_count > 80:
            log(f"  Purified item count outside expected range [50, 80]: {item_count}")
        else:
            log(f"  Purified item count in expected range: {item_count}")

        # Verify When items are excluded
        when_items = purified_items[purified_items['factor'] == 'when']
        if len(when_items) > 0:
            raise ValueError(f"When items not properly excluded! Found {len(when_items)} When items")
        else:
            log(f"  When items properly excluded (0 items with factor='when')")

        # Validation 4: Raw data filtered format
        log("  Raw data filtered format...")
        raw_validation = validate_data_format(
            df=raw_data_filtered,
            required_cols=['UID', 'TEST']
        )

        if not raw_validation['valid']:
            raise ValueError(f"Raw data validation failed: {raw_validation['message']}")

        # Check row count
        if len(raw_data_filtered) != 400:
            log(f"  Raw data row count not 400: {len(raw_data_filtered)}")
        else:
            log(f"  Raw data row count: {len(raw_data_filtered)}")

        # Check item columns (expected ~64 after When exclusion)
        item_col_count = len(raw_data_filtered.columns) - 2  # Subtract UID, TEST
        if item_col_count < 50:
            raise ValueError(f"Too few item columns in filtered data: {item_col_count} (expected >= 50)")
        else:
            log(f"  Item columns in filtered data: {item_col_count} (expected ~64)")

        # Validation 5: Composite ID alignment
        log("  Composite ID alignment across files...")
        theta_ids = set(irt_theta['composite_ID'].unique())
        tsvr_ids = set(tsvr_data['composite_ID'].unique())

        if theta_ids != tsvr_ids:
            missing_in_tsvr = theta_ids - tsvr_ids
            missing_in_theta = tsvr_ids - theta_ids
            msg = f"Composite ID mismatch between theta and TSVR files\n"
            if missing_in_tsvr:
                msg += f"  In theta but not TSVR: {len(missing_in_tsvr)} IDs\n"
            if missing_in_theta:
                msg += f"  In TSVR but not theta: {len(missing_in_theta)} IDs"
            raise ValueError(msg)
        else:
            log(f"  Composite IDs aligned: {len(theta_ids)} unique IDs")

        log("All validation checks passed")
        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
