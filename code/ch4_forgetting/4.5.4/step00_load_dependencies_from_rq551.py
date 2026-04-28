#!/usr/bin/env python3
"""Load Dependencies from RQ 5.5.1: Load IRT theta scores, purified items list, TSVR mapping from RQ 5.5.1 and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4
LOG_FILE = RQ_DIR / "logs" / "step00_load_dependencies_from_rq551.log"

# Dependency paths from RQ 5.5.1
RQ_551_DIR = PROJECT_ROOT / "results" / "ch5" / "5.5.1"
THETA_SCORES_PATH = RQ_551_DIR / "data" / "step03_theta_scores.csv"
PURIFIED_ITEMS_PATH = RQ_551_DIR / "data" / "step02_purified_items.csv"
TSVR_MAPPING_PATH = RQ_551_DIR / "data" / "step00_tsvr_mapping.csv"

# Project-level data
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Dependencies from RQ 5.5.1")
        # Check RQ 5.5.1 Files Exist (EXPECTATIONS ERROR if missing)

        log("Verifying RQ 5.5.1 dependency files exist...")

        missing_files = []
        for filepath in [THETA_SCORES_PATH, PURIFIED_ITEMS_PATH, TSVR_MAPPING_PATH]:
            if not filepath.exists():
                missing_files.append(str(filepath))

        if missing_files:
            log("EXPECTATIONS ERROR: RQ 5.5.1 dependency files missing")
            for missing in missing_files:
                log(f"{missing}")
            log("Run RQ 5.5.1 first to generate required dependency files")
            sys.exit(1)

        log("All RQ 5.5.1 dependency files exist")

        # Check dfData.csv exists
        if not DFDATA_PATH.exists():
            log(f"EXPECTATIONS ERROR: dfData.csv not found at {DFDATA_PATH}")
            log("Run data preparation pipeline to generate dfData.csv")
            sys.exit(1)

        log("dfData.csv exists")
        # Load RQ 5.5.1 Outputs

        log("Loading RQ 5.5.1 outputs...")

        # Load theta scores (wide format: composite_ID, theta_source, theta_destination, se_source, se_destination)
        theta_scores_wide = pd.read_csv(THETA_SCORES_PATH, encoding='utf-8')
        log(f"{THETA_SCORES_PATH.name} ({len(theta_scores_wide)} rows, {len(theta_scores_wide.columns)} cols)")

        # Load purified items (item_tag, factor, a, b, retention_reason)
        purified_items = pd.read_csv(PURIFIED_ITEMS_PATH, encoding='utf-8')
        log(f"{PURIFIED_ITEMS_PATH.name} ({len(purified_items)} rows, {len(purified_items.columns)} cols)")

        # Load TSVR mapping (composite_ID, UID, test, TSVR_hours)
        tsvr_mapping = pd.read_csv(TSVR_MAPPING_PATH, encoding='utf-8')
        log(f"{TSVR_MAPPING_PATH.name} ({len(tsvr_mapping)} rows, {len(tsvr_mapping.columns)} cols)")
        # Reshape Theta Scores to Long Format

        log("Converting theta scores from wide to long format...")

        # Split composite_ID into UID and test
        theta_scores_wide[['UID', 'test']] = theta_scores_wide['composite_ID'].str.split('_', expand=True)
        # UID is string format (e.g., "A010"), test is integer
        theta_scores_wide['test'] = theta_scores_wide['test'].astype(int)

        # Reshape to long format (source and destination as separate rows)
        theta_long_list = []

        for location_type, theta_col, se_col in [
            ('source', 'theta_source', 'se_source'),
            ('destination', 'theta_destination', 'se_destination')
        ]:
            df_subset = theta_scores_wide[['composite_ID', 'UID', 'test', theta_col, se_col]].copy()
            df_subset.rename(columns={theta_col: 'irt_theta', se_col: 'irt_se'}, inplace=True)
            df_subset['location_type'] = location_type
            theta_long_list.append(df_subset)

        theta_long = pd.concat(theta_long_list, ignore_index=True)
        log(f"Wide ({len(theta_scores_wide)} rows) -> Long ({len(theta_long)} rows)")
        # Merge TSVR_hours into Theta Long Format

        log("Adding TSVR_hours to theta long format...")

        theta_long = theta_long.merge(
            tsvr_mapping[['composite_ID', 'TSVR_hours']],
            on='composite_ID',
            how='left'
        )

        log(f"Theta long now has {len(theta_long)} rows with TSVR_hours")
        # Prepare Purified Items List

        log("Creating purified items list...")

        purified_items_list = purified_items[['item_tag', 'factor']].copy()
        purified_items_list.rename(columns={
            'item_tag': 'item_code',
            'factor': 'location_type'
        }, inplace=True)

        log(f"Purified items list: {len(purified_items_list)} items")
        # Load and Filter dfData to Purified Items

        log("Loading dfData.csv and filtering to purified items...")

        df_data_raw = pd.read_csv(DFDATA_PATH, encoding='utf-8')
        log(f"{DFDATA_PATH.name} ({len(df_data_raw)} rows, {len(df_data_raw.columns)} cols)")

        # Get list of purified item columns
        purified_item_cols = purified_items_list['item_code'].tolist()

        # Check which purified items exist in dfData
        available_items = [col for col in purified_item_cols if col in df_data_raw.columns]
        missing_items = [col for col in purified_item_cols if col not in df_data_raw.columns]

        if missing_items:
            log(f"{len(missing_items)} purified items not found in dfData.csv:")
            for item in missing_items[:5]:  # Show first 5 missing items
                log(f"  - {item}")
            if len(missing_items) > 5:
                log(f"  ... and {len(missing_items) - 5} more")

        log(f"{len(available_items)} / {len(purified_item_cols)} purified items in dfData.csv")

        # Create composite_ID in dfData (columns are uppercase: UID, TEST)
        df_data_raw['composite_ID'] = df_data_raw['UID'].astype(str) + '_' + df_data_raw['TEST'].astype(str)
        # Rename TEST to test for consistency
        df_data_raw['test'] = df_data_raw['TEST']

        # Filter to UID, test, composite_ID, and available purified items
        keep_cols = ['composite_ID', 'UID', 'test'] + available_items
        raw_responses = df_data_raw[keep_cols].copy()

        log(f"Raw responses: {len(raw_responses)} rows, {len(raw_responses.columns)} cols")
        # Validation Checks

        log("Running validation checks...")

        validation_errors = []

        # Check theta_long row count (800 expected)
        if len(theta_long) != 800:
            validation_errors.append(f"theta_long has {len(theta_long)} rows (expected 800)")
        else:
            log("theta_long has 800 rows")

        # Check raw_responses row count (400 expected)
        if len(raw_responses) != 400:
            validation_errors.append(f"raw_responses has {len(raw_responses)} rows (expected 400)")
        else:
            log("raw_responses has 400 rows")

        # Check purified_items count in [25, 32] range
        n_purified = len(purified_items_list)
        if not (25 <= n_purified <= 32):
            validation_errors.append(f"purified_items has {n_purified} items (expected 25-32)")
        else:
            log(f"purified_items has {n_purified} items (within 25-32 range)")

        # Check irt_theta range [-3, 3]
        theta_min = theta_long['irt_theta'].min()
        theta_max = theta_long['irt_theta'].max()
        if theta_min < -3 or theta_max > 3:
            validation_errors.append(f"irt_theta out of range [{theta_min:.2f}, {theta_max:.2f}] (expected [-3, 3])")
        else:
            log(f"irt_theta in range [{theta_min:.2f}, {theta_max:.2f}]")

        # Check irt_se range [0.1, 1.5]
        se_min = theta_long['irt_se'].min()
        se_max = theta_long['irt_se'].max()
        if se_min < 0.1 or se_max > 1.5:
            validation_errors.append(f"irt_se out of range [{se_min:.2f}, {se_max:.2f}] (expected [0.1, 1.5])")
        else:
            log(f"irt_se in range [{se_min:.2f}, {se_max:.2f}]")

        # Check TSVR_hours range [0, 360] - extended to 360h to account for participants tested beyond 1 week
        # This matches the TSVR_hours range extension applied in RQ 5.5.2
        tsvr_min = theta_long['TSVR_hours'].min()
        tsvr_max = theta_long['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 360:
            validation_errors.append(f"TSVR_hours out of range [{tsvr_min:.2f}, {tsvr_max:.2f}] (expected [0, 360])")
        else:
            log(f"TSVR_hours in range [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        # Check for NaN values
        nan_cols = []
        for col in ['irt_theta', 'irt_se', 'TSVR_hours']:
            if theta_long[col].isna().any():
                nan_count = theta_long[col].isna().sum()
                nan_cols.append(f"{col} ({nan_count} NaN)")

        if nan_cols:
            validation_errors.append(f"NaN values found: {', '.join(nan_cols)}")
        else:
            log("No NaN values in irt_theta, irt_se, TSVR_hours")

        # Check location_type values
        location_types = set(theta_long['location_type'].unique())
        expected_types = {'source', 'destination'}
        if location_types != expected_types:
            validation_errors.append(f"location_type has {location_types} (expected {expected_types})")
        else:
            log("location_type in {'source', 'destination'}")

        # Check for duplicate composite_IDs
        duplicates = theta_long[theta_long.duplicated(subset=['composite_ID', 'location_type'], keep=False)]
        if len(duplicates) > 0:
            validation_errors.append(f"Found {len(duplicates)} duplicate composite_ID x location_type combinations")
        else:
            log("No duplicate composite_IDs in theta_long")

        # Report validation results
        if validation_errors:
            log("Validation errors found:")
            for error in validation_errors:
                log(f"  - {error}")
            raise ValueError("Validation failed - see log for details")

        log("All validation checks passed")
        # Save Outputs

        log("Saving outputs...")

        # Save theta_long (800 rows)
        output_path_theta = RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv"
        theta_long.to_csv(output_path_theta, index=False, encoding='utf-8')
        log(f"{output_path_theta.name} ({len(theta_long)} rows, {len(theta_long.columns)} cols)")

        # Save purified_items_list (25-32 rows)
        output_path_items = RQ_DIR / "data" / "step00_purified_items_from_rq551.csv"
        purified_items_list.to_csv(output_path_items, index=False, encoding='utf-8')
        log(f"{output_path_items.name} ({len(purified_items_list)} rows, {len(purified_items_list.columns)} cols)")

        # Save raw_responses (400 rows)
        output_path_responses = RQ_DIR / "data" / "step00_raw_responses_filtered.csv"
        raw_responses.to_csv(output_path_responses, index=False, encoding='utf-8')
        log(f"{output_path_responses.name} ({len(raw_responses)} rows, {len(raw_responses.columns)} cols)")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
