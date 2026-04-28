#!/usr/bin/env python3
"""Load Data Sources: Load IRT item parameters, theta scores, TSVR mapping from RQ 5.2.1 and raw"""

import sys
from pathlib import Path
import pandas as pd
import traceback
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import check_file_exists, validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5
LOG_FILE = RQ_DIR / "logs" / "step00_load_data.log"


# Input/Output Paths

# Cross-RQ dependencies (RQ 5.2.1 outputs)
RQ51_DIR = PROJECT_ROOT / "results" / "ch5" / "5.2.1"
RQ51_STATUS = RQ51_DIR / "status.yaml"
RQ51_PURIFIED_ITEMS = RQ51_DIR / "data" / "step02_purified_items.csv"
RQ51_THETA_SCORES = RQ51_DIR / "data" / "step03_theta_scores.csv"
RQ51_TSVR_MAPPING = RQ51_DIR / "data" / "step00_tsvr_mapping.csv"

# Raw data source
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Local output paths (data/ folder)
OUTPUT_PURIFIED_ITEMS = RQ_DIR / "data" / "step00_irt_purified_items.csv"
OUTPUT_THETA_SCORES = RQ_DIR / "data" / "step00_theta_scores.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
OUTPUT_RAW_SCORES = RQ_DIR / "data" / "step00_raw_scores.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Load Data Sources")
        # Check RQ 5.1 Completion Status

        log("Verifying RQ 5.1 completion status...")

        if not RQ51_STATUS.exists():
            raise FileNotFoundError(
                f"RQ 5.1 status.yaml not found at {RQ51_STATUS}. "
                "RQ 5.1 must be completed before running RQ 5.12."
            )

        with open(RQ51_STATUS, 'r', encoding='utf-8') as f:
            status = yaml.safe_load(f)

        # Check step03_irt_calibration_pass2 completion (RQ 5.1 uses nested analysis_steps structure)
        if 'analysis_steps' not in status or 'step03_irt_calibration_pass2' not in status['analysis_steps']:
            raise ValueError(
                "RQ 5.1 status.yaml missing analysis_steps.step03_irt_calibration_pass2. "
                "RQ 5.1 must complete Step 3 before RQ 5.12 can proceed."
            )

        step03_data = status['analysis_steps']['step03_irt_calibration_pass2']
        step03_status = step03_data.get('status', 'unknown')
        if step03_status != 'success':
            raise ValueError(
                f"RQ 5.1 step03_irt_calibration_pass2 status = {step03_status}, expected 'success'. "
                "RQ 5.1 must complete Step 3 successfully before RQ 5.12 can proceed."
            )

        log(f"RQ 5.1 Step 3 completed successfully (status = {step03_status})")
        # Load RQ 5.1 Outputs

        log("Loading RQ 5.1 outputs...")

        # Load purified items
        log(f"Reading {RQ51_PURIFIED_ITEMS}...")
        df_purified_items = pd.read_csv(RQ51_PURIFIED_ITEMS, encoding='utf-8')
        log(f"Purified items (raw): {len(df_purified_items)} items, {len(df_purified_items.columns)} columns")
        log(f"Columns: {list(df_purified_items.columns)}")

        # FILTER: Exclude When domain items (floor effect in RQ 5.2.1)
        items_before = len(df_purified_items)
        df_purified_items = df_purified_items[df_purified_items['factor'] != 'when'].copy()
        items_after = len(df_purified_items)
        log(f"Excluded When domain: {items_before} -> {items_after} items ({items_before - items_after} removed)")

        # Load theta scores
        log(f"Reading {RQ51_THETA_SCORES}...")
        df_theta = pd.read_csv(RQ51_THETA_SCORES, encoding='utf-8')
        log(f"Theta scores (raw): {len(df_theta)} rows, {len(df_theta.columns)} columns")
        log(f"Columns: {list(df_theta.columns)}")

        # FILTER: Drop theta_when column (When domain excluded)
        if 'theta_when' in df_theta.columns:
            df_theta = df_theta.drop(columns=['theta_when'])
            log(f"Dropped theta_when column (When domain excluded)")
        log(f"Theta columns after filter: {list(df_theta.columns)}")

        # Load TSVR mapping
        log(f"Reading {RQ51_TSVR_MAPPING}...")
        df_tsvr = pd.read_csv(RQ51_TSVR_MAPPING, encoding='utf-8')
        log(f"TSVR mapping: {len(df_tsvr)} rows, {len(df_tsvr.columns)} columns")
        log(f"Columns: {list(df_tsvr.columns)}")
        # Load Raw Scores from dfData.csv

        log("Loading raw dichotomized scores from dfData.csv...")
        log(f"Reading {DFDATA_PATH}...")
        df_raw = pd.read_csv(DFDATA_PATH, encoding='utf-8')
        log(f"Raw scores: {len(df_raw)} rows, {len(df_raw.columns)} columns")

        # Count TQ_* columns (item responses)
        tq_columns = [col for col in df_raw.columns if col.startswith('TQ_')]
        log(f"Found {len(tq_columns)} TQ_* item columns")

        # Verify required columns present
        required_cols = ['UID', 'TEST']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"dfData.csv missing required columns: {missing_cols}")
        log(f"Required columns present: {required_cols}")
        # Create composite_ID in Raw Data

        log("Creating composite_ID in raw data...")
        log("Format: UID + '_' + TEST.astype(str) (e.g., A010_1)")

        df_raw['composite_ID'] = df_raw['UID'] + '_' + df_raw['TEST'].astype(str)
        log(f"composite_ID for {len(df_raw)} rows")
        log(f"Example composite_IDs: {df_raw['composite_ID'].head(3).tolist()}")
        # Verify Expected Columns

        log("Verifying expected columns...")

        # Validate purified items columns
        expected_purified = ['item_name', 'factor', 'a', 'b']
        result_purified = validate_data_columns(df_purified_items, expected_purified)
        if not result_purified['valid']:
            raise ValueError(
                f"Purified items column mismatch. Missing: {result_purified['missing_columns']}"
            )
        log(f"Purified items columns: {expected_purified}")

        # Validate theta scores columns (theta_when already dropped)
        expected_theta = ['composite_ID', 'theta_what', 'theta_where']
        result_theta = validate_data_columns(df_theta, expected_theta)
        if not result_theta['valid']:
            raise ValueError(
                f"Theta scores column mismatch. Missing: {result_theta['missing_columns']}"
            )
        log(f"Theta scores columns: {expected_theta}")

        # Validate TSVR mapping columns
        expected_tsvr = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        result_tsvr = validate_data_columns(df_tsvr, expected_tsvr)
        if not result_tsvr['valid']:
            raise ValueError(
                f"TSVR mapping column mismatch. Missing: {result_tsvr['missing_columns']}"
            )
        log(f"TSVR mapping columns: {expected_tsvr}")

        # Validate raw scores columns (now includes composite_ID)
        expected_raw = ['composite_ID', 'UID', 'TEST'] + tq_columns[:3]  # Check first 3 TQ_* as sample
        result_raw = validate_data_columns(df_raw, expected_raw)
        if not result_raw['valid']:
            raise ValueError(
                f"Raw scores column mismatch. Missing: {result_raw['missing_columns']}"
            )
        log(f"Raw scores columns include: composite_ID, UID, TEST, + {len(tq_columns)} TQ_* items")
        # Copy Files to Local data/ Folder

        log("Copying files to local data/ folder...")

        # Save purified items
        log(f"Writing {OUTPUT_PURIFIED_ITEMS}...")
        df_purified_items.to_csv(OUTPUT_PURIFIED_ITEMS, index=False, encoding='utf-8')
        log(f"{OUTPUT_PURIFIED_ITEMS.name} ({len(df_purified_items)} rows)")

        # Save theta scores
        log(f"Writing {OUTPUT_THETA_SCORES}...")
        df_theta.to_csv(OUTPUT_THETA_SCORES, index=False, encoding='utf-8')
        log(f"{OUTPUT_THETA_SCORES.name} ({len(df_theta)} rows)")

        # Save TSVR mapping
        log(f"Writing {OUTPUT_TSVR_MAPPING}...")
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"{OUTPUT_TSVR_MAPPING.name} ({len(df_tsvr)} rows)")

        # Save raw scores with composite_ID
        log(f"Writing {OUTPUT_RAW_SCORES}...")
        df_raw.to_csv(OUTPUT_RAW_SCORES, index=False, encoding='utf-8')
        log(f"{OUTPUT_RAW_SCORES.name} ({len(df_raw)} rows)")
        # Run Validation Tool
        # Validates: All 4 output files exist with minimum size >= 100 bytes
        # Threshold: 100 bytes ensures files are not empty (header + at least 1 data row)

        log("Running check_file_exists for all 4 output files...")

        output_files = [
            OUTPUT_PURIFIED_ITEMS,
            OUTPUT_THETA_SCORES,
            OUTPUT_TSVR_MAPPING,
            OUTPUT_RAW_SCORES
        ]

        validation_results = []
        for file_path in output_files:
            result = check_file_exists(file_path, min_size_bytes=100)
            validation_results.append(result)

            if result['valid']:
                log(f"{file_path.name}: PASS (size = {result['size_bytes']} bytes)")
            else:
                log(f"{file_path.name}: FAIL - {result['message']}")

        # Check if all validations passed
        all_valid = all(r['valid'] for r in validation_results)

        if not all_valid:
            failed_files = [f.name for f, r in zip(output_files, validation_results) if not r['valid']]
            raise FileNotFoundError(
                f"Validation failed for {len(failed_files)} files: {failed_files}. "
                "Step 0 data loading incomplete."
            )

        log(f"All {len(output_files)} output files validated successfully")

        log("Step 0 complete")
        log(f"Loaded 4 data sources from RQ 5.1 + dfData.csv")
        log(f"Created composite_ID in raw scores (UID_TEST format)")
        log(f"Saved 4 local copies to data/ folder")
        log(f"Files ready for CTT computation (Steps 1-3)")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
