#!/usr/bin/env python3
"""extract_merge_data: Load theta scores from RQ 5.1.1 (All composite factor), merge with TSVR time"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import check_file_exists, validate_data_format, check_missing_data

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_merge_data.log"

# Input file paths (cross-RQ dependencies from RQ 5.1.1)
THETA_FILE = PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_all.csv"
TSVR_FILE = PROJECT_ROOT / "results/ch5/5.1.1/data/step00_tsvr_mapping.csv"
AGE_FILE = PROJECT_ROOT / "data/cache/dfData.csv"

# Output file path
OUTPUT_FILE = RQ_DIR / "data" / "step00_lmm_input_raw.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: extract_merge_data")
        # VALIDATION 1: Check Cross-RQ Dependencies Exist
        # Before attempting data load, verify all source files exist
        # RQ 5.1.3 depends on RQ 5.7 outputs - user must execute RQ 5.7 first

        log("Checking cross-RQ dependencies...")

        # Check theta file from RQ 5.7
        theta_check = check_file_exists(THETA_FILE, min_size_bytes=1000)
        if not theta_check['valid']:
            error_msg = (
                f"Cross-RQ dependency error: {theta_check['message']}\n"
                f"File: {THETA_FILE}\n"
                f"Required: RQ 5.7 must complete successfully before running RQ 5.1.3\n"
                f"Action: Execute RQ 5.7 first to generate theta scores"
            )
            log(f"{error_msg}")
            raise FileNotFoundError(error_msg)
        log(f"Theta file exists: {THETA_FILE} ({theta_check['size_bytes']} bytes)")

        # Check TSVR file from RQ 5.7
        tsvr_check = check_file_exists(TSVR_FILE, min_size_bytes=1000)
        if not tsvr_check['valid']:
            error_msg = (
                f"Cross-RQ dependency error: {tsvr_check['message']}\n"
                f"File: {TSVR_FILE}\n"
                f"Required: RQ 5.7 Step 0 must complete to generate TSVR mapping\n"
                f"Action: Execute RQ 5.7 first"
            )
            log(f"{error_msg}")
            raise FileNotFoundError(error_msg)
        log(f"TSVR file exists: {TSVR_FILE} ({tsvr_check['size_bytes']} bytes)")

        # Check dfData file (participant demographics)
        age_check = check_file_exists(AGE_FILE, min_size_bytes=1000)
        if not age_check['valid']:
            error_msg = (
                f"Data dependency error: {age_check['message']}\n"
                f"File: {AGE_FILE}\n"
                f"Required: dfData.csv must exist in data/cache/\n"
                f"Action: Run data preparation pipeline to generate dfData.csv"
            )
            log(f"{error_msg}")
            raise FileNotFoundError(error_msg)
        log(f"Age file exists: {AGE_FILE} ({age_check['size_bytes']} bytes)")

        log("All 3 source files exist and are not empty")
        # Load Theta + TSVR from RQ 5.7

        log("Loading theta scores + TSVR from RQ 5.7 step04...")
        lmm_df = pd.read_csv(TSVR_FILE, encoding='utf-8')
        log(f"LMM data: {len(lmm_df)} rows, {len(lmm_df.columns)} columns")
        log(f"Columns: {lmm_df.columns.tolist()}")

        # Extract UID and TEST from composite_ID, then select needed columns
        log("Extracting UID and TEST from composite_ID...")
        lmm_df[['UID', 'TEST']] = lmm_df['composite_ID'].str.split('_', expand=True)

        # Select and rename columns
        lmm_df = lmm_df[['composite_ID', 'UID', 'TEST', 'Theta', 'SE', 'TSVR_hours']].copy()
        lmm_df = lmm_df.rename(columns={'Theta': 'theta_all', 'SE': 'se_all', 'TSVR_hours': 'TSVR'})

        log(f"UID range: {lmm_df['UID'].min()} - {lmm_df['UID'].max()}")
        log(f"TEST values: {sorted(lmm_df['TEST'].unique())}")
        log(f"TSVR range: {lmm_df['TSVR'].min():.2f} - {lmm_df['TSVR'].max():.2f} hours")
        log(f"Theta range: {lmm_df['theta_all'].min():.2f} - {lmm_df['theta_all'].max():.2f}")

        # Rename for consistency with merge logic below
        merged_df = lmm_df
        # Load Age from dfData

        log("Loading Age from dfData.csv...")
        age_df = pd.read_csv(AGE_FILE, encoding='utf-8')
        log(f"Demographics: {len(age_df)} rows, {len(age_df.columns)} columns")
        log(f"Demographics columns: {age_df.columns.tolist()}")

        # Select only UID and age columns, then get unique UID (age is constant per participant)
        age_df = age_df[['UID', 'age']].drop_duplicates(subset='UID').copy()
        log(f"Unique participants with age: {len(age_df)}")
        log(f"Age range: {age_df['age'].min():.1f} - {age_df['age'].max():.1f} years")
        # Merge with Age on UID
        # Merge type: Left join (keep all theta+TSVR observations, add age)

        log("Merging combined data with Age on UID...")
        merged_df = merged_df.merge(
            age_df,
            on='UID',
            how='left'
        )
        log(f"Final data: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        # Validate: Check for any missing Age values (critical error)
        age_missing = merged_df['age'].isna().sum()
        if age_missing > 0:
            missing_uids = merged_df[merged_df['age'].isna()]['UID'].unique()
            error_msg = (
                f"Age missing for {age_missing} rows across {len(missing_uids)} UIDs\n"
                f"Missing UIDs: {missing_uids}\n"
                f"All participants must have Age data in dfData.csv\n"
                f"Check dfData.csv completeness"
            )
            log(f"{error_msg}")
            raise ValueError(error_msg)
        log("No missing Age values - all participants have demographics")
        # Rename Columns for Analysis
        # theta_all -> theta (outcome variable)
        # TSVR -> TSVR_hours (explicit units for clarity)

        log("Renaming columns for analysis clarity...")
        merged_df = merged_df.rename(columns={
            'theta_all': 'theta',
            'TSVR': 'TSVR_hours'
        })
        log("Column renames applied: theta_all -> theta, TSVR -> TSVR_hours")
        # Select Final Columns and Reorder
        # Order: Identifiers, predictors, outcome, uncertainty

        log("Selecting final columns...")
        final_columns = ['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'theta', 'se_all', 'age']
        merged_df = merged_df[final_columns].copy()
        log(f"Final columns: {merged_df.columns.tolist()}")
        log(f"Final dimensions: {len(merged_df)} rows × {len(merged_df.columns)} columns")
        # Save Merged Data
        # These outputs will be used by: Step 01 (prepare predictors) and Step 02 (fit LMM)

        log(f"Saving merged data to {OUTPUT_FILE}...")
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE.name} ({len(merged_df)} rows, {len(merged_df.columns)} columns)")

        # Summary statistics for validation
        log("Dataset overview:")
        log(f"  N participants: {merged_df['UID'].nunique()}")
        log(f"  N tests per participant: {merged_df.groupby('UID')['TEST'].nunique().mode().values[0]}")
        log(f"  Total observations: {len(merged_df)}")
        log(f"  Age range: {merged_df['age'].min():.1f} - {merged_df['age'].max():.1f} years")
        log(f"  TSVR range: {merged_df['TSVR_hours'].min():.2f} - {merged_df['TSVR_hours'].max():.2f} hours")
        log(f"  Theta range: {merged_df['theta'].min():.3f} - {merged_df['theta'].max():.3f}")
        # VALIDATION 2: Check Output Data Format
        # Validates: All 7 required columns present in merged output

        log("Validating merged data format...")
        format_result = validate_data_format(
            df=merged_df,
            required_cols=['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'theta', 'se_all', 'age']
        )

        if not format_result['valid']:
            error_msg = f"Data format validation failed: {format_result['message']}"
            log(f"{error_msg}")
            raise ValueError(error_msg)
        log(f"{format_result['message']}")
        # VALIDATION 3: Check for Missing Data
        # Validates: Zero NaN values tolerated (all columns complete)

        log("Checking for missing data...")
        missing_result = check_missing_data(df=merged_df)

        if missing_result['has_missing']:
            missing_cols = {col: count for col, count in missing_result['missing_by_column'].items() if count > 0}
            error_msg = (
                f"Missing data detected: {missing_result['total_missing']} NaN values "
                f"({missing_result['percent_missing']:.2f}% of all cells)\n"
                f"Missing by column: {missing_cols}\n"
                f"RQ 5.1.3 requires complete cases (zero NaN tolerance)\n"
                f"Check merge logic and source data completeness"
            )
            log(f"{error_msg}")
            raise ValueError(error_msg)
        log(f"No missing data - all {missing_result['total_cells']} cells complete")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
