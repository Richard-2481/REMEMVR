#!/usr/bin/env python3
"""Merge Theta with TSVR: Merge IRT theta scores (confidence ability estimates) with TSVR time variable"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Merge Theta with TSVR")
        # Load Input Data
        #           TSVR mapping from data extraction (400 rows)

        log("Loading theta scores from step03...")
        df_theta = pd.read_csv(RQ_DIR / "data" / "step03_theta_confidence.csv")
        log(f"step03_theta_confidence.csv ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"Theta columns: {list(df_theta.columns)}")
        log(f"Theta composite_ID format example: {df_theta['composite_ID'].iloc[0]}")

        log("Loading TSVR mapping from step00...")
        df_tsvr = pd.read_csv(RQ_DIR / "data" / "step00_tsvr_mapping.csv")
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"TSVR columns: {list(df_tsvr.columns)}")
        log(f"TSVR composite_ID format example: {df_tsvr['composite_ID'].iloc[0]}")
        # Standardize composite_ID Formats
        # Problem: Theta uses UID_TEST (e.g., A010_T1), TSVR uses UID_test (e.g., A010_1)
        # Solution: Standardize TSVR to match Theta format (UID_TEST)
        # Why: Enables direct merge without data loss

        log("Standardizing composite_ID formats...")

        # Convert TSVR test column (1,2,3,4) to TEST format (T1,T2,T3,T4)
        df_tsvr['test_str'] = 'T' + df_tsvr['test'].astype(str)

        # Extract UID from TSVR composite_ID (everything before last underscore)
        df_tsvr['UID'] = df_tsvr['composite_ID'].str.rsplit('_', n=1).str[0]

        # Create standardized composite_ID in Theta format (UID_TEST)
        df_tsvr['composite_ID_standardized'] = df_tsvr['UID'] + '_' + df_tsvr['test_str']

        log(f"TSVR standardized composite_ID example: {df_tsvr['composite_ID_standardized'].iloc[0]}")

        # Use standardized composite_ID for merge
        df_tsvr_merge = df_tsvr[['composite_ID_standardized', 'TSVR_hours', 'test_str']].rename(
            columns={'composite_ID_standardized': 'composite_ID', 'test_str': 'test'}
        )
        # Merge Theta with TSVR
        # Merge type: Inner join (expect all rows to match)

        log("Merging theta scores with TSVR mapping...")
        df_merged = pd.merge(
            df_theta,
            df_tsvr_merge,
            on='composite_ID',
            how='inner',
            validate='one_to_one'  # Ensures no duplicate composite_IDs
        )
        log(f"Result: {len(df_merged)} rows, {len(df_merged.columns)} cols")

        # Check for data loss
        if len(df_merged) != len(df_theta):
            log(f"Data loss detected: {len(df_theta)} theta rows -> {len(df_merged)} merged rows")
            missing_theta = set(df_theta['composite_ID']) - set(df_merged['composite_ID'])
            log(f"Missing composite_IDs: {missing_theta}")
        else:
            log("No data loss from merge (all theta rows matched)")
        # Extract UID and Create Time Transformations
        # UID: Extracted from composite_ID (before first underscore)
        # Time transformations:
        #   - Days = TSVR_hours / 24 (continuous time in days)
        #   - Days_squared = Days^2 (for quadratic trajectory models)
        #   - log_Days_plus1 = log(Days + 1) (for logarithmic models, +1 handles Day 0)

        log("Extracting UID and creating time transformations...")

        # Extract UID (everything before first underscore)
        df_merged['UID'] = df_merged['composite_ID'].str.split('_').str[0]

        # Create time transformations
        df_merged['Days'] = df_merged['TSVR_hours'] / 24.0
        df_merged['Days_squared'] = df_merged['Days'] ** 2
        df_merged['log_Days_plus1'] = np.log(df_merged['Days'] + 1.0)

        log(f"UID count: {df_merged['UID'].nunique()} unique participants")
        log(f"TSVR_hours range: [{df_merged['TSVR_hours'].min():.2f}, {df_merged['TSVR_hours'].max():.2f}]")
        log(f"Days range: [{df_merged['Days'].min():.2f}, {df_merged['Days'].max():.2f}]")
        log(f"Days_squared range: [{df_merged['Days_squared'].min():.2f}, {df_merged['Days_squared'].max():.2f}]")
        log(f"log_Days_plus1 range: [{df_merged['log_Days_plus1'].min():.2f}, {df_merged['log_Days_plus1'].max():.2f}]")
        # Reorder Columns and Save Output
        # Output columns (as specified in 4_analysis.yaml):
        #   composite_ID, UID, test, theta_All, se_All,
        #   TSVR_hours, Days, Days_squared, log_Days_plus1

        log("Saving LMM input data...")

        # Select and reorder columns
        output_cols = [
            'composite_ID', 'UID', 'test', 'theta_All', 'se_All',
            'TSVR_hours', 'Days', 'Days_squared', 'log_Days_plus1'
        ]
        df_output = df_merged[output_cols]

        # Save to CSV
        output_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Validation
        # Validation criteria from 4_analysis.yaml:
        #   - 400 rows (no data loss)
        #   - No NaN in any column
        #   - TSVR_hours in [0, 250]
        #   - Days in [0, 11]
        #   - 100 unique UIDs

        log("Running data quality checks...")

        validation_passed = True

        # Check 1: Row count
        if len(df_output) != 400:
            log(f"Expected 400 rows, got {len(df_output)}")
            validation_passed = False
        else:
            log("Row count: 400 (expected)")

        # Check 2: No NaN values
        nan_counts = df_output.isnull().sum()
        if nan_counts.sum() > 0:
            log(f"NaN values detected:")
            for col, count in nan_counts[nan_counts > 0].items():
                log(f"  - {col}: {count} NaN values")
            validation_passed = False
        else:
            log("No NaN values in any column")

        # Check 3: TSVR_hours range
        tsvr_min = df_output['TSVR_hours'].min()
        tsvr_max = df_output['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 250:
            log(f"TSVR_hours out of expected range [0, 250]: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
            validation_passed = False
        else:
            log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] (within [0, 250])")

        # Check 4: Days range
        days_min = df_output['Days'].min()
        days_max = df_output['Days'].max()
        if days_min < 0 or days_max > 11:
            log(f"Days out of expected range [0, 11]: [{days_min:.2f}, {days_max:.2f}]")
            validation_passed = False
        else:
            log(f"Days range: [{days_min:.2f}, {days_max:.2f}] (within [0, 11])")

        # Check 5: UID count
        n_uids = df_output['UID'].nunique()
        if n_uids != 100:
            log(f"Expected 100 unique UIDs, got {n_uids}")
            validation_passed = False
        else:
            log(f"UID count: {n_uids} unique participants (expected)")

        # Check 6: Column count
        if len(df_output.columns) != 9:
            log(f"Expected 9 columns, got {len(df_output.columns)}")
            validation_passed = False
        else:
            log("Column count: 9 (expected)")

        # Final validation result
        if validation_passed:
            log("All validation checks passed")
            log("Step 04 complete")
            sys.exit(0)
        else:
            log("Validation failed - see above for details")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
