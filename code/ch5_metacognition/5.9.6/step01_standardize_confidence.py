#!/usr/bin/env python3
"""Load and Standardize Confidence Trajectories: Load confidence theta scores from Ch6 6.1.1, merge with TSVR time mapping,"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.6
LOG_FILE = RQ_DIR / "logs" / "step01_standardize_confidence.log"

# Input paths (from Ch6 6.1.1)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
THETA_PATH = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv"
TSVR_PATH = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step00_tsvr_mapping.csv"

# Output paths
OUTPUT_STANDARDIZED = RQ_DIR / "data" / "step01_confidence_standardized.csv"
OUTPUT_DESCRIPTIVES = RQ_DIR / "data" / "step01_confidence_descriptives.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: Load and Standardize Confidence Trajectories")
        log("=" * 80)
        # Load Input Data
        log("\nLoading confidence theta scores...")
        df_theta = pd.read_csv(THETA_PATH, encoding='utf-8')
        log(f"{THETA_PATH.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"         Columns: {list(df_theta.columns)}")

        log("\nLoading TSVR time mapping...")
        df_tsvr = pd.read_csv(TSVR_PATH, encoding='utf-8')
        log(f"{TSVR_PATH.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"         Columns: {list(df_tsvr.columns)}")
        # Merge Datasets
        log("\nMerging theta scores with time mapping on composite_ID...")
        df_merged = pd.merge(df_theta, df_tsvr, on='composite_ID', how='inner')
        log(f"{len(df_merged)} rows retained after inner join")

        # Verify merge didn't lose data
        if len(df_merged) != len(df_theta):
            log(f"Merge lost {len(df_theta) - len(df_merged)} rows")
            log(f"          Expected {len(df_theta)} rows, got {len(df_merged)}")

        if len(df_merged) < 300 or len(df_merged) > 500:
            raise ValueError(f"Unexpected row count after merge: {len(df_merged)} (expected ~400)")
        # Extract UID and Verify Participant Counts
        log("\nExtracting UID from composite_ID...")
        df_merged['UID'] = df_merged['composite_ID'].str.split('_').str[0]

        n_uids = df_merged['UID'].nunique()
        log(f"{n_uids} unique participants identified")

        if n_uids < 90 or n_uids > 110:
            log(f"Expected ~100 participants, found {n_uids}")

        # Verify 4 tests per participant
        tests_per_uid = df_merged.groupby('UID').size()
        n_tests_min = tests_per_uid.min()
        n_tests_max = tests_per_uid.max()
        n_tests_mode = tests_per_uid.mode().values[0] if len(tests_per_uid.mode()) > 0 else None

        log(f"Tests per participant: min={n_tests_min}, max={n_tests_max}, mode={n_tests_mode}")

        if n_tests_mode != 4:
            log(f"Expected 4 tests per participant, mode is {n_tests_mode}")

        # Identify unique test labels
        unique_tests = sorted(df_merged['test'].unique())
        log(f"Test labels: {unique_tests}")
        # Compute Z-Score Standardization
        log("\nComputing z-scores for theta_All...")

        # Extract baseline statistics BEFORE standardization
        mean_theta = df_merged['theta_All'].mean()
        sd_theta = df_merged['theta_All'].std()

        log(f"Raw theta_All statistics:")
        log(f"           Mean = {mean_theta:.4f}")
        log(f"           SD = {sd_theta:.4f}")
        log(f"           Range = [{df_merged['theta_All'].min():.4f}, {df_merged['theta_All'].max():.4f}]")

        # Z-score transformation: z = (x - mean) / sd
        df_merged['z_theta'] = (df_merged['theta_All'] - mean_theta) / sd_theta

        # Verify z-score properties
        mean_z = df_merged['z_theta'].mean()
        sd_z = df_merged['z_theta'].std()

        log(f"Z-score statistics:")
        log(f"               Mean = {mean_z:.6f} (expected ~0.0)")
        log(f"               SD = {sd_z:.6f} (expected ~1.0)")
        log(f"               Range = [{df_merged['z_theta'].min():.4f}, {df_merged['z_theta'].max():.4f}]")

        # Z-score validation (allow small numerical precision errors)
        if abs(mean_z) > 0.01:
            log(f"Z-score mean deviates from 0: {mean_z:.6f}")

        if abs(sd_z - 1.0) > 0.01:
            log(f"Z-score SD deviates from 1.0: {sd_z:.6f}")
        # Sort and Select Columns
        log("\nSorting by UID and test...")
        df_merged = df_merged.sort_values(['UID', 'test']).reset_index(drop=True)

        # Select output columns in specified order
        output_cols = ['UID', 'test', 'composite_ID', 'TSVR_hours', 'theta_All', 'se_All', 'z_theta']
        df_output = df_merged[output_cols].copy()

        log(f"Output columns: {output_cols}")
        log(f"            Output shape: {df_output.shape}")
        # Save Standardized Data
        log(f"\nSaving standardized confidence trajectories...")
        df_output.to_csv(OUTPUT_STANDARDIZED, index=False, encoding='utf-8')
        log(f"{OUTPUT_STANDARDIZED}")
        log(f"        {len(df_output)} rows x {len(df_output.columns)} columns")
        # Save Descriptive Statistics
        log(f"\nSaving descriptive statistics...")

        df_descriptives = pd.DataFrame({
            'mean_theta': [mean_theta],
            'sd_theta': [sd_theta],
            'mean_z': [mean_z],
            'sd_z': [sd_z]
        })

        df_descriptives.to_csv(OUTPUT_DESCRIPTIVES, index=False, encoding='utf-8')
        log(f"{OUTPUT_DESCRIPTIVES}")
        log(f"        {len(df_descriptives)} row x {len(df_descriptives.columns)} columns")
        # Run Validation
        log("\nRunning validate_data_columns...")

        # Validate standardized file columns
        required_columns = ['UID', 'test', 'composite_ID', 'TSVR_hours', 'theta_All', 'se_All', 'z_theta']
        validation_result = validate_data_columns(df_output, required_columns)

        if validation_result['valid']:
            log(f"All required columns present:")
            for col in validation_result['existing_columns']:
                log(f"       - {col}")
        else:
            log(f"Missing columns: {validation_result['missing_columns']}")
            raise ValueError(f"Validation failed: missing columns {validation_result['missing_columns']}")

        # Additional data quality checks
        log("\nData quality checks...")

        # Check for missing values in critical columns
        critical_cols = ['UID', 'test', 'theta_All', 'z_theta']
        for col in critical_cols:
            n_missing = df_output[col].isna().sum()
            if n_missing > 0:
                log(f"{col}: {n_missing} missing values (0 expected)")
                raise ValueError(f"Missing values in critical column {col}")
            else:
                log(f"{col}: no missing values")

        # Check se_All (allow <5% missing)
        n_missing_se = df_output['se_All'].isna().sum()
        pct_missing_se = (n_missing_se / len(df_output)) * 100
        if pct_missing_se < 5:
            log(f"se_All: {n_missing_se} missing ({pct_missing_se:.1f}%, <5% threshold)")
        else:
            log(f"se_All: {n_missing_se} missing ({pct_missing_se:.1f}%, exceeds 5% threshold)")
        # SUMMARY
        log("\n" + "=" * 80)
        log("Step 1 complete")
        log(f"  Loaded: {len(df_theta)} theta scores, {len(df_tsvr)} TSVR mappings")
        log(f"  Merged: {len(df_merged)} rows (inner join on composite_ID)")
        log(f"  Participants: {n_uids} unique UIDs")
        log(f"  Tests per participant: {n_tests_mode} (mode)")
        log(f"  Z-score validation: mean={mean_z:.6f}, sd={sd_z:.6f}")
        log(f"  Output: {OUTPUT_STANDARDIZED} ({len(df_output)} rows)")
        log(f"  Descriptives: {OUTPUT_DESCRIPTIVES}")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
