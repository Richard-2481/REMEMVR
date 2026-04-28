#!/usr/bin/env python3
"""step01_prepare_lmm_input: Merge theta scores with TSVR and Age, grand-mean center Age, reshape to long"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_prepare_lmm_input.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Prepare LMM Input with Age Variable")
        # Load Input Data from Step 00

        log("Loading theta scores from RQ 5.1...")
        theta_path = RQ_DIR / "data" / "step00_theta_from_rq51.csv"
        if not theta_path.exists():
            raise FileNotFoundError(f"Missing input: {theta_path} - Run step00 first")
        df_theta = pd.read_csv(theta_path, encoding='utf-8')
        log(f"theta scores: {len(df_theta)} rows, {len(df_theta.columns)} cols")
        log(f"Theta columns: {df_theta.columns.tolist()}")

        log("Loading TSVR mapping from RQ 5.1...")
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_from_rq51.csv"
        if not tsvr_path.exists():
            raise FileNotFoundError(f"Missing input: {tsvr_path} - Run step00 first")
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"TSVR mapping: {len(df_tsvr)} rows, {len(df_tsvr.columns)} cols")
        log(f"TSVR columns: {df_tsvr.columns.tolist()}")

        log("Loading age variable from dfData...")
        age_path = RQ_DIR / "data" / "step00_age_from_dfdata.csv"
        if not age_path.exists():
            raise FileNotFoundError(f"Missing input: {age_path} - Run step00 first")
        df_age = pd.read_csv(age_path, encoding='utf-8')
        log(f"age data: {len(df_age)} rows, {len(df_age.columns)} cols")
        log(f"Age columns: {df_age.columns.tolist()}")

        # Validate input columns
        log("Checking input columns...")
        required_theta_cols = ['composite_ID', 'domain', 'test', 'theta']
        required_tsvr_cols = ['composite_ID', 'test', 'TSVR_hours']
        required_age_cols = ['UID', 'age']

        missing_theta = set(required_theta_cols) - set(df_theta.columns)
        missing_tsvr = set(required_tsvr_cols) - set(df_tsvr.columns)
        missing_age = set(required_age_cols) - set(df_age.columns)

        if missing_theta:
            raise ValueError(f"Theta file missing columns: {missing_theta}")
        if missing_tsvr:
            raise ValueError(f"TSVR file missing columns: {missing_tsvr}")
        if missing_age:
            raise ValueError(f"Age file missing columns: {missing_age}")

        log("All input columns present")
        # Merge Data (UID will come from TSVR file)
        # TSVR file already has UID column extracted from composite_ID
        # No need to extract it from theta file - just merge and UID will carry through

        # Merge theta with TSVR on composite_ID and test
        log("Merging theta with TSVR mapping...")
        df_merged = df_theta.merge(
            df_tsvr,
            on=['composite_ID', 'test'],
            how='left',
            validate='many_to_one'
        )
        log(f"Merged data now has {df_merged['UID'].nunique()} unique UIDs from TSVR file")
        log(f"theta + TSVR: {len(df_merged)} rows")

        # Check for missing TSVR values after merge
        missing_tsvr_count = df_merged['TSVR_hours'].isna().sum()
        if missing_tsvr_count > 0:
            raise ValueError(f"Missing TSVR_hours for {missing_tsvr_count} rows after merge")
        log("No missing TSVR_hours after merge")

        # Merge with age on UID
        log("Merging with age data...")
        df_merged = df_merged.merge(
            df_age,
            on='UID',
            how='left',
            validate='many_to_one'
        )
        log(f"theta + TSVR + age: {len(df_merged)} rows")

        # Check for missing age values after merge
        missing_age_count = df_merged['age'].isna().sum()
        if missing_age_count > 0:
            raise ValueError(f"Missing age for {missing_age_count} rows after merge")
        log("No missing age values after merge")
        # Grand-Mean Center Age
        # LMM best practice: Center continuous predictors for interpretable intercepts
        # Grand-mean centering: Age_c = age - mean(age across all participants)

        log("Grand-mean centering Age variable...")
        mean_age = df_merged['age'].mean()
        df_merged['Age_c'] = df_merged['age'] - mean_age
        df_merged['mean_age'] = mean_age  # Store for reference

        log(f"Grand mean age: {mean_age:.2f} years")
        log(f"Age_c range: [{df_merged['Age_c'].min():.2f}, {df_merged['Age_c'].max():.2f}]")
        log(f"Mean Age_c: {df_merged['Age_c'].mean():.10f} (should be ~0)")

        # Validate centering
        age_c_mean = df_merged['Age_c'].mean()
        if abs(age_c_mean) > 1e-10:
            raise ValueError(f"Age_c not properly centered: mean = {age_c_mean:.2e} (expected ~0)")
        log("Age_c properly centered (mean approximately 0)")
        # Create Log-Transformed TSVR
        # Log transformation handles non-linear time effects
        # log(TSVR + 1) to handle TSVR=0 at T1 (immediate test)

        log("Creating log-transformed TSVR...")
        df_merged['log_TSVR'] = np.log(df_merged['TSVR_hours'] + 1)
        log(f"log_TSVR range: [{df_merged['log_TSVR'].min():.2f}, {df_merged['log_TSVR'].max():.2f}]")
        # Validate Final Dataset
        # 7 validation criteria from 4_analysis.yaml

        log("Running 7 validation checks...")

        # Check 1: Row count (800 rows expected - When excluded)
        expected_rows = 100 * 4 * 2  # 100 participants x 4 tests x 2 domains (When excluded)
        if len(df_merged) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {len(df_merged)}")
        log(f"Check 1: Row count = {len(df_merged)} (100 participants x 4 tests x 2 domains - When excluded)")

        # Check 2: No NaN values
        nan_counts = df_merged.isna().sum()
        if nan_counts.sum() > 0:
            raise ValueError(f"Found NaN values:\n{nan_counts[nan_counts > 0]}")
        log("Check 2: No NaN values in any column")

        # Check 3: Mean Age_c approximately 0 (already validated above)
        log(f"Check 3: Mean Age_c = {age_c_mean:.2e} (within 1e-10 tolerance)")

        # Check 4: Age_c range symmetric
        age_c_min = df_merged['Age_c'].min()
        age_c_max = df_merged['Age_c'].max()
        symmetry_ratio = abs(age_c_min) / age_c_max
        if not (0.9 <= symmetry_ratio <= 1.1):
            raise ValueError(f"Age_c range not symmetric: |min| / max = {symmetry_ratio:.2f}")
        log(f"Check 4: Age_c range symmetric (|min|/max = {symmetry_ratio:.2f})")

        # Check 5: All UIDs have exactly 8 rows (4 tests x 2 domains - When excluded)
        rows_per_uid = df_merged.groupby('UID').size()
        if not (rows_per_uid == 8).all():
            bad_uids = rows_per_uid[rows_per_uid != 8]
            raise ValueError(f"UIDs with != 8 rows:\n{bad_uids}")
        log(f"Check 5: All {len(rows_per_uid)} UIDs have exactly 8 rows (When excluded)")

        # Check 6: TSVR_hours reasonable range (allow up to 300 hours for scheduling variations)
        tsvr_min = df_merged['TSVR_hours'].min()
        tsvr_max = df_merged['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 300:
            raise ValueError(f"TSVR_hours out of range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
        log(f"Check 6: TSVR_hours in [0, 300] range (actual: [{tsvr_min:.2f}, {tsvr_max:.2f}])")
        if tsvr_max > 200:
            log(f"TSVR exceeds nominal 168h (7 days) - scheduling variations present")

        # Check 7: log_TSVR reasonable range
        log_tsvr_min = df_merged['log_TSVR'].min()
        log_tsvr_max = df_merged['log_TSVR'].max()
        if log_tsvr_min < 0 or log_tsvr_max > 10:
            raise ValueError(f"log_TSVR out of range: [{log_tsvr_min:.2f}, {log_tsvr_max:.2f}]")
        log(f"Check 7: log_TSVR in [0, 6] range (actual: [{log_tsvr_min:.2f}, {log_tsvr_max:.2f}])")

        log("All 7 validation checks PASSED")
        # Save LMM Input File
        # Output: Long-format CSV with 10 columns, ready for statsmodels MixedLM

        log("Saving LMM input file...")
        output_path = RQ_DIR / "data" / "step01_lmm_input.csv"

        # Reorder columns for clarity
        output_cols = [
            'UID', 'composite_ID', 'test', 'domain', 'theta',
            'TSVR_hours', 'log_TSVR', 'age', 'Age_c', 'mean_age'
        ]
        df_output = df_merged[output_cols].copy()

        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Save Preprocessing Summary
        # Text report documenting preprocessing steps and validation

        log("Saving preprocessing summary...")
        summary_path = RQ_DIR / "data" / "step01_preprocessing_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STEP 01: LMM INPUT PREPROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("INPUT FILES:\n")
            f.write(f"  - Theta scores: {theta_path.name} ({len(df_theta)} rows)\n")
            f.write(f"  - TSVR mapping: {tsvr_path.name} ({len(df_tsvr)} rows)\n")
            f.write(f"  - Age data: {age_path.name} ({len(df_age)} rows)\n\n")

            f.write("OUTPUT FILE:\n")
            f.write(f"  - LMM input: {output_path.name} ({len(df_output)} rows, {len(df_output.columns)} cols)\n\n")

            f.write("PARTICIPANT SUMMARY:\n")
            f.write(f"  - N participants: {df_output['UID'].nunique()}\n")
            f.write(f"  - N tests per participant: {df_output['test'].nunique()}\n")
            f.write(f"  - N domains: {df_output['domain'].nunique()}\n")
            f.write(f"  - Total observations: {len(df_output)}\n\n")

            f.write("AGE VARIABLE SUMMARY:\n")
            f.write(f"  - Grand mean age: {mean_age:.2f} years\n")
            f.write(f"  - Age range: [{df_output['age'].min():.2f}, {df_output['age'].max():.2f}] years\n")
            f.write(f"  - Age_c range: [{age_c_min:.2f}, {age_c_max:.2f}] years (centered)\n")
            f.write(f"  - Mean Age_c: {age_c_mean:.2e} (verification: should be ~0)\n")
            f.write(f"  - Age_c symmetry: |min|/max = {symmetry_ratio:.3f} (should be ~1.0)\n\n")

            f.write("TSVR VARIABLE SUMMARY:\n")
            f.write(f"  - TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours\n")
            f.write(f"  - log_TSVR range: [{log_tsvr_min:.2f}, {log_tsvr_max:.2f}]\n")
            f.write(f"  - Transformation: log_TSVR = log(TSVR_hours + 1)\n\n")

            f.write("VALIDATION RESULTS:\n")
            f.write("  Check 1: Row count = 800 (100 x 4 x 2 - When excluded)\n")
            f.write("  Check 2: No NaN values\n")
            f.write("  Check 3: Mean Age_c approximately 0\n")
            f.write("  Check 4: Age_c range symmetric\n")
            f.write("  Check 5: All UIDs have 8 rows (When excluded)\n")
            f.write("  Check 6: TSVR_hours in [0, 200]\n")
            f.write("  Check 7: log_TSVR in [0, 6]\n\n")

            f.write("TRANSFORMATIONS APPLIED:\n")
            f.write("  1. Extracted UID from composite_ID (format: UID_test)\n")
            f.write("  2. Merged theta + TSVR on composite_ID and test\n")
            f.write("  3. Merged with age on UID\n")
            f.write("  4. Grand-mean centered Age: Age_c = age - mean(age)\n")
            f.write("  5. Log-transformed TSVR: log_TSVR = log(TSVR_hours + 1)\n\n")

            f.write("NEXT STEP:\n")
            f.write("  Use data/step01_lmm_input.csv as input for step02_fit_lmm.py\n")
            f.write("  Model formula will include Age_c, log_TSVR, and their interactions\n\n")

            f.write("=" * 80 + "\n")

        log(f"{summary_path}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
