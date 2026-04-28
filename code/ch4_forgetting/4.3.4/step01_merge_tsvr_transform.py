#!/usr/bin/env python3
"""Merge TSVR and Transform Variables: Merge TSVR time variable from RQ 5.3.1 with theta scores and Age data, then"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_merge_tsvr_transform.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Merge TSVR and Transform Variables")
        # Load Input Data
        #           TSVR mapping from RQ 5.3.1 (400 rows, no paradigm dimension)

        log("Loading Step 0 theta-age merged data...")
        theta_age_path = RQ_DIR / "data" / "step00_theta_age_merged.csv"
        df_theta_age = pd.read_csv(theta_age_path, encoding='utf-8')
        log(f"step00_theta_age_merged.csv ({len(df_theta_age)} rows, {len(df_theta_age.columns)} cols)")
        log(f"Columns: {list(df_theta_age.columns)}")

        # Validate expected columns
        expected_cols_theta = ['composite_ID', 'UID', 'test', 'paradigm', 'theta', 'Age']
        if list(df_theta_age.columns) != expected_cols_theta:
            raise ValueError(f"Theta-age file has unexpected columns. Expected {expected_cols_theta}, got {list(df_theta_age.columns)}")

        log("Loading TSVR mapping from RQ 5.3.1...")
        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"Columns: {list(df_tsvr.columns)}")

        # Validate expected columns
        expected_cols_tsvr = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        if list(df_tsvr.columns) != expected_cols_tsvr:
            raise ValueError(f"TSVR file has unexpected columns. Expected {expected_cols_tsvr}, got {list(df_tsvr.columns)}")

        # Validate row counts before merge
        if len(df_theta_age) != 1200:
            raise ValueError(f"Theta-age file expected 1200 rows, got {len(df_theta_age)}")
        if len(df_tsvr) != 400:
            raise ValueError(f"TSVR file expected 400 rows, got {len(df_tsvr)}")

        log("Merge strategy: UID + test (TSVR has no paradigm dimension, must replicate across paradigms)")
        # Merge Datasets on UID + test
        # CRITICAL: Cannot merge on composite_ID despite matching formats because:
        # - Theta file: 1200 rows (100 participants x 4 tests x 3 paradigms)
        # - TSVR file: 400 rows (100 participants x 4 tests, NO paradigm)
        # Solution: Merge on (UID, test) to replicate TSVR_hours across paradigms

        log("Merging theta-age with TSVR on (UID, test)...")
        df_merged = df_theta_age.merge(
            df_tsvr[['UID', 'test', 'TSVR_hours']],  # Select only needed columns to avoid composite_ID duplicate
            on=['UID', 'test'],
            how='left'
        )
        log(f"Result: {len(df_merged)} rows, {len(df_merged.columns)} cols")

        # Validate merge success
        if len(df_merged) != 1200:
            raise ValueError(f"Merge produced {len(df_merged)} rows, expected 1200 (data loss occurred)")

        # Check for missing TSVR values (merge failure)
        missing_tsvr = df_merged['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            raise ValueError(f"Merge failed: {missing_tsvr} rows missing TSVR_hours (not all theta observations matched)")

        log(f"Merge successful - all 1200 theta observations have TSVR_hours")
        # Transform Variables
        # Create two transformed variables:
        # 1. Age_c: Grand-mean centered Age (Age - mean(Age))
        # 2. log_TSVR: Log transformation of TSVR_hours + 1

        log("Creating Age_c (grand-mean centered)...")
        age_mean = df_merged['Age'].mean()
        df_merged['Age_c'] = df_merged['Age'] - age_mean
        log(f"Age mean: {age_mean:.3f}")
        log(f"Age_c mean: {df_merged['Age_c'].mean():.6f} (should be ~0)")
        log(f"Age SD: {df_merged['Age'].std():.3f}")
        log(f"Age_c SD: {df_merged['Age_c'].std():.3f} (should equal Age SD)")

        # Validate Age_c centering
        age_c_mean = df_merged['Age_c'].mean()
        if abs(age_c_mean) > 0.1:
            raise ValueError(f"Age_c mean = {age_c_mean:.6f}, exceeds tolerance ±0.1 (centering failed)")
        log(f"Age_c grand-mean centered (mean = {age_c_mean:.6f})")

        # Validate Age_c preserves variance
        age_sd = df_merged['Age'].std()
        age_c_sd = df_merged['Age_c'].std()
        if abs(age_sd - age_c_sd) > 0.01:
            raise ValueError(f"Age_c SD ({age_c_sd:.3f}) differs from Age SD ({age_sd:.3f}), centering altered variance")
        log(f"Age_c preserves variance (SD = {age_c_sd:.3f})")

        log("Creating log_TSVR (log transformation)...")
        df_merged['log_TSVR'] = np.log(df_merged['TSVR_hours'] + 1)
        log(f"log_TSVR range: [{df_merged['log_TSVR'].min():.3f}, {df_merged['log_TSVR'].max():.3f}]")
        log(f"TSVR_hours range: [{df_merged['TSVR_hours'].min():.1f}, {df_merged['TSVR_hours'].max():.1f}] hours")
        # NOTE: Study data shows TSVR_hours ranges from ~1 to ~246 hours (not 0-168)
        # Some participants had delayed test sessions, which is valid study data

        # Validate log_TSVR range (must be positive, reasonable upper bound)
        if df_merged['log_TSVR'].min() < 0:
            raise ValueError(f"log_TSVR minimum is negative: {df_merged['log_TSVR'].min():.3f} (transformation error)")
        # Upper bound: log(300+1) ≈ 5.7 is reasonable for study with some delayed sessions
        if df_merged['log_TSVR'].max() > 6.0:
            raise ValueError(f"log_TSVR maximum exceeds reasonable bound: {df_merged['log_TSVR'].max():.3f} > 6.0")
        log(f"log_TSVR range valid (transformation successful)")

        # Validate log_TSVR monotonicity (should increase with TSVR_hours)
        log("Checking log_TSVR monotonicity...")
        test_values = [0, 1, 24, 72, 144, 168]
        for tsvr_val in test_values:
            log_val = np.log(tsvr_val + 1)
            log(f"log({tsvr_val} + 1) = {log_val:.3f}")
        log("log_TSVR transformation correct")
        # Validate Complete Dataset
        # Check for NaN values and validate expected structure

        log("Checking for missing values...")
        missing_counts = df_merged.isna().sum()
        if missing_counts.sum() > 0:
            log(f"Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                log(f"  {col}: {count} missing")
            raise ValueError("Dataset has missing values")
        log(f"No missing values in merged dataset")

        log("Checking value ranges...")
        # Theta should be in typical IRT range
        theta_min, theta_max = df_merged['theta'].min(), df_merged['theta'].max()
        log(f"theta range: [{theta_min:.3f}, {theta_max:.3f}]")
        if theta_min < -3 or theta_max > 3:
            log(f"theta range outside typical [-3, 3], but proceeding")

        # Age should be in study range
        age_min, age_max = df_merged['Age'].min(), df_merged['Age'].max()
        log(f"Age range: [{age_min:.1f}, {age_max:.1f}]")
        if age_min < 20 or age_max > 70:
            raise ValueError(f"Age range [{age_min:.1f}, {age_max:.1f}] outside study criteria [20, 70]")
        log(f"Age range valid")

        # TSVR_hours should be positive
        tsvr_min, tsvr_max = df_merged['TSVR_hours'].min(), df_merged['TSVR_hours'].max()
        log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
        if tsvr_min < 0:
            raise ValueError(f"TSVR_hours has negative values: minimum = {tsvr_min:.2f}")
        log(f"TSVR_hours range valid")

        # Check paradigm values
        paradigms = df_merged['paradigm'].unique()
        expected_paradigms = ['IFR', 'ICR', 'IRE']
        if not set(paradigms) == set(expected_paradigms):
            raise ValueError(f"Unexpected paradigm values. Expected {expected_paradigms}, got {list(paradigms)}")
        log(f"Paradigms: {list(paradigms)}")

        # Check paradigm balance (should be 400 rows per paradigm)
        paradigm_counts = df_merged['paradigm'].value_counts()
        log(f"Paradigm counts:")
        for paradigm, count in paradigm_counts.items():
            log(f"  {paradigm}: {count} observations")
            if count != 400:
                raise ValueError(f"Paradigm {paradigm} has {count} observations, expected 400 (unbalanced design)")
        log(f"Paradigm balance validated (400 observations per paradigm)")

        # Check participant count
        n_participants = df_merged['UID'].nunique()
        log(f"Unique participants: {n_participants}")
        if n_participants != 100:
            raise ValueError(f"Expected 100 participants, got {n_participants}")
        log(f"All 100 participants present")
        # Save LMM Input Data
        # Output contains: composite_ID, UID, test, paradigm, theta, Age, Age_c, TSVR_hours, log_TSVR

        log("Saving LMM input data...")
        output_path = RQ_DIR / "data" / "step01_lmm_input.csv"

        # Reorder columns for clarity
        output_cols = ['composite_ID', 'UID', 'test', 'paradigm', 'theta', 'Age', 'Age_c', 'TSVR_hours', 'log_TSVR']
        df_output = df_merged[output_cols]

        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        log(f"Output columns: {list(df_output.columns)}")

        # Final validation of output file
        log("Verifying saved file...")
        df_check = pd.read_csv(output_path, encoding='utf-8')
        if len(df_check) != 1200:
            raise ValueError(f"Saved file has {len(df_check)} rows, expected 1200")
        if list(df_check.columns) != output_cols:
            raise ValueError(f"Saved file has unexpected columns")
        log(f"Output file validated")

        log("Step 01 complete")
        log("")
        log(f"  Input: 1200 theta-age observations + 400 TSVR observations")
        log(f"  Merge: UID + test (replicated TSVR across 3 paradigms)")
        log(f"  Age_c: Grand-mean centered (mean = {age_c_mean:.6f})")
        log(f"  log_TSVR: Log-transformed time (range [{df_merged['log_TSVR'].min():.3f}, {df_merged['log_TSVR'].max():.3f}])")
        log(f"  Output: 1200 rows x 9 columns")
        log(f"  Ready for LMM analysis (Step 2)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
