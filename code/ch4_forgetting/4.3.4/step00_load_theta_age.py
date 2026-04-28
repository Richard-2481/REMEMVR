#!/usr/bin/env python3
"""Load Theta Scores and Age Data: Load paradigm-specific theta scores from RQ 5.3.1 (Free Recall, Cued Recall,"""

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
LOG_FILE = RQ_DIR / "logs" / "step00_load_theta_age.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Load Theta Scores and Age Data")
        # Load Theta Scores from RQ 5.3.1

        log("Loading theta scores from RQ 5.3.1...")
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step03_theta_scores.csv"

        if not theta_path.exists():
            raise FileNotFoundError(f"Theta scores file not found: {theta_path}")

        df_theta = pd.read_csv(theta_path, encoding='utf-8')
        log(f"Theta scores: {len(df_theta)} rows, {len(df_theta.columns)} columns")
        log(f"Columns: {list(df_theta.columns)}")
        log(f"Domain values: {sorted(df_theta['domain_name'].unique())}")
        # Parse composite_ID to Extract UID and Test
        # composite_ID format: 'UID_test' (e.g., 'A010_1')
        # Extract UID (participant ID) and test (session number)

        log("Extracting UID and test from composite_ID...")

        # Split composite_ID into components
        df_theta[['UID', 'test']] = df_theta['composite_ID'].str.split('_', expand=True)

        # Convert test to integer (T1=1, T2=2, T3=3, T4=4)
        df_theta['test'] = df_theta['test'].astype(int)

        log(f"UID range: {df_theta['UID'].nunique()} unique participants")
        log(f"Test range: {sorted(df_theta['test'].unique())}")
        # Map Domain Names to Paradigm Abbreviations
        # Map domain_name values to paradigm abbreviations for consistency with
        # other RQs and literature conventions

        log("Mapping domain names to paradigm abbreviations...")

        domain_to_paradigm = {
            'free_recall': 'IFR',
            'cued_recall': 'ICR',
            'recognition': 'IRE'
        }

        df_theta['paradigm'] = df_theta['domain_name'].map(domain_to_paradigm)

        # Validate mapping (no NaN values = all domains mapped)
        if df_theta['paradigm'].isna().any():
            unmapped = df_theta[df_theta['paradigm'].isna()]['domain_name'].unique()
            raise ValueError(f"Unmapped domain names found: {unmapped}")

        log(f"Paradigm distribution:")
        for paradigm, count in df_theta['paradigm'].value_counts().items():
            log(f"  {paradigm}: {count} observations")
        # Load Age Variable from dfData.csv

        log("Loading Age variable from dfData.csv...")
        age_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

        if not age_path.exists():
            raise FileNotFoundError(f"dfData.csv not found: {age_path}")

        # Load only UID and age columns (dfData has many columns we don't need)
        # NOTE: dfData has 400 rows (100 participants × 4 tests) - need to deduplicate
        df_age_raw = pd.read_csv(age_path, usecols=['UID', 'age'], encoding='utf-8')

        # Deduplicate: Get unique Age per UID (Age is constant within participant)
        df_age = df_age_raw.drop_duplicates(subset=['UID']).reset_index(drop=True)

        # Rename 'age' to 'Age' for consistency (output should have uppercase)
        df_age = df_age.rename(columns={'age': 'Age'})

        log(f"Age data: {len(df_age)} unique participants (deduplicated from {len(df_age_raw)} rows)")
        log(f"Age range: {df_age['Age'].min():.1f} - {df_age['Age'].max():.1f} years")
        log(f"Age mean: {df_age['Age'].mean():.1f}, SD: {df_age['Age'].std():.1f}")
        # Merge Theta Scores with Age on UID
        # Left join: Keep all theta observations (1200 rows), add Age for each

        log("Merging theta scores with Age on UID...")

        df_merged = df_theta.merge(df_age, on='UID', how='left')

        log(f"Output: {len(df_merged)} rows, {len(df_merged.columns)} columns")
        # Validation - Row Count
        # CRITICAL: Must have exactly 1200 rows (no data loss from merge)

        log("Checking row count...")

        expected_rows = 1200
        actual_rows = len(df_merged)

        if actual_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
            )

        log(f"Row count: {actual_rows} rows (100 participants x 4 tests x 3 paradigms)")
        # Validation - No Missing Values
        # CRITICAL: All theta observations must have Age (merge success)

        log("Checking for missing values...")

        # Check each column for NaN values
        missing_summary = df_merged.isna().sum()
        total_missing = missing_summary.sum()

        if total_missing > 0:
            log("Missing values detected:")
            for col, count in missing_summary[missing_summary > 0].items():
                log(f"  {col}: {count} missing")
            raise ValueError(f"Missing values found: {total_missing} total NaN values")

        log("No missing values (all 1200 theta observations matched with Age)")
        # Validation - Unique Participants

        log("Checking unique participants...")

        expected_uids = 100
        actual_uids = df_merged['UID'].nunique()

        if actual_uids != expected_uids:
            raise ValueError(
                f"UID count mismatch: expected {expected_uids}, got {actual_uids}"
            )

        log(f"Unique participants: {actual_uids} UIDs")
        # Validation - Balanced Paradigm Design

        log("Checking paradigm balance...")

        paradigm_counts = df_merged['paradigm'].value_counts()
        expected_per_paradigm = 400

        for paradigm, count in paradigm_counts.items():
            if count != expected_per_paradigm:
                raise ValueError(
                    f"Paradigm imbalance: {paradigm} has {count} observations, "
                    f"expected {expected_per_paradigm}"
                )

        log("Paradigm balance:")
        for paradigm in ['IFR', 'ICR', 'IRE']:
            log(f"  {paradigm}: {paradigm_counts[paradigm]} observations")
        # Validation - Value Ranges
        # theta: Typical IRT ability range is [-3, 3]
        # Age: Study inclusion criteria is 20-70 years

        log("Checking value ranges...")

        # Check theta range
        theta_min, theta_max = df_merged['theta'].min(), df_merged['theta'].max()
        log(f"Theta range: [{theta_min:.2f}, {theta_max:.2f}]")

        if theta_min < -3 or theta_max > 3:
            log(f"Theta values outside typical range [-3, 3]")
            # Not a critical error - some extreme abilities are possible
        else:
            log("Theta values in typical IRT range [-3, 3]")

        # Check Age range
        age_min, age_max = df_merged['Age'].min(), df_merged['Age'].max()
        log(f"Age range: [{age_min:.1f}, {age_max:.1f}]")

        if age_min < 20 or age_max > 70:
            raise ValueError(
                f"Age values outside study criteria [20, 70]: [{age_min}, {age_max}]"
            )

        log("Age values within study inclusion criteria [20, 70]")
        # Validation - Categorical Values
        # paradigm: Must be in {IFR, ICR, IRE}
        # test: Must be in {1, 2, 3, 4}

        log("Checking categorical values...")

        # Check paradigm values
        expected_paradigms = {'IFR', 'ICR', 'IRE'}
        actual_paradigms = set(df_merged['paradigm'].unique())

        if actual_paradigms != expected_paradigms:
            raise ValueError(
                f"Paradigm values mismatch: expected {expected_paradigms}, "
                f"got {actual_paradigms}"
            )

        log(f"Paradigm values: {sorted(actual_paradigms)}")

        # Check test values
        expected_tests = {1, 2, 3, 4}
        actual_tests = set(df_merged['test'].unique())

        if actual_tests != expected_tests:
            raise ValueError(
                f"Test values mismatch: expected {expected_tests}, got {actual_tests}"
            )

        log(f"Test values: {sorted(actual_tests)}")
        # Select and Order Output Columns
        # Output columns: composite_ID, UID, test, paradigm, theta, Age
        # Drop intermediate column: domain_name (replaced by paradigm)

        log("Selecting output columns...")

        output_columns = ['composite_ID', 'UID', 'test', 'paradigm', 'theta', 'Age']
        df_output = df_merged[output_columns].copy()

        log(f"Output columns: {list(df_output.columns)}")
        # Save Merged Data
        # Output: data/step00_theta_age_merged.csv
        # Contains: 1200 rows with theta scores + Age for 3-way interaction analysis

        log("Saving merged data...")

        output_path = RQ_DIR / "data" / "step00_theta_age_merged.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path}")
        log(f"Saved {len(df_output)} rows, {len(df_output.columns)} columns")
        # Final Summary

        log("Data merge complete:")
        log(f"  Input theta: 1200 rows (RQ 5.3.1 paradigm-specific theta scores)")
        log(f"  Input Age: 100 rows (participant demographics)")
        log(f"  Output: 1200 rows (theta + Age for all observations)")
        log(f"  Participants: {df_output['UID'].nunique()} unique UIDs")
        log(f"  Paradigms: {', '.join(sorted(df_output['paradigm'].unique()))}")
        log(f"  Age range: {df_output['Age'].min():.1f} - {df_output['Age'].max():.1f} years")
        log(f"  Theta range: {df_output['theta'].min():.2f} - {df_output['theta'].max():.2f}")

        log("Step 0 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
