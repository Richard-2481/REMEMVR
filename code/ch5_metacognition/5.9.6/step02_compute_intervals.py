#!/usr/bin/env python3
"""Compute Interval-Specific Improvements for Confidence: Calculate z-score changes across intervals (T1->T2, T2->T3, T3->T4) for confidence,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.6
LOG_FILE = RQ_DIR / "logs" / "step02_compute_intervals.log"

# Input path
INPUT_PATH = RQ_DIR / "data" / "step01_confidence_standardized.csv"

# Output paths
OUTPUT_INTERVALS = RQ_DIR / "data" / "step02_confidence_intervals.csv"
OUTPUT_WIDE = RQ_DIR / "data" / "step02_confidence_intervals_wide.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Compute Interval-Specific Improvements for Confidence")
        log("=" * 80)
        # Load Standardized Data
        log("\nLoading standardized confidence data...")
        df_long = pd.read_csv(INPUT_PATH, encoding='utf-8')
        log(f"{INPUT_PATH.name} ({len(df_long)} rows, {len(df_long.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'test', 'z_theta']
        missing_cols = [col for col in required_cols if col not in df_long.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Reshape Long to Wide Format
        log("\nPivoting to wide format (one row per participant)...")

        # Pivot on test to get z_theta_T1, z_theta_T2, z_theta_T3, z_theta_T4
        df_wide = df_long.pivot(index='UID', columns='test', values='z_theta')
        df_wide.columns = [f'z_theta_{col}' for col in df_wide.columns]
        df_wide = df_wide.reset_index()

        log(f"{len(df_wide)} rows (participants) x {len(df_wide.columns)} columns")
        log(f"           Columns: {list(df_wide.columns)}")

        # Verify expected columns (T1, T2, T3, T4)
        expected_test_cols = ['z_theta_T1', 'z_theta_T2', 'z_theta_T3', 'z_theta_T4']
        missing_test_cols = [col for col in expected_test_cols if col not in df_wide.columns]
        if missing_test_cols:
            log(f"Missing test columns: {missing_test_cols}")
            log(f"          Available: {[col for col in df_wide.columns if col.startswith('z_theta_')]}")
        # Compute Interval Improvements (Within-Participant Deltas)
        log("\nCalculating interval improvements (deltas)...")

        # Delta = later test - earlier test (positive = improvement)
        df_wide['delta_T1_T2'] = df_wide['z_theta_T2'] - df_wide['z_theta_T1']
        df_wide['delta_T2_T3'] = df_wide['z_theta_T3'] - df_wide['z_theta_T2']
        df_wide['delta_T3_T4'] = df_wide['z_theta_T4'] - df_wide['z_theta_T3']

        log(f"3 interval improvements:")
        log(f"           delta_T1_T2 (immediate practice, 0-24h)")
        log(f"           delta_T2_T3 (distributed practice, 24-72h)")
        log(f"           delta_T3_T4 (late practice, 72-144h)")

        # Check for missing deltas
        for delta_col in ['delta_T1_T2', 'delta_T2_T3', 'delta_T3_T4']:
            n_missing = df_wide[delta_col].isna().sum()
            pct_missing = (n_missing / len(df_wide)) * 100
            log(f"           {delta_col}: {n_missing} missing ({pct_missing:.1f}%)")
        # Compute Summary Statistics for Each Interval
        log("\nComputing summary statistics for each interval...")

        intervals = [
            ('T1->T2', 'delta_T1_T2'),
            ('T2->T3', 'delta_T2_T3'),
            ('T3->T4', 'delta_T3_T4')
        ]

        results = []

        for interval_name, delta_col in intervals:
            # Extract non-missing values
            delta_values = df_wide[delta_col].dropna()
            n = len(delta_values)

            if n == 0:
                log(f"{interval_name}: No valid delta values, skipping")
                continue

            # Compute statistics
            mean_improvement = delta_values.mean()
            se = delta_values.std() / np.sqrt(n)
            ci_lower = mean_improvement - 1.96 * se
            ci_upper = mean_improvement + 1.96 * se

            # One-sample t-test against zero (H0: no change)
            t_stat, p_uncorrected = stats.ttest_1samp(delta_values, 0.0)

            log(f"\n{interval_name}")
            log(f"           N = {n}")
            log(f"           Mean improvement = {mean_improvement:.4f}")
            log(f"           SE = {se:.4f}")
            log(f"           95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
            log(f"           t({n-1}) = {t_stat:.4f}, p = {p_uncorrected:.4f}")

            # Flag negative improvements (forgetting instead of practice)
            if mean_improvement < 0:
                log(f"Negative improvement detected in {interval_name} (forgetting instead of practice)")

            results.append({
                'interval': interval_name,
                'mean_improvement': mean_improvement,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                't_stat': t_stat,
                'p_uncorrected': p_uncorrected
            })

        # Create summary DataFrame
        df_intervals = pd.DataFrame(results)
        log(f"\nCreated interval summary: {len(df_intervals)} rows x {len(df_intervals.columns)} columns")
        # Save Interval Summary
        log(f"\nSaving interval summary...")
        df_intervals.to_csv(OUTPUT_INTERVALS, index=False, encoding='utf-8')
        log(f"{OUTPUT_INTERVALS}")
        log(f"        {len(df_intervals)} rows x {len(df_intervals.columns)} columns")
        # Save Wide Format (for Bootstrap in Step 3)
        log(f"\nSaving wide-format interval improvements...")

        # Select output columns (UID + deltas)
        df_wide_output = df_wide[['UID', 'delta_T1_T2', 'delta_T2_T3', 'delta_T3_T4']].copy()

        df_wide_output.to_csv(OUTPUT_WIDE, index=False, encoding='utf-8')
        log(f"{OUTPUT_WIDE}")
        log(f"        {len(df_wide_output)} rows x {len(df_wide_output.columns)} columns")
        # Run Validation
        log("\nRunning validate_data_columns on interval summary...")

        required_columns = ['interval', 'mean_improvement', 'se', 'ci_lower', 'ci_upper', 't_stat', 'p_uncorrected']
        validation_result = validate_data_columns(df_intervals, required_columns)

        if validation_result['valid']:
            log(f"All required columns present")
        else:
            log(f"Missing columns: {validation_result['missing_columns']}")
            raise ValueError(f"Validation failed: missing columns {validation_result['missing_columns']}")

        # Validate wide format
        log("\nValidating wide-format output...")

        required_wide_cols = ['UID', 'delta_T1_T2', 'delta_T2_T3', 'delta_T3_T4']
        validation_wide = validate_data_columns(df_wide_output, required_wide_cols)

        if validation_wide['valid']:
            log(f"Wide format columns correct")
        else:
            log(f"Missing columns: {validation_wide['missing_columns']}")
            raise ValueError(f"Wide format validation failed")

        # Additional validation: CI validity
        log("\nChecking CI validity (ci_lower < mean < ci_upper)...")

        for _, row in df_intervals.iterrows():
            if not (row['ci_lower'] < row['mean_improvement'] < row['ci_upper']):
                log(f"Invalid CI for {row['interval']}: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] doesn't contain {row['mean_improvement']:.4f}")
                raise ValueError(f"Invalid confidence interval for {row['interval']}")

        log(f"All CIs valid")
        # SUMMARY
        log("\n" + "=" * 80)
        log("Step 2 complete")
        log(f"  Computed 3 intervals: T1->T2, T2->T3, T3->T4")
        log(f"  Participants: {len(df_wide_output)} (100 expected)")
        log(f"  Interval summary: {OUTPUT_INTERVALS} ({len(df_intervals)} rows)")
        log(f"  Wide format: {OUTPUT_WIDE} ({len(df_wide_output)} rows)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
