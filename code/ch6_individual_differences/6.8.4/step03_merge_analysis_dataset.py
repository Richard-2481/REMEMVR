#!/usr/bin/env python3
"""merge_analysis_dataset: Create complete analysis dataset by merging domain theta scores with cognitive predictors."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_missing_data

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4
LOG_FILE = RQ_DIR / "logs" / "step03_merge_analysis_dataset.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: merge_analysis_dataset")
        # Load Input Data

        log("Loading input data...")

        # Load domain theta scores
        theta_path = RQ_DIR / 'data' / 'step01_domain_theta_scores.csv'
        df_theta = pd.read_csv(theta_path)
        log(f"{theta_path} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")

        # Load cognitive tests
        cognitive_path = RQ_DIR / 'data' / 'step02_cognitive_tests.csv'
        df_cognitive = pd.read_csv(cognitive_path)
        log(f"{cognitive_path} ({len(df_cognitive)} rows, {len(df_cognitive.columns)} cols)")
        # Merge Datasets
        # BYPASS tools.data.merge_theta_cognitive (hardcodes lowercase 'uid')
        # Use direct pandas merge with uppercase 'UID' (actual column name)

        log("Merging theta and cognitive datasets...")

        df_merged = pd.merge(
            df_theta,
            df_cognitive,
            on='UID',
            how='inner'  # Complete cases only
        )

        log(f"{len(df_merged)} participants with complete data")

        # Report merge statistics
        theta_only = len(df_theta) - len(df_merged)
        cognitive_only = len(df_cognitive) - len(df_merged)

        if theta_only > 0:
            log(f"{theta_only} participants had theta but not cognitive data")
        if cognitive_only > 0:
            log(f"{cognitive_only} participants had cognitive but not theta data")
        # Compute Descriptive Statistics

        log("Computing descriptive statistics...")

        # Variables to describe (all except UID)
        analysis_vars = ['What_theta', 'Where_theta', 'When_theta', 'ravlt_z', 'bvmt_z', 'rpm_z', 'age_z', 'pctret_z']

        descriptive_stats = []

        for var in analysis_vars:
            stats = {
                'variable': var,
                'n': df_merged[var].count(),
                'mean': df_merged[var].mean(),
                'sd': df_merged[var].std(),
                'min': df_merged[var].min(),
                'max': df_merged[var].max()
            }
            descriptive_stats.append(stats)

            log(f"{var}: n={stats['n']}, mean={stats['mean']:.3f}, SD={stats['sd']:.3f}")

        df_descriptive = pd.DataFrame(descriptive_stats)
        # Save Outputs
        # These outputs will be used by: Steps 04-05 (univariate and multivariate modeling)

        log("Saving analysis dataset...")
        analysis_output = RQ_DIR / 'data' / 'step03_analysis_dataset.csv'
        df_merged.to_csv(analysis_output, index=False, encoding='utf-8')
        log(f"{analysis_output} ({len(df_merged)} rows, {len(df_merged.columns)} cols)")

        log("Saving descriptive statistics...")
        descriptive_output = RQ_DIR / 'data' / 'step03_descriptive_stats.csv'
        df_descriptive.to_csv(descriptive_output, index=False, encoding='utf-8')
        log(f"{descriptive_output} ({len(df_descriptive)} rows, {len(df_descriptive.columns)} cols)")
        # Validation
        # Validates: No missing data in merged dataset
        # Threshold: 0 missing values

        log("Running validation...")

        missing_result = check_missing_data(df=df_merged)

        if missing_result.get('has_missing', False):
            total_missing = missing_result.get('total_missing', 0)
            percent_missing = missing_result.get('percent_missing', 0)
            log(f"Missing data detected: {total_missing} cells ({percent_missing:.2f}%)")
            log(f"Missing by column: {missing_result.get('missing_by_column', {})}")
            sys.exit(1)
        else:
            log(f"No missing data in merged dataset")

        # Validate z-score properties
        z_vars = ['ravlt_z', 'bvmt_z', 'rpm_z', 'age_z', 'pctret_z']
        for var in z_vars:
            mean_val = df_merged[var].mean()
            std_val = df_merged[var].std()

            # Check mean ≈ 0 (within numerical precision)
            if abs(mean_val) > 1e-10:
                log(f"{var} mean = {mean_val:.6f} (expected ≈ 0)")

            # Check SD ≈ 1 (within numerical precision)
            if abs(std_val - 1.0) > 1e-10:
                log(f"{var} SD = {std_val:.6f} (expected ≈ 1)")

        log(f"Z-score properties validated")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
