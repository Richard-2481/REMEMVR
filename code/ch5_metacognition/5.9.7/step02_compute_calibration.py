#!/usr/bin/env python3
"""compute_calibration: Compute calibration scores as z(confidence) - z(accuracy) with within-paradigm"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.7
LOG_FILE = RQ_DIR / "logs" / "step02_compute_calibration.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: compute_calibration")
        # Load Input Data

        log("Loading merged calibration data from Step 1...")

        input_path = RQ_DIR / "data" / "step01_merged_calibration_data.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")
        # Within-Paradigm Z-Standardization
        # Compute z-scores separately for each paradigm across all timepoints
        # This preserves temporal trajectories within paradigm while normalizing baselines

        log("Computing within-paradigm z-scores...")

        # Function to compute z-scores within group
        def standardize_within_paradigm(group):
            """Compute z-scores for theta_accuracy and theta_confidence within paradigm."""
            group['z_accuracy'] = zscore(group['theta_accuracy'], ddof=0, nan_policy='raise')
            group['z_confidence'] = zscore(group['theta_confidence'], ddof=0, nan_policy='raise')
            return group

        # Apply standardization by paradigm
        df = df.groupby('paradigm', group_keys=False).apply(standardize_within_paradigm)

        log(f"Z-scores computed for 3 paradigms")

        # Verify standardization by paradigm
        log("Checking standardization quality by paradigm:")
        for paradigm in df['paradigm'].unique():
            paradigm_data = df[df['paradigm'] == paradigm]
            mean_acc = paradigm_data['z_accuracy'].mean()
            std_acc = paradigm_data['z_accuracy'].std(ddof=0)
            mean_conf = paradigm_data['z_confidence'].mean()
            std_conf = paradigm_data['z_confidence'].std(ddof=0)

            log(f"  {paradigm}:")
            log(f"    z_accuracy: mean={mean_acc:.6f}, SD={std_acc:.6f}")
            log(f"    z_confidence: mean={mean_conf:.6f}, SD={std_conf:.6f}")

            # Check if within tolerance (mean ~ 0, SD ~ 1)
            if abs(mean_acc) > 0.01 or abs(std_acc - 1.0) > 0.01:
                log(f"z_accuracy standardization off for {paradigm}")
            if abs(mean_conf) > 0.01 or abs(std_conf - 1.0) > 0.01:
                log(f"z_confidence standardization off for {paradigm}")
        # Compute Calibration Scores
        # calibration = z_confidence - z_accuracy
        # Values near 0 = good calibration (confidence matches accuracy)

        log("Computing calibration = z_confidence - z_accuracy...")

        df['calibration'] = df['z_confidence'] - df['z_accuracy']

        log(f"Calibration scores for {len(df)} observations")
        log(f"  Range: [{df['calibration'].min():.3f}, {df['calibration'].max():.3f}]")
        log(f"  Mean: {df['calibration'].mean():.3f}, SD: {df['calibration'].std():.3f}")

        # Check for NaN or infinite values
        nan_count = df['calibration'].isna().sum()
        inf_count = np.isinf(df['calibration']).sum()
        if nan_count > 0 or inf_count > 0:
            log(f"Invalid calibration values: NaN={nan_count}, Inf={inf_count}")
            sys.exit(1)

        log("No invalid calibration values")
        # Validate Standardization

        log("Running standardization validation...")

        validation_result = validate_standardization(
            df,
            column_names=['z_accuracy', 'z_confidence'],
            tolerance=0.01
        )

        if not validation_result.get('valid', False):
            log(f"Standardization validation failed: {validation_result.get('message', 'Unknown error')}")
            sys.exit(1)

        log(f"Standardization validated: {validation_result.get('message', 'OK')}")
        # Compute Descriptive Statistics by Paradigm × Time

        log("Computing statistics by paradigm × timepoint...")

        desc_stats = df.groupby(['paradigm', 'test']).agg(
            n=('calibration', 'count'),
            mean_calibration=('calibration', 'mean'),
            sd_calibration=('calibration', 'std'),
            min_calibration=('calibration', 'min'),
            max_calibration=('calibration', 'max')
        ).reset_index()

        # Identify outliers (|calibration| > 3) per cell
        def count_outliers(group):
            return (np.abs(group['calibration']) > 3).sum()

        outlier_counts = df.groupby(['paradigm', 'test']).apply(count_outliers).reset_index(name='outlier_count')
        desc_stats = desc_stats.merge(outlier_counts, on=['paradigm', 'test'])

        log(f"Descriptive statistics: {len(desc_stats)} cells (3 paradigms × 4 times)")

        # Check overall outlier proportion
        total_outliers = desc_stats['outlier_count'].sum()
        outlier_pct = (total_outliers / len(df)) * 100
        log(f"Total outliers (|calibration| > 3): {total_outliers} ({outlier_pct:.2f}%)")

        if outlier_pct > 5.0:
            log(f"Outlier proportion exceeds 5% threshold")
        else:
            log("Outlier proportion acceptable (< 5%)")

        # Display descriptive statistics
        log("Descriptive statistics by paradigm × time:")
        for _, row in desc_stats.iterrows():
            log(f"  {row['paradigm']}, {row['test']}: n={row['n']}, mean={row['mean_calibration']:.3f}, SD={row['sd_calibration']:.3f}, outliers={row['outlier_count']}")
        # Save Outputs

        log("Saving calibration scores and descriptive statistics...")

        # Save full calibration dataset
        calibration_out = RQ_DIR / "data" / "step02_calibration_scores.csv"
        df.to_csv(calibration_out, index=False, encoding='utf-8')
        log(f"{calibration_out.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Save descriptive statistics
        desc_out = RQ_DIR / "data" / "step02_descriptive_stats.csv"
        desc_stats.to_csv(desc_out, index=False, encoding='utf-8')
        log(f"{desc_out.name} ({len(desc_stats)} rows)")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
