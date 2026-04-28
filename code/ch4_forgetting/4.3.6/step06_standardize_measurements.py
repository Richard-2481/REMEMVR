#!/usr/bin/env python3
"""Z-Standardize Measurements: Z-standardize (grand mean center and scale) all measurement paradigms:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.6
LOG_FILE = RQ_DIR / "logs" / "step06_standardize_measurements.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Z-Standardization Function

def z_standardize(series: pd.Series) -> pd.Series:
    """
    Z-standardize a series: z = (x - mean) / sd

    Parameters:
    -----------
    series : pd.Series
        Values to standardize

    Returns:
    --------
    pd.Series
        Z-standardized values (mean=0, sd=1)
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        log(f"Zero standard deviation for {series.name}, returning zeros")
        return pd.Series(0, index=series.index)

    z = (series - mean) / std
    return z

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Z-Standardize Measurements")
        # Load Input Data
        log("Loading input data...")

        # Load theta scores (long format)
        theta_long = pd.read_csv(RQ_DIR / "data" / "step00_theta_scores.csv")
        log(f"Theta scores: {len(theta_long)} rows, {len(theta_long.columns)} cols")
        log(f"Theta domains: {theta_long['domain_name'].unique().tolist()}")

        # Load full CTT scores
        ctt_full = pd.read_csv(RQ_DIR / "data" / "step02_ctt_full_scores.csv")
        log(f"Full CTT scores: {len(ctt_full)} rows, {len(ctt_full.columns)} cols")

        # Load purified CTT scores
        ctt_purified = pd.read_csv(RQ_DIR / "data" / "step03_ctt_purified_scores.csv")
        log(f"Purified CTT scores: {len(ctt_purified)} rows, {len(ctt_purified.columns)} cols")

        # Load TSVR mapping
        tsvr_mapping = pd.read_csv(RQ_DIR / "data" / "step00_tsvr_mapping.csv")
        log(f"TSVR mapping: {len(tsvr_mapping)} rows, {len(tsvr_mapping.columns)} cols")
        # Reshape Theta from Long to Wide Format
        log("Converting theta from long to wide format...")

        # Map domain names to paradigm codes
        domain_map = {
            'free_recall': 'IFR',
            'cued_recall': 'ICR',
            'recognition': 'IRE'
        }

        theta_long['paradigm'] = theta_long['domain_name'].map(domain_map)

        # Pivot to wide format
        theta_wide = theta_long.pivot(
            index='composite_ID',
            columns='paradigm',
            values='theta'
        ).reset_index()

        # Rename columns to theta_IFR, theta_ICR, theta_IRE
        theta_wide.columns = ['composite_ID'] + [f'theta_{col}' for col in theta_wide.columns if col != 'composite_ID']

        log(f"Theta wide format: {len(theta_wide)} rows, {len(theta_wide.columns)} cols")
        log(f"Theta columns: {theta_wide.columns.tolist()}")
        # Create composite_ID for CTT Files
        log("Creating composite_ID for CTT files...")

        # Full CTT: composite_ID = UID_test (test is string T1, T2, etc., strip "T" prefix)
        ctt_full['test_num'] = ctt_full['test'].str.replace('T', '')
        ctt_full['composite_ID'] = ctt_full['UID'] + "_" + ctt_full['test_num']

        # Purified CTT: composite_ID = UID_test (test is integer 1-4, need to convert)
        ctt_purified['test'] = ctt_purified['test'].astype(str)
        ctt_purified['composite_ID'] = ctt_purified['UID'] + "_" + ctt_purified['test']

        log(f"Full CTT composite_ID sample: {ctt_full['composite_ID'].head(3).tolist()}")
        log(f"Purified CTT composite_ID sample: {ctt_purified['composite_ID'].head(3).tolist()}")
        # Merge All Data on composite_ID
        log("Merging all datasets on composite_ID...")

        # Start with TSVR mapping (has composite_ID, UID, test, TSVR_hours)
        merged = tsvr_mapping.copy()
        log(f"Base (TSVR): {len(merged)} rows")

        # Merge theta (wide)
        merged = merged.merge(theta_wide, on='composite_ID', how='left')
        log(f"+ Theta: {len(merged)} rows")

        # Merge full CTT (drop duplicate UID, test columns, keep test_num for reference)
        ctt_full_merge = ctt_full[['composite_ID', 'CTT_full_IFR', 'CTT_full_ICR', 'CTT_full_IRE']]
        merged = merged.merge(ctt_full_merge, on='composite_ID', how='left')
        log(f"+ Full CTT: {len(merged)} rows")

        # Merge purified CTT (drop duplicate UID, test columns)
        ctt_purified_merge = ctt_purified.drop(columns=['UID', 'test'])
        merged = merged.merge(ctt_purified_merge, on='composite_ID', how='left')
        log(f"+ Purified CTT: {len(merged)} rows")

        log(f"Final dataset: {len(merged)} rows, {len(merged.columns)} cols")
        log(f"Columns: {merged.columns.tolist()}")

        # Check for missing values
        missing_counts = merged.isnull().sum()
        if missing_counts.sum() > 0:
            log("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                log(f"  {col}: {count} missing")
        # Z-Standardize All Measurements
        log("Z-standardizing all measurement paradigms...")

        # IRT Theta scores
        merged['z_theta_IFR'] = z_standardize(merged['theta_IFR'])
        merged['z_theta_ICR'] = z_standardize(merged['theta_ICR'])
        merged['z_theta_IRE'] = z_standardize(merged['theta_IRE'])
        log("IRT theta scores (IFR, ICR, IRE)")

        # Full CTT scores
        merged['z_CTT_full_IFR'] = z_standardize(merged['CTT_full_IFR'])
        merged['z_CTT_full_ICR'] = z_standardize(merged['CTT_full_ICR'])
        merged['z_CTT_full_IRE'] = z_standardize(merged['CTT_full_IRE'])
        log("Full CTT scores (IFR, ICR, IRE)")

        # Purified CTT scores
        merged['z_CTT_purified_IFR'] = z_standardize(merged['CTT_purified_IFR'])
        merged['z_CTT_purified_ICR'] = z_standardize(merged['CTT_purified_ICR'])
        merged['z_CTT_purified_IRE'] = z_standardize(merged['CTT_purified_IRE'])
        log("Purified CTT scores (IFR, ICR, IRE)")
        # Select Output Columns and Reorder
        log("Selecting and ordering output columns...")

        output_cols = [
            'composite_ID', 'UID', 'test', 'TSVR_hours',
            'z_theta_IFR', 'z_theta_ICR', 'z_theta_IRE',
            'z_CTT_full_IFR', 'z_CTT_full_ICR', 'z_CTT_full_IRE',
            'z_CTT_purified_IFR', 'z_CTT_purified_ICR', 'z_CTT_purified_IRE'
        ]

        output = merged[output_cols].copy()
        log(f"Output dataset: {len(output)} rows, {len(output.columns)} cols")
        # Validate Z-Standardization
        log("Validating z-standardization...")

        z_cols = [col for col in output.columns if col.startswith('z_')]

        validation_passed = True

        for col in z_cols:
            mean = output[col].mean()
            std = output[col].std()

            # Check mean within [-0.01, 0.01]
            if abs(mean) > 0.01:
                log(f"{col} mean = {mean:.6f} (expected ~0)")
                validation_passed = False
            else:
                log(f"{col} mean = {mean:.6f}")

            # Check SD within [0.99, 1.01]
            if not (0.99 <= std <= 1.01):
                log(f"{col} SD = {std:.6f} (expected ~1)")
                validation_passed = False
            else:
                log(f"{col} SD = {std:.6f}")

        # Check for NaN values
        nan_counts = output[z_cols].isnull().sum()
        if nan_counts.sum() > 0:
            log("NaN values detected in z_* columns:")
            for col, count in nan_counts[nan_counts > 0].items():
                log(f"  {col}: {count} NaN values")
            validation_passed = False
        else:
            log("No NaN values in z_* columns")

        # Check TSVR range (allow up to 250 hours for late participants)
        tsvr_min = output['TSVR_hours'].min()
        tsvr_max = output['TSVR_hours'].max()
        if not (0 <= tsvr_min and tsvr_max <= 250):
            log(f"TSVR_hours range [{tsvr_min:.2f}, {tsvr_max:.2f}] outside expected [0, 250]")
            validation_passed = False
        else:
            log(f"TSVR_hours range [{tsvr_min:.2f}, {tsvr_max:.2f}] within [0, 250]")

        # Check row count
        if len(output) != 400:
            log(f"Expected 400 rows, got {len(output)}")
            validation_passed = False
        else:
            log(f"400 rows present (100 participants × 4 tests)")

        if not validation_passed:
            log("Validation failed - see errors above")
            sys.exit(1)

        log("All checks passed")
        # Save Output
        output_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(output)} rows, {len(output.columns)} cols)")
        # Summary Statistics
        log("Z-standardization summary:")
        log(f"  Total observations: {len(output)}")
        log(f"  Participants: {output['UID'].nunique()}")
        log(f"  Test sessions: {output['test'].nunique()}")
        log(f"  Z-standardized paradigms: {len(z_cols)}")
        log("")
        log("  Descriptive statistics for each z_* column:")
        for col in z_cols:
            mean = output[col].mean()
            std = output[col].std()
            min_val = output[col].min()
            max_val = output[col].max()
            log(f"    {col:25s}: mean={mean:7.4f}, SD={std:7.4f}, range=[{min_val:7.4f}, {max_val:7.4f}]")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
