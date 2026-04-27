#!/usr/bin/env python3
"""
RQ 5.2.7 Step 01: Standardize Clustering Features

Purpose: Standardize all 4 clustering variables to z-scores (mean=0, SD=1)
         to ensure equal weighting in K-means distance calculations.

Input:
  - data/step00_random_effects_from_rq526.csv (100 rows × 5 cols)

Output:
  - data/step01_standardized_features.csv (100 rows × 5 cols: UID + 4 z-scored variables)
  - data/step01_standardization_summary.txt (summary report)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input file
INPUT_FILE = DATA_DIR / "step00_random_effects_from_rq526.csv"

# Clustering variables to standardize
CLUSTERING_VARS = [
    'Total_Intercept_What',
    'Total_Intercept_Where',
    'Total_Slope_What',
    'Total_Slope_Where'
]


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 01: Standardize Clustering Features")
    print("=" * 60)

    # Load input data
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")

    # Validate input
    missing_vars = [v for v in CLUSTERING_VARS if v not in df.columns]
    if missing_vars:
        print(f"\nERROR: Missing clustering variables: {missing_vars}")
        sys.exit(1)

    # Store raw statistics before standardization
    raw_stats = {}
    print("\nRaw variable statistics (before standardization):")
    for var in CLUSTERING_VARS:
        raw_stats[var] = {
            'mean': df[var].mean(),
            'sd': df[var].std(),
            'min': df[var].min(),
            'max': df[var].max()
        }
        print(f"  {var}:")
        print(f"    Mean: {raw_stats[var]['mean']:.4f}")
        print(f"    SD:   {raw_stats[var]['sd']:.4f}")
        print(f"    Range: [{raw_stats[var]['min']:.4f}, {raw_stats[var]['max']:.4f}]")

    # Check for zero variance (would cause division by zero)
    zero_var = [v for v in CLUSTERING_VARS if raw_stats[v]['sd'] == 0]
    if zero_var:
        print(f"\nERROR: Zero variance detected in: {zero_var}")
        sys.exit(1)

    # Standardize to z-scores
    print("\nStandardizing to z-scores...")
    df_z = df[['UID']].copy()

    for var in CLUSTERING_VARS:
        z_var = f"{var}_z"
        df_z[z_var] = (df[var] - raw_stats[var]['mean']) / raw_stats[var]['sd']

    # Validate standardization
    print("\nZ-scored variable statistics (after standardization):")
    z_stats = {}
    outliers = []

    for var in CLUSTERING_VARS:
        z_var = f"{var}_z"
        z_mean = df_z[z_var].mean()
        z_sd = df_z[z_var].std()
        z_min = df_z[z_var].min()
        z_max = df_z[z_var].max()

        z_stats[z_var] = {
            'mean': z_mean,
            'sd': z_sd,
            'min': z_min,
            'max': z_max
        }

        print(f"  {z_var}:")
        print(f"    Mean: {z_mean:.6f} (expected: ~0)")
        print(f"    SD:   {z_sd:.6f} (expected: ~1)")
        print(f"    Range: [{z_min:.3f}, {z_max:.3f}]")

        # Validate mean ~ 0
        if abs(z_mean) > 0.01:
            print(f"    ERROR: Mean not close to 0")
            sys.exit(1)

        # Validate SD ~ 1 (allowing for finite sample variance)
        if abs(z_sd - 1.0) > 0.05:
            print(f"    ERROR: SD not close to 1")
            sys.exit(1)

        # Check for outliers
        extreme = df_z[abs(df_z[z_var]) > 3][['UID', z_var]]
        for _, row in extreme.iterrows():
            outliers.append({
                'UID': row['UID'],
                'variable': z_var,
                'z_score': row[z_var]
            })
            severity = "EXTREME" if abs(row[z_var]) > 4 else "unusual"
            print(f"    Outlier ({severity}): UID={row['UID']}, z={row[z_var]:.3f}")

    print(f"\nValidation - PASS: All means within [-0.01, 0.01]")
    print(f"Validation - PASS: All SDs within [0.95, 1.05]")
    print(f"Standardized 4 variables to z-scores")

    # Check for extreme outliers (|z| > 4)
    extreme_outliers = [o for o in outliers if abs(o['z_score']) > 4]
    if extreme_outliers:
        print(f"\nWARNING: {len(extreme_outliers)} extreme outliers (|z| > 4) detected")
        for o in extreme_outliers:
            print(f"  {o['UID']}: {o['variable']} = {o['z_score']:.3f}")
        # Don't fail - document but proceed

    # Save standardized features
    output_file = DATA_DIR / "step01_standardized_features.csv"
    df_z.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  - Rows: {len(df_z)}")
    print(f"  - Columns: {list(df_z.columns)}")

    # Save summary report
    summary_file = DATA_DIR / "step01_standardization_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RQ 5.2.7 Step 01: Standardization Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Raw Variable Statistics (Before Standardization)\n")
        f.write("-" * 50 + "\n")
        for var in CLUSTERING_VARS:
            f.write(f"\n{var}:\n")
            f.write(f"  Mean: {raw_stats[var]['mean']:.6f}\n")
            f.write(f"  SD:   {raw_stats[var]['sd']:.6f}\n")
            f.write(f"  Range: [{raw_stats[var]['min']:.6f}, {raw_stats[var]['max']:.6f}]\n")

        f.write("\n\nZ-Scored Variable Statistics (After Standardization)\n")
        f.write("-" * 50 + "\n")
        for var in CLUSTERING_VARS:
            z_var = f"{var}_z"
            f.write(f"\n{z_var}:\n")
            f.write(f"  Mean: {z_stats[z_var]['mean']:.6f}\n")
            f.write(f"  SD:   {z_stats[z_var]['sd']:.6f}\n")
            f.write(f"  Range: [{z_stats[z_var]['min']:.6f}, {z_stats[z_var]['max']:.6f}]\n")

        f.write("\n\nOutliers (|z| > 3)\n")
        f.write("-" * 50 + "\n")
        if outliers:
            for o in outliers:
                f.write(f"  {o['UID']}: {o['variable']} = {o['z_score']:.4f}\n")
        else:
            f.write("  None detected\n")

        f.write("\n\nValidation Status: PASS\n")
        f.write("  - All means within [-0.01, 0.01]\n")
        f.write("  - All SDs within [0.95, 1.05]\n")
        f.write("  - No extreme outliers (|z| > 4) detected\n" if not extreme_outliers
                else f"  - {len(extreme_outliers)} extreme outliers detected (proceed with caution)\n")

    print(f"Saved: {summary_file}")

    print("\n" + "=" * 60)
    print("Step 01 COMPLETE: Features standardized successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
