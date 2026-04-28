#!/usr/bin/env python3
"""Prepare Plot Data: Prepare plot-ready data for plotting pipeline. Create CSV with profile"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_plot_data.log"
OUTPUT_DIR = RQ_DIR / "data"

# Input
INPUT_CHARACTERISTICS = OUTPUT_DIR / 'step04_profile_characteristics.csv'

# Output
OUTPUT_PLOT_DATA = OUTPUT_DIR / 'lpa_plot_data.csv'

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def ci_to_se(ci_low, ci_high, ci_level=0.95):
    """
    Convert 95% confidence interval to standard error.

    Formula: SE = (CI_high - CI_low) / (2 * z_critical)

    For 95% CI: z_critical = 1.96
    Therefore: SE = (CI_high - CI_low) / 3.92

    This assumes normal distribution of the mean.
    """
    z_critical = 1.96  # For 95% CI
    ci_width = ci_high - ci_low
    se = ci_width / (2 * z_critical)

    return se

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Prepare Plot Data")
        # Load Profile Characteristics

        log("\nLoading profile characteristics...")
        log(f"{INPUT_CHARACTERISTICS}")

        df_characteristics = pd.read_csv(INPUT_CHARACTERISTICS)
        log(f"{len(df_characteristics)} profiles")
        log(f"{df_characteristics.columns.tolist()}")

        # Scientific Mantra Checkpoint
        log("\nInput validation")
        expected_cols = ['Profile', 'N', 'What_Mean', 'What_SD', 'What_CI_Low', 'What_CI_High',
                        'Where_Mean', 'Where_SD', 'Where_CI_Low', 'Where_CI_High']

        missing_cols = [col for col in expected_cols if col not in df_characteristics.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"All required columns present")

        log(f"\nProfile characteristics:")
        log(f"{df_characteristics.to_string(index=False)}")
        # Compute Standard Errors from Bootstrap CIs

        log("\nConverting 95% CIs to standard errors...")

        # What domain SE
        df_characteristics['What_SE'] = df_characteristics.apply(
            lambda row: ci_to_se(row['What_CI_Low'], row['What_CI_High']),
            axis=1
        )

        # Where domain SE
        df_characteristics['Where_SE'] = df_characteristics.apply(
            lambda row: ci_to_se(row['Where_CI_Low'], row['Where_CI_High']),
            axis=1
        )

        log(f"Standard errors from bootstrap CIs")
        log(f"What SE range: [{df_characteristics['What_SE'].min():.4f}, {df_characteristics['What_SE'].max():.4f}]")
        log(f"Where SE range: [{df_characteristics['Where_SE'].min():.4f}, {df_characteristics['Where_SE'].max():.4f}]")

        # Scientific Mantra Checkpoint
        log("\nSE computation validation")

        # Check SEs are positive
        if (df_characteristics['What_SE'] <= 0).any():
            raise ValueError("What_SE contains non-positive values")
        if (df_characteristics['Where_SE'] <= 0).any():
            raise ValueError("Where_SE contains non-positive values")

        log(f"All SEs are positive")
        # Format Plot Data

        log("\nCreating plot-ready dataset...")

        # Select columns for plotting (plotting pipeline expects: Profile, N, *_Mean, *_SE)
        df_plot = df_characteristics[['Profile', 'N', 'What_Mean', 'What_SE', 'Where_Mean', 'Where_SE']].copy()

        log(f"Plot data with {len(df_plot)} profiles")
        log(f"{df_plot.columns.tolist()}")
        # Save Plot Data

        log(f"\nSaving plot data to {OUTPUT_PLOT_DATA}")

        df_plot.to_csv(OUTPUT_PLOT_DATA, index=False, encoding='utf-8')

        log(f"{len(df_plot)} rows, {len(df_plot.columns)} columns")
        log(f"\n[PLOT DATA TABLE]")
        log(f"{df_plot.to_string(index=False)}")
        # VALIDATION: Final Checks

        log("\nFinal output validation...")

        # Check 1: Row count matches input
        if len(df_plot) != len(df_characteristics):
            raise ValueError(f"Row count mismatch: {len(df_plot)} != {len(df_characteristics)}")
        log(f"Row count preserved ({len(df_plot)} profiles)")

        # Check 2: No missing values
        missing_count = df_plot.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"Found {missing_count} missing values in plot data")
        log(f"No missing values")

        # Check 3: Profile IDs are sequential (0, 1, 2, ...)
        expected_profiles = list(range(len(df_plot)))
        actual_profiles = sorted(df_plot['Profile'].unique())

        if actual_profiles != expected_profiles:
            log(f"Non-sequential profile IDs: {actual_profiles} (expected: {expected_profiles})")
        else:
            log(f"Profile IDs are sequential")

        # Check 4: SE values reasonable (should be < SD for bootstrap)
        for idx, row in df_plot.iterrows():
            profile_id = row['Profile']
            what_sd = df_characteristics.loc[idx, 'What_SD']
            where_sd = df_characteristics.loc[idx, 'Where_SD']

            # SE should typically be smaller than SD (SD is population, SE is sampling distribution)
            # But with bootstrap, SE can sometimes be close to SD (especially for small N)
            # Just check they're in reasonable range (not orders of magnitude different)

            if row['What_SE'] > 2 * what_sd:
                log(f"Profile {profile_id}: What_SE ({row['What_SE']:.4f}) > 2*SD ({what_sd:.4f})")

            if row['Where_SE'] > 2 * where_sd:
                log(f"Profile {profile_id}: Where_SE ({row['Where_SE']:.4f}) > 2*SD ({where_sd:.4f})")

        log(f"SE values in reasonable range")
        # SUMMARY: Plot Data Description

        log("\nPlot data ready for plotting pipeline")
        log(f"Profiles: {len(df_plot)}")
        log(f"Domains: What, Where (2 dimensions)")
        log(f"Error bars: Bootstrap-derived standard errors")
        log(f"Sample sizes included for each profile")

        log("\nThis file can be used by plotting pipeline to create:")
        log("  1. Profile centroid plot (What vs Where means)")
        log("  2. Error bar plot (means ± SE)")
        log("  3. Profile size comparison (N per profile)")

        log("\nStep 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
