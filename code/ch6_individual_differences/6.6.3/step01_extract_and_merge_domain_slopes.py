#!/usr/bin/env python3
"""extract_and_merge_domain_slopes: Load participant-level slopes from Ch5 5.2.6 random effects and create unified dataset."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.3
LOG_FILE = RQ_DIR / "logs" / "step01_extract_and_merge_domain_slopes.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_domain_slopes.csv"

# Input files
CH5_RANDOM_EFFECTS = PROJECT_ROOT / "results" / "ch5" / "5.2.6" / "data" / "step04_random_effects.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract and Merge Domain Slopes")
        # Load Ch5 5.2.6 Random Effects Data

        log(f"Reading Ch5 5.2.6 random effects...")
        df_random_effects = pd.read_csv(CH5_RANDOM_EFFECTS)
        log(f"{len(df_random_effects)} rows, {len(df_random_effects.columns)} columns")

        # Verify expected domains present
        domains = df_random_effects['domain'].unique()
        log(f"Found: {sorted(domains)}")

        if 'When' in domains:
            log(f"When domain present but will be excluded from analysis")
            # Filter to What and Where only
            df_random_effects = df_random_effects[df_random_effects['domain'].isin(['What', 'Where'])]
            log(f"Retained {len(df_random_effects)} rows (What + Where only)")
        # Pivot to Wide Format
        # Transform: Long format (domain column) -> Wide format (slope_what, slope_where)

        log(f"Pivoting to wide format...")

        # Extract slope data only
        df_slopes = df_random_effects[['UID', 'domain', 'Total_Slope']].copy()

        # Pivot: domain values become columns
        df_wide = df_slopes.pivot(index='UID', columns='domain', values='Total_Slope')
        df_wide.columns = [f'slope_{col.lower()}' for col in df_wide.columns]  # slope_what, slope_where
        df_wide = df_wide.reset_index()

        log(f"{len(df_wide)} participants, {len(df_wide.columns)} columns")
        log(f"{df_wide.columns.tolist()}")
        # Validate Complete Cases
        # Check: No missing slopes, no duplicates

        log(f"Checking data quality...")

        # Check for missing values
        missing_counts = df_wide.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            log(f"Missing values detected:")
            for col, count in missing_counts.items():
                if count > 0:
                    log(f"  {col}: {count} missing")

            # Drop incomplete cases
            df_wide = df_wide.dropna()
            log(f"Retained {len(df_wide)} complete cases")
        else:
            log(f"No missing values")

        # Check for duplicates
        n_duplicates = df_wide['UID'].duplicated().sum()
        if n_duplicates > 0:
            log(f"{n_duplicates} duplicate UIDs found")
            sys.exit(1)
        else:
            log(f"No duplicate UIDs")

        # Check final count
        if len(df_wide) != 100:
            log(f"Expected 100 participants, got {len(df_wide)}")
            sys.exit(1)
        else:
            log(f"Exactly 100 participants with complete data")
        # Validate Slope Ranges
        # Typical range: [-0.8, 0.2] for memory slopes

        log(f"Checking slope value ranges...")

        for col in ['slope_what', 'slope_where']:
            slope_min = df_wide[col].min()
            slope_max = df_wide[col].max()
            slope_mean = df_wide[col].mean()

            log(f"[{col.upper()}] Range: [{slope_min:.4f}, {slope_max:.4f}], Mean: {slope_mean:.4f}")

            # Check range
            if slope_min < -0.8 or slope_max > 0.2:
                log(f"{col} values outside typical range [-0.8, 0.2]")
                log(f"This may indicate extreme individual differences - acceptable")
        # Save Merged Domain Slopes
        # Output: Wide format CSV for ICC calculation
        # Downstream use: step02 (compute ICC for each domain)

        log(f"Writing domain slopes...")
        df_wide.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_wide)} rows, {len(df_wide.columns)} cols)")
        # Validation with tools.validation.validate_data_columns
        # Verify: Required columns present

        log(f"Running validate_data_columns...")
        required_cols = ['UID', 'slope_what', 'slope_where']

        validation_result = validate_data_columns(
            df=df_wide,
            required_columns=required_cols
        )

        if not validation_result.get('valid', False):
            log(f"Validation failed: {validation_result}")
            sys.exit(1)
        else:
            log(f"All required columns present")

        log(f"Step 01 complete")
        log(f"100 participants with complete slope data for What + Where domains")
        log(f"Proceed to step02 (compute ICC for each domain)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
