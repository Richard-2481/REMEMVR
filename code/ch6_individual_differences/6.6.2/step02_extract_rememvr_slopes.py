#!/usr/bin/env python3
"""extract_rememvr_slopes: Extract participant-specific forgetting slopes from Ch5 5.1.4 LMM random effects."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_rememvr_slopes.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract REMEMVR Slopes from Ch5 5.1.4")
        # Load Input Data from Ch5 5.1.4
        # CRITICAL: Using Ch5 5.1.4 (NOT 5.1.1) - corrected per user instruction

        log("Loading random effects from Ch5 5.1.4...")
        log("CRITICAL FIX: Using Ch5 5.1.4 (NOT 5.1.1) - proven working in RQ 7.6.1")

        input_path = PROJECT_ROOT / 'results' / 'ch5' / '5.1.4' / 'data' / 'step04_random_effects.csv'

        if not input_path.exists():
            log(f"Input file not found: {input_path}")
            log("Ch5 5.1.4 must be completed first")
            log("Expected: results/ch5/5.1.4/data/step04_random_effects.csv")
            sys.exit(1)

        df_random_effects = pd.read_csv(input_path, encoding='utf-8')
        log(f"step04_random_effects.csv ({len(df_random_effects)} rows, {len(df_random_effects.columns)} cols)")

        # Verify expected columns
        expected_cols = ['UID', 'random_intercept', 'random_slope']
        missing_cols = [col for col in expected_cols if col not in df_random_effects.columns]
        if missing_cols:
            log(f"Missing expected columns: {missing_cols}")
            log(f"Available columns: {df_random_effects.columns.tolist()}")
            sys.exit(1)

        log(f"All expected columns present: {expected_cols}")
        # Extract and Rename Columns
        # Extract only the columns we need and rename for clarity
        # random_slope -> REMEMVR_Slope (participant-specific forgetting trajectory)
        # random_intercept -> REMEMVR_Intercept (initial encoding level)

        log("Extracting random slopes and intercepts...")

        df_slopes = df_random_effects[['UID', 'random_slope', 'random_intercept']].copy()

        # Rename for clarity in downstream correlation analyses
        df_slopes = df_slopes.rename(columns={
            'random_slope': 'REMEMVR_Slope',
            'random_intercept': 'REMEMVR_Intercept'
        })

        log(f"Columns renamed to: {df_slopes.columns.tolist()}")
        # Data Quality Checks
        # Verify random effects are within expected bounds

        log("Verifying data quality...")

        # Check for missing values
        missing_counts = df_slopes.isnull().sum()
        if missing_counts.any():
            log("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                log(f"  - {col}: {count} missing values")
        else:
            log("No missing values in slope/intercept columns")

        # Check participant count
        n_participants = len(df_slopes)
        log(f"Participant count: {n_participants}")
        if n_participants != 100:
            log(f"Expected 100 participants, found {n_participants}")

        # Descriptive statistics
        log("REMEMVR_Slope: mean={:.4f}, std={:.4f}, range=[{:.4f}, {:.4f}]".format(
            df_slopes['REMEMVR_Slope'].mean(),
            df_slopes['REMEMVR_Slope'].std(),
            df_slopes['REMEMVR_Slope'].min(),
            df_slopes['REMEMVR_Slope'].max()
        ))
        log("REMEMVR_Intercept: mean={:.4f}, std={:.4f}, range=[{:.4f}, {:.4f}]".format(
            df_slopes['REMEMVR_Intercept'].mean(),
            df_slopes['REMEMVR_Intercept'].std(),
            df_slopes['REMEMVR_Intercept'].min(),
            df_slopes['REMEMVR_Intercept'].max()
        ))

        # Interpretation notes
        log("Slope interpretation:")
        log("- Negative slopes = forgetting over time")
        log("- Slopes near 0 = stable retention")
        log("- Positive slopes = improvement over time (rare)")

        n_negative = (df_slopes['REMEMVR_Slope'] < 0).sum()
        n_positive = (df_slopes['REMEMVR_Slope'] > 0).sum()
        log(f"Slope distribution: {n_negative} negative (forgetting), {n_positive} positive (improvement)")
        # Save Analysis Output
        # Save REMEMVR slopes for downstream correlation with RAVLT forgetting

        output_path = RQ_DIR / "data" / "step02_rememvr_slopes.csv"
        log(f"Saving REMEMVR slopes to: {output_path}")

        df_slopes.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_rememvr_slopes.csv ({len(df_slopes)} rows, {len(df_slopes.columns)} cols)")
        # Run Validation Tool
        # Validate slopes are within reasonable bounds

        log("Running validate_numeric_range on REMEMVR_Slope...")

        validation_result = validate_numeric_range(
            data=df_slopes['REMEMVR_Slope'],
            min_val=-0.5,  # Extreme forgetting (unlikely but possible)
            max_val=0.1,   # Slight improvement over time (rare)
            column_name='REMEMVR_Slope'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

            # Check if validation passed
            if validation_result.get('valid', False):
                log("PASS - Slopes within expected range [-0.5, 0.1]")
            else:
                log("FAIL - Some slopes outside expected range")
                # Note: This is a warning, not necessarily fatal (extreme cases may exist)
        else:
            log(f"{validation_result}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
