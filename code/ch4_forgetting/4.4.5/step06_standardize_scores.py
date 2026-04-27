#!/usr/bin/env python3
"""
Step 06: Z-Standardize All Measurements

Z-standardizes IRT theta, Full CTT, and Purified CTT scores for LMM coefficient comparability.
All scores transformed to mean=0, SD=1.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_standardize_scores.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 06: Z-Standardize All Measurements")

        # Load theta scores from RQ 5.4.1
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step03_theta_scores.csv"
        log(f"[LOAD] Reading {theta_path}")
        theta_scores = pd.read_csv(theta_path, encoding='utf-8')

        # Load Full CTT scores
        ctt_full_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        log(f"[LOAD] Reading {ctt_full_path}")
        ctt_full_scores = pd.read_csv(ctt_full_path, encoding='utf-8')

        # Load Purified CTT scores
        ctt_purified_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        log(f"[LOAD] Reading {ctt_purified_path}")
        ctt_purified_scores = pd.read_csv(ctt_purified_path, encoding='utf-8')

        # Load TSVR mapping
        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step00_tsvr_mapping.csv"
        log(f"[LOAD] Reading {tsvr_path}")
        tsvr_mapping = pd.read_csv(tsvr_path, encoding='utf-8')

        # Merge all datasets
        log("[MERGE] Merging all datasets on composite_ID")
        merged = theta_scores.merge(ctt_full_scores, on='composite_ID', how='inner')
        merged = merged.merge(ctt_purified_scores, on='composite_ID', how='inner')
        merged = merged.merge(tsvr_mapping[['composite_ID', 'TSVR_hours']], on='composite_ID', how='inner')
        log(f"[MERGED] {len(merged)} rows retained")

        # Identify score columns to standardize (9 total)
        score_columns = [
            'theta_common', 'theta_congruent', 'theta_incongruent',
            'ctt_full_common', 'ctt_full_congruent', 'ctt_full_incongruent',
            'ctt_purified_common', 'ctt_purified_congruent', 'ctt_purified_incongruent'
        ]

        log(f"[STANDARDIZE] Z-standardizing {len(score_columns)} score columns")

        # Z-standardize each column
        tolerance_mean = 0.01
        tolerance_sd = 0.01

        for col in score_columns:
            mean_val = merged[col].mean()
            sd_val = merged[col].std(ddof=1)

            # Z-standardize
            z_col = f'z_{col}'
            merged[z_col] = (merged[col] - mean_val) / sd_val

            # Verify standardization
            z_mean = merged[z_col].mean()
            z_sd = merged[z_col].std(ddof=1)

            log(f"  {col}: mean={mean_val:.3f}, SD={sd_val:.3f}")
            log(f"    -> {z_col}: mean={z_mean:.6f}, SD={z_sd:.6f}")

            # Check tolerance
            if abs(z_mean) > tolerance_mean:
                raise ValueError(f"{z_col} mean {z_mean:.6f} exceeds tolerance {tolerance_mean}")
            if abs(z_sd - 1.0) > tolerance_sd:
                raise ValueError(f"{z_col} SD {z_sd:.6f} deviates from 1.0 by more than {tolerance_sd}")

        log("[PASS] All z-scores have mean ~0 and SD ~1 within tolerance")

        # Select columns for output
        output_columns = ['composite_ID'] + [f'z_{col}' for col in score_columns] + ['TSVR_hours']
        standardized_scores = merged[output_columns]

        # Validation: Check for NaN values
        log("[VALIDATION] Checking for NaN values")
        nan_count = standardized_scores.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in standardized scores")
        log("[PASS] No NaN values found")

        # Validation: Check expected N
        log(f"[VALIDATION] Checking N = 400")
        if len(standardized_scores) != 400:
            log(f"[WARNING] Expected 400 rows, got {len(standardized_scores)}")

        # Save results
        output_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        log(f"[SAVE] Writing {output_path}")
        standardized_scores.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {len(standardized_scores)} rows, {len(standardized_scores.columns)} columns")

        log("[SUCCESS] Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
