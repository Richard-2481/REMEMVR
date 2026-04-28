#!/usr/bin/env python3
"""standardize_scores: Convert RAVLT and REMEMVR scores to z-scores for fair comparison across tests"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step02_standardize_scores.log"
INPUT_FILE = RQ_DIR / "data" / "step01_cognitive_scores.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step02_standardized_scores.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: standardize_scores")
        # Load Cognitive Scores
        log("Loading cognitive scores from step01...")

        df = pd.read_csv(INPUT_FILE)
        log(f"{INPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")
        # Compute Z-Scores
        log("Computing z-scores for RAVLT and REMEMVR...")

        # Z-score for RAVLT_Total
        # nan_policy='omit' excludes NaN from calculation but preserves them in output
        df['RAVLT_z'] = zscore(df['RAVLT_Total'], axis=0, nan_policy='omit')
        log(f"RAVLT_z: mean={df['RAVLT_z'].mean():.6f}, std={df['RAVLT_z'].std():.6f}")

        # Z-score for REMEMVR_theta
        # Note: REMEMVR_theta already approximately standardized (IRT scale ~N(0,1))
        # but re-standardizing ensures identical properties for classification
        df['REMEMVR_z'] = zscore(df['REMEMVR_theta'], axis=0, nan_policy='omit')
        log(f"REMEMVR_z: mean={df['REMEMVR_z'].mean():.6f}, std={df['REMEMVR_z'].std():.6f}")

        # Z-score for RAVLT_Pct_Ret (percent retention)
        df['RAVLT_Pct_Ret_z'] = zscore(df['RAVLT_Pct_Ret'], axis=0, nan_policy='omit')
        log(f"RAVLT_Pct_Ret_z: mean={df['RAVLT_Pct_Ret_z'].mean():.6f}, std={df['RAVLT_Pct_Ret_z'].std():.6f}")

        # Check for any missing z-scores
        missing_ravlt_z = df['RAVLT_z'].isnull().sum()
        missing_rememvr_z = df['REMEMVR_z'].isnull().sum()
        missing_pct_ret_z = df['RAVLT_Pct_Ret_z'].isnull().sum()

        if missing_ravlt_z > 0:
            log(f"{missing_ravlt_z} participants missing RAVLT_z scores")
        if missing_rememvr_z > 0:
            log(f"{missing_rememvr_z} participants missing REMEMVR_z scores")
        if missing_pct_ret_z > 0:
            log(f"{missing_pct_ret_z} participants missing RAVLT_Pct_Ret_z scores")

        if missing_ravlt_z == 0 and missing_rememvr_z == 0 and missing_pct_ret_z == 0:
            log(f"No missing z-scores")
        # Save Standardized Scores
        log("Saving standardized scores...")

        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")
        # Validate Standardization
        log("Running validate_standardization...")

        check_columns = ['RAVLT_z', 'REMEMVR_z', 'RAVLT_Pct_Ret_z']
        validation_result = validate_standardization(
            df=df,
            column_names=check_columns,
            tolerance=0.01
        )

        if validation_result.get('valid', False):
            log(f"Standardization validation passed")

            # Report detailed statistics
            for col in check_columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                log(f"{col}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.3f}, {max_val:.3f}]")
        else:
            log(f"Standardization validation failed: {validation_result.get('message', 'Unknown error')}")
            raise ValueError(f"Standardization validation failed: {validation_result}")

        # Additional manual checks (for transparency)
        log("Manual standardization checks...")

        for col in check_columns:
            mean_val = df[col].mean()
            std_val = df[col].std(ddof=0)  # Population std (same as scipy.stats.zscore uses)

            # Check mean ≈ 0
            if abs(mean_val) > 0.01:
                log(f"{col} mean={mean_val:.6f} deviates from 0.0 (tolerance=0.01)")
            else:
                log(f"{col} mean={mean_val:.6f} within tolerance")

            # Check std ≈ 1
            if abs(std_val - 1.0) > 0.01:
                log(f"{col} std={std_val:.6f} deviates from 1.0 (tolerance=0.01)")
            else:
                log(f"{col} std={std_val:.6f} within tolerance")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
