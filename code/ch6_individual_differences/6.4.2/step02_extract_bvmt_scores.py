#!/usr/bin/env python3
"""Extract BVMT Scores: Extract BVMT scores from dfnonvr.csv for domain-specificity analysis."""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_bvmt_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract BVMT Scores")
        # Load Input Data

        log("Loading cognitive test data from dfnonvr.csv...")
        # Load dfnonvr.csv - Ch7 uses prepared data, not master.xlsx
        # Expected columns: UID, bvmt-total-recall (among 235 total columns)
        # Expected rows: ~100 participants
        cognitive_df = pd.read_csv(PROJECT_ROOT / "data" / "dfnonvr.csv")
        log(f"dfnonvr.csv ({len(cognitive_df)} rows, {len(cognitive_df.columns)} cols)")
        
        # Verify required columns exist
        required_cols = ["UID", "bvmt-trial-1-score", "bvmt-trial-2-score", "bvmt-trial-3-score", "bvmt-percent-retained"]
        missing_cols = [col for col in required_cols if col not in cognitive_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Required columns present: {required_cols}")
        # Compute BVMT Total from Individual Trials
        # Recompute BVMT total as sum of trials 1-3 (more transparent than pre-computed column)
        # Also extract bvmt-percent-retained for retention-based prediction

        log("Computing BVMT Total from trials 1-3...")

        bvmt_df = cognitive_df[["UID"]].copy()

        # Sum trials 1-3 to get bvmt_total
        bvmt_df["bvmt_total"] = (
            cognitive_df["bvmt-trial-1-score"] +
            cognitive_df["bvmt-trial-2-score"] +
            cognitive_df["bvmt-trial-3-score"]
        )

        # Verify against pre-computed column if available
        if "bvmt-total-recall" in cognitive_df.columns:
            precomputed = cognitive_df["bvmt-total-recall"]
            mismatch = (bvmt_df["bvmt_total"] != precomputed).sum()
            log(f"Recomputed vs pre-computed bvmt-total-recall: {mismatch} mismatches out of {len(bvmt_df)}")
            if mismatch > 0:
                log(f"Mismatches found! Max difference: {(bvmt_df['bvmt_total'] - precomputed).abs().max():.2f}")

        log(f"bvmt_total = trial1 + trial2 + trial3 for {len(bvmt_df)} participants")

        # Extract percent retention
        bvmt_df["bvmt_pct_ret"] = cognitive_df["bvmt-percent-retained"].values
        log(f"bvmt_pct_ret from 'bvmt-percent-retained'")

        # Check for missing values
        missing_total = bvmt_df["bvmt_total"].isnull().sum()
        missing_pct_ret = bvmt_df["bvmt_pct_ret"].isnull().sum()
        missing_pct = (max(missing_total, missing_pct_ret) / len(bvmt_df)) * 100
        log(f"bvmt_total: {missing_total} missing; bvmt_pct_ret: {missing_pct_ret} missing ({missing_pct:.1f}%)")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 (merge with domain theta scores)

        output_path = RQ_DIR / "data" / "step02_bvmt_scores.csv"
        log(f"Saving {output_path}...")
        # Output: step02_bvmt_scores.csv
        # Contains: UID, recomputed BVMT total, and BVMT percent retention
        # Columns: ["UID", "bvmt_total", "bvmt_pct_ret"]
        bvmt_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(bvmt_df)} rows, {len(bvmt_df.columns)} cols)")
        # Run Validation Tool
        # Validates: BVMT scores in valid range [0, 36] with adequate variance
        # Threshold: Standard deviation > 2.0, missing data < 5%

        log("Running validate_numeric_range...")

        # Remove missing values for range validation
        bvmt_scores_clean = bvmt_df["bvmt_total"].dropna()
        bvmt_pct_ret_clean = bvmt_df["bvmt_pct_ret"].dropna()

        if len(bvmt_scores_clean) == 0:
            raise ValueError("No valid BVMT total scores found after removing missing values")
        if len(bvmt_pct_ret_clean) == 0:
            raise ValueError("No valid BVMT percent retention scores found after removing missing values")

        # Validate BVMT total scores are in valid range [0, 36]
        validation_result = validate_numeric_range(
            data=bvmt_scores_clean,
            min_val=0.0,
            max_val=36.0,
            column_name="bvmt_total"
        )

        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"bvmt_total {key}: {value}")
        else:
            log(f"bvmt_total: {validation_result}")

        # Validate BVMT percent retention (0-200+ range; can exceed 100 if delayed > immediate)
        validation_result_pct = validate_numeric_range(
            data=bvmt_pct_ret_clean,
            min_val=0.0,
            max_val=200.0,
            column_name="bvmt_pct_ret"
        )

        if isinstance(validation_result_pct, dict):
            for key, value in validation_result_pct.items():
                log(f"bvmt_pct_ret {key}: {value}")
        else:
            log(f"bvmt_pct_ret: {validation_result_pct}")

        # Additional validation checks
        n_participants = len(bvmt_df)
        n_valid_total = len(bvmt_scores_clean)
        n_valid_pct = len(bvmt_pct_ret_clean)
        missing_percentage = (1 - min(n_valid_total, n_valid_pct) / n_participants) * 100

        log(f"Participants with bvmt_total: {n_valid_total}/{n_participants}")
        log(f"Participants with bvmt_pct_ret: {n_valid_pct}/{n_participants}")
        log(f"Missing data percentage: {missing_percentage:.1f}%")

        if n_valid_total < 100:
            log(f"Expected 100 participants with bvmt_total, found {n_valid_total}")
        if n_valid_pct < 100:
            log(f"Expected 100 participants with bvmt_pct_ret, found {n_valid_pct}")

        if missing_percentage > 5.0:
            raise ValueError(f"Missing data percentage ({missing_percentage:.1f}%) exceeds 5% threshold")

        # Check adequate variance (standard deviation > 2.0)
        bvmt_std = bvmt_scores_clean.std()
        pct_ret_std = bvmt_pct_ret_clean.std()
        log(f"bvmt_total SD: {bvmt_std:.2f}")
        log(f"bvmt_pct_ret SD: {pct_ret_std:.2f}")

        if bvmt_std <= 2.0:
            log(f"bvmt_total SD ({bvmt_std:.2f}) may indicate insufficient variance")

        # Check score ranges
        bvmt_min, bvmt_max = bvmt_scores_clean.min(), bvmt_scores_clean.max()
        pct_min, pct_max = bvmt_pct_ret_clean.min(), bvmt_pct_ret_clean.max()
        log(f"bvmt_total range: [{bvmt_min:.1f}, {bvmt_max:.1f}]")
        log(f"bvmt_pct_ret range: [{pct_min:.1f}, {pct_max:.1f}]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)