#!/usr/bin/env python3
"""extract_ravlt_scores: Apply RAVLT ceiling fix, then compute forgetting index (Trial 5 - Delayed Recall)"""

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
LOG_FILE = RQ_DIR / "logs" / "step01_extract_ravlt_scores.log"

TRIAL_COLS = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
DR_COL = 'ravlt-delayed-recall-score'


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Ceiling Fix Function

def apply_ravlt_ceiling_fix(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials stored as 0.

    Logic: If a participant scored 14 or 15 on trial N, and trial N+1 is 0,
    the 0 likely represents an unadministered trial (ceiling reached), not a true score.
    Fix: set trial N+1 = 15.

    Applied sequentially from trial 1->2, 2->3, 3->4, 4->5 so that a fix
    on trial 2 can cascade to trial 3 if needed.
    """
    fixes_applied = 0
    for idx in df.index:
        for i in range(len(TRIAL_COLS) - 1):  # Check trials 2-5 against predecessor
            prev_col = TRIAL_COLS[i]
            current_col = TRIAL_COLS[i + 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                old_val = df.at[idx, current_col]
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} {old_val} -> 15 "
                       f"(prev trial {prev_col} = {df.at[idx, prev_col]})")
    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract RAVLT Scores, Ceiling Fix, Forgetting Index + Percent Retention")
        # Load Input Data

        log("Loading participant data from Step 00...")
        input_path = RQ_DIR / "data" / "step00_participant_data.csv"

        if not input_path.exists():
            log(f"Input file not found: {input_path}")
            log("Run step00_load_participant_data.py first")
            sys.exit(1)

        df_ravlt = pd.read_csv(input_path, encoding='utf-8')
        log(f"step00_participant_data.csv ({len(df_ravlt)} rows, {len(df_ravlt.columns)} cols)")

        # Verify expected columns
        expected_cols = ['UID'] + TRIAL_COLS + missing_cols = [col for col in expected_cols if col not in df_ravlt.columns]
        if missing_cols:
            log(f"Missing expected columns: {missing_cols}")
            sys.exit(1)
        # Apply RAVLT Ceiling Fix
        # Fix unadministered trials stored as 0 when previous trial >= 14
        # MUST be applied BEFORE computing forgetting index or percent retention

        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        log("[CEILING FIX] Rule: if trial N == 0 and trial N-1 >= 14, set trial N = 15")

        # Log pre-fix values for affected participants
        for col in TRIAL_COLS:
            zeros = df_ravlt[df_ravlt[col] == 0]
            if len(zeros) > 0:
                for _, row in zeros.iterrows():
                    log(f"[PRE-FIX] {row['UID']}: {col} = 0")

        df_ravlt = apply_ravlt_ceiling_fix(df_ravlt, log)
        # Compute Forgetting Index (post-ceiling-fix)
        # Forgetting = Trial 5 (encoding) - Delayed Recall (retention)
        # Positive values = forgetting, Negative values = retention > encoding

        log("Calculating RAVLT_Forgetting (Trial 5 - Delayed Recall) [post-ceiling-fix]...")

        df_ravlt['RAVLT_Forgetting'] = (
            df_ravlt['ravlt-trial-5-score'] - df_ravlt)

        log(f"RAVLT_Forgetting column added")
        log(f"Forgetting score: mean={df_ravlt['RAVLT_Forgetting'].mean():.2f}, "
            f"std={df_ravlt['RAVLT_Forgetting'].std():.2f}, "
            f"range=[{df_ravlt['RAVLT_Forgetting'].min():.0f}, {df_ravlt['RAVLT_Forgetting'].max():.0f}]")

        # Check for negative forgetting (retention > encoding)
        n_negative = (df_ravlt['RAVLT_Forgetting'] < 0).sum()
        if n_negative > 0:
            log(f"{n_negative} participants have negative forgetting (retention > encoding)")
            log(f"This can occur due to practice effects or measurement noise")
        else:
            log("No negative forgetting values detected")
        # Compute Percent Retention (post-ceiling-fix)
        # RAVLT_Pct_Ret = (Delayed Recall / Best Learning Trial) * 100
        # Best Learning Trial = max of trials 1-5 (after ceiling fix)
        # If best == 0, set to NaN (cannot compute ratio)

        log("Calculating RAVLT_Pct_Ret (DR / best_learning_trial * 100) [post-ceiling-fix]...")

        best_learning = df_ravlt.max(axis=1)
        log(f"Best learning trial: mean={best_learning.mean():.2f}, "
            f"range=[{best_learning.min():.0f}, {best_learning.max():.0f}]")

        # Compute percent retention, handling division by zero
        df_ravlt['RAVLT_Pct_Ret'] = np.where(
            best_learning > 0,
            (df_ravlt/ best_learning) * 100,
            np.nan
        )

        n_valid_pct = df_ravlt['RAVLT_Pct_Ret'].notna().sum()
        n_nan_pct = df_ravlt['RAVLT_Pct_Ret'].isna().sum()
        log(f"RAVLT_Pct_Ret: {n_valid_pct} valid, {n_nan_pct} NaN (best_trial == 0)")
        log(f"Pct Retention: mean={df_ravlt['RAVLT_Pct_Ret'].mean():.2f}, "
            f"std={df_ravlt['RAVLT_Pct_Ret'].std():.2f}, "
            f"range=[{df_ravlt['RAVLT_Pct_Ret'].min():.1f}, {df_ravlt['RAVLT_Pct_Ret'].max():.1f}]")
        # Standardize to Z-Scores
        # Convert raw scores to standardized z-scores using sample mean and SD

        log("Standardizing RAVLT_Forgetting to z-scores...")
        df_ravlt['RAVLT_Forgetting_z'] = (
            (df_ravlt['RAVLT_Forgetting'] - df_ravlt['RAVLT_Forgetting'].mean()) /
            df_ravlt['RAVLT_Forgetting'].std()
        )
        log(f"RAVLT_Forgetting_z: mean={df_ravlt['RAVLT_Forgetting_z'].mean():.4f}, "
            f"std={df_ravlt['RAVLT_Forgetting_z'].std():.4f}")

        log("Standardizing RAVLT_Pct_Ret to z-scores...")
        df_ravlt['RAVLT_Pct_Ret_z'] = (
            (df_ravlt['RAVLT_Pct_Ret'] - df_ravlt['RAVLT_Pct_Ret'].mean()) /
            df_ravlt['RAVLT_Pct_Ret'].std()
        )
        log(f"RAVLT_Pct_Ret_z: mean={df_ravlt['RAVLT_Pct_Ret_z'].mean():.4f}, "
            f"std={df_ravlt['RAVLT_Pct_Ret_z'].std():.4f}")
        # Data Quality Checks

        log("Verifying data quality...")

        for col_name in ['RAVLT_Forgetting', 'RAVLT_Forgetting_z', 'RAVLT_Pct_Ret', 'RAVLT_Pct_Ret_z']:
            n_missing = df_ravlt[col_name].isnull().sum()
            if n_missing > 0:
                log(f"{n_missing} missing values in {col_name}")
            else:
                log(f"No missing values in {col_name}")

        # Check for extreme outliers (|z| > 3) in forgetting
        n_outliers_f = (np.abs(df_ravlt['RAVLT_Forgetting_z']) > 3).sum()
        if n_outliers_f > 0:
            log(f"{n_outliers_f} participants with extreme forgetting (|z| > 3)")
        else:
            log("No extreme forgetting outliers detected")

        # Check for extreme outliers (|z| > 3) in pct retention
        n_outliers_p = (np.abs(df_ravlt['RAVLT_Pct_Ret_z'].dropna()) > 3).sum()
        if n_outliers_p > 0:
            log(f"{n_outliers_p} participants with extreme pct retention (|z| > 3)")
        else:
            log("No extreme pct retention outliers detected")
        # Save Analysis Output

        output_path = RQ_DIR / "data" / "step01_ravlt_scores.csv"
        log(f"Saving RAVLT scores to: {output_path}")

        df_ravlt.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step01_ravlt_scores.csv ({len(df_ravlt)} rows, {len(df_ravlt.columns)} cols)")
        log(f"Output columns: {df_ravlt.columns.tolist()}")
        # Run Validation Tool

        log("Running validate_numeric_range on RAVLT_Forgetting...")

        validation_result = validate_numeric_range(
            data=df_ravlt['RAVLT_Forgetting'],
            min_val=-15.0,
            max_val=15.0,
            column_name='RAVLT_Forgetting'
        )

        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            if validation_result.get('valid', False):
                log("PASS - Forgetting scores within expected range [-15, 15]")
            else:
                log("FAIL - Forgetting scores outside expected range")
        else:
            log(f"{validation_result}")

        log("Running validate_numeric_range on RAVLT_Pct_Ret...")

        validation_result_pct = validate_numeric_range(
            data=df_ravlt['RAVLT_Pct_Ret'].dropna(),
            min_val=0.0,
            max_val=200.0,  # Generous upper bound (can exceed 100 if DR > best trial)
            column_name='RAVLT_Pct_Ret'
        )

        if isinstance(validation_result_pct, dict):
            for key, value in validation_result_pct.items():
                log(f"{key}: {value}")
            if validation_result_pct.get('valid', False):
                log("PASS - Percent retention within expected range [0, 200]")
            else:
                log("FAIL - Percent retention outside expected range")
        else:
            log(f"{validation_result_pct}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
