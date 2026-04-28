#!/usr/bin/env python3
"""extract_cognitive_tests: Extract RAVLT cognitive test scores from dfnonvr.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_extract_cognitive_tests.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    Logic: If a participant scored 14 or 15 on trial N, and trial N+1 is 0,
    the 0 likely represents an unadministered trial (ceiling reached), not a true score.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0
    for idx in df.index:
        for i in range(1, 5):  # Check trials 2-5 against their predecessor
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df.at[idx, prev_col]})")
    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df


def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT percent retention: delayed recall / best learning trial * 100.

    Uses the best (highest) learning trial as denominator to avoid inflation
    from early low-scoring trials.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    dr_col = 'ravlt-delayed-recall-score'
    pct_ret = np.full(len(df), np.nan)
    for i, idx in enumerate(df.index):
        dr = df.at[idx, dr_col]
        # Find best (highest) learning trial as denominator
        denom = np.nan
        for trial_col in reversed(trial_cols):
            val = df.at[idx, trial_col]
            if val > 0:
                denom = val
                break
        if denom > 0:
            pct_ret[i] = (dr / denom) * 100
    n_valid = np.sum(~np.isnan(pct_ret))
    log_fn(f"RAVLT Percent Retention: {n_valid}/{len(df)} valid")
    return pct_ret


if __name__ == "__main__":
    try:
        log("Step 0: extract_cognitive_tests")
        # Load Input Data

        log("Loading dfnonvr.csv...")
        # Based on gcode_lessons.md: Column names are exact from DATA_DICTIONARY.md
        # CRITICAL: Column names are lowercase with hyphens
        input_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'
        df_cognitive = pd.read_csv(input_path)
        log(f"dfnonvr.csv ({len(df_cognitive)} rows, {len(df_cognitive.columns)} cols)")
        # Extract RAVLT Trial Scores

        log("Extracting RAVLT cognitive test scores...")

        # Define RAVLT trial score columns (exact names from DATA_DICTIONARY.md)
        ravlt_trial_cols = [
            'ravlt-trial-1-score',
            'ravlt-trial-2-score',
            'ravlt-trial-3-score',
            'ravlt-trial-4-score',
            'ravlt-trial-5-score'
        ]
        dr_col = 'ravlt-delayed-recall-score'

        # Verify all required columns exist
        required_cols = ['UID'] + ravlt_trial_cols + [dr_col]
        missing_cols = [col for col in required_cols if col not in df_cognitive.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"All required RAVLT columns: {ravlt_trial_cols + [dr_col]}")

        # Extract participants with complete RAVLT data
        cols_needed = ['UID'] + ravlt_trial_cols + [dr_col]
        df_ravlt = df_cognitive[cols_needed].copy()

        # Check for missing data
        initial_count = len(df_ravlt)
        df_ravlt = df_ravlt.dropna(subset=ravlt_trial_cols + [dr_col])
        final_count = len(df_ravlt)

        if initial_count != final_count:
            log(f"Dropped {initial_count - final_count} participants with incomplete RAVLT data")

        log(f"{final_count} participants with complete RAVLT data")
        # STEP 2b: Apply Ceiling Fix BEFORE computing totals
        # Fix unadministered trials stored as 0 when previous trial was >= 14

        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        df_ravlt = fix_ravlt_ceiling(df_ravlt, log)

        # Compute RAVLT total (sum of 5 learning trials, after ceiling fix)
        df_ravlt['ravlt_total'] = df_ravlt[ravlt_trial_cols].sum(axis=1)

        log(f"RAVLT total scores (range: {df_ravlt['ravlt_total'].min()}-{df_ravlt['ravlt_total'].max()})")
        # STEP 2c: Compute Percent Retention
        # percent retention = delayed recall / best learning trial * 100

        log("Computing RAVLT percent retention...")
        df_ravlt['ravlt_pct_ret'] = compute_ravlt_percent_retention(df_ravlt, log)

        log(f"RAVLT pct retention (range: {df_ravlt['ravlt_pct_ret'].min():.1f}-{df_ravlt['ravlt_pct_ret'].max():.1f})")

        # Prepare output dataset with standardized column names
        # Change UID to lowercase 'uid' for consistency with Ch7 conventions
        output_df = pd.DataFrame({
            'uid': df_ravlt['UID'],
            'ravlt_total': df_ravlt['ravlt_total'],
            'ravlt_pct_ret': df_ravlt['ravlt_pct_ret']
        })

        log("RAVLT extraction complete")
        # Save Analysis Output
        # These outputs will be used by: Step 2 (merge with paradigm theta scores)

        output_path = RQ_DIR / "data" / "step00_cognitive_tests.csv"
        log(f"Saving {output_path}...")
        # Output: step00_cognitive_tests.csv
        # Contains: UID, RAVLT total score, and percent retention
        # Columns: ['uid', 'ravlt_total', 'ravlt_pct_ret']
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step00_cognitive_tests.csv ({len(output_df)} rows, {len(output_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns present in output
        # Threshold: All required columns must exist

        log("Running validate_data_format...")
        validation_result = validate_data_format(
            df=output_df,  # Output DataFrame
            required_cols=["uid", "ravlt_total", "ravlt_pct_ret"]  # Required columns for next step
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation checks specific to RAVLT
        log("Running RAVLT-specific validation...")

        # Check expected participant count
        if len(output_df) != 100:
            log(f"Expected 100 participants, got {len(output_df)}")
        else:
            log("Participant count: 100 (expected)")

        # Check RAVLT total range (0-75 theoretical max: 5 trials x 15 words each)
        min_ravlt = output_df['ravlt_total'].min()
        max_ravlt = output_df['ravlt_total'].max()
        if min_ravlt < 0 or max_ravlt > 75:
            log(f"RAVLT total outside expected range [0,75]: {min_ravlt}-{max_ravlt}")
        else:
            log(f"RAVLT total range: {min_ravlt}-{max_ravlt} (within [0,75])")

        # Check ravlt_pct_ret range
        min_pct = output_df['ravlt_pct_ret'].min()
        max_pct = output_df['ravlt_pct_ret'].max()
        n_missing_pct = output_df['ravlt_pct_ret'].isna().sum()
        log(f"RAVLT pct retention range: {min_pct:.1f}-{max_pct:.1f}")
        if n_missing_pct > 0:
            log(f"{n_missing_pct} missing ravlt_pct_ret values")
        else:
            log("No missing ravlt_pct_ret values")

        # Check for missing values in core columns
        missing_count = output_df[['uid', 'ravlt_total']].isnull().sum().sum()
        if missing_count > 0:
            log(f"Found {missing_count} missing values in uid/ravlt_total")
        else:
            log("No missing values in uid/ravlt_total")

        # Verify RAVLT total calculation by spot checking first few rows
        log("Spot checking RAVLT total calculation...")
        for idx in range(min(3, len(df_ravlt))):
            uid = df_ravlt.iloc[idx]['UID']
            trial_scores = df_ravlt.iloc[idx][ravlt_trial_cols].values
            computed_total = trial_scores.sum()
            output_total = output_df[output_df['uid'] == uid]['ravlt_total'].iloc[0]

            if computed_total != output_total:
                log(f"RAVLT total mismatch for {uid}: computed={computed_total}, output={output_total}")
            else:
                log(f"{uid}: trials {trial_scores} -> total {computed_total} (correct)")

        log("Step 0 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)