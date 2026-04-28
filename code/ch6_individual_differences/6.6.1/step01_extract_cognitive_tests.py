#!/usr/bin/env python3
"""Extract Cognitive Test Data: Extract cognitive test scores from dfnonvr.csv and convert to T-scores."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

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


if __name__ == "__main__":
    try:
        log("Step 01: Extract Cognitive Test Data")
        # Load Input Data

        log("Loading dfnonvr.csv...")
        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'

        if not dfnonvr_path.exists():
            log(f"Input file not found: {dfnonvr_path}")
            sys.exit(1)

        df = pd.read_csv(dfnonvr_path)
        log(f"dfnonvr.csv ({len(df)} rows, {len(df.columns)} cols)")
        # Apply RAVLT Ceiling Fix BEFORE computing composites
        # Fix unadministered trials stored as 0 when previous trial was >= 14
        # Affects participants: A064, A070, A077, A103

        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        ravlt_trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]

        # Verify trial columns exist
        for col in ravlt_trial_cols:
            if col not in df.columns:
                log(f"Expected column '{col}' not found")
                sys.exit(1)

        df = fix_ravlt_ceiling(df, log)
        # Extract RAVLT Composite (Sum of Trials 1-5, after ceiling fix)
        # Per 4_analysis.yaml lines 90-93: Sum ravlt-trial-1-score through ravlt-trial-5-score
        # CRITICAL: Do NOT include distraction trial (per gcode_lessons.md bug #6)

        log("Computing RAVLT composite score (post-ceiling fix)...")

        if len(ravlt_trial_cols) != 5:
            log(f"Expected 5 RAVLT trial columns, found {len(ravlt_trial_cols)}")
            sys.exit(1)

        df['RAVLT_raw'] = df[ravlt_trial_cols].sum(axis=1)
        log(f"RAVLT composite from {len(ravlt_trial_cols)} trials")
        log(f"RAVLT_raw: mean={df['RAVLT_raw'].mean():.2f}, SD={df['RAVLT_raw'].std():.2f}, range=[{df['RAVLT_raw'].min():.0f}, {df['RAVLT_raw'].max():.0f}]")
        # STEP 3b: Compute RAVLT Percent Retention
        # RAVLT_Pct_Ret = delayed_recall / best_learning_trial * 100
        # Best learning trial = highest of trials 1-5 (after ceiling fix)

        log("Computing RAVLT percent retention...")
        dr_col = 'ravlt-delayed-recall-score'
        if dr_col not in df.columns:
            log(f"Expected column '{dr_col}' not found")
            sys.exit(1)

        # Best learning trial = max of trials 1-5 (after ceiling fix)
        df['best_learning_trial'] = df[ravlt_trial_cols].max(axis=1)

        # Compute percent retention; handle edge case where best_trial == 0
        df['RAVLT_Pct_Ret_raw'] = np.where(
            df['best_learning_trial'] > 0,
            df[dr_col] / df['best_learning_trial'] * 100,
            np.nan
        )
        n_valid_pct = df['RAVLT_Pct_Ret_raw'].notna().sum()
        log(f"RAVLT Percent Retention: {n_valid_pct}/{len(df)} valid")
        log(f"RAVLT_Pct_Ret_raw: mean={df['RAVLT_Pct_Ret_raw'].mean():.2f}, SD={df['RAVLT_Pct_Ret_raw'].std():.2f}, range=[{df['RAVLT_Pct_Ret_raw'].min():.1f}, {df['RAVLT_Pct_Ret_raw'].max():.1f}]")
        # Extract BVMT Composite (Sum of Trials 1-3)
        # Per 4_analysis.yaml lines 94-96: Sum bvmt-trial-1-score through bvmt-trial-3-score

        log("Computing BVMT composite score...")
        bvmt_trial_cols = []
        for i in range(1, 4):  # Trials 1-3
            col = f'bvmt-trial-{i}-score'
            if col in df.columns:
                bvmt_trial_cols.append(col)
            else:
                log(f"Expected column '{col}' not found")

        if len(bvmt_trial_cols) != 3:
            log(f"Expected 3 BVMT trial columns, found {len(bvmt_trial_cols)}")
            sys.exit(1)

        df['BVMT_raw'] = df[bvmt_trial_cols].sum(axis=1)
        log(f"BVMT composite from {len(bvmt_trial_cols)} trials")
        log(f"BVMT_raw: mean={df['BVMT_raw'].mean():.2f}, SD={df['BVMT_raw'].std():.2f}, range=[{df['BVMT_raw'].min():.0f}, {df['BVMT_raw'].max():.0f}]")
        # STEP 4b: Extract BVMT Percent Retention
        # Pre-computed column from dfnonvr.csv

        log("Extracting BVMT percent retention...")
        bvmt_pct_col = 'bvmt-percent-retained'
        if bvmt_pct_col not in df.columns:
            log(f"Expected column '{bvmt_pct_col}' not found")
            sys.exit(1)

        df['BVMT_Pct_Ret_raw'] = df[bvmt_pct_col]
        n_valid_bvmt_pct = df['BVMT_Pct_Ret_raw'].notna().sum()
        log(f"BVMT Percent Retention: {n_valid_bvmt_pct}/{len(df)} valid")
        log(f"BVMT_Pct_Ret_raw: mean={df['BVMT_Pct_Ret_raw'].mean():.2f}, SD={df['BVMT_Pct_Ret_raw'].std():.2f}, range=[{df['BVMT_Pct_Ret_raw'].min():.1f}, {df['BVMT_Pct_Ret_raw'].max():.1f}]")
        # Extract RPM Score (Direct Use)
        # Per 4_analysis.yaml lines 97-99: Use rpm-score directly

        log("Extracting RPM score...")
        if 'rpm-score' not in df.columns:
            log("Expected column 'rpm-score' not found")
            sys.exit(1)

        df['RPM_raw'] = df['rpm-score']
        log(f"RPM score")
        log(f"RPM_raw: mean={df['RPM_raw'].mean():.2f}, SD={df['RPM_raw'].std():.2f}, range=[{df['RPM_raw'].min():.0f}, {df['RPM_raw'].max():.0f}]")
        # Convert All Raw Scores to T-Scores
        # T-score formula: T = 50 + 10 * (raw - mean) / sd
        # Per 4_analysis.yaml line 101: t_score_conversion: true

        log("Converting raw scores to T-scores...")

        # RAVLT T-score
        ravlt_mean = df['RAVLT_raw'].mean()
        ravlt_sd = df['RAVLT_raw'].std()
        df['RAVLT_T'] = 50 + 10 * (df['RAVLT_raw'] - ravlt_mean) / ravlt_sd

        # BVMT T-score
        bvmt_mean = df['BVMT_raw'].mean()
        bvmt_sd = df['BVMT_raw'].std()
        df['BVMT_T'] = 50 + 10 * (df['BVMT_raw'] - bvmt_mean) / bvmt_sd

        # RPM T-score
        rpm_mean = df['RPM_raw'].mean()
        rpm_sd = df['RPM_raw'].std()
        df['RPM_T'] = 50 + 10 * (df['RPM_raw'] - rpm_mean) / rpm_sd

        # RAVLT Percent Retention T-score
        ravlt_pct_mean = df['RAVLT_Pct_Ret_raw'].mean()
        ravlt_pct_sd = df['RAVLT_Pct_Ret_raw'].std()
        df['RAVLT_Pct_Ret_T'] = 50 + 10 * (df['RAVLT_Pct_Ret_raw'] - ravlt_pct_mean) / ravlt_pct_sd

        # BVMT Percent Retention T-score
        bvmt_pct_mean = df['BVMT_Pct_Ret_raw'].mean()
        bvmt_pct_sd = df['BVMT_Pct_Ret_raw'].std()
        df['BVMT_Pct_Ret_T'] = 50 + 10 * (df['BVMT_Pct_Ret_raw'] - bvmt_pct_mean) / bvmt_pct_sd

        log(f"All scores to T-scores (mean=50, SD=10)")
        log(f"RAVLT_T: mean={df['RAVLT_T'].mean():.2f}, SD={df['RAVLT_T'].std():.2f}, range=[{df['RAVLT_T'].min():.1f}, {df['RAVLT_T'].max():.1f}]")
        log(f"BVMT_T: mean={df['BVMT_T'].mean():.2f}, SD={df['BVMT_T'].std():.2f}, range=[{df['BVMT_T'].min():.1f}, {df['BVMT_T'].max():.1f}]")
        log(f"RPM_T: mean={df['RPM_T'].mean():.2f}, SD={df['RPM_T'].std():.2f}, range=[{df['RPM_T'].min():.1f}, {df['RPM_T'].max():.1f}]")
        log(f"RAVLT_Pct_Ret_T: mean={df['RAVLT_Pct_Ret_T'].mean():.2f}, SD={df['RAVLT_Pct_Ret_T'].std():.2f}, range=[{df['RAVLT_Pct_Ret_T'].min():.1f}, {df['RAVLT_Pct_Ret_T'].max():.1f}]")
        log(f"BVMT_Pct_Ret_T: mean={df['BVMT_Pct_Ret_T'].mean():.2f}, SD={df['BVMT_Pct_Ret_T'].std():.2f}, range=[{df['BVMT_Pct_Ret_T'].min():.1f}, {df['BVMT_Pct_Ret_T'].max():.1f}]")
        # Extract Demographics
        # Per 4_analysis.yaml line 100: demographic_columns: ["age", "sex", "education"]

        log("Extracting demographic variables...")
        required_demo_cols = ['age', 'sex', 'education']
        missing_demo = [col for col in required_demo_cols if col not in df.columns]

        if missing_demo:
            log(f"Missing demographic columns: {missing_demo}")
            sys.exit(1)

        log(f"Demographics: {', '.join(required_demo_cols)}")
        # Create Output DataFrame
        # Columns: UID, RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T, age, sex, education

        log("Creating output dataset...")
        output_cols = ['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        output_df = df[output_cols].copy()

        log(f"Output dataset ({len(output_df)} rows, {len(output_df.columns)} cols)")
        # Save Output
        # Output: results/ch7/7.6.1/data/step01_cognitive_tests.csv

        output_path = RQ_DIR / 'data' / 'step01_cognitive_tests.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"{len(output_df)} participants with 5 T-scores + demographics")
        # Validation
        # - t_score_range: [20, 80]
        # - expected_participants: 100

        log("Validating T-score ranges...")

        # Check T-score ranges (20-80 is reasonable, not strict)
        for col in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
            out_of_range = (output_df[col] < 20) | (output_df[col] > 80)
            n_out = out_of_range.sum()
            if n_out > 0:
                log(f"{col}: {n_out} values outside [20, 80] range")
            else:
                log(f"{col}: All values within [20, 80] range")

        # Check participant count
        expected_n = 100
        actual_n = len(output_df)
        if actual_n == expected_n:
            log(f"Participant count: {actual_n} (expected {expected_n})")
        else:
            log(f"Participant count: {actual_n} (expected {expected_n})")

        # Check for missing values
        missing_counts = output_df.isnull().sum()
        if missing_counts.sum() == 0:
            log("No missing values in output dataset")
        else:
            log(f"Missing values detected:")
            for col in missing_counts[missing_counts > 0].index:
                log(f"  {col}: {missing_counts[col]} missing")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
