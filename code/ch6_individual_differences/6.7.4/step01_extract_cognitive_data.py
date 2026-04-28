#!/usr/bin/env python3
"""extract_cognitive_data: Load REMEMVR theta scores from Ch5 and RAVLT data from dfnonvr.csv, then merge"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_data.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_cognitive_scores.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# RAVLT Ceiling Fix

def fix_ravlt_ceiling(df, log_fn):
    """If trial N == 0 and trial N-1 >= 14, set trial N = 15.

    Known issue: Some participants stopped responding on a trial after near-perfect
    performance, likely due to fatigue or disengagement rather than memory failure.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0
    for idx in df.index:
        for i in range(1, 5):
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df.at[idx, prev_col]})")
    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_cognitive_data")
        # Load Ch5 Theta Scores
        log("Loading Ch5 REMEMVR theta scores...")

        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        df_theta = pd.read_csv(theta_path)
        log(f"Theta file: {len(df_theta)} rows, {len(df_theta.columns)} columns")
        log(f"Columns: {df_theta.columns.tolist()}")

        # Aggregate theta by UID (mean across 4 test sessions)
        # Clinical comparison needs participant-level scores, not session-level
        df_theta_agg = df_theta.groupby('UID')['Theta_All'].mean().reset_index()
        df_theta_agg.rename(columns={'Theta_All': 'REMEMVR_theta'}, inplace=True)
        log(f"Mean theta per participant: {len(df_theta_agg)} participants")
        log(f"REMEMVR_theta range: [{df_theta_agg['REMEMVR_theta'].min():.3f}, {df_theta_agg['REMEMVR_theta'].max():.3f}]")
        # Load RAVLT Data from dfnonvr.csv
        log("Loading RAVLT data from dfnonvr.csv...")

        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        df_nonvr = pd.read_csv(dfnonvr_path)
        log(f"dfnonvr.csv: {len(df_nonvr)} rows, {len(df_nonvr.columns)} columns")

        # Calculate RAVLT Total (sum of trials 1-5 ONLY, not distraction)
        # Bug #6 from gcode_lessons: Be explicit about which trials to sum
        ravlt_trial_cols = []
        for i in range(1, 6):  # Trials 1-5 only
            col = f'ravlt-trial-{i}-score'
            if col in df_nonvr.columns:
                ravlt_trial_cols.append(col)
            else:
                log(f"Missing expected column: {col}")
                raise ValueError(f"Missing RAVLT trial column: {col}")

        log(f"RAVLT trial columns: {ravlt_trial_cols}")

        # Apply ceiling fix BEFORE computing RAVLT_Total
        df_nonvr = fix_ravlt_ceiling(df_nonvr, log)

        df_nonvr['RAVLT_Total'] = df_nonvr[ravlt_trial_cols].sum(axis=1)
        log(f"RAVLT_Total range: [{df_nonvr['RAVLT_Total'].min()}, {df_nonvr['RAVLT_Total'].max()}]")

        # Compute RAVLT Percent Retention: (delayed recall / best learning trial) * 100
        best_trial = df_nonvr[ravlt_trial_cols].max(axis=1)
        df_nonvr['RAVLT_Pct_Ret'] = np.where(
            best_trial > 0,
            df_nonvr['ravlt-delayed-recall-score'] / best_trial * 100,
            np.nan
        )
        log(f"RAVLT_Pct_Ret range: [{df_nonvr['RAVLT_Pct_Ret'].min():.1f}, {df_nonvr['RAVLT_Pct_Ret'].max():.1f}]")
        pct_ret_missing = df_nonvr['RAVLT_Pct_Ret'].isnull().sum()
        if pct_ret_missing > 0:
            log(f"RAVLT_Pct_Ret missing for {pct_ret_missing} participants (best_trial=0)")

        # Extract demographics (rename to standard format)
        df_demographics = df_nonvr[['UID', 'RAVLT_Total', 'RAVLT_Pct_Ret', 'age', 'education', 'vr-exposure', 'nart-score']].copy()
        df_demographics.rename(columns={
            'age': 'Age',
            'education': 'Education',
            'vr-exposure': 'VR_Experience',
            'nart-score': 'NART_Score'
        }, inplace=True)

        log(f"Demographics for {len(df_demographics)} participants")
        # Merge Datasets
        log("Combining REMEMVR theta and RAVLT data...")

        df_merged = pd.merge(df_theta_agg, df_demographics, on='UID', how='inner')
        log(f"Final dataset: {len(df_merged)} participants")

        # Check for missing data
        missing_counts = df_merged.isnull().sum()
        if missing_counts.sum() > 0:
            log(f"Missing values detected:")
            for col, count in missing_counts.items():
                if count > 0:
                    log(f"  {col}: {count} missing")
        else:
            log(f"No missing values in merged dataset")
        # Save Output
        log("Saving cognitive scores dataset...")

        df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_merged)} rows, {len(df_merged.columns)} columns)")
        # Validate Output
        log("Running validate_data_columns...")

        required_columns = ['UID', 'REMEMVR_theta', 'RAVLT_Total', 'RAVLT_Pct_Ret',
                          'Age', 'Education', 'VR_Experience', 'NART_Score']

        validation_result = validate_data_columns(
            df=df_merged,
            required_columns=required_columns
        )

        if validation_result.get('valid', False):
            log(f"All required columns present")
        else:
            log(f"Validation failed: {validation_result.get('message', 'Unknown error')}")
            raise ValueError(f"Column validation failed: {validation_result}")

        # Validate value ranges
        log("Checking value ranges...")

        ranges_valid = True

        # REMEMVR_theta: [-3, 3]
        theta_min, theta_max = df_merged['REMEMVR_theta'].min(), df_merged['REMEMVR_theta'].max()
        if theta_min < -3 or theta_max > 3:
            log(f"REMEMVR_theta out of expected range [-3, 3]: [{theta_min:.3f}, {theta_max:.3f}]")
            # Not a hard failure - theta can occasionally exceed ±3
        else:
            log(f"REMEMVR_theta in expected range: [{theta_min:.3f}, {theta_max:.3f}]")

        # RAVLT_Total: [15, 75]
        ravlt_min, ravlt_max = df_merged['RAVLT_Total'].min(), df_merged['RAVLT_Total'].max()
        if ravlt_min < 15 or ravlt_max > 75:
            log(f"RAVLT_Total out of expected range [15, 75]: [{ravlt_min}, {ravlt_max}]")
        else:
            log(f"RAVLT_Total in expected range: [{ravlt_min}, {ravlt_max}]")

        # RAVLT_Pct_Ret: [0, 200]
        pct_ret_min, pct_ret_max = df_merged['RAVLT_Pct_Ret'].min(), df_merged['RAVLT_Pct_Ret'].max()
        if pct_ret_min < 0 or pct_ret_max > 200:
            log(f"RAVLT_Pct_Ret out of expected range [0, 200]: [{pct_ret_min:.1f}, {pct_ret_max:.1f}]")
        else:
            log(f"RAVLT_Pct_Ret in expected range: [{pct_ret_min:.1f}, {pct_ret_max:.1f}]")

        # Age: [18, 89]
        age_min, age_max = df_merged['Age'].min(), df_merged['Age'].max()
        if age_min < 18 or age_max > 89:
            log(f"Age out of expected range [18, 89]: [{age_min}, {age_max}]")
        else:
            log(f"Age in expected range: [{age_min}, {age_max}]")

        # Education: [1, 10]
        edu_min, edu_max = df_merged['Education'].min(), df_merged['Education'].max()
        if edu_min < 1 or edu_max > 10:
            log(f"Education out of expected range [1, 10]: [{edu_min}, {edu_max}]")
        else:
            log(f"Education in expected range: [{edu_min}, {edu_max}]")

        # VR_Experience: [0, 4]
        vr_min, vr_max = df_merged['VR_Experience'].min(), df_merged['VR_Experience'].max()
        if vr_min < 0 or vr_max > 4:
            log(f"VR_Experience out of expected range [0, 4]: [{vr_min}, {vr_max}]")
        else:
            log(f"VR_Experience in expected range: [{vr_min}, {vr_max}]")

        # NART_Score: [0, 50]
        nart_min, nart_max = df_merged['NART_Score'].min(), df_merged['NART_Score'].max()
        if nart_min < 0 or nart_max > 50:
            log(f"NART_Score out of expected range [0, 50]: [{nart_min}, {nart_max}]")
        else:
            log(f"NART_Score in expected range: [{nart_min}, {nart_max}]")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
