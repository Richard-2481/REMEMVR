#!/usr/bin/env python3
"""Extract Cognitive Tests: Extract cognitive test scores from dfnonvr.csv and compute T-score transformations"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column mappings from DATA_DICTIONARY.md
RAVLT_COLUMNS = ['ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
                 'ravlt-trial-4-score', 'ravlt-trial-5-score']
BVMT_COLUMN = 'bvmt-total-recall'
RPM_COLUMN = 'rpm-score'
DASS_COLUMNS = ['total-dass-depression-items', 'total-dass-anxiety-items', 'total-dass-stress-items']
SLEEP_COLUMN = 'typical-sleep-hours'
DEMOGRAPHIC_COLUMNS = ['UID', 'age', 'sex', 'education']

# T-score parameters
TARGET_MEAN = 50.0
TARGET_SD = 10.0

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def compute_t_score(raw_scores: pd.Series, target_mean: float = 50.0, target_sd: float = 10.0) -> pd.Series:
    """
    Convert raw scores to T-scores (M=50, SD=10).

    Formula: T = (raw - mean_raw) / sd_raw * target_sd + target_mean
    """
    mean_raw = raw_scores.mean()
    sd_raw = raw_scores.std()

    if sd_raw == 0:
        log(f"Zero SD for variable, cannot compute T-scores")
        return pd.Series([target_mean] * len(raw_scores))

    t_scores = ((raw_scores - mean_raw) / sd_raw) * target_sd + target_mean
    return t_scores

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract Cognitive Tests")
        # Load dfnonvr.csv
        log("Loading dfnonvr.csv...")

        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'
        df_raw = pd.read_csv(dfnonvr_path)

        log(f"dfnonvr.csv: {len(df_raw)} rows, {len(df_raw.columns)} columns")
        # Validate Required Columns
        log("Checking required columns...")

        required_columns = (DEMOGRAPHIC_COLUMNS + RAVLT_COLUMNS + [BVMT_COLUMN, RPM_COLUMN] +
                          DASS_COLUMNS + +
                          ['ravlt-delayed-recall-score', 'bvmt-percent-retained'])

        validation_result = validate_data_columns(df=df_raw, required_columns=required_columns)

        if not validation_result.get('valid', False):
            missing = validation_result.get('missing_columns', [])
            log(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        log(f"All {len(required_columns)} required columns present")
        # Extract and Compute Variables
        log("Computing cognitive test scores...")

        # Initialize output dataframe
        df_output = pd.DataFrame()

        # Extract demographics (UID already uppercase)
        df_output['UID'] = df_raw['UID']  # Already uppercase in dfnonvr.csv
        df_output['Age'] = df_raw['age']
        df_output['Sex'] = df_raw['sex']  # 0=female, 1=male
        df_output['Education'] = df_raw['education']

        log(f"Demographics: UID, Age, Sex, Education")

        # RAVLT Ceiling Fix: if trial N == 0 and trial N-1 >= 14, set trial N = 15
        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        fixes_applied = 0
        for idx in df_raw.index:
            for i in range(1, 5):
                current_col = RAVLT_COLUMNS[i]
                prev_col = RAVLT_COLUMNS[i - 1]
                if df_raw.at[idx, current_col] == 0 and df_raw.at[idx, prev_col] >= 14:
                    uid = df_raw.at[idx, 'UID']
                    df_raw.at[idx, current_col] = 15
                    fixes_applied += 1
                    log(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df_raw.at[idx, prev_col]})")
        log(f"[CEILING FIX] Total fixes applied: {fixes_applied}")

        # Compute RAVLT Total (trials 1-5 only, no distraction trial)
        ravlt_scores = df_raw.sum(axis=1)
        log(f"RAVLT Total: mean={ravlt_scores.mean():.2f}, SD={ravlt_scores.std():.2f}, range=[{ravlt_scores.min():.0f}, {ravlt_scores.max():.0f}]")

        # Compute RAVLT T-score
        df_output['RAVLT_T'] = compute_t_score(ravlt_scores, TARGET_MEAN, TARGET_SD)
        log(f"[T-SCORE] RAVLT_T: mean={df_output['RAVLT_T'].mean():.2f}, SD={df_output['RAVLT_T'].std():.2f}")

        # Extract BVMT Total and compute T-score
        bvmt_scores = df_rawlog(f"BVMT Total: mean={bvmt_scores.mean():.2f}, SD={bvmt_scores.std():.2f}, range=[{bvmt_scores.min():.0f}, {bvmt_scores.max():.0f}]")
        df_output['BVMT_T'] = compute_t_score(bvmt_scores, TARGET_MEAN, TARGET_SD)
        log(f"[T-SCORE] BVMT_T: mean={df_output['BVMT_T'].mean():.2f}, SD={df_output['BVMT_T'].std():.2f}")

        # Extract RPM and compute T-score
        rpm_scores = df_rawlog(f"RPM Score: mean={rpm_scores.mean():.2f}, SD={rpm_scores.std():.2f}, range=[{rpm_scores.min():.0f}, {rpm_scores.max():.0f}]")
        df_output['RPM_T'] = compute_t_score(rpm_scores, TARGET_MEAN, TARGET_SD)
        log(f"[T-SCORE] RPM_T: mean={df_output['RPM_T'].mean():.2f}, SD={df_output['RPM_T'].std():.2f}")

        # RAVLT Percent Retention: DR / max(trials 1-5 after ceiling fix) * 100
        dr_col = 'ravlt-delayed-recall-score'
        ravlt_max_trial = df_raw.max(axis=1)
        ravlt_pct_ret = np.where(ravlt_max_trial > 0, (df_raw[dr_col] / ravlt_max_trial) * 100, np.nan)
        df_output['RAVLT_Pct_Ret_T'] = compute_t_score(pd.Series(ravlt_pct_ret), TARGET_MEAN, TARGET_SD)
        log(f"[T-SCORE] RAVLT_Pct_Ret_T: mean={df_output['RAVLT_Pct_Ret_T'].mean():.2f}, SD={df_output['RAVLT_Pct_Ret_T'].std():.2f}")

        # BVMT Percent Retained: pre-computed column from dfnonvr.csv
        bvmt_pct_ret = df_raw['bvmt-percent-retained']
        df_output['BVMT_Pct_Ret_T'] = compute_t_score(bvmt_pct_ret, TARGET_MEAN, TARGET_SD)
        log(f"[T-SCORE] BVMT_Pct_Ret_T: mean={df_output['BVMT_Pct_Ret_T'].mean():.2f}, SD={df_output['BVMT_Pct_Ret_T'].std():.2f}")

        # Compute DASS Total (sum of 3 subscales)
        dass_scores = df_raw.sum(axis=1)
        log(f"DASS Total: mean={dass_scores.mean():.2f}, SD={dass_scores.std():.2f}, range=[{dass_scores.min():.0f}, {dass_scores.max():.0f}]")
        df_output['DASS_Total'] = dass_scores  # Keep raw (not T-score)

        # Extract Sleep hours
        df_output['Sleep'] = df_rawlog(f"Sleep: mean={df_output['Sleep'].mean():.2f}, SD={df_output['Sleep'].std():.2f}, range=[{df_output['Sleep'].min():.1f}, {df_output['Sleep'].max():.1f}]")

        log(f"Extracted {len(df_output.columns)} variables for {len(df_output)} participants")
        # Compute Descriptive Statistics
        log("Computing descriptive statistics...")

        descriptive_stats = []
        for col in df_output.columns:
            if col == 'UID':
                continue

            stats = {
                'variable': col,
                'mean': df_output[col].mean(),
                'sd': df_output[col].std(),
                'min': df_output[col].min(),
                'max': df_output[col].max(),
                'missing_count': df_output[col].isna().sum()
            }
            descriptive_stats.append(stats)

        df_stats = pd.DataFrame(descriptive_stats)
        log(f"Descriptive statistics for {len(df_stats)} variables")
        # Save Outputs
        output_file = OUTPUT_DIR / "step01_cognitive_tests.csv"
        log(f"Saving cognitive test data to {output_file}")
        df_output.to_csv(output_file, index=False, encoding='utf-8')
        log(f"{len(df_output)} rows, {len(df_output.columns)} columns")

        stats_file = OUTPUT_DIR / "step01_descriptive_stats.csv"
        log(f"Saving descriptive statistics to {stats_file}")
        df_stats.to_csv(stats_file, index=False, encoding='utf-8')
        log(f"{len(df_stats)} rows")
        # Validation Check
        log("Checking T-score transformations...")

        # Check T-scores have approximately correct mean and SD
        for col in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
            mean_val = df_output[col].mean()
            sd_val = df_output[col].std()

            # Allow ±2 points for mean, ±2 points for SD (sample variation)
            mean_ok = abs(mean_val - TARGET_MEAN) < 2.0
            sd_ok = abs(sd_val - TARGET_SD) < 2.0

            if mean_ok and sd_ok:
                log(f"{col}: mean={mean_val:.2f} (target={TARGET_MEAN}), SD={sd_val:.2f} (target={TARGET_SD})")
            else:
                log(f"{col}: mean={mean_val:.2f} (target={TARGET_MEAN}), SD={sd_val:.2f} (target={TARGET_SD})")

        # Check for unexpected missing data
        total_missing = df_output.isna().sum().sum()
        if total_missing > 0:
            log(f"{total_missing} missing values detected")
            for col in df_output.columns:
                n_missing = df_output[col].isna().sum()
                if n_missing > 0:
                    log(f"  {col}: {n_missing} missing")
        else:
            log("No missing data")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
