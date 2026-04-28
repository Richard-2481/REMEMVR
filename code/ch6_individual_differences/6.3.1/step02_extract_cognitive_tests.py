#!/usr/bin/env python3
"""extract_cognitive_tests: Extract cognitive test scores from dfnonvr.csv and convert to T-scores (M=50, SD=10)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_cognitive_tests.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Analysis Functions

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0)."""
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


def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT percent retention: delayed recall / best learning trial * 100."""
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    dr_col = 'ravlt-delayed-recall-score'
    pct_ret = np.full(len(df), np.nan)
    for i, idx in enumerate(df.index):
        dr = df.at[idx, dr_col]
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


def extract_cognitive_tests_custom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cognitive test scores and compute totals.
    
    Custom implementation due to signature mismatch with tools.data.extract_cognitive_tests.
    Actual function has (uid_list, data_path) but spec expects (df, tests).
    """
    log("Starting cognitive test extraction...")
    
    # Initialize output dataframe
    result = pd.DataFrame()
    result['UID'] = df['UID'].copy()
    
    # Extract RAVLT total (trials 1-5 sum)
    # CRITICAL: From lessons learned - be explicit about which trials to include
    ravlt_cols = []
    for i in range(1, 6):  # Trials 1-5 only (exclude distraction trial)
        col = f'ravlt-trial-{i}-score'
        if col in df.columns:
            ravlt_cols.append(col)
        else:
            log(f"Missing RAVLT column: {col}")
    
    if len(ravlt_cols) == 5:
        result['RAVLT_total'] = df[ravlt_cols].sum(axis=1)
        log(f"RAVLT_total from {len(ravlt_cols)} trials")
    else:
        log(f"Expected 5 RAVLT columns, found {len(ravlt_cols)}")
        raise ValueError(f"RAVLT columns incomplete: found {ravlt_cols}")
    
    # Extract BVMT total (trials 1-3 sum)
    bvmt_cols = []
    for i in range(1, 4):  # Trials 1-3 
        col = f'bvmt-trial-{i}-score'
        if col in df.columns:
            bvmt_cols.append(col)
        else:
            log(f"Missing BVMT column: {col}")
    
    if len(bvmt_cols) == 3:
        result['BVMT_total'] = df[bvmt_cols].sum(axis=1)
        log(f"BVMT_total from {len(bvmt_cols)} trials")
    else:
        log(f"Expected 3 BVMT columns, found {len(bvmt_cols)}")
        raise ValueError(f"BVMT columns incomplete: found {bvmt_cols}")
    
    # Extract RPM score (direct score)
    if 'rpm-score' in df.columns:
        result['RPM_total'] = df['rpm-score'].copy()
        log("RPM_total from rpm-score")
    else:
        log("Missing RPM column: rpm-score")
        raise ValueError("RPM column missing: rpm-score")
    
    # Extract demographics
    demo_cols = ['age', 'sex', 'education']
    for col in demo_cols:
        if col in df.columns:
            result[col] = df[col].copy()
            log(f"Demographics: {col}")
        else:
            log(f"Missing demographic column: {col}")
            result[col] = np.nan
    
    log(f"Extracted data for {len(result)} participants")
    return result

def convert_to_t_scores(df: pd.DataFrame, score_cols: List[str], target_mean: float = 50, target_sd: float = 10) -> pd.DataFrame:
    """
    Convert raw scores to T-scores (M=50, SD=10).
    
    T-score formula: T = (raw - raw_mean) / raw_sd * target_sd + target_mean
    """
    log(f"[T-SCORE] Converting {len(score_cols)} measures to T-scores...")
    
    result = df.copy()
    
    for col in score_cols:
        # Support both '{col}_total' convention and direct column names
        raw_col = f"{col}_total"
        if raw_col not in df.columns:
            raw_col = col  # Fall back to direct column name
        t_col = f"{col}_T"

        if raw_col not in df.columns:
            log(f"Column {raw_col} not found for T-score conversion")
            continue
            
        # Get valid (non-null) scores
        valid_mask = pd.notna(df[raw_col])
        valid_scores = df.loc[valid_mask, raw_col]
        
        if len(valid_scores) == 0:
            log(f"No valid scores for {col}")
            result[t_col] = np.nan
            continue
            
        # Compute raw statistics
        raw_mean = valid_scores.mean()
        raw_sd = valid_scores.std()
        
        log(f"[T-SCORE] {col}: Raw M={raw_mean:.2f}, SD={raw_sd:.2f}")
        
        if raw_sd == 0:
            log(f"Zero SD for {col} - all participants have same score")
            result[t_col] = target_mean  # All get mean T-score
        else:
            # Apply T-score transformation
            t_scores = (df[raw_col] - raw_mean) / raw_sd * target_sd + target_mean
            result[t_col] = t_scores
            
            # Report T-score statistics
            t_valid = t_scores[valid_mask]
            t_mean = t_valid.mean()
            t_sd = t_valid.std()
            log(f"[T-SCORE] {col}: T-score M={t_mean:.2f}, SD={t_sd:.2f}")
    
    return result

def validate_t_scores(df: pd.DataFrame, t_score_cols: List[str], expected_mean: float = 50, expected_sd: float = 10) -> Dict[str, Any]:
    """
    Validate T-score computation.
    
    Custom validation due to signature mismatch with tools.validation.validate_numeric_range.
    Actual function expects (data, min_val, max_val, column_name) but we need DataFrame validation.
    """
    log("Validating T-score computation...")
    
    results = {
        'valid': True,
        'messages': [],
        'statistics': {}
    }
    
    for col in t_score_cols:
        if col not in df.columns:
            results['valid'] = False
            results['messages'].append(f"Missing T-score column: {col}")
            continue
            
        # Get valid T-scores
        valid_scores = df[col].dropna()
        
        if len(valid_scores) == 0:
            results['valid'] = False
            results['messages'].append(f"No valid T-scores for {col}")
            continue
            
        # Check statistics
        actual_mean = valid_scores.mean()
        actual_sd = valid_scores.std()
        
        # Allow reasonable tolerance (±5 for mean, ±3 for SD)
        mean_ok = abs(actual_mean - expected_mean) <= 5
        sd_ok = abs(actual_sd - expected_sd) <= 3
        
        # Check reasonable range (T-scores typically 20-80)
        min_score = valid_scores.min()
        max_score = valid_scores.max()
        range_ok = (min_score >= 20) and (max_score <= 80)
        
        results['statistics'][col] = {
            'mean': actual_mean,
            'sd': actual_sd,
            'min': min_score,
            'max': max_score,
            'n_valid': len(valid_scores),
            'mean_ok': mean_ok,
            'sd_ok': sd_ok,
            'range_ok': range_ok
        }
        
        if not mean_ok:
            results['messages'].append(f"{col}: Mean {actual_mean:.1f} not close to {expected_mean}")
        if not sd_ok:
            results['messages'].append(f"{col}: SD {actual_sd:.1f} not close to {expected_sd}")
        if not range_ok:
            results['messages'].append(f"{col}: Range [{min_score:.1f}, {max_score:.1f}] outside [20, 80]")
            
        if not (mean_ok and sd_ok and range_ok):
            results['valid'] = False
    
    # Overall validation
    if results['valid']:
        log("T-score validation PASS")
    else:
        log(f"T-score validation FAIL: {'; '.join(results['messages'])}")
    
    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract Cognitive Tests")
        # Load Input Data

        log("Loading dfnonvr.csv...")
        input_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        df_cognitive = pd.read_csv(input_path, encoding='utf-8')
        log(f"dfnonvr.csv ({len(df_cognitive)} rows, {len(df_cognitive.columns)} cols)")
        
        # Validate required columns exist
        required_cols = [
            'UID', 'ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
            'ravlt-trial-4-score', 'ravlt-trial-5-score', 'ravlt-delayed-recall-score',
            'bvmt-trial-1-score', 'bvmt-trial-2-score', 'bvmt-trial-3-score',
            'bvmt-delayed-recall-score', 'bvmt-percent-retained',
            'rpm-score', 'age', 'sex', 'education'
        ]
        
        missing_cols = [col for col in required_cols if col not in df_cognitive.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All {len(required_cols)} required columns present")
        # STEP 2a: Apply RAVLT Ceiling Fix
        # Fix ceiling effects BEFORE computing RAVLT totals
        # Participants who scored 15 on a trial may not have been administered
        # the next trial, stored as 0. Substitute 15 for these cases.

        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        df_cognitive = fix_ravlt_ceiling(df_cognitive, log)
        # STEP 2b: Extract Raw Totals

        log("Extracting cognitive test raw totals...")
        df_extracted = extract_cognitive_tests_custom(df_cognitive)
        log("Cognitive test extraction complete")
        # STEP 2c: Compute Percent Retention Measures
        # RAVLT percent retention: delayed recall / best learning trial * 100
        # BVMT percent retention: from bvmt-percent-retained column in dfnonvr.csv

        log("Computing percent retention measures...")
        df_extracted['RAVLT_Pct_Ret_total'] = compute_ravlt_percent_retention(df_cognitive, log)
        df_extracted['BVMT_Pct_Ret_total'] = df_cognitive['bvmt-percent-retained'].values
        n_valid_bvmt_pct = df_extracted['BVMT_Pct_Ret_total'].notna().sum()
        log(f"BVMT Percent Retention: {n_valid_bvmt_pct}/{len(df_extracted)} valid")
        
        # Report raw total statistics
        score_cols = ['RAVLT_total', 'BVMT_total', 'RPM_total', 'RAVLT_Pct_Ret_total', 'BVMT_Pct_Ret_total']
        for col in score_cols:
            valid_scores = df_extracted[col].dropna()
            if len(valid_scores) > 0:
                log(f"[RAW STATS] {col}: M={valid_scores.mean():.2f}, SD={valid_scores.std():.2f}, N={len(valid_scores)}")
        # Convert to T-scores

        log("Converting to T-scores (M=50, SD=10)...")
        df_t_scored = convert_to_t_scores(df_extracted, ['RAVLT', 'BVMT', 'RPM', 'RAVLT_Pct_Ret', 'BVMT_Pct_Ret'])
        log("T-score conversion complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 (standardization validation), Step 4 (merging)

        log("Saving cognitive tests with T-scores...")
        output_cols = ['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        
        # Ensure all output columns exist
        for col in output_cols:
            if col not in df_t_scored.columns:
                log(f"Output column {col} missing, adding as NaN")
                df_t_scored[col] = np.nan
        
        df_output = df_t_scored[output_cols].copy()
        
        # Drop rows with missing T-scores (allow 5% exclusion as per validation criteria)
        initial_n = len(df_output)
        t_score_cols = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        df_output = df_output.dropna(subset=t_score_cols)
        final_n = len(df_output)
        
        exclusion_rate = (initial_n - final_n) / initial_n
        log(f"Dropped {initial_n - final_n} participants with missing T-scores ({exclusion_rate:.1%})")
        
        if exclusion_rate > 0.05:  # More than 5% excluded
            log(f"Exclusion rate {exclusion_rate:.1%} exceeds 5% threshold")
        
        if final_n < 95:  # Less than 95 participants remaining
            log(f"Final sample size {final_n} below 95 participant threshold")
        
        output_path = RQ_DIR / "data" / "step02_cognitive_tests.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_cognitive_tests.csv ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Run Validation Tool
        # Validates: T-score means ~50, SDs ~10, reasonable ranges [20, 80]
        # Threshold: Sample size >= 95 participants

        log("Running T-score validation...")
        validation_result = validate_t_scores(df_output, t_score_cols)
        
        # Report validation results
        if validation_result['valid']:
            log("T-score validation PASS")
            for col, stats in validation_result['statistics'].items():
                log(f"{col}: M={stats['mean']:.1f}, SD={stats['sd']:.1f}, N={stats['n_valid']}")
        else:
            log("T-score validation FAIL")
            for msg in validation_result['messages']:
                log(f"{msg}")
        
        # Check sample size criterion
        if final_n >= 95:
            log(f"Sample size criterion met: N={final_n} >= 95")
        else:
            log(f"Sample size criterion NOT met: N={final_n} < 95")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)