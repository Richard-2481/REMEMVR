#!/usr/bin/env python3
"""extract_cognitive_tests: Extract and T-score cognitive test data from dfnonvr.csv. Compute RAVLT sum"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_cognitive_tests.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Functions (Due to Signature Mismatches)

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

def extract_cognitive_tests_custom(df, cognitive_tests, demographics):
    """
    Custom implementation of cognitive test extraction.
    
    Args:
        df: DataFrame with cognitive test data
        cognitive_tests: Dict mapping test names to column lists
        demographics: List of demographic column names
        
    Returns:
        DataFrame with extracted tests and demographics
    """
    log("Starting custom cognitive test extraction...")
    
    results = {'UID': df['UID']}
    
    # Extract RAVLT (sum of trials 1-5)
    ravlt_cols = cognitive_tests['RAVLT']
    log(f"Extracting columns: {ravlt_cols}")
    ravlt_data = df[ravlt_cols]
    results['RAVLT_Total'] = ravlt_data.sum(axis=1)
    log(f"Computed total scores, range: {results['RAVLT_Total'].min():.1f} to {results['RAVLT_Total'].max():.1f}")
    
    # Extract BVMT (sum of trials 1-3, NOT delayed recall)
    bvmt_cols = cognitive_tests['BVMT'][:3]  # Only first 3 trials
    log(f"Extracting columns: {bvmt_cols}")
    bvmt_data = df[bvmt_cols]
    results['BVMT_Total'] = bvmt_data.sum(axis=1)
    log(f"Computed total scores, range: {results['BVMT_Total'].min():.1f} to {results['BVMT_Total'].max():.1f}")
    
    # Extract RPM
    rpm_col = cognitive_tests['RPM']
    log(f"Extracting column: {rpm_col}")
    results['RPM'] = df[rpm_col]
    log(f"Raw scores, range: {results['RPM'].min():.1f} to {results['RPM'].max():.1f}")
    
    # Extract demographics
    log(f"Extracting columns: {demographics}")
    for demo_col in demographics:
        results[demo_col] = df[demo_col]
    
    result_df = pd.DataFrame(results)
    log(f"Complete: {len(result_df)} participants, {len(result_df.columns)} variables")
    
    return result_df

def standardize_to_t_scores_custom(scores, target_mean=50, target_sd=10):
    """
    Custom T-score conversion (M=50, SD=10).
    
    Args:
        scores: pandas Series with raw scores
        target_mean: Target mean for T-scores (default 50)
        target_sd: Target SD for T-scores (default 10)
        
    Returns:
        pandas Series with T-scores
    """
    # Remove missing values for calculation
    valid_scores = scores.dropna()
    
    if len(valid_scores) == 0:
        log("No valid scores for T-score conversion")
        return scores  # Return original if all missing
    
    # Calculate sample statistics
    sample_mean = valid_scores.mean()
    sample_sd = valid_scores.std()
    
    if sample_sd == 0:
        log("Zero standard deviation - all scores identical")
        return pd.Series([target_mean] * len(scores), index=scores.index)
    
    # Z-score transformation: (X - M) / SD
    z_scores = (scores - sample_mean) / sample_sd
    
    # T-score transformation: Z * target_SD + target_Mean
    t_scores = z_scores * target_sd + target_mean
    
    log(f"[T-SCORE] Raw: M={sample_mean:.2f}, SD={sample_sd:.2f} -> T-score: M={target_mean}, SD={target_sd}")
    log(f"[T-SCORE] Range: {t_scores.min():.1f} to {t_scores.max():.1f}")
    
    return t_scores

def validate_numeric_range_custom(data, min_val, max_val, column_name):
    """
    Custom numeric range validation.
    
    Args:
        data: pandas Series with data to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        column_name: Name of column for reporting
        
    Returns:
        Dict with validation results
    """
    valid_data = data.dropna()
    n_total = len(data)
    n_valid = len(valid_data)
    n_missing = n_total - n_valid
    
    if n_valid == 0:
        return {
            'valid': False,
            'message': f"{column_name}: All values missing",
            'n_total': n_total,
            'n_valid': 0,
            'n_missing': n_missing,
            'n_out_of_range': 0
        }
    
    # Check range
    out_of_range = valid_data[(valid_data < min_val) | (valid_data > max_val)]
    n_out_of_range = len(out_of_range)
    
    min_observed = valid_data.min()
    max_observed = valid_data.max()
    
    is_valid = n_out_of_range == 0
    
    result = {
        'valid': is_valid,
        'message': f"{column_name}: {n_valid}/{n_total} valid, range [{min_observed:.1f}, {max_observed:.1f}]",
        'n_total': n_total,
        'n_valid': n_valid,
        'n_missing': n_missing,
        'n_out_of_range': n_out_of_range,
        'min_observed': min_observed,
        'max_observed': max_observed,
        'min_expected': min_val,
        'max_expected': max_val
    }
    
    if n_out_of_range > 0:
        result['message'] += f", {n_out_of_range} out of range [{min_val}, {max_val}]"
        result['out_of_range_values'] = out_of_range.tolist()
    
    return result

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract Cognitive Tests")
        # Load Input Data

        log("Loading dfnonvr.csv...")
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        df = pd.read_csv(dfnonvr_path)
        log(f"dfnonvr.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns exist
        required_cols = [
            'UID', 'age', 'sex', 'education',
            'ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
            'ravlt-trial-4-score', 'ravlt-trial-5-score',
            'ravlt-delayed-recall-score',
            'bvmt-trial-1-score', 'bvmt-trial-2-score', 'bvmt-trial-3-score',
            'bvmt-percent-retained',
            'rpm-score'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All {len(required_cols)} required columns present")
        # Run Analysis Tool (Custom Implementation)

        # Apply RAVLT ceiling fix BEFORE computing totals
        log("Applying RAVLT ceiling fix...")
        df = fix_ravlt_ceiling(df, log)

        cognitive_tests = {
            'RAVLT': ['ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
                      'ravlt-trial-4-score', 'ravlt-trial-5-score'],
            'BVMT': ['bvmt-trial-1-score', 'bvmt-trial-2-score', 'bvmt-trial-3-score'],
            'RPM': 'rpm-score'
        }
        demographics = ['age', 'sex', 'education']

        log("Running extract_cognitive_tests_custom...")
        cognitive_df = extract_cognitive_tests_custom(df, cognitive_tests, demographics)
        log("Cognitive test extraction complete")

        # Compute percent retention measures
        log("Computing percent retention measures...")
        cognitive_df['RAVLT_Pct_Ret'] = compute_ravlt_percent_retention(df, log)
        cognitive_df['BVMT_Pct_Ret'] = df['bvmt-percent-retained'].values
        n_bvmt_valid = cognitive_df['BVMT_Pct_Ret'].notna().sum()
        log(f"BVMT Percent Retention: {n_bvmt_valid}/{len(cognitive_df)} valid")
        # Apply T-Score Conversion

        log("[T-SCORE] Converting cognitive scores to T-scores (M=50, SD=10)...")
        
        cognitive_df['RAVLT_T'] = standardize_to_t_scores_custom(cognitive_df['RAVLT_Total'])
        cognitive_df['BVMT_T'] = standardize_to_t_scores_custom(cognitive_df['BVMT_Total'])
        cognitive_df['RPM_T'] = standardize_to_t_scores_custom(cognitive_df['RPM'])
        cognitive_df['RAVLT_Pct_Ret_T'] = standardize_to_t_scores_custom(cognitive_df['RAVLT_Pct_Ret'])
        cognitive_df['BVMT_Pct_Ret_T'] = standardize_to_t_scores_custom(cognitive_df['BVMT_Pct_Ret'])

        log("[T-SCORE] Conversion complete")

        # Create final output with only required columns
        output_cols = ['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        final_df = cognitive_df[output_cols].copy()
        # Save Analysis Output
        # Output: step02_cognitive_tests.csv
        # Contains: T-scored cognitive tests with demographics
        # Columns: UID, RAVLT_T, BVMT_T, RPM_T, age, sex, education

        output_path = RQ_DIR / "data" / "step02_cognitive_tests.csv"
        log(f"Saving {output_path}...")
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_cognitive_tests.csv ({len(final_df)} rows, {len(final_df.columns)} cols)")
        # Run Validation Tool (Custom Implementation)
        # Validates: T-scores are in reasonable range [20, 80] (±3 SD)
        # Threshold: T-score range validation

        log("Running validate_numeric_range_custom...")
        
        validation_results = {}
        
        # Validate RAVLT_T
        ravlt_result = validate_numeric_range_custom(final_df['RAVLT_T'], 20, 80, 'RAVLT_T')
        validation_results['RAVLT_T'] = ravlt_result
        log(f"RAVLT_T: {ravlt_result['message']}")

        # Validate BVMT_T
        bvmt_result = validate_numeric_range_custom(final_df['BVMT_T'], 20, 80, 'BVMT_T')
        validation_results['BVMT_T'] = bvmt_result
        log(f"BVMT_T: {bvmt_result['message']}")

        # Validate RPM_T
        rpm_result = validate_numeric_range_custom(final_df['RPM_T'], 20, 80, 'RPM_T')
        validation_results['RPM_T'] = rpm_result
        log(f"RPM_T: {rpm_result['message']}")

        # Validate RAVLT_Pct_Ret_T
        ravlt_pct_result = validate_numeric_range_custom(final_df['RAVLT_Pct_Ret_T'], 20, 80, 'RAVLT_Pct_Ret_T')
        validation_results['RAVLT_Pct_Ret_T'] = ravlt_pct_result
        log(f"RAVLT_Pct_Ret_T: {ravlt_pct_result['message']}")

        # Validate BVMT_Pct_Ret_T
        bvmt_pct_result = validate_numeric_range_custom(final_df['BVMT_Pct_Ret_T'], 20, 80, 'BVMT_Pct_Ret_T')
        validation_results['BVMT_Pct_Ret_T'] = bvmt_pct_result
        log(f"BVMT_Pct_Ret_T: {bvmt_pct_result['message']}")

        # Overall validation summary
        all_valid = all(result['valid'] for result in validation_results.values())
        if all_valid:
            log("All T-score ranges PASS")
        else:
            failed_tests = [test for test, result in validation_results.items() if not result['valid']]
            log(f"FAIL - Issues with: {failed_tests}")
            for test in failed_tests:
                result = validation_results[test]
                if result.get('n_out_of_range', 0) > 0:
                    log(f"{test} out-of-range values: {result['out_of_range_values']}")

        # Summary statistics
        log("Final dataset summary:")
        log(f"Participants: {len(final_df)}")
        log(f"RAVLT_T: M={final_df['RAVLT_T'].mean():.1f}, SD={final_df['RAVLT_T'].std():.1f}")
        log(f"BVMT_T: M={final_df['BVMT_T'].mean():.1f}, SD={final_df['BVMT_T'].std():.1f}")
        log(f"RPM_T: M={final_df['RPM_T'].mean():.1f}, SD={final_df['RPM_T'].std():.1f}")
        log(f"RAVLT_Pct_Ret_T: M={final_df['RAVLT_Pct_Ret_T'].mean():.1f}, SD={final_df['RAVLT_Pct_Ret_T'].std():.1f}")
        log(f"BVMT_Pct_Ret_T: M={final_df['BVMT_Pct_Ret_T'].mean():.1f}, SD={final_df['BVMT_Pct_Ret_T'].std():.1f}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)