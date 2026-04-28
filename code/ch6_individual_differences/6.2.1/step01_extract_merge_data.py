#!/usr/bin/env python3
"""extract_merge_data: Extract and merge data by combining Ch5 5.1.1 theta_all scores (IRT ability estimates)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import zscore
import traceback
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import missing data utilities
try:
    sys.path.insert(0, str(PROJECT_ROOT / "results" / "ch7"))
    from missing_data_handler import analyze_missing_pattern, create_missing_data_report
except ImportError:
    # Utilities not available - continue without
    pass


from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_merge_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Real-time monitoring compatibility
    print(msg, flush=True)  # -u flag compatibility

# T-Score Calculation Function

def calculate_t_score(raw_scores):
    """
    Convert raw scores to T-scores (mean=50, SD=10).
    
    Formula: T = 50 + 10 * ((X - mean(X)) / SD(X))
    This standardizes scores to T-score metric commonly used in neuropsychology.
    """
    mean_score = np.mean(raw_scores)
    std_score = np.std(raw_scores, ddof=1)  # Sample standard deviation
    if std_score == 0:
        # Handle case where all scores are identical
        return np.full(len(raw_scores), 50.0)
    return 50 + 10 * ((raw_scores - mean_score) / std_score)

# RAVLT Ceiling Fix

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    Logic: If a participant scored >= 14 on trial N-1 and trial N == 0,
    trial N was not administered (ceiling). Substitute 15.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0
    for idx in df.index:
        for i in range(1, 5):  # Check trials 2,3,4,5
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df.at[idx, prev_col]})")
    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df

# RAVLT Percent Retention

def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT Percent Retention = (Delayed Recall / best available trial) * 100.

    Uses the last non-zero learning trial as the denominator (scanning trials 5->1).
    """
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

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract and Merge Data")
        # Load and Aggregate Ch5 Theta Scores
        # NOTE: Column adaptation - file has 'Theta_All' not 'theta_all', no 'SE' column

        log("Loading Ch5 5.1.1 theta scores...")
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        theta_df = pd.read_csv(theta_path)
        log(f"{theta_path.name} ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        log(f"Theta columns: {theta_df.columns.tolist()}")
        
        # Verify expected structure (400 rows = 100 participants × 4 test sessions)
        expected_theta_rows = 400
        if len(theta_df) != expected_theta_rows:
            log(f"Expected {expected_theta_rows} theta rows, got {len(theta_df)}")
        
        # Check column adaptation requirement
        if 'Theta_All' not in theta_df.columns:
            raise ValueError(f"Expected 'Theta_All' column but found: {theta_df.columns.tolist()}")
            
        # Aggregate theta scores by UID (mean across 4 test sessions)
        # This creates participant-level IRT ability estimates for cross-sectional analysis
        log("Aggregating theta scores by UID (mean across test sessions)...")
        theta_agg = theta_df.groupby('UID')['Theta_All'].mean().reset_index()
        theta_agg.columns = ['UID', 'theta_all']  # Rename to match spec output
        log(f"{len(theta_agg)} participants (expected: 100)")
        
        if len(theta_agg) != 100:
            log(f"Expected 100 participants, got {len(theta_agg)}")
        # Load and Process Cognitive Test Data
        # NOTE: Multiple column adaptations required from spec

        log("Loading cognitive test data from dfnonvr.csv...")
        cognitive_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        cognitive_df = pd.read_csv(cognitive_path)
        log(f"{cognitive_path.name} ({len(cognitive_df)} rows, {len(cognitive_df.columns)} cols)")
        
        # Verify expected participant count
        if len(cognitive_df) != 100:
            log(f"Expected 100 participants in cognitive data, got {len(cognitive_df)}")
            
        # Extract required columns with adaptations
        log("Extracting cognitive measures with column adaptations...")

        # Age adaptation: 'age' not 'Age'
        if 'age' not in cognitive_df.columns:
            raise ValueError("Expected 'age' column in dfnonvr.csv")
        age_data = cognitive_df[['UID', 'age']].copy()
        age_data.columns = ['UID', 'Age']  # Rename for consistency

        # RPM adaptation: 'rpm-score' is raw score, need T-score conversion
        if 'rpm-score' not in cognitive_df.columns:
            raise ValueError("Expected 'rpm-score' column in dfnonvr.csv")
        rpm_raw = cognitive_df['rpm-score'].values
        rpm_t_scores = calculate_t_score(rpm_raw)
        # RAVLT: Apply ceiling fix BEFORE computing totals
        log("Applying RAVLT ceiling fix...")
        ravlt_trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
        for col in ravlt_trial_cols:
            if col not in cognitive_df.columns:
                raise ValueError(f"Expected '{col}' column in dfnonvr.csv")
        cognitive_df = fix_ravlt_ceiling(cognitive_df, log)

        # RAVLT total: Sum trials 1-5 (after ceiling fix)
        log("Calculating RAVLT total from trials 1-5 (post ceiling fix)...")
        ravlt_total_raw = cognitive_df[ravlt_trial_cols].sum(axis=1).values
        ravlt_t_scores = calculate_t_score(ravlt_total_raw)
        # BVMT: Recompute from sum(trials 1-3) instead of pre-computed column
        log("Computing BVMT total from sum(trials 1-3)...")
        bvmt_trial_cols = [f'bvmt-trial-{i}-score' for i in range(1, 4)]
        for col in bvmt_trial_cols:
            if col not in cognitive_df.columns:
                raise ValueError(f"Expected '{col}' column in dfnonvr.csv")
        bvmt_raw = cognitive_df[bvmt_trial_cols].sum(axis=1).values
        bvmt_t_scores = calculate_t_score(bvmt_raw)
        # RAVLT Percent Retention
        log("Computing RAVLT Percent Retention...")
        if 'ravlt-delayed-recall-score' not in cognitive_df.columns:
            raise ValueError("Expected 'ravlt-delayed-recall-score' column in dfnonvr.csv")
        ravlt_pct_ret_raw = compute_ravlt_percent_retention(cognitive_df, log)
        ravlt_pct_ret_t = calculate_t_score(ravlt_pct_ret_raw)
        # BVMT Percent Retained (from pre-computed column)
        log("Extracting BVMT Percent Retained...")
        if 'bvmt-percent-retained' not in cognitive_df.columns:
            raise ValueError("Expected 'bvmt-percent-retained' column in dfnonvr.csv")
        bvmt_pct_ret_raw = cognitive_df['bvmt-percent-retained'].values
        bvmt_pct_ret_t = calculate_t_score(bvmt_pct_ret_raw)

        # Create cognitive measures dataframe with T-scores
        cognitive_clean = pd.DataFrame({
            'UID': cognitive_df['UID'],
            'Age': age_data['Age'],
            'RAVLT_T': ravlt_t_scores,        # T-score from summed trials 1-5
            'BVMT_T': bvmt_t_scores,          # T-score from sum(trials 1-3)
            'RPM_T': rpm_t_scores,            # T-score from raw score
            'RAVLT_Pct_Ret_T': ravlt_pct_ret_t,  # T-score from percent retention
            'BVMT_Pct_Ret_T': bvmt_pct_ret_t     # T-score from percent retained
        })

        log(f"Cognitive measures with T-scores calculated")
        log(f"RAVLT raw range: {ravlt_total_raw.min():.1f} - {ravlt_total_raw.max():.1f}")
        log(f"BVMT raw range: {bvmt_raw.min():.1f} - {bvmt_raw.max():.1f}")
        log(f"RPM raw range: {rpm_raw.min():.1f} - {rpm_raw.max():.1f}")
        log(f"RAVLT Pct Ret raw range: {np.nanmin(ravlt_pct_ret_raw):.1f} - {np.nanmax(ravlt_pct_ret_raw):.1f}")
        log(f"BVMT Pct Ret raw range: {np.nanmin(bvmt_pct_ret_raw):.1f} - {np.nanmax(bvmt_pct_ret_raw):.1f}")
        # Merge Datasets
        # Merge theta scores with cognitive measures on UID

        log("Merging theta scores with cognitive measures...")
        merged_df = pd.merge(theta_agg, cognitive_clean, on='UID', how='inner')
        log(f"{len(merged_df)} participants after merge (expected: 100)")
        
        if len(merged_df) != 100:
            log(f"Expected 100 participants after merge, got {len(merged_df)}")
            
        # Check for missing data - analysis requires complete cases
        missing_count = merged_df.isnull().sum().sum()
        if missing_count > 0:
            log(f"Missing data detected: {missing_count} values")
            log("Missing data by column:")
            for col in merged_df.columns:
                missing = merged_df[col].isnull().sum()
                if missing > 0:
                    log(f"  {col}: {missing} missing")
        else:
            log("No missing data - complete case analysis dataset")
        # Create Standardized Predictors
        # Create z-scored versions of predictors for regression analysis

        log("Creating standardized predictor variables...")
        merged_df['Age_std'] = zscore(merged_df['Age'])
        merged_df['RAVLT_T_std'] = zscore(merged_df['RAVLT_T'])
        merged_df['BVMT_T_std'] = zscore(merged_df['BVMT_T'])
        merged_df['RPM_T_std'] = zscore(merged_df['RPM_T'])
        merged_df['RAVLT_Pct_Ret_T_std'] = zscore(merged_df['RAVLT_Pct_Ret_T'])
        merged_df['BVMT_Pct_Ret_T_std'] = zscore(merged_df['BVMT_Pct_Ret_T'])

        log("Standardized variables created (mean=0, SD=1)")

        # Report descriptive statistics
        log("Final dataset summary:")
        log(f"  N participants: {len(merged_df)}")
        log(f"  Age range: {merged_df['Age'].min():.1f} - {merged_df['Age'].max():.1f} years")
        log(f"  Theta_all range: {merged_df['theta_all'].min():.2f} - {merged_df['theta_all'].max():.2f}")
        log(f"  RAVLT_T range: {merged_df['RAVLT_T'].min():.1f} - {merged_df['RAVLT_T'].max():.1f}")
        log(f"  BVMT_T range: {merged_df['BVMT_T'].min():.1f} - {merged_df['BVMT_T'].max():.1f}")
        log(f"  RPM_T range: {merged_df['RPM_T'].min():.1f} - {merged_df['RPM_T'].max():.1f}")
        log(f"  RAVLT_Pct_Ret_T range: {merged_df['RAVLT_Pct_Ret_T'].min():.1f} - {merged_df['RAVLT_Pct_Ret_T'].max():.1f}")
        log(f"  BVMT_Pct_Ret_T range: {merged_df['BVMT_Pct_Ret_T'].min():.1f} - {merged_df['BVMT_Pct_Ret_T'].max():.1f}")
        # Save Analysis Dataset
        # Save merged analysis dataset for subsequent hierarchical regression steps
        # Format: CSV with all required variables for age moderation analysis

        log("Saving analysis dataset...")
        output_path = RQ_DIR / "data" / "step01_analysis_dataset.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        # Run Validation Tool
        # Validate final dataset structure meets requirements for downstream analyses

        log("Running validate_dataframe_structure...")
        expected_columns = ['UID', 'theta_all', 'Age', 'RAVLT_T', 'BVMT_T', 'RPM_T',
                           'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T',
                           'Age_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                           'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        
        validation_result = validate_dataframe_structure(
            df=merged_df,
            expected_rows=100,
            expected_columns=expected_columns,
            column_types=None  # Skip type checking for this step
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 01 complete - analysis dataset ready for hierarchical regression")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)