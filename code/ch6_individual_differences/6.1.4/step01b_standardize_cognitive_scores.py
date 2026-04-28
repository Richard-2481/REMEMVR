#!/usr/bin/env python3
"""
Step 01b: Standardize Cognitive Scores to T-scores
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Convert raw cognitive test scores to T-scores (M=50, SD=10)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
RQ_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = RQ_DIR.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Set up logging
LOG_FILE = RQ_DIR / "logs" / "step01b_standardize_cognitive_scores.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def standardize_to_t_score(scores, t_mean=50.0, t_sd=10.0):
    """
    Convert raw scores to T-scores.
    T-score = 50 + 10 * (score - sample_mean) / sample_sd
    """
    # Remove NaN values for mean/std calculation
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return np.full_like(scores, np.nan)
    
    sample_mean = np.mean(valid_scores)
    sample_sd = np.std(valid_scores, ddof=1)
    
    if sample_sd == 0:
        return np.full_like(scores, t_mean)
    
    # Calculate T-scores
    t_scores = t_mean + t_sd * (scores - sample_mean) / sample_sd
    return t_scores

def main():
    """Main execution."""
    log("Step 01b: Standardize cognitive scores to T-scores")
    
    # Load cognitive test data from Step 01
    input_path = RQ_DIR / "data" / "step01_cognitive_tests.csv"
    log(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    log(f"Loaded {len(df)} participants")
    
    # Create output dataframe
    output_df = pd.DataFrame()
    output_df['uid'] = df['uid']
    
    # List of columns to standardize
    test_columns = ['RAVLT_T', 'RAVLT_DR_T', 'RAVLT_Pct_Ret', 'BVMT_T', 'BVMT_Pct_Ret', 'NART_T', 'RPM_T']
    
    log("Converting to T-scores (M=50, SD=10)...")
    for col in test_columns:
        if col in df.columns:
            # Calculate T-scores
            t_scores = standardize_to_t_score(df[col].values)
            output_df[col] = t_scores
            
            # Report statistics
            valid_scores = t_scores[~np.isnan(t_scores)]
            if len(valid_scores) > 0:
                mean_t = np.mean(valid_scores)
                std_t = np.std(valid_scores, ddof=1)
                min_t = np.min(valid_scores)
                max_t = np.max(valid_scores)
                n_missing = np.isnan(t_scores).sum()
                
                log(f"  - {col}:")
                log(f"    Raw: M={df[col].mean():.1f}, SD={df[col].std():.1f}")
                log(f"    T-score: M={mean_t:.1f}, SD={std_t:.1f}, Range=[{min_t:.1f}, {max_t:.1f}]")
                if n_missing > 0:
                    log(f"    Missing: {n_missing} values")
    
    # Verify T-score properties
    log("Checking T-score properties...")
    for col in test_columns:
        if col in output_df.columns:
            valid = output_df[col].dropna()
            if len(valid) > 0:
                mean_check = abs(valid.mean() - 50.0) < 0.5
                sd_check = abs(valid.std() - 10.0) < 0.5
                status = "PASS" if mean_check and sd_check else "WARNING"
                log(f"  - {col}: M={valid.mean():.2f}, SD={valid.std():.2f} [{status}]")
    
    # Save output
    output_path = RQ_DIR / "data" / "step01b_cognitive_t_scores.csv"
    output_path.parent.mkdir(exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Saved T-scores to {output_path}")
    log(f"Shape: {output_df.shape}")
    
    log("Step 01b complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())