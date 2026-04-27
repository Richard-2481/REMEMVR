#!/usr/bin/env python3
"""
Step 05: Merge All Predictors
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Merge cognitive tests, demographics, self-report, and theta scores
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from functools import reduce

# Add project root to path
RQ_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = RQ_DIR.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Set up logging
LOG_FILE = RQ_DIR / "logs" / "step05_merge_predictors.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def main():
    """Main execution."""
    log("[START] Step 05: Merge all predictors")
    
    # Load all data files from previous steps
    log("[LOAD] Loading data from previous steps...")
    
    # 1. Cognitive T-scores
    cognitive_path = RQ_DIR / "data" / "step01b_cognitive_t_scores.csv"
    df_cognitive = pd.read_csv(cognitive_path)
    df_cognitive['uid'] = df_cognitive['uid'].astype(str)
    log(f"  - Cognitive tests: {df_cognitive.shape}")
    
    # 2. Demographics
    demographics_path = RQ_DIR / "data" / "step02_demographics.csv"
    df_demographics = pd.read_csv(demographics_path)
    df_demographics['uid'] = df_demographics['uid'].astype(str)
    log(f"  - Demographics: {df_demographics.shape}")
    
    # 3. Self-report (DASS, VR, Sleep)
    self_report_path = RQ_DIR / "data" / "step03_self_report.csv"
    df_self_report = pd.read_csv(self_report_path)
    df_self_report['uid'] = df_self_report['uid'].astype(str)
    log(f"  - Self-report: {df_self_report.shape}")
    
    # 4. Theta scores (outcome variable)
    theta_path = RQ_DIR / "data" / "step04_theta_scores.csv"
    df_theta = pd.read_csv(theta_path)
    df_theta['uid'] = df_theta['uid'].astype(str)
    log(f"  - Theta scores: {df_theta.shape}")
    
    # Sequential merge
    log("[MERGE] Merging all datasets on uid...")
    
    # Start with theta (outcome)
    merged = df_theta.copy()
    
    # Add cognitive tests
    merged = pd.merge(merged, df_cognitive, on='uid', how='left')
    log(f"  - After adding cognitive: {merged.shape}")
    
    # Add demographics
    merged = pd.merge(merged, df_demographics, on='uid', how='left')
    log(f"  - After adding demographics: {merged.shape}")
    
    # Add self-report
    merged = pd.merge(merged, df_self_report, on='uid', how='left')
    log(f"  - After adding self-report: {merged.shape}")
    
    # Check for missing values
    log("[CHECK] Missing values per column:")
    missing = merged.isnull().sum()
    for col, n_missing in missing.items():
        if n_missing > 0:
            log(f"  - {col}: {n_missing} missing ({n_missing/len(merged)*100:.1f}%)")
    
    # Standardize predictors (z-scores)
    log("[STANDARDIZE] Converting predictors to z-scores...")
    
    # List columns to standardize (exclude uid, theta, and binary variables)
    cols_to_standardize = [
        'RAVLT_T', 'RAVLT_DR_T', 'RAVLT_Pct_Ret', 'BVMT_T', 'BVMT_Pct_Ret', 'NART_T', 'RPM_T',  # Cognitive
        'age', 'education',  # Demographics (not sex - it's binary)
        'DASS_Dep', 'DASS_Anx', 'DASS_Str', 'VR_Exp', 'Sleep'  # Self-report
    ]
    
    for col in cols_to_standardize:
        if col in merged.columns:
            # Calculate z-scores
            valid_data = merged[col].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                if std_val > 0:
                    merged[f'{col}_z'] = (merged[col] - mean_val) / std_val
                else:
                    merged[f'{col}_z'] = 0
                log(f"  - {col}: M={mean_val:.2f}, SD={std_val:.2f}")
    
    # Report predictor correlations with outcome
    log("[CORRELATIONS] Predictor correlations with theta:")
    z_cols = [col for col in merged.columns if col.endswith('_z')]
    for col in z_cols:
        if col in merged.columns:
            valid_mask = ~(merged[col].isna() | merged['theta'].isna())
            if valid_mask.sum() > 0:
                corr = merged.loc[valid_mask, col].corr(merged.loc[valid_mask, 'theta'])
                log(f"  - {col}: r={corr:.3f}")
    
    # Summary statistics
    log("[SUMMARY] Final merged dataset:")
    log(f"  - N participants: {len(merged)}")
    log(f"  - N columns: {len(merged.columns)}")
    log(f"  - Complete cases: {merged.dropna().shape[0]}")
    
    # Save output
    output_path = RQ_DIR / "data" / "step05_merged_predictors.csv"
    output_path.parent.mkdir(exist_ok=True)
    merged.to_csv(output_path, index=False)
    log(f"[SAVE] Saved merged data to {output_path}")
    
    log("[SUCCESS] Step 05 complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())