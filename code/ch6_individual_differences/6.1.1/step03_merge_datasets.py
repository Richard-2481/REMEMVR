#!/usr/bin/env python3
"""Merge Datasets: Merge cognitive tests with theta means to create final analysis dataset"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_datasets.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Merge Datasets")
        # Load Input Data

        log("Loading cognitive test data from step01...")
        # Load cognitive test T-scores
        # Expected columns: UID, RAVLT_T, RAVLT_DR_T, RAVLT_Pct_Ret_T, BVMT_T, BVMT_Pct_Ret_T, NART_T, RPM_T
        # Expected rows: ~100 participants
        cognitive_tests = pd.read_csv(RQ_DIR / "data" / "step01_cognitive_tests.csv")
        log(f"Cognitive tests ({len(cognitive_tests)} rows, {len(cognitive_tests.columns)} cols)")
        log(f"Cognitive test columns: {cognitive_tests.columns.tolist()}")
        
        log("Loading theta means data from step02...")
        # Load theta means per participant
        # Expected columns: UID, theta_mean
        # Expected rows: ~100 participants
        theta_means = pd.read_csv(RQ_DIR / "data" / "step02_theta_means.csv")
        log(f"Theta means ({len(theta_means)} rows, {len(theta_means.columns)} cols)")
        log(f"Theta means columns: {theta_means.columns.tolist()}")
        # Handle Missing Data (NART Specifically)
        # User mentioned 3 participants with missing NART
        # Check for and handle missing data before merge
        
        log("Checking for missing data in cognitive tests...")
        missing_nart = cognitive_tests['NART_T'].isna().sum()
        missing_any = cognitive_tests.isna().any(axis=1).sum()
        log(f"Participants missing NART: {missing_nart}")
        log(f"Participants missing any cognitive test: {missing_any}")
        
        if missing_nart > 0:
            log(f"Found {missing_nart} participants with missing NART")
            # For regression analysis, we need complete cases
            # Remove participants with missing NART for analysis
            cognitive_complete = cognitive_tests.dropna()
            log(f"After removing missing NART: {len(cognitive_complete)} participants remain")
        else:
            cognitive_complete = cognitive_tests.copy()
            log("No missing NART data - all participants retained")
        # Merge Datasets
        # Custom merge logic instead of tools.data.merge_theta_cognitive
        # Inner join to ensure only participants with both cognitive and theta data
        
        log("Merging cognitive tests with theta means on UID...")
        # Inner join on UID column
        # This ensures only participants with both datasets are retained
        merged_data = pd.merge(
            cognitive_complete,
            theta_means,
            on='UID',
            how='inner'
        )
        log(f"Successfully merged datasets: {len(merged_data)} participants")
        
        # Check merge quality
        n_cognitive = len(cognitive_complete)
        n_theta = len(theta_means)
        n_merged = len(merged_data)
        log(f"Merge retention: {n_merged}/{n_cognitive} cognitive, {n_merged}/{n_theta} theta")
        
        if n_merged < n_cognitive * 0.9:  # Less than 90% retention
            log(f"Low merge retention: {n_merged}/{n_cognitive} = {n_merged/n_cognitive:.2%}")
        else:
            log(f"Good merge retention: {n_merged}/{n_cognitive} = {n_merged/n_cognitive:.2%}")
        # Final Data Validation
        # Ensure we have complete cases with all required columns
        
        log("Validating merged dataset...")
        expected_cols = ['UID', 'RAVLT_T', 'RAVLT_DR_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'NART_T', 'RPM_T', 'theta_mean']
        
        # Check all expected columns present
        missing_cols = [col for col in expected_cols if col not in merged_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for any missing values in final dataset
        missing_any_final = merged_data[expected_cols].isna().any(axis=1).sum()
        if missing_any_final > 0:
            log(f"{missing_any_final} participants have missing values in final dataset")
            merged_data = merged_data[expected_cols].dropna()
            log(f"After removing incomplete cases: {len(merged_data)} participants")
        
        # Final dataset summary
        log(f"Final analysis dataset: {len(merged_data)} participants")
        log(f"Columns: {merged_data.columns.tolist()}")
        log(f"Theta range: [{merged_data['theta_mean'].min():.3f}, {merged_data['theta_mean'].max():.3f}]")
        # Save Analysis Output
        # Save merged dataset for downstream regression analysis
        
        output_path = RQ_DIR / "data" / "step03_merged_analysis.csv"
        log(f"Saving merged analysis dataset to {output_path}...")
        merged_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(merged_data)} rows, {len(merged_data.columns)} cols)")
        # Run Validation Tool
        # Validate merged dataset meets criteria for regression analysis
        
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=merged_data,
            required_columns=expected_cols
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
        
        # Additional validation checks
        if len(merged_data) >= 90:
            log("Sample size adequate (N >= 90)")
        elif len(merged_data) >= 80:
            log("Sample size acceptable (N >= 80) but less than target")
        else:
            log(f"Sample size low (N = {len(merged_data)} < 80)")
        
        if merged_data[expected_cols].isna().sum().sum() == 0:
            log("Complete cases only - no missing data")
        else:
            log("Missing data detected in final dataset")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)