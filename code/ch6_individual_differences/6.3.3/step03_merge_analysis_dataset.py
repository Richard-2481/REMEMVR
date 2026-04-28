#!/usr/bin/env python3
"""Merge Analysis Dataset: Merge cognitive tests and HCE rates into analysis dataset with centered predictors"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data import merge_theta_cognitive

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_analysis_dataset.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Merge Analysis Dataset")
        # Load Input Data

        log("Loading cognitive test data...")
        cognitive_df = pd.read_csv(RQ_DIR / "data" / "step01_cognitive_tests.csv")
        log(f"step01_cognitive_tests.csv ({len(cognitive_df)} rows, {len(cognitive_df.columns)} cols)")
        log(f"Cognitive columns: {cognitive_df.columns.tolist()}")
        
        log("Loading HCE rate data...")
        hce_df = pd.read_csv(RQ_DIR / "data" / "step02_hce_rates.csv")
        log(f"step02_hce_rates.csv ({len(hce_df)} rows, {len(hce_df.columns)} cols)")
        log(f"HCE columns: {hce_df.columns.tolist()}")
        # Handle UID Column Case Differences
        # Tool expects both DataFrames to have consistent column naming
        # cognitive_df has 'uid', hce_df has 'UID' - need to standardize
        
        log("Standardizing UID column names...")
        # The merge_theta_cognitive function expects 'uid' in lowercase
        # So rename HCE data 'UID' to 'uid' to match cognitive data
        hce_df = hce_df.rename(columns={'UID': 'uid'})
        log(f"HCE data now has uid column (was UID)")
        
        # Both datasets now have 'uid' in lowercase which the function expects
        # theta_df parameter will get hce_df, cognitive_df parameter gets cognitive_df
        # Run Analysis Tool (Merge Datasets)

        log("Running merge_theta_cognitive...")
        merged_df = merge_theta_cognitive(
            theta_df=hce_df,       # HCE rates (theta_df parameter used for first dataset)
            cognitive_df=cognitive_df,  # Cognitive tests
            how="inner"            # Inner join - only participants with both HCE and cognitive data
        )
        log("Merge complete")
        log(f"Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} cols")
        log(f"Merged columns: {merged_df.columns.tolist()}")
        # Center Continuous Predictors
        # Center age, RAVLT, BVMT, RPM around sample means
        # Creates _c suffix versions for use in regression
        
        log("Creating centered predictors...")
        
        # Continuous predictors to center
        predictors_to_center = ['age', 'ravlt_total', 'bvmt_total', 'rpm_score', 'ravlt_pct_ret', 'bvmt_pct_ret']

        # Mapping from raw name to centered name
        center_name_map = {
            'age': 'age_c',
            'ravlt_total': 'ravlt_c',
            'bvmt_total': 'bvmt_c',
            'rpm_score': 'rpm_c',
            'ravlt_pct_ret': 'ravlt_pct_ret_c',
            'bvmt_pct_ret': 'bvmt_pct_ret_c',
        }

        for predictor in predictors_to_center:
            if predictor in merged_df.columns:
                # Calculate sample mean
                sample_mean = merged_df[predictor].mean()
                # Create centered version
                centered_col = center_name_map[predictor]
                merged_df[centered_col] = merged_df[predictor] - sample_mean
                log(f"{predictor} -> {centered_col} (mean={sample_mean:.3f})")
            else:
                log(f"Predictor {predictor} not found in merged data")
        # Standardize Final Column Names
        # Ensure final output has 'uid' (lowercase) for consistency with other RQ outputs
        
        log("Standardizing final column names...")
        merged_df = merged_df.rename(columns={'UID': 'uid'})
        log(f"Final columns: {merged_df.columns.tolist()}")
        # Save Analysis Output
        # This output will be used by: Step 04 hierarchical regression analysis

        log("Saving step03_analysis_dataset.csv...")
        # Output: step03_analysis_dataset.csv
        # Contains: Merged HCE + cognitive data with centered predictors ready for regression
        # Columns: uid, hce_rate, cognitive variables, and centered versions
        output_path = RQ_DIR / "data" / "step03_analysis_dataset.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_analysis_dataset.csv ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns are present for downstream regression analysis
        # Required: uid, hce_rate, centered predictors, and sex

        log("Running validate_data_columns...")
        required_columns = ["uid", "hce_rate", "ravlt_c", "bvmt_c", "rpm_c", "age_c", "sex", "ravlt_pct_ret_c", "bvmt_pct_ret_c"]
        
        validation_result = validate_data_columns(
            df=merged_df,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional data quality checks
        log("Additional quality checks...")
        log(f"Missing data summary:")
        for col in merged_df.columns:
            missing_count = merged_df[col].isnull().sum()
            if missing_count > 0:
                log(f"{col}: {missing_count} missing ({missing_count/len(merged_df)*100:.1f}%)")
            else:
                log(f"{col}: 0 missing")

        # Check centered variables have mean ~0
        log(f"Centered variable means (should be ~0):")
        centered_vars = ['age_c', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        for var in centered_vars:
            if var in merged_df.columns:
                mean_val = merged_df[var].mean()
                log(f"{var}: mean={mean_val:.6f}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)