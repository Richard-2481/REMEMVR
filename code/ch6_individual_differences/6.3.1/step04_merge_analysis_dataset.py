#!/usr/bin/env python3
"""Merge Analysis Dataset: Merge confidence theta scores with cognitive tests and demographics to create"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_merge_analysis_dataset.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Merge Analysis Dataset")
        # Load Input Data

        log("Loading confidence theta scores...")
        # Load confidence theta scores (100 participants expected)
        # Expected columns: ['UID', 'confidence_theta', 'se_theta']
        # Expected rows: ~100
        confidence_df = pd.read_csv(RQ_DIR / "data/step01_confidence_theta.csv")
        log(f"step01_confidence_theta.csv ({len(confidence_df)} rows, {len(confidence_df.columns)} cols)")
        
        log("Loading cognitive test scores...")
        # Load cognitive test T-scores and demographics (100 participants expected)
        # Expected columns: ['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'age', 'sex', 'education']
        # Expected rows: ~100
        # Note: Step 03 was skipped because Step 02 already created T-scores
        cognitive_df = pd.read_csv(RQ_DIR / "data/step02_cognitive_tests.csv")
        log(f"step02_cognitive_tests.csv ({len(cognitive_df)} rows, {len(cognitive_df.columns)} cols)")
        # Run Analysis Tool (pandas merge)

        log("Running pd.merge for inner join on UID...")
        
        # Merge datasets using inner join on UID
        # This ensures only participants with both confidence and cognitive data are included
        merged_df = pd.merge(
            left=confidence_df,
            right=cognitive_df,
            on="UID",  # merge_key parameter
            how="inner"  # merge_type parameter  
        )
        log(f"Inner join complete: {len(merged_df)} participants with complete data")
        
        # Drop se_theta column as it's not needed for regression analysis
        # Final dataset should have: UID, confidence_theta, RAVLT_T, BVMT_T, RPM_T, age, sex, education
        if 'se_theta' in merged_df.columns:
            merged_df = merged_df.drop(columns=['se_theta'])
            log("Removed se_theta column (not needed for regression)")
        
        # Check for missing values and report
        missing_counts = merged_df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            log(f"Found {total_missing} missing values:")
            for col, count in missing_counts.items():
                if count > 0:
                    log(f"  {col}: {count} missing")
            
            # Drop rows with any missing values (drop_missing parameter)
            initial_rows = len(merged_df)
            merged_df = merged_df.dropna()
            final_rows = len(merged_df)
            
            if final_rows < initial_rows:
                log(f"Dropped {initial_rows - final_rows} rows with missing values")
                log(f"Final complete-case dataset: {final_rows} participants")
        else:
            log("No missing values found - dataset is complete")

        log("Analysis complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 05 (hierarchical regression) and subsequent analyses

        log(f"Saving step04_analysis_dataset.csv...")
        # Output: step04_analysis_dataset.csv
        # Contains: Merged analysis-ready dataset with complete cases
        # Columns: ['UID', 'confidence_theta', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'age', 'sex', 'education']
        output_path = RQ_DIR / "data/step04_analysis_dataset.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step04_analysis_dataset.csv ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        
        # Log final dataset summary
        log(f"Final dataset columns: {list(merged_df.columns)}")
        log(f"Sample size: {len(merged_df)} participants")
        # Run Validation Tool
        # Validates: DataFrame has expected structure and sample size
        # Threshold: Final sample size >= 90 participants

        log("Running validate_dataframe_structure...")
        
        # Expected columns for final dataset
        expected_cols = ['UID', 'confidence_theta', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        
        validation_result = validate_dataframe_structure(
            df=merged_df,
            expected_rows=(90, 105),  # Range: minimum 90, maximum ~105
            expected_columns=expected_cols,  # Required columns
            column_types=None  # No specific type checking needed
        )

        # Report validation results
        if isinstance(validation_result, dict):
            if validation_result.get('valid', False):
                log("Dataset structure validation PASSED")
                for check_name, check_result in validation_result.get('checks', {}).items():
                    status = "PASS" if check_result else "FAIL"
                    log(f"{check_name}: {status}")
            else:
                log(f"Dataset structure validation FAILED: {validation_result.get('message', 'Unknown error')}")
        else:
            log(f"{validation_result}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)