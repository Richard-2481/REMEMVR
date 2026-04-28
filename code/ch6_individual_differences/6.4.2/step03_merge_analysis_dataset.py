#!/usr/bin/env python3
"""merge_analysis_dataset: Merge domain theta scores with BVMT scores to create analysis-ready dataset"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_analysis_dataset.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Custom Merge Function (Parameter Mismatch Workaround)

def merge_theta_cognitive_custom(theta_df, cognitive_df, uid_col='UID'):
    """
    Custom merge function to work around parameter mismatch in tools.data.merge_theta_cognitive.
    
    Issue: tools.data.merge_theta_cognitive has parameters (theta_df, cognitive_df, how)
    But 4_analysis.yaml expects (theta_df, cognitive_df, uid_col).
    
    Solution: Implement custom merge logic per gcode_lessons.md #9.
    """
    log(f"Custom merge on column: {uid_col}")
    log(f"Theta data shape: {theta_df.shape}")
    log(f"Cognitive data shape: {cognitive_df.shape}")
    
    # Verify UID column exists in both dataframes
    if uid_col not in theta_df.columns:
        raise ValueError(f"UID column '{uid_col}' not found in theta_df. Available: {theta_df.columns.tolist()}")
    if uid_col not in cognitive_df.columns:
        raise ValueError(f"UID column '{uid_col}' not found in cognitive_df. Available: {cognitive_df.columns.tolist()}")
    
    # Check for duplicate UIDs
    theta_dups = theta_df[uid_col].duplicated().sum()
    cognitive_dups = cognitive_df[uid_col].duplicated().sum()
    if theta_dups > 0:
        raise ValueError(f"Duplicate UIDs in theta_df: {theta_dups} duplicates found")
    if cognitive_dups > 0:
        raise ValueError(f"Duplicate UIDs in cognitive_df: {cognitive_dups} duplicates found")
    
    # Inner join to ensure complete data only
    merged_df = pd.merge(theta_df, cognitive_df, on=uid_col, how='inner')
    
    log(f"Merged shape: {merged_df.shape}")
    log(f"Participants retained: {len(merged_df)}")
    
    # Verify no data loss (should have exactly 100 participants)
    expected_n = 100
    if len(merged_df) != expected_n:
        log(f"Expected {expected_n} participants, got {len(merged_df)}")
    
    return merged_df

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: merge_analysis_dataset")
        # Load Input Data

        log("Loading input data...")
        
        # Load Step 01 output: domain theta scores
        # Expected columns: UID, Where_mean, What_mean
        # Expected rows: ~100
        theta_input = pd.read_csv(RQ_DIR / "data/step01_domain_theta_scores.csv")
        log(f"step01_domain_theta_scores.csv ({len(theta_input)} rows, {len(theta_input.columns)} cols)")
        log(f"Theta data: {theta_input.columns.tolist()}")
        
        # Load Step 02 output: BVMT scores
        # Expected columns: UID, bvmt_total, bvmt_pct_ret
        # Expected rows: ~100
        bvmt_input = pd.read_csv(RQ_DIR / "data/step02_bvmt_scores.csv")
        log(f"step02_bvmt_scores.csv ({len(bvmt_input)} rows, {len(bvmt_input.columns)} cols)")
        log(f"BVMT data: {bvmt_input.columns.tolist()}")
        # Run Analysis Tool (Custom Merge Implementation)

        log("Running custom merge_theta_cognitive...")
        merged_data = merge_theta_cognitive_custom(
            theta_df=theta_input,
            cognitive_df=bvmt_input,
            uid_col="UID"  # Standard UID column for merge
        )
        log("Merge analysis complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 04 correlation analysis and Step 05 Steiger test

        log(f"Saving step03_analysis_dataset.csv...")
        # Output: step03_analysis_dataset.csv
        # Contains: Complete analysis dataset with domain theta and BVMT scores
        # Columns: UID, bvmt_total, Where_mean, What_mean
        merged_data.to_csv(RQ_DIR / "data/step03_analysis_dataset.csv", index=False, encoding='utf-8')
        log(f"step03_analysis_dataset.csv ({len(merged_data)} rows, {len(merged_data.columns)} cols)")
        
        # Verify expected column order
        expected_columns = ["UID", "bvmt_total", "bvmt_pct_ret", "Where_mean", "What_mean"]
        actual_columns = merged_data.columns.tolist()
        if actual_columns == expected_columns:
            log("Output columns match expected format")
        else:
            log(f"Column order mismatch. Expected: {expected_columns}, Actual: {actual_columns}")
        # Run Validation Tool
        # Validates: Complete dataset structure and data integrity
        # Threshold: 100 participants expected, no missing data allowed

        log("Running validate_dataframe_structure...")
        validation_result = validate_dataframe_structure(
            merged_data,
            expected_columns=expected_columns,
            expected_rows=100
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional manual validation checks
        log("Additional checks...")
        
        # Check for exactly 100 participants (no data loss)
        if len(merged_data) == 100:
            log("Participant count: PASS (100 retained)")
        else:
            log(f"Participant count: WARNING ({len(merged_data)} instead of 100)")
        
        # Check for no duplicate UIDs
        duplicate_count = merged_data['UID'].duplicated().sum()
        if duplicate_count == 0:
            log("Duplicate UIDs: PASS (none found)")
        else:
            log(f"Duplicate UIDs: FAIL ({duplicate_count} duplicates)")
            
        # Check for missing values
        missing_count = merged_data.isnull().sum().sum()
        if missing_count == 0:
            log("Missing values: PASS (none found)")
        else:
            log(f"Missing values: FAIL ({missing_count} missing)")
            
        # Check variable ranges
        # BVMT range: [0, 36] (theoretical max)
        bvmt_min, bvmt_max = merged_data['bvmt_total'].min(), merged_data['bvmt_total'].max()
        if 0 <= bvmt_min and bvmt_max <= 36:
            log(f"BVMT range: PASS ({bvmt_min:.1f} to {bvmt_max:.1f})")
        else:
            log(f"BVMT range: WARNING ({bvmt_min:.1f} to {bvmt_max:.1f}, expected [0,36])")
            
        # Theta range: [-3, 3] (typical IRT range)
        where_min, where_max = merged_data['Where_mean'].min(), merged_data['Where_mean'].max()
        what_min, what_max = merged_data['What_mean'].min(), merged_data['What_mean'].max()
        if -3 <= where_min and where_max <= 3:
            log(f"Where theta range: PASS ({where_min:.2f} to {where_max:.2f})")
        else:
            log(f"Where theta range: WARNING ({where_min:.2f} to {where_max:.2f}, expected [-3,3])")
        if -3 <= what_min and what_max <= 3:
            log(f"What theta range: PASS ({what_min:.2f} to {what_max:.2f})")
        else:
            log(f"What theta range: WARNING ({what_min:.2f} to {what_max:.2f}, expected [-3,3])")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)