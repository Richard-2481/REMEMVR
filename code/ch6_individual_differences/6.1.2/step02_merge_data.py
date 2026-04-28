#!/usr/bin/env python3
"""
Step 02: Merge Datasets
RQ: ch7/7.1.2
Purpose: Merge cognitive tests with random effects for regression analysis
Output: results/ch7/7.1.2/data/step02_regression_input.csv
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.2
LOG_FILE = RQ_DIR / "logs" / "step02_merge_data.log"
OUTPUT_FILE = RQ_DIR / "data" / "step02_regression_input.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 02: Merge Datasets")
        # Load random effects data
        log("Loading random effects from Step 00...")
        random_effects_file = RQ_DIR / "data" / "step00_random_effects.csv"
        random_effects_df = pd.read_csv(random_effects_file)
        log(f"Loaded random effects: {random_effects_df.shape[0]} participants")
        log(f"Random effects columns: {list(random_effects_df.columns)}")
        # Load cognitive test data
        log("Loading cognitive tests from Step 01...")
        cognitive_file = RQ_DIR / "data" / "step01_cognitive_tests.csv"
        cognitive_df = pd.read_csv(cognitive_file)
        log(f"Loaded cognitive tests: {cognitive_df.shape[0]} participants")
        log(f"Cognitive test columns: {list(cognitive_df.columns)}")
        
        # Select only needed cognitive tests (exclude NART per concept)
        cognitive_subset = cognitive_df[['UID', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']].copy()
        log("Selected cognitive tests: RAVLT_T, RAVLT_Pct_Ret_T, BVMT_T, BVMT_Pct_Ret_T, RPM_T (excluding NART)")
        # Merge datasets on UID
        log("Merging random effects with cognitive tests...")
        
        # Use inner join to ensure complete cases
        merged_df = pd.merge(
            random_effects_df,
            cognitive_subset,
            on='UID',
            how='inner',
            validate='one_to_one'
        )
        
        log(f"Merged dataset: {merged_df.shape[0]} participants x {merged_df.shape[1]} columns")
        log(f"Final columns: {list(merged_df.columns)}")
        # Validate merged data
        log("Checking merged dataset...")
        
        # Check for missing values
        missing_counts = merged_df.isnull().sum()
        if missing_counts.sum() > 0:
            log("Missing values found:")
            for col in missing_counts[missing_counts > 0].index:
                log(f"  - {col}: {missing_counts[col]} missing")
        else:
            log("No missing values in merged dataset")
        
        # Check expected columns
        expected_cols = ['UID', 'intercept', 'slope', 'se_intercept', 'se_slope',
                        'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        missing_cols = set(expected_cols) - set(merged_df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        log(f"All expected columns present")
        
        # Check value ranges
        log("Checking value ranges...")
        log(f"  - Intercepts: [{merged_df['intercept'].min():.3f}, {merged_df['intercept'].max():.3f}]")
        log(f"  - Slopes: [{merged_df['slope'].min():.3f}, {merged_df['slope'].max():.3f}]")
        log(f"  - RAVLT_T: [{merged_df['RAVLT_T'].min():.1f}, {merged_df['RAVLT_T'].max():.1f}]")
        log(f"  - RAVLT_Pct_Ret_T: [{merged_df['RAVLT_Pct_Ret_T'].min():.1f}, {merged_df['RAVLT_Pct_Ret_T'].max():.1f}]")
        log(f"  - BVMT_T: [{merged_df['BVMT_T'].min():.1f}, {merged_df['BVMT_T'].max():.1f}]")
        log(f"  - BVMT_Pct_Ret_T: [{merged_df['BVMT_Pct_Ret_T'].min():.1f}, {merged_df['BVMT_Pct_Ret_T'].max():.1f}]")
        log(f"  - RPM_T: [{merged_df['RPM_T'].min():.1f}, {merged_df['RPM_T'].max():.1f}]")
        # Save merged dataset
        log(f"Saving merged dataset to {OUTPUT_FILE}...")
        merged_df.to_csv(OUTPUT_FILE, index=False)
        log(f"{merged_df.shape[0]} participants x {merged_df.shape[1]} columns")
        
        # Final validation
        from tools.validation import validate_data_columns
        
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(merged_df, expected_cols)
        
        if validation_result['valid']:
            log("PASSED - All required columns present")
        else:
            raise ValueError(f"Validation failed: {validation_result['message']}")
        
        log("Step 02 complete - Datasets merged successfully")
        
    except Exception as e:
        log(f"{str(e)}")
        raise