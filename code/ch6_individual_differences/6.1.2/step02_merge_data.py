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

# Add project root to path for imports
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
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("[START] Step 02: Merge Datasets")
        
        # =========================================================================
        # STEP 1: Load random effects data
        # =========================================================================
        log("[DATA] Loading random effects from Step 00...")
        random_effects_file = RQ_DIR / "data" / "step00_random_effects.csv"
        random_effects_df = pd.read_csv(random_effects_file)
        log(f"[DATA] Loaded random effects: {random_effects_df.shape[0]} participants")
        log(f"[DATA] Random effects columns: {list(random_effects_df.columns)}")
        
        # =========================================================================
        # STEP 2: Load cognitive test data
        # =========================================================================
        log("[DATA] Loading cognitive tests from Step 01...")
        cognitive_file = RQ_DIR / "data" / "step01_cognitive_tests.csv"
        cognitive_df = pd.read_csv(cognitive_file)
        log(f"[DATA] Loaded cognitive tests: {cognitive_df.shape[0]} participants")
        log(f"[DATA] Cognitive test columns: {list(cognitive_df.columns)}")
        
        # Select only needed cognitive tests (exclude NART per concept)
        cognitive_subset = cognitive_df[['UID', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']].copy()
        log("[DATA] Selected cognitive tests: RAVLT_T, RAVLT_Pct_Ret_T, BVMT_T, BVMT_Pct_Ret_T, RPM_T (excluding NART)")
        
        # =========================================================================
        # STEP 3: Merge datasets on UID
        # =========================================================================
        log("[MERGE] Merging random effects with cognitive tests...")
        
        # Use inner join to ensure complete cases
        merged_df = pd.merge(
            random_effects_df,
            cognitive_subset,
            on='UID',
            how='inner',
            validate='one_to_one'
        )
        
        log(f"[MERGE] Merged dataset: {merged_df.shape[0]} participants x {merged_df.shape[1]} columns")
        log(f"[MERGE] Final columns: {list(merged_df.columns)}")
        
        # =========================================================================
        # STEP 4: Validate merged data
        # =========================================================================
        log("[VALIDATION] Checking merged dataset...")
        
        # Check for missing values
        missing_counts = merged_df.isnull().sum()
        if missing_counts.sum() > 0:
            log("[WARNING] Missing values found:")
            for col in missing_counts[missing_counts > 0].index:
                log(f"  - {col}: {missing_counts[col]} missing")
        else:
            log("[VALIDATION] No missing values in merged dataset")
        
        # Check expected columns
        expected_cols = ['UID', 'intercept', 'slope', 'se_intercept', 'se_slope',
                        'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        missing_cols = set(expected_cols) - set(merged_df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        log(f"[VALIDATION] All expected columns present")
        
        # Check value ranges
        log("[VALIDATION] Checking value ranges...")
        log(f"  - Intercepts: [{merged_df['intercept'].min():.3f}, {merged_df['intercept'].max():.3f}]")
        log(f"  - Slopes: [{merged_df['slope'].min():.3f}, {merged_df['slope'].max():.3f}]")
        log(f"  - RAVLT_T: [{merged_df['RAVLT_T'].min():.1f}, {merged_df['RAVLT_T'].max():.1f}]")
        log(f"  - RAVLT_Pct_Ret_T: [{merged_df['RAVLT_Pct_Ret_T'].min():.1f}, {merged_df['RAVLT_Pct_Ret_T'].max():.1f}]")
        log(f"  - BVMT_T: [{merged_df['BVMT_T'].min():.1f}, {merged_df['BVMT_T'].max():.1f}]")
        log(f"  - BVMT_Pct_Ret_T: [{merged_df['BVMT_Pct_Ret_T'].min():.1f}, {merged_df['BVMT_Pct_Ret_T'].max():.1f}]")
        log(f"  - RPM_T: [{merged_df['RPM_T'].min():.1f}, {merged_df['RPM_T'].max():.1f}]")
        
        # =========================================================================
        # STEP 5: Save merged dataset
        # =========================================================================
        log(f"[SAVE] Saving merged dataset to {OUTPUT_FILE}...")
        merged_df.to_csv(OUTPUT_FILE, index=False)
        log(f"[SAVED] {merged_df.shape[0]} participants x {merged_df.shape[1]} columns")
        
        # Final validation
        from tools.validation import validate_data_columns
        
        log("[VALIDATION] Running validate_data_columns...")
        validation_result = validate_data_columns(merged_df, expected_cols)
        
        if validation_result['valid']:
            log("[VALIDATION] PASSED - All required columns present")
        else:
            raise ValueError(f"Validation failed: {validation_result['message']}")
        
        log("[SUCCESS] Step 02 complete - Datasets merged successfully")
        
    except Exception as e:
        log(f"[ERROR] {str(e)}")
        raise