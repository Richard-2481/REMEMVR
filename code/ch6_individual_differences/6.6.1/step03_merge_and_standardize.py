#!/usr/bin/env python3
"""Merge Datasets and Standardize Predictors: Merge cognitive test data with slope estimates and standardize continuous"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step03_merge_and_standardize.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Merge Datasets and Standardize Predictors")
        # Load Input Files

        log("Loading step01 cognitive tests...")
        cognitive_path = RQ_DIR / 'data' / 'step01_cognitive_tests.csv'

        if not cognitive_path.exists():
            log(f"Step 01 output not found: {cognitive_path}")
            log("Run step01_extract_cognitive_tests.py first")
            sys.exit(1)

        df_cognitive = pd.read_csv(cognitive_path)
        log(f"{cognitive_path.name} ({len(df_cognitive)} rows, {len(df_cognitive.columns)} cols)")

        log("Loading step02 slopes...")
        slopes_path = RQ_DIR / 'data' / 'step02_slopes_extracted.csv'

        if not slopes_path.exists():
            log(f"Step 02 output not found: {slopes_path}")
            log("Run step02_extract_slopes.py first")
            sys.exit(1)

        df_slopes = pd.read_csv(slopes_path)
        log(f"{slopes_path.name} ({len(df_slopes)} rows, {len(df_slopes.columns)} cols)")
        # Verify Required Columns
        # Per 4_analysis.yaml lines 188-191

        log("Checking required columns...")

        # Check cognitive test columns
        required_cognitive = ['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        missing_cognitive = [col for col in required_cognitive if col not in df_cognitive.columns]
        if missing_cognitive:
            log(f"Missing columns in cognitive data: {missing_cognitive}")
            sys.exit(1)

        # Check slopes columns
        required_slopes = ['UID', 'slope']
        missing_slopes = [col for col in required_slopes if col not in df_slopes.columns]
        if missing_slopes:
            log(f"Missing columns in slopes data: {missing_slopes}")
            sys.exit(1)

        log("All required columns present")
        # Merge Datasets on UID (Inner Join)
        # Per 4_analysis.yaml lines 182-183: merge_on: "UID", merge_how: "inner"

        log("Merging cognitive tests with slopes on UID (inner join)...")
        df_merged = pd.merge(df_cognitive, df_slopes, on='UID', how='inner')

        log(f"{len(df_merged)} participants after inner join")
        log(f"Cognitive data: {len(df_cognitive)} rows")
        log(f"Slopes data: {len(df_slopes)} rows")
        log(f"Merged data: {len(df_merged)} rows")

        if len(df_merged) == 0:
            log("Merge resulted in 0 rows - no matching UIDs")
            sys.exit(1)
        # Standardize Continuous Predictors
        # Per 4_analysis.yaml lines 183-185:
        # - standardize_columns: ["age", "education", "RAVLT_T", "BVMT_T", "RPM_T"]
        # - exclude_from_standardization: ["sex"]  (binary variable)
        # - standardization_method: "z_score"  (mean=0, SD=1)

        log("Computing z-scores for continuous predictors...")

        # Variables to standardize
        std_vars = ['age', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']

        for var in std_vars:
            if var not in df_merged.columns:
                log(f"Variable '{var}' not found for standardization")
                sys.exit(1)

            # Compute z-score: (x - mean) / SD
            var_mean = df_merged[var].mean()
            var_sd = df_merged[var].std()

            df_merged[f'{var}_std'] = (df_merged[var] - var_mean) / var_sd

            log(f"{var}: mean={var_mean:.4f}, SD={var_sd:.4f} -> {var}_std (z-score)")

        # sex stays binary (NOT standardized per 4_analysis.yaml line 184)
        log("sex column preserved as binary (NOT standardized)")
        # Create Final Output Dataset
        # Per 4_analysis.yaml lines 194-195:
        # Columns: UID, slope, age_std, sex, education_std, RAVLT_T_std, BVMT_T_std, RPM_T_std

        log("Creating final analysis dataset...")
        output_cols = [
            'UID', 'slope',
            'age_std', 'sex', 'education_std',
            'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
            'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std'
        ]

        missing_output = [col for col in output_cols if col not in df_merged.columns]
        if missing_output:
            log(f"Missing columns for output: {missing_output}")
            sys.exit(1)

        output_df = df_merged[output_cols].copy()
        log(f"Output dataset ({len(output_df)} rows, {len(output_df.columns)} cols)")
        # Validate Standardization
        # Per 4_analysis.yaml lines 200-205:
        # - expected_means: 0.0
        # - expected_sds: 1.0
        # - tolerance: 0.01

        log("Validating standardization...")

        tolerance = 0.01
        std_cols = ['age_std', 'education_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std', 'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']

        all_valid = True
        for col in std_cols:
            col_mean = output_df[col].mean()
            col_sd = output_df[col].std()

            # Check mean ~= 0
            mean_ok = abs(col_mean) < tolerance
            # Check SD ~= 1
            sd_ok = abs(col_sd - 1.0) < tolerance

            if mean_ok and sd_ok:
                log(f"{col}: mean={col_mean:.6f}, SD={col_sd:.6f}")
            else:
                log(f"{col}: mean={col_mean:.6f} (expected ~0), SD={col_sd:.6f} (expected ~1)")
                all_valid = False

        if not all_valid:
            log("Standardization validation failed")
            sys.exit(1)

        log("All standardized variables have mean~=0, SD~=1")

        # Check for missing values
        missing_counts = output_df.isnull().sum()
        if missing_counts.sum() == 0:
            log("No missing values in output dataset")
        else:
            log(f"Missing values detected:")
            for col in missing_counts[missing_counts > 0].index:
                log(f"  {col}: {missing_counts[col]} missing")
            sys.exit(1)

        # Check participant count
        expected_n = 100
        actual_n = len(output_df)
        if actual_n == expected_n:
            log(f"Participant count: {actual_n} (expected {expected_n})")
        else:
            log(f"Participant count: {actual_n} (expected {expected_n})")
        # Save Output
        # Output: results/ch7/7.6.1/data/step03_analysis_input.csv
        # Per 4_analysis.yaml lines 194-196

        output_path = RQ_DIR / 'data' / 'step03_analysis_input.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"{len(output_df)} participants ready for regression analysis")

        # Log summary statistics
        log("Final dataset statistics:")
        log(f"  Slope: mean={output_df['slope'].mean():.4f}, SD={output_df['slope'].std():.4f}")
        log(f"  age_std: mean={output_df['age_std'].mean():.6f}, SD={output_df['age_std'].std():.6f}")
        log(f"  sex: mean={output_df['sex'].mean():.4f} (proportion)")
        log(f"  education_std: mean={output_df['education_std'].mean():.6f}, SD={output_df['education_std'].std():.6f}")
        log(f"  RAVLT_T_std: mean={output_df['RAVLT_T_std'].mean():.6f}, SD={output_df['RAVLT_T_std'].std():.6f}")
        log(f"  BVMT_T_std: mean={output_df['BVMT_T_std'].mean():.6f}, SD={output_df['BVMT_T_std'].std():.6f}")
        log(f"  RPM_T_std: mean={output_df['RPM_T_std'].mean():.6f}, SD={output_df['RPM_T_std'].std():.6f}")
        log(f"  RAVLT_Pct_Ret_T_std: mean={output_df['RAVLT_Pct_Ret_T_std'].mean():.6f}, SD={output_df['RAVLT_Pct_Ret_T_std'].std():.6f}")
        log(f"  BVMT_Pct_Ret_T_std: mean={output_df['BVMT_Pct_Ret_T_std'].mean():.6f}, SD={output_df['BVMT_Pct_Ret_T_std'].std():.6f}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
