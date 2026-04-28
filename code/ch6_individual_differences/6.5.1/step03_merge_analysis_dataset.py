#!/usr/bin/env python3
"""merge_analysis_dataset: Combine self-report measures and theta scores into analysis-ready dataset"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Import scipy for standardization
from scipy import stats

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_analysis_dataset.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: merge_analysis_dataset")
        # Load Input Data

        log("Loading input data...")
        
        # Load step01_self_report_data.csv
        # Expected columns: ['UID', 'Education', 'VR_Experience', 'Typical_Sleep', 'Age']
        # Expected rows: ~100 participants
        self_report = pd.read_csv(RQ_DIR / "data/step01_self_report_data.csv")
        log(f"step01_self_report_data.csv ({len(self_report)} rows, {len(self_report.columns)} cols)")
        
        # Load step02_theta_scores.csv
        # Expected columns: ['UID', 'theta_all', 'theta_SE']
        # Expected rows: ~100 participants
        theta_scores = pd.read_csv(RQ_DIR / "data/step02_theta_scores.csv")
        log(f"step02_theta_scores.csv ({len(theta_scores)} rows, {len(theta_scores.columns)} cols)")
        # Run Analysis (Merge and Standardize)

        log("Merging datasets...")
        # Merge on UID (inner join for complete cases only)
        analysis_df = pd.merge(self_report, theta_scores, on='UID', how='inner')
        log(f"Inner join complete: {len(analysis_df)} complete cases from {len(self_report)} + {len(theta_scores)}")

        log("Standardizing predictors...")
        # Standardize predictors (z-scores)
        predictors = ['Education', 'VR_Experience', 'Typical_Sleep', 'Age']
        for pred in predictors:
            analysis_df[f'{pred}_z'] = stats.zscore(analysis_df[pred])
            log(f"{pred} -> {pred}_z (mean=0, std=1)")

        log("Creating correlation matrix...")
        # Create correlation matrix for multicollinearity screening
        corr_matrix = analysis_df[[f'{p}_z' for p in predictors]].corr()
        
        # Check multicollinearity
        max_corr = corr_matrix.abs().where(~np.eye(len(corr_matrix), dtype=bool)).max().max()
        log(f"Maximum absolute correlation: {max_corr:.3f}")

        log("Analysis complete")
        # Save Analysis Outputs
        # These outputs will be used by: Hierarchical regression analysis (step04)

        log("Saving analysis outputs...")
        
        # Output: step03_analysis_dataset.csv
        # Contains: Analysis-ready dataset with standardized predictors
        # Columns: ['UID', 'theta_all', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z', 'Age_z']
        final_cols = ['UID', 'theta_all'] + [f'{p}_z' for p in predictors]
        output_path = RQ_DIR / "data/step03_analysis_dataset.csv"
        analysis_df[final_cols].to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} rows, {len(final_cols)} cols)")

        # Output: step03_correlation_matrix.csv
        # Contains: Predictor correlation matrix for multicollinearity assessment
        # Columns: 4x4 matrix of standardized predictor correlations
        corr_path = RQ_DIR / "data/step03_correlation_matrix.csv"
        corr_matrix.to_csv(corr_path, encoding='utf-8')
        log(f"step03_correlation_matrix.csv ({len(corr_matrix)} rows, {len(corr_matrix.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns exist in final analysis dataset
        # Threshold: All required columns must be present

        log("Running validate_data_columns...")
        
        # CRITICAL LESSON #15: validate_data_columns expects DataFrame, not path
        # Wrong: validate_data_columns(df_path=..., required_columns=...)
        # Correct: Load DataFrame first, then pass it
        required_columns = ['UID', 'theta_all', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z', 'Age_z']
        validation_result = validate_data_columns(
            df=analysis_df[final_cols],  # Pass DataFrame directly, not path
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            
            if validation_result.get('valid', False):
                log("PASS - All required columns present")
            else:
                missing = validation_result.get('missing_columns', [])
                log(f"FAIL - Missing columns: {missing}")
        else:
            log(f"{validation_result}")

        # Additional validation checks
        log("Additional data quality checks...")
        assert len(analysis_df) >= 95, f"Insufficient complete cases: {len(analysis_df)} (need >=95)"
        log(f"Sample size check: {len(analysis_df)} >= 95 complete cases")
        
        assert max_corr < 0.90, f"Multicollinearity detected: max r={max_corr:.3f} (threshold <0.90)"
        log(f"Multicollinearity check: max r={max_corr:.3f} < 0.90")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)