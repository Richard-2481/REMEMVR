#!/usr/bin/env python3
"""merge_and_check_assumptions: Merge calibration quality metrics from Ch6 with cognitive test data from dfnonvr.csv,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_and_check_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Assumption Testing Functions

def check_normality(data: pd.Series, name: str) -> Dict[str, Any]:
    """Check normality using Shapiro-Wilk test."""
    try:
        statistic, p_value = stats.shapiro(data.dropna())
        result = "NORMAL" if p_value > 0.05 else "NON_NORMAL"
        return {
            'test': f'Shapiro_Wilk_{name}',
            'statistic': statistic,
            'p_value': p_value,
            'result': result
        }
    except Exception as e:
        return {
            'test': f'Shapiro_Wilk_{name}',
            'statistic': np.nan,
            'p_value': np.nan,
            'result': f'ERROR: {str(e)}'
        }

def check_linearity(x: pd.Series, y: pd.Series, x_name: str, y_name: str) -> Dict[str, Any]:
    """Check linearity using Pearson correlation (preliminary check)."""
    try:
        # Remove rows with missing values
        valid_mask = ~(x.isna() | y.isna())
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            return {
                'test': f'Linearity_{x_name}_vs_{y_name}',
                'statistic': np.nan,
                'p_value': np.nan,
                'result': 'ERROR: Insufficient data'
            }
        
        correlation, p_value = stats.pearsonr(x_valid, y_valid)
        result = "SIGNIFICANT" if p_value < 0.05 else "NON_SIGNIFICANT"
        
        return {
            'test': f'Linearity_{x_name}_vs_{y_name}',
            'statistic': correlation,
            'p_value': p_value,
            'result': result
        }
    except Exception as e:
        return {
            'test': f'Linearity_{x_name}_vs_{y_name}',
            'statistic': np.nan,
            'p_value': np.nan,
            'result': f'ERROR: {str(e)}'
        }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: merge_and_check_assumptions")
        # Load Input Data

        log("Loading calibration metrics from step01...")
        calibration_df = pd.read_csv(RQ_DIR / "data" / "step01_calibration_metrics.csv")
        log(f"step01_calibration_metrics.csv ({len(calibration_df)} rows, {len(calibration_df.columns)} cols)")
        log(f"Calibration metrics columns: {calibration_df.columns.tolist()}")

        log("Loading cognitive test data from step02...")
        cognitive_df = pd.read_csv(RQ_DIR / "data" / "step02_cognitive_tests.csv")
        log(f"step02_cognitive_tests.csv ({len(cognitive_df)} rows, {len(cognitive_df.columns)} cols)")
        log(f"Cognitive test columns: {cognitive_df.columns.tolist()}")
        # Merge Datasets

        log("Merging calibration and cognitive data on UID...")
        
        # Check for UID overlap before merging
        calib_uids = set(calibration_df['UID'].unique())
        cognitive_uids = set(cognitive_df['UID'].unique())
        overlap_uids = calib_uids & cognitive_uids
        
        log(f"Calibration UIDs: {len(calib_uids)}")
        log(f"Cognitive test UIDs: {len(cognitive_uids)}")
        log(f"Overlapping UIDs: {len(overlap_uids)}")
        
        if len(overlap_uids) == 0:
            raise ValueError("No overlapping UIDs found between calibration and cognitive data")
        
        # Perform inner join on UID
        analysis_dataset = pd.merge(
            calibration_df, 
            cognitive_df, 
            on='UID', 
            how='inner',
            suffixes=('_calib', '_cog')
        )
        
        log(f"Analysis dataset created ({len(analysis_dataset)} rows, {len(analysis_dataset.columns)} cols)")
        log(f"Final columns: {analysis_dataset.columns.tolist()}")

        # Check for missing data
        missing_summary = analysis_dataset.isnull().sum()
        if missing_summary.sum() > 0:
            log("Missing data found:")
            for col, missing_count in missing_summary.items():
                if missing_count > 0:
                    log(f"  {col}: {missing_count} missing ({missing_count/len(analysis_dataset)*100:.1f}%)")
        else:
            log("No missing data in merged dataset")
        # Save Analysis Dataset
        # Output: results/ch7/7.3.2/data/step03_analysis_dataset.csv
        # Contains: Complete merged data for regression analysis
        # Columns: ['UID', 'calibration_quality', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'age', 'sex', 'education']

        output_path = RQ_DIR / "data" / "step03_analysis_dataset.csv"
        log(f"Saving analysis dataset to {output_path}...")
        analysis_dataset.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_analysis_dataset.csv ({len(analysis_dataset)} rows, {len(analysis_dataset.columns)} cols)")
        # Run Assumption Checks
        # Validates: Basic normality and linearity assumptions for regression
        # Threshold: Standard alpha = 0.05 for preliminary checks

        log("Running preliminary assumption checks...")
        
        assumption_results = []
        
        # Check normality of outcome variable (calibration_quality)
        log("Checking normality of calibration_quality...")
        normality_result = check_normality(analysis_dataset['calibration_quality'], 'calibration_quality')
        assumption_results.append(normality_result)
        log(f"{normality_result['test']}: {normality_result['result']} (p={normality_result['p_value']:.4f})")
        
        # Check normality of key predictors
        predictors_to_test = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age']
        for predictor in predictors_to_test:
            if predictor in analysis_dataset.columns:
                log(f"Checking normality of {predictor}...")
                pred_normality = check_normality(analysis_dataset[predictor], predictor)
                assumption_results.append(pred_normality)
                log(f"{pred_normality['test']}: {pred_normality['result']} (p={pred_normality['p_value']:.4f})")
        
        # Check linearity (preliminary correlation tests)
        cognitive_predictors = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        for predictor in cognitive_predictors:
            if predictor in analysis_dataset.columns:
                log(f"Checking linearity: {predictor} vs calibration_quality...")
                linearity_result = check_linearity(
                    analysis_dataset[predictor], 
                    analysis_dataset['calibration_quality'], 
                    predictor, 
                    'calibration_quality'
                )
                assumption_results.append(linearity_result)
                log(f"{linearity_result['test']}: {linearity_result['result']} (r={linearity_result['statistic']:.4f}, p={linearity_result['p_value']:.4f})")

        # Create assumption results DataFrame
        assumption_df = pd.DataFrame(assumption_results)
        
        # Save assumption check results
        assumption_output_path = RQ_DIR / "data" / "step03_assumption_checks.csv"
        log(f"Saving assumption check results to {assumption_output_path}...")
        assumption_df.to_csv(assumption_output_path, index=False, encoding='utf-8')
        log(f"step03_assumption_checks.csv ({len(assumption_df)} rows, {len(assumption_df.columns)} cols)")
        # Run Validation Tool
        # Validates: All required columns present in final dataset
        # Required: ['UID', 'calibration_quality', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'age', 'sex', 'education']

        log("Running validate_data_columns...")
        required_columns = ['UID', 'calibration_quality', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        validation_result = validate_data_columns(analysis_dataset, required_columns)

        # Report validation results
        if isinstance(validation_result, dict):
            log(f"Column validation result:")
            for key, value in validation_result.items():
                log(f"  {key}: {value}")
            
            # Check if validation passed
            if validation_result.get('valid', False):
                log("PASS - All required columns present")
            else:
                log("FAIL - Missing required columns")
                missing_cols = validation_result.get('missing_columns', [])
                if missing_cols:
                    log(f"Missing: {missing_cols}")
        else:
            log(f"{validation_result}")

        # Additional validation checks
        log("Additional data quality checks:")
        
        # Check UID uniqueness
        uid_duplicates = analysis_dataset['UID'].duplicated().sum()
        if uid_duplicates > 0:
            log(f"{uid_duplicates} duplicate UIDs found")
        else:
            log("All UIDs are unique")
        
        # Check T-score ranges (should be roughly 20-80)
        t_score_cols = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        for col in t_score_cols:
            if col in analysis_dataset.columns:
                col_min = analysis_dataset[col].min()
                col_max = analysis_dataset[col].max()
                col_mean = analysis_dataset[col].mean()
                col_std = analysis_dataset[col].std()
                
                if col_min < 10 or col_max > 90:
                    log(f"{col} range unusual: {col_min:.1f} to {col_max:.1f}")
                else:
                    log(f"{col} range reasonable: {col_min:.1f} to {col_max:.1f} (M={col_mean:.1f}, SD={col_std:.1f})")

        log("Step 03 complete")
        log(f"Created analysis dataset with {len(analysis_dataset)} participants")
        log(f"Ran {len(assumption_results)} assumption tests")
        log("Ready for hierarchical regression analysis in step04")
        
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)