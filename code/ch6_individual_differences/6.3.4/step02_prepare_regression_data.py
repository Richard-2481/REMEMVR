#!/usr/bin/env python3
"""prepare_regression_data: Merge DASS predictors with Ch5 theta and Ch6 confidence/calibration outcomes."""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.4
LOG_FILE = RQ_DIR / "logs" / "step02_prepare_regression_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def parse_composite_id(composite_id_str):
    """Parse composite_ID from Ch6 to extract UID."""
    # composite_ID format: "UID_test" (e.g., "101_1", "102_2")
    try:
        return composite_id_str.split('_')[0]
    except:
        return str(composite_id_str)

def check_coefficient_variation(df, columns):
    """Check coefficient of variation > 0.10 for outcome variables."""
    cv_results = {}
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            cv = std_val / abs(mean_val) if mean_val != 0 else 0
            cv_results[col] = cv
            log(f"[CV] {col}: CV = {cv:.3f} (mean={mean_val:.3f}, std={std_val:.3f})")
    return cv_results

def detect_outliers_iqr(df, column, threshold=1.5):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Prepare Regression Data")
        # Load Input Data

        log("Loading input data...")

        # Load DASS predictors (already participant-level, z-standardized)
        dass_df = pd.read_csv(RQ_DIR / "data" / "step01_dass_scores.csv")
        log(f"DASS scores ({len(dass_df)} rows, {len(dass_df.columns)} cols)")
        log(f"DASS columns: {dass_df.columns.tolist()}")

        # Load Ch5 theta scores (test-level, need aggregation)  
        theta_df = pd.read_csv(PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv")
        log(f"Ch5 theta scores ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        log(f"Ch5 columns: {theta_df.columns.tolist()}")

        # Load Ch6 confidence scores (test-level, need aggregation and UID parsing)
        confidence_df = pd.read_csv(PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv")
        log(f"Ch6 confidence scores ({len(confidence_df)} rows, {len(confidence_df.columns)} cols)")
        log(f"Ch6 confidence columns: {confidence_df.columns.tolist()}")

        # Load Ch6 calibration scores (participant-level, but check structure)
        calibration_df = pd.read_csv(PROJECT_ROOT / "results" / "ch6" / "6.2.1" / "data" / "step02_calibration_scores.csv")
        log(f"Ch6 calibration scores ({len(calibration_df)} rows, {len(calibration_df.columns)} cols)")
        log(f"Ch6 calibration columns: {calibration_df.columns.tolist()}")
        # Handle Column Name Variations and Data Preparation
        # Handle known column name differences from actual data vs 4_analysis.yaml
        
        log("Handling column name variations...")

        # Ch5 theta: has "Theta_All" instead of "theta", has "test" column for aggregation
        if "Theta_All" in theta_df.columns:
            theta_df = theta_df.rename(columns={"Theta_All": "theta"})
            log("Renamed Theta_All -> theta in Ch5 data")

        # Aggregate Ch5 theta to participant level (mean across 4 tests)
        theta_participant = theta_df.groupby("UID")["theta"].mean().reset_index()
        theta_participant = theta_participant.rename(columns={"theta": "theta_accuracy"})
        log(f"Ch5 aggregated to participant level: {len(theta_participant)} participants")

        # Ch6 confidence: has "composite_ID" and "theta_All", need to parse UID and aggregate
        if "composite_ID" in confidence_df.columns and "theta_All" in confidence_df.columns:
            # Parse UID from composite_ID
            confidence_df["UID"] = confidence_df["composite_ID"].apply(parse_composite_id)
            confidence_df = confidence_df.rename(columns={"theta_All": "theta_confidence"})
            log("Parsed UID from composite_ID and renamed theta_All -> theta_confidence")
            
            # Aggregate to participant level (mean across tests per participant)
            confidence_participant = confidence_df.groupby("UID")["theta_confidence"].mean().reset_index()
            confidence_participant = confidence_participant.rename(columns={"theta_confidence": "confidence"})
            log(f"Ch6 confidence aggregated to participant level: {len(confidence_participant)} participants")
        else:
            raise ValueError("Ch6 confidence file missing expected columns: composite_ID, theta_All")

        # Ch6 calibration: check if already participant-level or needs aggregation
        if "test" in calibration_df.columns:
            # Test-level, need aggregation
            calibration_participant = calibration_df.groupby("UID")["calibration"].mean().reset_index()
            log(f"Ch6 calibration aggregated to participant level: {len(calibration_participant)} participants")
        else:
            # Already participant-level, just select needed columns
            calibration_participant = calibration_df[["UID", "calibration"]].drop_duplicates()
            log(f"Ch6 calibration already participant-level: {len(calibration_participant)} participants")

        # Ensure all UIDs are strings for consistent merging
        dass_df["UID"] = dass_df["UID"].astype(str)
        theta_participant["UID"] = theta_participant["UID"].astype(str)
        confidence_participant["UID"] = confidence_participant["UID"].astype(str)
        calibration_participant["UID"] = calibration_participant["UID"].astype(str)
        # Merge All Datasets (Inner Join Strategy)
        # Merge strategy: inner join to ensure complete cases only

        log("Merging all datasets with inner join...")

        # Start with DASS predictors (base dataset)
        merged_df = dass_df[["UID", "z_Dep", "z_Anx", "z_Str"]].copy()
        log(f"Starting with DASS predictors: {len(merged_df)} participants")

        # Merge Ch5 theta (memory accuracy)
        merged_df = merged_df.merge(theta_participant, on="UID", how="inner")
        log(f"After Ch5 theta merge: {len(merged_df)} participants")

        # Merge Ch6 confidence
        merged_df = merged_df.merge(confidence_participant, on="UID", how="inner")
        log(f"After Ch6 confidence merge: {len(merged_df)} participants")

        # Merge Ch6 calibration
        merged_df = merged_df.merge(calibration_participant, on="UID", how="inner")
        log(f"After Ch6 calibration merge: {len(merged_df)} participants")

        log(f"Final dataset: {len(merged_df)} participants with complete data")
        log(f"Final columns: {merged_df.columns.tolist()}")
        # Outlier Detection and Data Quality Checks
        # Check for outliers using IQR method and assess data quality

        log("Checking for outliers in outcome variables...")

        outcome_vars = ["theta_accuracy", "confidence", "calibration"]
        outlier_summary = {}

        for var in outcome_vars:
            outliers, lower, upper = detect_outliers_iqr(merged_df, var, threshold=1.5)
            outlier_summary[var] = {
                "n_outliers": len(outliers),
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_rate": len(outliers) / len(merged_df)
            }
            log(f"{var}: {len(outliers)} outliers ({outlier_summary[var]['outlier_rate']:.2%})")
            if len(outliers) > 0:
                log(f"{var}: bounds = [{lower:.3f}, {upper:.3f}]")

        # Check missing values
        missing_counts = merged_df.isnull().sum()
        log("Missing value check:")
        for col, count in missing_counts.items():
            if count > 0:
                log(f"{col}: {count} missing values")
            else:
                log(f"{col}: No missing values")
        # Coefficient of Variation Check
        # Verify sufficient variance in outcome variables (CV > 0.10)

        log("Checking coefficient of variation for outcome variables...")
        cv_results = check_coefficient_variation(merged_df, outcome_vars)

        # Check CV threshold
        low_variance_vars = [var for var, cv in cv_results.items() if cv < 0.10]
        if low_variance_vars:
            log(f"Variables with low variance (CV < 0.10): {low_variance_vars}")
        else:
            log("All outcome variables have sufficient variance (CV > 0.10)")
        # Save Analysis Dataset
        # Save merged dataset for downstream regression analysis

        output_path = RQ_DIR / "data" / "step02_analysis_dataset.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"Analysis dataset: {output_path}")
        log(f"Dataset shape: {merged_df.shape} (rows, columns)")

        # Log descriptive statistics
        log("Descriptive statistics for final dataset:")
        for col in merged_df.columns:
            if col != "UID" and merged_df[col].dtype in ['float64', 'int64']:
                stats = {
                    'mean': merged_df[col].mean(),
                    'std': merged_df[col].std(), 
                    'min': merged_df[col].min(),
                    'max': merged_df[col].max()
                }
                log(f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        # Run Validation Tool
        # Validate final dataset structure matches requirements

        log("Running validate_data_columns...")
        required_columns = ["UID", "z_Dep", "z_Anx", "z_Str", "theta_accuracy", "confidence", "calibration"]
        
        validation_result = validate_data_columns(merged_df, required_columns)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation checks per 4_analysis.yaml criteria
        log("Checking analysis criteria...")

        # Check 1: All required columns present (7 total)
        missing_cols = [col for col in required_columns if col not in merged_df.columns]
        if missing_cols:
            log(f"FAIL - Missing columns: {missing_cols}")
        else:
            log("PASS - All 7 required columns present")

        # Check 2: N >= 90 participants (sufficient power)
        n_participants = len(merged_df)
        if n_participants >= 90:
            log(f"PASS - Sufficient sample size: N={n_participants} >= 90")
        else:
            log(f"WARNING - Small sample size: N={n_participants} < 90")

        # Check 3: No missing values
        total_missing = merged_df.isnull().sum().sum()
        if total_missing == 0:
            log("PASS - No missing values in analysis dataset")
        else:
            log(f"FAIL - {total_missing} missing values found")

        # Check 4: CV > 0.10 for outcome variables
        low_cv_count = len(low_variance_vars)
        if low_cv_count == 0:
            log("PASS - All outcome variables have CV > 0.10")
        else:
            log(f"WARNING - {low_cv_count} variables with CV < 0.10")

        # Check 5: DASS predictors z-standardized (mean ~0, std ~1)
        log("Checking DASS z-standardization...")
        dass_cols = ["z_Dep", "z_Anx", "z_Str"]
        for col in dass_cols:
            mean_val = merged_df[col].mean()
            std_val = merged_df[col].std()
            if abs(mean_val) < 0.1 and 0.9 < std_val < 1.1:
                log(f"PASS - {col}: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                log(f"WARNING - {col}: mean={mean_val:.3f}, std={std_val:.3f} (not properly standardized)")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)