#!/usr/bin/env python3
"""prepare_predictors: Grand-mean center Age variable and create time transformations (linear + log) to prepare"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_standardization, validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_prepare_predictors.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: prepare_predictors")
        # Load Input Data

        log("Loading merged data from Step 0...")
        input_path = RQ_DIR / "data" / "step00_lmm_input_raw.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify expected columns exist
        required_cols = ['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'theta', 'se_all', 'age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All required columns present: {required_cols}")
        # Create Age-Centered Predictor

        log("Computing grand-mean centered Age...")
        mean_age = df['age'].mean()
        df['Age_c'] = df['age'] - mean_age
        log(f"Grand mean age = {mean_age:.2f} years")
        log(f"Age_c: mean = {df['Age_c'].mean():.6f}, SD = {df['Age_c'].std():.2f}")

        # Verify centering worked (mean should be ≈ 0)
        if abs(df['Age_c'].mean()) > 0.01:
            log(f"Age_c mean = {df['Age_c'].mean():.6f} (expected ≈ 0)")
        # Create Time Transformations

        log("Creating time transformations...")

        # Linear time (direct copy)
        df['Time'] = df['TSVR_hours']
        log(f"Time (linear): range = [{df['Time'].min():.1f}, {df['Time'].max():.1f}] hours")

        # Logarithmic time (log offset = 1 to avoid log(0))
        df['Time_log'] = np.log(df['TSVR_hours'] + 1)
        log(f"Time_log: range = [{df['Time_log'].min():.3f}, {df['Time_log'].max():.3f}]")

        # Check for invalid values (NaN or inf)
        if df['Time_log'].isna().any() or np.isinf(df['Time_log']).any():
            raise ValueError("Time_log contains NaN or inf values")
        log("Time_log contains no NaN or inf values")
        # Save Prepared Data
        # These outputs will be used by: Step 2 (fit_lmm)

        log("Saving prepared data...")
        output_path = RQ_DIR / "data" / "step01_lmm_input_prepared.csv"
        output_cols = ['composite_ID', 'UID', 'TEST', 'TSVR_hours', 'theta', 'se_all',
                      'age', 'Age_c', 'Time', 'Time_log']
        df[output_cols].to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df)} rows, {len(output_cols)} cols)")
        # Run Validation Tool #1 (validate_standardization)
        # Validates: Age_c has mean ≈ 0 (within tolerance 0.01)
        # Threshold: Centering deviation < 0.01

        log("Running validate_standardization...")
        validation_result_1 = validate_standardization(
            df=df,
            column_names=['Age_c'],
            tolerance=0.01  # Age_c mean must be within ±0.01 of 0
        )

        # Report validation results
        if isinstance(validation_result_1, dict):
            for key, value in validation_result_1.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result_1}")

        # Check if validation passed (if dict has 'status' key)
        if isinstance(validation_result_1, dict) and 'status' in validation_result_1:
            if validation_result_1['status'] != 'pass':
                log(f"validate_standardization did not pass (non-fatal)")
        # Run Validation Tool #2 (validate_numeric_range)
        # Validates: Age_c in [-30, 30], Time_log in [0, 6]
        # Threshold: Values must be within specified ranges

        log("Running validate_numeric_range for Age_c...")
        validation_result_2a = validate_numeric_range(
            data=df['Age_c'],
            min_val=-30.0,
            max_val=30.0,
            column_name='Age_c'  # Age_c should be in [-30, 30] (centered around 0)
        )

        if isinstance(validation_result_2a, dict):
            for key, value in validation_result_2a.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result_2a}")

        # Check if validation passed
        if isinstance(validation_result_2a, dict) and 'status' in validation_result_2a:
            if validation_result_2a['status'] != 'pass':
                raise ValueError(f"validate_numeric_range failed for Age_c: {validation_result_2a.get('message', 'Unknown error')}")

        log("Running validate_numeric_range for Time_log...")
        validation_result_2b = validate_numeric_range(
            data=df['Time_log'],
            min_val=0.0,
            max_val=6.0,
            column_name='Time_log'  # Time_log should be in [0, 6] (log(169) ≈ 5.13)
        )

        if isinstance(validation_result_2b, dict):
            for key, value in validation_result_2b.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result_2b}")

        # Check if validation passed
        if isinstance(validation_result_2b, dict) and 'status' in validation_result_2b:
            if validation_result_2b['status'] != 'pass':
                raise ValueError(f"validate_numeric_range failed for Time_log: {validation_result_2b.get('message', 'Unknown error')}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
