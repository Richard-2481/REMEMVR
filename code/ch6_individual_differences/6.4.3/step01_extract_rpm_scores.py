#!/usr/bin/env python3
"""extract_rpm_scores: Extract RPM scores from dfnonvr.csv (CRITICAL: NOT master.xlsx) using exact column name 'rpm-score'"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_rpm_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: extract_rpm_scores")
        # Load Input Data

        log("Loading RPM data from dfnonvr.csv...")
        log("CRITICAL: Using dfnonvr.csv NOT master.xlsx per Ch7 v5.3.0 compliance")
        
        # Read dfnonvr.csv with exact column names from specification
        # CRITICAL: Column name is 'rpm-score' (with hyphen) per DATA_DICTIONARY.md
        input_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'
        df_rpm = pd.read_csv(input_path, usecols=['UID', 'rpm-score'])
        log(f"dfnonvr.csv ({len(df_rpm)} rows, {len(df_rpm.columns)} cols)")
        # Data Processing and Standardization

        log("Cleaning and standardizing RPM data...")
        
        # Rename column from 'rpm-score' to 'rpm_score' (remove hyphen for easier Python access)
        df_rpm = df_rpm.rename(columns={'rpm-score': 'rpm_score'})
        log(f"Column 'rpm-score' -> 'rpm_score'")
        
        # Remove rows with missing RPM scores
        initial_count = len(df_rpm)
        df_rpm = df_rpm.dropna(subset=['rpm_score'])
        final_count = len(df_rpm)
        removed_count = initial_count - final_count
        log(f"Removed {removed_count} rows with missing RPM scores ({final_count} remain)")
        
        # Validate RPM score range (0-12 per DATA_DICTIONARY)
        min_rpm = df_rpm['rpm_score'].min()
        max_rpm = df_rpm['rpm_score'].max()
        mean_rpm = df_rpm['rpm_score'].mean()
        log(f"RPM scores: min={min_rpm}, max={max_rpm}, mean={mean_rpm:.2f}")
        
        if min_rpm < 0 or max_rpm > 12:
            log(f"RPM scores outside expected range [0,12]: min={min_rpm}, max={max_rpm}")
        
        # Create standardized z-score (rpm_standardized)
        # Formula: (x - mean) / std
        rpm_mean = df_rpm['rpm_score'].mean()
        rpm_std = df_rpm['rpm_score'].std()
        df_rpm['rpm_standardized'] = (df_rpm['rpm_score'] - rpm_mean) / rpm_std
        
        # Validate standardization
        z_mean = df_rpm['rpm_standardized'].mean()
        z_std = df_rpm['rpm_standardized'].std()
        log(f"rpm_standardized: mean={z_mean:.6f}, std={z_std:.6f}")
        
        if abs(z_mean) > 1e-6 or abs(z_std - 1.0) > 1e-6:
            log(f"Standardization check failed: mean should be ~0, std should be ~1")

        log("Data processing complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 (correlation analysis)

        log("Saving processed RPM scores...")
        
        # Output: results/ch7/7.4.3/data/step01_rpm_scores.csv
        # Contains: UID, rpm_score (0-12 range), rpm_standardized (z-score)
        # Columns: ['UID', 'rpm_score', 'rpm_standardized']
        output_path = RQ_DIR / 'data' / 'step01_rpm_scores.csv'
        df_rpm.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_rpm)} rows, {len(df_rpm.columns)} cols)")
        
        # Log column summary for verification
        log(f"Output columns: {df_rpm.columns.tolist()}")
        log(f"First few UIDs: {df_rpm['UID'].head(3).tolist()}")
        # Run Validation Tool
        # Validates: Required columns present in output
        # Threshold: All expected columns must exist

        log("Running validate_data_columns...")
        required_columns = ['UID', 'rpm_score', 'rpm_standardized']
        validation_result = validate_data_columns(
            df=df_rpm,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            if validation_result.get('valid', False):
                log("PASS - All required columns present")
                for key, value in validation_result.items():
                    if key != 'valid':
                        log(f"{key}: {value}")
            else:
                log("FAIL - Missing required columns")
                log(f"Missing: {validation_result.get('missing_columns', [])}")
                sys.exit(1)
        else:
            log(f"{validation_result}")

        log("Step 1 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)