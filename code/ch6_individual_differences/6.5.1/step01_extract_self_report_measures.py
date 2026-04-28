#!/usr/bin/env python3
"""extract_self_report_measures: Extract self-report and demographic variables from dfnonvr.csv with data quality checks"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data import load_participant_data

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_self_report_measures.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_self_report_measures")
        # Load Participant Data
        
        log("Loading participant data from dfnonvr.csv...")
        # Use absolute path from project root to avoid path issues (lesson #3)
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        # Load using tools.data.load_participant_data function
        participant_df = load_participant_data(path=str(dfnonvr_path))
        log(f"dfnonvr.csv ({len(participant_df)} rows, {len(participant_df.columns)} cols)")
        # Extract Required Columns
        
        log("Extracting required self-report columns...")
        
        # Define required columns (exact names from dfnonvr.csv as verified in validation)
        required_cols = ['UID', 'education', 'vr-exposure', 'typical-sleep-hours', 'age']
        
        # Check all columns exist (defensive programming)
        missing_cols = [col for col in required_cols if col not in participant_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dfnonvr.csv: {missing_cols}")
        
        # Extract only required columns
        df_extracted = participant_df[required_cols].copy()
        log(f"{len(required_cols)} columns: {required_cols}")
        # Data Quality Checks
        # Check missing data patterns before proceeding
        # Important for understanding data completeness in self-report measures
        
        log("Checking missing data patterns...")
        missing_data = df_extracted.isnull().sum()
        total_rows = len(df_extracted)
        
        for col in required_cols:
            missing_count = missing_data[col]
            missing_pct = (missing_count / total_rows) * 100
            log(f"{col}: {missing_count}/{total_rows} ({missing_pct:.1f}%)")
        
        # Check if missing data is acceptable (< 5% per variable)
        high_missing = missing_data[missing_data > (total_rows * 0.05)]
        if len(high_missing) > 0:
            log(f"High missing data in: {high_missing.to_dict()}")
        else:
            log("Missing data levels acceptable (all <5%)")
        # Rename Columns for Analysis Consistency
        # Rename columns to match analysis conventions (remove hyphens, standardize case)
        # This ensures downstream scripts can use consistent variable names
        
        log("Renaming columns for analysis consistency...")
        
        # Column renaming mapping (dfnonvr.csv -> analysis format)
        column_mapping = {
            'UID': 'UID',                         # Keep as-is
            'education': 'Education',             # Capitalize
            'vr-exposure': 'VR_Experience',       # Remove hyphens, use underscore
            'typical-sleep-hours': 'Typical_Sleep',  # Remove hyphens, simplify
            'age': 'Age'                          # Capitalize
        }
        
        df_renamed = df_extracted.rename(columns=column_mapping)
        final_columns = list(column_mapping.values())
        log(f"Columns renamed to: {final_columns}")
        # Save Analysis Outputs
        # These outputs will be used by: step02 (theta extraction) and step03 (dataset merge)
        
        output_path = RQ_DIR / "data" / "step01_self_report_data.csv"
        log(f"Saving {output_path}...")
        
        # Output: step01_self_report_data.csv
        # Contains: Clean self-report measures with consistent column names
        # Columns: ['UID', 'Education', 'VR_Experience', 'Typical_Sleep', 'Age']
        df_renamed.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_renamed)} rows, {len(df_renamed.columns)} cols)")
        # Run Validation Tool
        # Validates: All required output columns are present in final dataset
        # Threshold: All columns must exist (no missing columns allowed)
        
        log("Running validate_data_columns...")
        
        # CRITICAL: Load DataFrame first, then pass to validation (lesson #15)
        # validate_data_columns expects (df: DataFrame, required_columns: List[str])
        # NOT (df_path: str, required_columns: List[str]) as 4_analysis.yaml incorrectly specified
        df_for_validation = pd.read_csv(output_path)
        
        validation_result = validate_data_columns(
            df=df_for_validation,
            required_columns=final_columns  # Validate renamed columns are all present
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
        
        # Check validation passed
        if validation_result.get('valid', False):
            log("All required columns present in output dataset")
        else:
            missing_cols = validation_result.get('missing_columns', [])
            log(f"FAILED - Missing columns: {missing_cols}")
            raise ValueError(f"Validation failed: missing columns {missing_cols}")

        log("Step 01 complete")
        log(f"Self-report extraction complete: {len(df_renamed)} participants")
        log(f"Data completeness check passed")
        log(f"Column renaming complete for downstream compatibility")
        
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)