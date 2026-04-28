#!/usr/bin/env python3
"""extract_hce_rates: Load and prepare HCE rates from Ch6 analysis - aggregates test-level data to"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_hce_rates.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: extract_hce_rates")
        # Load Input Data

        log("Loading HCE rates from Ch6 analysis...")
        input_path = PROJECT_ROOT / "results" / "ch6" / "6.6.1" / "data" / "step01_hce_rates.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        hce_data = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path} ({len(hce_data)} rows, {len(hce_data.columns)} cols)")
        log(f"Columns found: {list(hce_data.columns)}")

        # Verify required columns exist
        required_cols = ['UID', 'HCE_rate']
        missing_cols = [col for col in required_cols if col not in hce_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {list(hce_data.columns)}")
        
        # Report data structure
        n_unique_participants = hce_data['UID'].nunique()
        n_total_rows = len(hce_data)
        avg_tests_per_participant = n_total_rows / n_unique_participants
        log(f"Found {n_unique_participants} unique participants")
        log(f"Average {avg_tests_per_participant:.1f} tests per participant")
        # Run Analysis - Aggregate to Participant Level

        log("Aggregating HCE rates to participant level...")
        
        # Check for missing HCE_rate values
        missing_hce = hce_data['HCE_rate'].isna().sum()
        if missing_hce > 0:
            log(f"Found {missing_hce} missing HCE_rate values - will exclude from aggregation")
        
        # Validate HCE rates are in [0,1] range before aggregation
        hce_values = hce_data['HCE_rate'].dropna()
        out_of_range = ((hce_values < 0) | (hce_values > 1)).sum()
        if out_of_range > 0:
            log(f"Found {out_of_range} HCE rates outside [0,1] range")
            invalid_rows = hce_data[(hce_data['HCE_rate'] < 0) | (hce_data['HCE_rate'] > 1)]
            log(f"Invalid values: {invalid_rows[['UID', 'HCE_rate']].to_string()}")
            raise ValueError("HCE rates must be in [0,1] range")
        
        log(f"All HCE rates in valid [0,1] range")
        
        # Aggregate by participant using mean
        participant_hce = hce_data.groupby('UID').agg({
            'HCE_rate': ['mean', 'count', 'std']
        }).round(6)
        
        # Flatten column names  
        participant_hce.columns = ['hce_rate', 'n_tests', 'hce_std']
        participant_hce = participant_hce.reset_index()
        
        log(f"Aggregated to {len(participant_hce)} participants")
        log(f"Mean tests per participant: {participant_hce['n_tests'].mean():.1f}")
        log(f"HCE rate range: {participant_hce['hce_rate'].min():.3f} - {participant_hce['hce_rate'].max():.3f}")
        log(f"Mean HCE rate: {participant_hce['hce_rate'].mean():.3f} (SD = {participant_hce['hce_rate'].std():.3f})")
        # Save Analysis Outputs
        # These outputs will be used by: step03_merge_analysis_dataset for regression

        log("Saving participant-level HCE rates...")
        
        # Prepare output with only required columns (UID, hce_rate)
        output_data = participant_hce[['UID', 'hce_rate']].copy()
        
        # Save to hierarchical path
        output_path = RQ_DIR / "data" / "step02_hce_rates.csv"
        output_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(output_data)} rows, {len(output_data.columns)} cols)")
        
        # Log summary statistics for verification
        log(f"Participant HCE rate statistics:")
        log(f"  - N participants: {len(output_data)}")
        log(f"  - Mean HCE rate: {output_data['hce_rate'].mean():.4f}")
        log(f"  - SD HCE rate: {output_data['hce_rate'].std():.4f}")
        log(f"  - Min HCE rate: {output_data['hce_rate'].min():.4f}")
        log(f"  - Max HCE rate: {output_data['hce_rate'].max():.4f}")
        # Run Validation Tool
        # Validates: Required columns ['UID', 'hce_rate'] are present
        # Threshold: All required columns must exist

        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=output_data,
            required_columns=['UID', 'hce_rate']
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
                
            # Check if validation passed
            if validation_result.get('valid', False):
                log("Column validation PASSED")
            else:
                log("Column validation FAILED")
                missing = validation_result.get('missing_columns', [])
                if missing:
                    log(f"Missing columns: {missing}")
                raise ValueError(f"Validation failed: missing columns {missing}")
        else:
            log(f"{validation_result}")

        # Additional range validation
        log("Verifying HCE rates in [0,1] range...")
        out_of_range_final = ((output_data['hce_rate'] < 0) | (output_data['hce_rate'] > 1)).sum()
        if out_of_range_final > 0:
            log(f"Final output has {out_of_range_final} HCE rates outside [0,1] range")
            raise ValueError("Final HCE rates contain invalid values")
        else:
            log("All final HCE rates in valid [0,1] range")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)