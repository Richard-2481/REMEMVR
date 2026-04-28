#!/usr/bin/env python3
"""extract_sleep_data: Extract and clean per-test sleep data from dfvr.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.4
LOG_FILE = RQ_DIR / "logs" / "step01_extract_sleep_data.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_sleep_data")
        # Load dfvr.csv Data

        log("Loading dfvr.csv data...")
        dfvr_path = PROJECT_ROOT / "data" / "dfvr.csv"
        
        if not dfvr_path.exists():
            log(f"dfvr.csv not found at {dfvr_path}")
            sys.exit(1)
            
        dfvr_df = pd.read_csv(dfvr_path)
        log(f"dfvr.csv ({len(dfvr_df)} rows, {len(dfvr_df.columns)} cols)")
        # Extract Sleep Variables
        # Custom extraction (tools function has mismatched signature)

        log("Extracting sleep variables...")
        
        # Required columns
        required_cols = ["UID", "TEST", "hours-slept-night-before", "sleep-quality--1=bad-1=good"]
        missing_cols = [col for col in required_cols if col not in dfvr_df.columns]
        
        if missing_cols:
            log(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Extract and rename sleep variables
        sleep_df = dfvr_df[required_cols].copy()
        sleep_df = sleep_df.rename(columns={
            'hours-slept-night-before': 'Sleep_Hours',
            'sleep-quality--1=bad-1=good': 'Sleep_Quality'
        })
        
        log(f"Sleep variables for {len(sleep_df)} observations")
        
        # Check data types and convert if needed
        sleep_df['Sleep_Hours'] = pd.to_numeric(sleep_df['Sleep_Hours'], errors='coerce')
        sleep_df['Sleep_Quality'] = pd.to_numeric(sleep_df['Sleep_Quality'], errors='coerce')
        
        # Report missing data after conversion
        hours_missing = sleep_df['Sleep_Hours'].isna().sum()
        quality_missing = sleep_df['Sleep_Quality'].isna().sum()
        log(f"Sleep_Hours missing: {hours_missing}/{len(sleep_df)} ({hours_missing/len(sleep_df):.1%})")
        log(f"Sleep_Quality missing: {quality_missing}/{len(sleep_df)} ({quality_missing/len(sleep_df):.1%})")

        log("Sleep extraction complete")
        # Save Sleep Data
        # Output: Clean sleep dataset for downstream analysis

        log("Saving sleep data...")
        
        sleep_output_path = RQ_DIR / "data" / "step01_sleep_data.csv"
        sleep_df.to_csv(sleep_output_path, index=False, encoding='utf-8')
        log(f"{sleep_output_path} ({len(sleep_df)} rows, {len(sleep_df.columns)} cols)")
        # Run Sleep Hours Validation
        # Validates: Sleep hours in reasonable range [0, 24]

        log("Running sleep hours validation...")
        
        # Validate sleep hours range
        sleep_hours_clean = sleep_df['Sleep_Hours'].dropna()
        hours_validation = validate_numeric_range(
            data=sleep_hours_clean.values,
            min_val=0.0,
            max_val=24.0,
            column_name="Sleep_Hours"
        )
        # Run Sleep Quality Validation
        # Validates: Sleep quality in expected range [-1, 1]

        log("Running sleep quality validation...")
        
        # Validate sleep quality range  
        sleep_quality_clean = sleep_df['Sleep_Quality'].dropna()
        quality_validation = validate_numeric_range(
            data=sleep_quality_clean.values,
            min_val=-1.0,
            max_val=1.0,
            column_name="Sleep_Quality"
        )
        # Check Participant Coverage
        # Additional validation: Check minimum tests per participant

        log("Checking participant coverage...")
        
        # Count tests per participant
        tests_per_uid = sleep_df.groupby('UID').size()
        min_tests = tests_per_uid.min()
        max_tests = tests_per_uid.max()
        mean_tests = tests_per_uid.mean()
        
        uids_with_few_tests = tests_per_uid[tests_per_uid < 3].index.tolist()
        
        log(f"Tests per UID - Min: {min_tests}, Max: {max_tests}, Mean: {mean_tests:.1f}")
        log(f"UIDs with <3 tests: {len(uids_with_few_tests)}")
        
        coverage_acceptable = len(uids_with_few_tests) <= 10  # Allow some participants with missing sessions

        # Report validation results
        if hours_validation.get('valid', False):
            log("Sleep hours range: PASS")
        else:
            log(f"Sleep hours range: FAIL - {hours_validation.get('message', 'Unknown error')}")
            log(f"Out of range count: {hours_validation.get('out_of_range_count', 0)}")
            
        if quality_validation.get('valid', False):
            log("Sleep quality range: PASS")
        else:
            log(f"Sleep quality range: FAIL - {quality_validation.get('message', 'Unknown error')}")
            log(f"Out of range count: {quality_validation.get('out_of_range_count', 0)}")
            
        if coverage_acceptable:
            log("Participant coverage: PASS")
        else:
            log(f"Participant coverage: FAIL - too many UIDs with insufficient data")

        # Scientific Mantra logging between steps
        log("")
        log("=== SCIENTIFIC MANTRA ===")
        log("1. What question did we ask?")
        log("   -> Can we extract clean per-test sleep data for within-person analysis?")
        log("2. What did we find?")
        log(f"   -> {len(sleep_df)} observations, sleep hours range valid: {hours_validation.get('valid', False)}")
        log(f"   -> Sleep quality range valid: {quality_validation.get('valid', False)}")
        log("3. What does it mean?")
        log("   -> Clean sleep data enables multilevel modeling of within-person effects")
        log("4. What should we do next?")
        log("   -> Proceed to step02: create person-centered sleep variables")
        log("=========================")
        log("")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)