#!/usr/bin/env python3
"""validate_data_availability: Validate dfvr.csv data availability and sleep variable completeness"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_data_columns, validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.4
LOG_FILE = RQ_DIR / "logs" / "step00_validate_data_availability.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: validate_data_availability")
        # Load dfvr.csv Data

        log("Loading dfvr.csv data...")
        dfvr_path = PROJECT_ROOT / "data" / "dfvr.csv"
        
        if not dfvr_path.exists():
            log(f"dfvr.csv not found at {dfvr_path}")
            sys.exit(1)
            
        dfvr_df = pd.read_csv(dfvr_path)
        log(f"dfvr.csv ({len(dfvr_df)} rows, {len(dfvr_df.columns)} cols)")
        # Run Data Column Validation

        log("Running validate_data_columns...")
        required_columns = ["UID", "TEST", "hours-slept-night-before", "sleep-quality--1=bad-1=good"]
        
        validation_result = validate_data_columns(
            df=dfvr_df,
            required_columns=required_columns
        )
        log("Column validation complete")
        # Check Sleep Data Completeness
        # Additional validation: Check sleep variable completeness
        # Criteria: <10% missing data acceptable

        log("Checking sleep data completeness...")
        
        # Check missing data in sleep columns
        sleep_cols = ["hours-slept-night-before", "sleep-quality--1=bad-1=good"]
        missing_counts = {}
        
        for col in sleep_cols:
            if col in dfvr_df.columns:
                missing_count = dfvr_df[col].isna().sum()
                missing_pct = missing_count / len(dfvr_df)
                missing_counts[col] = {
                    'missing_count': missing_count,
                    'missing_pct': missing_pct,
                    'acceptable': missing_pct <= 0.10
                }
                log(f"{col}: {missing_count}/{len(dfvr_df)} missing ({missing_pct:.1%})")
        
        # Overall sleep completeness
        all_sleep_acceptable = all(
            info['acceptable'] for info in missing_counts.values()
        )
        # Save Validation Results
        # Output: Validation summary for downstream steps

        log("Saving validation results...")
        
        # Create validation summary
        validation_summary = pd.DataFrame([{
            'validation_result': validation_result.get('valid', False),
            'missing_columns': str(validation_result.get('missing_cols', [])),
            'row_count': len(dfvr_df),
            'sleep_completeness': all_sleep_acceptable,
            'hours_slept_missing_pct': missing_counts.get('hours-slept-night-before', {}).get('missing_pct', 1.0),
            'sleep_quality_missing_pct': missing_counts.get('sleep-quality--1=bad-1=good', {}).get('missing_pct', 1.0),
            'expected_rows': 400,
            'row_count_acceptable': len(dfvr_df) >= 380  # Allow for some missing participants
        }])
        
        validation_path = RQ_DIR / "data" / "step00_data_validation.csv"
        validation_summary.to_csv(validation_path, index=False, encoding='utf-8')
        log(f"{validation_path} (1 row, {len(validation_summary.columns)} cols)")
        # Run Format Validation
        # Validates: Basic DataFrame structure meets expectations
        
        log("Running validate_data_format...")
        
        format_validation = validate_data_format(
            df=dfvr_df, 
            required_cols=required_columns
        )

        # Report validation results
        if validation_result.get('valid', False):
            log("Column validation: PASS")
        else:
            log(f"Column validation: FAIL - {validation_result.get('message', 'Unknown error')}")
            
        if all_sleep_acceptable:
            log("Sleep completeness: PASS")
        else:
            log("Sleep completeness: FAIL - excessive missing data")
            
        if format_validation.get('valid', False):
            log("Format validation: PASS")
        else:
            log(f"Format validation: FAIL - {format_validation.get('message', 'Unknown error')}")

        # Scientific Mantra logging between steps
        log("")
        log("=== SCIENTIFIC MANTRA ===")
        log("1. What question did we ask?")
        log("   -> Are sleep variables available and complete in dfvr.csv for 400 observations?")
        log("2. What did we find?")
        log(f"   -> {len(dfvr_df)} total rows, sleep completeness: {all_sleep_acceptable}")
        log("3. What does it mean?")
        log("   -> Data validation determines if sleep-performance analysis can proceed")
        log("4. What should we do next?")
        log("   -> If validation passed: proceed to step01 sleep extraction")
        log("   -> If validation failed: investigate data quality issues")
        log("=========================")
        log("")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)