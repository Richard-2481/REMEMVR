#!/usr/bin/env python3
"""extract_calibration_metrics: Extract calibration quality metrics from Ch6 outputs - identifies per-participant"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import glob
import os

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_calibration_metrics.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract Calibration Metrics")
        # Discover Ch6 Calibration Data Source
        
        log("Searching for Ch6 calibration data...")
        
        # Specifically use step02_calibration_scores.csv which has per-test data we can aggregate
        calibration_file = str(PROJECT_ROOT / 'results' / 'ch6' / '6.2.1' / 'data' / 'step02_calibration_scores.csv')
        
        if not os.path.exists(calibration_file):
            # Fall back to searching
            ch6_patterns = [
                str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*calibration*.csv'),
                str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*resolution*.csv'),
                str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*brier*.csv')
            ]
            
            calibration_file = None
            for pattern in ch6_patterns:
                files = glob.glob(pattern)
                # Skip trajectory files which don't have per-participant data
                files = [f for f in files if 'trajectory' not in f and 'SEM' not in f]
                if files:
                    calibration_file = files[0]
                    break
        
        if calibration_file and os.path.exists(calibration_file):
            log(f"Ch6 calibration source: {calibration_file}")
        else:
            calibration_file = None
        
        if not calibration_file:
            raise FileNotFoundError("No Ch6 calibration data found. Expected files matching patterns: " + str(ch6_patterns))
        # Load Ch6 Calibration Data
        # Contains: UID, test, calibration (plus other columns)
        
        log("Loading Ch6 calibration data...")
        df_calibration = pd.read_csv(calibration_file)
        log(f"{calibration_file} ({len(df_calibration)} rows, {len(df_calibration.columns)} cols)")
        log(f"Available columns: {list(df_calibration.columns)}")
        
        # Verify required columns exist
        required_cols = ['UID', 'calibration']
        missing_cols = [col for col in required_cols if col not in df_calibration.columns]
        if missing_cols:
            # Check for alternative column names
            metric_cols = [col for col in df_calibration.columns if any(term in col.lower() 
                          for term in ['calibration', 'resolution', 'brier'])]
            if metric_cols:
                log(f"Required 'calibration' column not found, but found calibration metrics: {metric_cols}")
                calibration_col = metric_cols[0]
                log(f"Using '{calibration_col}' as calibration metric")
            else:
                raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(df_calibration.columns)}")
        else:
            calibration_col = 'calibration'
        # Aggregate Calibration by Participant
        # Goal: Convert per-test calibration scores to per-participant calibration quality
        # Method: Mean calibration across 4 tests per participant
        
        log("Aggregating calibration scores by participant...")
        
        # Check for missing data in calibration metric
        n_missing = df_calibration[calibration_col].isna().sum()
        if n_missing > 0:
            log(f"Found {n_missing} missing values in {calibration_col} column")
        
        # Aggregate by UID (mean across tests)
        df_calibration_agg = df_calibration.groupby('UID')[calibration_col].agg(['mean', 'count']).reset_index()
        df_calibration_agg.columns = ['UID', 'calibration_quality', 'n_tests']
        
        log(f"{len(df_calibration_agg)} participants")
        log(f"Mean calibration quality: {df_calibration_agg['calibration_quality'].mean():.3f}")
        log(f"Tests per participant - Min: {df_calibration_agg['n_tests'].min()}, Max: {df_calibration_agg['n_tests'].max()}")
        
        # Check for participants with < 4 tests
        incomplete_participants = df_calibration_agg[df_calibration_agg['n_tests'] < 4]
        if len(incomplete_participants) > 0:
            log(f"{len(incomplete_participants)} participants have < 4 tests:")
            for _, row in incomplete_participants.iterrows():
                log(f"  {row['UID']}: {int(row['n_tests'])} tests")
        # Save Analysis Output
        # Output: Per-participant calibration quality for downstream regression analysis
        
        output_path = RQ_DIR / "data" / "step01_calibration_metrics.csv"
        log(f"Saving calibration metrics to {output_path}...")
        
        # Create final output with only required columns
        df_output = df_calibration_agg[['UID', 'calibration_quality']].copy()
        
        # Remove participants with missing calibration quality
        n_before = len(df_output)
        df_output = df_output.dropna()
        n_after = len(df_output)
        if n_before != n_after:
            log(f"{n_before - n_after} participants with missing calibration quality")
        
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Run Validation Tool
        # Validates: Output has required columns for downstream regression
        # Required columns: ["UID", "calibration_quality"]
        
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=df_output,
            required_columns=["UID", "calibration_quality"]
        )

        # Report validation results
        for key, value in validation_result.items():
            log(f"{key}: {value}")

        if not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown validation error')}")

        log("Step 01 complete - calibration metrics extracted and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)