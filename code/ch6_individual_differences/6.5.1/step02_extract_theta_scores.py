#!/usr/bin/env python3
"""extract_theta_scores: Load and aggregate mean theta_all scores from Ch5 5.1.1"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_theta_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: extract_theta_scores")
        # Load Input Data

        log("Loading Ch5 theta scores...")
        # Load results/ch5/5.1.1/data/step03_theta_scores.csv
        # Expected columns: ["UID", "test", "Theta_All"]
        # Expected rows: ~400
        input_path = PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv"
        theta_df = pd.read_csv(input_path)
        log(f"step03_theta_scores.csv ({len(theta_df)} rows, {len(theta_df.columns)} cols)")

        # Verify expected format
        expected_cols = ["UID", "test", "Theta_All"]
        if list(theta_df.columns) != expected_cols:
            raise ValueError(f"Column mismatch. Expected: {expected_cols}, Got: {list(theta_df.columns)}")
        
        log(f"Column check passed: {expected_cols}")
        log(f"Data format: {len(theta_df)} rows (expected ~400)")
        
        # Check for expected format (4 tests per participant)
        n_participants = theta_df['UID'].nunique()
        n_tests_per_uid = theta_df.groupby('UID')['test'].count()
        log(f"{n_participants} unique participants")
        log(f"Tests per participant: min={n_tests_per_uid.min()}, max={n_tests_per_uid.max()}")
        # Run Analysis Operations
        # Operations: Aggregate Theta_All by UID (mean, std, count)

        log("Aggregating theta scores by participant...")
        
        # Calculate mean theta across all 4 tests per participant
        theta_summary = theta_df.groupby('UID').agg({
            'Theta_All': ['mean', 'std', 'count']
        }).round(4)
        
        # Flatten column names
        theta_summary.columns = ['theta_all', 'theta_SE', 'n_tests']
        theta_summary = theta_summary.reset_index()
        
        # Verify expected format
        if len(theta_summary) != 100:
            log(f"Expected 100 participants, got {len(theta_summary)}")
        
        if not all(theta_summary['n_tests'] == 4):
            log(f"Not all participants have 4 tests: {theta_summary['n_tests'].value_counts()}")
        
        # Check theta value bounds
        theta_min = theta_summary['theta_all'].min()
        theta_max = theta_summary['theta_all'].max()
        if not theta_summary['theta_all'].between(-3, 3).all():
            log(f"Some theta values outside expected range [-3, 3]: min={theta_min:.3f}, max={theta_max:.3f}")
        
        log(f"Aggregation complete: {len(theta_summary)} participants")
        log(f"Theta range: [{theta_min:.3f}, {theta_max:.3f}]")
        # Save Analysis Outputs
        # These outputs will be used by: Step 03 (merge with self-report data)

        log("Saving aggregated theta scores...")
        # Output: results/ch7/7.5.1/data/step02_theta_scores.csv
        # Contains: Participant-level theta means and standard errors
        # Columns: ["UID", "theta_all", "theta_SE"]
        output_path = RQ_DIR / "data/step02_theta_scores.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only required columns (drop n_tests helper column)
        output_df = theta_summary[['UID', 'theta_all', 'theta_SE']].copy()
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_theta_scores.csv ({len(output_df)} rows, {len(output_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Theta values within expected bounds [-3, 3]
        # Threshold: Custom range validation for IRT theta scores

        log("Running theta bounds validation (manual)...")
        # Manual validation since validate_probability_range doesn't accept custom_range
        theta_values = output_df['theta_all']
        within_bounds = theta_values.between(-3, 3)
        
        validation_result = {
            'valid': within_bounds.all(),
            'out_of_bounds_count': (~within_bounds).sum(),
            'min_value': theta_values.min(),
            'max_value': theta_values.max(),
            'range_check': f"[{theta_values.min():.3f}, {theta_values.max():.3f}]"
        }

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)