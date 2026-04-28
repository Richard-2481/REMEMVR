#!/usr/bin/env python3
"""create_time_transformations: Create time variables for quadratic model (Time, Time_squared, Time_log) and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import assign_piecewise_segments

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_create_time_transformations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Create Time Transformations")
        # Load Input Data

        log("Loading input data...")
        input_path = RQ_DIR / "data" / "step00_theta_tsvr.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        theta_tsvr_data = pd.read_csv(input_path, encoding='utf-8')
        log(f"step00_theta_tsvr.csv ({len(theta_tsvr_data)} rows, {len(theta_tsvr_data.columns)} cols)")
        log(f"Columns: {list(theta_tsvr_data.columns)}")
        # Run Analysis Tool (Piecewise Segment Assignment)
        #               (time since segment start) for piecewise LMM

        log("Running assign_piecewise_segments...")
        log("Parameters: tsvr_col='TSVR_hours', early_cutoff_hours=48.0")

        # Call assign_piecewise_segments to create Segment and Days_within columns
        time_transformed_data = assign_piecewise_segments(
            df=theta_tsvr_data,
            tsvr_col='TSVR_hours',
            early_cutoff_hours=48.0  # RQ 5.1.2 uses 48h inflection (one night's sleep)
        )
        log("Piecewise segment assignment complete")
        log(f"Segments created: {time_transformed_data['Segment'].unique().tolist()}")
        # Create Quadratic and Log Transformations
        # Quadratic model needs: Time, Time_squared
        # Log model needs: Time_log
        # These are simple column operations (no function call needed)

        log("Creating quadratic and log transformations...")

        # Copy TSVR_hours to Time column (for consistency with quadratic formula)
        time_transformed_data['Time'] = time_transformed_data['TSVR_hours'].copy()

        # Compute Time_squared
        time_transformed_data['Time_squared'] = time_transformed_data['TSVR_hours'] ** 2

        # Compute Time_log (add 1 to handle TSVR=0)
        time_transformed_data['Time_log'] = np.log(time_transformed_data['TSVR_hours'] + 1)

        log("Transformations complete")
        log(f"Added columns: Time, Time_squared, Time_log")
        # Save Analysis Outputs
        # This output will be used by: Step 2 (quadratic model) and Step 3 (piecewise model)

        output_path = RQ_DIR / "data" / "step01_time_transformed.csv"
        log(f"Saving {output_path.name}...")

        # Save with all transformations
        time_transformed_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step01_time_transformed.csv ({len(time_transformed_data)} rows, {len(time_transformed_data.columns)} cols)")
        log(f"Final columns: {list(time_transformed_data.columns)}")
        # Run Validation Tool
        # Validates: All 9 expected columns present, no missing columns
        # Threshold: Exact column set required

        log("Running validate_data_columns...")

        required_columns = [
            'UID', 'test', 'TSVR_hours', 'theta',  # Original columns
            'Time', 'Time_squared', 'Time_log',     # Quadratic/log transformations
            'Segment', 'Days_within'                # Piecewise transformations
        ]

        validation_result = validate_data_columns(
            df=time_transformed_data,
            required_columns=required_columns
        )

        # Report validation results
        log(f"valid: {validation_result['valid']}")
        log(f"n_required: {validation_result['n_required']}")
        log(f"n_missing: {validation_result['n_missing']}")

        if validation_result['missing_columns']:
            log(f"missing_columns: {validation_result['missing_columns']}")

        if not validation_result['valid']:
            raise ValueError(f"Validation failed: {validation_result}")

        # Additional validation checks (beyond validate_data_columns)
        log("Additional checks...")

        # Check for NaN values (transformations should be deterministic)
        nan_counts = time_transformed_data[required_columns].isna().sum()
        if nan_counts.sum() > 0:
            log(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        else:
            log("No NaN values (deterministic transformations confirmed)")

        # Check segment distribution
        segment_counts = time_transformed_data['Segment'].value_counts()
        log(f"Segment distribution: {segment_counts.to_dict()}")
        early_pct = segment_counts.get('Early', 0) / len(time_transformed_data) * 100
        late_pct = segment_counts.get('Late', 0) / len(time_transformed_data) * 100
        log(f"Early: {early_pct:.1f}%, Late: {late_pct:.1f}%")

        # Check Days_within starts at 0 for both segments
        early_min_days = time_transformed_data[time_transformed_data['Segment'] == 'Early']['Days_within'].min()
        late_min_days = time_transformed_data[time_transformed_data['Segment'] == 'Late']['Days_within'].min()
        log(f"Days_within min (Early): {early_min_days:.2f}")
        log(f"Days_within min (Late): {late_min_days:.2f}")

        if early_min_days != 0.0 or late_min_days != 0.0:
            log("Days_within does not start at 0 for both segments")
        else:
            log("Days_within starts at 0 for both segments (correct)")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
