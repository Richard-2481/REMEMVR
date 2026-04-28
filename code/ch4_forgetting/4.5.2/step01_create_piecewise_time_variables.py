#!/usr/bin/env python3
"""create_piecewise_time_variables: Create piecewise time variables for piecewise LMM analysis: Segment (Early/Late)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import assign_piecewise_segments

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_create_piecewise_time_variables.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Create Piecewise Time Variables")
        # Load Input Data

        log("Loading input data from step00_theta_from_rq551.csv...")
        input_path = RQ_DIR / "data" / "step00_theta_from_rq551.csv"
        df_theta = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")

        # Verify expected columns
        expected_input_cols = ['composite_ID', 'UID', 'test', 'theta_source', 'theta_destination',
                               'se_source', 'se_destination', 'TSVR_hours']
        actual_cols = list(df_theta.columns)
        if actual_cols != expected_input_cols:
            log(f"Column mismatch!")
            log(f"Expected: {expected_input_cols}")
            log(f"Actual: {actual_cols}")
            raise ValueError("Input file column mismatch")

        log(f"Input columns verified: {len(actual_cols)} columns match expected")
        log(f"TSVR_hours range: [{df_theta['TSVR_hours'].min():.1f}, {df_theta['TSVR_hours'].max():.1f}]")
        # Run Analysis Tool
        #               Days_within (recentered time within each segment)

        log("Running assign_piecewise_segments with 48h cutoff...")
        # CRITICAL: RQ 5.5.2 uses 48-hour cutoff (not default 24h)
        # Early segment: 0-48h (2 days, consolidation-dominated)
        # Late segment: 48-168h (5 days, decay-dominated)
        df_piecewise = assign_piecewise_segments(
            df=df_theta,
            tsvr_col='TSVR_hours',
            early_cutoff_hours=48.0  # RQ 5.5.2 breakpoint (not default 24h)
        )
        log("Segment assignment complete")

        # Report segment distribution
        segment_counts = df_piecewise['Segment'].value_counts()
        log(f"Segment distribution:")
        for segment, count in segment_counts.items():
            log(f"{segment}: {count} observations ({count/len(df_piecewise)*100:.1f}%)")

        # Report Days_within range per segment
        for segment in ['Early', 'Late']:
            segment_data = df_piecewise[df_piecewise['Segment'] == segment]
            days_min = segment_data['Days_within'].min()
            days_max = segment_data['Days_within'].max()
            log(f"{segment} Days_within range: [{days_min:.2f}, {days_max:.2f}]")
        # Save Analysis Outputs
        # Output: data/step01_piecewise_time_variables.csv
        # Contains: All input columns + Segment (str) + Days_within (float)
        # Downstream usage: Step 2 (reshape wide-to-long) and Step 3 (fit piecewise LMM)

        output_path = RQ_DIR / "data" / "step01_piecewise_time_variables.csv"
        log(f"Saving {output_path.name}...")
        df_piecewise.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_piecewise)} rows, {len(df_piecewise.columns)} cols)")

        # Log output columns
        output_cols = list(df_piecewise.columns)
        log(f"Output columns ({len(output_cols)}): {output_cols}")
        # Run Validation Tool
        # Validates: Row count (400), column count (10), column presence
        # Threshold: Must match expected structure exactly

        log("Running validate_dataframe_structure...")
        expected_output_cols = [
            'composite_ID', 'UID', 'test', 'theta_source', 'theta_destination',
            'se_source', 'se_destination', 'TSVR_hours', 'Segment', 'Days_within'
        ]
        validation_result = validate_dataframe_structure(
            df=df_piecewise,
            expected_rows=400,  # 100 UIDs × 4 tests
            expected_columns=expected_output_cols
        )

        # Report validation results
        if validation_result['valid']:
            log("DataFrame structure validation passed")
            for key, value in validation_result.items():
                if key != 'valid':
                    log(f"{key}: {value}")
        else:
            log("DataFrame structure validation failed")
            log(f"Message: {validation_result.get('message', 'No message')}")
            raise ValueError("Validation failed - see log for details")

        # Additional manual validation checks
        log("Running additional checks...")

        # Check 1: Segment values
        unique_segments = set(df_piecewise['Segment'].unique())
        expected_segments = {'Early', 'Late'}
        if unique_segments != expected_segments:
            log(f"Segment values mismatch")
            log(f"Expected: {expected_segments}")
            log(f"Actual: {unique_segments}")
            raise ValueError("Invalid Segment values")
        log(f"Segment values: {unique_segments}")

        # Check 2: Days_within range
        days_min = df_piecewise['Days_within'].min()
        days_max = df_piecewise['Days_within'].max()
        if days_min < 0 or days_max > 10.0:  # Extended: some TSVR up to 246h = 8+ days after 48h breakpoint
            log(f"Days_within out of expected range")
            log(f"Expected: [0, 10]")
            log(f"Actual: [{days_min:.2f}, {days_max:.2f}]")
            raise ValueError("Days_within out of range")
        log(f"Days_within range: [{days_min:.2f}, {days_max:.2f}]")

        # Check 3: No NaN values in new columns
        nan_segment = df_piecewise['Segment'].isna().sum()
        nan_days = df_piecewise['Days_within'].isna().sum()
        if nan_segment > 0 or nan_days > 0:
            log(f"NaN values detected")
            log(f"Segment NaNs: {nan_segment}")
            log(f"Days_within NaNs: {nan_days}")
            raise ValueError("NaN values in new columns")
        log(f"No NaN values in Segment or Days_within")

        # Check 4: Segment distribution (~50/50 split expected)
        early_count = (df_piecewise['Segment'] == 'Early').sum()
        late_count = (df_piecewise['Segment'] == 'Late').sum()
        early_pct = early_count / len(df_piecewise) * 100
        late_pct = late_count / len(df_piecewise) * 100
        log(f"Distribution: {early_count} Early ({early_pct:.1f}%), {late_count} Late ({late_pct:.1f}%)")

        # 48h breakpoint should give ~2 tests in Early (T1, T2 at Day 0-1), ~2 tests in Late (T3, T4 at Day 3-6)
        # With 100 UIDs, expect ~200 observations per segment
        if early_count < 150 or early_count > 250:
            log(f"Early segment count unexpected: {early_count} (expected ~200)")
        if late_count < 150 or late_count > 250:
            log(f"Late segment count unexpected: {late_count} (expected ~200)")

        log("Step 01 complete")
        log(f"Output: {output_path}")
        log(f"Created piecewise time variables: Segment (Early/Late), Days_within (recentered time)")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
