#!/usr/bin/env python3
"""extract_overall_theta: Extract overall omnibus theta scores representing complex integration performance from Ch5 5.1.1."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_extract_overall_theta.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: extract_overall_theta")
        # Load Input Data

        log("Loading Ch5 5.1.1 theta scores...")
        # Load results/ch5/5.1.1/data/step03_theta_scores.csv
        # Expected columns: ['UID', 'test', 'Theta_All']
        # Expected rows: ~400 (4 tests × 100 participants)
        input_file = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        df_overall = pd.read_csv(input_file)
        log(f"{input_file.name} ({len(df_overall)} rows, {len(df_overall.columns)} cols)")

        # Validate expected columns exist
        expected_cols = ['UID', 'test', 'Theta_All']
        if not all(col in df_overall.columns for col in expected_cols):
            raise ValueError(f"Missing expected columns. Found: {list(df_overall.columns)}, Expected: {expected_cols}")
        
        # Log data summary
        log(f"Unique participants: {df_overall['UID'].nunique()}")
        log(f"Tests per participant: {df_overall.groupby('UID').size().mean():.1f} (expected: 4)")
        log(f"Theta_All range: [{df_overall['Theta_All'].min():.3f}, {df_overall['Theta_All'].max():.3f}]")
        # Run Analysis Tool

        log("Computing overall theta scores by participant...")
        
        # Group by UID and compute mean, std, and count for Theta_All
        df_theta_overall = df_overall.groupby('UID')['Theta_All'].agg(['mean', 'std', 'count']).reset_index()
        df_theta_overall.columns = ['UID', 'theta_overall', 'se_overall', 'n_tests']
        
        # Convert std to standard error: SE = std / sqrt(n)
        df_theta_overall['se_overall'] = df_theta_overall['se_overall'] / np.sqrt(df_theta_overall['n_tests'])
        
        # Handle missing SE values (when participant has only 1 test, std is NaN)
        # Fill with mean SE across all participants
        mean_se = df_theta_overall['se_overall'].mean()
        df_theta_overall['se_overall'] = df_theta_overall['se_overall'].fillna(mean_se)
        log(f"Filled {df_theta_overall['se_overall'].isna().sum()} missing SE values with mean SE ({mean_se:.3f})")
        
        # Drop count column (not needed in output)
        df_theta_overall = df_theta_overall.drop('n_tests', axis=1)
        
        log("Analysis complete")
        log(f"Output contains {len(df_theta_overall)} participants")
        log(f"theta_overall range: [{df_theta_overall['theta_overall'].min():.3f}, {df_theta_overall['theta_overall'].max():.3f}]")
        log(f"se_overall range: [{df_theta_overall['se_overall'].min():.3f}, {df_theta_overall['se_overall'].max():.3f}]")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 correlation analysis

        output_file = RQ_DIR / "data" / "step02_overall_theta.csv"
        log(f"Saving {output_file.name}...")
        # Output: step02_overall_theta.csv
        # Contains: Overall omnibus theta scores (complex integration performance)
        # Columns: ['UID', 'theta_overall', 'se_overall']
        df_theta_overall.to_csv(output_file, index=False, encoding='utf-8')
        log(f"{output_file.name} ({len(df_theta_overall)} rows, {len(df_theta_overall.columns)} cols)")
        # Run Validation Tool
        # Validates: theta_overall values are in expected IRT range [-4, 4]
        # Threshold: Standard IRT theta score range

        log("Running validate_numeric_range...")
        validation_result = validate_numeric_range(
            data=df_theta_overall['theta_overall'],
            min_val=-4.0,
            max_val=4.0,
            column_name='theta_overall'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)