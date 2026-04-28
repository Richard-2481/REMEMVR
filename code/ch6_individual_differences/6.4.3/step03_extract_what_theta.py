#!/usr/bin/env python3
"""extract_what_theta: Extract What-domain theta scores representing simple single-domain performance from Ch5 5.2.1"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_extract_what_theta.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: extract_what_theta")
        # Load Input Data

        log("Loading domain-specific theta scores from Ch5 5.2.1...")
        # Load results/ch5/5.2.1/data/step03_theta_scores.csv
        # Expected columns: ['composite_ID', 'theta_what', 'theta_where', 'theta_when']
        # Expected rows: ~400
        input_df = pd.read_csv(PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv")
        log(f"step03_theta_scores.csv ({len(input_df)} rows, {len(input_df.columns)} cols)")
        # Run Analysis Tool (Custom pandas processing)

        log("Extracting UID from composite_ID and aggregating What theta scores...")
        
        # Extract UID from composite_ID (format: A010_1 -> A010)
        # Using regex pattern to extract participant ID before underscore
        input_df['UID'] = input_df['composite_ID'].str.extract(r'([A-Z]\d+)')
        log(f"UIDs from composite_ID format (pattern: A###_N -> A###)")
        
        # Average What theta scores across tests for each participant
        # Group by UID and compute mean, std, count for theta_what
        df_theta_what = input_df.groupby('UID')['theta_what'].agg(['mean', 'std', 'count']).reset_index()
        df_theta_what.columns = ['UID', 'theta_what', 'se_what', 'n_tests']
        log(f"What theta scores by UID ({len(df_theta_what)} unique participants)")
        
        # Convert std to standard error (SE = std / sqrt(n))
        # Standard error represents uncertainty in individual participant's mean theta
        df_theta_what['se_what'] = df_theta_what['se_what'] / np.sqrt(df_theta_what['n_tests'])
        df_theta_what = df_theta_what.drop('n_tests', axis=1)
        log("Standard errors from within-participant variability")
        
        # Fill missing SE values with mean SE
        # Handles cases where participants have only 1 test (std = NaN)
        mean_se = df_theta_what['se_what'].mean()
        df_theta_what['se_what'] = df_theta_what['se_what'].fillna(mean_se)
        log(f"Missing SE values with mean SE ({mean_se:.4f})")

        log("Analysis complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 correlation analysis with RPM scores

        log("Saving step03_what_theta.csv...")
        # Output: results/ch7/7.4.3/data/step03_what_theta.csv
        # Contains: What-domain theta scores aggregated by participant
        # Columns: ['UID', 'theta_what', 'se_what']
        output_path = RQ_DIR / "data" / "step03_what_theta.csv"
        df_theta_what.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_what_theta.csv ({len(df_theta_what)} rows, {len(df_theta_what.columns)} cols)")
        # Run Validation Tool
        # Validates: theta_what values fall within expected IRT range [-4, 4]
        # Threshold: Standard IRT theta range validation

        log("Running validate_numeric_range on theta_what...")
        # Note: Using actual function signature (data, min_val, max_val, column_name)
        # not the 4_analysis.yaml signature (df, column, min_val, max_val, allow_missing)
        validation_result = validate_numeric_range(
            data=df_theta_what['theta_what'],
            min_val=-4.0,
            max_val=4.0,
            column_name='theta_what'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)