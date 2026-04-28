#!/usr/bin/env python3
"""step02_reshape_wide_to_long: Convert wide format (2 theta columns: theta_source, theta_destination) to long"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_reshape_wide_to_long.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Reshape Wide to Long")
        # Load Input Data
        #          with LocationType factor for LMM analysis

        log("Loading wide-format data from Step 1...")
        input_file = RQ_DIR / "data" / "step01_piecewise_time_variables.csv"
        df_wide = pd.read_csv(input_file, encoding='utf-8')
        log(f"step01_piecewise_time_variables.csv ({len(df_wide)} rows, {len(df_wide.columns)} cols)")
        log(f"Columns: {list(df_wide.columns)}")
        # Reshape Using pandas.melt()
        #               (single theta column with LocationType identifier)

        log("Reshaping from wide to long format...")

        # Define ID variables (preserved across reshape)
        id_vars = ['UID', 'test', 'TSVR_hours', 'Segment', 'Days_within']

        # Define value variables (theta columns to melt)
        value_vars = ['theta_source', 'theta_destination']

        # Melt theta columns
        df_long = pd.melt(
            df_wide,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='LocationType_raw',
            value_name='theta'
        )
        log(f"After melt: {len(df_long)} rows")
        # Create LocationType Factor
        # Transform LocationType_raw ('theta_source', 'theta_destination')
        # into clean factor labels ('Source', 'Destination')
        # Set Source as reference level (treatment coding)

        log("Creating LocationType factor...")
        df_long['LocationType'] = df_long['LocationType_raw'].map({
            'theta_source': 'Source',
            'theta_destination': 'Destination'
        })

        # Verify mapping successful
        if df_long['LocationType'].isna().any():
            raise ValueError("LocationType mapping failed - unexpected values in LocationType_raw")

        log(f"LocationType distribution:\n{df_long['LocationType'].value_counts()}")
        # Match SE to Theta
        # When LocationType = 'Source': use se_source
        # When LocationType = 'Destination': use se_destination
        # This preserves SE-theta correspondence from wide format

        log("Matching SE values to theta...")

        # Create SE column by conditional assignment
        df_long['se'] = np.where(
            df_long['LocationType'] == 'Source',
            df_wide.loc[df_long.index % len(df_wide), 'se_source'].values,
            df_wide.loc[df_long.index % len(df_wide), 'se_destination'].values
        )

        # Verify SE values are reasonable
        se_range = (df_long['se'].min(), df_long['se'].max())
        log(f"SE range: [{se_range[0]:.3f}, {se_range[1]:.3f}]")
        # Clean Up and Reorder Columns
        # Drop intermediate columns, keep only required LMM input columns
        # Final column order: UID, test, LocationType, theta, se, TSVR_hours, Segment, Days_within

        log("Finalizing column structure...")

        # Drop LocationType_raw (intermediate variable)
        df_long = df_long.drop(columns=['LocationType_raw'])

        # Reorder columns to match specification
        final_columns = ['UID', 'test', 'LocationType', 'theta', 'se',
                        'TSVR_hours', 'Segment', 'Days_within']
        df_long = df_long[final_columns]

        log(f"Final columns: {list(df_long.columns)}")
        # Save Long-Format Output
        # This output will be used by Step 3 (Fit Piecewise LMM)

        log("Saving long-format LMM input...")
        output_file = RQ_DIR / "data" / "step02_lmm_input_long.csv"
        df_long.to_csv(output_file, index=False, encoding='utf-8')
        log(f"step02_lmm_input_long.csv ({len(df_long)} rows, {len(df_long.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns present, expected structure

        log("Running validate_data_format()...")

        required_cols = ['UID', 'test', 'LocationType', 'theta', 'se',
                        'TSVR_hours', 'Segment', 'Days_within']

        validation_result = validate_data_format(
            df=df_long,
            required_cols=required_cols
        )

        # Report validation results
        if validation_result['valid']:
            log(f"{validation_result['message']}")
        else:
            log(f"[VALIDATION ERROR] {validation_result['message']}")
            log(f"[VALIDATION ERROR] Missing columns: {validation_result['missing_cols']}")
            raise ValueError("Validation failed - required columns missing")
        # Additional Validation Checks
        # Verify data quality beyond column presence

        log("Performing additional data quality checks...")

        # Check row count (should be 800 = 400 x 2)
        expected_rows = len(df_wide) * 2
        if len(df_long) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {len(df_long)}")
        log(f"Row count: {len(df_long)} (expected {expected_rows})")

        # Check LocationType distribution (should be 400 each)
        location_counts = df_long['LocationType'].value_counts()
        if location_counts['Source'] != len(df_wide):
            raise ValueError(f"Expected {len(df_wide)} Source rows, got {location_counts['Source']}")
        if location_counts['Destination'] != len(df_wide):
            raise ValueError(f"Expected {len(df_wide)} Destination rows, got {location_counts['Destination']}")
        log(f"LocationType distribution: {location_counts['Source']} Source, {location_counts['Destination']} Destination")

        # Check each UID x test appears exactly 2 times
        uid_test_counts = df_long.groupby(['UID', 'test']).size()
        if not (uid_test_counts == 2).all():
            bad_counts = uid_test_counts[uid_test_counts != 2]
            raise ValueError(f"Some UID x test combinations don't appear exactly 2 times:\n{bad_counts.head()}")
        log(f"Each UID x test appears exactly 2 times")

        # Check for NaN values
        nan_counts = df_long.isna().sum()
        if nan_counts.any():
            raise ValueError(f"NaN values detected:\n{nan_counts[nan_counts > 0]}")
        log(f"No NaN values in any column")

        # Check theta range (should be in [-3, 3])
        theta_min, theta_max = df_long['theta'].min(), df_long['theta'].max()
        if theta_min < -3 or theta_max > 3:
            raise ValueError(f"Theta values outside expected range [-3, 3]: [{theta_min:.3f}, {theta_max:.3f}]")
        log(f"Theta range: [{theta_min:.3f}, {theta_max:.3f}] (within [-3, 3])")

        # Check SE range (should be in [0.1, 1.0])
        se_min, se_max = df_long['se'].min(), df_long['se'].max()
        if se_min < 0.1 or se_max > 1.0:
            raise ValueError(f"SE values outside expected range [0.1, 1.0]: [{se_min:.3f}, {se_max:.3f}]")
        log(f"SE range: [{se_min:.3f}, {se_max:.3f}] (within [0.1, 1.0])")

        log("Step 02 complete - wide-to-long reshape successful")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
