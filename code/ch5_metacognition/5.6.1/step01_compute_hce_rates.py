#!/usr/bin/env python3
"""Compute HCE Rate Per Participant Per Timepoint: Aggregate item-level confidence-accuracy data to compute HCE rate (high-confidence"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_compute_hce_rates.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Compute HCE Rate Per Participant Per Timepoint")
        # Load Input Data

        log("Loading input data...")
        input_path = RQ_DIR / "data" / "step00_item_level.csv"
        df_item = pd.read_csv(input_path, encoding='utf-8')
        log(f"step00_item_level.csv ({len(df_item)} rows, {len(df_item.columns)} cols)")
        log(f"Input columns: {df_item.columns.tolist()}")
        log(f"Confidence scale: {sorted(df_item['confidence'].dropna().unique())}")
        log(f"Accuracy values: {sorted(df_item['accuracy'].dropna().unique())}")
        # Compute HCE Rates Per Participant Per Test
        #               and compute proportion per participant per test

        log("Computing HCE rates...")

        # Filter to valid rows (non-NaN confidence AND accuracy)
        df_valid = df_item.dropna(subset=['confidence', 'accuracy']).copy()
        log(f"Valid rows after NaN filter: {len(df_valid)} (dropped {len(df_item) - len(df_valid)} rows)")

        # Define HCE condition: confidence >= 0.75 AND accuracy = 0
        # Note: Confidence scale is {0.2, 0.4, 0.6, 0.8, 1.0}, so >= 0.75 means {0.8, 1.0}
        df_valid['is_HCE'] = (df_valid['confidence'] >= 0.75) & (df_valid['accuracy'] == 0)

        # Group by UID and TEST
        agg_dict = {
            'is_HCE': 'sum',       # Count HCE instances
            'item_code': 'count',  # Count total items (valid rows)
            'TSVR': 'first'        # Get TSVR (should be constant per UID-TEST)
        }

        df_grouped = df_valid.groupby(['UID', 'TEST'], as_index=False).agg(agg_dict)

        # Rename columns
        df_grouped.rename(columns={
            'is_HCE': 'n_HCE',
            'item_code': 'n_total'
        }, inplace=True)

        # Compute HCE_rate = n_HCE / n_total
        df_grouped['HCE_rate'] = df_grouped['n_HCE'] / df_grouped['n_total']

        # Reorder columns to match specification
        df_hce_rates = df_grouped[['UID', 'TEST', 'TSVR', 'HCE_rate', 'n_HCE', 'n_total']]

        log(f"Computed HCE rates for {len(df_hce_rates)} observations")
        log(f"Unique participants: {df_hce_rates['UID'].nunique()}")
        log(f"Unique tests: {df_hce_rates['TEST'].nunique()} ({sorted(df_hce_rates['TEST'].unique())})")
        log(f"HCE_rate range: [{df_hce_rates['HCE_rate'].min():.4f}, {df_hce_rates['HCE_rate'].max():.4f}]")
        log(f"Mean HCE_rate: {df_hce_rates['HCE_rate'].mean():.4f}")
        log(f"n_HCE range: [{df_hce_rates['n_HCE'].min()}, {df_hce_rates['n_HCE'].max()}]")
        log(f"n_total range: [{df_hce_rates['n_total'].min()}, {df_hce_rates['n_total'].max()}]")
        # Save Analysis Output
        # Output: data/step01_hce_rates.csv
        # Contains: HCE rates per participant-test for LMM trajectory analysis

        log("Saving output data...")
        output_path = RQ_DIR / "data" / "step01_hce_rates.csv"
        df_hce_rates.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step01_hce_rates.csv ({len(df_hce_rates)} rows, {len(df_hce_rates.columns)} cols)")
        # Run Validation Tool
        # Validates: HCE_rate in [0, 1] range (proportion validation)
        # Threshold: No NaN, no out-of-range values

        log("Running validate_probability_range...")
        validation_result = validate_probability_range(
            probability_df=df_hce_rates,
            prob_columns=['HCE_rate']
        )

        # Report validation results
        if validation_result['valid']:
            log(f"Probability range validation: {validation_result['message']}")
        else:
            log(f"Probability range validation: {validation_result['message']}")
            if 'violations' in validation_result and validation_result['violations']:
                for violation in validation_result['violations']:
                    log(f"- {violation}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional logical checks not covered by validate_probability_range
        log("Running additional logical checks...")

        # Check: n_HCE <= n_total
        invalid_counts = df_hce_rates[df_hce_rates['n_HCE'] > df_hce_rates['n_total']]
        if len(invalid_counts) > 0:
            log(f"Found {len(invalid_counts)} rows where n_HCE > n_total")
            log(f"Examples: {invalid_counts.head()}")
            raise ValueError("Logical constraint violated: n_HCE > n_total")
        else:
            log("All rows satisfy n_HCE <= n_total")

        # Check: Expected 400 rows (100 participants × 4 tests)
        expected_rows = 400
        if len(df_hce_rates) != expected_rows:
            log(f"Expected {expected_rows} rows, found {len(df_hce_rates)}")
            raise ValueError(f"Expected {expected_rows} rows, found {len(df_hce_rates)}")
        else:
            log(f"Row count matches expected: {expected_rows}")

        # Check: All 100 participants present
        n_participants = df_hce_rates['UID'].nunique()
        if n_participants != 100:
            log(f"Expected 100 participants, found {n_participants}")
            raise ValueError(f"Expected 100 participants, found {n_participants}")
        else:
            log(f"All 100 participants present")

        # Check: All 4 tests per participant
        tests_per_participant = df_hce_rates.groupby('UID')['TEST'].nunique()
        if not (tests_per_participant == 4).all():
            missing_tests = tests_per_participant[tests_per_participant < 4]
            log(f"{len(missing_tests)} participants missing test sessions")
            log(f"Examples: {missing_tests.head()}")
            raise ValueError(f"{len(missing_tests)} participants missing test sessions")
        else:
            log(f"All participants have 4 test sessions")

        # Check: No NaN in HCE_rate
        nan_count = df_hce_rates['HCE_rate'].isna().sum()
        if nan_count > 0:
            log(f"Found {nan_count} NaN values in HCE_rate")
            raise ValueError(f"Found {nan_count} NaN values in HCE_rate")
        else:
            log("No NaN values in HCE_rate")

        log("Step 01 complete - All validations passed")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
