#!/usr/bin/env python3
"""load_participant_data: Load participant demographic and cognitive test data from prepared dataset."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_load_participant_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Participant Data")
        # Load Input Data

        log("Loading dfnonvr.csv from data/ directory...")
        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'

        if not dfnonvr_path.exists():
            log(f"dfnonvr.csv not found at: {dfnonvr_path}")
            sys.exit(1)

        df_full = pd.read_csv(dfnonvr_path, encoding='utf-8')
        log(f"dfnonvr.csv ({len(df_full)} rows, {len(df_full.columns)} cols)")
        # Extract RAVLT Columns
        # Extract only the columns we need for forgetting slope analysis
        # CRITICAL: Column names are hyphenated lowercase (from DATA_DICTIONARY.md v2026-01-05)

        log("Selecting all RAVLT trial columns (trials 1-5 + delayed recall)...")

        required_columns = [
            'UID',
            'ravlt-trial-1-score',
            'ravlt-trial-2-score',
            'ravlt-trial-3-score',
            'ravlt-trial-4-score',
            'ravlt-trial-5-score',
            'ravlt-delayed-recall-score',
        ]

        # Check that all required columns exist
        missing_cols = [col for col in required_columns if col not in df_full.columns]
        if missing_cols:
            log(f"Missing required columns: {missing_cols}")
            log(f"Available columns: {df_full.columns.tolist()[:10]}...")
            sys.exit(1)

        # Extract subset
        df_participant = df_full[required_columns].copy()
        log(f"Subset created ({len(df_participant)} rows, {len(df_participant.columns)} cols)")
        # Data Quality Checks
        # Verify data integrity before saving

        log("Verifying data quality...")

        # Check for missing values
        missing_counts = df_participant.isnull().sum()
        if missing_counts.any():
            log("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                log(f"  - {col}: {count} missing values")
        else:
            log("No missing values in RAVLT columns")

        # Check participant count
        n_participants = len(df_participant)
        log(f"Participant count: {n_participants}")
        if n_participants != 100:
            log(f"Expected 100 participants, found {n_participants}")

        # Check UID format (should be A###)
        uid_pattern_valid = df_participant['UID'].str.match(r'^A\d{3}$').all()
        if uid_pattern_valid:
            log("All UIDs match expected format (A###)")
        else:
            log("Some UIDs don't match expected format")

        # Descriptive statistics
        for trial_num in range(1, 6):
            col = f'ravlt-trial-{trial_num}-score'
            log("RAVLT Trial {} score: mean={:.2f}, std={:.2f}, range=[{:.0f}, {:.0f}]".format(
                trial_num,
                df_participant[col].mean(),
                df_participant[col].std(),
                df_participant[col].min(),
                df_participant[col].max()
            ))
        log("RAVLT Delayed Recall score: mean={:.2f}, std={:.2f}, range=[{:.0f}, {:.0f}]".format(
            df_participant['ravlt-delayed-recall-score'].mean(),
            df_participant['ravlt-delayed-recall-score'].std(),
            df_participant['ravlt-delayed-recall-score'].min(),
            df_participant['ravlt-delayed-recall-score'].max()
        ))
        # Save Analysis Output
        # Save participant data for downstream steps

        output_path = RQ_DIR / "data" / "step00_participant_data.csv"
        log(f"Saving participant data to: {output_path}")

        df_participant.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step00_participant_data.csv ({len(df_participant)} rows, {len(df_participant.columns)} cols)")
        # Run Validation Tool
        # Validate that output has expected columns

        log("Running validate_data_columns...")

        validation_result = validate_data_columns(
            df=df_participant,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

            # Check if validation passed
            if validation_result.get('valid', False):
                log("PASS - All required columns present")
            else:
                log("FAIL - Missing required columns")
                sys.exit(1)
        else:
            log(f"{validation_result}")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
