#!/usr/bin/env python3
"""validate_dependencies: Validate Ch5 5.1.1 theta scores and dfnonvr.csv data exist before proceeding."""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.2
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")
        # Validate Ch5 5.1.1 Status
        log("Validating Ch5 5.1.1 completion status...")

        ch5_status_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "status.yaml"
        if not ch5_status_path.exists():
            raise FileNotFoundError(f"Ch5 5.1.1 status file not found: {ch5_status_path}")

        with open(ch5_status_path, 'r', encoding='utf-8') as f:
            status = yaml.safe_load(f)

        # Handle both dict and string formats (from gcode_lessons.md Bug #5)
        results analysis_status = status.get('results analysis', {})
        if isinstance(results analysis_status, dict):
            actual_status = results analysis_status.get('status')
        elif isinstance(results analysis_status, str):
            actual_status = results analysis_status
        else:
            actual_status = 'unknown'

        if actual_status != 'success':
            raise ValueError(f"Ch5 5.1.1 not completed - results analysis status is '{actual_status}', expected 'success'")

        log(f"Ch5 5.1.1 status: {actual_status}")
        # Locate Theta Scores File
        log("Locating Ch5 5.1.1 theta scores file...")

        # Try multiple possible locations
        theta_candidates = [
            PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv",
            PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "theta_omnibus_scores.csv",
            PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "theta_all_scores.csv"
        ]

        theta_path = None
        for candidate in theta_candidates:
            if candidate.exists():
                theta_path = candidate
                break

        if theta_path is None:
            raise FileNotFoundError(f"Ch5 5.1.1 theta scores file not found. Tried: {[str(c) for c in theta_candidates]}")

        log(f"Theta scores file: {theta_path.relative_to(PROJECT_ROOT)}")

        # Load and validate theta file structure
        theta_df = pd.read_csv(theta_path)

        # Find theta column (may be 'Theta_All' or 'theta_all')
        theta_col = None
        for col in theta_df.columns:
            if col.lower() == 'theta_all':
                theta_col = col
                break

        if theta_col is None:
            raise ValueError(f"No omnibus theta column found in {theta_path}. Available columns: {theta_df.columns.tolist()}")

        # Verify required columns
        required_theta_cols = ['UID', 'test', theta_col]
        missing_cols = [col for col in required_theta_cols if col not in theta_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in theta file: {missing_cols}")

        n_theta_rows = len(theta_df)
        n_theta_participants = theta_df['UID'].nunique()

        log(f"Theta file structure valid: {n_theta_rows} rows, {n_theta_participants} unique participants")
        log(f"Theta column name: '{theta_col}'")
        # Validate dfnonvr.csv
        log("Validating dfnonvr.csv data file...")

        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        if not dfnonvr_path.exists():
            raise FileNotFoundError(f"Prepared data file not found: {dfnonvr_path}")

        dfnonvr_df = pd.read_csv(dfnonvr_path)

        # Verify key columns exist (exact names from DATA_DICTIONARY.md)
        required_demo_cols = ['UID', 'age', 'education', 'vr-exposure']
        required_ravlt_cols = [
            'ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
            'ravlt-trial-4-score', 'ravlt-trial-5-score'
        ]

        all_required = required_demo_cols + required_ravlt_cols
        missing_cols = [col for col in all_required if col not in dfnonvr_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns in dfnonvr.csv: {missing_cols}")

        n_demo_participants = len(dfnonvr_df)

        log(f"dfnonvr.csv structure valid: {n_demo_participants} participants")
        log(f"All required demographic and RAVLT columns present")
        # Validation Tool Check
        log("Running validate_data_format on theta file...")

        validation_result = validate_data_format(
            df=theta_df,
            required_cols=required_theta_cols
        )

        if not validation_result.get('valid', False):
            log(f"Theta file validation warnings: {validation_result}")
        else:
            log("Theta file validation successful")
        # Save Validation Results
        output_file = RQ_DIR / "data" / "step00_dependency_validation.txt"

        validation_summary = f"""DEPENDENCY VALIDATION RESULTS
Generated: {pd.Timestamp.now().isoformat()}

Ch5 5.1.1 Status:
  Status File: {ch5_status_path.relative_to(PROJECT_ROOT)}
  RQ Results Status: {actual_status}
  Validation: PASS

Theta Scores File:
  Path: {theta_path.relative_to(PROJECT_ROOT)}
  Theta Column: {theta_col}
  Total Rows: {n_theta_rows}
  Unique Participants: {n_theta_participants}
  Validation: PASS

dfnonvr.csv File:
  Path: {dfnonvr_path.relative_to(PROJECT_ROOT)}
  Total Participants: {n_demo_participants}
  Required Columns Present: {len(all_required)}
  Validation: PASS

Summary:
  All dependencies validated successfully
  Ready to proceed with discrepancy analysis
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(validation_summary)

        log(f"Validation results: {output_file.relative_to(RQ_DIR)}")

        # Print summary to console
        print("\n" + "="*80)
        print(validation_summary)
        print("="*80 + "\n")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
