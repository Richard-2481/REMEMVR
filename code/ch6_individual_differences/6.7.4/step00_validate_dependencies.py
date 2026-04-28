#!/usr/bin/env python3
"""validate_dependencies: Validate Ch5 theta outputs and dfnonvr.csv accessibility before analysis."""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: validate_dependencies")

        # Create output list for validation report
        validation_report = []
        all_valid = True
        # Validate Ch5 Status File
        log("Checking Ch5 5.1.1 status.yaml...")

        status_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "status.yaml"
        result = check_file_exists(file_path=status_path, min_size_bytes=100)

        if result.get('valid', False):
            log(f"Ch5 status file exists: {status_path}")
            validation_report.append(f"Ch5 5.1.1 status file: FOUND")

            # Check status content
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = yaml.safe_load(f)

            # Handle both dict and string status formats (Bug #5 from gcode_lessons)
            results analysis = status_data.get('results analysis', {})
            if isinstance(results analysis, dict):
                status_value = results analysis.get('status', 'unknown')
            elif isinstance(results analysis, str):
                status_value = results analysis
            else:
                status_value = 'unknown'

            if status_value == 'success':
                log(f"Ch5 5.1.1 status: success")
                validation_report.append(f"Ch5 5.1.1 status: success")
            else:
                log(f"Ch5 5.1.1 status: {status_value} (expected: success)")
                validation_report.append(f"Ch5 5.1.1 status: {status_value} (WARNING)")
                all_valid = False
        else:
            log(f"Ch5 status file not found or too small: {status_path}")
            validation_report.append(f"Ch5 5.1.1 status file: MISSING")
            all_valid = False
        # Validate Ch5 Theta Scores File
        log("Checking Ch5 theta scores file...")

        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        result = check_file_exists(file_path=theta_path, min_size_bytes=100)

        if result.get('valid', False):
            log(f"Theta file exists: {theta_path}")
            validation_report.append(f"Theta file found: {theta_path}")

            # Validate structure
            df_theta = pd.read_csv(theta_path)
            required_cols = ['UID', 'Theta_All']

            missing_cols = [col for col in required_cols if col not in df_theta.columns]
            if missing_cols:
                log(f"Theta file missing columns: {missing_cols}")
                log(f"Available columns: {df_theta.columns.tolist()}")
                validation_report.append(f"Theta file: MISSING COLUMNS {missing_cols}")
                all_valid = False
            else:
                log(f"Theta file has required columns: {required_cols}")
                log(f"Theta file shape: {df_theta.shape}")
                validation_report.append(f"Theta file structure: VALID ({df_theta.shape[0]} rows)")

                # Count unique participants
                n_uids = df_theta['UID'].nunique()
                log(f"Unique participants in theta file: {n_uids}")
                validation_report.append(f"Theta participants: {n_uids}")
        else:
            log(f"Theta file not found or too small: {theta_path}")
            validation_report.append(f"Theta file: MISSING")
            all_valid = False
        # Validate dfnonvr.csv File
        log("Checking dfnonvr.csv accessibility...")

        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        result = check_file_exists(file_path=dfnonvr_path, min_size_bytes=100)

        if result.get('valid', False):
            log(f"dfnonvr.csv exists: {dfnonvr_path}")
            validation_report.append(f"dfnonvr.csv accessible: YES")

            # Validate RAVLT columns
            df_nonvr = pd.read_csv(dfnonvr_path)
            required_cols = ['UID', 'ravlt-trial-1-score', 'ravlt-trial-2-score',
                           'ravlt-trial-3-score', 'ravlt-trial-4-score', 'ravlt-trial-5-score']

            missing_cols = [col for col in required_cols if col not in df_nonvr.columns]
            if missing_cols:
                log(f"dfnonvr.csv missing columns: {missing_cols}")
                log(f"Available columns (first 20): {df_nonvr.columns.tolist()[:20]}")
                validation_report.append(f"dfnonvr.csv: MISSING COLUMNS {missing_cols}")
                all_valid = False
            else:
                log(f"dfnonvr.csv has required RAVLT columns")
                log(f"dfnonvr.csv shape: {df_nonvr.shape}")
                validation_report.append(f"dfnonvr.csv structure: VALID ({df_nonvr.shape[0]} rows)")

                # Count participants
                n_uids = df_nonvr['UID'].nunique()
                log(f"Unique participants in dfnonvr.csv: {n_uids}")
                validation_report.append(f"dfnonvr participants: {n_uids}")
        else:
            log(f"dfnonvr.csv not found or too small: {dfnonvr_path}")
            validation_report.append(f"dfnonvr.csv accessible: NO")
            all_valid = False
        # Write Validation Report
        log("Writing validation report...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 7.7.4 - Dependency Validation Report\n")
            f.write("=" * 80 + "\n\n")

            for line in validation_report:
                f.write(f"{line}\n")

            f.write("\n" + "=" * 80 + "\n")
            if all_valid:
                f.write("OVERALL STATUS: ALL DEPENDENCIES VALID\n")
                f.write("Ready to proceed with false negative analysis\n")
            else:
                f.write("OVERALL STATUS: VALIDATION FAILED\n")
                f.write("Fix missing dependencies before proceeding\n")
            f.write("=" * 80 + "\n")

        log(f"Validation report: {OUTPUT_FILE}")
        # Final Status
        if all_valid:
            log("Step 00 complete - all dependencies valid")
            sys.exit(0)
        else:
            log("Step 00 failed - dependency validation errors found")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
