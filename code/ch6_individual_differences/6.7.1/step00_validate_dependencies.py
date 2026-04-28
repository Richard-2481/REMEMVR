#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.1.1 theta outputs and dfnonvr.csv accessibility before analysis"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import check_file_exists, validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.1
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Dependencies to check
DEPENDENCIES = {
    'ch5_status': PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'status.yaml',
    'ch5_theta': PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv',
    'dfnonvr': PROJECT_ROOT / 'data' / 'dfnonvr.csv'
}

# Expected columns
EXPECTED_COLUMNS = {
    'ch5_theta': ['UID', 'test', 'Theta_All'],
    'dfnonvr': ['UID', 'ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
                'ravlt-trial-4-score', 'ravlt-trial-5-score', 'bvmt-trial-1-score',
                'bvmt-trial-2-score', 'bvmt-trial-3-score']
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")

        # Create output directory if needed
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Track validation results
        validation_results = []
        all_passed = True
        # Check Ch5 5.1.1 status.yaml

        log("Validating Ch5 5.1.1 status.yaml...")
        status_path = DEPENDENCIES['ch5_status']

        if not status_path.exists():
            log(f"Ch5 status file not found: {status_path}")
            validation_results.append(f"FAIL: Ch5 5.1.1 status.yaml not found")
            all_passed = False
        else:
            # Load and check status
            with open(status_path, 'r') as f:
                status_data = yaml.safe_load(f)

            # Handle both dict and string formats (gcode_lessons.md #5)
            results analysis_status = status_data.get('results analysis', {})
            if isinstance(results analysis_status, dict):
                actual_status = results analysis_status.get('status')
            elif isinstance(results analysis_status, str):
                actual_status = results analysis_status
            else:
                actual_status = 'unknown'

            if actual_status == 'success':
                log(f"Ch5 5.1.1 status: {actual_status}")
                validation_results.append(f"PASS: Ch5 5.1.1 status = success")
            else:
                log(f"Ch5 5.1.1 status: {actual_status} (expected 'success')")
                validation_results.append(f"FAIL: Ch5 5.1.1 status = {actual_status}")
                all_passed = False
        # Check Ch5 theta scores file

        log("Validating Ch5 theta scores CSV...")
        theta_path = DEPENDENCIES['ch5_theta']

        # Check file exists with minimum size (gcode_lessons.md pattern)
        file_check = check_file_exists(theta_path, min_size_bytes=500)

        if not file_check.get('valid', False):
            log(f"Ch5 theta file validation: {file_check.get('message', 'Unknown error')}")
            validation_results.append(f"FAIL: Ch5 theta file - {file_check.get('message')}")
            all_passed = False
        else:
            log(f"Ch5 theta file exists ({file_check.get('size_bytes', 0)} bytes)")

            # Validate columns
            df_theta = pd.read_csv(theta_path)
            col_check = validate_data_columns(df_theta, EXPECTED_COLUMNS['ch5_theta'])

            if not col_check.get('valid', False):
                log(f"Ch5 theta columns: {col_check.get('missing_columns', [])}")
                validation_results.append(f"FAIL: Ch5 theta missing columns: {col_check.get('missing_columns')}")
                all_passed = False
            else:
                log(f"Ch5 theta has required columns ({len(df_theta)} rows)")
                validation_results.append(f"PASS: Ch5 theta scores ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        # Check dfnonvr.csv file

        log("Validating dfnonvr.csv...")
        dfnonvr_path = DEPENDENCIES['dfnonvr']

        # Check file exists with minimum size
        file_check = check_file_exists(dfnonvr_path, min_size_bytes=10000)

        if not file_check.get('valid', False):
            log(f"dfnonvr.csv validation: {file_check.get('message', 'Unknown error')}")
            validation_results.append(f"FAIL: dfnonvr.csv - {file_check.get('message')}")
            all_passed = False
        else:
            log(f"dfnonvr.csv exists ({file_check.get('size_bytes', 0)} bytes)")

            # Validate columns (critical cognitive test columns)
            df_nonvr = pd.read_csv(dfnonvr_path)
            col_check = validate_data_columns(df_nonvr, EXPECTED_COLUMNS['dfnonvr'])

            if not col_check.get('valid', False):
                log(f"dfnonvr.csv columns: {col_check.get('missing_columns', [])}")
                validation_results.append(f"FAIL: dfnonvr.csv missing columns: {col_check.get('missing_columns')}")
                all_passed = False
            else:
                log(f"dfnonvr.csv has required columns ({len(df_nonvr)} rows)")
                validation_results.append(f"PASS: dfnonvr.csv ({len(df_nonvr)} rows, {len(df_nonvr.columns)} cols)")
        # Write validation report
        # Output: Text file summarizing all validation checks

        log("Writing validation report...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RQ 7.7.1 DEPENDENCY VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("OVERALL STATUS: ")
            if all_passed:
                f.write("ALL CHECKS PASSED\n\n")
            else:
                f.write("SOME CHECKS FAILED\n\n")

            f.write("INDIVIDUAL CHECKS:\n")
            f.write("-" * 70 + "\n")
            for result in validation_results:
                f.write(f"{result}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("RECOMMENDATION:\n")
            if all_passed:
                f.write("All dependencies validated. Ready to proceed with Step 01.\n")
            else:
                f.write("Dependency validation failed. Address issues before proceeding.\n")

        log(f"Validation report: {OUTPUT_FILE}")

        # Final status
        if all_passed:
            log("Step 00 complete - All dependencies validated")
            sys.exit(0)
        else:
            log("Step 00 - Dependency validation failed")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
