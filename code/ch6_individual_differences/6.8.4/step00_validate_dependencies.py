#!/usr/bin/env python3
"""validate_dependencies: Validate Ch5 domain-specific RQ completion (5.2.1, 5.2.2, 5.2.3) and data accessibility"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4 (derived from script location)
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
        log("Step 00: validate_dependencies")
        # Define Dependency Paths

        log("Defining dependency paths...")

        # Ch5 domain-specific status files
        ch5_dependencies = {
            'Ch5_5.2.1_What': PROJECT_ROOT / 'results' / 'ch5' / '5.2.1' / 'status.yaml',
            'Ch5_5.2.2_Where': PROJECT_ROOT / 'results' / 'ch5' / '5.2.2' / 'status.yaml',
            'Ch5_5.2.3_When': PROJECT_ROOT / 'results' / 'ch5' / '5.2.3' / 'status.yaml'
        }

        # Cognitive test data (Ch7 CRITICAL: NEVER use master.xlsx)
        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'

        log(f"Will validate {len(ch5_dependencies)} Ch5 status files + 1 data file")
        # Validate Ch5 Status Files

        log("Validating Ch5 status files...")

        validation_results = []
        all_valid = True

        for name, path in ch5_dependencies.items():
            log(f"Checking {name}...")
            result = check_file_exists(file_path=str(path), min_size_bytes=100)

            if result.get('valid', False):
                log(f"{name}: {path} ({result.get('size_bytes', 0)} bytes)")
                validation_results.append(f"{name}: EXISTS (size={result.get('size_bytes', 0)} bytes)")
            else:
                log(f"{name}: {result.get('message', 'Unknown error')}")
                validation_results.append(f"{name}: {result.get('message', 'Unknown error')}")
                all_valid = False
        # Validate dfnonvr.csv Existence

        log("Validating dfnonvr.csv...")
        dfnonvr_result = check_file_exists(file_path=str(dfnonvr_path), min_size_bytes=100)

        if dfnonvr_result.get('valid', False):
            log(f"dfnonvr.csv: {dfnonvr_path} ({dfnonvr_result.get('size_bytes', 0)} bytes)")
            validation_results.append(f"dfnonvr.csv: EXISTS (size={dfnonvr_result.get('size_bytes', 0)} bytes)")
        else:
            log(f"dfnonvr.csv: {dfnonvr_result.get('message', 'Unknown error')}")
            validation_results.append(f"dfnonvr.csv: {dfnonvr_result.get('message', 'Unknown error')}")
            all_valid = False
        # Validate dfnonvr.csv Columns
        # Validates: Required cognitive test columns exist
        # Threshold: All 5 columns must be present

        log("Validating dfnonvr.csv columns...")

        # NOTE: NART not available per 4_analysis.yaml verification report
        # Only 4 predictors: ravlt, bvmt, rpm, age
        required_columns = [
            'UID',
            'ravlt-trial-5-score',  # Ch7 exact column name (lowercase with hyphens)
            'bvmt-total-recall',    # Ch7 exact column name
            'rpm-score',            # Ch7 exact column name
            'age'                   # Ch7 exact column name
        ]

        try:
            df_nonvr = pd.read_csv(dfnonvr_path)
            log(f"dfnonvr.csv ({len(df_nonvr)} rows, {len(df_nonvr.columns)} cols)")

            column_result = validate_data_columns(df=df_nonvr, required_columns=required_columns)

            if column_result.get('valid', False):
                log(f"All {len(required_columns)} required columns present in dfnonvr.csv")
                validation_results.append(f"dfnonvr.csv COLUMNS: All {len(required_columns)} required columns present")
            else:
                missing = column_result.get('missing_columns', [])
                log(f"dfnonvr.csv missing columns: {missing}")
                validation_results.append(f"dfnonvr.csv COLUMNS: Missing {missing}")
                all_valid = False

        except Exception as e:
            log(f"Could not read dfnonvr.csv: {str(e)}")
            validation_results.append(f"dfnonvr.csv READ ERROR: {str(e)}")
            all_valid = False
        # Save Validation Report
        # These outputs will be used by: Master to determine if Step 01 can proceed

        log("Saving validation report...")
        output_path = RQ_DIR / 'data' / 'step00_dependency_validation.txt'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RQ 7.8.4 Dependency Validation Report\n")
            f.write("=" * 60 + "\n\n")

            for result_line in validation_results:
                f.write(result_line + "\n")

            f.write("\n" + "=" * 60 + "\n")
            if all_valid:
                f.write("OVERALL STATUS: PASS - All dependencies validated\n")
            else:
                f.write("OVERALL STATUS: FAIL - See failures above\n")

        log(f"{output_path}")
        # Report Validation Status
        # Validates: Overall success/failure of dependency checks

        log("Reporting overall status...")

        if all_valid:
            log("All dependency checks PASSED")
            log("Step 00 complete - Ready for Step 01")
            sys.exit(0)
        else:
            log("Some dependency checks FAILED")
            log("Step 00 incomplete - Fix dependencies before proceeding to Step 01")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
