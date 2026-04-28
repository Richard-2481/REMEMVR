#!/usr/bin/env python3
"""Validate Dependencies: Validate cross-RQ dependencies exist before proceeding with analysis. Ensures"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dependencies to check
DEPENDENCIES = {
    'ch5_theta': {
        'path': PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv',
        'required_columns': ['UID', 'Theta_All'],  # Actual column name is Theta_All (uppercase)
        'description': 'Ch5 5.1.1 theta_all scores',
    },
    'dfnonvr': {
        'path': PROJECT_ROOT / 'data' / 'dfnonvr.csv',
        'required_columns': ['UID', 'age', 'sex', 'education', 'ravlt-trial-1-score',
                             'bvmt-total-recall', 'rpm-score'],  # UID is uppercase in dfnonvr.csv
        'description': 'Cognitive test and participant data',
    }
}

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
        # Check File Existence
        log("Checking file existence...")

        all_valid = True
        validation_results = []

        for dep_name, dep_info in DEPENDENCIES.items():
            file_path = dep_info['path']
            description = dep_info['description']

            log(f"{description}: {file_path}")

            # Check file exists with minimum size
            result = check_file_exists(str(file_path), min_size_bytes=1000)

            if result.get('valid', False):
                log(f"File exists ({result.get('size_bytes', 0)} bytes)")
                validation_results.append({
                    'dependency': dep_name,
                    'status': 'PASS',
                    'message': f"File exists ({result.get('size_bytes', 0)} bytes)"
                })
            else:
                log(f"{result.get('message', 'File validation failed')}")
                validation_results.append({
                    'dependency': dep_name,
                    'status': 'FAIL',
                    'message': result.get('message', 'File validation failed')
                })
                all_valid = False
        # Check Column Presence
        if all_valid:
            log("Checking required columns...")

            for dep_name, dep_info in DEPENDENCIES.items():
                file_path = dep_info['path']
                required_columns = dep_info['required_columns']
                description = dep_info['description']

                log(f"{description} columns: {required_columns}")

                # Load first row to check columns
                df_check = pd.read_csv(file_path, nrows=1)
                actual_columns = df_check.columns.tolist()

                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in actual_columns]

                if missing_columns:
                    log(f"Missing columns: {missing_columns}")
                    log(f"Available columns: {actual_columns}")
                    validation_results.append({
                        'dependency': dep_name,
                        'status': 'FAIL',
                        'message': f"Missing columns: {missing_columns}"
                    })
                    all_valid = False
                else:
                    log(f"All required columns present")
                    validation_results.append({
                        'dependency': dep_name,
                        'status': 'PASS',
                        'message': f"All {len(required_columns)} required columns present"
                    })
        # Save Validation Results
        output_file = OUTPUT_DIR / "step00_dependency_validation.txt"
        log(f"Writing validation results to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RQ 7.8.3 Dependency Validation Results\n")
            f.write("=" * 60 + "\n\n")

            for result in validation_results:
                f.write(f"Dependency: {result['dependency']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Message: {result['message']}\n")
                f.write("-" * 60 + "\n")

            f.write("\n")
            if all_valid:
                f.write("OVERALL STATUS: PASS\n")
                f.write("All dependencies validated successfully.\n")
            else:
                f.write("OVERALL STATUS: FAIL\n")
                f.write("One or more dependencies failed validation.\n")

        log(f"Validation results: {output_file}")
        # Validation Check
        if not all_valid:
            log("Dependency validation failed")
            log("Cannot proceed with analysis - missing required dependencies")
            sys.exit(1)

        log("Step 00 complete - all dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
