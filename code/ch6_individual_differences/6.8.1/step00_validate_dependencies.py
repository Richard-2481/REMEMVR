#!/usr/bin/env python3
"""Validate Dependencies: Verify Ch5 domain-specific theta scores (from 5.2.1 ONLY) and cognitive test"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Files to validate
DEPENDENCIES = [
    {
        'path': PROJECT_ROOT / 'results' / 'ch5' / '5.2.1' / 'data' / 'step03_theta_scores.csv',
        'description': 'Ch5 5.2.1 domain theta scores (What/Where/When)',
        'required_columns': ['composite_ID', 'theta_what', 'theta_where', 'theta_when'],
        'min_size_bytes': 1000,
        'expected_rows': 400
    },
    {
        'path': PROJECT_ROOT / 'data' / 'dfnonvr.csv',
        'description': 'Participant cognitive test data',
        'required_columns': ['UID', 'age', 'nart-score', 'rpm-score'],
        'min_size_bytes': 1000,
        'expected_rows': 100
    }
]

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
        # VALIDATION: Check All Dependency Files

        validation_results = []
        all_valid = True

        for dep in DEPENDENCIES:
            log(f"\nValidating: {dep['description']}")
            log(f"{dep['path']}")

            result = {
                'file': dep['description'],
                'path': str(dep['path']),
                'exists': False,
                'size_ok': False,
                'columns_ok': False,
                'rows_ok': False,
                'status': 'FAIL'
            }

            # Check 1: File exists
            if not dep['path'].exists():
                log(f"File does not exist: {dep['path']}")
                all_valid = False
                validation_results.append(result)
                continue

            result['exists'] = True
            log(f"File exists")

            # Check 2: File size
            file_size = dep['path'].stat().st_size
            if file_size < dep['min_size_bytes']:
                log(f"File too small: {file_size} bytes (min: {dep['min_size_bytes']})")
                all_valid = False
                validation_results.append(result)
                continue

            result['size_ok'] = True
            log(f"File size OK: {file_size} bytes")

            # Check 3: CSV columns (header only for speed)
            try:
                df_header = pd.read_csv(dep['path'], nrows=0)
                actual_columns = df_header.columns.tolist()

                # Check required columns are present (not exact match - allows extra columns)
                missing_cols = [col for col in dep['required_columns'] if col not in actual_columns]

                if missing_cols:
                    log(f"Missing columns: {missing_cols}")
                    log(f"Expected: {dep['required_columns']}")
                    log(f"Found: {actual_columns}")
                    all_valid = False
                    validation_results.append(result)
                    continue

                result['columns_ok'] = True
                log(f"Required columns present: {dep['required_columns']}")

                # Check 4: Row count (load full file)
                df_full = pd.read_csv(dep['path'])
                actual_rows = len(df_full)

                # Allow some flexibility in row count (at least 90% of expected)
                min_expected_rows = int(dep['expected_rows'] * 0.9)

                if actual_rows < min_expected_rows:
                    log(f"Too few rows: {actual_rows} (expected ~{dep['expected_rows']})")
                    all_valid = False
                    validation_results.append(result)
                    continue

                result['rows_ok'] = True
                log(f"Row count OK: {actual_rows} rows (expected ~{dep['expected_rows']})")

                # All checks passed
                result['status'] = 'PASS'
                log(f"{dep['description']} validated successfully")

            except Exception as e:
                log(f"Error reading CSV: {str(e)}")
                all_valid = False
                validation_results.append(result)
                continue

            validation_results.append(result)
        # SAVE VALIDATION RESULTS

        log(f"\nWriting validation results to {OUTPUT_FILE}")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DEPENDENCY VALIDATION REPORT - RQ 7.8.1\n")
            f.write("=" * 80 + "\n\n")

            f.write("CRITICAL CORRECTION:\n")
            f.write("- Original 4_analysis.yaml referenced THREE RQs (5.2.1, 5.2.2, 5.2.3)\n")
            f.write("- ACTUAL source: Ch5 5.2.1 ONLY (all domains in one file)\n")
            f.write("- Validation adjusted accordingly\n\n")

            for result in validation_results:
                f.write("-" * 80 + "\n")
                f.write(f"File: {result['file']}\n")
                f.write(f"Path: {result['path']}\n")
                f.write(f"Exists: {'YES' if result['exists'] else 'NO'}\n")
                f.write(f"Size OK: {'YES' if result['size_ok'] else 'NO'}\n")
                f.write(f"Columns OK: {'YES' if result['columns_ok'] else 'NO'}\n")
                f.write(f"Rows OK: {'YES' if result['rows_ok'] else 'NO'}\n")
                f.write(f"Status: {result['status']}\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"OVERALL STATUS: {'PASS - All dependencies valid' if all_valid else 'FAIL - See errors above'}\n")
            f.write("=" * 80 + "\n")

        log(f"Validation report written")
        # FINAL STATUS

        if all_valid:
            log("\nStep 00 complete - All dependencies validated")
            sys.exit(0)
        else:
            log("\nStep 00 failed - Dependency validation errors detected")
            log("Fix missing/invalid dependencies before proceeding")
            sys.exit(1)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
