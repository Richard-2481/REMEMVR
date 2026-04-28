#!/usr/bin/env python3
"""Validate Dependencies: Verify required outputs from Ch5 5.1.4, Ch6 6.1.4, and Ch6 6.1.1 exist before"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict
import traceback
import yaml

# Configuration

# RQ directory (4 levels up from code/ to project root, then back to this RQ)
SCRIPT_PATH = Path(__file__).resolve()
RQ_DIR = SCRIPT_PATH.parents[1]  # results/ch6/6.9.2
PROJECT_ROOT = SCRIPT_PATH.parents[4]  # REMEMVR/

LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")
        # Define Required Dependencies
        # Cross-RQ dependencies that must exist before proceeding

        dependencies = [
            {
                'path': PROJECT_ROOT / 'results/ch5/5.1.4/data/step06_averaged_random_effects.csv',
                'description': 'Accuracy model-averaged random effects',
                'required_columns': ['UID', 'intercept_avg', 'slope_avg'],
                'expected_rows': (90, 110),  # ~100 participants, allow range
                'rq_status': PROJECT_ROOT / 'results/ch5/5.1.4/status.yaml'
            },
            {
                'path': PROJECT_ROOT / 'results/ch6/6.1.4/data/step01_variance_components.csv',
                'description': 'Confidence single-model variance components',
                'required_columns': ['component', 'value'],
                'expected_rows': (4, 10),  # Variance components
                'rq_status': PROJECT_ROOT / 'results/ch6/6.1.4/status.yaml'
            },
            {
                'path': PROJECT_ROOT / 'results/ch6/6.1.1/data/step05b_model_averaged_random_effects.csv',
                'description': 'Confidence model-averaged random effects',
                'required_columns': ['UID', 'ma_intercept', 'ma_slope'],
                'expected_rows': (90, 110),
                'rq_status': PROJECT_ROOT / 'results/ch6/6.1.1/status.yaml'
            },
            {
                'path': PROJECT_ROOT / 'results/ch6/6.1.1/data/step03_theta_confidence.csv',
                'description': 'Theta scores for re-fitting Steps 4-5',
                'required_columns': ['composite_ID', 'theta_All', 'se_All'],
                'expected_rows': (380, 420),  # ~400 observations
                'rq_status': PROJECT_ROOT / 'results/ch6/6.1.1/status.yaml'
            },
            {
                'path': PROJECT_ROOT / 'results/ch6/6.1.1/data/step00_tsvr_mapping.csv',
                'description': 'Time mapping composite_ID to test/TSVR_hours',
                'required_columns': ['composite_ID', 'TSVR_hours', 'test'],
                'expected_rows': (380, 420),  # ~400 observations
                'rq_status': PROJECT_ROOT / 'results/ch6/6.1.1/status.yaml'
            },
            {
                'path': PROJECT_ROOT / 'results/ch6/6.1.1/data/step00_irt_input.csv',
                'description': 'Wide-format IRT input for binary collapse (Step 5)',
                'required_columns': ['composite_ID'],  # Plus 105 item columns
                'expected_rows': (380, 420),  # ~400 observations
                'rq_status': PROJECT_ROOT / 'results/ch6/6.1.1/status.yaml'
            }
        ]

        log(f"Validating {len(dependencies)} cross-RQ dependencies...")
        # Check Each Dependency
        # For each file: check exists, check columns, check row count

        validation_results = []
        all_passed = True
        missing_files = []

        for dep in dependencies:
            file_path = dep['path']
            desc = dep['description']

            log(f"\n{desc}")
            log(f"  Path: {file_path}")

            # Check file exists
            if not file_path.exists():
                msg = f"FAIL: {file_path.name} - File does not exist"
                log(f"  {msg}")
                validation_results.append(msg)
                missing_files.append(str(file_path))
                all_passed = False
                continue

            # Check status.yaml for results analysis: success
            status_file = dep['rq_status']
            if status_file.exists():
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        status_data = yaml.safe_load(f)

                    # Check if results analysis is success
                    if status_data.get('results analysis', {}).get('status') != 'success':
                        msg = f"FAIL: {file_path.name} - Source RQ not complete (status != success)"
                        log(f"  {msg}")
                        validation_results.append(msg)
                        all_passed = False
                        continue
                except Exception as e:
                    log(f"  WARNING: Could not parse {status_file.name}: {e}")

            # Load file and check structure
            try:
                df = pd.read_csv(file_path)
                n_rows = len(df)
                n_cols = len(df.columns)

                log(f"  File loaded: {n_rows} rows, {n_cols} columns")

                # Check required columns
                missing_cols = set(dep['required_columns']) - set(df.columns)
                if missing_cols:
                    msg = f"FAIL: {file_path.name} - Missing columns: {missing_cols}"
                    log(f"  {msg}")
                    validation_results.append(msg)
                    all_passed = False
                    continue

                # Check row count
                min_rows, max_rows = dep['expected_rows']
                if not (min_rows <= n_rows <= max_rows):
                    msg = f"FAIL: {file_path.name} - Row count {n_rows} outside expected range [{min_rows}, {max_rows}]"
                    log(f"  {msg}")
                    validation_results.append(msg)
                    all_passed = False
                    continue

                # All checks passed
                msg = f"PASS: {file_path.name} ({n_rows} rows, {n_cols} cols)"
                log(f"  {msg}")
                validation_results.append(msg)

            except Exception as e:
                msg = f"FAIL: {file_path.name} - Could not read file: {str(e)}"
                log(f"  {msg}")
                validation_results.append(msg)
                all_passed = False
        # Write Validation Report
        # Save results to output file

        output_file = RQ_DIR / "data" / "step00_dependency_validation.txt"
        log(f"\nWriting validation report to {output_file.name}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Dependency Validation Report - RQ 6.9.2\n")
            f.write("=" * 70 + "\n\n")
            for result in validation_results:
                f.write(f"{result}\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Overall Status: {'PASS' if all_passed else 'FAIL'}\n")
            if not all_passed:
                f.write(f"\nMissing Files ({len(missing_files)}):\n")
                for missing in missing_files:
                    f.write(f"  - {missing}\n")

        log(f"{output_file.name}")
        # Exit with Appropriate Status
        # QUIT with error if any validation failed

        if not all_passed:
            log("\nDependency validation FAILED")
            log(f"  {len(missing_files)} file(s) missing or invalid")
            log("  Action: Complete prerequisite RQs (Ch5 5.1.4, Ch6 6.1.4, Ch6 6.1.1)")
            log("  Recommendation: Run prerequisite RQs before proceeding with RQ 6.9.2")
            sys.exit(1)

        log(f"\nStep 00 complete - All {len(dependencies)} dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
