#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.3.1 (accuracy) and Ch6 6.4.1 (confidence) outputs exist and are complete."""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.7
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"

# Dependency file paths
ACCURACY_PATH = Path("/home/etai/projects/REMEMVR/results/ch5/5.3.1/data/step03_theta_scores.csv")
CONFIDENCE_PATH = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.1/data/step03_theta_confidence.csv")


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: validate_dependencies")
        # Validate Accuracy File (Ch5 5.3.1)

        log("Checking accuracy file from Ch5 5.3.1...")

        # Check file exists
        validation_result = check_file_exists(ACCURACY_PATH, min_size_bytes=100)
        if not validation_result.get('valid', False):
            log(f"Accuracy file not found: {ACCURACY_PATH}")
            log("SOURCE RQ INCOMPLETE: Ch5 5.3.1 accuracy data missing")
            sys.exit(1)

        # Load accuracy data
        df_accuracy = pd.read_csv(ACCURACY_PATH, encoding='utf-8')
        log(f"Accuracy file: {len(df_accuracy)} rows, {len(df_accuracy.columns)} columns")

        # Validate structure
        expected_cols = ['composite_ID', 'domain_name', 'theta']
        missing_cols = [c for c in expected_cols if c not in df_accuracy.columns]
        if missing_cols:
            log(f"Missing columns in accuracy file: {missing_cols}")
            log(f"Found columns: {list(df_accuracy.columns)}")
            sys.exit(1)

        # Validate row count
        if len(df_accuracy) != 1200:
            log(f"Accuracy file has {len(df_accuracy)} rows, expected 1200")
            log("ACCURACY DATA INCOMPLETE: Expected 100 UID × 4 tests × 3 paradigms")
            sys.exit(1)

        # Check for missing values
        missing_theta = df_accuracy['theta'].isna().sum()
        if missing_theta > 0:
            log(f"Accuracy file has {missing_theta} missing theta values")
            sys.exit(1)

        # Validate domain_name values
        unique_domains = df_accuracy['domain_name'].unique()
        expected_domains = ['free_recall', 'cued_recall', 'recognition']
        if set(unique_domains) != set(expected_domains):
            log(f"Unexpected domain_name values: {unique_domains}")
            log(f"Expected: {expected_domains}")
            sys.exit(1)

        log("Accuracy file validation complete")
        # Validate Confidence File (Ch6 6.4.1)

        log("Checking confidence file from Ch6 6.4.1...")

        # Check file exists
        validation_result = check_file_exists(CONFIDENCE_PATH, min_size_bytes=100)
        if not validation_result.get('valid', False):
            log(f"Confidence file not found: {CONFIDENCE_PATH}")
            log("SOURCE RQ INCOMPLETE: Ch6 6.4.1 confidence data missing")
            sys.exit(1)

        # Load confidence data
        df_confidence = pd.read_csv(CONFIDENCE_PATH, encoding='utf-8')
        log(f"Confidence file: {len(df_confidence)} rows, {len(df_confidence.columns)} columns")

        # Validate structure
        expected_cols = ['composite_ID', 'theta_IFR', 'theta_ICR', 'theta_IRE']
        missing_cols = [c for c in expected_cols if c not in df_confidence.columns]
        if missing_cols:
            log(f"Missing columns in confidence file: {missing_cols}")
            log(f"Found columns: {list(df_confidence.columns)}")
            sys.exit(1)

        # Validate row count
        if len(df_confidence) != 400:
            log(f"Confidence file has {len(df_confidence)} rows, expected 400")
            log("CONFIDENCE DATA INCOMPLETE: Expected 100 UID × 4 tests")
            sys.exit(1)

        # Check for missing values in theta columns
        theta_cols = ['theta_IFR', 'theta_ICR', 'theta_IRE']
        for col in theta_cols:
            missing_count = df_confidence[col].isna().sum()
            if missing_count > 0:
                log(f"Confidence file has {missing_count} missing values in {col}")
                sys.exit(1)

        log("Confidence file validation complete")
        # Save Validated Data
        # These copies ensure analysis pipeline uses validated inputs

        log("Saving validated dependency data...")

        # Save accuracy data
        accuracy_out = RQ_DIR / "data" / "step00_accuracy_raw.csv"
        df_accuracy.to_csv(accuracy_out, index=False, encoding='utf-8')
        log(f"{accuracy_out.name} ({len(df_accuracy)} rows)")

        # Save confidence data
        confidence_out = RQ_DIR / "data" / "step00_confidence_raw.csv"
        df_confidence.to_csv(confidence_out, index=False, encoding='utf-8')
        log(f"{confidence_out.name} ({len(df_confidence)} rows)")
        # Write Validation Report
        # Comprehensive validation summary for audit trail

        report_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DEPENDENCY VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"RQ: 6.9.7 - Paradigm-Specific Calibration Trajectory\n")
            f.write(f"Validation Date: 2026-01-19\n\n")

            f.write("SOURCE RQ 1: Ch5 5.3.1 (Accuracy by Paradigm)\n")
            f.write("-"*70 + "\n")
            f.write(f"File: {ACCURACY_PATH}\n")
            f.write(f"Status: File exists and validated\n")
            f.write(f"Rows: {len(df_accuracy)} (expected 1200)\n")
            f.write(f"Columns: {list(df_accuracy.columns)}\n")
            f.write(f"Domain values: {sorted(df_accuracy['domain_name'].unique())}\n")
            f.write(f"Missing theta values: 0\n\n")

            f.write("SOURCE RQ 2: Ch6 6.4.1 (Confidence by Paradigm)\n")
            f.write("-"*70 + "\n")
            f.write(f"File: {CONFIDENCE_PATH}\n")
            f.write(f"Status: File exists and validated\n")
            f.write(f"Rows: {len(df_confidence)} (expected 400)\n")
            f.write(f"Columns: {list(df_confidence.columns)}\n")
            f.write(f"Missing IFR values: 0\n")
            f.write(f"Missing ICR values: 0\n")
            f.write(f"Missing IRE values: 0\n\n")

            f.write("OVERALL VALIDATION RESULT\n")
            f.write("-"*70 + "\n")
            f.write("All validation checks passed\n")
            f.write("Ready to proceed with Step 1 (reshape and merge)\n")

        log(f"{report_path.name}")

        log("Step 0 complete - all dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
