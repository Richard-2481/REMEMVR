#!/usr/bin/env python3
"""Validate Cross-RQ Dependencies: Verify that Ch5 5.5.1 (accuracy theta scores) and Ch6 6.8.1 (confidence theta"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.validation import check_file_exists, validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.5
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"

# Cross-RQ dependencies
CH5_RQ = PROJECT_ROOT / "results" / "ch5" / "5.5.1"
CH6_RQ = PROJECT_ROOT / "results" / "ch6" / "6.8.1"

DEPENDENCIES = [
    {
        "path": CH5_RQ / "data" / "step05_lmm_coefficients.csv",
        "description": "Accuracy model LMM coefficients (Location x Time interaction)",
        "min_size": 100
    },
    {
        "path": CH5_RQ / "data" / "step03_theta_scores.csv",
        "description": "Accuracy theta scores (WIDE format: theta_source, theta_destination)",
        "min_size": 1000
    },
    {
        "path": CH6_RQ / "results" / "summary.md",
        "description": "Confidence model summary (p=0.553 for interaction)",
        "min_size": 100
    },
    {
        "path": CH6_RQ / "data" / "step03_theta_confidence.csv",
        "description": "Confidence theta scores (WIDE format: theta_Source, theta_Destination)",
        "min_size": 1000
    }
]

STATUS_CHECKS = [
    {
        "path": CH5_RQ / "status.yaml",
        "rq_id": "ch5/5.5.1",
        "required_status": "results analysis: success"
    },
    {
        "path": CH6_RQ / "status.yaml",
        "rq_id": "ch6/6.8.1",
        "required_status": "results analysis: success"
    }
]

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 0: Validate Cross-RQ Dependencies")

        validation_report = []
        all_checks_passed = True
        # CHECK 1: Status Files
        log("\n[CHECK 1] Validating RQ Status Files")
        validation_report.append("="*80)
        validation_report.append("CROSS-RQ DEPENDENCY VALIDATION")
        validation_report.append("="*80)
        validation_report.append("")
        validation_report.append("CHECK 1: RQ Status Files")
        validation_report.append("-"*80)

        for status_check in STATUS_CHECKS:
            status_path = status_check["path"]
            rq_id = status_check["rq_id"]

            if not status_path.exists():
                log(f"{rq_id}: status.yaml not found at {status_path}")
                validation_report.append(f"{rq_id}: status.yaml NOT FOUND")
                all_checks_passed = False
                continue

            # Parse status.yaml - handle both nested and flat formats
            # Ch5 5.5.1: results analysis: {status: success, context_dump: ...}
            # Ch6 6.8.1: results analysis: success
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Simple text search for "results analysis: success" or "results analysis:" + "status: success"
                results analysis_status = "unknown"

                if 'results analysis: success' in content:
                    # Flat format: results analysis: success
                    results analysis_status = "success"
                elif 'results analysis:' in content:
                    # Nested format: check for status: success in subsequent lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'results analysis:' in line:
                            # Check next few lines for status: success
                            for j in range(i+1, min(i+10, len(lines))):
                                if 'status: success' in lines[j]:
                                    results analysis_status = "success"
                                    break
                                elif not lines[j].startswith((' ', '\t')) and lines[j].strip():
                                    # New top-level key, stop looking
                                    break
                            break

                if results analysis_status == 'success':
                    log(f"{rq_id}: results analysis = success")
                    validation_report.append(f"{rq_id}: results analysis = success")
                else:
                    log(f"{rq_id}: results analysis = {results analysis_status} (expected: success)")
                    validation_report.append(f"{rq_id}: results analysis = {results analysis_status}")
                    all_checks_passed = False

            except Exception as e:
                log(f"{rq_id}: Could not parse status.yaml: {e}")
                validation_report.append(f"{rq_id}: Status parse error")
                all_checks_passed = False
        # CHECK 2: Data Files Existence
        log("\n[CHECK 2] Validating Data File Existence")
        validation_report.append("")
        validation_report.append("CHECK 2: Data File Existence")
        validation_report.append("-"*80)

        for dep in DEPENDENCIES:
            file_path = dep["path"]
            description = dep["description"]
            min_size = dep["min_size"]

            result = check_file_exists(str(file_path), min_size_bytes=min_size)

            if result['valid']:
                size_kb = result['size_bytes'] / 1024
                log(f"{file_path.name}: {size_kb:.1f} KB")
                validation_report.append(f"{file_path.name}: {size_kb:.1f} KB")
                validation_report.append(f"       {description}")
            else:
                log(f"{file_path.name}: {result['message']}")
                validation_report.append(f"{file_path.name}: {result['message']}")
                all_checks_passed = False
        # CHECK 3: Data Structure Verification
        log("\n[CHECK 3] Validating Data Structure")
        validation_report.append("")
        validation_report.append("CHECK 3: Data Structure Verification")
        validation_report.append("-"*80)

        # Check accuracy data structure
        accuracy_path = CH5_RQ / "data" / "step03_theta_scores.csv"
        if accuracy_path.exists():
            df_accuracy = pd.read_csv(accuracy_path)
            expected_cols_accuracy = ['composite_ID', 'theta_source', 'theta_destination', 'se_source', 'se_destination']

            result = validate_data_format(df_accuracy, expected_cols_accuracy)
            if result['valid']:
                log(f"Accuracy data: {len(df_accuracy)} rows, columns OK")
                validation_report.append(f"Accuracy data: {len(df_accuracy)} rows")
                validation_report.append(f"       Columns: {', '.join(expected_cols_accuracy)}")
                validation_report.append(f"       Note: UID/test will be parsed from composite_ID")
            else:
                log(f"Accuracy data: {result['message']}")
                validation_report.append(f"Accuracy data: {result['message']}")
                all_checks_passed = False

        # Check confidence data structure
        confidence_path = CH6_RQ / "data" / "step03_theta_confidence.csv"
        if confidence_path.exists():
            df_confidence = pd.read_csv(confidence_path)
            expected_cols_confidence = ['composite_ID', 'theta_Source', 'theta_Destination']

            result = validate_data_format(df_confidence, expected_cols_confidence)
            if result['valid']:
                log(f"Confidence data: {len(df_confidence)} rows, columns OK")
                validation_report.append(f"Confidence data: {len(df_confidence)} rows")
                validation_report.append(f"       Columns: {', '.join(expected_cols_confidence)}")
                validation_report.append(f"       Note: UID/test will be parsed from composite_ID")
            else:
                log(f"Confidence data: {result['message']}")
                validation_report.append(f"Confidence data: {result['message']}")
                all_checks_passed = False
        # FINAL VERDICT
        validation_report.append("")
        validation_report.append("="*80)
        if all_checks_passed:
            validation_report.append("FINAL VERDICT: ALL DEPENDENCIES VALIDATED")
            validation_report.append("="*80)
            log("\nAll dependencies validated - ready to proceed")
        else:
            validation_report.append("FINAL VERDICT: DEPENDENCY VALIDATION FAILED")
            validation_report.append("="*80)
            log("\nDependency validation failed - cannot proceed")
        # SAVE VALIDATION REPORT
        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(validation_report))
        log(f"Validation report: {output_path}")

        # Exit with appropriate code
        if all_checks_passed:
            log("Step 0 complete")
            sys.exit(0)
        else:
            log("Step 0 failed - dependencies not met")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
