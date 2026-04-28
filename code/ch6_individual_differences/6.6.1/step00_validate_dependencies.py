#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.1.4 slope outputs and dfnonvr.csv cognitive data availability."""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"


# Dependencies to Validate

DEPENDENCIES = {
    "ch5_status": {
        "path": PROJECT_ROOT / "results" / "ch5" / "5.1.4" / "status.yaml",
        "min_size": 100,
        "check_content": True,
        "required_content": "results analysis",
        "expected_status": "success",
        "description": "Ch5 5.1.4 completion status"
    },
    "ch5_slopes": {
        "path": PROJECT_ROOT / "results" / "ch5" / "5.1.4" / "data" / "step04_random_effects.csv",
        "min_size": 1000,
        "check_columns": True,
        "expected_columns": ["UID", "random_slope"],
        "description": "Individual slope estimates from Ch5 5.1.4"
    },
    "dfnonvr": {
        "path": PROJECT_ROOT / "data" / "dfnonvr.csv",
        "min_size": 10000,
        "check_columns": True,
        "expected_columns": ["UID", "ravlt-trial-1-score", "bvmt-trial-1-score", "rpm-score"],
        "description": "Cognitive test data"
    }
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Validation Functions

def validate_yaml_content(yaml_path: Path, required_key: str, expected_status: str) -> Dict[str, Any]:
    """
    Validate that YAML file contains required key with expected status.
    Handles nested dict structure (results analysis.status) or flat string (results analysis: 'success').
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if required_key not in data:
            return {
                'valid': False,
                'reason': f"Key '{required_key}' not found in YAML"
            }

        # Handle both dict and string formats (Lesson #5 from gcode_lessons.md)
        results analysis_value = data.get(required_key, {})
        if isinstance(results analysis_value, dict):
            actual_status = results analysis_value.get('status')
        elif isinstance(results analysis_value, str):
            actual_status = results analysis_value
        else:
            actual_status = 'unknown'

        if actual_status == expected_status:
            return {
                'valid': True,
                'status': actual_status
            }
        else:
            return {
                'valid': False,
                'reason': f"Expected status '{expected_status}', found '{actual_status}'"
            }

    except Exception as e:
        return {
            'valid': False,
            'reason': f"Error reading YAML: {str(e)}"
        }

def validate_csv_columns(csv_path: Path, expected_columns: List[str]) -> Dict[str, Any]:
    """
    Validate that CSV file has expected columns (exact match).
    """
    try:
        # Read only header (nrows=0 for speed)
        df = pd.read_csv(csv_path, nrows=0)
        actual_columns = df.columns.tolist()

        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in actual_columns]

        if not missing_columns:
            return {
                'valid': True,
                'columns': actual_columns,
                'n_columns': len(actual_columns)
            }
        else:
            return {
                'valid': False,
                'reason': f"Missing columns: {missing_columns}",
                'expected': expected_columns,
                'actual': actual_columns
            }

    except Exception as e:
        return {
            'valid': False,
            'reason': f"Error reading CSV: {str(e)}"
        }

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")
        log("")

        # Initialize output report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DEPENDENCY VALIDATION REPORT")
        report_lines.append("RQ 7.6.1: Cognitive Tests Predicting Individual Differences in Forgetting Rate")
        report_lines.append("=" * 80)
        report_lines.append("")

        all_valid = True
        validation_results = {}
        # Validate Each Dependency

        for dep_name, dep_config in DEPENDENCIES.items():
            log(f"Checking {dep_name}: {dep_config['description']}")
            report_lines.append(f"Dependency: {dep_name}")
            report_lines.append(f"Description: {dep_config['description']}")
            report_lines.append(f"Path: {dep_config['path']}")

            # Check file existence and size
            file_check = check_file_exists(
                file_path=dep_config['path'],
                min_size_bytes=dep_config['min_size']
            )

            if not file_check.get('valid', False):
                all_valid = False
                log(f"{dep_name}: File check failed")
                report_lines.append(f"Status: ")
                report_lines.append(f"Reason: {file_check.get('reason', 'File does not exist or too small')}")
            else:
                log(f"{dep_name}: File exists ({file_check.get('size_bytes', 0)} bytes)")
                report_lines.append(f"Status: File exists")
                report_lines.append(f"Size: {file_check.get('size_bytes', 0)} bytes")

                # Additional validation based on dependency type
                if dep_config.get('check_content', False):
                    # Validate YAML content
                    yaml_result = validate_yaml_content(
                        dep_config['path'],
                        dep_config['required_content'],
                        dep_config['expected_status']
                    )

                    if yaml_result.get('valid', False):
                        log(f"{dep_name}: Status = {yaml_result.get('status')}")
                        report_lines.append(f"Content: {dep_config['required_content']} = {yaml_result.get('status')}")
                    else:
                        all_valid = False
                        log(f"{dep_name}: {yaml_result.get('reason')}")
                        report_lines.append(f"Content: {yaml_result.get('reason')}")

                elif dep_config.get('check_columns', False):
                    # Validate CSV columns
                    csv_result = validate_csv_columns(
                        dep_config['path'],
                        dep_config['expected_columns']
                    )

                    if csv_result.get('valid', False):
                        log(f"{dep_name}: Columns validated ({csv_result.get('n_columns')} total)")
                        report_lines.append(f"Columns: Expected columns present")
                        report_lines.append(f"Expected: {', '.join(dep_config['expected_columns'])}")
                        report_lines.append(f"Total columns: {csv_result.get('n_columns')}")
                    else:
                        all_valid = False
                        log(f"{dep_name}: {csv_result.get('reason')}")
                        report_lines.append(f"Columns: {csv_result.get('reason')}")
                        report_lines.append(f"Expected: {', '.join(dep_config['expected_columns'])}")
                        report_lines.append(f"Actual: {', '.join(csv_result.get('actual', []))}")

            validation_results[dep_name] = file_check
            report_lines.append("")
            log("")
        # Generate Summary Report

        report_lines.append("=" * 80)
        report_lines.append("VALIDATION SUMMARY")
        report_lines.append("=" * 80)

        if all_valid:
            log("All dependencies validated successfully")
            report_lines.append("Overall Status: ")
            report_lines.append("")
            report_lines.append("All required dependencies are in place.")
            report_lines.append("Ready to proceed with RQ 7.6.1 analysis.")
        else:
            log("One or more dependencies failed validation")
            report_lines.append("Overall Status: ")
            report_lines.append("")
            report_lines.append("One or more dependencies are missing or invalid.")
            report_lines.append("Please resolve issues before proceeding.")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        # Save Report to Output File

        log(f"Writing validation report to {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        log(f"Validation report written")

        # Exit with appropriate code
        if all_valid:
            log("Step 00 complete")
            sys.exit(0)
        else:
            log("Step 00 failed - dependencies not satisfied")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
