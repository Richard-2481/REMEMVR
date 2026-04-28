#!/usr/bin/env python3
"""Validate Cross-RQ Dependencies: Verify that prerequisite RQs (Ch5 5.1.1 and Ch6 6.1.1) have completed"""

import sys
from pathlib import Path
import pandas as pd
import re
from typing import Dict, List, Tuple

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.1

LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
VALIDATION_OUTPUT = RQ_DIR / "data" / "step00_dependency_validation.txt"


# Prerequisite RQ Paths

# Ch5 5.1.1 (Accuracy IRT - 2PL)
CH5_RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.1.1"
CH5_STATUS = CH5_RQ_DIR / "status.yaml"
CH5_DATA_PRIMARY = CH5_RQ_DIR / "data" / "step03_theta_scores.csv"
CH5_DATA_PATTERN = CH5_RQ_DIR / "data" / "*theta*.csv"

# Ch6 6.1.1 (Confidence IRT - GRM)
CH6_RQ_DIR = PROJECT_ROOT / "results" / "ch6" / "6.1.1"
CH6_STATUS = CH6_RQ_DIR / "status.yaml"
CH6_CONF_PRIMARY = CH6_RQ_DIR / "data" / "step03_theta_confidence.csv"
CH6_CONF_PATTERN = CH6_RQ_DIR / "data" / "*theta*.csv"
CH6_TIME_PRIMARY = CH6_RQ_DIR / "data" / "step00_tsvr_mapping.csv"
CH6_TIME_PATTERN = CH6_RQ_DIR / "data" / "*tsvr*.csv"

# Expected file structures
EXPECTED_STRUCTURE = {
    "accuracy": {
        "columns": ["UID", "test", "Theta_All"],
        "rows": 400,
        "description": "Accuracy theta scores (2PL IRT)"
    },
    "confidence": {
        "columns": ["composite_ID", "theta_All", "se_All"],
        "rows": 400,
        "description": "Confidence theta scores (GRM)"
    },
    "time_mapping": {
        "columns": ["composite_ID", "TSVR_hours", "test"],
        "rows": 400,
        "description": "Time since VR mapping"
    }
}

# Logging Function

def log(msg: str, validation_log: List[str] = None):
    """Write to log file, console, and optional validation log."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)
    if validation_log is not None:
        validation_log.append(msg)

# Validation Functions

def check_status_file(status_path: Path, rq_name: str, validation_log: List[str]) -> bool:
    """
    Check that prerequisite RQ completed successfully.

    Returns:
        True if results analysis: status: success found, False otherwise
    """
    log(f"Checking {rq_name} status file...", validation_log)

    if not status_path.exists():
        log(f"Status file not found: {status_path}", validation_log)
        return False

    # Read status.yaml and search for completion indicators
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for multiple patterns - status.yaml formats vary between chapters
        # Pattern 1: results analysis: followed by status: success (multi-line)
        # Pattern 2: results analysis: success (single line)
        # Pattern 3: Just "results analysis: success" at start of line
        patterns = [
            r'results analysis:\s*success',  # Single line format
            r'results analysis:\s*\n\s*status:\s*success',  # Multi-line format
            r'^results analysis:\s*success'  # At start of line only
        ]

        found = False
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                found = True
                log(f"{rq_name} status: success (pattern: {pattern})", validation_log)
                break

        if found:
            return True
        else:
            log(f"{rq_name}: results analysis pattern not found, checking for validation status", validation_log)
            # Check for quality validation (highest level, implies completion)
            if re.search(r'certification:\s*\n\s*status:\s*validation', content):
                log(f"{rq_name} has quality validation (highest validation level)", validation_log)
                return True
            else:
                log(f"{rq_name} not complete - neither results analysis: success nor validation found", validation_log)
                log(f"Searched for patterns: {patterns}", validation_log)
                return False

    except Exception as e:
        log(f"Failed to read status file: {e}", validation_log)
        return False

def locate_file(primary_path: Path, pattern: str, file_type: str, validation_log: List[str]) -> Path:
    """
    Locate data file, trying primary path first, then glob pattern.

    Returns:
        Path to file if found, None otherwise
    """
    log(f"Searching for {file_type}...", validation_log)

    # Try primary path
    if primary_path.exists():
        log(f"Primary path: {primary_path}", validation_log)
        return primary_path

    # Try alternative pattern
    log(f"Primary not found, trying pattern: {pattern}", validation_log)
    parent_dir = Path(pattern).parent
    pattern_str = Path(pattern).name

    matches = list(parent_dir.glob(pattern_str))
    if matches:
        found_path = matches[0]
        log(f"Alternative path: {found_path}", validation_log)
        if len(matches) > 1:
            log(f"Multiple matches found, using first: {[str(m) for m in matches]}", validation_log)
        return found_path

    log(f"File not found: {file_type}", validation_log)
    log(f"Primary: {primary_path}", validation_log)
    log(f"Pattern: {pattern}", validation_log)
    return None

def verify_file_structure(file_path: Path, expected: Dict, file_type: str, validation_log: List[str]) -> bool:
    """
    Load CSV file and verify columns and row count.

    Returns:
        True if structure matches, False otherwise
    """
    log(f"Checking structure of {file_type}...", validation_log)

    try:
        # Load file
        df = pd.read_csv(file_path, encoding='utf-8')

        # Check columns
        actual_cols = df.columns.tolist()
        expected_cols = expected["columns"]

        if actual_cols != expected_cols:
            log(f"Column mismatch for {file_type}", validation_log)
            log(f"Expected: {expected_cols}", validation_log)
            log(f"Actual: {actual_cols}", validation_log)
            return False

        # Check row count (allow 10% tolerance)
        actual_rows = len(df)
        expected_rows = expected["rows"]
        tolerance = 0.1 * expected_rows

        if abs(actual_rows - expected_rows) > tolerance:
            log(f"Row count mismatch for {file_type}", validation_log)
            log(f"Expected: ~{expected_rows} rows", validation_log)
            log(f"Actual: {actual_rows} rows", validation_log)
            return False

        log(f"{file_type}: {actual_rows} rows, {len(actual_cols)} columns", validation_log)
        log(f"Columns: {actual_cols}", validation_log)

        # Check for missing data
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            log(f"Missing data detected in {file_type}:", validation_log)
            for col, count in missing_counts[missing_counts > 0].items():
                pct = (count / len(df)) * 100
                log(f"{col}: {count} missing ({pct:.1f}%)", validation_log)

        return True

    except Exception as e:
        log(f"Failed to verify {file_type}: {e}", validation_log)
        return False

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 0: Validate Cross-RQ Dependencies")

        # Initialize validation log (for txt output)
        validation_log = []
        validation_log.append("=" * 80)
        validation_log.append("RQ 6.9.1 - DEPENDENCY VALIDATION REPORT")
        validation_log.append("=" * 80)
        validation_log.append("")

        # Track validation status
        all_checks_passed = True
        # Check Prerequisite Status Files
        log("[STEP 1] Checking prerequisite RQ status files...")
        validation_log.append("-" * 80)
        validation_log.append("STEP 1: Status File Checks")
        validation_log.append("-" * 80)

        # Check Ch5 5.1.1 status
        ch5_status_ok = check_status_file(CH5_STATUS, "Ch5 5.1.1", validation_log)
        if not ch5_status_ok:
            all_checks_passed = False

        # Check Ch6 6.1.1 status
        ch6_status_ok = check_status_file(CH6_STATUS, "Ch6 6.1.1", validation_log)
        if not ch6_status_ok:
            all_checks_passed = False

        if ch5_status_ok and ch6_status_ok:
            log("Both prerequisite RQs completed successfully")
            validation_log.append("")
            validation_log.append("Ch5 5.1.1 COMPLETE")
            validation_log.append("Ch6 6.1.1 COMPLETE")
        else:
            log("Prerequisite status check failed")
            validation_log.append("")
            validation_log.append("STATUS CHECK FAILED - Prerequisites not complete")
        # Locate Data Files
        log("[STEP 2] Locating required data files...")
        validation_log.append("")
        validation_log.append("-" * 80)
        validation_log.append("STEP 2: Data File Location")
        validation_log.append("-" * 80)

        # Locate accuracy theta
        accuracy_path = locate_file(CH5_DATA_PRIMARY, str(CH5_DATA_PATTERN), "accuracy theta", validation_log)

        # Locate confidence theta
        confidence_path = locate_file(CH6_CONF_PRIMARY, str(CH6_CONF_PATTERN), "confidence theta", validation_log)

        # Locate time mapping
        time_path = locate_file(CH6_TIME_PRIMARY, str(CH6_TIME_PATTERN), "time mapping", validation_log)

        # Check if all files found
        if accuracy_path and confidence_path and time_path:
            log("All 3 required files located")
            validation_log.append("")
            validation_log.append("All 3 files located")
        else:
            log("One or more required files not found")
            validation_log.append("")
            validation_log.append("MISSING FILES - Cannot proceed with analysis")
            all_checks_passed = False
        # Verify File Structures
        if all_checks_passed:
            log("[STEP 3] Verifying file structures...")
            validation_log.append("")
            validation_log.append("-" * 80)
            validation_log.append("STEP 3: File Structure Verification")
            validation_log.append("-" * 80)

            # Verify accuracy theta
            acc_ok = verify_file_structure(accuracy_path, EXPECTED_STRUCTURE["accuracy"],
                                         "accuracy theta", validation_log)

            # Verify confidence theta
            conf_ok = verify_file_structure(confidence_path, EXPECTED_STRUCTURE["confidence"],
                                          "confidence theta", validation_log)

            # Verify time mapping
            time_ok = verify_file_structure(time_path, EXPECTED_STRUCTURE["time_mapping"],
                                          "time mapping", validation_log)

            if acc_ok and conf_ok and time_ok:
                log("All file structures validated")
                validation_log.append("")
                validation_log.append("All file structures valid")
                validation_log.append(f"Accuracy: {len(pd.read_csv(accuracy_path))} rows")
                validation_log.append(f"Confidence: {len(pd.read_csv(confidence_path))} rows")
                validation_log.append(f"Time mapping: {len(pd.read_csv(time_path))} rows")
            else:
                log("File structure validation failed")
                validation_log.append("")
                validation_log.append("STRUCTURE MISMATCH - File formats invalid")
                all_checks_passed = False
        else:
            log("Skipping structure verification (prerequisite checks failed)")
        # Write Validation Report
        log("[STEP 4] Writing validation report...")
        validation_log.append("")
        validation_log.append("=" * 80)
        validation_log.append("FINAL VALIDATION STATUS")
        validation_log.append("=" * 80)

        if all_checks_passed:
            validation_log.append("All dependency checks passed")
            validation_log.append("")
            validation_log.append("Ready for analysis pipeline:")
            validation_log.append("  - Step 1: Merge trajectories")
            validation_log.append("  - Step 2: Compute decline rates")
            validation_log.append("  - Step 3: Paired t-test (PRIMARY)")
            log("All dependency checks passed")
        else:
            validation_log.append("Dependency validation failed")
            validation_log.append("")
            validation_log.append("Action required:")
            validation_log.append("  1. Complete prerequisite RQs (Ch5 5.1.1, Ch6 6.1.1)")
            validation_log.append("  2. Verify all required output files exist")
            validation_log.append("  3. Re-run this validation script")
            log("Dependency validation failed")

        validation_log.append("")
        validation_log.append("=" * 80)

        # Write validation report
        with open(VALIDATION_OUTPUT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(validation_log))

        log(f"Validation report: {VALIDATION_OUTPUT}")

        # Exit with appropriate code
        if all_checks_passed:
            log("Step 0 complete - dependencies validated")
            sys.exit(0)
        else:
            log("Step 0 failed - dependencies not satisfied")
            sys.exit(1)

    except Exception as e:
        log(f"Unexpected error during validation: {str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
