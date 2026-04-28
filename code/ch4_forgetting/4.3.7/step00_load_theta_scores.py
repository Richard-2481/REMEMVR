#!/usr/bin/env python3
"""Load Theta Scores from RQ 5.3.1: Load paradigm-specific theta scores from RQ 5.3.1 and validate data structure."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.7 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_load_theta_scores.log"

# Input file from RQ 5.3.1
SOURCE_FILE = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step04_lmm_input.csv"

# Output files
OUTPUT_VALIDATED = RQ_DIR / "data" / "step00_theta_scores_validated.csv"
OUTPUT_SUMMARY = RQ_DIR / "data" / "step00_validation_summary.txt"

# Expected structure (actual from RQ 5.3.1)
EXPECTED_ROWS = 1200
EXPECTED_COLS = ['composite_ID', 'UID', 'test', 'TSVR_hours', 'TSVR_hours_sq', 'TSVR_hours_log', 'paradigm', 'theta']
EXPECTED_PARADIGMS = {'free_recall': 400, 'cued_recall': 400, 'recognition': 400}
EXPECTED_TESTS = {1: 300, 2: 300, 3: 300, 4: 300}
EXPECTED_PARTICIPANTS = 100
ROWS_PER_PARTICIPANT = 12  # 4 tests × 3 paradigms

# Value ranges
THETA_RANGE = (-4, 4)
TSVR_RANGE = (0, 300)  # Extended to 300 hours to accommodate all TSVR values


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Validation Functions

def validate_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if source file exists."""
    if not file_path.exists():
        return False, f"Source file not found: {file_path}"
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    return True, f"Source file exists: {file_path}"

def validate_row_count(df: pd.DataFrame, expected: int) -> Tuple[bool, str]:
    """Check if DataFrame has expected number of rows."""
    actual = len(df)
    if actual != expected:
        return False, f"Row count mismatch: expected {expected}, got {actual}"
    return True, f"Row count valid: {actual} rows"

def validate_columns(df: pd.DataFrame, expected_cols: List[str]) -> Tuple[bool, str]:
    """Check if DataFrame has expected columns."""
    actual_cols = df.columns.tolist()
    if actual_cols != expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing columns: {missing}")
        if extra:
            msg_parts.append(f"Extra columns: {extra}")
        return False, f"Column mismatch: {'; '.join(msg_parts)}"
    return True, f"Columns valid: {len(actual_cols)} columns match expected"

def validate_paradigm_balance(df: pd.DataFrame, expected: Dict[str, int]) -> Tuple[bool, str]:
    """Check if paradigm categories have expected row counts."""
    actual = df['paradigm'].value_counts().to_dict()
    mismatches = []
    for paradigm, expected_count in expected.items():
        actual_count = actual.get(paradigm, 0)
        if actual_count != expected_count:
            mismatches.append(f"{paradigm}: expected {expected_count}, got {actual_count}")

    if mismatches:
        return False, f"Paradigm balance mismatch: {'; '.join(mismatches)}"
    return True, f"Paradigm balance valid: {', '.join([f'{k}={v}' for k, v in expected.items()])}"

def validate_test_balance(df: pd.DataFrame, expected: Dict[str, int]) -> Tuple[bool, str]:
    """Check if test session categories have expected row counts."""
    actual = df['test'].value_counts().to_dict()
    mismatches = []
    for test, expected_count in expected.items():
        actual_count = actual.get(test, 0)
        if actual_count != expected_count:
            mismatches.append(f"{test}: expected {expected_count}, got {actual_count}")

    if mismatches:
        return False, f"Test balance mismatch: {'; '.join(mismatches)}"
    return True, f"Test balance valid: {', '.join([f'{k}={v}' for k, v in expected.items()])}"

def validate_missing_data(df: pd.DataFrame, critical_cols: List[str]) -> Tuple[bool, str]:
    """Check for missing data in critical columns."""
    missing_counts = df[critical_cols].isna().sum()
    if missing_counts.sum() > 0:
        missing_report = ', '.join([f"{col}={count}" for col, count in missing_counts.items() if count > 0])
        return False, f"Missing data detected: {missing_report}"
    return True, f"No missing data in critical columns: {critical_cols}"

def validate_value_range(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> Tuple[bool, str]:
    """Check if values in column are within expected range."""
    values = df[col].dropna()
    out_of_range = ((values < min_val) | (values > max_val)).sum()
    if out_of_range > 0:
        min_actual = values.min()
        max_actual = values.max()
        return False, f"{col} range violation: expected [{min_val}, {max_val}], got [{min_actual:.2f}, {max_actual:.2f}] ({out_of_range} values out of range)"
    return True, f"{col} range valid: all values in [{min_val}, {max_val}]"

def validate_participant_balance(df: pd.DataFrame, expected_participants: int, rows_per_participant: int) -> Tuple[bool, str]:
    """Check if all participants have expected number of rows."""
    uid_counts = df['UID'].value_counts()
    total_uids = len(uid_counts)

    if total_uids != expected_participants:
        return False, f"Participant count mismatch: expected {expected_participants}, got {total_uids}"

    # Check if all participants have correct row count
    incorrect_counts = uid_counts[uid_counts != rows_per_participant]
    if len(incorrect_counts) > 0:
        examples = incorrect_counts.head(5).to_dict()
        return False, f"Some participants have incorrect row counts: {examples}"

    return True, f"Participant balance valid: {total_uids} participants with {rows_per_participant} rows each"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Load Theta Scores from RQ 5.3.1")
        log("")
        # VALIDATION 1: Check Source File Exists

        log("Validating source file exists...")
        valid, msg = validate_file_exists(SOURCE_FILE)
        log(f"  {msg}")
        if not valid:
            log("EXPECTATIONS ERROR: RQ 5.3.1 must complete before RQ 5.3.7")
            sys.exit(1)
        log("")
        # Load Input Data

        log("Loading theta scores from RQ 5.3.1...")
        df = pd.read_csv(SOURCE_FILE)
        log(f"{len(df)} rows, {len(df.columns)} columns")
        log(f"  Columns: {df.columns.tolist()}")
        log("")
        # Run Comprehensive Validation

        log("Running comprehensive validation checks...")

        validation_results = []
        all_valid = True

        # Check 1: Row count
        valid, msg = validate_row_count(df, EXPECTED_ROWS)
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Row count: {msg}")
        all_valid = all_valid and valid

        # Check 2: Columns
        valid, msg = validate_columns(df, EXPECTED_COLS)
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Columns: {msg}")
        all_valid = all_valid and valid

        # Check 3: Paradigm balance
        valid, msg = validate_paradigm_balance(df, EXPECTED_PARADIGMS)
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Paradigm balance: {msg}")
        all_valid = all_valid and valid

        # Check 4: Test balance
        valid, msg = validate_test_balance(df, EXPECTED_TESTS)
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Test balance: {msg}")
        all_valid = all_valid and valid

        # Check 5: Missing data
        valid, msg = validate_missing_data(df, ['theta', 'TSVR_hours', 'paradigm'])
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Missing data: {msg}")
        all_valid = all_valid and valid

        # Check 6: Theta range
        valid, msg = validate_value_range(df, 'theta', THETA_RANGE[0], THETA_RANGE[1])
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Theta range: {msg}")
        all_valid = all_valid and valid

        # Check 7: TSVR range
        valid, msg = validate_value_range(df, 'TSVR_hours', TSVR_RANGE[0], TSVR_RANGE[1])
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] TSVR_hours range: {msg}")
        all_valid = all_valid and valid

        # Check 8: Participant balance
        valid, msg = validate_participant_balance(df, EXPECTED_PARTICIPANTS, ROWS_PER_PARTICIPANT)
        validation_results.append(f"[{'PASS' if valid else 'FAIL'}] Participant balance: {msg}")
        all_valid = all_valid and valid

        # Log validation results
        for result in validation_results:
            log(f"  {result}")
        log("")

        if not all_valid:
            log("Validation failed - see errors above")
            sys.exit(1)

        log("All validation checks passed")
        log("")
        # Save Validated Copy

        log("Saving validated theta scores...")
        df.to_csv(OUTPUT_VALIDATED, index=False, encoding='utf-8')
        log(f"{OUTPUT_VALIDATED} ({len(df)} rows, {len(df.columns)} columns)")
        log("")
        # Write Validation Summary

        log("Writing validation summary...")
        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.3.7 Step 0: Theta Scores Validation Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write("SOURCE FILE:\n")
            f.write(f"  {SOURCE_FILE}\n\n")

            f.write("DATA STRUCTURE:\n")
            f.write(f"  Rows: {len(df)}\n")
            f.write(f"  Columns: {len(df.columns)}\n")
            f.write(f"  Column names: {df.columns.tolist()}\n\n")

            f.write("PARADIGM DISTRIBUTION:\n")
            for paradigm, count in df['paradigm'].value_counts().sort_index().items():
                f.write(f"  {paradigm}: {count} rows\n")
            f.write("\n")

            f.write("TEST SESSION DISTRIBUTION:\n")
            for test, count in df['test'].value_counts().sort_index().items():
                f.write(f"  {test}: {count} rows\n")
            f.write("\n")

            f.write("PARTICIPANT BALANCE:\n")
            f.write(f"  Total participants: {df['UID'].nunique()}\n")
            f.write(f"  Rows per participant: {df.groupby('UID').size().mean():.1f} (expected: {ROWS_PER_PARTICIPANT})\n\n")

            f.write("VALUE RANGES:\n")
            f.write(f"  theta: [{df['theta'].min():.3f}, {df['theta'].max():.3f}] (expected: [{THETA_RANGE[0]}, {THETA_RANGE[1]}])\n")
            f.write(f"  TSVR_hours: [{df['TSVR_hours'].min():.3f}, {df['TSVR_hours'].max():.3f}] (expected: [{TSVR_RANGE[0]}, {TSVR_RANGE[1]}])\n\n")

            f.write("MISSING DATA:\n")
            missing_counts = df.isna().sum()
            if missing_counts.sum() == 0:
                f.write("  No missing data detected\n\n")
            else:
                for col, count in missing_counts.items():
                    if count > 0:
                        f.write(f"  {col}: {count} missing ({count/len(df)*100:.2f}%)\n")
                f.write("\n")

            f.write("VALIDATION RESULTS:\n")
            for result in validation_results:
                f.write(f"  {result}\n")
            f.write("\n")

            f.write("CONCLUSION:\n")
            if all_valid:
                f.write("  All validation checks PASSED\n")
                f.write("  Data is ready for variance decomposition analysis\n")
            else:
                f.write("  Validation FAILED - see errors above\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        log(f"{OUTPUT_SUMMARY}")
        log("")

        log("Step 0 complete")
        log(f"  Validated data: {OUTPUT_VALIDATED}")
        log(f"  Summary report: {OUTPUT_SUMMARY}")
        log(f"  Next: Run step01_load_model_metadata.py")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
