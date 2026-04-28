#!/usr/bin/env python3
"""step00_load_dependencies: Load theta scores from RQ 5.4.1, TSVR mapping from RQ 5.4.1, and Age variable"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.4.3
LOG_FILE = RQ_DIR / "logs" / "step00_load_dependencies.log"

# Dependency file paths (relative to project root)
THETA_SCORES_PATH = PROJECT_ROOT / "results/ch5/5.4.1/data/step03_theta_scores.csv"
TSVR_MAPPING_PATH = PROJECT_ROOT / "results/ch5/5.4.1/data/step00_tsvr_mapping.csv"
MASTER_DATA_PATH = PROJECT_ROOT / "data/cache/dfData.csv"

# Output file paths
OUTPUT_THETA_PATH = RQ_DIR / "data" / "step00_theta_wide.csv"
OUTPUT_TSVR_PATH = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
OUTPUT_AGE_PATH = RQ_DIR / "data" / "step00_age_data.csv"

# Validation parameters
EXPECTED_THETA_ROWS = 400
EXPECTED_TSVR_ROWS = 400
EXPECTED_MIN_UNIQUE_UIDS = 100  # At least 100 unique participants

VALUE_RANGES = {
    'theta_common': (-4.0, 4.0),
    'theta_congruent': (-4.0, 4.0),
    'theta_incongruent': (-4.0, 4.0),
    'se_common': (0.0, 2.0),  # Allow some flexibility (0.1-1.5 expected, but allow wider)
    'se_congruent': (0.0, 2.0),
    'se_incongruent': (0.0, 2.0),
    'TSVR_hours': (0.0, 250.0),  # Actual max ~148h, allow margin
    'Age': (20.0, 70.0)
}


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Validation Functions

def validate_file_exists(file_path: Path, file_label: str) -> None:
    """Validate that a file exists."""
    if not file_path.exists():
        raise FileNotFoundError(f"{file_label} not found at: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"{file_label} is not a file: {file_path}")
    log(f"{file_label} exists: {file_path}")


def validate_columns(df: pd.DataFrame, expected_cols: List[str],
                      data_label: str) -> None:
    """Validate that DataFrame has expected columns."""
    df_cols = df.columns.tolist()
    missing_cols = [c for c in expected_cols if c not in df_cols]
    extra_cols = [c for c in df_cols if c not in expected_cols]

    if missing_cols:
        raise ValueError(
            f"{data_label} missing columns: {missing_cols}\n"
            f"  Expected: {expected_cols}\n"
            f"  Found: {df_cols}"
        )

    log(f"{data_label} has all expected columns: {expected_cols}")
    if extra_cols:
        log(f"{data_label} has additional columns: {extra_cols}")


def validate_row_count(df: pd.DataFrame, expected_rows: int, data_label: str,
                        tolerance: int = 10) -> None:
    """Validate that DataFrame has approximately expected row count."""
    actual_rows = len(df)

    if abs(actual_rows - expected_rows) > tolerance:
        raise ValueError(
            f"{data_label} row count mismatch: "
            f"expected ~{expected_rows} (±{tolerance}), got {actual_rows}"
        )

    log(f"{data_label} row count: {actual_rows} (expected ~{expected_rows})")


def validate_no_nan(df: pd.DataFrame, columns: List[str], data_label: str) -> None:
    """Validate that specified columns have no NaN values."""
    nan_counts = {}
    has_nan = False

    for col in columns:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                nan_counts[col] = n_nan
                has_nan = True

    if has_nan:
        raise ValueError(
            f"{data_label} has NaN values:\n" +
            "\n".join([f"  {col}: {count} NaN" for col, count in nan_counts.items()])
        )

    log(f"{data_label} has no NaN in critical columns: {columns}")


def validate_value_ranges(df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]],
                          data_label: str) -> None:
    """Validate that numeric columns fall within expected ranges."""
    violations = []

    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            continue

        below_min = (df[col] < min_val).sum()
        above_max = (df[col] > max_val).sum()

        if below_min > 0 or above_max > 0:
            actual_min = df[col].min()
            actual_max = df[col].max()
            violations.append(
                f"  {col}: expected [{min_val}, {max_val}], "
                f"got [{actual_min:.2f}, {actual_max:.2f}] "
                f"({below_min} below, {above_max} above)"
            )

    if violations:
        raise ValueError(
            f"{data_label} has values out of expected ranges:\n" +
            "\n".join(violations)
        )

    log(f"{data_label} values within expected ranges")


def parse_uids_from_composite_id(composite_ids: pd.Series) -> pd.Series:
    """
    Parse UIDs from composite_ID column.

    composite_ID format: "UID_testnum" (e.g., "A010_1", "A010_2")
    Returns UIDs: "A010", "A010"
    """
    return composite_ids.str.split('_').str[0]


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load and Validate Dependency Files")
        log("")
        # Create output directories if they don't exist
        (RQ_DIR / "data").mkdir(parents=True, exist_ok=True)
        (RQ_DIR / "logs").mkdir(parents=True, exist_ok=True)
        log("Created output directories (if not exist)")
        log("")
        # Validate Dependency Files Exist
        log("[STEP 1] Validate Dependency Files Exist")
        log("-" * 70)

        validate_file_exists(THETA_SCORES_PATH, "Theta scores (RQ 5.4.1)")
        validate_file_exists(TSVR_MAPPING_PATH, "TSVR mapping (RQ 5.4.1)")
        validate_file_exists(MASTER_DATA_PATH, "Master data (dfData.csv)")
        log("")
        # Load Theta Scores from RQ 5.4.1
        log("[STEP 2] Load Theta Scores from RQ 5.4.1")
        log("-" * 70)

        df_theta = pd.read_csv(THETA_SCORES_PATH)
        log(f"Theta scores: {len(df_theta)} rows, {len(df_theta.columns)} columns")

        # Validate theta scores structure
        expected_theta_cols = [
            'composite_ID', 'theta_common', 'theta_congruent', 'theta_incongruent',
            'se_common', 'se_congruent', 'se_incongruent'
        ]
        validate_columns(df_theta, expected_theta_cols, "Theta scores")
        validate_row_count(df_theta, EXPECTED_THETA_ROWS, "Theta scores")

        # Validate no NaN in theta columns
        theta_critical_cols = [
            'composite_ID', 'theta_common', 'theta_congruent', 'theta_incongruent',
            'se_common', 'se_congruent', 'se_incongruent'
        ]
        validate_no_nan(df_theta, theta_critical_cols, "Theta scores")

        # Validate theta value ranges
        theta_ranges = {
            'theta_common': VALUE_RANGES['theta_common'],
            'theta_congruent': VALUE_RANGES['theta_congruent'],
            'theta_incongruent': VALUE_RANGES['theta_incongruent'],
            'se_common': VALUE_RANGES['se_common'],
            'se_congruent': VALUE_RANGES['se_congruent'],
            'se_incongruent': VALUE_RANGES['se_incongruent']
        }
        validate_value_ranges(df_theta, theta_ranges, "Theta scores")

        log(f"Theta score sample:")
        log(f"  theta_common: [{df_theta['theta_common'].min():.2f}, {df_theta['theta_common'].max():.2f}]")
        log(f"  theta_congruent: [{df_theta['theta_congruent'].min():.2f}, {df_theta['theta_congruent'].max():.2f}]")
        log(f"  theta_incongruent: [{df_theta['theta_incongruent'].min():.2f}, {df_theta['theta_incongruent'].max():.2f}]")
        log("")
        # Load TSVR Mapping from RQ 5.4.1
        log("[STEP 3] Load TSVR Mapping from RQ 5.4.1")
        log("-" * 70)

        df_tsvr = pd.read_csv(TSVR_MAPPING_PATH)
        log(f"TSVR mapping: {len(df_tsvr)} rows, {len(df_tsvr.columns)} columns")

        # Validate TSVR structure
        expected_tsvr_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        validate_columns(df_tsvr, expected_tsvr_cols, "TSVR mapping")
        validate_row_count(df_tsvr, EXPECTED_TSVR_ROWS, "TSVR mapping")

        # Validate no NaN in TSVR columns
        tsvr_critical_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        validate_no_nan(df_tsvr, tsvr_critical_cols, "TSVR mapping")

        # Validate TSVR value ranges
        tsvr_ranges = {'TSVR_hours': VALUE_RANGES['TSVR_hours']}
        validate_value_ranges(df_tsvr, tsvr_ranges, "TSVR mapping")

        log(f"TSVR_hours range: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")
        log("")
        # Validate Theta <-> TSVR Composite ID Match
        log("[STEP 4] Validate Theta <-> TSVR Composite ID Match")
        log("-" * 70)

        theta_composite_ids = set(df_theta['composite_ID'])
        tsvr_composite_ids = set(df_tsvr['composite_ID'])

        missing_in_tsvr = theta_composite_ids - tsvr_composite_ids
        missing_in_theta = tsvr_composite_ids - theta_composite_ids

        if missing_in_tsvr:
            raise ValueError(
                f"{len(missing_in_tsvr)} composite_IDs in theta but not in TSVR: "
                f"{list(missing_in_tsvr)[:10]}..."
            )

        if missing_in_theta:
            log(f"{len(missing_in_theta)} composite_IDs in TSVR but not in theta "
                f"(acceptable if theta is filtered): {list(missing_in_theta)[:10]}...")
        else:
            log("All composite_IDs match between theta and TSVR")

        log("")
        # Load Age Data from Master
        log("[STEP 5] Load Age Data from Master")
        log("-" * 70)

        # Load master data (only needed columns)
        df_master = pd.read_csv(MASTER_DATA_PATH, usecols=['UID', 'age'])
        log(f"Master data: {len(df_master)} rows, {len(df_master.columns)} columns")

        # Rename 'age' to 'Age' for consistency
        df_master = df_master.rename(columns={'age': 'Age'})
        log("Renamed 'age' column to 'Age' for consistency")

        # Extract unique UIDs (master has 400 rows = 100 UIDs x 4 tests)
        df_age = df_master[['UID', 'Age']].drop_duplicates(subset=['UID']).reset_index(drop=True)
        log(f"Extracted {len(df_age)} unique UIDs from master data")

        # Validate Age data structure
        validate_columns(df_age, ['UID', 'Age'], "Age data")

        if len(df_age) < EXPECTED_MIN_UNIQUE_UIDS:
            raise ValueError(
                f"Age data has only {len(df_age)} unique UIDs, "
                f"expected at least {EXPECTED_MIN_UNIQUE_UIDS}"
            )
        log(f"Age data has {len(df_age)} unique UIDs (>= {EXPECTED_MIN_UNIQUE_UIDS})")

        # Validate no NaN in Age
        validate_no_nan(df_age, ['UID', 'Age'], "Age data")

        # Validate Age value ranges
        age_ranges = {'Age': VALUE_RANGES['Age']}
        validate_value_ranges(df_age, age_ranges, "Age data")

        log(f"Age range: [{df_age['Age'].min():.1f}, {df_age['Age'].max():.1f}]")
        log(f"Age mean: {df_age['Age'].mean():.1f} (SD: {df_age['Age'].std():.1f})")
        log("")
        # Validate All Theta UIDs Have Age Data
        log("[STEP 6] Validate All Theta UIDs Have Age Data")
        log("-" * 70)

        # Parse UIDs from composite_IDs
        theta_uids = parse_uids_from_composite_id(df_theta['composite_ID'])
        unique_theta_uids = set(theta_uids.unique())
        log(f"Parsed {len(unique_theta_uids)} unique UIDs from theta composite_IDs")

        # Check all theta UIDs have Age data
        age_uids = set(df_age['UID'])
        missing_age = unique_theta_uids - age_uids

        if missing_age:
            raise ValueError(
                f"{len(missing_age)} UIDs from theta scores missing Age data: "
                f"{list(missing_age)[:10]}..."
            )

        log("All UIDs from theta scores have matching Age data")
        log("")
        # Save Validated Copies
        log("[STEP 7] Save Validated Copies")
        log("-" * 70)

        # Save theta scores
        df_theta.to_csv(OUTPUT_THETA_PATH, index=False, encoding='utf-8')
        log(f"Theta scores: {OUTPUT_THETA_PATH}")
        log(f"  {len(df_theta)} rows, {len(df_theta.columns)} columns")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_PATH, index=False, encoding='utf-8')
        log(f"TSVR mapping: {OUTPUT_TSVR_PATH}")
        log(f"  {len(df_tsvr)} rows, {len(df_tsvr.columns)} columns")

        # Save Age data
        df_age.to_csv(OUTPUT_AGE_PATH, index=False, encoding='utf-8')
        log(f"Age data: {OUTPUT_AGE_PATH}")
        log(f"  {len(df_age)} rows, {len(df_age.columns)} columns")
        log("")
        # Final Validation Summary
        log("[STEP 8] Final Validation Summary")
        log("-" * 70)
        log("All dependency files loaded and validated successfully")
        log("")
        log("Summary:")
        log(f"  Theta scores: {len(df_theta)} rows (400 expected)")
        log(f"  TSVR mapping: {len(df_tsvr)} rows (400 expected)")
        log(f"  Age data: {len(df_age)} rows (100 unique UIDs)")
        log(f"  Composite IDs: {len(theta_composite_ids)} in theta, {len(tsvr_composite_ids)} in TSVR")
        log(f"  UIDs in theta: {len(unique_theta_uids)}")
        log(f"  UIDs with Age: {len(age_uids)}")
        log("")
        log("Value Ranges:")
        log(f"  theta_common: [{df_theta['theta_common'].min():.2f}, {df_theta['theta_common'].max():.2f}]")
        log(f"  theta_congruent: [{df_theta['theta_congruent'].min():.2f}, {df_theta['theta_congruent'].max():.2f}]")
        log(f"  theta_incongruent: [{df_theta['theta_incongruent'].min():.2f}, {df_theta['theta_incongruent'].max():.2f}]")
        log(f"  TSVR_hours: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")
        log(f"  Age: [{df_age['Age'].min():.1f}, {df_age['Age'].max():.1f}]")
        log("")

        log("Step 00 complete - all dependencies validated and saved")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
