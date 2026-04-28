#!/usr/bin/env python3
"""Load Dependency Data from RQ 5.5.1: Load IRT theta scores by location type from RQ 5.5.1, TSVR mapping, and Age"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_load_dependency_data.log"

# Dependency paths (RQ 5.5.1)
RQ_551_DIR = PROJECT_ROOT / "results" / "ch5" / "5.5.1"
RQ_551_STATUS = RQ_551_DIR / "status.yaml"
RQ_551_THETA = RQ_551_DIR / "data" / "step03_theta_scores.csv"
RQ_551_TSVR = RQ_551_DIR / "data" / "step00_tsvr_mapping.csv"

# Project-level data
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Load Dependency Data from RQ 5.5.1")
        # Check RQ 5.5.1 Completion Status

        log("Verifying RQ 5.5.1 completion status...")

        if not RQ_551_STATUS.exists():
            raise FileNotFoundError(f"RQ 5.5.1 status file not found: {RQ_551_STATUS}")

        with open(RQ_551_STATUS, 'r', encoding='utf-8') as f:
            status_data = yaml.safe_load(f)

        results analysis_status = status_data.get('results analysis', {}).get('status', 'unknown')
        log(f"RQ 5.5.1 status: {results analysis_status}")

        if results analysis_status != 'success':
            raise ValueError(
                f"RQ 5.5.1 not complete (status: {results analysis_status}). "
                f"Cannot proceed with RQ 5.5.3 until dependency is satisfied."
            )

        log("RQ 5.5.1 completed successfully")
        # Load RQ 5.5.1 Theta Scores by Location Type

        log("Loading theta scores from RQ 5.5.1...")

        if not RQ_551_THETA.exists():
            raise FileNotFoundError(f"Theta scores file not found: {RQ_551_THETA}")

        theta_from_rq551 = pd.read_csv(RQ_551_THETA, encoding='utf-8')
        log(f"{RQ_551_THETA.name} ({len(theta_from_rq551)} rows, {len(theta_from_rq551.columns)} cols)")

        # Validate columns present (order doesn't matter for pandas)
        required_theta_cols = ['composite_ID', 'theta_source', 'se_source', 'theta_destination', 'se_destination']
        missing_theta_cols = set(required_theta_cols) - set(theta_from_rq551.columns)
        if missing_theta_cols:
            raise ValueError(f"Theta scores missing required columns: {missing_theta_cols}")

        # Validate row count
        if len(theta_from_rq551) != 400:
            raise ValueError(f"Expected 400 rows in theta scores, got {len(theta_from_rq551)}")

        # Validate no NaN values
        theta_cols_to_check = ['theta_source', 'theta_destination', 'se_source', 'se_destination']
        for col in theta_cols_to_check:
            n_missing = theta_from_rq551[col].isna().sum()
            if n_missing > 0:
                raise ValueError(f"Theta scores has {n_missing} NaN values in {col} column")

        # Validate theta ranges: [-4, 4]
        for col in ['theta_source', 'theta_destination']:
            theta_min = theta_from_rq551[col].min()
            theta_max = theta_from_rq551[col].max()
            if theta_min < -4.0 or theta_max > 4.0:
                raise ValueError(
                    f"{col} out of range [-4, 4]: min={theta_min:.3f}, max={theta_max:.3f}"
                )
        log(f"Theta ranges: source [{theta_from_rq551['theta_source'].min():.3f}, {theta_from_rq551['theta_source'].max():.3f}], "
            f"destination [{theta_from_rq551['theta_destination'].min():.3f}, {theta_from_rq551['theta_destination'].max():.3f}]")

        # Validate SE ranges: [0.1, 1.5]
        for col in ['se_source', 'se_destination']:
            se_min = theta_from_rq551[col].min()
            se_max = theta_from_rq551[col].max()
            if se_min < 0.1 or se_max > 1.5:
                raise ValueError(
                    f"{col} out of range [0.1, 1.5]: min={se_min:.3f}, max={se_max:.3f}"
                )
        log(f"SE ranges: source [{theta_from_rq551['se_source'].min():.3f}, {theta_from_rq551['se_source'].max():.3f}], "
            f"destination [{theta_from_rq551['se_destination'].min():.3f}, {theta_from_rq551['se_destination'].max():.3f}]")
        # Load RQ 5.5.1 TSVR Mapping

        log("Loading TSVR mapping from RQ 5.5.1...")

        if not RQ_551_TSVR.exists():
            raise FileNotFoundError(f"TSVR mapping file not found: {RQ_551_TSVR}")

        tsvr_from_rq551 = pd.read_csv(RQ_551_TSVR, encoding='utf-8')
        log(f"{RQ_551_TSVR.name} ({len(tsvr_from_rq551)} rows, {len(tsvr_from_rq551.columns)} cols)")

        # Validate columns present
        required_tsvr_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        missing_tsvr_cols = set(required_tsvr_cols) - set(tsvr_from_rq551.columns)
        if missing_tsvr_cols:
            raise ValueError(f"TSVR mapping missing required columns: {missing_tsvr_cols}")

        # Validate row count
        if len(tsvr_from_rq551) != 400:
            raise ValueError(f"Expected 400 rows in TSVR mapping, got {len(tsvr_from_rq551)}")

        # Validate no NaN values in TSVR_hours
        n_missing_tsvr = tsvr_from_rq551['TSVR_hours'].isna().sum()
        if n_missing_tsvr > 0:
            raise ValueError(f"TSVR mapping has {n_missing_tsvr} NaN values in TSVR_hours column")

        # Validate TSVR range: [0, 360] (extended range to handle late tests)
        tsvr_min = tsvr_from_rq551['TSVR_hours'].min()
        tsvr_max = tsvr_from_rq551['TSVR_hours'].max()
        if tsvr_min < 0.0 or tsvr_max > 360.0:
            raise ValueError(
                f"TSVR_hours out of range [0, 360]: min={tsvr_min:.2f}, max={tsvr_max:.2f}"
            )
        log(f"TSVR range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")
        # Load Age Variable from dfData.csv

        log("Loading Age variable from dfData.csv...")

        if not DFDATA_PATH.exists():
            raise FileNotFoundError(f"dfData.csv not found: {DFDATA_PATH}")

        # Load only UID and age columns (age is lowercase in dfData.csv)
        age_from_dfdata = pd.read_csv(DFDATA_PATH, usecols=['UID', 'age'], encoding='utf-8')
        log(f"{DFDATA_PATH.name} (extracted 2 columns from 214 total)")

        # Get unique UID-Age pairs (should be 100 participants)
        age_from_dfdata = age_from_dfdata.drop_duplicates(subset='UID')
        log(f"{len(age_from_dfdata)} unique participants")

        # Rename lowercase 'age' to 'Age' for consistency with spec
        age_from_dfdata = age_from_dfdata.rename(columns={'age': 'Age'})

        # Validate row count
        if len(age_from_dfdata) != 100:
            raise ValueError(f"Expected 100 unique participants, got {len(age_from_dfdata)}")

        # Validate no NaN values in Age
        n_missing_age = age_from_dfdata['Age'].isna().sum()
        if n_missing_age > 0:
            raise ValueError(f"Age data has {n_missing_age} NaN values")

        # Validate Age range: [20, 70]
        age_min = age_from_dfdata['Age'].min()
        age_max = age_from_dfdata['Age'].max()
        if age_min < 20 or age_max > 70:
            raise ValueError(
                f"Age out of range [20, 70]: min={age_min}, max={age_max}"
            )
        log(f"Age range: [{age_min}, {age_max}] years")
        log(f"Age mean={age_from_dfdata['Age'].mean():.1f}, SD={age_from_dfdata['Age'].std():.1f}")
        # Save All Outputs
        # These outputs will be used by: Step 1 (prepare LMM input)

        log("Saving dependency data copies...")

        # Output 1: Theta scores from RQ 5.5.1
        output_theta_path = RQ_DIR / "data" / "step00_theta_from_rq551.csv"
        theta_from_rq551.to_csv(output_theta_path, index=False, encoding='utf-8')
        log(f"{output_theta_path.name} ({len(theta_from_rq551)} rows, {len(theta_from_rq551.columns)} cols)")

        # Output 2: TSVR mapping from RQ 5.5.1
        output_tsvr_path = RQ_DIR / "data" / "step00_tsvr_from_rq551.csv"
        tsvr_from_rq551.to_csv(output_tsvr_path, index=False, encoding='utf-8')
        log(f"{output_tsvr_path.name} ({len(tsvr_from_rq551)} rows, {len(tsvr_from_rq551.columns)} cols)")

        # Output 3: Age variable from dfData.csv
        output_age_path = RQ_DIR / "data" / "step00_age_from_dfdata.csv"
        age_from_dfdata.to_csv(output_age_path, index=False, encoding='utf-8')
        log(f"{output_age_path.name} ({len(age_from_dfdata)} rows, {len(age_from_dfdata.columns)} cols)")
        # Final Validation Summary
        # Validates: All data loaded correctly with expected structure and ranges

        log("Final validation summary:")
        validation_checks = [
            ("RQ 5.5.1 completion status", results analysis_status == 'success'),
            ("Theta scores row count", len(theta_from_rq551) == 400),
            ("TSVR mapping row count", len(tsvr_from_rq551) == 400),
            ("Age data row count", len(age_from_dfdata) == 100),
            ("Theta scores columns", set(required_theta_cols).issubset(set(theta_from_rq551.columns))),
            ("TSVR mapping columns", set(required_tsvr_cols).issubset(set(tsvr_from_rq551.columns))),
            ("Age data columns", set(['UID', 'Age']).issubset(set(age_from_dfdata.columns))),
            ("No NaN in theta_source", theta_from_rq551['theta_source'].isna().sum() == 0),
            ("No NaN in theta_destination", theta_from_rq551['theta_destination'].isna().sum() == 0),
            ("No NaN in TSVR_hours", tsvr_from_rq551['TSVR_hours'].isna().sum() == 0),
            ("No NaN in Age", age_from_dfdata['Age'].isna().sum() == 0),
            ("Theta source range [-4, 4]",
             theta_from_rq551['theta_source'].min() >= -4.0 and theta_from_rq551['theta_source'].max() <= 4.0),
            ("Theta destination range [-4, 4]",
             theta_from_rq551['theta_destination'].min() >= -4.0 and theta_from_rq551['theta_destination'].max() <= 4.0),
            ("SE source range [0.1, 1.5]",
             theta_from_rq551['se_source'].min() >= 0.1 and theta_from_rq551['se_source'].max() <= 1.5),
            ("SE destination range [0.1, 1.5]",
             theta_from_rq551['se_destination'].min() >= 0.1 and theta_from_rq551['se_destination'].max() <= 1.5),
            ("TSVR range [0, 360]",
             tsvr_from_rq551['TSVR_hours'].min() >= 0.0 and tsvr_from_rq551['TSVR_hours'].max() <= 360.0),
            ("Age range [20, 70]",
             age_from_dfdata['Age'].min() >= 20 and age_from_dfdata['Age'].max() <= 70),
        ]

        all_passed = True
        for check_name, check_result in validation_checks:
            status_str = "" if check_result else ""
            log(f"{status_str} {check_name}")
            if not check_result:
                all_passed = False

        if not all_passed:
            raise ValueError("Validation failed - see log for details")

        log(f"All {len(validation_checks)} checks passed")

        log("Step 0 complete - All dependency data loaded and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
