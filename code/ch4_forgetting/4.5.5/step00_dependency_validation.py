#!/usr/bin/env python3
"""Dependency Validation: Verify RQ 5.5.1 completed successfully and load required outputs for downstream"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_dependency_validation.log"

# Dependency paths (RQ 5.5.1 outputs)
RQ_551_DIR = PROJECT_ROOT / "results" / "ch5" / "5.5.1"
STATUS_FILE = RQ_551_DIR / "status.yaml"
PURIFIED_ITEMS_FILE = RQ_551_DIR / "data" / "step02_purified_items.csv"
THETA_SCORES_FILE = RQ_551_DIR / "data" / "step03_theta_scores.csv"
TSVR_MAPPING_FILE = RQ_551_DIR / "data" / "step00_tsvr_mapping.csv"
RAW_DATA_FILE = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Output paths
VALIDATION_REPORT = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Validation thresholds
THETA_MIN = -4.0
THETA_MAX = 4.0
RETENTION_RATE_MIN = 0.55
RETENTION_RATE_MAX = 0.85
EXPECTED_ROWS_THETA = 400
EXPECTED_ROWS_PURIFIED_MIN = 32
EXPECTED_ROWS_PURIFIED_MAX = 36
MIN_ITEMS_PER_LOCATION = 10


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Dependency Validation")

        # Initialize validation report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RQ 5.5.5 DEPENDENCY VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Generated: {pd.Timestamp.now()}")
        report_lines.append(f"Purpose: Verify RQ 5.5.1 completed successfully before RQ 5.5.5 execution")
        report_lines.append("")

        validation_passed = True
        # Check RQ 5.5.1 Completion Status

        log("[VALIDATION 1/6] Checking RQ 5.5.1 completion status...")
        report_lines.append("VALIDATION 1: RQ 5.5.1 Status")
        report_lines.append("-" * 80)

        if not STATUS_FILE.exists():
            validation_passed = False
            error_msg = f"Status file not found: {STATUS_FILE}"
            log(error_msg)
            report_lines.append(error_msg)
            report_lines.append("RESULT: FAIL - RQ 5.5.1 has not been executed")
        else:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status_data = yaml.safe_load(f)

            results analysis_status = status_data.get('results analysis', {}).get('status', None)

            if results analysis_status == 'success':
                log(f"RQ 5.5.1 status = 'success'")
                report_lines.append(f"RQ 5.5.1 status = 'success'")
                report_lines.append(f"Status file: {STATUS_FILE}")
            else:
                validation_passed = False
                error_msg = f"RQ 5.5.1 status = '{results analysis_status}' (expected 'success')"
                log(error_msg)
                report_lines.append(error_msg)

        report_lines.append("")
        # Validate Purified Items File
        # Columns: item_tag, factor, a, b, retention_reason

        log("[VALIDATION 2/6] Validating purified items file...")
        report_lines.append("VALIDATION 2: Purified Items")
        report_lines.append("-" * 80)

        if not PURIFIED_ITEMS_FILE.exists():
            validation_passed = False
            error_msg = f"Purified items file not found: {PURIFIED_ITEMS_FILE}"
            log(error_msg)
            report_lines.append(error_msg)
        else:
            df_purified = pd.read_csv(PURIFIED_ITEMS_FILE, encoding='utf-8')
            n_items = len(df_purified)

            report_lines.append(f"File: {PURIFIED_ITEMS_FILE}")
            report_lines.append(f"Rows: {n_items}")
            report_lines.append(f"Columns: {list(df_purified.columns)}")

            # Check expected columns
            expected_cols = ['item_tag', 'factor', 'a', 'b', 'retention_reason']
            missing_cols = [col for col in expected_cols if col not in df_purified.columns]

            if missing_cols:
                validation_passed = False
                error_msg = f"Missing columns: {missing_cols}"
                log(error_msg)
                report_lines.append(error_msg)
            else:
                log(f"All expected columns present")
                report_lines.append(f"All expected columns present")

            # Check row count
            if EXPECTED_ROWS_PURIFIED_MIN <= n_items <= EXPECTED_ROWS_PURIFIED_MAX:
                log(f"Row count in expected range [{EXPECTED_ROWS_PURIFIED_MIN}, {EXPECTED_ROWS_PURIFIED_MAX}]")
                report_lines.append(f"Row count in expected range [{EXPECTED_ROWS_PURIFIED_MIN}, {EXPECTED_ROWS_PURIFIED_MAX}]")
            else:
                validation_passed = False
                error_msg = f"Row count {n_items} outside expected range [{EXPECTED_ROWS_PURIFIED_MIN}, {EXPECTED_ROWS_PURIFIED_MAX}]"
                log(error_msg)
                report_lines.append(error_msg)

            # Check retention per location type (source vs destination)
            if 'factor' in df_purified.columns:
                retention_counts = df_purified['factor'].value_counts()
                report_lines.append("")
                report_lines.append("Retention by location type:")

                for location_type, count in retention_counts.items():
                    report_lines.append(f"  {location_type}: {count} items")

                    # Check minimum items per location
                    if count < MIN_ITEMS_PER_LOCATION:
                        validation_passed = False
                        error_msg = f"{location_type}: only {count} items (minimum {MIN_ITEMS_PER_LOCATION} required)"
                        log(error_msg)
                        report_lines.append(f"  {error_msg}")
                    else:
                        log(f"{location_type}: {count} items (>= {MIN_ITEMS_PER_LOCATION})")
                        report_lines.append(f"  >= {MIN_ITEMS_PER_LOCATION} items")

        report_lines.append("")
        # Validate Theta Scores File
        # Columns: composite_ID, theta_source, theta_destination, se_source, se_destination

        log("[VALIDATION 3/6] Validating theta scores file...")
        report_lines.append("VALIDATION 3: Theta Scores")
        report_lines.append("-" * 80)

        if not THETA_SCORES_FILE.exists():
            validation_passed = False
            error_msg = f"Theta scores file not found: {THETA_SCORES_FILE}"
            log(error_msg)
            report_lines.append(error_msg)
        else:
            df_theta = pd.read_csv(THETA_SCORES_FILE, encoding='utf-8')
            n_rows = len(df_theta)

            report_lines.append(f"File: {THETA_SCORES_FILE}")
            report_lines.append(f"Rows: {n_rows}")
            report_lines.append(f"Columns: {list(df_theta.columns)}")

            # Check expected columns
            expected_cols = ['composite_ID', 'theta_source', 'theta_destination', 'se_source', 'se_destination']
            missing_cols = [col for col in expected_cols if col not in df_theta.columns]

            if missing_cols:
                validation_passed = False
                error_msg = f"Missing columns: {missing_cols}"
                log(error_msg)
                report_lines.append(error_msg)
            else:
                log(f"All expected columns present")
                report_lines.append(f"All expected columns present")

            # Check row count
            if n_rows == EXPECTED_ROWS_THETA:
                log(f"Row count = {EXPECTED_ROWS_THETA}")
                report_lines.append(f"Row count = {EXPECTED_ROWS_THETA}")
            else:
                validation_passed = False
                error_msg = f"Row count {n_rows} != {EXPECTED_ROWS_THETA}"
                log(error_msg)
                report_lines.append(error_msg)

            # Check theta ranges
            theta_cols = ['theta_source', 'theta_destination']
            for col in theta_cols:
                if col in df_theta.columns:
                    theta_min = df_theta[col].min()
                    theta_max = df_theta[col].max()
                    theta_nan = df_theta[col].isna().sum()

                    report_lines.append(f"")
                    report_lines.append(f"{col}:")
                    report_lines.append(f"  Range: [{theta_min:.2f}, {theta_max:.2f}]")
                    report_lines.append(f"  NaN count: {theta_nan}")

                    if theta_nan > 0:
                        validation_passed = False
                        error_msg = f"{col}: {theta_nan} NaN values found"
                        log(error_msg)
                        report_lines.append(f"  {error_msg}")
                    else:
                        log(f"{col}: No NaN values")
                        report_lines.append(f"  No NaN values")

                    if THETA_MIN <= theta_min and theta_max <= THETA_MAX:
                        log(f"{col}: Range within [{THETA_MIN}, {THETA_MAX}]")
                        report_lines.append(f"  Range within [{THETA_MIN}, {THETA_MAX}]")
                    else:
                        validation_passed = False
                        error_msg = f"{col}: Range outside [{THETA_MIN}, {THETA_MAX}]"
                        log(error_msg)
                        report_lines.append(f"  {error_msg}")

        report_lines.append("")
        # Validate TSVR Mapping File
        # Columns: composite_ID, UID, test, TSVR_hours

        log("[VALIDATION 4/6] Validating TSVR mapping file...")
        report_lines.append("VALIDATION 4: TSVR Mapping")
        report_lines.append("-" * 80)

        if not TSVR_MAPPING_FILE.exists():
            validation_passed = False
            error_msg = f"TSVR mapping file not found: {TSVR_MAPPING_FILE}"
            log(error_msg)
            report_lines.append(error_msg)
        else:
            df_tsvr = pd.read_csv(TSVR_MAPPING_FILE, encoding='utf-8')
            n_rows = len(df_tsvr)

            report_lines.append(f"File: {TSVR_MAPPING_FILE}")
            report_lines.append(f"Rows: {n_rows}")
            report_lines.append(f"Columns: {list(df_tsvr.columns)}")

            # Check expected columns
            expected_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours']
            missing_cols = [col for col in expected_cols if col not in df_tsvr.columns]

            if missing_cols:
                validation_passed = False
                error_msg = f"Missing columns: {missing_cols}"
                log(error_msg)
                report_lines.append(error_msg)
            else:
                log(f"All expected columns present")
                report_lines.append(f"All expected columns present")

            # Check row count
            if n_rows == EXPECTED_ROWS_THETA:
                log(f"Row count = {EXPECTED_ROWS_THETA}")
                report_lines.append(f"Row count = {EXPECTED_ROWS_THETA}")
            else:
                validation_passed = False
                error_msg = f"Row count {n_rows} != {EXPECTED_ROWS_THETA}"
                log(error_msg)
                report_lines.append(error_msg)

            # Check UID/test alignment with theta scores
            if THETA_SCORES_FILE.exists() and 'composite_ID' in df_theta.columns and 'composite_ID' in df_tsvr.columns:
                theta_ids = set(df_theta['composite_ID'])
                tsvr_ids = set(df_tsvr['composite_ID'])

                if theta_ids == tsvr_ids:
                    log(f"composite_ID sets match between theta and TSVR files")
                    report_lines.append(f"composite_ID sets match between theta and TSVR files")
                else:
                    validation_passed = False
                    missing_in_tsvr = theta_ids - tsvr_ids
                    missing_in_theta = tsvr_ids - theta_ids
                    error_msg = f"composite_ID mismatch (theta-only: {len(missing_in_tsvr)}, tsvr-only: {len(missing_in_theta)})"
                    log(error_msg)
                    report_lines.append(error_msg)

        report_lines.append("")
        # Validate Raw Data File

        log("[VALIDATION 5/6] Validating raw data file...")
        report_lines.append("VALIDATION 5: Raw Data File")
        report_lines.append("-" * 80)

        if not RAW_DATA_FILE.exists():
            validation_passed = False
            error_msg = f"Raw data file not found: {RAW_DATA_FILE}"
            log(error_msg)
            report_lines.append(error_msg)
        else:
            # Read only header to check columns (faster than loading full file)
            df_raw_header = pd.read_csv(RAW_DATA_FILE, nrows=0, encoding='utf-8')
            columns = list(df_raw_header.columns)

            # Count -U- and -D- item columns
            source_items = [col for col in columns if '-U-' in col]
            destination_items = [col for col in columns if '-D-' in col]

            report_lines.append(f"File: {RAW_DATA_FILE}")
            report_lines.append(f"Total columns: {len(columns)}")
            report_lines.append(f"Source items (-U-): {len(source_items)}")
            report_lines.append(f"Destination items (-D-): {len(destination_items)}")

            # Check minimum item counts (18 source + 18 destination = 36 total before purification)
            min_items = 18
            if len(source_items) >= min_items:
                log(f"Source items: {len(source_items)} >= {min_items}")
                report_lines.append(f"Source items: {len(source_items)} >= {min_items}")
            else:
                validation_passed = False
                error_msg = f"Source items: {len(source_items)} < {min_items}"
                log(error_msg)
                report_lines.append(error_msg)

            if len(destination_items) >= min_items:
                log(f"Destination items: {len(destination_items)} >= {min_items}")
                report_lines.append(f"Destination items: {len(destination_items)} >= {min_items}")
            else:
                validation_passed = False
                error_msg = f"Destination items: {len(destination_items)} < {min_items}"
                log(error_msg)
                report_lines.append(error_msg)

            # Check required base columns
            required_base_cols = ['UID', 'TEST']
            missing_base = [col for col in required_base_cols if col not in columns]

            if missing_base:
                validation_passed = False
                error_msg = f"Missing required base columns: {missing_base}"
                log(error_msg)
                report_lines.append(error_msg)
            else:
                log(f"All required base columns present: {required_base_cols}")
                report_lines.append(f"All required base columns present: {required_base_cols}")

        report_lines.append("")
        # Write Validation Report

        log("[VALIDATION 6/6] Writing validation report...")
        report_lines.append("=" * 80)
        report_lines.append("FINAL RESULT")
        report_lines.append("=" * 80)

        if validation_passed:
            result_msg = "All validations PASSED - RQ 5.5.5 dependencies satisfied"
            log(result_msg)
            report_lines.append(result_msg)
            report_lines.append("")
            report_lines.append("RQ 5.5.5 may proceed with:")
            report_lines.append("  - Purified items from RQ 5.5.1 Step 2")
            report_lines.append("  - Theta scores from RQ 5.5.1 Step 3")
            report_lines.append("  - TSVR mapping from RQ 5.5.1 Step 0")
            report_lines.append("  - Raw binary responses from dfData.csv")
        else:
            result_msg = "One or more validations FAILED - cannot proceed with RQ 5.5.5"
            log(result_msg)
            report_lines.append(result_msg)
            report_lines.append("")
            report_lines.append("Action required:")
            report_lines.append("  1. Review failed validations above")
            report_lines.append("  2. Fix issues in RQ 5.5.1 if applicable")
            report_lines.append("  3. Re-run RQ 5.5.1 if necessary")
            report_lines.append("  4. Re-run this validation step")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Write report to file
        with open(VALIDATION_REPORT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        log(f"Validation report: {VALIDATION_REPORT}")
        # Final Result

        if validation_passed:
            log("Step 00: Dependency Validation complete")
            sys.exit(0)
        else:
            log("[EXPECTATIONS ERROR] RQ 5.5.1 dependencies not satisfied")
            log("Action: QUIT - RQ 5.5.1 must complete successfully before RQ 5.5.5 execution")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
