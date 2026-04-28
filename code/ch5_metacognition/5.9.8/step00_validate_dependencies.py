#!/usr/bin/env python3
"""Validate Cross-RQ Dependencies: Verify that Ch5 5.4.1 (accuracy by schema) and Ch6 6.5.1 (confidence by schema)"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.8
PROJECT_ROOT = RQ_DIR.parents[2]  # /home/etai/projects/REMEMVR

LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Dependency paths
CH5_STATUS = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "status.yaml"
CH6_STATUS = PROJECT_ROOT / "results" / "ch6" / "6.5.1" / "status.yaml"
CH5_DATA = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step03_theta_scores.csv"
CH6_DATA = PROJECT_ROOT / "results" / "ch6" / "6.5.1" / "data" / "step03_theta_confidence.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 00: Validate Cross-RQ Dependencies")

        validation_report = []
        validation_report.append("=" * 80)
        validation_report.append("DEPENDENCY VALIDATION REPORT - RQ 6.9.8")
        validation_report.append("=" * 80)
        validation_report.append("")

        all_passed = True
        # Check File Existence
        log("Verifying file existence...")

        files_to_check = [
            ("Ch5 5.4.1 status.yaml", CH5_STATUS),
            ("Ch6 6.5.1 status.yaml", CH6_STATUS),
            ("Ch5 5.4.1 accuracy data", CH5_DATA),
            ("Ch6 6.5.1 confidence data", CH6_DATA)
        ]

        validation_report.append("FILE EXISTENCE CHECKS:")
        for name, path in files_to_check:
            if path.exists():
                log(f"{name} found")
                validation_report.append(f"  {name}")
                validation_report.append(f"         Path: {path}")
            else:
                log(f"{name} NOT FOUND: {path}")
                validation_report.append(f"  {name} NOT FOUND")
                validation_report.append(f"         Expected: {path}")
                all_passed = False

        if not all_passed:
            validation_report.append("")
            validation_report.append("VALIDATION - FAIL")
            validation_report.append("One or more required files missing. Cannot proceed.")

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("\n".join(validation_report))

            log("Dependency validation failed - see report")
            sys.exit(1)

        validation_report.append("")
        # Check Status YAML Files
        log("Verifying RQ completion status...")

        validation_report.append("STATUS CHECKS:")

        # Check Ch5 5.4.1
        with open(CH5_STATUS, 'r', encoding='utf-8') as f:
            ch5_status = yaml.safe_load(f)

        # Handle both nested (Ch5) and flat (Ch6) status.yaml formats
        ch5_results analysis = ch5_status.get('results analysis')
        if isinstance(ch5_results analysis, dict):
            ch5_status_value = ch5_results analysis.get('status', 'MISSING')
        else:
            ch5_status_value = ch5_results analysis

        if ch5_status_value == 'success':
            log("Ch5 5.4.1 status: success")
            validation_report.append("  Ch5 5.4.1 status: success")
        else:
            log(f"Ch5 5.4.1 status: {ch5_status_value}")
            validation_report.append(f"  Ch5 5.4.1 status: {ch5_status_value}")
            all_passed = False

        # Check Ch6 6.5.1
        with open(CH6_STATUS, 'r', encoding='utf-8') as f:
            ch6_status = yaml.safe_load(f)

        # Handle both nested and flat formats
        ch6_results analysis = ch6_status.get('results analysis')
        if isinstance(ch6_results analysis, dict):
            ch6_status_value = ch6_results analysis.get('status', 'MISSING')
        else:
            ch6_status_value = ch6_results analysis

        if ch6_status_value == 'success':
            log("Ch6 6.5.1 status: success")
            validation_report.append("  Ch6 6.5.1 status: success")
        else:
            log(f"Ch6 6.5.1 status: {ch6_status_value}")
            validation_report.append(f"  Ch6 6.5.1 status: {ch6_status_value}")
            all_passed = False

        if not all_passed:
            validation_report.append("")
            validation_report.append("VALIDATION - FAIL")
            validation_report.append("One or more source RQs not complete. Run Ch5 5.4.1 and Ch6 6.5.1 first.")

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("\n".join(validation_report))

            log("Dependency validation failed - see report")
            sys.exit(1)

        validation_report.append("")
        # Verify Data File Structure
        log("Verifying data file structure...")

        validation_report.append("DATA STRUCTURE CHECKS:")

        # Check Ch5 accuracy file
        df_acc = pd.read_csv(CH5_DATA)
        log(f"Accuracy file: {len(df_acc)} rows, {len(df_acc.columns)} columns")

        acc_checks = []
        acc_checks.append(("Row count", len(df_acc) == 400, f"Expected 400, got {len(df_acc)}"))
        acc_checks.append(("composite_ID column", "composite_ID" in df_acc.columns, "Required ID column"))
        acc_checks.append(("theta_common column", "theta_common" in df_acc.columns, "Required accuracy column"))
        acc_checks.append(("theta_congruent column", "theta_congruent" in df_acc.columns, "Required accuracy column"))
        acc_checks.append(("theta_incongruent column", "theta_incongruent" in df_acc.columns, "Required accuracy column"))

        validation_report.append("")
        validation_report.append(f"  Accuracy file found: {len(df_acc)} rows, {len(df_acc.columns)} columns")
        validation_report.append(f"  Columns: {', '.join(df_acc.columns.tolist())}")

        for check_name, passed, message in acc_checks:
            if passed:
                log(f"Accuracy {check_name}")
                validation_report.append(f"    {check_name}")
            else:
                log(f"Accuracy {check_name}: {message}")
                validation_report.append(f"    {check_name}: {message}")
                all_passed = False

        # Check Ch6 confidence file
        df_conf = pd.read_csv(CH6_DATA)
        log(f"Confidence file: {len(df_conf)} rows, {len(df_conf.columns)} columns")

        conf_checks = []
        conf_checks.append(("Row count", len(df_conf) == 400, f"Expected 400, got {len(df_conf)}"))
        conf_checks.append(("composite_ID column", "composite_ID" in df_conf.columns, "Required ID column"))
        conf_checks.append(("theta_Common column", "theta_Common" in df_conf.columns, "Required confidence column (title case)"))
        conf_checks.append(("theta_Congruent column", "theta_Congruent" in df_conf.columns, "Required confidence column (title case)"))
        conf_checks.append(("theta_Incongruent column", "theta_Incongruent" in df_conf.columns, "Required confidence column (title case)"))

        validation_report.append("")
        validation_report.append(f"  Confidence file found: {len(df_conf)} rows, {len(df_conf.columns)} columns")
        validation_report.append(f"  Columns: {', '.join(df_conf.columns.tolist())}")

        for check_name, passed, message in conf_checks:
            if passed:
                log(f"Confidence {check_name}")
                validation_report.append(f"    {check_name}")
            else:
                log(f"Confidence {check_name}: {message}")
                validation_report.append(f"    {check_name}: {message}")
                all_passed = False

        validation_report.append("")
        # Write Validation Report
        if all_passed:
            validation_report.append("VALIDATION - PASS")
            validation_report.append("")
            validation_report.append("All dependencies satisfied. Ready to proceed with RQ 6.9.8 analysis.")
            log("All dependency checks passed")
        else:
            validation_report.append("VALIDATION - FAIL")
            validation_report.append("")
            validation_report.append("One or more checks failed. Review report above.")
            log("One or more dependency checks failed")

        validation_report.append("")
        validation_report.append("=" * 80)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(validation_report))

        log(f"Validation report: {OUTPUT_FILE}")

        if all_passed:
            log("Step 00 complete")
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
