#!/usr/bin/env python3
"""validate_dependencies: Verify required Ch5 5.1.1 outputs and dfnonvr.csv accessibility before proceeding."""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Validation

if __name__ == "__main__":
    try:
        log("Step 00: Validate Dependencies")

        validation_results = []
        all_passed = True
        # CHECK 1: Ch5 5.1.1 Status
        log("[CHECK 1] Validating Ch5 5.1.1 completion status...")

        ch5_status_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "status.yaml"

        if not ch5_status_path.exists():
            validation_results.append("Ch5 5.1.1 status.yaml not found")
            all_passed = False
            log(f"{ch5_status_path} does not exist")
        else:
            with open(ch5_status_path, 'r', encoding='utf-8') as f:
                status_data = yaml.safe_load(f)

            # Handle both dict and string formats (from gcode_lessons bug #5)
            results analysis_status = status_data.get('results analysis', {})
            if isinstance(results analysis_status, dict):
                actual_status = results analysis_status.get('status')
            elif isinstance(results analysis_status, str):
                actual_status = results analysis_status
            else:
                actual_status = 'unknown'

            if actual_status == 'success':
                validation_results.append("Ch5 5.1.1 status: success")
                log("Ch5 5.1.1 completed successfully")
            else:
                validation_results.append(f"Ch5 5.1.1 status: {actual_status} (expected 'success')")
                all_passed = False
                log(f"Ch5 5.1.1 status is '{actual_status}', expected 'success'")
        # CHECK 2: Ch5 Theta Scores File
        log("[CHECK 2] Validating Ch5 theta scores file...")

        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        required_theta_cols = ['UID', 'test', 'Theta_All']

        if not theta_path.exists():
            validation_results.append("Ch5 theta scores file not found")
            all_passed = False
            log(f"{theta_path} does not exist")
        else:
            theta_df = pd.read_csv(theta_path, nrows=0)
            actual_cols = theta_df.columns.tolist()

            missing_cols = [col for col in required_theta_cols if col not in actual_cols]

            if missing_cols:
                validation_results.append(f"Ch5 theta scores missing columns: {missing_cols}")
                all_passed = False
                log(f"Missing columns: {missing_cols}")
                log(f"Found columns: {actual_cols}")
            else:
                # Check file size
                theta_full = pd.read_csv(theta_path)
                n_rows = len(theta_full)
                validation_results.append(f"Ch5 theta scores file valid ({n_rows} rows, {len(actual_cols)} columns)")
                log(f"Theta scores file has all required columns ({n_rows} rows)")
        # CHECK 3: dfnonvr.csv RAVLT Columns
        log("[CHECK 3] Validating dfnonvr.csv RAVLT columns...")

        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        required_ravlt_cols = [
            'UID',
            'ravlt-trial-1-score',
            'ravlt-trial-2-score',
            'ravlt-trial-3-score',
            'ravlt-trial-4-score',
            'ravlt-trial-5-score',
            'ravlt-delayed-recall-score',
            'ravlt-recognition-hits-'
        ]

        if not dfnonvr_path.exists():
            validation_results.append("dfnonvr.csv not found")
            all_passed = False
            log(f"{dfnonvr_path} does not exist")
        else:
            dfnonvr = pd.read_csv(dfnonvr_path, nrows=0)
            actual_cols = dfnonvr.columns.tolist()

            missing_cols = [col for col in required_ravlt_cols if col not in actual_cols]

            if missing_cols:
                validation_results.append(f"dfnonvr.csv missing columns: {missing_cols}")
                all_passed = False
                log(f"Missing RAVLT columns: {missing_cols}")
            else:
                # Check file size
                dfnonvr_full = pd.read_csv(dfnonvr_path)
                n_rows = len(dfnonvr_full)
                validation_results.append(f"dfnonvr.csv RAVLT columns valid ({n_rows} participants)")
                log(f"dfnonvr.csv has all required RAVLT columns ({n_rows} participants)")
        # WRITE VALIDATION REPORT
        log("Writing validation report...")

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 7.7.3 DEPENDENCY VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            for result in validation_results:
                f.write(f"{result}\n")

            f.write("\n" + "=" * 80 + "\n")
            if all_passed:
                f.write("OVERALL STATUS: All dependencies validated\n")
            else:
                f.write("OVERALL STATUS: Some dependencies missing or invalid\n")
            f.write("=" * 80 + "\n")

        log(f"{OUTPUT_FILE}")
        # FINAL STATUS
        if all_passed:
            log("Step 00 complete - all dependencies validated")
            sys.exit(0)
        else:
            log("Step 00 failed - dependency validation errors detected")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
