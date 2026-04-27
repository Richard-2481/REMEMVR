#!/usr/bin/env python3
"""
Step 01: Load Dependency Data from RQ 5.4.1

PURPOSE:
Load theta scores, TSVR mapping from RQ 5.4.1. Verify data structure before
proceeding to congruence-stratified analysis.

EXPECTED INPUTS:
- results/ch5/5.4.1/status.yaml: RQ completion status
- results/ch5/5.4.1/data/step03_theta_scores.csv: Theta scores (400 rows)
- results/ch5/5.4.1/data/step04_lmm_input.csv: LMM input (1200 rows)

EXPECTED OUTPUTS:
- data/step01_dependency_validation_report.txt: Validation report
- data/step01_loaded_lmm_input.csv: Cached LMM input (1200 rows x 6 columns)

VALIDATION CRITERIA:
- RQ 5.4.1 status = 'success'
- All dependency files exist
- LMM input: 1200 rows, required columns present, 3 unique congruence levels
- No NaN in theta or TSVR_hours
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_load_dependency_data.log"

# Dependency paths (RQ 5.4.1)
DEPENDENCY_RQ = PROJECT_ROOT / "results" / "ch5" / "5.4.1"
STATUS_FILE = DEPENDENCY_RQ / "status.yaml"
THETA_FILE = DEPENDENCY_RQ / "data" / "step03_theta_scores.csv"
LMM_INPUT_FILE = DEPENDENCY_RQ / "data" / "step04_lmm_input.csv"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 01: Load Dependency Data from RQ 5.4.1")

        # =====================================================================
        # STEP 1: Check RQ 5.4.1 Completion Status
        # =====================================================================
        log("[CHECK] Verifying RQ 5.4.1 completion status...")

        if not STATUS_FILE.exists():
            raise FileNotFoundError(f"EXPECTATIONS ERROR: RQ 5.4.1 status.yaml not found at {STATUS_FILE}")

        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            status_data = yaml.safe_load(f)

        # Check if rq_results section exists and has success status
        if 'rq_results' not in status_data:
            raise ValueError("EXPECTATIONS ERROR: RQ 5.4.1 status.yaml missing 'rq_results' section")

        rq_results_status = status_data.get('rq_results', {}).get('status', 'unknown')

        if rq_results_status != 'success':
            raise ValueError(f"EXPECTATIONS ERROR: RQ 5.4.1 not complete (status: {rq_results_status}). Cannot proceed with derived analysis.")

        log(f"[PASS] RQ 5.4.1 completion verified (status: {rq_results_status})")

        # =====================================================================
        # STEP 2: Check Dependency Files Exist
        # =====================================================================
        log("[CHECK] Verifying dependency files exist...")

        required_files = {
            'theta_scores': THETA_FILE,
            'lmm_input': LMM_INPUT_FILE
        }

        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")

        if missing_files:
            raise FileNotFoundError(f"EXPECTATIONS ERROR: Missing dependency files:\n" + "\n".join(missing_files))

        log(f"[PASS] All dependency files exist")

        # =====================================================================
        # STEP 3: Load and Validate Theta Scores
        # =====================================================================
        log("[LOAD] Loading theta scores from RQ 5.4.1...")

        df_theta = pd.read_csv(THETA_FILE, encoding='utf-8')
        log(f"[LOADED] Theta scores: {len(df_theta)} rows, {len(df_theta.columns)} columns")

        # Validate structure
        expected_theta_cols = ['composite_ID', 'theta_common', 'theta_congruent', 'theta_incongruent',
                              'se_common', 'se_congruent', 'se_incongruent']
        missing_theta_cols = [col for col in expected_theta_cols if col not in df_theta.columns]

        if missing_theta_cols:
            raise ValueError(f"Theta scores missing columns: {missing_theta_cols}")

        if len(df_theta) != 400:
            log(f"[WARNING] Expected 400 rows (100 UID x 4 tests), got {len(df_theta)}")

        log(f"[PASS] Theta scores validated: {len(df_theta)} rows, all required columns present")

        # =====================================================================
        # STEP 4: Load and Validate LMM Input
        # =====================================================================
        log("[LOAD] Loading LMM input from RQ 5.4.1...")

        df_lmm_input = pd.read_csv(LMM_INPUT_FILE, encoding='utf-8')
        log(f"[LOADED] LMM input: {len(df_lmm_input)} rows, {len(df_lmm_input.columns)} columns")

        # Validate structure - use lowercase 'se' as that's what RQ 5.4.1 produces
        expected_lmm_cols = ['UID', 'test', 'TSVR_hours', 'congruence', 'theta', 'se']
        missing_lmm_cols = [col for col in expected_lmm_cols if col not in df_lmm_input.columns]

        if missing_lmm_cols:
            raise ValueError(f"LMM input missing columns: {missing_lmm_cols}")

        if len(df_lmm_input) != 1200:
            log(f"[WARNING] Expected 1200 rows (100 UID x 4 tests x 3 congruence), got {len(df_lmm_input)}")

        # Check congruence levels (case-insensitive check since they might be lowercase)
        congruence_levels = df_lmm_input['congruence'].unique()
        if len(congruence_levels) != 3:
            raise ValueError(f"Expected 3 congruence levels, found {len(congruence_levels)}: {congruence_levels}")

        # Normalize to title case for comparison
        actual_congruence = {c.title() for c in congruence_levels}
        expected_congruence = {'Common', 'Congruent', 'Incongruent'}

        if actual_congruence != expected_congruence:
            raise ValueError(f"Unexpected congruence levels. Expected {expected_congruence}, got {actual_congruence}")

        log(f"[PASS] Congruence levels validated: {sorted(congruence_levels)}")

        # Check for NaN values
        nan_theta = df_lmm_input['theta'].isna().sum()
        nan_tsvr = df_lmm_input['TSVR_hours'].isna().sum()

        if nan_theta > 0:
            raise ValueError(f"Found {nan_theta} NaN values in theta column")
        if nan_tsvr > 0:
            raise ValueError(f"Found {nan_tsvr} NaN values in TSVR_hours column")

        log(f"[PASS] No NaN values in theta or TSVR_hours columns")

        # Check TSVR_hours range
        tsvr_min = df_lmm_input['TSVR_hours'].min()
        tsvr_max = df_lmm_input['TSVR_hours'].max()

        log(f"[INFO] TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours")

        # =====================================================================
        # STEP 5: Cache LMM Input Locally (with normalized congruence names)
        # =====================================================================
        log("[SAVE] Caching LMM input for local workflow...")

        # Normalize congruence to title case for consistency
        df_lmm_input['congruence'] = df_lmm_input['congruence'].str.title()

        output_lmm_input = RQ_DIR / "data" / "step01_loaded_lmm_input.csv"
        df_lmm_input.to_csv(output_lmm_input, index=False, encoding='utf-8')

        log(f"[SAVED] {output_lmm_input.name} ({len(df_lmm_input)} rows, {len(df_lmm_input.columns)} columns)")

        # =====================================================================
        # STEP 6: Write Validation Report
        # =====================================================================
        log("[REPORT] Writing validation report...")

        report_path = RQ_DIR / "data" / "step01_dependency_validation_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.4.6 DEPENDENCY VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("DEPENDENCY: RQ 5.4.1 (Schema-Specific IRT Calibration)\n\n")

            f.write(f"RQ 5.4.1 Status: {rq_results_status}\n\n")

            f.write("DEPENDENCY FILES:\n")
            f.write(f"  1. Theta scores: {THETA_FILE.name}\n")
            f.write(f"     - Rows: {len(df_theta)}\n")
            f.write(f"     - Columns: {len(df_theta.columns)}\n")
            f.write(f"  2. LMM input: {LMM_INPUT_FILE.name}\n")
            f.write(f"     - Rows: {len(df_lmm_input)}\n")
            f.write(f"     - Columns: {len(df_lmm_input.columns)}\n")
            f.write(f"     - Congruence levels: {', '.join(sorted(df_lmm_input['congruence'].unique()))}\n")
            f.write("\n")

            f.write("DATA VALIDATION:\n")
            f.write(f"  - Congruence levels: {len(df_lmm_input['congruence'].unique())} unique ({', '.join(sorted(df_lmm_input['congruence'].unique()))})\n")
            f.write(f"  - NaN in theta: {nan_theta}\n")
            f.write(f"  - NaN in TSVR_hours: {nan_tsvr}\n")
            f.write(f"  - TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] hours\n")
            f.write("\n")

            f.write("VALIDATION STATUS: PASS\n")
            f.write("\n")
            f.write("All dependency checks passed. Ready to proceed with congruence-stratified analysis.\n")

        log(f"[SAVED] {report_path.name}")

        log("[SUCCESS] Step 01 complete - All dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
