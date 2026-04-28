#!/usr/bin/env python3
"""Get Data (Load RQ 5.1.1 Outputs): Load theta scores, TSVR mapping, and best continuous model from RQ 5.1.1 outputs."""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_get_data.log"

# RQ 5.1.1 dependency paths (absolute)
RQ7_DIR = PROJECT_ROOT / "results" / "ch5" / "rq7"
RQ7_LMM_INPUT_FILE = RQ7_DIR / "data" / "step04_lmm_input.csv"  # Has UID, test, Theta, TSVR_hours
RQ7_MODEL_COMPARISON_FILE = RQ7_DIR / "results" / "step05_model_comparison.csv"  # Has best model AIC


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Get Data (Load RQ 5.1.1 Outputs)")
        # Validate RQ 5.1.1 Dependency Files Exist

        log("Checking RQ 5.1.1 dependency files exist...")

        # Check RQ 5.1.1 LMM input file exists
        validation_result = check_file_exists(
            file_path=str(RQ7_LMM_INPUT_FILE),
            min_size_bytes=1000  # Should be ~35KB
        )

        if not validation_result.get('valid', False):
            raise FileNotFoundError(
                f"EXPECTATIONS ERROR: RQ 5.1.1 dependency missing or invalid: {RQ7_LMM_INPUT_FILE}\n"
                f"Validation message: {validation_result.get('message', 'unknown')}\n"
                f"RQ 5.1.1 must complete successfully before RQ 5.1.2 can proceed."
            )

        log(f"Dependency exists: {RQ7_LMM_INPUT_FILE.name} ({validation_result['size_bytes']} bytes)")
        # Load RQ 5.1.1 LMM Input Data
        # RQ 5.1.1 step04_lmm_input.csv already has UID, test, Theta, TSVR_hours merged
        # We just need to select relevant columns and rename Theta -> theta

        log("Loading RQ 5.1.1 LMM input data...")
        rq7_data = pd.read_csv(RQ7_LMM_INPUT_FILE, encoding='utf-8')
        log(f"{RQ7_LMM_INPUT_FILE.name} ({len(rq7_data)} rows, {len(rq7_data.columns)} cols)")
        log(f"Columns: {list(rq7_data.columns)}")

        # Load RQ 5.1.1 model comparison results (for AIC)
        log("Loading RQ 5.1.1 model comparison...")
        rq7_models = pd.read_csv(RQ7_MODEL_COMPARISON_FILE, encoding='utf-8')
        log(f"{RQ7_MODEL_COMPARISON_FILE.name} ({len(rq7_models)} models)")

        # Get best model (lowest AIC = delta_AIC == 0)
        best_model = rq7_models[rq7_models['delta_AIC'] == 0].iloc[0]
        rq7_aic = best_model['AIC']
        rq7_model_name = best_model['model_name']
        log(f"RQ 5.1.1 best model: {rq7_model_name} (AIC = {rq7_aic:.2f})")
        # Extract relevant columns and rename
        # RQ 5.1.1 has: composite_ID, UID, test, Theta, SE, TSVR_hours, Days, Days_squared, log_Days_plus1
        # RQ 5.1.2 needs: UID, test, theta (lowercase), TSVR_hours

        log("Selecting columns (UID, test, Theta, TSVR_hours) and renaming...")
        theta_tsvr = rq7_data[['UID', 'test', 'Theta', 'TSVR_hours']].copy()
        theta_tsvr.rename(columns={'Theta': 'theta'}, inplace=True)
        log(f"Extracted shape: {theta_tsvr.shape}")
        log(f"Columns: {list(theta_tsvr.columns)}")
        # Validation
        # Check row count (expect ~400: 100 participants x 4 tests)
        expected_rows_min = 380
        expected_rows_max = 420

        if len(theta_tsvr) < expected_rows_min:
            log(f"Data has {len(theta_tsvr)} rows (expected ~400)")
        elif len(theta_tsvr) > expected_rows_max:
            log(f"Data has {len(theta_tsvr)} rows (expected ~400)")
        else:
            log(f"Row count reasonable: {len(theta_tsvr)} rows")

        # Check for NaN values
        nan_theta = theta_tsvr['theta'].isna().sum()
        nan_tsvr = theta_tsvr['TSVR_hours'].isna().sum()

        if nan_theta > 0:
            raise ValueError(
                f"Merged data contains {nan_theta} NaN values in theta column\n"
                f"Check RQ 5.1.1 theta scores for missing values"
            )

        if nan_tsvr > 0:
            raise ValueError(
                f"Merged data contains {nan_tsvr} NaN values in TSVR_hours column\n"
                f"Check RQ 5.1.1 TSVR mapping for missing values"
            )

        log(f"No NaN values in theta or TSVR_hours")
        # Save Outputs
        # Save merged theta + TSVR dataset
        # Save RQ 5.1.1 AIC value (for Step 3 comparison)
        # Save RQ 5.1.1 convergence status (CRITICAL for interpretation)

        output_theta_tsvr = RQ_DIR / "data" / "step00_theta_tsvr.csv"
        log(f"Saving merged theta + TSVR to {output_theta_tsvr.name}...")
        theta_tsvr.to_csv(output_theta_tsvr, index=False, encoding='utf-8')
        log(f"{output_theta_tsvr.name} ({len(theta_tsvr)} rows, {len(theta_tsvr.columns)} cols)")

        output_aic = RQ_DIR / "data" / "step00_best_continuous_aic.txt"
        log(f"Saving RQ 5.1.1 AIC to {output_aic.name}...")
        with open(output_aic, 'w', encoding='utf-8') as f:
            f.write(f"{rq7_aic:.6f}\n")
        log(f"{output_aic.name} (AIC = {rq7_aic:.2f})")

        output_convergence = RQ_DIR / "data" / "step00_rq57_convergence.txt"
        log(f"Saving RQ 5.1.1 model info to {output_convergence.name}...")
        with open(output_convergence, 'w', encoding='utf-8') as f:
            f.write(f"RQ 5.1.1 Best Continuous Model Information\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Best model: {rq7_model_name}\n")
            f.write(f"AIC: {rq7_aic:.6f}\n")
            f.write(f"\n")
            f.write(f"INTERPRETATION FOR RQ 5.1.2 STEP 3:\n")
            f.write(f"-" * 60 + "\n")
            f.write(f"RQ 5.1.1 completed successfully with {rq7_model_name} as best model.\n")
            f.write(f"AIC comparison in Step 3 uses this value as reference.\n")
        log(f"{output_convergence.name}")

        log("Step 0 complete")
        log(f"Outputs:")
        log(f"  - {output_theta_tsvr.name}: {len(theta_tsvr)} rows (merged theta + TSVR)")
        log(f"  - {output_aic.name}: AIC = {rq7_aic:.2f} (for Step 3 comparison)")
        log(f"  - {output_convergence.name}: Convergence status (CRITICAL for interpretation)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
