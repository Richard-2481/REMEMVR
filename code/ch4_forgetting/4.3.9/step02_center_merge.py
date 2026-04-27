#!/usr/bin/env python3
"""
Step 02: Center Difficulty Variable and Merge TSVR Time

Purpose:
- Grand-mean center item difficulty predictor (Difficulty_c = Difficulty - mean[Difficulty])
- Merge TSVR time variable per Decision D070
- Create Time variable for LMM modeling

Inputs:
- data/step01_analysis_ready.csv (composite_ID, UID, Test, Item, Response, paradigm, Difficulty)
- data/step00_tsvr_mapping.csv (composite_ID, TSVR_hours)

Outputs:
- data/step02_lmm_input.csv (all input columns + Difficulty_c, Time)

Validation:
- mean(Difficulty_c) ≈ 0 (within ±0.01 tolerance)
- No NaN in Difficulty_c or Time columns
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_missing_data

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_center_merge.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def main():
    try:
        log("[START] Step 02: Center Difficulty and Merge TSVR")

        # =====================================================================
        # STEP 1: Load Analysis-Ready Data
        # =====================================================================
        log("[LOAD] Loading step01_analysis_ready.csv...")
        input_path = RQ_DIR / "data" / "step01_analysis_ready.csv"
        df = pd.read_csv(input_path)
        log(f"[LOADED] {len(df)} rows, {len(df.columns)} columns")
        log(f"[INFO] Columns: {list(df.columns)}")

        # =====================================================================
        # STEP 2: Center Difficulty Variable
        # =====================================================================
        log("[ANALYSIS] Centering Difficulty variable...")

        # Compute grand mean
        mean_difficulty = df['Difficulty'].mean()
        log(f"[INFO] Grand mean of Difficulty: {mean_difficulty:.6f}")

        # Create centered variable
        df['Difficulty_c'] = df['Difficulty'] - mean_difficulty

        # Verify centering
        centered_mean = df['Difficulty_c'].mean()
        centered_sd = df['Difficulty_c'].std()
        log(f"[INFO] Mean of Difficulty_c: {centered_mean:.6f}")
        log(f"[INFO] SD of Difficulty_c: {centered_sd:.4f}")

        # Validate centering (mean ≈ 0)
        tolerance = 0.01
        if abs(centered_mean) > tolerance:
            log(f"[ERROR] Centering failed: mean = {centered_mean:.6f} (expected ≈ 0, tolerance ±{tolerance})")
            raise ValueError(f"Centering validation failed: mean = {centered_mean:.6f}")
        else:
            log(f"[PASS] Centering successful: mean(Difficulty_c) = {centered_mean:.6f} (within ±{tolerance})")

        # =====================================================================
        # STEP 3: Load and Merge TSVR Time Variable
        # =====================================================================
        log("[LOAD] Loading TSVR mapping...")
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path)
        log(f"[LOADED] {len(df_tsvr)} TSVR rows")
        log(f"[INFO] TSVR columns: {list(df_tsvr.columns)}")

        # Merge by composite_ID
        log("[MERGE] Merging TSVR_hours by composite_ID...")
        df_before = len(df)
        df = df.merge(df_tsvr[['composite_ID', 'TSVR_hours']], on='composite_ID', how='left')
        df_after = len(df)

        if df_before != df_after:
            log(f"[ERROR] Row count changed during merge: {df_before} -> {df_after}")
            raise ValueError("Merge operation changed row count")

        log(f"[PASS] Merge successful: row count unchanged ({df_after} rows)")

        # Check for missing TSVR_hours
        missing_tsvr = df['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            log(f"[ERROR] {missing_tsvr} rows have missing TSVR_hours after merge")
            raise ValueError(f"{missing_tsvr} composite_IDs not found in TSVR mapping")

        log("[PASS] No missing TSVR_hours values")

        # Create Time variable
        df['Time'] = df['TSVR_hours']
        log("[INFO] Created Time variable (Time = TSVR_hours)")

        # =====================================================================
        # STEP 4: Save LMM Input
        # =====================================================================
        log("[SAVE] Saving LMM input data...")
        output_path = RQ_DIR / "data" / "step02_lmm_input.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {output_path}")
        log(f"[INFO] Output shape: {len(df)} rows, {len(df.columns)} columns")
        log(f"[INFO] Columns: {list(df.columns)}")

        # =====================================================================
        # STEP 5: Run Validation Tools
        # =====================================================================
        log("[VALIDATION] Validating centering (mean ≈ 0)...")

        # Manual validation for centering (mean ≈ 0, NOT standardization with SD ≈ 1)
        centering_valid = abs(df['Difficulty_c'].mean()) < tolerance
        if not centering_valid:
            log(f"[ERROR] Centering validation failed: mean = {df['Difficulty_c'].mean():.6f}")
            raise ValueError("Centering validation failed")

        log(f"[PASS] Centering validated: mean(Difficulty_c) = {df['Difficulty_c'].mean():.6f}")

        log("[VALIDATION] Running check_missing_data...")
        missing_result = check_missing_data(df)
        log(f"[VALIDATION] Missing data result: {missing_result}")

        # Check that Difficulty_c and Time have no missing values
        if missing_result.get('missing_by_column', {}).get('Difficulty_c', 0) > 0:
            log("[ERROR] Difficulty_c has missing values")
            raise ValueError("Difficulty_c has missing values")

        if missing_result.get('missing_by_column', {}).get('Time', 0) > 0:
            log("[ERROR] Time has missing values")
            raise ValueError("Time has missing values")

        log("[PASS] No missing values in Difficulty_c and Time")

        log("[SUCCESS] Step 02 complete")
        return 0

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK]")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        return 1

if __name__ == "__main__":
    sys.exit(main())
