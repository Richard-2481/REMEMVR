#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: step00_extract_data
Step Name: Extract VR Data for Omnibus IRT Analysis
RQ: ch5/5.1.1
Generated: 2025-12-01

PURPOSE:
Extract VR test item responses from dfData.csv, dichotomize scores,
create Q-matrix for single omnibus "All" factor IRT, and prepare TSVR mapping.

This is the ROOT extraction for General type RQs (5.1.X).
5.1.1 extracts independently from dfData.csv - NO cross-type dependencies.

EXPECTED INPUTS:
- data/step00_input_data.csv (LOCAL to this RQ - no external dependencies)
  Columns: [UID, TEST, TSVR, TQ_* columns]
  Format: CSV with UTF-8 encoding
  Expected rows: ~400 (100 participants x 4 tests)

EXPECTED OUTPUTS:
- data/step00_irt_input.csv
  Columns: [composite_ID, <item_tags>]
  Format: CSV with UTF-8 encoding, wide format
  Expected rows: ~400

- data/step00_tsvr_mapping.csv
  Columns: [composite_ID, UID, test, TSVR_hours]
  Format: CSV with UTF-8 encoding
  Expected rows: ~400

- data/step00_q_matrix.csv
  Columns: [item_name, All]
  Format: CSV with UTF-8 encoding
  Expected rows: ~100-200 items (all items load on single "All" factor)

VALIDATION CRITERIA:
- Output files exist: CRITICAL
- Row counts: ~400 for irt_input/tsvr_mapping: CRITICAL
- Item values binary: All item values in {0, 1, NaN} only: CRITICAL
- TSVR range: TSVR_hours in [0, 300]: CRITICAL
- Q-matrix: Each row sums to exactly 1 (all items load on "All"): CRITICAL
- All UIDs present: 100 unique UIDs in output: CRITICAL

IMPLEMENTATION NOTES:
- Analysis type: stdlib (pure pandas/numpy operations)
- Omnibus factor: ALL items (What/Where/When) assigned to single "All" factor
- This differs from 5.2.1 which has What/Where/When factors
- Interactive paradigms only: IFR, ICR, IRE (excludes RFR, TCR)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import traceback
from typing import List, Dict

# =============================================================================
# Configuration
# =============================================================================

# Project root and RQ directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step00_extract_data.log"

# Input file - LOCAL to this RQ (no external dependencies)
INPUT_FILE = RQ_DIR / "data" / "step00_input_data.csv"

# Output files (relative to RQ_DIR)
OUTPUT_IRT_INPUT = RQ_DIR / "data" / "step00_irt_input.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
OUTPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"

# Parameters
DICHOTOMIZATION_THRESHOLD = 1.0

# Interactive paradigm prefixes to KEEP (consistent with other RQs)
PARADIGM_PREFIXES_KEEP = ["TQ_IFR-", "TQ_ICR-", "TQ_IRE-"]

# Prefixes to explicitly EXCLUDE (Room Free Recall, Task Cued Recall)
PARADIGM_PREFIXES_EXCLUDE = ["TQ_RFR-", "TQ_TCR-", "TQ_RRE-"]

# =============================================================================
# Logging Function
# =============================================================================

def log(msg: str) -> None:
    """Write to both log file and console with ASCII-only output."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Validation Functions
# =============================================================================

def validate_outputs(
    df_irt: pd.DataFrame,
    df_tsvr: pd.DataFrame,
    df_qmatrix: pd.DataFrame,
    item_cols: List[str]
) -> None:
    """
    Validate all outputs against criteria.
    Raises ValueError with specific message on any failure.
    """
    log("[VALIDATION] Running validation checks...")

    # 1. Row counts
    irt_rows = len(df_irt)
    tsvr_rows = len(df_tsvr)
    qmatrix_rows = len(df_qmatrix)

    log(f"[VALIDATION] Row counts: IRT input={irt_rows}, TSVR mapping={tsvr_rows}, Q-matrix={qmatrix_rows}")

    if irt_rows < 300 or irt_rows > 500:
        raise ValueError(f"IRT input row count ({irt_rows}) outside expected range [300, 500]")

    if tsvr_rows < 300 or tsvr_rows > 500:
        raise ValueError(f"TSVR mapping row count ({tsvr_rows}) outside expected range [300, 500]")

    if qmatrix_rows < 50 or qmatrix_rows > 300:
        raise ValueError(f"Q-matrix row count ({qmatrix_rows}) outside expected range [50, 300]")

    # 2. Item values binary (0, 1, or NaN only)
    log("[VALIDATION] Checking item values are binary...")
    for col in item_cols:
        if col in df_irt.columns:
            unique_vals = df_irt[col].dropna().unique()
            invalid = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
            if len(invalid) > 0:
                raise ValueError(f"Item {col} has non-binary values: {invalid}")

    # 3. TSVR range check
    log("[VALIDATION] Checking TSVR_hours range...")
    tsvr_min = df_tsvr["TSVR_hours"].min()
    tsvr_max = df_tsvr["TSVR_hours"].max()
    log(f"[VALIDATION] TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

    if tsvr_min < 0 or tsvr_max > 300:
        raise ValueError(f"TSVR_hours outside expected range [0, 300]: min={tsvr_min}, max={tsvr_max}")

    # 4. Q-matrix row sum check (each item loads on exactly 1 factor = "All")
    log("[VALIDATION] Checking Q-matrix row sums...")
    qmatrix_rowsums = df_qmatrix[["All"]].sum(axis=1)
    invalid_rowsums = qmatrix_rowsums[qmatrix_rowsums != 1]
    if len(invalid_rowsums) > 0:
        bad_items = df_qmatrix.loc[invalid_rowsums.index, "item_name"].tolist()
        raise ValueError(f"Q-matrix has {len(invalid_rowsums)} items not loading on exactly 1 factor: {bad_items[:5]}...")

    # 5. Unique UIDs count
    log("[VALIDATION] Checking unique UIDs...")
    unique_uids = df_tsvr["UID"].nunique()
    log(f"[VALIDATION] Unique UIDs: {unique_uids}")

    if unique_uids < 90:  # Allow some tolerance for missing data
        raise ValueError(f"Expected ~100 unique UIDs, found only {unique_uids}")

    # 6. composite_ID format check
    log("[VALIDATION] Checking composite_ID format...")
    sample_ids = df_irt["composite_ID"].head(10).tolist()
    pattern = re.compile(r"^.+_\d+$")  # UID_test pattern
    invalid_ids = [id for id in sample_ids if not pattern.match(str(id))]
    if len(invalid_ids) > 0:
        log(f"[WARNING] Some composite_IDs may not match expected pattern: {invalid_ids}")

    # 7. Consistency check: IRT and TSVR have same composite_IDs
    irt_ids = set(df_irt["composite_ID"])
    tsvr_ids = set(df_tsvr["composite_ID"])
    if irt_ids != tsvr_ids:
        missing_in_tsvr = irt_ids - tsvr_ids
        missing_in_irt = tsvr_ids - irt_ids
        raise ValueError(f"composite_ID mismatch: {len(missing_in_tsvr)} in IRT but not TSVR, {len(missing_in_irt)} in TSVR but not IRT")

    log("[VALIDATION] All validation checks PASSED")

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        # Initialize log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Step 00: Extract VR Data for Omnibus IRT Analysis (General Type ROOT)\n")
            f.write("=" * 80 + "\n\n")

        log("[START] Step 00: Extract VR Data for Omnibus IRT Analysis")
        log(f"[INFO] RQ: 5.1.1 (General Type ROOT)")
        log(f"[INFO] Source: step00_input_data.csv (LOCAL - no external dependencies)")

        # =====================================================================
        # STEP 1: Load Input Data
        # =====================================================================
        log("[LOAD] Loading input data from step00_input_data.csv...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"[LOADED] step00_input_data.csv ({len(df)} rows, {len(df.columns)} cols)")

        # =====================================================================
        # STEP 2: Identify Columns
        # =====================================================================
        log("[PROCESS] Identifying columns...")

        # Required metadata columns
        meta_cols = ["UID", "TEST", "TSVR"]
        for col in meta_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Item columns are all TQ_* columns
        all_item_cols = [col for col in df.columns if col.startswith("TQ_")]
        log(f"[INFO] Found {len(all_item_cols)} total item columns (TQ_*)")

        # Filter to interactive paradigms only (IFR, ICR, IRE)
        item_cols = []
        for col in all_item_cols:
            is_interactive = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_KEEP)
            is_excluded = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_EXCLUDE)
            if is_interactive and not is_excluded:
                item_cols.append(col)

        log(f"[INFO] Filtered to {len(item_cols)} interactive paradigm items (IFR/ICR/IRE)")
        log(f"[INFO] Excluded {len(all_item_cols) - len(item_cols)} non-interactive items (RFR/TCR)")

        # =====================================================================
        # STEP 3: Create composite_ID
        # =====================================================================
        log("[PROCESS] Creating composite_ID...")
        df["composite_ID"] = df["UID"].astype(str) + "_" + df["TEST"].astype(str)
        log(f"[INFO] Created {df['composite_ID'].nunique()} unique composite_IDs")

        # =====================================================================
        # STEP 4: Dichotomize Item Responses
        # =====================================================================
        log(f"[PROCESS] Dichotomizing item responses (threshold={DICHOTOMIZATION_THRESHOLD})...")

        df_items = df[item_cols].copy()

        # Apply dichotomization: >= threshold -> 1, < threshold -> 0, NaN stays NaN
        df_items_binary = df_items.apply(
            lambda x: np.where(x >= DICHOTOMIZATION_THRESHOLD, 1,
                              np.where(x < DICHOTOMIZATION_THRESHOLD, 0, np.nan))
        )

        # Convert to appropriate type (float to preserve NaN)
        df_items_binary = df_items_binary.astype(float)

        # Report dichotomization stats
        total_responses = df_items_binary.count().sum()
        ones = (df_items_binary == 1).sum().sum()
        zeros = (df_items_binary == 0).sum().sum()
        nan_count = df_items_binary.isna().sum().sum()
        log(f"[INFO] Dichotomization results:")
        log(f"       - Total valid responses: {total_responses}")
        log(f"       - Correct (1): {ones} ({100*ones/total_responses:.1f}%)")
        log(f"       - Incorrect (0): {zeros} ({100*zeros/total_responses:.1f}%)")
        log(f"       - Missing (NaN): {nan_count}")

        # =====================================================================
        # STEP 5: Create IRT Input (Wide Format)
        # =====================================================================
        log("[PROCESS] Creating IRT input file (wide format)...")

        df_irt = pd.DataFrame({
            "composite_ID": df["composite_ID"]
        })
        df_irt = pd.concat([df_irt, df_items_binary], axis=1)

        log(f"[INFO] IRT input shape: {df_irt.shape}")

        # =====================================================================
        # STEP 6: Create TSVR Mapping
        # =====================================================================
        log("[PROCESS] Creating TSVR mapping file...")

        df_tsvr = df[["composite_ID", "UID", "TEST", "TSVR"]].copy()
        df_tsvr = df_tsvr.rename(columns={
            "TEST": "test",
            "TSVR": "TSVR_hours"
        })

        log(f"[INFO] TSVR mapping shape: {df_tsvr.shape}")
        log(f"[INFO] TSVR_hours range: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")

        # =====================================================================
        # STEP 7: Create Q-Matrix (Omnibus "All" Factor)
        # =====================================================================
        log("[PROCESS] Creating Q-matrix with omnibus 'All' factor...")

        # ALL items load on single "All" factor (omnibus)
        qmatrix_data = []
        for item in item_cols:
            qmatrix_data.append({
                "item_name": item,
                "All": 1  # All items load on single omnibus factor
            })

        df_qmatrix = pd.DataFrame(qmatrix_data)

        log(f"[INFO] Q-matrix shape: {df_qmatrix.shape}")
        log(f"[INFO] All {len(df_qmatrix)} items load on single 'All' factor (omnibus)")

        # =====================================================================
        # STEP 8: Validate Outputs
        # =====================================================================
        validate_outputs(df_irt, df_tsvr, df_qmatrix, item_cols)

        # =====================================================================
        # STEP 9: Save Output Files
        # =====================================================================
        log("[SAVE] Saving output files...")

        # Ensure output directories exist
        OUTPUT_IRT_INPUT.parent.mkdir(parents=True, exist_ok=True)

        # Save IRT input
        df_irt.to_csv(OUTPUT_IRT_INPUT, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_IRT_INPUT} ({len(df_irt)} rows, {len(df_irt.columns)} cols)")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_TSVR_MAPPING} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")

        # Save Q-matrix
        df_qmatrix.to_csv(OUTPUT_Q_MATRIX, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_Q_MATRIX} ({len(df_qmatrix)} rows, {len(df_qmatrix.columns)} cols)")

        # =====================================================================
        # STEP 10: Final Summary
        # =====================================================================
        log("")
        log("=" * 80)
        log("[SUCCESS] Step 00 complete - General Type ROOT extraction")
        log("=" * 80)
        log(f"Outputs created:")
        log(f"  1. IRT input: {OUTPUT_IRT_INPUT}")
        log(f"     -> {len(df_irt)} observations, {len(item_cols)} items")
        log(f"  2. TSVR mapping: {OUTPUT_TSVR_MAPPING}")
        log(f"     -> {len(df_tsvr)} observations, TSVR_hours for Decision D070")
        log(f"  3. Q-matrix: {OUTPUT_Q_MATRIX}")
        log(f"     -> {len(df_qmatrix)} items, 1 factor (omnibus 'All')")
        log("")
        log("[INFO] This is the ROOT extraction for General type (5.1.X)")
        log("[INFO] NO cross-type dependencies - extracts directly from dfData.csv")
        log("")
        log("Ready for Step 01: IRT Calibration Pass 1")

        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
