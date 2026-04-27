#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: step00
Step Name: Extract VR Data for Congruence IRT Analysis
RQ: results/ch5/5.4.1
Generated: 2025-12-01 (Updated from 2025-11-24)

PURPOSE:
Extract VR test item responses from dfData.csv, dichotomize scores,
create Q-matrix for congruence-based IRT with Common/Congruent/Incongruent factors.

This is the ROOT extraction for Congruence type RQs (5.4.X).
5.4.1 extracts independently from dfData.csv - NO cross-type dependencies.

EXPECTED INPUTS:
  - data/step00_input_data.csv (LOCAL to this RQ - no external dependencies)
    Columns: [UID, TEST, TSVR, TQ_* columns]
    Format: CSV with UTF-8 encoding
    Expected rows: ~400 (100 participants x 4 test sessions)

EXPECTED OUTPUTS:
  - data/step00_irt_input.csv
    Columns: composite_ID + interactive paradigm item columns
    Format: Wide-format IRT input (binary 0/1)
    Expected rows: ~400

  - data/step00_q_matrix.csv
    Columns: item_name, common, congruent, incongruent
    Format: Q-matrix with binary loadings (each item loads on exactly 1 dimension)
    Expected rows: ~72 items (24 items x 3 paradigms)

  - data/step00_tsvr_mapping.csv
    Columns: composite_ID, UID, test, TSVR_hours
    Format: CSV with time-since-VR-encoding mapping
    Expected rows: ~400

VALIDATION CRITERIA:
  - Output files created
  - Row count: ~400
  - Q-matrix structure valid (each item loads on exactly 1 dimension)
  - All congruence categories present (common, congruent, incongruent)

IMPLEMENTATION NOTES:
- Item naming: TQ_{paradigm}-{domain}-{item} (e.g., TQ_IFR-N-i1)
- Interactive paradigms: IFR (Free Recall), ICR (Cued Recall), IRE (Recognition)
- Excludes: RFR (Room Free Recall), TCR (Task Cued Recall) - different format
- Congruence mapping: i1,i2 -> common; i3,i4 -> congruent; i5,i6 -> incongruent
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import traceback
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.4.1
LOG_FILE = RQ_DIR / "logs" / "step00_extract_congruence_data.log"

# Input file - LOCAL to this RQ (no external dependencies)
INPUT_FILE = RQ_DIR / "data" / "step00_input_data.csv"

# Output paths
OUTPUT_IRT_INPUT = RQ_DIR / "data" / "step00_irt_input.csv"
OUTPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"

# Parameters
DICHOTOMIZATION_THRESHOLD = 1.0

# Interactive paradigm prefixes to KEEP
PARADIGM_PREFIXES_KEEP = ["TQ_IFR-", "TQ_ICR-", "TQ_IRE-"]

# Prefixes to explicitly EXCLUDE (Room Free Recall, Task Cued Recall)
PARADIGM_PREFIXES_EXCLUDE = ["TQ_RFR-", "TQ_TCR-", "TQ_RRE-"]

# Congruence mapping based on item suffix
CONGRUENCE_MAPPING = {
    "common": ["-i1", "-i2"],
    "congruent": ["-i3", "-i4"],
    "incongruent": ["-i5", "-i6"]
}

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{formatted_msg}\n")
    print(formatted_msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        # Initialize log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Step 00: Extract VR Data for Congruence IRT Analysis (Congruence Type ROOT)\n")
            f.write(f"{'='*80}\n\n")

        log("[START] Step 00: Extract VR Data for Congruence IRT Analysis")
        log(f"[INFO] RQ: 5.4.1 (Congruence Type ROOT)")
        log(f"[INFO] Source: step00_input_data.csv (LOCAL - no external dependencies)")

        # =========================================================================
        # STEP 1: Load Input Data from step00_input_data.csv
        # =========================================================================
        log("[LOAD] Loading raw data from step00_input_data.csv...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"[LOADED] step00_input_data.csv ({len(df)} rows, {len(df.columns)} cols)")

        # =========================================================================
        # STEP 2: Identify and Filter Columns
        # =========================================================================
        log("[PROCESS] Identifying columns...")

        # Required metadata columns
        meta_cols = ["UID", "TEST", "TSVR"]
        for col in meta_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Item columns are all TQ_* columns
        all_item_cols = [col for col in df.columns if col.startswith("TQ_")]
        log(f"[INFO] Found {len(all_item_cols)} total item columns (TQ_*)")

        # Filter to interactive paradigm items only (IFR, ICR, IRE)
        interactive_cols = []
        for col in all_item_cols:
            is_interactive = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_KEEP)
            is_excluded = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_EXCLUDE)
            if is_interactive and not is_excluded:
                interactive_cols.append(col)

        log(f"[FILTERED] Retained {len(interactive_cols)} interactive paradigm items (IFR/ICR/IRE)")
        log(f"[FILTERED] Excluded {len(all_item_cols) - len(interactive_cols)} non-interactive items")

        # =========================================================================
        # STEP 3: Create composite_ID
        # =========================================================================
        log("[PROCESS] Creating composite_ID...")
        df["composite_ID"] = df["UID"].astype(str) + "_" + df["TEST"].astype(str)
        log(f"[INFO] Created {df['composite_ID'].nunique()} unique composite_IDs")

        # =========================================================================
        # STEP 4: Dichotomize Item Responses
        # =========================================================================
        log(f"[PROCESS] Dichotomizing item responses (threshold={DICHOTOMIZATION_THRESHOLD})...")

        df_items = df[interactive_cols].copy()

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

        # =========================================================================
        # STEP 5: Create IRT Input (Wide Format)
        # =========================================================================
        log("[PROCESS] Creating IRT input file (wide format)...")

        df_irt = pd.DataFrame({
            "composite_ID": df["composite_ID"]
        })
        df_irt = pd.concat([df_irt, df_items_binary], axis=1)

        log(f"[INFO] IRT input shape: {df_irt.shape}")

        # =========================================================================
        # STEP 6: Create TSVR Mapping
        # =========================================================================
        log("[PROCESS] Creating TSVR mapping file...")

        df_tsvr = df[["composite_ID", "UID", "TEST", "TSVR"]].copy()
        df_tsvr = df_tsvr.rename(columns={
            "TEST": "test",
            "TSVR": "TSVR_hours"
        })

        log(f"[INFO] TSVR mapping shape: {df_tsvr.shape}")
        log(f"[INFO] TSVR_hours range: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")

        # =========================================================================
        # STEP 7: Create Q-Matrix (Congruence Factors)
        # =========================================================================
        log("[PROCESS] Creating Q-matrix with congruence factors...")

        q_matrix_data = []
        skipped_items = []
        for item_name in interactive_cols:
            # Determine congruence category from item suffix
            congruence_found = None
            for congruence, suffixes in CONGRUENCE_MAPPING.items():
                if any(item_name.endswith(suffix) for suffix in suffixes):
                    congruence_found = congruence
                    break

            if congruence_found is None:
                skipped_items.append(item_name)
                continue

            q_matrix_data.append({
                "item_name": item_name,
                "common": 1 if congruence_found == "common" else 0,
                "congruent": 1 if congruence_found == "congruent" else 0,
                "incongruent": 1 if congruence_found == "incongruent" else 0
            })

        df_q_matrix = pd.DataFrame(q_matrix_data)

        if skipped_items:
            log(f"[WARN] {len(skipped_items)} items don't match any congruence suffix - skipped")
            log(f"[WARN] First 5 skipped: {skipped_items[:5]}")

        # Count items per congruence category
        n_common = df_q_matrix["common"].sum()
        n_congruent = df_q_matrix["congruent"].sum()
        n_incongruent = df_q_matrix["incongruent"].sum()

        log(f"[INFO] Q-matrix shape: {df_q_matrix.shape}")
        log(f"[INFO] Q-matrix factor counts:")
        log(f"       - common: {n_common} items")
        log(f"       - congruent: {n_congruent} items")
        log(f"       - incongruent: {n_incongruent} items")

        # =========================================================================
        # STEP 8: Validation
        # =========================================================================
        log("[VALIDATION] Running validation checks...")

        # Validation 1: Row count
        assert len(df_irt) == 400, f"Row count mismatch: expected 400, got {len(df_irt)}"
        log("[PASS] Row count: 400 rows")

        # Validation 2: Q-matrix structure (each item loads on exactly 1 dimension)
        row_sums = df_q_matrix[["common", "congruent", "incongruent"]].sum(axis=1)
        if not all(row_sums == 1):
            invalid_items = df_q_matrix[row_sums != 1]["item_name"].tolist()
            raise ValueError(f"Q-matrix invalid: items don't load on exactly 1 dimension: {invalid_items}")
        log("[PASS] Q-matrix structure: Each item in exactly one factor")

        # Validation 3: All congruence categories present
        assert n_common > 0, "No common items found"
        assert n_congruent > 0, "No congruent items found"
        assert n_incongruent > 0, "No incongruent items found"
        log("[PASS] All congruence categories present")

        # Validation 4: Item values valid (0, 1, or NaN)
        item_values = df_irt.drop(columns=["composite_ID"]).values.flatten()
        valid_values = {0.0, 1.0}
        unique_values = set(item_values[~np.isnan(item_values)])
        invalid_values = unique_values - valid_values
        if invalid_values:
            log(f"[WARN] Found unexpected item values: {invalid_values}")
        else:
            log("[PASS] Item values valid (0, 1, NaN only)")

        # Validation 5: TSVR range
        tsvr_min = df_tsvr["TSVR_hours"].min()
        tsvr_max = df_tsvr["TSVR_hours"].max()
        assert tsvr_min >= 0 and tsvr_max <= 300, f"TSVR_hours outside expected range [0, 300]"
        log(f"[PASS] TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        log("[VALIDATION] All validation checks PASSED")

        # =========================================================================
        # STEP 9: Save Output Files
        # =========================================================================
        log("[SAVE] Saving output files...")

        # Ensure data directory exists
        OUTPUT_IRT_INPUT.parent.mkdir(parents=True, exist_ok=True)

        # Save IRT input (filtered)
        df_irt.to_csv(OUTPUT_IRT_INPUT, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_IRT_INPUT} ({len(df_irt)} rows, {len(df_irt.columns)} cols)")

        # Save Q-matrix
        df_q_matrix.to_csv(OUTPUT_Q_MATRIX, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_Q_MATRIX} ({len(df_q_matrix)} rows)")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_TSVR_MAPPING} ({len(df_tsvr)} rows)")

        # =========================================================================
        # STEP 10: Summary
        # =========================================================================
        log("")
        log("=" * 80)
        log("[SUCCESS] Step 00 Complete - Congruence Type ROOT extraction")
        log("=" * 80)
        log(f"Outputs created:")
        log(f"  1. IRT input: {OUTPUT_IRT_INPUT}")
        log(f"     -> {len(df_irt)} observations, {len(interactive_cols)} items")
        log(f"  2. TSVR mapping: {OUTPUT_TSVR_MAPPING}")
        log(f"     -> {len(df_tsvr)} observations, TSVR_hours for Decision D070")
        log(f"  3. Q-matrix: {OUTPUT_Q_MATRIX}")
        log(f"     -> {len(df_q_matrix)} items, 3 factors (Common/Congruent/Incongruent)")
        log(f"")
        log(f"  Congruence breakdown:")
        log(f"    - Common (i1, i2): {n_common} items")
        log(f"    - Congruent (i3, i4): {n_congruent} items")
        log(f"    - Incongruent (i5, i6): {n_incongruent} items")
        if skipped_items:
            log(f"    - Skipped (no congruence suffix): {len(skipped_items)} items")
        log("")
        log("[INFO] This is the ROOT extraction for Congruence type (5.4.X)")
        log("[INFO] NO cross-type dependencies - extracts directly from dfData.csv")
        log("")
        log("[NEXT] Step 01: IRT Calibration Pass 1")

        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
