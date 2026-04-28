#!/usr/bin/env python3
"""Extract VR Data for Paradigm IRT Analysis: Extract VR test item responses from dfData.csv, dichotomize scores,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1
LOG_FILE = RQ_DIR / "logs" / "step00_prepare_paradigm_data.log"

# Input file - LOCAL to this RQ (no external dependencies)
INPUT_FILE = RQ_DIR / "data" / "step00_input_data.csv"

# Output files
OUTPUT_IRT_INPUT = RQ_DIR / "data" / "step00_irt_input.csv"
OUTPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"

# Parameters
DICHOTOMIZATION_THRESHOLD = 1.0

# Paradigm prefixes to KEEP
PARADIGM_PREFIXES_KEEP = ["TQ_IFR-", "TQ_ICR-", "TQ_IRE-"]

# Prefixes to explicitly EXCLUDE
PARADIGM_PREFIXES_EXCLUDE = ["TQ_RFR-", "TQ_TCR-", "TQ_RRE-"]

# Paradigm patterns for Q-matrix assignment
PARADIGM_PATTERNS = {
    'free_recall': 'IFR',      # Item Free Recall
    'cued_recall': 'ICR',      # Item Cued Recall
    'recognition': 'IRE'       # Item Recognition
}

# Logging Function

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{formatted_msg}\n")
    print(formatted_msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Initialize log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Step 00: Extract VR Data for Paradigm IRT Analysis (Paradigms Type ROOT)\n")
            f.write(f"{'='*80}\n\n")

        log("Step 00: Extract VR Data for Paradigm IRT Analysis")
        log(f"RQ: 5.3.1 (Paradigms Type ROOT)")
        log(f"Source: step00_input_data.csv (LOCAL - no external dependencies)")
        # Load Input Data from step00_input_data.csv
        log("Loading raw data from step00_input_data.csv...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"step00_input_data.csv ({len(df)} rows, {len(df.columns)} cols)")
        # Identify and Filter Columns
        log("Identifying columns...")

        # Required metadata columns
        meta_cols = ["UID", "TEST", "TSVR"]
        for col in meta_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Item columns are all TQ_* columns
        all_item_cols = [col for col in df.columns if col.startswith("TQ_")]
        log(f"Found {len(all_item_cols)} total item columns (TQ_*)")

        # Filter to paradigm items only (IFR, ICR, IRE)
        paradigm_cols = []
        for col in all_item_cols:
            is_paradigm = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_KEEP)
            is_excluded = any(col.startswith(prefix) for prefix in PARADIGM_PREFIXES_EXCLUDE)
            if is_paradigm and not is_excluded:
                paradigm_cols.append(col)

        log(f"Retained {len(paradigm_cols)} paradigm items (IFR/ICR/IRE)")
        log(f"Excluded {len(all_item_cols) - len(paradigm_cols)} non-paradigm items")

        # Count by paradigm
        ifr_count = sum(1 for c in paradigm_cols if 'IFR' in c)
        icr_count = sum(1 for c in paradigm_cols if 'ICR' in c)
        ire_count = sum(1 for c in paradigm_cols if 'IRE' in c)
        log(f"Free Recall (IFR): {ifr_count} items")
        log(f"Cued Recall (ICR): {icr_count} items")
        log(f"Recognition (IRE): {ire_count} items")
        # Create composite_ID
        log("Creating composite_ID...")
        df["composite_ID"] = df["UID"].astype(str) + "_" + df["TEST"].astype(str)
        log(f"Created {df['composite_ID'].nunique()} unique composite_IDs")
        # Dichotomize Item Responses
        log(f"Dichotomizing item responses (threshold={DICHOTOMIZATION_THRESHOLD})...")

        df_items = df[paradigm_cols].copy()

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
        log(f"Dichotomization results:")
        log(f"       - Total valid responses: {total_responses}")
        log(f"       - Correct (1): {ones} ({100*ones/total_responses:.1f}%)")
        log(f"       - Incorrect (0): {zeros} ({100*zeros/total_responses:.1f}%)")
        log(f"       - Missing (NaN): {nan_count}")
        # Create IRT Input (Wide Format)
        log("Creating IRT input file (wide format)...")

        df_irt = pd.DataFrame({
            "composite_ID": df["composite_ID"]
        })
        df_irt = pd.concat([df_irt, df_items_binary], axis=1)

        log(f"IRT input shape: {df_irt.shape}")
        # Create TSVR Mapping
        log("Creating TSVR mapping file...")

        df_tsvr = df[["composite_ID", "UID", "TEST", "TSVR"]].copy()
        df_tsvr = df_tsvr.rename(columns={
            "TEST": "test",
            "TSVR": "TSVR_hours"
        })

        log(f"TSVR mapping shape: {df_tsvr.shape}")
        log(f"TSVR_hours range: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")
        # Create Q-Matrix (Paradigm Factors)
        log("Creating Q-matrix with paradigm factors...")

        q_matrix_rows = []
        for col in paradigm_cols:
            row = {
                'item_name': col,
                'free_recall': 1 if 'IFR' in col else 0,
                'cued_recall': 1 if 'ICR' in col else 0,
                'recognition': 1 if 'IRE' in col else 0
            }
            q_matrix_rows.append(row)

        df_qmatrix = pd.DataFrame(q_matrix_rows)

        log(f"Q-matrix shape: {df_qmatrix.shape}")
        log(f"Q-matrix factor counts:")
        log(f"       - free_recall: {df_qmatrix['free_recall'].sum()}")
        log(f"       - cued_recall: {df_qmatrix['cued_recall'].sum()}")
        log(f"       - recognition: {df_qmatrix['recognition'].sum()}")
        # Validation
        log("Running validation checks...")

        # Validation 1: Row count
        assert len(df_irt) == 400, f"Row count mismatch: expected 400, got {len(df_irt)}"
        log("Row count: 400 rows")

        # Validation 2: Q-matrix structure (each item in exactly one factor)
        factor_sums = df_qmatrix[['free_recall', 'cued_recall', 'recognition']].sum(axis=1)
        assert (factor_sums == 1).all(), "Q-matrix validation failed: Items not in exactly one factor"
        log("Q-matrix structure: Each item in exactly one factor")

        # Validation 3: No RFR/TCR columns
        filtered_cols = df_irt.columns.tolist()
        rfr_count_check = sum(1 for c in filtered_cols if 'RFR' in c)
        tcr_count_check = sum(1 for c in filtered_cols if 'TCR' in c)
        assert rfr_count_check == 0, f"Found {rfr_count_check} RFR columns (should be 0)"
        assert tcr_count_check == 0, f"Found {tcr_count_check} TCR columns (should be 0)"
        log("No RFR/TCR columns in output")

        # Validation 4: Minimum items per paradigm (at least 10)
        min_items_per_paradigm = 10
        assert ifr_count >= min_items_per_paradigm, f"IFR items ({ifr_count}) below minimum ({min_items_per_paradigm})"
        assert icr_count >= min_items_per_paradigm, f"ICR items ({icr_count}) below minimum ({min_items_per_paradigm})"
        assert ire_count >= min_items_per_paradigm, f"IRE items ({ire_count}) below minimum ({min_items_per_paradigm})"
        log(f"Minimum items per paradigm: IFR={ifr_count}, ICR={icr_count}, IRE={ire_count}")

        # Validation 5: Q-matrix item names match IRT columns
        qmatrix_items = set(df_qmatrix['item_name'].tolist())
        irt_items = set(paradigm_cols)
        assert qmatrix_items == irt_items, "Q-matrix items don't match IRT columns"
        log("Q-matrix item names match IRT input columns")

        # Validation 6: TSVR range
        tsvr_min = df_tsvr["TSVR_hours"].min()
        tsvr_max = df_tsvr["TSVR_hours"].max()
        assert tsvr_min >= 0 and tsvr_max <= 300, f"TSVR_hours outside expected range [0, 300]"
        log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        log("All validation checks PASSED")
        # Save Output Files
        log("Saving output files...")

        # Ensure output directories exist
        OUTPUT_IRT_INPUT.parent.mkdir(parents=True, exist_ok=True)

        # Save IRT input
        df_irt.to_csv(OUTPUT_IRT_INPUT, index=False, encoding='utf-8')
        log(f"{OUTPUT_IRT_INPUT} ({len(df_irt)} rows, {len(df_irt.columns)} cols)")

        # Save Q-matrix
        df_qmatrix.to_csv(OUTPUT_Q_MATRIX, index=False, encoding='utf-8')
        log(f"{OUTPUT_Q_MATRIX} ({len(df_qmatrix)} rows)")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"{OUTPUT_TSVR_MAPPING} ({len(df_tsvr)} rows)")
        # Summary
        log("")
        log("=" * 80)
        log("Step 00 Complete - Paradigms Type ROOT extraction")
        log("=" * 80)
        log(f"Outputs created:")
        log(f"  1. IRT input: {OUTPUT_IRT_INPUT}")
        log(f"     -> {len(df_irt)} observations, {len(paradigm_cols)} items")
        log(f"  2. TSVR mapping: {OUTPUT_TSVR_MAPPING}")
        log(f"     -> {len(df_tsvr)} observations, TSVR_hours for Decision D070")
        log(f"  3. Q-matrix: {OUTPUT_Q_MATRIX}")
        log(f"     -> {len(df_qmatrix)} items, 3 factors (IFR/ICR/IRE)")
        log(f"")
        log(f"  Paradigm breakdown:")
        log(f"    - Free Recall (IFR): {ifr_count} items")
        log(f"    - Cued Recall (ICR): {icr_count} items")
        log(f"    - Recognition (IRE): {ire_count} items")
        log("")
        log("This is the ROOT extraction for Paradigms type (5.3.X)")
        log("NO cross-type dependencies - extracts directly from dfData.csv")
        log("")
        log("Step 01: IRT Calibration Pass 1 (All Items)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
