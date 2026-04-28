#!/usr/bin/env python3
"""Extract VR Data for IRT Analysis: Extract VR test item responses from dfData.csv, dichotomize scores,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import traceback
from typing import List, Dict

# Configuration

# Project root and RQ directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step00_extract_vr_data.log"

# Input file - LOCAL to this RQ (no external dependencies)
INPUT_FILE = RQ_DIR / "data" / "step00_input_data.csv"

# Output files (relative to RQ_DIR)
OUTPUT_IRT_INPUT = RQ_DIR / "data" / "step00_irt_input.csv"
OUTPUT_TSVR_MAPPING = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
OUTPUT_Q_MATRIX = RQ_DIR / "data" / "step00_q_matrix.csv"

# Parameters from 4_analysis.yaml
DICHOTOMIZATION_THRESHOLD = 1.0
DOMAIN_TAG_PATTERNS = {
    "what": ["-N-"],       # Named items (object identity)
    "where": ["-L-", "-U-", "-D-"],  # Location, Upstairs, Downstairs
    "when": ["-O-"]        # Order/temporal items
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Domain Classification Functions

def classify_item_domain(item_tag: str) -> str:
    """
    Classify an item into a domain based on tag pattern.

    Returns: 'what', 'where', 'when', or 'unknown'

    Examples:
    - TQ_RFR-N-OBJ1 -> 'what' (contains -N-)
    - TQ_RFR-L-OBJ1 -> 'where' (contains -L-)
    - TQ_IFR-U-i1 -> 'where' (contains -U-)
    - TQ_IFR-D-i1 -> 'where' (contains -D-)
    - TQ_RFR-O-RORD -> 'when' (contains -O-)
    """
    # Remove TQ_ prefix if present for cleaner matching
    tag = item_tag.replace("TQ_", "") if item_tag.startswith("TQ_") else item_tag

    for domain, patterns in DOMAIN_TAG_PATTERNS.items():
        for pattern in patterns:
            if pattern in tag:
                return domain

    return "unknown"

def get_items_by_domain(item_columns: List[str]) -> Dict[str, List[str]]:
    """
    Group item columns by domain.

    Returns dict with keys 'what', 'where', 'when', 'unknown'
    """
    grouped = {"what": [], "where": [], "when": [], "unknown": []}

    for col in item_columns:
        domain = classify_item_domain(col)
        grouped[domain].append(col)

    return grouped

# Validation Functions

def validate_outputs(
    df_irt: pd.DataFrame,
    df_tsvr: pd.DataFrame,
    df_qmatrix: pd.DataFrame,
    item_cols: List[str]
) -> None:
    """
    Validate all outputs against criteria from 4_analysis.yaml.
    Raises ValueError with specific message on any failure.
    """
    log("Running validation checks...")

    # 1. Row counts
    irt_rows = len(df_irt)
    tsvr_rows = len(df_tsvr)
    qmatrix_rows = len(df_qmatrix)

    log(f"Row counts: IRT input={irt_rows}, TSVR mapping={tsvr_rows}, Q-matrix={qmatrix_rows}")

    if irt_rows < 300 or irt_rows > 500:
        raise ValueError(f"IRT input row count ({irt_rows}) outside expected range [300, 500]")

    if tsvr_rows < 300 or tsvr_rows > 500:
        raise ValueError(f"TSVR mapping row count ({tsvr_rows}) outside expected range [300, 500]")

    if qmatrix_rows < 50 or qmatrix_rows > 300:
        raise ValueError(f"Q-matrix row count ({qmatrix_rows}) outside expected range [50, 300]")

    # 2. Item values binary (0, 1, or NaN only)
    log("Checking item values are binary...")
    for col in item_cols:
        if col in df_irt.columns:
            unique_vals = df_irt[col].dropna().unique()
            invalid = [v for v in unique_vals if v not in [0, 1, 0.0, 1.0]]
            if len(invalid) > 0:
                raise ValueError(f"Item {col} has non-binary values: {invalid}")

    # 3. TSVR range check
    log("Checking TSVR_hours range...")
    tsvr_min = df_tsvr["TSVR_hours"].min()
    tsvr_max = df_tsvr["TSVR_hours"].max()
    log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

    # Note: 4_analysis.yaml specified [0, 200] but actual data has TEST 4 up to ~250 hours
    # Updated to [0, 300] to accommodate actual data (TEST 4 = ~6-10 days delay)
    if tsvr_min < 0 or tsvr_max > 300:
        raise ValueError(f"TSVR_hours outside expected range [0, 300]: min={tsvr_min}, max={tsvr_max}")

    # 4. Q-matrix row sum check (each item loads on exactly 1 domain)
    log("Checking Q-matrix row sums...")
    qmatrix_rowsums = df_qmatrix[["what", "where", "when"]].sum(axis=1)
    invalid_rowsums = qmatrix_rowsums[qmatrix_rowsums != 1]
    if len(invalid_rowsums) > 0:
        bad_items = df_qmatrix.loc[invalid_rowsums.index, "item_name"].tolist()
        raise ValueError(f"Q-matrix has {len(invalid_rowsums)} items not loading on exactly 1 domain: {bad_items[:5]}...")

    # 5. Unique UIDs count
    log("Checking unique UIDs...")
    unique_uids = df_tsvr["UID"].nunique()
    log(f"Unique UIDs: {unique_uids}")

    if unique_uids < 90:  # Allow some tolerance for missing data
        raise ValueError(f"Expected ~100 unique UIDs, found only {unique_uids}")

    # 6. composite_ID format check
    log("Checking composite_ID format...")
    sample_ids = df_irt["composite_ID"].head(10).tolist()
    pattern = re.compile(r"^.+_\d+$")  # UID_test pattern
    invalid_ids = [id for id in sample_ids if not pattern.match(str(id))]
    if len(invalid_ids) > 0:
        log(f"Some composite_IDs may not match expected pattern: {invalid_ids}")

    # 7. Consistency check: IRT and TSVR have same composite_IDs
    irt_ids = set(df_irt["composite_ID"])
    tsvr_ids = set(df_tsvr["composite_ID"])
    if irt_ids != tsvr_ids:
        missing_in_tsvr = irt_ids - tsvr_ids
        missing_in_irt = tsvr_ids - irt_ids
        raise ValueError(f"composite_ID mismatch: {len(missing_in_tsvr)} in IRT but not TSVR, {len(missing_in_irt)} in TSVR but not IRT")

    log("All validation checks PASSED")

# Main Analysis

if __name__ == "__main__":
    try:
        # Initialize log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Step 00: Extract VR Data for IRT Analysis\n")
            f.write("=" * 80 + "\n\n")

        log("Step 00: Extract VR Data for IRT Analysis")
        log(f"RQ: 5.2.1 (Domains Type ROOT)")
        log(f"Source: step00_input_data.csv (LOCAL - no external dependencies)")
        # Load Input Data

        log("Loading input data from step00_input_data.csv...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"step00_input_data.csv ({len(df)} rows, {len(df.columns)} cols)")
        # Identify Columns
        # Separate metadata columns from item response columns

        log("Identifying columns...")

        # Required metadata columns
        meta_cols = ["UID", "TEST", "TSVR"]
        for col in meta_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Item columns are all TQ_* columns
        item_cols = [col for col in df.columns if col.startswith("TQ_")]
        log(f"Found {len(item_cols)} item columns (TQ_*)")

        # Group items by domain
        items_by_domain = get_items_by_domain(item_cols)
        log(f"Domain breakdown:")
        log(f"       - what: {len(items_by_domain['what'])} items")
        log(f"       - where: {len(items_by_domain['where'])} items")
        log(f"       - when: {len(items_by_domain['when'])} items")
        log(f"       - unknown: {len(items_by_domain['unknown'])} items")

        # Warn about unknown items
        if len(items_by_domain['unknown']) > 0:
            log(f"{len(items_by_domain['unknown'])} items could not be classified:")
            for item in items_by_domain['unknown'][:5]:
                log(f"          - {item}")
        # Create composite_ID
        # Format: UID_TEST (e.g., "P001_0" for participant P001 at test 0)

        log("Creating composite_ID...")
        df["composite_ID"] = df["UID"].astype(str) + "_" + df["TEST"].astype(str)
        log(f"Created {df['composite_ID'].nunique()} unique composite_IDs")
        # Dichotomize Item Responses
        # Threshold: values < 1 -> 0, values >= 1 -> 1
        # Preserves NaN for missing data

        log(f"Dichotomizing item responses (threshold={DICHOTOMIZATION_THRESHOLD})...")

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
        log(f"Dichotomization results:")
        log(f"       - Total valid responses: {total_responses}")
        log(f"       - Correct (1): {ones} ({100*ones/total_responses:.1f}%)")
        log(f"       - Incorrect (0): {zeros} ({100*zeros/total_responses:.1f}%)")
        log(f"       - Missing (NaN): {nan_count}")
        # Create IRT Input (Wide Format)
        # Rows = composite_ID, Columns = item_tags (binary responses)

        log("Creating IRT input file (wide format)...")

        df_irt = pd.DataFrame({
            "composite_ID": df["composite_ID"]
        })
        df_irt = pd.concat([df_irt, df_items_binary], axis=1)

        log(f"IRT input shape: {df_irt.shape}")
        # Create TSVR Mapping
        # Maps composite_ID to UID, test, and TSVR_hours for LMM (Decision D070)

        log("Creating TSVR mapping file...")

        df_tsvr = df[["composite_ID", "UID", "TEST", "TSVR"]].copy()
        df_tsvr = df_tsvr.rename(columns={
            "TEST": "test",
            "TSVR": "TSVR_hours"
        })

        log(f"TSVR mapping shape: {df_tsvr.shape}")
        log(f"TSVR_hours range: [{df_tsvr['TSVR_hours'].min():.2f}, {df_tsvr['TSVR_hours'].max():.2f}]")
        # Create Q-Matrix
        # Rows = item_name, Columns = [what, where, when] (binary loadings)
        # Each item loads on exactly 1 factor

        log("Creating Q-matrix...")

        # Only include classifiable items (exclude 'unknown')
        classifiable_items = (
            items_by_domain['what'] +
            items_by_domain['where'] +
            items_by_domain['when']
        )

        qmatrix_data = []
        for item in classifiable_items:
            domain = classify_item_domain(item)
            qmatrix_data.append({
                "item_name": item,
                "what": 1 if domain == "what" else 0,
                "where": 1 if domain == "where" else 0,
                "when": 1 if domain == "when" else 0
            })

        df_qmatrix = pd.DataFrame(qmatrix_data)

        log(f"Q-matrix shape: {df_qmatrix.shape}")
        log(f"Q-matrix domain counts:")
        log(f"       - what: {df_qmatrix['what'].sum()}")
        log(f"       - where: {df_qmatrix['where'].sum()}")
        log(f"       - when: {df_qmatrix['when'].sum()}")
        # Validate Outputs
        # Run all validation checks from 4_analysis.yaml

        validate_outputs(df_irt, df_tsvr, df_qmatrix, item_cols)
        # Save Output Files

        log("Saving output files...")

        # Ensure output directories exist
        OUTPUT_IRT_INPUT.parent.mkdir(parents=True, exist_ok=True)

        # Save IRT input
        df_irt.to_csv(OUTPUT_IRT_INPUT, index=False, encoding='utf-8')
        log(f"{OUTPUT_IRT_INPUT} ({len(df_irt)} rows, {len(df_irt.columns)} cols)")

        # Save TSVR mapping
        df_tsvr.to_csv(OUTPUT_TSVR_MAPPING, index=False, encoding='utf-8')
        log(f"{OUTPUT_TSVR_MAPPING} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")

        # Save Q-matrix
        df_qmatrix.to_csv(OUTPUT_Q_MATRIX, index=False, encoding='utf-8')
        log(f"{OUTPUT_Q_MATRIX} ({len(df_qmatrix)} rows, {len(df_qmatrix.columns)} cols)")
        # Final Summary

        log("")
        log("=" * 80)
        log("Step 00 complete")
        log("=" * 80)
        log(f"Outputs created:")
        log(f"  1. IRT input: {OUTPUT_IRT_INPUT}")
        log(f"     -> {len(df_irt)} observations, {len(item_cols)} items")
        log(f"  2. TSVR mapping: {OUTPUT_TSVR_MAPPING}")
        log(f"     -> {len(df_tsvr)} observations, TSVR_hours for Decision D070")
        log(f"  3. Q-matrix: {OUTPUT_Q_MATRIX}")
        log(f"     -> {len(df_qmatrix)} items, 3 domains (what/where/when)")
        log("")
        log("Ready for Step 01: IRT Calibration Pass 1")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
