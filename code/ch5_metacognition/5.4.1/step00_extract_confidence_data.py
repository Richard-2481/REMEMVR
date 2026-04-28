#!/usr/bin/env python3
"""extract_confidence_data: Extract TC_* confidence items from dfData.csv, filter to interactive paradigms"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.4.1
LOG_FILE = RQ_DIR / "logs" / "step00_extract_confidence_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Paradigm Assignment Function

def assign_paradigm_from_tag(item_name: str) -> str:
    """
    Parse paradigm from TC_* column name based on embedded tags.

    Paradigm tags:
      - IFR: Interactive Free Recall
      - ICR: Interactive Cued Recall
      - IRE: Interactive Recognition

    Args:
        item_name: Column name like TC_IFR-N-i1

    Returns:
        Paradigm string: "IFR", "ICR", or "IRE"

    Raises:
        ValueError if paradigm tag not recognized
    """
    if 'IFR' in item_name:
        return "IFR"
    elif 'ICR' in item_name:
        return "ICR"
    elif 'IRE' in item_name:
        return "IRE"
    else:
        raise ValueError(f"Cannot determine paradigm for item: {item_name}")

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Extract Confidence Data (Paradigm-Based)")
        # Load Wide-Format Data

        log("Loading data/cache/dfData.csv...")
        df_raw = pd.read_csv(PROJECT_ROOT / "data" / "cache" / "dfData.csv")
        log(f"dfData.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Verify required columns exist
        required_cols = ['UID', 'TEST', 'TSVR']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Required columns present: {required_cols}")
        # Filter to Interactive Paradigm TC_* Columns
        # Filter: TC_* columns containing IFR, ICR, or IRE paradigms

        log("Filtering to interactive paradigm TC_* columns...")

        # Get all TC_* columns
        tc_cols = [col for col in df_raw.columns if col.startswith('TC_')]
        log(f"{len(tc_cols)} total TC_* columns")

        # Filter to interactive paradigms (IFR, ICR, IRE)
        interactive_paradigms = ['IFR', 'ICR', 'IRE']
        tc_interactive = [col for col in tc_cols
                         if any(paradigm in col for paradigm in interactive_paradigms)]

        log(f"{len(tc_interactive)} interactive TC_* columns (IFR, ICR, IRE)")
        log(f"IFR: {sum(1 for c in tc_interactive if 'IFR' in c)} items")
        log(f"ICR: {sum(1 for c in tc_interactive if 'ICR' in c)} items")
        log(f"IRE: {sum(1 for c in tc_interactive if 'IRE' in c)} items")
        # Create Composite ID
        # Format: {UID}_T{test_number}
        # Example: A010_T1, A010_T2, A010_T3, A010_T4

        log("Creating composite_ID from UID and TEST...")

        # Ensure TEST is integer for T1/T2/T3/T4 formatting
        df_raw['test_int'] = df_raw['TEST'].astype(int)
        df_raw['composite_ID'] = df_raw['UID'] + '_T' + df_raw['test_int'].astype(str)

        log(f"composite_ID for {len(df_raw)} observations")
        log(f"First 3 composite_IDs: {df_raw['composite_ID'].head(3).tolist()}")
        # Create IRT Input (Wide Format)
        # Output: composite_ID + TC_* items

        log("Creating IRT input wide-format DataFrame...")

        irt_input_cols = ['composite_ID'] + tc_interactive
        df_irt_input = df_raw[irt_input_cols].copy()

        log(f"IRT input: {len(df_irt_input)} rows x {len(df_irt_input.columns)} cols")
        # CRITICAL FIX: Convert fractional values to integers for IRT
        # IRT GRM expects integer categories 0, 1, 2, 3, 4 for 5-category ordinal data
        # But dfData.csv contains fractional values: 0.2, 0.4, 0.6, 0.8, 1.0
        # This mapping preserves the ordinal structure while providing correct input for IRT

        log("Converting fractional confidence values to integers for IRT...")
        value_mapping = {
            0.2: 0,   # Lowest confidence -> 0
            0.4: 1,
            0.6: 2,   # Middle confidence -> 2
            0.8: 3,
            1.0: 4    # Highest confidence -> 4
        }

        # Apply conversion to all TC_* columns
        for col in tc_interactive:
            df_irt_input[col] = df_irt_input[col].map(lambda x: value_mapping.get(x, x) if pd.notna(x) else x)

        log(f"Values mapped: 0.2→0, 0.4→1, 0.6→2, 0.8→3, 1.0→4")
        # ADAPTIVE n_cats DETECTION
        # Detect actual unique non-NaN values in extracted TC_* items
        #
        # Background: After integer conversion, TC_* values are {0, 1, 2, 3, 4}
        # This is 5 categories for GRM (n_cats=5)
        #
        # Detection logic:
        #   - Collect all non-NaN unique values across all TC_* columns
        #   - Sort and report
        #   - Compute n_cats = len(unique_values)
        #   - Print recommendation for Step 01 parameters

        log("Detecting actual unique TC_* values for n_cats determination...")

        all_tc_values = set()
        for col in tc_interactive:
            col_values = df_irt_input[col].dropna().unique()
            all_tc_values.update(col_values)

        # Sort unique values for display
        unique_sorted = sorted(all_tc_values)
        n_cats_detected = len(unique_sorted)

        log(f"Unique TC_* values found: {unique_sorted}")
        log(f"Number of categories (n_cats): {n_cats_detected}")

        # Compute number of thresholds (k = n_cats - 1)
        n_thresholds = n_cats_detected - 1
        log(f"Number of thresholds for GRM: {n_thresholds} (b1 to b{n_thresholds})")

        # Print recommendation for Step 01
        log("")
        log("=" * 80)
        log("RECOMMENDATION FOR STEP 01 (IRT Calibration):")
        log(f"  Set n_cats = {n_cats_detected} in 4_analysis.yaml Step 01 parameters")
        log(f"  This will create {n_thresholds} thresholds: b1, b2, ..., b{n_thresholds}")
        log(f"  Item parameter CSV will have columns: item_name, factor, a, b1, b2, ..., b{n_thresholds}")
        log("=" * 80)
        log("")
        # Create TSVR Time Mapping
        # Output: composite_ID -> TSVR_hours (actual hours since encoding per D070)

        log("Creating TSVR time mapping...")

        df_tsvr = df_raw[['composite_ID', 'UID', 'TSVR', 'test_int']].copy()
        df_tsvr.rename(columns={'TSVR': 'TSVR_hours', 'test_int': 'test'}, inplace=True)

        # Convert test to T1/T2/T3/T4 format for consistency
        df_tsvr['test'] = 'T' + df_tsvr['test'].astype(str)

        log(f"TSVR mapping: {len(df_tsvr)} rows")
        log(f"[TSVR RANGE] Min: {df_tsvr['TSVR_hours'].min():.2f} hours, Max: {df_tsvr['TSVR_hours'].max():.2f} hours")
        # Create Q-Matrix (3-Factor Structure BY PARADIGM)
        # Assign each TC_* item to paradigm (IFR/ICR/IRE) based on embedded tags
        # Q-matrix encoding: IFR=factor1, ICR=factor2, IRE=factor3
        #
        # KEY DIFFERENCE from RQ 6.3.1: Q-matrix assigns by PARADIGM (not domain)
        # RQ 6.3.1: Q-matrix by domain (What/Where/When) for domain-based confidence
        # RQ 6.4.1: Q-matrix by paradigm (IFR/ICR/IRE) for paradigm-based confidence

        log("Creating 3-factor Q-matrix BY PARADIGM...")

        q_matrix_data = []
        for item in tc_interactive:
            paradigm = assign_paradigm_from_tag(item)

            # Create binary loadings for 3-factor structure
            # Each item loads on exactly ONE factor (its paradigm)
            q_matrix_data.append({
                'item_name': item,
                'factor1_IFR': 1 if paradigm == 'IFR' else 0,
                'factor2_ICR': 1 if paradigm == 'ICR' else 0,
                'factor3_IRE': 1 if paradigm == 'IRE' else 0
            })

        df_q_matrix = pd.DataFrame(q_matrix_data)

        log(f"Q-matrix: {len(df_q_matrix)} items")
        log(f"IFR: {df_q_matrix['factor1_IFR'].sum()} items")
        log(f"ICR: {df_q_matrix['factor2_ICR'].sum()} items")
        log(f"IRE: {df_q_matrix['factor3_IRE'].sum()} items")
        # Save Outputs
        # These outputs will be used by:
        #   - step00_irt_input.csv -> Step 1 (IRT Pass 1 calibration)
        #   - step00_tsvr_mapping.csv -> Step 4 (merge with theta scores)
        #   - step00_q_matrix.csv -> Step 1 (factor structure definition)

        log("Saving outputs...")

        # Save IRT input
        irt_input_path = RQ_DIR / "data" / "step00_irt_input.csv"
        df_irt_input.to_csv(irt_input_path, index=False, encoding='utf-8')
        log(f"{irt_input_path.name} ({len(df_irt_input)} rows, {len(df_irt_input.columns)} cols)")

        # Save TSVR mapping
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        df_tsvr.to_csv(tsvr_path, index=False, encoding='utf-8')
        log(f"{tsvr_path.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")

        # Save Q-matrix
        q_matrix_path = RQ_DIR / "data" / "step00_q_matrix.csv"
        df_q_matrix.to_csv(q_matrix_path, index=False, encoding='utf-8')
        log(f"{q_matrix_path.name} ({len(df_q_matrix)} rows, {len(df_q_matrix.columns)} cols)")
        # Validation
        # Validate outputs meet expected criteria from 4_analysis.yaml

        log("Running inline validation checks...")

        validation_passed = True

        # Check 1: All output files exist
        if not irt_input_path.exists() or not tsvr_path.exists() or not q_matrix_path.exists():
            log("Not all output files created")
            validation_passed = False
        else:
            log("All 3 output files exist")

        # Check 2: Expected dimensions for IRT input
        if len(df_irt_input) != 400:
            log(f"IRT input rows: expected 400, got {len(df_irt_input)}")
            validation_passed = False
        else:
            log(f"IRT input dimensions: 400 rows x {len(df_irt_input.columns)} cols")

        # Check 3: TSVR_hours range (actual hours, not nominal days - can exceed 168)
        tsvr_min = df_tsvr['TSVR_hours'].min()
        tsvr_max = df_tsvr['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 300:
            log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] outside [0, 300]")
            validation_passed = False
        else:
            log(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}] within [0, 300]")

        # Check 4: All 3 paradigms represented in Q-matrix
        paradigms_ifr = df_q_matrix['factor1_IFR'].sum()
        paradigms_icr = df_q_matrix['factor2_ICR'].sum()
        paradigms_ire = df_q_matrix['factor3_IRE'].sum()

        if paradigms_ifr == 0 or paradigms_icr == 0 or paradigms_ire == 0:
            log(f"Q-matrix missing paradigm representation: IFR={paradigms_ifr}, ICR={paradigms_icr}, IRE={paradigms_ire}")
            validation_passed = False
        else:
            log(f"Q-matrix has all 3 paradigms (IFR={paradigms_ifr}, ICR={paradigms_icr}, IRE={paradigms_ire})")

        # Check 5: Each item loads on exactly ONE factor
        row_sums = df_q_matrix[['factor1_IFR', 'factor2_ICR', 'factor3_IRE']].sum(axis=1)
        if not (row_sums == 1).all():
            log(f"Some items load on multiple factors (expected 1 per item)")
            validation_passed = False
        else:
            log(f"Each item loads on exactly one factor")

        # Check 6: Missing data report (informational)
        missing_pct_per_item = df_irt_input[tc_interactive].isnull().mean() * 100
        high_missing_items = missing_pct_per_item[missing_pct_per_item > 10].sort_values(ascending=False)

        if len(high_missing_items) > 0:
            log(f"{len(high_missing_items)} items have >10% missing data:")
            for item, pct in high_missing_items.head(10).items():
                log(f"  - {item}: {pct:.1f}% missing")
        else:
            log(f"All items have <10% missing data")

        # Overall validation result
        if validation_passed:
            log("All critical checks PASSED")
        else:
            log("Some checks FAILED - see details above")
            sys.exit(1)

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
