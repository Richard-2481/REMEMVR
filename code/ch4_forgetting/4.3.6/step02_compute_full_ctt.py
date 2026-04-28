#!/usr/bin/env python3
"""Compute Full CTT Scores (All Items Pre-Purification): Compute CTT (Classical Test Theory) scores using ALL 72 paradigm items (24 per paradigm)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.6
LOG_FILE = RQ_DIR / "logs" / "step02_compute_full_ctt.log"

# All 72 paradigm items (24 per paradigm) from RQ 5.3.1 step00_irt_input.csv
# These are ALL items before IRT purification
ALL_IFR_ITEMS = [
    'TQ_IFR-N-i1', 'TQ_IFR-N-i2', 'TQ_IFR-N-i3', 'TQ_IFR-N-i4', 'TQ_IFR-N-i5', 'TQ_IFR-N-i6',
    'TQ_IFR-U-i1', 'TQ_IFR-U-i2', 'TQ_IFR-U-i3', 'TQ_IFR-U-i4', 'TQ_IFR-U-i5', 'TQ_IFR-U-i6',
    'TQ_IFR-D-i1', 'TQ_IFR-D-i2', 'TQ_IFR-D-i3', 'TQ_IFR-D-i4', 'TQ_IFR-D-i5', 'TQ_IFR-D-i6',
    'TQ_IFR-O-i1', 'TQ_IFR-O-i2', 'TQ_IFR-O-i3', 'TQ_IFR-O-i4', 'TQ_IFR-O-i5', 'TQ_IFR-O-i6'
]

ALL_ICR_ITEMS = [
    'TQ_ICR-N-i1', 'TQ_ICR-N-i2', 'TQ_ICR-N-i3', 'TQ_ICR-N-i4', 'TQ_ICR-N-i5', 'TQ_ICR-N-i6',
    'TQ_ICR-U-i1', 'TQ_ICR-U-i2', 'TQ_ICR-U-i3', 'TQ_ICR-U-i4', 'TQ_ICR-U-i5', 'TQ_ICR-U-i6',
    'TQ_ICR-D-i1', 'TQ_ICR-D-i2', 'TQ_ICR-D-i3', 'TQ_ICR-D-i4', 'TQ_ICR-D-i5', 'TQ_ICR-D-i6',
    'TQ_ICR-O-i1', 'TQ_ICR-O-i2', 'TQ_ICR-O-i3', 'TQ_ICR-O-i4', 'TQ_ICR-O-i5', 'TQ_ICR-O-i6'
]

ALL_IRE_ITEMS = [
    'TQ_IRE-N-i1', 'TQ_IRE-N-i2', 'TQ_IRE-N-i3', 'TQ_IRE-N-i4', 'TQ_IRE-N-i5', 'TQ_IRE-N-i6',
    'TQ_IRE-U-i1', 'TQ_IRE-U-i2', 'TQ_IRE-U-i3', 'TQ_IRE-U-i4', 'TQ_IRE-U-i5', 'TQ_IRE-U-i6',
    'TQ_IRE-D-i1', 'TQ_IRE-D-i2', 'TQ_IRE-D-i3', 'TQ_IRE-D-i4', 'TQ_IRE-D-i5', 'TQ_IRE-D-i6',
    'TQ_IRE-O-i1', 'TQ_IRE-O-i2', 'TQ_IRE-O-i3', 'TQ_IRE-O-i4', 'TQ_IRE-O-i5', 'TQ_IRE-O-i6'
]


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute Full CTT Scores (All Items Pre-Purification)")
        # Load Raw Response Data

        log("Loading raw response data from dfData.csv...")
        raw_data_path = RQ_DIR / "data/cache/dfData.csv"

        if not raw_data_path.exists():
            raise FileNotFoundError(f"dfData.csv not found at {raw_data_path}")

        df_raw = pd.read_csv(raw_data_path, encoding='utf-8')
        log(f"dfData.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Verify required columns exist
        required_base_cols = ['UID', 'TEST']
        missing_base = [col for col in required_base_cols if col not in df_raw.columns]
        if missing_base:
            raise ValueError(f"Missing required columns: {missing_base}")

        log(f"Found UID and TEST columns")
        # Verify All 72 Paradigm Items Exist
        # Check that all expected item columns are present in dfData.csv

        log("Verifying all 72 paradigm items exist in dfData.csv...")

        all_items = ALL_IFR_ITEMS + ALL_ICR_ITEMS + ALL_IRE_ITEMS
        missing_items = [item for item in all_items if item not in df_raw.columns]

        if missing_items:
            log(f"Missing {len(missing_items)} items: {missing_items[:5]}...")
            raise ValueError(f"Missing {len(missing_items)} expected paradigm items in dfData.csv")

        log(f"All 72 paradigm items found (24 IFR + 24 ICR + 24 IRE)")
        # Compute Full CTT Scores Per Paradigm
        # For each paradigm (IFR, ICR, IRE):
        #   - Extract all 24 items for that paradigm
        #   - Compute mean proportion correct per UID × TEST
        #   - Result: CTT_full_[paradigm] in range [0, 1]

        log("Computing Full CTT scores using all items per paradigm...")

        # Create result DataFrame with UID and test columns
        # Map TEST values (1,2,3,4) to test labels (T1,T2,T3,T4)
        df_result = df_raw[['UID', 'TEST']].copy()
        df_result['test'] = 'T' + df_result['TEST'].astype(str)
        df_result.drop(columns=['TEST'], inplace=True)

        # Compute CTT_full_IFR (mean of all 24 IFR items)
        log(f"CTT_full_IFR (24 items)...")
        ifr_responses = df_raw.copy()
        # Convert to numeric (handle any non-numeric values)
        ifr_responses = ifr_responses.apply(pd.to_numeric, errors='coerce')
        # Dichotomize: TQ < 1 → 0, TQ >= 1 → 1 (per specification)
        ifr_binary = (ifr_responses >= 1).astype(int)
        df_result['CTT_full_IFR'] = ifr_binary.mean(axis=1)
        log(f"CTT_full_IFR: mean={df_result['CTT_full_IFR'].mean():.3f}, range=[{df_result['CTT_full_IFR'].min():.3f}, {df_result['CTT_full_IFR'].max():.3f}]")

        # Compute CTT_full_ICR (mean of all 24 ICR items)
        log(f"CTT_full_ICR (24 items)...")
        icr_responses = df_raw.copy()
        icr_responses = icr_responses.apply(pd.to_numeric, errors='coerce')
        icr_binary = (icr_responses >= 1).astype(int)
        df_result['CTT_full_ICR'] = icr_binary.mean(axis=1)
        log(f"CTT_full_ICR: mean={df_result['CTT_full_ICR'].mean():.3f}, range=[{df_result['CTT_full_ICR'].min():.3f}, {df_result['CTT_full_ICR'].max():.3f}]")

        # Compute CTT_full_IRE (mean of all 24 IRE items)
        log(f"CTT_full_IRE (24 items)...")
        ire_responses = df_raw.copy()
        ire_responses = ire_responses.apply(pd.to_numeric, errors='coerce')
        ire_binary = (ire_responses >= 1).astype(int)
        df_result['CTT_full_IRE'] = ire_binary.mean(axis=1)
        log(f"CTT_full_IRE: mean={df_result['CTT_full_IRE'].mean():.3f}, range=[{df_result['CTT_full_IRE'].min():.3f}, {df_result['CTT_full_IRE'].max():.3f}]")
        # Validate CTT Scores
        # Criteria:
        #   - All CTT scores in [0, 1]
        #   - No NaN values
        #   - No duplicate UID × test combinations
        #   - Expected 400 rows (100 participants × 4 tests)

        log("Validating CTT scores...")

        # Check score ranges
        ctt_cols = ['CTT_full_IFR', 'CTT_full_ICR', 'CTT_full_IRE']
        for col in ctt_cols:
            if df_result[col].min() < 0 or df_result[col].max() > 1:
                raise ValueError(f"{col} out of range [0, 1]: min={df_result[col].min()}, max={df_result[col].max()}")
            log(f"{col} in range [0, 1]")

        # Check for NaN values
        nan_counts = df_result[ctt_cols].isna().sum()
        if nan_counts.sum() > 0:
            raise ValueError(f"NaN values found: {nan_counts.to_dict()}")
        log(f"No NaN values in CTT columns")

        # Check for duplicates
        duplicates = df_result.duplicated(subset=['UID', 'test']).sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate UID × test combinations")
        log(f"No duplicate UID × test combinations")

        # Check row count
        expected_rows = 400  # 100 participants × 4 tests
        if len(df_result) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(df_result)}")
        else:
            log(f"{expected_rows} rows present (100 participants × 4 tests)")

        # Check all test sessions present
        test_sessions = sorted(df_result['test'].unique())
        expected_sessions = ['T1', 'T2', 'T3', 'T4']
        if test_sessions != expected_sessions:
            raise ValueError(f"Expected test sessions {expected_sessions}, found {test_sessions}")
        log(f"All 4 test sessions present: {test_sessions}")
        # Save Full CTT Scores
        # Output: data/step02_ctt_full_scores.csv (400 rows × 5 columns)

        output_path = RQ_DIR / "data/step02_ctt_full_scores.csv"
        log(f"Saving Full CTT scores to {output_path.name}...")

        # Reorder columns: UID, test, CTT_full_IFR, CTT_full_ICR, CTT_full_IRE
        df_result = df_result[['UID', 'test', 'CTT_full_IFR', 'CTT_full_ICR', 'CTT_full_IRE']]

        df_result.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_result)} rows, {len(df_result.columns)} cols)")
        # Summary Statistics

        log("Full CTT Score Statistics:")
        for col in ctt_cols:
            log(f"  {col}: mean={df_result[col].mean():.4f}, SD={df_result[col].std():.4f}, range=[{df_result[col].min():.4f}, {df_result[col].max():.4f}]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
