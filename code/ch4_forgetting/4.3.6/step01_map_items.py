#!/usr/bin/env python3
"""Map Items by Paradigm (Retained vs Removed): Map purified items to paradigms (IFR/ICR/IRE) and create retention summary"""

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
LOG_FILE = RQ_DIR / "logs" / "step01_map_items.log"

# Domain to paradigm mapping
DOMAIN_TO_PARADIGM = {
    'free_recall': 'IFR',
    'cued_recall': 'ICR',
    'recognition': 'IRE'
}

# Original item counts from RQ 5.3.1 (before purification)
ORIGINAL_ITEM_COUNTS = {
    'IFR': 24,
    'ICR': 24,
    'IRE': 24
}


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Map Items by Paradigm (Retained vs Removed)")
        # Load Purified Items

        log("Loading purified items...")
        input_path = RQ_DIR / "data" / "step00_purified_items.csv"
        df_purified = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df_purified)} rows, {len(df_purified.columns)} cols)")

        # Verify expected columns
        expected_cols = ['item', 'domain', 'Discrimination', 'Difficulty_1']
        if list(df_purified.columns) != expected_cols:
            log(f"Column mismatch. Expected: {expected_cols}, Got: {list(df_purified.columns)}")
            sys.exit(1)
        # Map Domain to Paradigm
        #               to paradigm codes (IFR, ICR, IRE)

        log("Mapping domains to paradigms...")
        df_purified['paradigm'] = df_purified['domain'].map(DOMAIN_TO_PARADIGM)

        # Check for unmapped domains
        unmapped = df_purified['paradigm'].isna().sum()
        if unmapped > 0:
            log(f"{unmapped} items have unmapped domains")
            log(f"Unmapped domains: {df_purified[df_purified['paradigm'].isna()]['domain'].unique()}")
            sys.exit(1)

        log(f"Mapped {len(df_purified)} items to paradigms")
        log(f"Paradigm distribution: {df_purified['paradigm'].value_counts().to_dict()}")
        # Create Item Mapping Table
        # Output: data/step01_item_mapping.csv
        # Contains: One row per retained item with paradigm, IRT parameters
        # Columns: item_name, paradigm, retained, a, b, exclusion_reason

        log("Creating item mapping table...")
        df_mapping = pd.DataFrame({
            'item_name': df_purified['item'],
            'paradigm': df_purified['paradigm'],
            'retained': True,  # All items in this list were retained after purification
            'a': df_purified['Discrimination'],  # IRT discrimination parameter
            'b': df_purified['Difficulty_1'],    # IRT difficulty parameter
            'exclusion_reason': 'retained'       # All passed purification
        })

        output_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        df_mapping.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_mapping)} rows, {len(df_mapping.columns)} cols)")
        # Create Retention Summary
        # Output: data/step01_retention_summary.csv
        # Contains: 3 rows (IFR, ICR, IRE) with retention statistics
        # Columns: paradigm, total_items, retained_items, removed_items, retention_rate

        log("Creating retention summary...")

        # Count retained items per paradigm
        retained_counts = df_purified['paradigm'].value_counts().to_dict()

        # Build summary table
        summary_data = []
        for paradigm in ['IFR', 'ICR', 'IRE']:
            total = ORIGINAL_ITEM_COUNTS[paradigm]
            retained = retained_counts.get(paradigm, 0)
            removed = total - retained
            retention_rate = retained / total if total > 0 else 0.0

            summary_data.append({
                'paradigm': paradigm,
                'total_items': total,
                'retained_items': retained,
                'removed_items': removed,
                'retention_rate': retention_rate
            })

        df_summary = pd.DataFrame(summary_data)
        output_path = RQ_DIR / "data" / "step01_retention_summary.csv"
        df_summary.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_summary)} rows, {len(df_summary.columns)} cols)")

        # Log retention statistics
        for _, row in df_summary.iterrows():
            log(f"{row['paradigm']}: {row['retained_items']}/{row['total_items']} retained ({row['retention_rate']:.1%})")
        # Run Validation
        # Validates: All 3 paradigms present, retention rates valid, no NaNs
        # Threshold: retention_rate in [0, 1], retained + removed = total

        log("Validating outputs...")

        # Check 1: All 3 paradigms present
        paradigms = set(df_summary['paradigm'])
        if paradigms != {'IFR', 'ICR', 'IRE'}:
            log(f"[VALIDATION FAIL] Expected paradigms ['IFR', 'ICR', 'IRE'], got {paradigms}")
            sys.exit(1)
        log("[VALIDATION PASS] All 3 paradigms present")

        # Check 2: retention_rate in [0, 1]
        invalid_rates = df_summary[(df_summary['retention_rate'] < 0) | (df_summary['retention_rate'] > 1)]
        if len(invalid_rates) > 0:
            log(f"[VALIDATION FAIL] Invalid retention rates: {invalid_rates['retention_rate'].tolist()}")
            sys.exit(1)
        log("[VALIDATION PASS] All retention rates in [0, 1]")

        # Check 3: retained + removed = total
        df_summary['sum_check'] = df_summary['retained_items'] + df_summary['removed_items']
        mismatch = df_summary[df_summary['sum_check'] != df_summary['total_items']]
        if len(mismatch) > 0:
            log(f"[VALIDATION FAIL] retained + removed != total for: {mismatch['paradigm'].tolist()}")
            sys.exit(1)
        log("[VALIDATION PASS] retained + removed = total for all paradigms")

        # Check 4: No NaN values
        if df_mapping.isna().any().any():
            log("[VALIDATION FAIL] NaN values found in item mapping")
            log(f"Columns with NaN: {df_mapping.columns[df_mapping.isna().any()].tolist()}")
            sys.exit(1)
        if df_summary.isna().any().any():
            log("[VALIDATION FAIL] NaN values found in retention summary")
            log(f"Columns with NaN: {df_summary.columns[df_summary.isna().any()].tolist()}")
            sys.exit(1)
        log("[VALIDATION PASS] No NaN values in outputs")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
