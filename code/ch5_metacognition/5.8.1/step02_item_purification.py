#!/usr/bin/env python3
"""Item Purification (Decision D039): Filter TC_* confidence items by quality thresholds per Decision D039 2-pass IRT workflow."""

import sys
from pathlib import Path
import pandas as pd
from typing import Tuple
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

from tools.validation import validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.1
LOG_FILE = RQ_DIR / "logs" / "step02_item_purification.log"

# Decision D039 thresholds
A_THRESHOLD = 0.4  # Minimum discrimination
B_THRESHOLD = 3.0  # Maximum absolute difficulty


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Ensure immediate write (from step01 pattern)
    print(msg, flush=True)  # Unbuffered console output

# Main Analysis

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 02: Item Purification (Decision D039)")
        log("=" * 80)
        # Load Pass 1 Item Parameters

        log("\nLoading Pass 1 item parameters...")
        input_path = RQ_DIR / "data" / "step01_pass1_item_params.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_items_pass1 = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name}")
        log(f"  Rows: {len(df_items_pass1)} items")
        log(f"  Columns: {list(df_items_pass1.columns)}")
        log(f"  Difficulty range: [{df_items_pass1['Difficulty'].min():.2f}, {df_items_pass1['Difficulty'].max():.2f}]")
        log(f"  Overall_Discrimination range: [{df_items_pass1['Overall_Discrimination'].min():.2f}, {df_items_pass1['Overall_Discrimination'].max():.2f}]")
        # Run Item Purification

        log("\nRunning filter_items_by_quality...")
        log(f"  Thresholds: a >= {A_THRESHOLD}, |b| <= {B_THRESHOLD}")

        df_purified, df_excluded = filter_items_by_quality(
            df_items=df_items_pass1,
            a_threshold=A_THRESHOLD,
            b_threshold=B_THRESHOLD
        )

        retention_pct = 100 * len(df_purified) / len(df_items_pass1)
        exclusion_pct = 100 * len(df_excluded) / len(df_items_pass1)

        log(f"Purification complete")
        log(f"  Retained: {len(df_purified)} items ({retention_pct:.1f}%)")
        log(f"  Excluded: {len(df_excluded)} items ({exclusion_pct:.1f}%)")

        # Check if retention rate is within expected range (20-80%)
        if retention_pct < 20 or retention_pct > 80:
            log(f"Retention rate {retention_pct:.1f}% outside typical 20-80% range")
            log(f"          This may indicate calibration problems in Pass 1")
        # Save Purification Outputs
        # Output 1: Purified items (retained for Pass 2)
        # Output 2: Excluded items (with exclusion reasons)

        log("\nSaving purification results...")

        # Save purified items
        purified_path = RQ_DIR / "data" / "step02_purified_items.csv"
        df_purified.to_csv(purified_path, index=False, encoding='utf-8')
        log(f"{purified_path.name}")
        log(f"  Rows: {len(df_purified)} retained items")
        log(f"  Columns: {list(df_purified.columns)}")

        # Save excluded items
        excluded_path = RQ_DIR / "data" / "step02_excluded_items.csv"
        df_excluded.to_csv(excluded_path, index=False, encoding='utf-8')
        log(f"{excluded_path.name}")
        log(f"  Rows: {len(df_excluded)} excluded items")
        log(f"  Columns: {list(df_excluded.columns)}")

        # Domain-wise breakdown
        if 'factor' in df_purified.columns:
            log("\n[DOMAIN BREAKDOWN]")
            log("  Retained items by domain:")
            for domain in df_purified['factor'].unique():
                n_domain = len(df_purified[df_purified['factor'] == domain])
                log(f"    {domain}: {n_domain} items")

            if 'factor' in df_excluded.columns:
                log("  Excluded items by domain:")
                for domain in df_excluded['factor'].unique():
                    n_domain = len(df_excluded[df_excluded['factor'] == domain])
                    log(f"    {domain}: {n_domain} items")
        # Validate Purification Results
        # Validates: All retained items meet thresholds (a >= 0.4, |b| <= 3.0)
        # Threshold: 100% compliance required

        log("\nValidating purified items...")

        validation_result = validate_irt_parameters(
            df_items=df_purified,
            a_min=A_THRESHOLD,
            b_max=B_THRESHOLD,
            a_col='a',  # filter_items_by_quality normalizes column names
            b_col='b'
        )

        if validation_result.get('valid', True):
            log("[VALIDATION PASSED]")
            if 'message' in validation_result:
                log(f"  {validation_result['message']}")
            log(f"  Valid items: {validation_result.get('n_valid', len(df_purified))} / {validation_result.get('n_items', len(df_purified))}")
        else:
            log("[VALIDATION FAILED]")
            if 'message' in validation_result:
                log(f"  {validation_result['message']}")
            log(f"  Invalid items: {validation_result.get('n_invalid', 0)} / {validation_result.get('n_items', len(df_purified))}")
            if validation_result.get('invalid_items'):
                log(f"  Invalid items: {validation_result['invalid_items']}")
            raise ValueError("Purification validation failed - some items don't meet thresholds")
        # FINAL SUMMARY

        log("\n" + "=" * 80)
        log("Step 02 complete")
        log("=" * 80)
        log(f"Pass 1 items: {len(df_items_pass1)}")
        log(f"Retained (Pass 2): {len(df_purified)} ({retention_pct:.1f}%)")
        log(f"Excluded: {len(df_excluded)} ({exclusion_pct:.1f}%)")
        log(f"Thresholds: a >= {A_THRESHOLD}, |b| <= {B_THRESHOLD}")
        log("\nNext: Step 03 will recalibrate using {len(df_purified)} purified items")
        log("=" * 80)

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("\nFull error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
            f.flush()
        traceback.print_exc()
        sys.exit(1)
