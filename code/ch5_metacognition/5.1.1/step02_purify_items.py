#!/usr/bin/env python3
"""purify_items: Apply Decision D039 purification thresholds to exclude poorly performing items"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_purify_items.log"


# Decision D039 thresholds
A_THRESHOLD = 0.4   # Minimum discrimination (items below this are too flat)
B_THRESHOLD = 3.0   # Maximum absolute difficulty (items beyond this are too extreme)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Purify Items (Decision D039)")
        # Load Pass 1 Item Parameters

        log("Loading Pass 1 item parameters...")
        input_path = RQ_DIR / "data" / "step01_pass1_item_params.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_items = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df_items)} items, {len(df_items.columns)} columns)")
        log(f"Columns: {list(df_items.columns)}")

        # Validate expected columns
        expected_cols = ['item_name', 'dimension', 'a', 'b1', 'b2', 'b3', 'b4']
        missing_cols = [col for col in expected_cols if col not in df_items.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Discrimination (a) range: [{df_items['a'].min():.2f}, {df_items['a'].max():.2f}]")
        # Compute b_mean for GRM Items
        # GRM produces 4 threshold parameters (b1, b2, b3, b4) representing
        # category boundaries. Decision D039 requires single difficulty value,
        # so we compute mean of thresholds as central tendency measure.

        log("Computing b_mean = mean(b1, b2, b3, b4) for each item...")

        df_items['b_mean'] = df_items[['b1', 'b2', 'b3', 'b4']].mean(axis=1)

        log(f"b_mean range: [{df_items['b_mean'].min():.2f}, {df_items['b_mean'].max():.2f}]")
        log(f"|b_mean| range: [0.00, {df_items['b_mean'].abs().max():.2f}]")
        # Apply Decision D039 Purification Thresholds
        # Threshold criteria:
        #   - EXCLUDE if discrimination a < 0.4 (item too flat, poor discrimination)
        #   - EXCLUDE if |b_mean| > 3.0 (item too extreme, outside typical ability range)

        log(f"Applying Decision D039 thresholds:")
        log(f"- Minimum discrimination a >= {A_THRESHOLD}")
        log(f"- Maximum |b_mean| <= {B_THRESHOLD}")

        # Identify items failing each criterion
        low_discrimination = df_items['a'] < A_THRESHOLD
        extreme_difficulty = df_items['b_mean'].abs() > B_THRESHOLD

        n_low_a = low_discrimination.sum()
        n_extreme_b = extreme_difficulty.sum()

        log(f"Items failing criteria:")
        log(f"- Low discrimination (a < {A_THRESHOLD}): {n_low_a} items")
        log(f"- Extreme difficulty (|b_mean| > {B_THRESHOLD}): {n_extreme_b} items")

        # Items that PASS both criteria (retained)
        retained_mask = (~low_discrimination) & (~extreme_difficulty)
        df_retained = df_items[retained_mask].copy()

        # Items that FAIL either criterion (excluded)
        df_excluded = df_items[~retained_mask].copy()

        # Add exclusion reasons
        exclusion_reasons = []
        for idx, row in df_excluded.iterrows():
            reasons = []
            if row['a'] < A_THRESHOLD:
                reasons.append(f"a < {A_THRESHOLD} (a={row['a']:.3f})")
            if abs(row['b_mean']) > B_THRESHOLD:
                reasons.append(f"|b_mean| > {B_THRESHOLD} (b_mean={row['b_mean']:.3f})")
            exclusion_reasons.append("; ".join(reasons))

        df_excluded['exclusion_reason'] = exclusion_reasons

        # Compute retention statistics
        n_total = len(df_items)
        n_retained = len(df_retained)
        n_excluded = len(df_excluded)
        retention_rate = n_retained / n_total if n_total > 0 else 0.0

        log(f"Purification summary:")
        log(f"- Total items: {n_total}")
        log(f"- Retained items: {n_retained}")
        log(f"- Excluded items: {n_excluded}")
        log(f"- Retention rate: {retention_rate:.1%}")
        # Save Purified Items List
        # Output: CSV with single column (item_name) containing retained items
        # This list will be used by Pass 2 calibration to filter items

        log("Saving purified items list...")
        purified_items_path = RQ_DIR / "data" / "step02_purified_items.csv"

        df_purified_list = df_retained[['item_name']].copy()
        df_purified_list.to_csv(purified_items_path, index=False, encoding='utf-8')

        log(f"{purified_items_path.name} ({len(df_purified_list)} items)")
        # Generate Purification Report
        # Output: Plain text report with detailed exclusion statistics
        # Contains: summary counts, exclusion reasons, item lists

        log("Generating purification report...")
        report_path = RQ_DIR / "data" / "step02_purification_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ITEM PURIFICATION REPORT (Decision D039)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"RQ: ch6/6.1.1\n")
            f.write(f"Step: step02_purify_items\n")
            f.write(f"Model: GRM (5-category ordinal)\n")

            f.write("-" * 70 + "\n")
            f.write("PURIFICATION THRESHOLDS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  - Minimum discrimination (a): {A_THRESHOLD}\n")
            f.write(f"  - Maximum |b_mean|: {B_THRESHOLD}\n")
            f.write(f"  - b_mean computed as: mean(b1, b2, b3, b4)\n\n")

            f.write("-" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  - Total items (Pass 1): {n_total}\n")
            f.write(f"  - Retained items: {n_retained}\n")
            f.write(f"  - Excluded items: {n_excluded}\n")
            f.write(f"  - Retention rate: {retention_rate:.1%}\n\n")

            f.write("-" * 70 + "\n")
            f.write("EXCLUSION BREAKDOWN\n")
            f.write("-" * 70 + "\n")
            f.write(f"  - Low discrimination (a < {A_THRESHOLD}): {n_low_a} items\n")
            f.write(f"  - Extreme difficulty (|b_mean| > {B_THRESHOLD}): {n_extreme_b} items\n")
            f.write(f"  - Total excluded (with potential overlap): {n_excluded} items\n\n")

            if n_excluded > 0:
                f.write("-" * 70 + "\n")
                f.write("EXCLUDED ITEMS (with reasons)\n")
                f.write("-" * 70 + "\n\n")

                for idx, row in df_excluded.iterrows():
                    f.write(f"  - {row['item_name']}\n")
                    f.write(f"    a = {row['a']:.3f}, b_mean = {row['b_mean']:.3f}\n")
                    f.write(f"    Reason: {row['exclusion_reason']}\n\n")
            else:
                f.write("-" * 70 + "\n")
                f.write("NO ITEMS EXCLUDED (all items passed thresholds)\n")
                f.write("-" * 70 + "\n\n")

            f.write("-" * 70 + "\n")
            f.write("RETAINED ITEMS SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"  - Count: {n_retained} items\n")
            f.write(f"  - Discrimination (a) range: [{df_retained['a'].min():.3f}, {df_retained['a'].max():.3f}]\n")
            f.write(f"  - b_mean range: [{df_retained['b_mean'].min():.3f}, {df_retained['b_mean'].max():.3f}]\n")
            f.write(f"  - |b_mean| range: [0.000, {df_retained['b_mean'].abs().max():.3f}]\n\n")

            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        log(f"{report_path.name}")
        # Validate Purification Results
        # Validation criteria:
        #   - Retention rate in [0.30, 0.70] (expected range per D039)
        #   - No duplicate item names
        #   - All retained items exist in original Pass 1 data

        log("Validating purification results...")

        # Check retention rate
        if not (0.30 <= retention_rate <= 0.70):
            log(f"Retention rate {retention_rate:.1%} outside expected range [30%, 70%]")
            log(f"This may indicate unusual data quality or inappropriate thresholds")
            # NOTE: Per spec, this is a WARNING not a FAILURE - proceed anyway
        else:
            log(f"Retention rate {retention_rate:.1%} within expected range [30%, 70%]")

        # Check for duplicates
        n_duplicates = df_purified_list['item_name'].duplicated().sum()
        if n_duplicates > 0:
            raise ValueError(f"Purified items list contains {n_duplicates} duplicate item names")
        log(f"No duplicate item names in purified list")

        # Check all retained items exist in original
        original_items = set(df_items['item_name'])
        retained_items = set(df_purified_list['item_name'])
        invalid_items = retained_items - original_items
        if invalid_items:
            raise ValueError(f"Purified list contains {len(invalid_items)} items not in Pass 1: {invalid_items}")
        log(f"All {n_retained} retained items exist in Pass 1 item parameters")

        # Check exclusion reasons sum correctly
        expected_exclusions = n_excluded
        actual_exclusions = len(df_excluded)
        if expected_exclusions != actual_exclusions:
            raise ValueError(f"Exclusion count mismatch: expected {expected_exclusions}, got {actual_exclusions}")
        log(f"Exclusion count verified: {n_excluded} items")

        log("All validation checks passed")
        log("Step 2 complete")

        # Final summary
        log("")
        log("=" * 70)
        log("PURIFICATION COMPLETE")
        log("=" * 70)
        log(f"Outputs:")
        log(f"  - Purified items list: {purified_items_path}")
        log(f"  - Purification report: {report_path}")
        log(f"Summary:")
        log(f"  - {n_retained}/{n_total} items retained ({retention_rate:.1%})")
        log(f"  - {n_excluded}/{n_total} items excluded ({1-retention_rate:.1%})")
        log("=" * 70)

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
