#!/usr/bin/env python3
"""Purify Items by Quality Thresholds (Decision D039): Filter items based on Decision D039 purification thresholds to retain only"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

from tools.validation import validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_purify_items.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Purify Items by Quality Thresholds (Decision D039)")
        # Load Input Data

        log("Loading Pass 1 item parameters...")
        input_path = RQ_DIR / "data" / "step01_pass1_item_params.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        pass1_params = pd.read_csv(input_path, encoding='utf-8')
        log(f"step01_pass1_item_params.csv ({len(pass1_params)} rows, {len(pass1_params.columns)} cols)")

        # Validate input columns
        required_cols = ['item_tag', 'factor', 'a', 'b']
        missing_cols = [col for col in required_cols if col not in pass1_params.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Show input summary
        log(f"Total items: {len(pass1_params)}")
        log(f"Items by factor:")
        for factor in pass1_params['factor'].unique():
            count = len(pass1_params[pass1_params['factor'] == factor])
            log(f"  - {factor}: {count} items")
        # Run Analysis Tool (filter_items_by_quality)
        #               to identify items that should be retained vs excluded

        log("Running filter_items_by_quality (Decision D039)...")
        log("Thresholds: a >= 0.4, |b| <= 3.0")

        # CRITICAL: The tool expects 'item_name' column but our data has 'item_tag'
        # Rename temporarily for tool compatibility
        pass1_params_renamed = pass1_params.rename(columns={'item_tag': 'item_name'})

        # Run purification
        purified_items, excluded_items = filter_items_by_quality(
            df_items=pass1_params_renamed,
            a_threshold=0.4,    # Decision D039: minimum discrimination
            b_threshold=3.0     # Decision D039: maximum absolute difficulty
        )

        # Rename back to 'item_tag' for output consistency
        purified_items = purified_items.rename(columns={'item_name': 'item_tag'})
        excluded_items = excluded_items.rename(columns={'item_name': 'item_tag'})

        log("Item purification complete")
        log(f"Retained: {len(purified_items)} items")
        log(f"Excluded: {len(excluded_items)} items")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 (Pass 2 calibration)

        # Output 1: Purified items CSV (retained items only)
        log("Saving purified items...")
        output_path_purified = RQ_DIR / "data" / "step02_purified_items.csv"

        # Add retention_reason column (all retained items passed thresholds)
        purified_items['retention_reason'] = 'PASS'

        # Ensure column order matches specification
        output_cols = ['item_tag', 'factor', 'a', 'b', 'retention_reason']
        purified_items_output = purified_items[output_cols]

        purified_items_output.to_csv(output_path_purified, index=False, encoding='utf-8')
        log(f"step02_purified_items.csv ({len(purified_items_output)} rows, {len(purified_items_output.columns)} cols)")

        # Show retention breakdown by factor
        log("Retained items by factor:")
        for factor in purified_items['factor'].unique():
            count = len(purified_items[purified_items['factor'] == factor])
            log(f"  - {factor}: {count} items")

        # Output 2: Purification report TXT (summary statistics)
        log("Generating purification report...")
        output_path_report = RQ_DIR / "data" / "step02_purification_report.txt"

        with open(output_path_report, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ITEM PURIFICATION REPORT (Decision D039)\n")
            f.write("=" * 80 + "\n\n")

            # Summary statistics
            total_items = len(pass1_params)
            retained_count = len(purified_items)
            excluded_count = len(excluded_items)
            retained_pct = 100 * retained_count / total_items
            excluded_pct = 100 * excluded_count / total_items

            f.write(f"Total items analyzed: {total_items}\n")
            f.write(f"Items retained: {retained_count} ({retained_pct:.1f}%)\n")
            f.write(f"Items excluded: {excluded_count} ({excluded_pct:.1f}%)\n\n")

            # Breakdown by exclusion reason
            f.write("-" * 80 + "\n")
            f.write("EXCLUSION BREAKDOWN BY REASON\n")
            f.write("-" * 80 + "\n\n")

            if len(excluded_items) > 0:
                # Count exclusions by reason
                exclusion_reasons = excluded_items['exclusion_reason'].value_counts()
                for reason, count in exclusion_reasons.items():
                    f.write(f"  {reason}: {count} items\n")
            else:
                f.write("  No items excluded (all items passed thresholds)\n")

            f.write("\n")

            # Factor-specific retention counts
            f.write("-" * 80 + "\n")
            f.write("RETENTION BY FACTOR\n")
            f.write("-" * 80 + "\n\n")

            for factor in sorted(pass1_params['factor'].unique()):
                factor_total = len(pass1_params[pass1_params['factor'] == factor])
                factor_retained = len(purified_items[purified_items['factor'] == factor])
                factor_pct = 100 * factor_retained / factor_total
                f.write(f"  {factor}: {factor_retained}/{factor_total} ({factor_pct:.1f}%)\n")

            f.write("\n")

            # List of excluded items with reasons
            f.write("-" * 80 + "\n")
            f.write("EXCLUDED ITEMS (WITH REASONS)\n")
            f.write("-" * 80 + "\n\n")

            if len(excluded_items) > 0:
                for _, row in excluded_items.iterrows():
                    item = row['item_tag']
                    factor = row['factor']
                    a = row['a']
                    b = row['b']
                    reason = row['exclusion_reason']
                    f.write(f"  {item:20s} (factor={factor:12s}, a={a:.3f}, b={b:+.3f}): {reason}\n")
            else:
                f.write("  No items excluded\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        log(f"step02_purification_report.txt")
        # Run Validation Tool (validate_irt_parameters)
        # Validates: All retained items meet purification criteria
        # Threshold: a >= 0.4, |b| <= 3.0, minimum 10 items per factor

        log("Running validate_irt_parameters...")

        # Prepare validation parameters (need to match tool signature)
        validation_result = validate_irt_parameters(
            df_items=purified_items.rename(columns={'item_tag': 'item_name', 'a': 'Discrimination', 'b': 'Difficulty'}),
            a_min=0.4,
            b_max=3.0,
            a_col='Discrimination',
            b_col='Difficulty'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation: minimum items per factor
        log("Checking minimum items per factor (threshold: 10)...")
        for factor in purified_items['factor'].unique():
            factor_count = len(purified_items[purified_items['factor'] == factor])
            if factor_count < 10:
                raise ValueError(f"Factor '{factor}' has only {factor_count} items (minimum 10 required)")
            log(f"Factor '{factor}': {factor_count} items ")

        # Check for NaN values
        log("Checking for NaN values...")
        nan_count = purified_items.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in purified_items")
        log("No NaN values found ")

        # Check for duplicates
        log("Checking for duplicate items...")
        duplicate_count = purified_items['item_tag'].duplicated().sum()
        if duplicate_count > 0:
            duplicates = purified_items[purified_items['item_tag'].duplicated()]['item_tag'].tolist()
            raise ValueError(f"Found {duplicate_count} duplicate items: {duplicates}")
        log("No duplicate items found ")

        log("Step 2 complete - All validations passed")
        log(f"{retained_count} items retained, {excluded_count} items excluded")
        log(f"Ready for Pass 2 calibration (Step 3)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
