#!/usr/bin/env python3
"""Item Purification: Filter items by quality thresholds per Decision D039 (a >= 0.4, |b| <= 3.0)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

from tools.validation import validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1
LOG_FILE = RQ_DIR / "logs" / "step02_purify_items.log"


# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file for fresh run
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 02: Item Purification")
        log("=" * 60)
        # Load Input Data

        log("Loading Pass 1 item parameters...")

        input_path = RQ_DIR / "logs" / "step01_pass1_item_params.csv"
        pass1_params = pd.read_csv(input_path, encoding='utf-8')

        log(f"{input_path.name} ({len(pass1_params)} rows, {len(pass1_params.columns)} cols)")
        log(f"Columns: {list(pass1_params.columns)}")

        # Log summary statistics before purification
        log(f"Discrimination range: {pass1_params['Discrimination'].min():.3f} - {pass1_params['Discrimination'].max():.3f}")
        log(f"Difficulty range: {pass1_params['Difficulty_1'].min():.3f} - {pass1_params['Difficulty_1'].max():.3f}")

        # Items per paradigm before purification
        items_by_paradigm = pass1_params['domain'].value_counts().to_dict()
        log(f"Items by paradigm (before): {items_by_paradigm}")
        # Run Analysis Tool
        # Thresholds: a >= 0.4 (discrimination), |b| <= 3.0 (difficulty)

        log("Running filter_items_by_quality...")
        log(f"a_threshold=0.4, b_threshold=3.0 (Decision D039)")

        # The filter_items_by_quality function has two paths:
        # 1. Multivariate: expects 'Difficulty' + 'Discrim_*' columns
        # 2. Univariate: expects 'factor', 'a', 'b' columns
        #
        # Our data has 'domain', 'Discrimination', 'Difficulty_1' columns
        # We need to convert to univariate format for compatibility

        pass1_params_formatted = pass1_params.copy()
        pass1_params_formatted = pass1_params_formatted.rename(columns={
            'item': 'item_name',
            'domain': 'factor',
            'Discrimination': 'a',
            'Difficulty_1': 'b'
        })

        log(f"Reformatted columns for univariate path: {list(pass1_params_formatted.columns)}")

        retained_items, removed_items = filter_items_by_quality(
            df_items=pass1_params_formatted,
            a_threshold=0.4,  # Decision D039: Minimum discrimination
            b_threshold=3.0   # Decision D039: Maximum |difficulty|
        )

        log("Analysis complete")
        log(f"Retained: {len(retained_items)} items, Removed: {len(removed_items)} items")
        # Save Analysis Outputs
        # These outputs will be used by: Step 03 (IRT Pass 2 calibration)

        # Rename columns back to original format for consistency
        retained_items_output = retained_items.rename(columns={
            'item_name': 'item',
            'factor': 'domain',
            'a': 'Discrimination',
            'b': 'Difficulty_1'
        })

        # Ensure output has expected columns in right order
        output_columns = ['item', 'domain', 'Discrimination', 'Difficulty_1']
        available_cols = [c for c in output_columns if c in retained_items_output.columns]
        retained_items_output = retained_items_output[available_cols].copy()

        # Save retained items
        output_path = RQ_DIR / "data" / "step02_purified_items.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log(f"Saving {output_path.name}...")
        retained_items_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(retained_items_output)} rows, {len(retained_items_output.columns)} cols)")

        # Save removed items (for diagnostic purposes)
        if len(removed_items) > 0:
            removed_items_output = removed_items.rename(columns={
                'item_name': 'item',
                'factor': 'domain',
                'a': 'Discrimination',
                'b': 'Difficulty_1'
            })
            removed_output_path = RQ_DIR / "logs" / "step02_removed_items.csv"
            log(f"Saving {removed_output_path.name}...")
            removed_items_output.to_csv(removed_output_path, index=False, encoding='utf-8')
            log(f"{removed_output_path.name} ({len(removed_items_output)} rows)")
        else:
            log("No items removed - all items met quality thresholds")
        # Run Validation Tool
        # Validates: All retained items meet quality thresholds
        # Thresholds: a >= 0.4, |b| <= 3.0

        log("Running validate_irt_parameters...")

        # Prepare validation input - need 'a' and 'b' columns
        validation_df = retained_items_output.copy()
        validation_df['a'] = validation_df['Discrimination']
        validation_df['b'] = validation_df['Difficulty_1']

        validation_result = validate_irt_parameters(
            df_items=validation_df,
            a_min=0.4,           # Decision D039 threshold
            b_max=3.0,           # Decision D039 threshold
            a_col='a',           # Column name for discrimination
            b_col='b'            # Column name for difficulty
        )

        # Report validation results
        log(f"Valid: {validation_result['valid']}")
        log(f"Total items: {validation_result['total_items']}")
        log(f"Flagged items: {validation_result['n_flagged']}")

        if not validation_result['valid']:
            log("Some items still flagged after purification - checking details...")
            for item in validation_result['flagged_items']:
                log(f"  - {item['item_name']}: {item['reasons']}")
            raise ValueError(f"Validation failed: {validation_result['n_flagged']} items still flagged")

        # Additional validation: Check items per paradigm
        items_by_paradigm_after = retained_items_output['domain'].value_counts().to_dict()
        log(f"Items by paradigm (after): {items_by_paradigm_after}")

        # Check minimum items per paradigm
        min_items_per_paradigm = 10
        for paradigm, count in items_by_paradigm_after.items():
            if count < min_items_per_paradigm:
                log(f"Paradigm '{paradigm}' has only {count} items (minimum: {min_items_per_paradigm})")

        # Check no paradigm completely eliminated
        expected_paradigms = {'free_recall', 'cued_recall', 'recognition'}
        actual_paradigms = set(items_by_paradigm_after.keys())
        missing_paradigms = expected_paradigms - actual_paradigms
        if missing_paradigms:
            raise ValueError(f"CRITICAL: Paradigms completely eliminated: {missing_paradigms}")

        # Summary statistics
        log("=" * 60)
        log("Item Purification Results:")
        log(f"  - Original items: {len(pass1_params)}")
        log(f"  - Retained items: {len(retained_items_output)}")
        log(f"  - Removed items: {len(removed_items)}")
        log(f"  - Retention rate: {len(retained_items_output)/len(pass1_params)*100:.1f}%")
        log("=" * 60)

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
