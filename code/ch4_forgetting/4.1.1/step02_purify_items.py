#!/usr/bin/env python3
"""Item Purification: Apply Decision D039 quality thresholds (|b| <= 3.0, a >= 0.4) to item parameters"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# parents[4] = REMEMVR/ (code -> rq7 -> ch5 -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

from tools.validation import validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
LOG_FILE = RQ_DIR / "logs" / "step02_purification_report.txt"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Item Purification")
        # Load Pass 1 Item Parameters

        log("Loading Pass 1 item parameters...")
        input_path = RQ_DIR / "logs" / "step01_item_parameters.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Step 1 output missing: {input_path}\n"
                                     "Run step01_irt_calibration_omnibus.py first")

        pass1_params = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(pass1_params)} items)")
        log(f"  Columns: {pass1_params.columns.tolist()}")
        log(f"  Discrimination range: [{pass1_params['a'].min():.3f}, {pass1_params['a'].max():.3f}]")
        log(f"  Difficulty range: [{pass1_params['b'].min():.3f}, {pass1_params['b'].max():.3f}]")
        # Apply Quality Thresholds (Decision D039)

        log("Applying Decision D039 thresholds (a>=0.4, |b|<=3.0)...")
        purified_items, removed_items = filter_items_by_quality(
            df_items=pass1_params,
            a_threshold=0.4,  # Minimum discrimination (items with a<0.4 excluded)
            b_threshold=3.0   # Maximum |difficulty| (items with |b|>3.0 excluded)
        )

        retention_rate = len(purified_items) / len(pass1_params) * 100
        log(f"Purification complete")
        log(f"  Retained: {len(purified_items)} items ({retention_rate:.1f}%)")
        log(f"  Removed: {len(removed_items)} items ({100-retention_rate:.1f}%)")
        # Save Purification Outputs
        # These outputs will be used by: Step 3 (Pass 2 IRT calibration)

        # Save purified item list
        output_path = RQ_DIR / "data" / "step02_purified_items.csv"
        log(f"Saving purified items to {output_path.name}...")
        purified_items.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(purified_items)} items, {len(purified_items.columns)} cols)")

        # Save exclusion report
        log(f"Saving exclusion report to {LOG_FILE.name}...")
        log("")  # Blank line separator
        log("=" * 80)
        log("EXCLUSION REPORT - Items Removed from Analysis")
        log("=" * 80)
        log(f"\nTotal items removed: {len(removed_items)}")

        if len(removed_items) > 0:
            log("\nRemoved items by reason:")
            log(removed_items.to_string(index=False))
        else:
            log("\nNo items removed (all items passed quality thresholds)")

        log("\n" + "=" * 80)
        log(f"Exclusion report appended to {LOG_FILE.name}")
        # Validate Purified Items
        # Validates: All retained items meet thresholds, retention rate reasonable
        # Threshold: a>=0.4, |b|<=3.0, retention 30-70%

        log("Validating purified item parameters...")
        validation_result = validate_irt_parameters(
            df_items=purified_items,
            a_min=0.4,        # Minimum discrimination threshold
            b_max=3.0,        # Maximum |difficulty| threshold
            a_col='pass1_a',  # Column name for discrimination in purified dataset
            b_col='pass1_b'   # Column name for difficulty in purified dataset
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check retention rate (expected 30-70%)
        if retention_rate < 30:
            log(f"Low retention rate ({retention_rate:.1f}%) - fewer items than expected")
        elif retention_rate > 70:
            log(f"High retention rate ({retention_rate:.1f}%) - thresholds may be too lenient")
        else:
            log(f"Retention rate within expected range ({retention_rate:.1f}%)")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
