#!/usr/bin/env python3
"""Purify Items (Decision D039): Filter IRT items by D039 thresholds to retain only psychometrically sound items."""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

from tools.validation import validate_irt_parameters

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_purify_items.log"

# Decision D039 thresholds
A_THRESHOLD = 0.4   # Minimum discrimination
B_THRESHOLD = 3.0   # Maximum |difficulty|

# Logging Function

def log(msg):
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file at start
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 02: Purify Items (Decision D039)")
        log(f"a_threshold >= {A_THRESHOLD}")
        log(f"|b_threshold| <= {B_THRESHOLD}")
        # Load Input Data
        # Contains: item_name, Difficulty, Overall_Discrimination, Discrim_* columns

        log("Loading Pass 1 item parameters...")
        input_path = RQ_DIR / "logs" / "step01_pass1_item_params.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                "Step 01 (IRT Calibration Pass 1) must complete first."
            )

        pass1_params = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(pass1_params)} rows, {len(pass1_params.columns)} cols)")
        log(f"Columns: {list(pass1_params.columns)}")
        # Run Analysis Tool

        log("Running filter_items_by_quality...")
        log(f"df_items shape: {pass1_params.shape}")
        log(f"a_threshold: {A_THRESHOLD}")
        log(f"b_threshold: {B_THRESHOLD}")

        retained_items, removed_items = filter_items_by_quality(
            df_items=pass1_params,
            a_threshold=A_THRESHOLD,  # Decision D039: minimum discrimination
            b_threshold=B_THRESHOLD   # Decision D039: maximum |difficulty|
        )

        log("Analysis complete")
        log(f"Retained: {len(retained_items)} items")
        log(f"Removed: {len(removed_items)} items")

        # Calculate retention rate
        total_items = len(pass1_params)
        retention_rate = len(retained_items) / total_items * 100 if total_items > 0 else 0
        log(f"Retention rate: {retention_rate:.1f}%")
        # Save Analysis Outputs
        # These outputs will be used by:
        # - step03_irt_calibration_pass2 (uses purified items only)
        # - Diagnostics and reporting (removed items with reasons)

        # Save retained items
        retained_path = RQ_DIR / "data" / "step02_purified_items.csv"
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Saving purified items...")
        retained_items.to_csv(retained_path, index=False, encoding='utf-8')
        log(f"{retained_path.name} ({len(retained_items)} rows, {len(retained_items.columns)} cols)")

        # Save removed items (diagnostic output)
        removed_path = RQ_DIR / "logs" / "step02_removed_items.csv"
        log(f"Saving removed items...")
        removed_items.to_csv(removed_path, index=False, encoding='utf-8')
        log(f"{removed_path.name} ({len(removed_items)} rows)")

        # Log exclusion reasons summary
        if len(removed_items) > 0 and 'exclusion_reason' in removed_items.columns:
            log("Exclusion reasons summary:")
            reason_counts = removed_items['exclusion_reason'].value_counts()
            for reason, count in reason_counts.items():
                log(f"  - {reason}: {count} items")
        # Run Validation Tool
        # Validates: All retained items meet D039 criteria
        # Thresholds: a >= 0.4, |b| <= 3.0

        log("Running validate_irt_parameters...")

        # Note: validate_irt_parameters checks that NO items are flagged
        # (i.e., all items should pass the thresholds after purification)
        # We need to map column names based on what's in the retained items

        # Determine column names for validation
        # filter_items_by_quality returns original columns from calibrate_irt
        # which uses 'Difficulty' and 'Overall_Discrimination'
        a_col = 'Overall_Discrimination' if 'Overall_Discrimination' in retained_items.columns else 'a'
        b_col = 'Difficulty' if 'Difficulty' in retained_items.columns else 'b'

        log(f"Validation using a_col='{a_col}', b_col='{b_col}'")

        validation_result = validate_irt_parameters(
            df_items=retained_items,
            a_min=A_THRESHOLD,   # All items should have a >= 0.4
            b_max=B_THRESHOLD,   # All items should have |b| <= 3.0
            a_col=a_col,
            b_col=b_col
        )

        # Report validation results
        log(f"valid: {validation_result.get('valid', 'N/A')}")
        log(f"n_flagged: {validation_result.get('n_flagged', 'N/A')}")
        log(f"total_items: {validation_result.get('total_items', 'N/A')}")

        if not validation_result.get('valid', True):
            flagged = validation_result.get('flagged_items', [])
            log(f"{len(flagged)} items still flagged after purification:")
            for item in flagged[:5]:  # Show first 5
                log(f"  - {item.get('item_name')}: {item.get('reasons')}")
            if len(flagged) > 5:
                log(f"  ... and {len(flagged) - 5} more")

        # Additional validation: Check domain coverage
        log("Checking domain coverage...")

        # Try to determine domain from Discrim_* columns
        discrim_cols = [col for col in retained_items.columns if col.startswith('Discrim_')]
        if discrim_cols:
            domain_counts = {}
            for _, row in retained_items.iterrows():
                # Find which domain has highest discrimination (primary loading)
                discrim_values = {col.replace('Discrim_', ''): row[col] for col in discrim_cols}
                primary_domain = max(discrim_values.items(), key=lambda x: x[1])[0]
                domain_counts[primary_domain] = domain_counts.get(primary_domain, 0) + 1

            log(f"Items per domain:")
            min_per_domain = float('inf')
            for domain, count in sorted(domain_counts.items()):
                log(f"  - {domain}: {count} items")
                min_per_domain = min(min_per_domain, count)

            # Check minimum items per domain criterion
            if min_per_domain < 10:
                log(f"Some domains have fewer than 10 items (min={min_per_domain})")
        else:
            log("Domain coverage check skipped (no Discrim_* columns)")

        # Check retention rate bounds
        if retention_rate < 20:
            log(f"Retention rate ({retention_rate:.1f}%) below 20% - too aggressive filtering")
        elif retention_rate > 95:
            log(f"Retention rate ({retention_rate:.1f}%) above 95% - filtering may be too lenient")
        else:
            log(f"Retention rate ({retention_rate:.1f}%) within expected range [20%, 95%]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
