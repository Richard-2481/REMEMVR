#!/usr/bin/env python3
"""Item Purification (Decision D039): Remove items with extreme difficulty or low discrimination per Decision D039 thresholds."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_irt import filter_items_by_quality

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_purify_items.log"

# Input files
INPUT_ITEM_PARAMS = RQ_DIR / "logs" / "step01_pass1_item_params.csv"

# Output files
OUTPUT_PURIFIED = RQ_DIR / "data" / "step02_purified_items.csv"
OUTPUT_REMOVED = RQ_DIR / "data" / "step02_removed_items.csv"

# D039 Thresholds
A_THRESHOLD = 0.4  # Minimum discrimination
B_THRESHOLD = 3.0  # Maximum |difficulty|

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Item Purification (Decision D039)")
        log(f"RQ Directory: {RQ_DIR}")
        log(f"Thresholds: a >= {A_THRESHOLD}, |b| <= {B_THRESHOLD}")
        # Load Pass 1 Item Parameters
        log("\nLoading Pass 1 item parameters...")

        df_pass1_params = pd.read_csv(INPUT_ITEM_PARAMS)
        log(f"{INPUT_ITEM_PARAMS.name} ({len(df_pass1_params)} items)")

        # Show initial parameter ranges
        a_min, a_max = df_pass1_params["a"].min(), df_pass1_params["a"].max()
        b_min, b_max = df_pass1_params["b"].min(), df_pass1_params["b"].max()
        log(f"  Initial a range: [{a_min:.3f}, {a_max:.3f}]")
        log(f"  Initial b range: [{b_min:.3f}, {b_max:.3f}]")
        # Apply Purification Thresholds
        log("\nApplying D039 purification thresholds...")

        # Identify items to remove
        purified_items = []
        removed_items = []

        for _, row in df_pass1_params.iterrows():
            item_name = row["item_name"]
            dimension = row["dimension"]
            a = row["a"]
            b = row["b"]

            reasons = []

            # Check discrimination threshold
            if a < A_THRESHOLD:
                reasons.append(f"a < {A_THRESHOLD} ({a:.3f})")

            # Check difficulty threshold (absolute value)
            if abs(b) > B_THRESHOLD:
                reasons.append(f"|b| > {B_THRESHOLD} ({b:.3f})")

            if reasons:
                removed_items.append({
                    "item_name": item_name,
                    "dimension": dimension,
                    "a": a,
                    "b": b,
                    "removal_reason": "; ".join(reasons)
                })
            else:
                purified_items.append({
                    "item_name": item_name,
                    "dimension": dimension,
                    "a": a,
                    "b": b,
                    "retention_reason": f"Meets D039 thresholds (a >= {A_THRESHOLD}, |b| <= {B_THRESHOLD})"
                })

        df_purified = pd.DataFrame(purified_items)
        df_removed = pd.DataFrame(removed_items)
        # Report Purification Results
        log("\nPurification summary:")

        n_total = len(df_pass1_params)
        n_purified = len(df_purified)
        n_removed = len(df_removed)
        retention_rate = n_purified / n_total * 100

        log(f"  Total items: {n_total}")
        log(f"  Purified (retained): {n_purified} ({retention_rate:.1f}%)")
        log(f"  Removed: {n_removed} ({100 - retention_rate:.1f}%)")

        # Items per dimension
        log("\n  Items per dimension after purification:")
        for dim in ["common", "congruent", "incongruent"]:
            n_dim = len(df_purified[df_purified["dimension"] == dim])
            n_dim_original = len(df_pass1_params[df_pass1_params["dimension"] == dim])
            log(f"    {dim}: {n_dim} / {n_dim_original} retained")

        # List removed items
        if n_removed > 0:
            log("\n  Removed items:")
            for _, row in df_removed.iterrows():
                log(f"    - {row['item_name']} ({row['dimension']}): {row['removal_reason']}")
        # Save Outputs
        log("\nSaving output files...")

        # Ensure data directory exists
        (RQ_DIR / "data").mkdir(parents=True, exist_ok=True)

        # Save purified items
        df_purified.to_csv(OUTPUT_PURIFIED, index=False, encoding='utf-8')
        log(f"{OUTPUT_PURIFIED.name} ({len(df_purified)} items)")

        # Save removed items
        df_removed.to_csv(OUTPUT_REMOVED, index=False, encoding='utf-8')
        log(f"{OUTPUT_REMOVED.name} ({len(df_removed)} items)")
        # Validation
        log("\nValidating purification results...")

        # Check all retained items meet thresholds
        if len(df_purified) > 0:
            a_violations = df_purified[df_purified["a"] < A_THRESHOLD]
            b_violations = df_purified[df_purified["b"].abs() > B_THRESHOLD]

            if len(a_violations) > 0:
                raise ValueError(f"Retained items violate a threshold: {a_violations['item_name'].tolist()}")
            log(f"All retained items have a >= {A_THRESHOLD}")

            if len(b_violations) > 0:
                raise ValueError(f"Retained items violate b threshold: {b_violations['item_name'].tolist()}")
            log(f"All retained items have |b| <= {B_THRESHOLD}")

        # Check minimum items per dimension
        min_items_per_dim = 5
        for dim in ["common", "congruent", "incongruent"]:
            n_dim = len(df_purified[df_purified["dimension"] == dim])
            if n_dim < min_items_per_dim:
                log(f"Only {n_dim} items retained for {dim} (minimum {min_items_per_dim})")
            else:
                log(f"{dim}: {n_dim} items (>= {min_items_per_dim})")

        # Check minimum total items
        min_total = 30
        if n_purified < min_total:
            log(f"Only {n_purified} items retained (recommended minimum {min_total})")
        else:
            log(f"Total items: {n_purified} (>= {min_total})")

        # Check retention rate
        if retention_rate < 40:
            log(f"Low retention rate: {retention_rate:.1f}% (expected 40-80%)")
        elif retention_rate > 80:
            log(f"High retention rate: {retention_rate:.1f}% (may indicate lenient thresholds)")
        else:
            log(f"Retention rate: {retention_rate:.1f}% (within expected 40-80%)")

        # Show final parameter ranges
        if len(df_purified) > 0:
            a_min, a_max = df_purified["a"].min(), df_purified["a"].max()
            b_min, b_max = df_purified["b"].min(), df_purified["b"].max()
            log(f"\n  Final a range: [{a_min:.3f}, {a_max:.3f}]")
            log(f"  Final b range: [{b_min:.3f}, {b_max:.3f}]")

        log("\nStep 02 complete (Item Purification)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
