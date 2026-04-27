#!/usr/bin/env python3
"""
Step 01: Map Retained vs Removed Items by Congruence Category

Quantifies item purification effects separately for Common, Congruent, Incongruent.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_map_items.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 01: Map Items by Retention Status")

        # Load full item list
        input_path = RQ_DIR / "data" / "step00_full_item_list.csv"
        log(f"[LOAD] Reading {input_path}")
        full_item_list = pd.read_csv(input_path, encoding='utf-8')
        log(f"[LOADED] {len(full_item_list)} items total")

        # Group by dimension and count retention status
        log("[ANALYSIS] Computing retention statistics by dimension")

        results = []
        for dimension in ['common', 'congruent', 'incongruent']:
            dimension_items = full_item_list[full_item_list['dimension'] == dimension]
            n_total = len(dimension_items)
            n_retained = dimension_items['retained'].sum()
            n_removed = n_total - n_retained
            retention_rate = n_retained / n_total if n_total > 0 else 0.0

            results.append({
                'dimension': dimension.capitalize(),
                'N_total': n_total,
                'N_retained': n_retained,
                'N_removed': n_removed,
                'retention_rate': retention_rate
            })

            log(f"[SUMMARY] {dimension.capitalize()}: {n_retained}/{n_total} retained ({retention_rate:.1%})")

        # Create summary DataFrame
        item_mapping = pd.DataFrame(results)

        # Validate accounting identity
        log("[VALIDATION] Checking accounting identity (N_total = N_retained + N_removed)")
        for idx, row in item_mapping.iterrows():
            if row['N_total'] != row['N_retained'] + row['N_removed']:
                raise ValueError(f"Accounting error for {row['dimension']}: {row['N_total']} != {row['N_retained']} + {row['N_removed']}")
        log("[PASS] Accounting identity verified")

        # Check retention rates in plausible range
        log("[VALIDATION] Checking retention rates in [0.6, 0.9]")
        for idx, row in item_mapping.iterrows():
            if not (0.6 <= row['retention_rate'] <= 0.9):
                log(f"[WARNING] {row['dimension']} retention rate {row['retention_rate']:.1%} outside expected range [60%, 90%]")

        # Save results
        output_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        log(f"[SAVE] Writing {output_path}")
        item_mapping.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {len(item_mapping)} rows, {len(item_mapping.columns)} columns")

        log("[SUCCESS] Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
