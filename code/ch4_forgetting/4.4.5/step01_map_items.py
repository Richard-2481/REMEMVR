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
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 01: Map Items by Retention Status")

        # Load full item list
        input_path = RQ_DIR / "data" / "step00_full_item_list.csv"
        log(f"Reading {input_path}")
        full_item_list = pd.read_csv(input_path, encoding='utf-8')
        log(f"{len(full_item_list)} items total")

        # Group by dimension and count retention status
        log("Computing retention statistics by dimension")

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

            log(f"{dimension.capitalize()}: {n_retained}/{n_total} retained ({retention_rate:.1%})")

        # Create summary DataFrame
        item_mapping = pd.DataFrame(results)

        # Validate accounting identity
        log("Checking accounting identity (N_total = N_retained + N_removed)")
        for idx, row in item_mapping.iterrows():
            if row['N_total'] != row['N_retained'] + row['N_removed']:
                raise ValueError(f"Accounting error for {row['dimension']}: {row['N_total']} != {row['N_retained']} + {row['N_removed']}")
        log("Accounting identity verified")

        # Check retention rates in plausible range
        log("Checking retention rates in [0.6, 0.9]")
        for idx, row in item_mapping.iterrows():
            if not (0.6 <= row['retention_rate'] <= 0.9):
                log(f"{row['dimension']} retention rate {row['retention_rate']:.1%} outside expected range [60%, 90%]")

        # Save results
        output_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        log(f"Writing {output_path}")
        item_mapping.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{len(item_mapping)} rows, {len(item_mapping.columns)} columns")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
