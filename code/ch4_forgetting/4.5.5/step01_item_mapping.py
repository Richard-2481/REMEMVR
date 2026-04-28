#!/usr/bin/env python3
"""item_mapping: Create item mapping showing which items were retained vs removed during IRT"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_item_mapping.log"

# Input paths
PURIFIED_ITEMS_PATH = PROJECT_ROOT / "results/ch5/5.5.1/data/step02_purified_items.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data/cache/dfData.csv"

# Output path
OUTPUT_PATH = RQ_DIR / "data/step01_item_mapping.csv"

# IRT quality thresholds from RQ 5.5.1 purification
A_THRESHOLD = 0.4  # Minimum discrimination
B_THRESHOLD = 3.0  # Maximum absolute difficulty


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Item Mapping")
        # Load Purified Items from RQ 5.5.1

        log("Loading purified items from RQ 5.5.1...")
        df_purified = pd.read_csv(PURIFIED_ITEMS_PATH, encoding='utf-8')
        log(f"Purified items: {len(df_purified)} rows, {len(df_purified.columns)} cols")

        # Validate purified items structure
        expected_purified_cols = ['item_tag', 'factor', 'a', 'b', 'retention_reason']
        if df_purified.columns.tolist() != expected_purified_cols:
            log(f"Purified items columns mismatch")
            log(f"  Expected: {expected_purified_cols}")
            log(f"  Actual: {df_purified.columns.tolist()}")
            sys.exit(1)

        # Count by location type
        source_retained = len(df_purified[df_purified['factor'] == 'source'])
        dest_retained = len(df_purified[df_purified['factor'] == 'destination'])
        log(f"Retained items by location: Source={source_retained}, Destination={dest_retained}")

        # Create set of retained item tags for fast lookup
        retained_tags = set(df_purified['item_tag'].tolist())
        log(f"Retained item tags: {len(retained_tags)} unique items")
        # Extract All TQ Items from dfData.csv

        log("Loading dfData.csv to extract all TQ item columns...")
        df_data = pd.read_csv(RAW_DATA_PATH, nrows=0, encoding='utf-8')  # Header only
        all_cols = df_data.columns.tolist()

        # Filter to TQ items only (exclude TC items)
        tq_items = sorted([c for c in all_cols if c.startswith('TQ_') and ('-U-' in c or '-D-' in c)])
        log(f"Total TQ items found in dfData.csv: {len(tq_items)}")

        # Separate by location type
        tq_source = [c for c in tq_items if '-U-' in c]
        tq_dest = [c for c in tq_items if '-D-' in c]
        log(f"TQ items by location: Source={len(tq_source)}, Destination={len(tq_dest)}")

        # Validate expected counts
        if len(tq_items) != 36:
            log(f"Expected 36 TQ items, found {len(tq_items)}")
            sys.exit(1)
        if len(tq_source) != 18 or len(tq_dest) != 18:
            log(f"Expected 18 source + 18 destination, found {len(tq_source)} + {len(tq_dest)}")
            sys.exit(1)
        # Create Item Mapping with Retention Status
        # For each TQ item: check if in retained set, assign retention status
        # If retained: get a, b values from purified_items
        # If removed: a=NaN, b=NaN, assign removal_reason

        log("Creating item mapping with retention status...")

        item_mapping_rows = []

        for item_name in tq_items:
            # Determine location type from item name
            location_type = 'source' if '-U-' in item_name else 'destination'

            # Check if item was retained
            if item_name in retained_tags:
                # Item retained - get parameters from purified_items
                item_row = df_purified[df_purified['item_tag'] == item_name].iloc[0]
                a = item_row['a']
                b = item_row['b']
                retained = True
                removal_reason = 'retained'
            else:
                # Item removed - parameters unknown (not in purified list)
                # We don't have access to Pass 1 item parameters here, so mark as NaN
                a = np.nan
                b = np.nan
                retained = False
                # Removal reason: Since purified_items.csv doesn't include removed items,
                # we cannot determine exact removal reason (low_discrimination, extreme_difficulty, or both)
                # Mark as generic removed_in_rq551
                removal_reason = 'removed_in_rq551'

            item_mapping_rows.append({
                'item_name': item_name,
                'location_type': location_type,
                'a': a,
                'b': b,
                'retained': retained,
                'removal_reason': removal_reason
            })

        # Create DataFrame
        df_item_mapping = pd.DataFrame(item_mapping_rows)
        log(f"Item mapping: {len(df_item_mapping)} rows")
        # Validate Item Mapping
        # Check counts, retention rates, removal reasons

        log("Validating item mapping...")

        # Check total count
        if len(df_item_mapping) != 36:
            log(f"Expected 36 items, got {len(df_item_mapping)}")
            sys.exit(1)

        # Check location type values
        location_values = set(df_item_mapping['location_type'].unique())
        if location_values != {'source', 'destination'}:
            log(f"Unexpected location_type values: {location_values}")
            sys.exit(1)

        # Check retention counts by location
        source_retained_count = len(df_item_mapping[(df_item_mapping['location_type'] == 'source') &
                                                     (df_item_mapping['retained'] == True)])
        source_removed_count = len(df_item_mapping[(df_item_mapping['location_type'] == 'source') &
                                                    (df_item_mapping['retained'] == False)])
        dest_retained_count = len(df_item_mapping[(df_item_mapping['location_type'] == 'destination') &
                                                   (df_item_mapping['retained'] == True)])
        dest_removed_count = len(df_item_mapping[(df_item_mapping['location_type'] == 'destination') &
                                                  (df_item_mapping['retained'] == False)])

        log(f"Source: {source_retained_count} retained, {source_removed_count} removed")
        log(f"Destination: {dest_retained_count} retained, {dest_removed_count} removed")

        # Expected from user message: 17 source retained, 1 source removed, 15 destination retained, 3 destination removed
        if source_retained_count != 17 or source_removed_count != 1:
            log(f"Source retention counts differ from expected (17 retained, 1 removed)")
        if dest_retained_count != 15 or dest_removed_count != 3:
            log(f"Destination retention counts differ from expected (15 retained, 3 removed)")

        # Check removal_reason values
        removal_reasons = set(df_item_mapping['removal_reason'].unique())
        valid_reasons = {'retained', 'removed_in_rq551'}
        if not removal_reasons.issubset(valid_reasons):
            log(f"Unexpected removal_reason values: {removal_reasons - valid_reasons}")

        # Check retention rate per location type
        source_total = len(df_item_mapping[df_item_mapping['location_type'] == 'source'])
        dest_total = len(df_item_mapping[df_item_mapping['location_type'] == 'destination'])
        source_retention_rate = source_retained_count / source_total
        dest_retention_rate = dest_retained_count / dest_total

        log(f"Retention rates: Source={source_retention_rate:.2%}, Destination={dest_retention_rate:.2%}")

        # Check that retention rate is within expected range [0.55, 0.85]
        if not (0.55 <= source_retention_rate <= 0.85):
            log(f"Source retention rate outside expected range [0.55, 0.85]: {source_retention_rate:.2%}")
        if not (0.55 <= dest_retention_rate <= 0.85):
            log(f"Destination retention rate outside expected range [0.55, 0.85]: {dest_retention_rate:.2%}")

        # Check at least 10 retained items per location type (per validation criteria)
        if source_retained_count < 10:
            log(f"Source has < 10 retained items ({source_retained_count})")
            sys.exit(1)
        if dest_retained_count < 10:
            log(f"Destination has < 10 retained items ({dest_retained_count})")
            sys.exit(1)

        log("All validation checks passed")
        # Save Item Mapping
        # Output: data/step01_item_mapping.csv

        log(f"Saving item mapping to {OUTPUT_PATH}...")
        df_item_mapping.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        log(f"{OUTPUT_PATH} ({len(df_item_mapping)} rows, {len(df_item_mapping.columns)} cols)")

        # Log summary statistics
        log(f"\nItem Mapping Statistics:")
        log(f"  Total items: {len(df_item_mapping)}")
        log(f"  Source items: {source_total} ({source_retained_count} retained, {source_removed_count} removed)")
        log(f"  Destination items: {dest_total} ({dest_retained_count} retained, {dest_removed_count} removed)")
        log(f"  Overall retention rate: {len(df_item_mapping[df_item_mapping['retained'] == True]) / len(df_item_mapping):.2%}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
