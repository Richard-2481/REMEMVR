#!/usr/bin/env python3
"""Map Items to Full vs Purified Sets: Identify which TQ_* items in dfData.csv were retained vs excluded by RQ 5.2.1"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_map_items.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: Map Items to Full vs Purified Sets")
        # Load Input Data

        log("Loading input data...")

        # Load purified items from RQ 5.1 Step 2 (retained items after quality filtering)
        # Expected columns: item_name, factor, a, b
        # Expected rows: ~38 items
        df_purified = pd.read_csv(RQ_DIR / "data/step00_irt_purified_items.csv")
        log(f"step00_irt_purified_items.csv ({len(df_purified)} rows, {len(df_purified.columns)} cols)")

        # Load raw scores from Step 0 (dfData with composite_ID added)
        # Expected columns: composite_ID, UID, TEST, TQ_* (many item columns)
        # Expected rows: ~400 (100 participants x 4 tests)
        df_raw = pd.read_csv(RQ_DIR / "data/step00_raw_scores.csv")
        log(f"step00_raw_scores.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")
        # Extract Item Lists

        log("Extracting item lists...")

        # Extract all column names matching pattern 'TQ_*' -> full_item_list
        # This represents ALL items in the original test battery
        full_item_list = [col for col in df_raw.columns if col.startswith('TQ_')]
        log(f"Full item list: {len(full_item_list)} items")

        # Extract item_name values from purified items -> purified_item_list
        # This represents ONLY items retained after RQ 5.1 quality filtering
        purified_item_list = df_purified['item_name'].tolist()
        log(f"Purified item list: {len(purified_item_list)} items")

        # Compute retention rate
        retention_rate = len(purified_item_list) / len(full_item_list) * 100
        log(f"Retention rate: {retention_rate:.1f}% ({len(purified_item_list)}/{len(full_item_list)})")
        # Classify Items by Domain (What/Where only - When EXCLUDED)
        # Tag patterns for domain classification:
        #   What (Nominal): TQ_*-N-*
        #   Where (Spatial): TQ_*-U-* + TQ_*-D-* + TQ_*-L-*
        #   When (Temporal): TQ_*-O-* -> EXCLUDED due to floor effect
        # Note: When items are filtered out, not classified

        log("Classifying items by domain using tag patterns...")
        log("When domain (-O-) items will be EXCLUDED per RQ 5.2.1 floor effect")

        def classify_domain(item_name: str) -> str:
            """Classify item into domain based on tag pattern.
            Returns None for When items (to be excluded).
            """
            if '-N-' in item_name:
                return 'what'
            elif '-U-' in item_name or '-D-' in item_name or '-L-' in item_name:
                return 'where'
            elif '-O-' in item_name or '-T-' in item_name:
                return None  # EXCLUDED - When domain
            else:
                # Fallback for unexpected pattern
                return 'unknown'

        # Create item mapping DataFrame (EXCLUDING When items)
        # Columns: item_name, domain, retained
        item_mapping_data = []
        when_excluded_count = 0
        for item in full_item_list:
            domain = classify_domain(item)
            if domain is None:
                # When item - skip (excluded)
                when_excluded_count += 1
                continue
            retained = item in purified_item_list
            item_mapping_data.append({
                'item_name': item,
                'domain': domain,
                'retained': retained
            })

        log(f"Excluded {when_excluded_count} When domain items")

        df_mapping = pd.DataFrame(item_mapping_data)
        log(f"Created mapping for {len(df_mapping)} items")

        # Count items per domain
        domain_counts = df_mapping.groupby('domain').agg({
            'item_name': 'count',
            'retained': 'sum'
        }).rename(columns={'item_name': 'total', 'retained': 'retained'})
        domain_counts['removed'] = domain_counts['total'] - domain_counts['retained']
        domain_counts['retention_rate'] = domain_counts['retained'] / domain_counts['total'] * 100

        log("Item counts per domain (What/Where only):")
        log(f"  What:  {domain_counts.loc['what', 'retained']:.0f} retained / {domain_counts.loc['what', 'total']:.0f} total ({domain_counts.loc['what', 'retention_rate']:.1f}%)")
        log(f"  Where: {domain_counts.loc['where', 'retained']:.0f} retained / {domain_counts.loc['where', 'total']:.0f} total ({domain_counts.loc['where', 'retention_rate']:.1f}%)")
        log(f"  When:  EXCLUDED (floor effect in RQ 5.2.1 - {when_excluded_count} items removed)")
        # Save Outputs
        # Outputs:
        #   - data/step01_item_mapping.csv: Item mapping with retention status
        #   - logs/step01_item_counts.txt: Text report of counts

        log("Saving outputs...")

        # Save item mapping CSV
        # Columns: item_name, domain, retained
        # Expected rows: ~50 items
        output_mapping_path = RQ_DIR / "data/step01_item_mapping.csv"
        df_mapping.to_csv(output_mapping_path, index=False, encoding='utf-8')
        log(f"{output_mapping_path.name} ({len(df_mapping)} rows, {len(df_mapping.columns)} cols)")

        # Save item counts report (text file in logs/)
        # Contains: Domain-wise breakdown of full vs purified counts (What/Where only)
        counts_report_path = RQ_DIR / "logs/step01_item_counts.txt"
        with open(counts_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ITEM MAPPING REPORT - RQ 5.2.5 Step 1\n")
            f.write("(When domain EXCLUDED due to floor effect in RQ 5.2.1)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total items in test battery (raw): {len(full_item_list)}\n")
            f.write(f"When domain items excluded: {when_excluded_count}\n")
            f.write(f"Items analyzed (What/Where only): {len(df_mapping)}\n")
            f.write(f"Items retained by RQ 5.2.1 purification: {len(purified_item_list)}\n\n")
            f.write("-" * 70 + "\n")
            f.write("DOMAIN-WISE BREAKDOWN (What/Where only)\n")
            f.write("-" * 70 + "\n\n")
            for domain in ['what', 'where']:
                if domain in domain_counts.index:
                    total = int(domain_counts.loc[domain, 'total'])
                    retained = int(domain_counts.loc[domain, 'retained'])
                    removed = int(domain_counts.loc[domain, 'removed'])
                    rate = domain_counts.loc[domain, 'retention_rate']
                    f.write(f"{domain.upper()} Domain:\n")
                    f.write(f"  Total items:    {total}\n")
                    f.write(f"  Retained items: {retained}\n")
                    f.write(f"  Removed items:  {removed}\n")
                    f.write(f"  Retention rate: {rate:.1f}%\n\n")
            f.write("-" * 70 + "\n")
            f.write("WHEN Domain: EXCLUDED\n")
            f.write(f"  Reason: Floor effect discovered in RQ 5.2.1\n")
            f.write(f"  Items excluded: {when_excluded_count}\n")
            f.write("=" * 70 + "\n")

        log(f"{counts_report_path.name} (item counts report)")
        # Run Validation (What/Where only - When excluded)
        # Validates: Row count in [60, 90], all required columns present,
        #            domain values in {what, where} ONLY, retention rate in typical range

        log("Running validate_dataframe_structure...")

        validation_result = validate_dataframe_structure(
            df=df_mapping,
            expected_rows=(60, 100),  # Range for What+Where item count
            expected_columns=['item_name', 'domain', 'retained'],
            column_types={
                'item_name': (object,),
                'domain': (object,),
                'retained': (bool,)
            }
        )

        # Report validation results
        if validation_result['valid']:
            log("DataFrame structure valid")
            log(f"Row count: {len(df_mapping)} (What/Where items only)")
            log(f"Columns: {list(df_mapping.columns)}")
            log(f"Domain values: {df_mapping['domain'].unique().tolist()}")

            # Compute retention rate for included items only
            included_retention = df_mapping['retained'].sum() / len(df_mapping) * 100
            log(f"Retention rate (What/Where): {included_retention:.1f}%")

            # Additional validation checks - ensure only what/where (no when)
            domain_values = set(df_mapping['domain'].unique())
            expected_domains = {'what', 'where'}
            if domain_values != expected_domains:
                unexpected = domain_values - expected_domains
                if unexpected:
                    log(f"Unexpected domain values: {unexpected}")
                    raise ValueError(f"When domain should be excluded but found: {unexpected}")
                missing = expected_domains - domain_values
                if missing:
                    log(f"Missing expected domains: {missing}")

            # Check if 'when' was accidentally included
            if 'when' in domain_values:
                raise ValueError("When domain should be EXCLUDED but was found in mapping!")

            log("When domain correctly excluded")

        else:
            log(f"{validation_result['message']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log("Step 1 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
