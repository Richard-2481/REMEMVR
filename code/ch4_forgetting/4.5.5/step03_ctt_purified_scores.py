#!/usr/bin/env python3
"""Compute Purified CTT Sum Scores: Compute Classical Test Theory (CTT) sum scores using ONLY IRT-retained items"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5
LOG_FILE = RQ_DIR / "logs" / "step03_ctt_purified_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Compute Purified CTT Sum Scores")
        # Load Item Mapping and Identify Retained Items

        log("Loading item mapping from Step 1...")
        item_mapping_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        df_items = pd.read_csv(item_mapping_path, encoding='utf-8')
        log(f"item_mapping: {len(df_items)} items")

        # Filter to retained items only
        df_retained = df_items[df_items['retained'] == True].copy()
        log(f"Retained items: {len(df_retained)} / {len(df_items)} items")

        # Separate by location_type
        source_retained = df_retained[df_retained['location_type'] == 'source']['item_name'].tolist()
        dest_retained = df_retained[df_retained['location_type'] == 'destination']['item_name'].tolist()

        log(f"Source items: {len(source_retained)}")
        log(f"Destination items: {len(dest_retained)}")

        # Validate retention counts
        if len(source_retained) < 10 or len(dest_retained) < 10:
            log(f"Insufficient retained items (source={len(source_retained)}, dest={len(dest_retained)})")
            log("Need at least 10 items per location_type for reliable CTT")
            sys.exit(1)
        # Load Raw Binary Responses

        log("Loading raw binary responses from dfData.csv...")
        dfData_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        df_raw = pd.read_csv(dfData_path, encoding='utf-8')
        log(f"dfData: {len(df_raw)} rows, {len(df_raw.columns)} columns")

        # Map item_name to TQ_ column names in dfData
        # Item mapping uses: TQ_ICR-D-i1, TQ_ICR-U-i2, etc.
        # dfData columns use: TQ_ICR-D-i1, TQ_ICR-U-i2, etc. (same format)

        # Verify all retained items exist as columns in dfData
        all_retained_items = source_retained + dest_retained
        missing_items = [item for item in all_retained_items if item not in df_raw.columns]
        if missing_items:
            log(f"Missing items in dfData: {missing_items}")
            sys.exit(1)

        log("All retained items found in dfData")
        # Compute Purified CTT Scores

        log("Computing purified CTT scores...")

        results = []

        # Process each UID x TEST combination
        for _, row in df_raw.iterrows():
            uid = row['UID']
            test_val = row['TEST']

            # Convert TEST column (1-4) to test format (T1-T4)
            if pd.notna(test_val):
                test = f"T{int(test_val)}"
            else:
                log(f"Skipping row with NaN TEST value for UID {uid}")
                continue

            # Compute source CTT (mean of retained source items)
            source_responses = row[source_retained]
            # Handle NaN values: only count non-NaN responses
            source_valid = source_responses[source_responses.notna()]
            if len(source_valid) > 0:
                source_ctt = source_valid.mean()
            else:
                source_ctt = np.nan

            # Compute destination CTT (mean of retained destination items)
            dest_responses = row[dest_retained]
            dest_valid = dest_responses[dest_responses.notna()]
            if len(dest_valid) > 0:
                dest_ctt = dest_valid.mean()
            else:
                dest_ctt = np.nan

            # Append results
            results.append({
                'UID': uid,
                'test': test,
                'location_type': 'source',
                'ctt_purified_score': source_ctt
            })
            results.append({
                'UID': uid,
                'test': test,
                'location_type': 'destination',
                'ctt_purified_score': dest_ctt
            })

        df_purified_ctt = pd.DataFrame(results)
        log(f"Purified CTT scores: {len(df_purified_ctt)} rows")
        # Validate Purified CTT Scores
        # Validates: row count, value range, no NaN values
        # Threshold: correlation with Full CTT > 0.85

        log("Validating purified CTT scores...")

        # Check row count
        expected_rows = 800  # 100 UID x 4 tests x 2 location_types
        if len(df_purified_ctt) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(df_purified_ctt)}")
            sys.exit(1)

        # Check for NaN values
        nan_count = df_purified_ctt['ctt_purified_score'].isna().sum()
        if nan_count > 0:
            log(f"Found {nan_count} NaN values in ctt_purified_score")
            sys.exit(1)

        # Check value range [0, 1]
        min_score = df_purified_ctt['ctt_purified_score'].min()
        max_score = df_purified_ctt['ctt_purified_score'].max()
        if min_score < 0.0 or max_score > 1.0:
            log(f"CTT scores out of [0,1] range: min={min_score:.4f}, max={max_score:.4f}")
            sys.exit(1)

        log(f"Row count: {len(df_purified_ctt)} = {expected_rows}")
        log(f"No NaN values")
        log(f"CTT scores in [0,1]: min={min_score:.4f}, max={max_score:.4f}")

        # Validate correlation with Full CTT
        log("Computing correlation with Full CTT...")
        full_ctt_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        df_full_ctt = pd.read_csv(full_ctt_path, encoding='utf-8')

        # Merge on UID, test, location_type
        df_merged = pd.merge(
            df_purified_ctt,
            df_full_ctt,
            on=['UID', 'test', 'location_type'],
            how='inner'
        )

        if len(df_merged) != expected_rows:
            log(f"Merge mismatch: expected {expected_rows} rows, got {len(df_merged)}")
            sys.exit(1)

        # Compute correlation
        corr = df_merged['ctt_purified_score'].corr(df_merged['ctt_full_score'])
        log(f"Purified vs Full CTT: r = {corr:.4f}")

        if corr < 0.85:
            log(f"Correlation below expected threshold (r < 0.85)")
            log(f"This may indicate poor convergence between purified and full CTT")
        else:
            log(f"Correlation r = {corr:.4f} > 0.85 (high convergence)")
        # Save Purified CTT Scores
        # Output: CSV file with purified CTT scores
        # Contains: UID, test, location_type, ctt_purified_score

        log("Saving purified CTT scores...")
        output_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        df_purified_ctt.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"{len(df_purified_ctt)} rows, {len(df_purified_ctt.columns)} columns")

        # Report summary statistics
        log("Purified CTT Score Statistics:")
        for loc_type in ['source', 'destination']:
            subset = df_purified_ctt[df_purified_ctt['location_type'] == loc_type]['ctt_purified_score']
            log(f"  {loc_type.capitalize()}: mean={subset.mean():.4f}, SD={subset.std():.4f}, "
                f"min={subset.min():.4f}, max={subset.max():.4f}")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
