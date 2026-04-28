#!/usr/bin/env python3
"""ctt_full_scores: Compute Classical Test Theory (CTT) sum scores using ALL source and destination"""

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
LOG_FILE = RQ_DIR / "logs" / "step02_ctt_full_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute Full CTT Sum Scores")
        # Load Input Data

        log("Loading raw binary responses...")
        dfData_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        dfData = pd.read_csv(dfData_path, encoding='utf-8')
        log(f"dfData.csv ({len(dfData)} rows, {len(dfData.columns)} cols)")

        # Load item mapping to get complete list of source and destination items
        log("Loading item mapping...")
        item_mapping_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        item_mapping = pd.read_csv(item_mapping_path, encoding='utf-8')
        log(f"step01_item_mapping.csv ({len(item_mapping)} rows)")
        # Identify Source and Destination Items

        log("Identifying source and destination items...")

        # Get item names from item_mapping (authoritative source)
        source_items = item_mapping[item_mapping['location_type'] == 'source']['item_name'].tolist()
        destination_items = item_mapping[item_mapping['location_type'] == 'destination']['item_name'].tolist()

        log(f"{len(source_items)} source items (location_type='source')")
        log(f"{len(destination_items)} destination items (location_type='destination')")

        # Verify all items exist in dfData
        missing_source = [item for item in source_items if item not in dfData.columns]
        missing_destination = [item for item in destination_items if item not in dfData.columns]

        if missing_source:
            log(f"Missing source items in dfData: {missing_source}")
            sys.exit(1)
        if missing_destination:
            log(f"Missing destination items in dfData: {missing_destination}")
            sys.exit(1)

        log("All items present in dfData.csv")
        # Compute CTT Sum Scores

        log("Computing CTT sum scores (proportion correct)...")

        # Source CTT score: mean of 18 source item responses (NaN-tolerant)
        # Values are already 0/1 binary (TQ columns pre-dichotomized)
        # NaN values are excluded from mean automatically by pandas
        dfData['ctt_source'] = dfData[source_items].mean(axis=1)

        # Destination CTT score: mean of 18 destination item responses
        dfData['ctt_destination'] = dfData[destination_items].mean(axis=1)

        log(f"Source CTT mean={dfData['ctt_source'].mean():.4f}, SD={dfData['ctt_source'].std():.4f}")
        log(f"Destination CTT mean={dfData['ctt_destination'].mean():.4f}, SD={dfData['ctt_destination'].std():.4f}")
        # Reshape to Long Format

        log("Converting to long format...")

        # Convert TEST column from numeric (1,2,3,4) to string (T1,T2,T3,T4)
        # This matches the format from theta_scores.csv (RQ 5.5.1 output)
        dfData['test'] = 'T' + dfData['TEST'].astype(str)

        # Select relevant columns for reshaping
        df_wide = dfData[['UID', 'test', 'ctt_source', 'ctt_destination']].copy()

        # Reshape to long format: UID, test, location_type, ctt_full_score
        df_long = pd.melt(
            df_wide,
            id_vars=['UID', 'test'],
            value_vars=['ctt_source', 'ctt_destination'],
            var_name='location_type',
            value_name='ctt_full_score'
        )

        # Clean up location_type column: 'ctt_source' -> 'source', 'ctt_destination' -> 'destination'
        df_long['location_type'] = df_long['location_type'].str.replace('ctt_', '')

        log(f"Long format: {len(df_long)} rows, {len(df_long.columns)} cols")
        # Validation
        # Validates: Shape, range, completeness per specification

        log("Validating output...")

        # Check 1: 800 rows exactly
        expected_rows = 800  # 100 UIDs × 4 tests × 2 location_types
        if len(df_long) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(df_long)}")
            sys.exit(1)
        log(f"Row count: {len(df_long)} rows")

        # Check 2: ctt_full_score in [0, 1]
        min_score = df_long['ctt_full_score'].min()
        max_score = df_long['ctt_full_score'].max()
        if min_score < 0 or max_score > 1:
            log(f"ctt_full_score out of range [0,1]: min={min_score}, max={max_score}")
            sys.exit(1)
        log(f"ctt_full_score in [0, 1]: min={min_score:.4f}, max={max_score:.4f}")

        # Check 3: No NaN values
        n_missing = df_long['ctt_full_score'].isna().sum()
        if n_missing > 0:
            log(f"{n_missing} NaN values in ctt_full_score")
            sys.exit(1)
        log("No NaN values in ctt_full_score")

        # Check 4: 400 rows per location_type
        location_counts = df_long['location_type'].value_counts()
        for location_type, count in location_counts.items():
            if count != 400:
                log(f"Expected 400 rows for {location_type}, got {count}")
                sys.exit(1)
        log("400 rows per location_type")

        # Check 5: 100 unique UIDs
        n_unique_uids = df_long['UID'].nunique()
        if n_unique_uids != 100:
            log(f"Expected 100 unique UIDs, got {n_unique_uids}")
            sys.exit(1)
        log("100 unique UIDs")

        # Check 6: 4 tests per UID
        tests_per_uid = df_long.groupby('UID')['test'].nunique()
        if not (tests_per_uid == 4).all():
            log(f"Not all UIDs have 4 tests: {tests_per_uid.value_counts()}")
            sys.exit(1)
        log("4 tests per UID (T1, T2, T3, T4)")

        log("All checks passed")
        # Save Output
        # Output: data/step02_ctt_full_scores.csv
        # Contains: UID, test, location_type, ctt_full_score
        # These scores will be used for: Correlation analysis (Step 5), LMM comparison (Step 7)

        log("Saving output...")
        output_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        df_long.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_long)} rows, {len(df_long.columns)} cols)")

        # Report summary statistics
        log("CTT Full Scores Summary:")
        for location_type in ['source', 'destination']:
            subset = df_long[df_long['location_type'] == location_type]
            mean_score = subset['ctt_full_score'].mean()
            sd_score = subset['ctt_full_score'].std()
            min_score_loc = subset['ctt_full_score'].min()
            max_score_loc = subset['ctt_full_score'].max()
            log(f"  {location_type.capitalize()}: mean={mean_score:.4f}, SD={sd_score:.4f}, range=[{min_score_loc:.4f}, {max_score_loc:.4f}]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
