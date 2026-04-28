#!/usr/bin/env python3
"""Compute Purified CTT Scores: Calculate Classical Test Theory (CTT) scores using ONLY IRT-retained items"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_compute_purified_ctt.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Compute Purified CTT Scores")
        # Load Input Data
        #           and item mapping with retention status from RQ 5.1 (Step 1)

        log("Loading input data...")

        # Load raw scores with all TQ_* item columns
        # Contains dichotomized responses (0/1) for ~105 VR items
        df_raw = pd.read_csv(RQ_DIR / "data/step00_raw_scores.csv")
        log(f"step00_raw_scores.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Load item mapping with retention status
        # Identifies which items were retained by RQ 5.1 IRT purification (a > 0.4, |b| < 3.0)
        df_mapping = pd.read_csv(RQ_DIR / "data/step01_item_mapping.csv")
        log(f"step01_item_mapping.csv ({len(df_mapping)} rows)")

        # Report retention counts per domain (What/Where only)
        retention_counts = df_mapping.groupby('domain')['retained'].value_counts().unstack(fill_value=0)
        log(f"Item retention by domain (What/Where only - When excluded):")
        log(f"  What:  {retention_counts.loc['what', True] if 'what' in retention_counts.index else 0} retained, {retention_counts.loc['what', False] if 'what' in retention_counts.index else 0} removed")
        log(f"  Where: {retention_counts.loc['where', True] if 'where' in retention_counts.index else 0} retained, {retention_counts.loc['where', False] if 'where' in retention_counts.index else 0} removed")
        log(f"  When:  EXCLUDED (floor effect in RQ 5.2.1)")

        total_retained = df_mapping['retained'].sum()
        total_items = len(df_mapping)
        retention_rate = total_retained / total_items
        log(f"Overall retention: {total_retained}/{total_items} items ({retention_rate:.1%})")
        # Compute Purified CTT Scores
        # For each domain: Filter to ONLY retained items, compute mean per composite_ID
        # This creates CTT scores using high-quality items only (IRT-informed selection)

        log("Computing purified CTT scores per domain...")
        log("When domain EXCLUDED per RQ 5.2.1 floor effect")

        # Initialize output DataFrame with composite_ID, UID, TEST
        df_output = df_raw[['composite_ID', 'UID', 'TEST']].copy()

        # Process each domain (What/Where only - When excluded)
        for domain in ['what', 'where']:
            # Get retained items for this domain
            retained_items = df_mapping[
                (df_mapping['domain'] == domain) &
                (df_mapping['retained'] == True)
            ]['item_name'].tolist()

            log(f"Domain '{domain}': {len(retained_items)} retained items")

            # Select only retained item columns from raw data
            # Handle case where item might not be in raw data columns
            available_items = [item for item in retained_items if item in df_raw.columns]

            if len(available_items) < len(retained_items):
                missing_items = set(retained_items) - set(available_items)
                log(f"Domain '{domain}': {len(missing_items)} retained items not found in raw data: {missing_items}")

            # Compute mean of retained items per composite_ID
            # Mean of dichotomized responses (0/1) = proportion correct
            df_output[f'CTT_purified_{domain}'] = df_raw[available_items].mean(axis=1)

            # Report score statistics
            mean_score = df_output[f'CTT_purified_{domain}'].mean()
            sd_score = df_output[f'CTT_purified_{domain}'].std()
            min_score = df_output[f'CTT_purified_{domain}'].min()
            max_score = df_output[f'CTT_purified_{domain}'].max()

            log(f"CTT_purified_{domain}: mean={mean_score:.3f}, SD={sd_score:.3f}, range=[{min_score:.3f}, {max_score:.3f}]")

        log("Purified CTT computation complete")
        # Save Purified CTT Scores
        # Output will be used by Step 5 (correlation analysis) and Step 7 (parallel LMM)

        log("Saving purified CTT scores...")
        output_path = RQ_DIR / "data/step03_ctt_purified_scores.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")

        # Report output structure
        log(f"Output columns: {df_output.columns.tolist()}")
        # Validation
        # Validate purified CTT scores are in valid proportion range [0, 1]
        # Tests: No negative values, no values > 1.0, scores are proportions

        log("Running validate_numeric_range for purified CTT scores...")

        validation_passed = True

        for domain in ['what', 'where']:
            col_name = f'CTT_purified_{domain}'

            result = validate_numeric_range(
                data=df_output[col_name],
                min_val=0.0,
                max_val=1.0,
                column_name=col_name
            )

            if result['valid']:
                log(f"{col_name}: PASS - {result['message']}")
            else:
                log(f"{col_name}: FAIL - {result['message']}")
                log(f"Out of range count: {result['out_of_range_count']}")
                if result['violations']:
                    log(f"Sample violations: {result['violations'][:5]}")
                validation_passed = False

        if not validation_passed:
            log("Validation failed - purified CTT scores out of valid range [0, 1]")
            sys.exit(1)

        log("All purified CTT scores validated successfully")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
