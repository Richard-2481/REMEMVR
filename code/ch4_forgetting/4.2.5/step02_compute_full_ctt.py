#!/usr/bin/env python3
"""Compute Full CTT Scores: Calculate Classical Test Theory (CTT) scores using ALL items (full item set)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_compute_full_ctt.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Compute Full CTT Scores")
        # Load Input Data

        log("Loading input data...")

        # Load raw scores with dichotomized item responses
        df_raw = pd.read_csv(RQ_DIR / "data" / "step00_raw_scores.csv")
        log(f"step00_raw_scores.csv ({len(df_raw)} rows, {len(df_raw.columns)} cols)")

        # Load item-to-domain mapping
        df_mapping = pd.read_csv(RQ_DIR / "data" / "step01_item_mapping.csv")
        log(f"step01_item_mapping.csv ({len(df_mapping)} items)")

        # Verify required columns
        required_raw_cols = ['composite_ID', 'UID', 'TEST']
        missing_raw = [c for c in required_raw_cols if c not in df_raw.columns]
        if missing_raw:
            raise ValueError(f"Missing required columns in step00_raw_scores.csv: {missing_raw}")

        required_mapping_cols = ['item_name', 'domain']
        missing_mapping = [c for c in required_mapping_cols if c not in df_mapping.columns]
        if missing_mapping:
            raise ValueError(f"Missing required columns in step01_item_mapping.csv: {missing_mapping}")

        # Extract TQ_* item columns
        tq_cols = [c for c in df_raw.columns if c.startswith('TQ_')]
        log(f"Found {len(tq_cols)} TQ_* item columns in raw data")
        # Compute CTT Scores for Each Domain

        log("Computing full CTT scores using ALL items per domain...")
        log("When domain EXCLUDED per RQ 5.2.1 floor effect")

        # Initialize results dictionary
        ctt_scores = {'composite_ID': df_raw['composite_ID'].values}

        # Process each domain (What/Where only - When excluded)
        domains = ['what', 'where']

        for domain in domains:
            log(f"Processing domain: {domain}")

            # Get items for this domain (ALL items, not filtered by retained status)
            domain_items = df_mapping[df_mapping['domain'] == domain]['item_name'].tolist()
            log(f"  - Domain '{domain}' has {len(domain_items)} items (full item set)")

            # Verify all domain items exist in raw data
            missing_items = [item for item in domain_items if item not in df_raw.columns]
            if missing_items:
                log(f"  {len(missing_items)} items from mapping not found in raw data: {missing_items[:5]}...")
                # Filter to existing items only
                domain_items = [item for item in domain_items if item in df_raw.columns]
                log(f"  - Using {len(domain_items)} items that exist in raw data")

            if len(domain_items) == 0:
                raise ValueError(f"No items found for domain '{domain}' in raw data")

            # Compute mean across domain items (proportion correct)
            # axis=1 computes row-wise mean (across items for each participant)
            df_raw[f'CTT_full_{domain}'] = df_raw[domain_items].mean(axis=1)

            # Count non-NaN values
            non_nan_count = df_raw[f'CTT_full_{domain}'].notna().sum()
            log(f"  - Computed CTT_full_{domain}: {non_nan_count}/{len(df_raw)} non-NaN scores")

            # Report score range
            score_min = df_raw[f'CTT_full_{domain}'].min()
            score_max = df_raw[f'CTT_full_{domain}'].max()
            score_mean = df_raw[f'CTT_full_{domain}'].mean()
            log(f"  - Score range: [{score_min:.3f}, {score_max:.3f}], mean = {score_mean:.3f}")

        log("All domain CTT scores computed")
        # Save Analysis Outputs
        # These outputs will be used by: Step 5 (correlation analysis), Step 6 (standardization)

        log("Saving CTT full scores...")

        # Select output columns (What/Where only - no When)
        output_cols = ['composite_ID', 'UID', 'TEST', 'CTT_full_what', 'CTT_full_where']
        df_output = df_raw[output_cols].copy()

        # Save to CSV
        output_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step02_ctt_full_scores.csv ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Run Validation Tool
        # Validates: CTT scores in [0, 1] range (proportion correct)
        # Threshold: min=0.0, max=1.0 (no negative values, no values > 1)

        log("Running validate_numeric_range on CTT scores...")

        validation_passed = True

        for domain in domains:
            col_name = f'CTT_full_{domain}'

            # Run validation
            result = validate_numeric_range(
                data=df_output[col_name],
                min_val=0.0,
                max_val=1.0,
                column_name=col_name
            )

            # Check validation result
            if result['valid']:
                log(f"{col_name}: PASS (all values in [0, 1])")
            else:
                log(f"{col_name}: FAIL - {result['message']}")
                log(f"  - Out of range count: {result['out_of_range_count']}")
                if result['violations']:
                    log(f"  - First violations: {result['violations'][:5]}")
                validation_passed = False

        if not validation_passed:
            raise ValueError("Validation failed: CTT scores out of valid range [0, 1]")

        log("All CTT scores validated successfully")
        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
