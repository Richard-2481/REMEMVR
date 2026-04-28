#!/usr/bin/env python3
"""Standardize Outcomes for Parallel LMM: Standardize all three measurement approaches (Full CTT, Purified CTT, IRT theta)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_standardize_outcomes.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Standardize Outcomes for Parallel LMM")
        # Load Input Data

        log("Loading input data...")

        # Load IRT theta scores (from RQ 5.1 Step 3 Pass 2)
        df_theta = pd.read_csv(RQ_DIR / "data/step00_theta_scores.csv")
        log(f"step00_theta_scores.csv ({len(df_theta)} rows, {len(df_theta.columns)} cols)")

        # Load full CTT scores (from Step 2)
        df_full_ctt = pd.read_csv(RQ_DIR / "data/step02_ctt_full_scores.csv")
        log(f"step02_ctt_full_scores.csv ({len(df_full_ctt)} rows, {len(df_full_ctt.columns)} cols)")

        # Load purified CTT scores (from Step 3)
        df_purified_ctt = pd.read_csv(RQ_DIR / "data/step03_ctt_purified_scores.csv")
        log(f"step03_ctt_purified_scores.csv ({len(df_purified_ctt)} rows, {len(df_purified_ctt.columns)} cols)")

        # Load TSVR mapping (from RQ 5.1 Step 0)
        df_tsvr = pd.read_csv(RQ_DIR / "data/step00_tsvr_mapping.csv")
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        # Merge Data on composite_ID

        log("Merging all measurement approaches on composite_ID...")

        # Start with TSVR (includes UID which we need for grouping)
        df_merged = df_tsvr.copy()

        # Merge theta scores (What/Where only - no When)
        df_merged = df_merged.merge(
            df_theta[['composite_ID', 'theta_what', 'theta_where']],
            on='composite_ID',
            how='left'
        )

        # Merge full CTT scores (What/Where only - no When)
        df_merged = df_merged.merge(
            df_full_ctt[['composite_ID', 'CTT_full_what', 'CTT_full_where']],
            on='composite_ID',
            how='left'
        )

        # Merge purified CTT scores (What/Where only - no When)
        df_merged = df_merged.merge(
            df_purified_ctt[['composite_ID', 'CTT_purified_what', 'CTT_purified_where']],
            on='composite_ID',
            how='left'
        )

        log(f"All data merged ({len(df_merged)} rows, {len(df_merged.columns)} cols)")

        # Check for merge failures (NaN values)
        na_counts = df_merged.isna().sum()
        if na_counts.sum() > 0:
            log(f"Merge produced NaN values:")
            for col, count in na_counts[na_counts > 0].items():
                log(f"  - {col}: {count} NaN values")
        # Reshape to Long Format

        log("Converting to long format (2 rows per composite_ID, 1 per domain)...")
        log("When domain EXCLUDED per RQ 5.2.1 floor effect")

        # Create list to store long-format data
        rows = []

        for _, row in df_merged.iterrows():
            # Process each domain (What/Where only - When excluded)
            for domain in ['what', 'where']:
                rows.append({
                    'composite_ID': row['composite_ID'],
                    'UID': row['UID'],
                    'TSVR_hours': row['TSVR_hours'],
                    'domain': domain,
                    'theta_raw': row[f'theta_{domain}'],
                    'ctt_full_raw': row[f'CTT_full_{domain}'],
                    'ctt_purified_raw': row[f'CTT_purified_{domain}']
                })

        df_long = pd.DataFrame(rows)
        log(f"Long format created ({len(df_long)} rows = {len(df_merged)} composite_IDs x 2 domains)")
        # Compute Z-Scores per Measurement Type x Domain

        log("Computing z-scores per measurement type x domain...")

        # Group by domain and compute z-scores for each measurement type
        # Z-score formula: (value - mean) / sd

        for domain in ['what', 'where']:
            domain_mask = df_long['domain'] == domain

            # IRT theta z-scores
            theta_vals = df_long.loc[domain_mask, 'theta_raw']
            theta_mean = theta_vals.mean()
            theta_std = theta_vals.std()
            df_long.loc[domain_mask, 'z_irt_theta'] = (theta_vals - theta_mean) / theta_std
            log(f"{domain} - IRT theta: mean={theta_mean:.4f}, SD={theta_std:.4f}")

            # Full CTT z-scores
            full_vals = df_long.loc[domain_mask, 'ctt_full_raw']
            full_mean = full_vals.mean()
            full_std = full_vals.std()
            df_long.loc[domain_mask, 'z_full_ctt'] = (full_vals - full_mean) / full_std
            log(f"{domain} - Full CTT: mean={full_mean:.4f}, SD={full_std:.4f}")

            # Purified CTT z-scores
            purified_vals = df_long.loc[domain_mask, 'ctt_purified_raw']
            purified_mean = purified_vals.mean()
            purified_std = purified_vals.std()
            df_long.loc[domain_mask, 'z_purified_ctt'] = (purified_vals - purified_mean) / purified_std
            log(f"{domain} - Purified CTT: mean={purified_mean:.4f}, SD={purified_std:.4f}")
        # Save Standardized Outcomes
        # Output: data/step06_standardized_outcomes.csv
        # Contains: composite_ID, UID, TSVR_hours, domain, z_full_ctt, z_purified_ctt, z_irt_theta

        log("Saving standardized outcomes...")

        # Select final columns for output
        df_output = df_long[['composite_ID', 'UID', 'TSVR_hours', 'domain', 'z_full_ctt', 'z_purified_ctt', 'z_irt_theta']]

        output_path = RQ_DIR / "data/step06_standardized_outcomes.csv"
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_output)} rows, {len(df_output.columns)} cols)")
        # Run Validation Tool
        # Validates: mean ≈ 0, SD ≈ 1 for all z-score columns (tolerance ±0.01)
        # Threshold: Passes only if ALL 9 combinations (3 measurements x 3 domains) meet criteria

        log("Running validate_standardization...")

        validation_result = validate_standardization(
            df=df_output,
            column_names=['z_full_ctt', 'z_purified_ctt', 'z_irt_theta'],
            tolerance=0.01
        )

        # Report validation results
        if validation_result['valid']:
            log("PASS - All z-scores meet standardization criteria (mean ≈ 0, SD ≈ 1)")
            log(f"Mean values: {validation_result['mean_values']}")
            log(f"SD values: {validation_result['sd_values']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            log(f"Mean values: {validation_result['mean_values']}")
            log(f"SD values: {validation_result['sd_values']}")
            raise ValueError(f"Standardization validation failed: {validation_result['message']}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
