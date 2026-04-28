#!/usr/bin/env python3
"""Merge Theta with TSVR Time Data: Merge theta estimates from Pass 2 IRT calibration with TSVR time variable"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Merge Theta with TSVR Time Data")
        # Load Input Data
        #           TSVR mapping with hours since encoding (Decision D070)

        log("Loading Pass 2 theta estimates (wide format)...")
        input_theta_path = RQ_DIR / "data/step03_pass2_theta.csv"
        df_theta = pd.read_csv(input_theta_path, encoding='utf-8')
        log(f"{input_theta_path.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"Theta columns: {df_theta.columns.tolist()}")

        log("Loading TSVR time mapping...")
        input_tsvr_path = RQ_DIR / "data/step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(input_tsvr_path, encoding='utf-8')
        log(f"{input_tsvr_path.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"TSVR columns: {df_tsvr.columns.tolist()}")
        # Merge Theta with TSVR (Left Join on composite_ID)

        log("Merging theta with TSVR on composite_ID (left join)...")
        df_merged = pd.merge(
            df_theta,
            df_tsvr,
            on='composite_ID',
            how='left',
            validate='1:1'  # Ensure one-to-one mapping (no duplicates)
        )
        log(f"Result: {len(df_merged)} rows, {len(df_merged.columns)} cols")

        # Validate merge completeness (no missing TSVR data)
        n_missing_tsvr = df_merged['TSVR_hours'].isna().sum()
        if n_missing_tsvr > 0:
            log(f"Found {n_missing_tsvr} rows with missing TSVR_hours (merge failed)")
            sys.exit(1)
        else:
            log(f"100% merge rate - all {len(df_merged)} theta observations matched with TSVR")
        # Reshape from Wide to Long Format (Melt)

        log("Melting wide theta format to long format...")
        df_long = pd.melt(
            df_merged,
            id_vars=['composite_ID', 'TSVR_hours', 'test'],
            value_vars=['theta_What', 'theta_Where', 'theta_When'],
            var_name='domain',
            value_name='theta'
        )
        log(f"Result: {len(df_long)} rows (expected: {len(df_merged) * 3})")

        # Clean domain names (remove "theta_" prefix)
        df_long['domain'] = df_long['domain'].str.replace('theta_', '')
        log(f"Domains: {df_long['domain'].unique().tolist()}")
        # Parse UID from composite_ID and Add Time Transformations

        log("Extracting UID from composite_ID...")
        df_long['UID'] = df_long['composite_ID'].str.split('_').str[0]
        n_unique_uids = df_long['UID'].nunique()
        log(f"Extracted {n_unique_uids} unique UIDs")

        log("Adding time transformations for LMM...")
        # log_TSVR: log(TSVR_hours + 1) to handle TSVR=0 at encoding
        df_long['log_TSVR'] = np.log(df_long['TSVR_hours'] + 1)
        log(f"log_TSVR range: [{df_long['log_TSVR'].min():.3f}, {df_long['log_TSVR'].max():.3f}]")

        # Reorder columns for clarity
        df_long = df_long[[
            'composite_ID', 'UID', 'test', 'TSVR_hours', 'log_TSVR', 'domain', 'theta'
        ]]
        # Save LMM-Ready Output
        # Output: Long-format DataFrame with TSVR time variable per Decision D070
        # Contains: composite_ID, UID, test, TSVR_hours, log_TSVR, domain, theta

        output_path = RQ_DIR / "data/step04_lmm_input.csv"
        log(f"Saving LMM-ready input to {output_path.name}...")
        df_long.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_long)} rows, {len(df_long.columns)} cols)")
        # Validation Report
        # Validates: Merge completeness, row count, TSVR range, UID parsing

        log("Running inline validation checks...")

        # Check 1: Row count preserved (df_theta rows * 3 domains = df_long rows)
        expected_rows = len(df_theta) * 3
        if len(df_long) == expected_rows:
            log(f"Row count: {len(df_long)} rows = {len(df_theta)} theta rows * 3 domains")
        else:
            log(f"Row count mismatch: Expected {expected_rows}, got {len(df_long)}")
            sys.exit(1)

        # Check 2: TSVR range (note: actual data may exceed 168h due to real-world scheduling)
        tsvr_min = df_long['TSVR_hours'].min()
        tsvr_max = df_long['TSVR_hours'].max()
        if 0 <= tsvr_min and tsvr_max <= 168:
            log(f"TSVR_hours range: [{tsvr_min:.1f}, {tsvr_max:.1f}] hours (within 0-168)")
        elif 0 <= tsvr_min and tsvr_max <= 336:  # Accept up to 2 weeks
            log(f"TSVR_hours range: [{tsvr_min:.1f}, {tsvr_max:.1f}] exceeds expected 0-168 but acceptable")
        else:
            log(f"TSVR_hours range: [{tsvr_min:.1f}, {tsvr_max:.1f}] (expected 0-336)")
            sys.exit(1)

        # Check 3: log_TSVR range
        log_tsvr_min = df_long['log_TSVR'].min()
        log_tsvr_max = df_long['log_TSVR'].max()
        expected_log_max = np.log(168 + 1)  # ~5.13
        log(f"log_TSVR range: [{log_tsvr_min:.3f}, {log_tsvr_max:.3f}] (expected max ~5.13)")

        # Check 4: No missing TSVR values
        n_missing = df_long['TSVR_hours'].isna().sum()
        if n_missing == 0:
            log(f"No missing TSVR_hours (0 NaN values)")
        else:
            log(f"Found {n_missing} missing TSVR_hours values")
            sys.exit(1)

        # Check 5: UID parsing successful
        if n_unique_uids > 0:
            sample_uids = df_long['UID'].unique()[:5].tolist()
            log(f"UID parsing successful ({n_unique_uids} unique UIDs, sample: {sample_uids})")
        else:
            log(f"UID parsing failed (0 unique UIDs)")
            sys.exit(1)

        # Check 6: Domain values correct
        expected_domains = {'What', 'Where', 'When'}
        actual_domains = set(df_long['domain'].unique())
        if actual_domains == expected_domains:
            log(f"Domains correct: {sorted(actual_domains)}")
        else:
            log(f"Domain mismatch - Expected: {expected_domains}, Got: {actual_domains}")
            sys.exit(1)

        # Summary statistics
        log("Summary statistics by domain:")
        for domain in sorted(df_long['domain'].unique()):
            domain_data = df_long[df_long['domain'] == domain]
            theta_mean = domain_data['theta'].mean()
            theta_std = domain_data['theta'].std()
            n_obs = len(domain_data)
            log(f"{domain}: N={n_obs}, theta mean={theta_mean:.3f}, SD={theta_std:.3f}")

        log("Step 4 complete - LMM input ready with TSVR time variable")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
