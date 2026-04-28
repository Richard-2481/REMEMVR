#!/usr/bin/env python3
"""Merge Theta Scores with TSVR (Decision D070): Merge Pass 2 theta scores with TSVR mapping, reshape to long format for LMM."""

import sys
from pathlib import Path
import pandas as pd
import traceback

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file at start
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 04: Merge Theta Scores with TSVR (Decision D070)")
        # Load Input Data

        log("Loading input data...")

        # Load theta scores from Pass 2 (FINAL)
        # Expected columns: composite_ID, theta_what, theta_where, theta_when
        theta_path = RQ_DIR / "data" / "step03_theta_scores.csv"
        df_theta = pd.read_csv(theta_path, encoding='utf-8')
        log(f"step03_theta_scores.csv ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"         Columns: {list(df_theta.columns)}")

        # Load TSVR mapping from Step 0
        # Expected columns: composite_ID, UID, test, TSVR_hours
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"         Columns: {list(df_tsvr.columns)}")
        # Merge on composite_ID
        # Method: Inner join - only keep composite_IDs present in both files

        log("Merging theta scores with TSVR mapping on composite_ID...")

        # Inner merge to combine theta scores with TSVR data
        df_merged = pd.merge(
            df_theta,
            df_tsvr,
            on='composite_ID',
            how='inner'  # Only keep matching composite_IDs
        )
        log(f"Result has {len(df_merged)} rows")

        # Check for merge loss
        theta_ids = set(df_theta['composite_ID'])
        merged_ids = set(df_merged['composite_ID'])
        missing_ids = theta_ids - merged_ids
        if missing_ids:
            log(f"{len(missing_ids)} composite_IDs lost in merge: {list(missing_ids)[:5]}...")
        else:
            log("No composite_IDs lost in merge")
        # Melt to Long Format
        # Method: pandas melt with theta columns as value variables
        # Result: One row per composite_ID x domain combination

        log("Melting to long format (one row per composite_ID x domain)...")

        # Define melt parameters per 4_analysis.yaml
        id_vars = ['composite_ID', 'UID', 'test', 'TSVR_hours']
        value_vars = ['theta_what', 'theta_where', 'theta_when']

        df_long = pd.melt(
            df_merged,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='domain',
            value_name='theta'
        )
        log(f"Result has {len(df_long)} rows (expected: {len(df_merged)} x 3 = {len(df_merged) * 3})")
        # Clean Domain Column
        # Method: String replacement per 4_analysis.yaml domain_cleanup

        log("Cleaning domain column (theta_what -> what, etc.)...")

        domain_cleanup = {
            'theta_what': 'what',
            'theta_where': 'where',
            'theta_when': 'when'
        }
        df_long['domain'] = df_long['domain'].replace(domain_cleanup)

        unique_domains = df_long['domain'].unique()
        log(f"Domain values: {list(unique_domains)}")
        # Save Output
        # Output: Long-format LMM input ready for Step 5 trajectory analysis
        # Downstream usage: LMM model fitting with Domain x Time interaction

        output_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_long.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"        {len(df_long)} rows, {len(df_long.columns)} cols")
        log(f"        Columns: {list(df_long.columns)}")
        # Inline Validation (per 4_analysis.yaml)
        # Validates: File existence, row counts, data integrity, domain coverage
        # Threshold: All CRITICAL criteria must pass

        log("Running inline validation criteria...")

        validation_errors = []

        # Criterion 1: Output file exists (CRITICAL)
        if not output_path.exists():
            validation_errors.append("CRITICAL: Output file does not exist")
        else:
            log("Output file exists")

        # Criterion 2: Row count ~1200 (CRITICAL)
        expected_rows = len(df_merged) * 3
        actual_rows = len(df_long)
        if actual_rows != expected_rows:
            validation_errors.append(f"CRITICAL: Row count mismatch (expected {expected_rows}, got {actual_rows})")
        else:
            log(f"Row count correct ({actual_rows} = {len(df_merged)} x 3)")

        # Criterion 3: No merge loss (CRITICAL)
        if missing_ids:
            validation_errors.append(f"CRITICAL: {len(missing_ids)} composite_IDs lost in merge")
        else:
            log("No merge loss")

        # Criterion 4: Domain coverage (CRITICAL)
        domain_counts = df_long.groupby('composite_ID')['domain'].count()
        incorrect_counts = domain_counts[domain_counts != 3]
        if len(incorrect_counts) > 0:
            validation_errors.append(f"CRITICAL: {len(incorrect_counts)} composite_IDs don't have exactly 3 domains")
        else:
            log("Each composite_ID appears exactly 3 times (once per domain)")

        # Criterion 5: No NaN in TSVR_hours (CRITICAL)
        nan_tsvr = df_long['TSVR_hours'].isna().sum()
        if nan_tsvr > 0:
            validation_errors.append(f"CRITICAL: {nan_tsvr} NaN values in TSVR_hours")
        else:
            log("No NaN in TSVR_hours")

        # Criterion 6: No NaN in theta (CRITICAL)
        nan_theta = df_long['theta'].isna().sum()
        if nan_theta > 0:
            validation_errors.append(f"CRITICAL: {nan_theta} NaN values in theta")
        else:
            log("No NaN in theta")

        # Criterion 7: Domain values (CRITICAL)
        expected_domains = {'what', 'where', 'when'}
        actual_domains = set(df_long['domain'].unique())
        if actual_domains != expected_domains:
            validation_errors.append(f"CRITICAL: Domain values incorrect (expected {expected_domains}, got {actual_domains})")
        else:
            log("Domain values correct (what, where, when)")

        # Criterion 8: TSVR range [0, 200] (MODERATE)
        tsvr_min = df_long['TSVR_hours'].min()
        tsvr_max = df_long['TSVR_hours'].max()
        if tsvr_min < 0 or tsvr_max > 200:
            log(f"TSVR_hours range outside [0, 200]: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
        else:
            log(f"TSVR_hours range valid: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

        # Report validation summary
        if validation_errors:
            log("[VALIDATION FAILED] Critical errors found:")
            for err in validation_errors:
                log(f"  - {err}")
            raise ValueError(f"Validation failed: {validation_errors[0]}")
        else:
            log("[VALIDATION PASSED] All criteria met")
        # Report Summary Statistics

        log("")
        log("Output Data Statistics:")
        log(f"  - Total rows: {len(df_long)}")
        log(f"  - Unique composite_IDs: {df_long['composite_ID'].nunique()}")
        log(f"  - Unique UIDs: {df_long['UID'].nunique()}")
        log(f"  - Test sessions: {sorted(df_long['test'].unique())}")
        log(f"  - Domains: {sorted(df_long['domain'].unique())}")
        log(f"  - TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")
        log(f"  - theta range: [{df_long['theta'].min():.4f}, {df_long['theta'].max():.4f}]")
        log(f"  - theta mean: {df_long['theta'].mean():.4f}")
        log(f"  - theta std: {df_long['theta'].std():.4f}")

        # Per-domain statistics
        log("")
        log("Per-Domain Statistics:")
        for domain in ['what', 'where', 'when']:
            domain_data = df_long[df_long['domain'] == domain]
            log(f"  - {domain}: n={len(domain_data)}, mean_theta={domain_data['theta'].mean():.4f}, std={domain_data['theta'].std():.4f}")

        log("")
        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
