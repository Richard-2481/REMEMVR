#!/usr/bin/env python3
"""
RQ 5.2.6 Step 00: Load and Filter Data (Exclude When Domain)

Purpose:
    Load theta scores from RQ 5.2.1 and filter out When domain due to floor effect.

Input:
    - results/ch5/5.2.1/data/step04_lmm_input.csv (1200 rows: 100 UID x 4 tests x 3 domains)

Output:
    - data/step00_lmm_input_filtered.csv (800 rows: 100 UID x 4 tests x 2 domains)

When Domain Exclusion Rationale:
    - Floor effect discovered in RQ 5.2.1
    - 77% item attrition after IRT purification (26 -> 6 items)
    - 6-9% participants at floor
    - Only What and Where domains provide reliable theta estimates

Author: Claude (g_code agent)
Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step00_load_and_filter_data.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Load RQ 5.2.1 data and filter out When domain."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 00: Load and Filter Data (Exclude When Domain)")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load source data from RQ 5.2.1
    # ---------------------------------------------------------------------
    source_path = Path("results/ch5/5.2.1/data/step04_lmm_input.csv")

    if not source_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Source file not found: {source_path}")
        logger.error("RQ 5.2.1 must complete Steps 1-4 before this RQ can run")
        sys.exit(1)

    df = pd.read_csv(source_path)
    logger.info(f"Loaded source data: {len(df)} rows from {source_path}")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate expected columns
    required_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours', 'domain', 'theta']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 2. Check domain distribution before filtering
    # ---------------------------------------------------------------------
    domain_counts_before = df['domain'].value_counts()
    logger.info(f"\nDomain distribution BEFORE filtering:")
    for domain, count in domain_counts_before.items():
        logger.info(f"  {domain}: {count} rows")

    unique_domains = set(df['domain'].unique())
    logger.info(f"Unique domains: {unique_domains}")

    # ---------------------------------------------------------------------
    # 3. Filter out When domain
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("WHEN DOMAIN EXCLUSION")
    logger.info("Reason: Floor effect in RQ 5.2.1 (77% item attrition, 6-9% floor)")
    logger.info("=" * 60)

    # Normalize domain names to lowercase for consistent filtering
    df['domain_lower'] = df['domain'].str.lower()

    # Filter to keep only what and where
    df_filtered = df[df['domain_lower'].isin(['what', 'where'])].copy()

    # Drop the temporary column
    df_filtered = df_filtered.drop(columns=['domain_lower'])

    rows_removed = len(df) - len(df_filtered)
    logger.info(f"\nFiltering results:")
    logger.info(f"  Original rows: {len(df)}")
    logger.info(f"  Rows removed (When domain): {rows_removed}")
    logger.info(f"  Remaining rows: {len(df_filtered)}")

    # ---------------------------------------------------------------------
    # 4. Validate filtered data
    # ---------------------------------------------------------------------
    domain_counts_after = df_filtered['domain'].value_counts()
    logger.info(f"\nDomain distribution AFTER filtering:")
    for domain, count in domain_counts_after.items():
        logger.info(f"  {domain}: {count} rows")

    # Validate expected row count
    n_participants = df_filtered['UID'].nunique()
    n_tests = df_filtered['test'].nunique()
    n_domains = df_filtered['domain'].nunique()
    expected_rows = n_participants * n_tests * n_domains

    logger.info(f"\nData structure:")
    logger.info(f"  Unique participants (UID): {n_participants}")
    logger.info(f"  Unique tests: {n_tests}")
    logger.info(f"  Unique domains: {n_domains}")
    logger.info(f"  Expected rows (N×T×D): {expected_rows}")
    logger.info(f"  Actual rows: {len(df_filtered)}")

    if len(df_filtered) != expected_rows:
        logger.warning(f"Row count mismatch! Expected {expected_rows}, got {len(df_filtered)}")

    # Check for missing data
    missing_theta = df_filtered['theta'].isna().sum()
    if missing_theta > 0:
        logger.warning(f"Missing theta values: {missing_theta}")
    else:
        logger.info("No missing theta values - data complete")

    # Validate TSVR_hours range
    tsvr_min = df_filtered['TSVR_hours'].min()
    tsvr_max = df_filtered['TSVR_hours'].max()
    logger.info(f"TSVR_hours range: [{tsvr_min:.1f}, {tsvr_max:.1f}] hours")

    # Validate theta range
    theta_min = df_filtered['theta'].min()
    theta_max = df_filtered['theta'].max()
    theta_mean = df_filtered['theta'].mean()
    theta_std = df_filtered['theta'].std()
    logger.info(f"Theta range: [{theta_min:.3f}, {theta_max:.3f}]")
    logger.info(f"Theta mean: {theta_mean:.3f}, SD: {theta_std:.3f}")

    # ---------------------------------------------------------------------
    # 5. Save filtered data
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step00_lmm_input_filtered.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_filtered.to_csv(output_path, index=False)
    logger.info(f"\nSaved filtered data to: {output_path}")
    logger.info(f"Output: {len(df_filtered)} rows, {len(df_filtered.columns)} columns")

    # ---------------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 00 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input:  {len(df)} rows (What + Where + When)")
    logger.info(f"Output: {len(df_filtered)} rows (What + Where only)")
    logger.info(f"When domain excluded due to floor effect")
    logger.info("Ready for Step 01: Fit Domain-Stratified LMMs")

    return df_filtered


if __name__ == "__main__":
    main()
