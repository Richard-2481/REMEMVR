#!/usr/bin/env python3
"""
RQ 5.5.7 Step 01: Standardize Features to Z-Scores

Purpose: Standardize all 4 features to z-scores (mean=0, SD=1) to equalize
scale across intercepts (theta scale) and slopes (theta/day scale).

Input:
- data/step00_random_effects_from_rq556.csv (100 rows)
  Columns: UID, Source_intercept, Source_slope, Destination_intercept, Destination_slope

Output:
- data/step01_standardized_features.csv (100 rows)
  Same columns, z-scored (mean ~ 0, SD ~ 1)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "step01_standardize_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Standardize features to z-scores."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 01: Standardize Features to Z-Scores")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load input data
    # -------------------------------------------------------------------------
    input_path = DATA_DIR / "step00_random_effects_from_rq556.csv"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run Step 00 first")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from Step 00")

    # -------------------------------------------------------------------------
    # 2. Validate input
    # -------------------------------------------------------------------------
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']

    if len(df) != 100:
        logger.error(f"Expected 100 rows, got {len(df)}")
        sys.exit(1)

    for col in feature_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            sys.exit(1)

    logger.info("Input validation PASSED")

    # -------------------------------------------------------------------------
    # 3. Standardize features to z-scores
    # -------------------------------------------------------------------------
    logger.info("Standardizing 4 features to z-scores...")

    df_std = df.copy()

    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=0)  # Population SD for z-score

        if std_val == 0:
            logger.error(f"Column {col} has zero variance, cannot standardize")
            sys.exit(1)

        df_std[col] = (df[col] - mean_val) / std_val

        logger.info(f"  {col}: original mean={mean_val:.6f}, SD={std_val:.6f}")

    logger.info("Standardized 4 features to z-scores")

    # -------------------------------------------------------------------------
    # 4. Validate z-score transformation
    # -------------------------------------------------------------------------
    logger.info("Validating z-score transformation...")

    tolerance = 0.01
    all_pass = True

    for col in feature_cols:
        z_mean = df_std[col].mean()
        z_std = df_std[col].std(ddof=0)

        mean_ok = abs(z_mean) < tolerance
        std_ok = abs(z_std - 1.0) < tolerance

        status_mean = "PASS" if mean_ok else "FAIL"
        status_std = "PASS" if std_ok else "FAIL"

        logger.info(f"  {col}: mean={z_mean:.6f} [{status_mean}], "
                   f"SD={z_std:.6f} [{status_std}]")

        if not mean_ok or not std_ok:
            all_pass = False

    if all_pass:
        logger.info("Mean check: all features in [-0.01, 0.01] - PASS")
        logger.info("SD check: all features in [0.99, 1.01] - PASS")
    else:
        logger.error("Z-score validation FAILED")
        sys.exit(1)

    # Check for NaN values
    if df_std.isna().any().any():
        logger.error("NaN values detected after standardization")
        sys.exit(1)

    logger.info("No NaN after standardization - PASS")

    # -------------------------------------------------------------------------
    # 5. Save output
    # -------------------------------------------------------------------------
    output_path = DATA_DIR / "step01_standardized_features.csv"
    df_std.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # -------------------------------------------------------------------------
    # 6. Summary statistics for z-scored data
    # -------------------------------------------------------------------------
    logger.info("\nZ-scored feature summary:")
    for col in feature_cols:
        min_val = df_std[col].min()
        max_val = df_std[col].max()
        logger.info(f"  {col}: z_min={min_val:.3f}, z_max={max_val:.3f}")

    # -------------------------------------------------------------------------
    # 7. Final summary
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 01 COMPLETE")
    logger.info(f"  Input:  {input_path} (100 rows, original scale)")
    logger.info(f"  Output: {output_path} (100 rows, z-scored)")
    logger.info("  All 4 features standardized: mean ~ 0, SD ~ 1")
    logger.info("=" * 60)

    return df_std

if __name__ == "__main__":
    main()
