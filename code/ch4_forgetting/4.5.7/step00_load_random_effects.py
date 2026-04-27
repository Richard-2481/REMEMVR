#!/usr/bin/env python3
"""
RQ 5.5.7 Step 00: Load Random Effects from RQ 5.5.6

Purpose: Load random effects from RQ 5.5.6 and reshape from 200 rows
(100 UID x 2 location types) to 100 rows x 4 features.

Input:
- results/ch5/5.5.6/data/step04_random_effects.csv (200 rows)
  Columns: UID, location, random_intercept, random_slope

Output:
- data/step00_random_effects_from_rq556.csv (100 rows)
  Columns: UID, Source_intercept, Source_slope, Destination_intercept, Destination_slope
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
        logging.FileHandler(LOG_DIR / "step00_load_random_effects.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Load and reshape random effects from RQ 5.5.6."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 00: Load Random Effects from RQ 5.5.6")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load input data from RQ 5.5.6
    # -------------------------------------------------------------------------
    input_path = Path("results/ch5/5.5.6/data/step04_random_effects.csv")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("EXPECTATIONS ERROR: RQ 5.5.6 must complete before 5.5.7")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from RQ 5.5.6")
    logger.info(f"Columns: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # 2. Validate input structure
    # -------------------------------------------------------------------------
    expected_cols = ['UID', 'location', 'random_intercept', 'random_slope']

    if list(df.columns) != expected_cols:
        logger.error(f"Column mismatch: expected {expected_cols}, got {list(df.columns)}")
        sys.exit(1)

    if len(df) != 200:
        logger.error(f"Expected 200 rows (100 UID x 2 locations), got {len(df)}")
        sys.exit(1)

    # Check location values
    locations = df['location'].unique()
    if set(locations) != {'Source', 'Destination'}:
        logger.error(f"Expected locations ['Source', 'Destination'], got {list(locations)}")
        sys.exit(1)

    logger.info("Input validation PASSED: 200 rows, 4 columns, correct locations")

    # -------------------------------------------------------------------------
    # 3. Reshape from long to wide format (pivot)
    # -------------------------------------------------------------------------
    logger.info("Reshaping from long (200 rows) to wide (100 rows)...")

    # Pivot the data
    df_wide = df.pivot(
        index='UID',
        columns='location',
        values=['random_intercept', 'random_slope']
    )

    # Flatten multi-index columns
    df_wide.columns = [f"{loc}_{stat}" for stat, loc in df_wide.columns]

    # Reset index to make UID a column
    df_wide = df_wide.reset_index()

    # Rename columns to match specification
    df_wide = df_wide.rename(columns={
        'Source_random_intercept': 'Source_intercept',
        'Source_random_slope': 'Source_slope',
        'Destination_random_intercept': 'Destination_intercept',
        'Destination_random_slope': 'Destination_slope'
    })

    # Reorder columns to match spec
    df_wide = df_wide[['UID', 'Source_intercept', 'Source_slope',
                        'Destination_intercept', 'Destination_slope']]

    logger.info(f"Reshaped to {len(df_wide)} rows x {len(df_wide.columns)} columns")

    # -------------------------------------------------------------------------
    # 4. Validate reshaping
    # -------------------------------------------------------------------------
    if len(df_wide) != 100:
        logger.error(f"Expected 100 rows after reshaping, got {len(df_wide)}")
        sys.exit(1)

    expected_wide_cols = ['UID', 'Source_intercept', 'Source_slope',
                          'Destination_intercept', 'Destination_slope']
    if list(df_wide.columns) != expected_wide_cols:
        logger.error(f"Column mismatch after reshaping: {list(df_wide.columns)}")
        sys.exit(1)

    # Check for NaN values
    nan_count = df_wide.isna().sum().sum()
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN values after reshaping")
        sys.exit(1)

    logger.info("No missing values detected")

    # Check for duplicate UIDs
    if df_wide['UID'].duplicated().any():
        logger.error("Duplicate UIDs found after reshaping")
        sys.exit(1)

    logger.info("Reshaping validation PASSED: 100 rows, 5 columns, no NaN, no duplicates")

    # -------------------------------------------------------------------------
    # 5. Validate value ranges
    # -------------------------------------------------------------------------
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']

    for col in feature_cols:
        min_val = df_wide[col].min()
        max_val = df_wide[col].max()
        mean_val = df_wide[col].mean()
        std_val = df_wide[col].std()

        logger.info(f"{col}: min={min_val:.4f}, max={max_val:.4f}, "
                   f"mean={mean_val:.4f}, SD={std_val:.4f}")

        # Check intercepts in [-3, 3]
        if 'intercept' in col.lower():
            if min_val < -3 or max_val > 3:
                logger.warning(f"{col} outside expected range [-3, 3]")

        # Check slopes in [-1, 1]
        if 'slope' in col.lower():
            if min_val < -1 or max_val > 1:
                logger.warning(f"{col} outside expected range [-1, 1]")

    # -------------------------------------------------------------------------
    # 6. Save output
    # -------------------------------------------------------------------------
    output_path = DATA_DIR / "step00_random_effects_from_rq556.csv"
    df_wide.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # -------------------------------------------------------------------------
    # 7. Final validation summary
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 00 COMPLETE")
    logger.info(f"  Input:  {input_path} (200 rows)")
    logger.info(f"  Output: {output_path} (100 rows)")
    logger.info("  Features: Source_intercept, Source_slope, "
               "Destination_intercept, Destination_slope")
    logger.info("=" * 60)

    return df_wide

if __name__ == "__main__":
    main()
