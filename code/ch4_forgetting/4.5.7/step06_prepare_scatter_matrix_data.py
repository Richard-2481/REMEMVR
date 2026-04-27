#!/usr/bin/env python3
"""
RQ 5.5.7 Step 06: Create Scatter Plot Matrix Data

Purpose: Create plot source CSV for 4x4 scatter plot matrix showing
pairwise relationships among 4 features, colored by cluster membership.

Input:
- data/step01_standardized_features.csv (100 rows, z-scored)
- data/step04_cluster_assignments.csv (100 rows: UID, cluster)

Output:
- plots/step06_cluster_scatter_matrix_data.csv (100 rows: UID, 4 features, cluster)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
PLOTS_DIR = RQ_DIR / "plots"
LOG_DIR = RQ_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "step06_prepare_scatter_matrix_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Create plot source data for scatter matrix."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 06: Create Scatter Plot Matrix Data")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load inputs
    # -------------------------------------------------------------------------
    features_path = DATA_DIR / "step01_standardized_features.csv"
    assignments_path = DATA_DIR / "step04_cluster_assignments.csv"

    if not features_path.exists():
        logger.error(f"Input file not found: {features_path}")
        sys.exit(1)

    if not assignments_path.exists():
        logger.error(f"Input file not found: {assignments_path}")
        sys.exit(1)

    df_features = pd.read_csv(features_path)
    df_assignments = pd.read_csv(assignments_path)

    logger.info(f"Loaded {len(df_features)} rows (z-scored features)")
    logger.info(f"Loaded {len(df_assignments)} cluster assignments")

    # -------------------------------------------------------------------------
    # 2. Merge data
    # -------------------------------------------------------------------------
    df = df_features.merge(df_assignments, on='UID')
    logger.info(f"Merged data: {len(df)} rows")

    # Reorder columns to match spec
    expected_cols = ['UID', 'Source_intercept', 'Source_slope',
                     'Destination_intercept', 'Destination_slope', 'cluster']
    df = df[expected_cols]

    # -------------------------------------------------------------------------
    # 3. Validate data
    # -------------------------------------------------------------------------
    logger.info("\nValidating plot data...")

    # Check row count
    if len(df) != 100:
        logger.error(f"Expected 100 rows, got {len(df)}")
        sys.exit(1)
    logger.info("Plot data prepared: 100 participants")

    # Check columns
    if list(df.columns) != expected_cols:
        logger.error(f"Column mismatch: expected {expected_cols}, got {list(df.columns)}")
        sys.exit(1)
    logger.info("Columns match specification: PASS")

    # Check no NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN values in plot data")
        sys.exit(1)
    logger.info("No NaN values in features or cluster column: PASS")

    # Check all clusters represented
    K = df['cluster'].nunique()
    expected_clusters = set(range(K))
    actual_clusters = set(df['cluster'].unique())

    if expected_clusters != actual_clusters:
        logger.error(f"Missing clusters: expected {expected_clusters}, got {actual_clusters}")
        sys.exit(1)
    logger.info(f"All {K} clusters represented in plot data: PASS")

    # Check no duplicate UIDs
    if df['UID'].duplicated().any():
        logger.error("Duplicate UIDs found in plot data")
        sys.exit(1)
    logger.info("No duplicate UIDs: PASS")

    # Check feature ranges (z-scores typically in [-3, 3])
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val < -4 or max_val > 4:
            logger.warning(f"{col} has extreme values: min={min_val:.3f}, max={max_val:.3f}")
        else:
            logger.info(f"  {col}: z_min={min_val:.3f}, z_max={max_val:.3f}")

    # -------------------------------------------------------------------------
    # 4. Save output
    # -------------------------------------------------------------------------
    output_path = PLOTS_DIR / "step06_cluster_scatter_matrix_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # -------------------------------------------------------------------------
    # 5. Print cluster distribution
    # -------------------------------------------------------------------------
    logger.info("\nCluster distribution in plot data:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        logger.info(f"  Cluster {cluster_id}: {count} participants")

    # -------------------------------------------------------------------------
    # 6. Final summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 06 COMPLETE")
    logger.info(f"  Plot data: 100 participants, {K} clusters")
    logger.info(f"  Features: 4 z-scored (Source_intercept, Source_slope, "
               "Destination_intercept, Destination_slope)")
    logger.info(f"  All clusters represented")
    logger.info(f"  Output: {output_path}")
    logger.info("  Ready for rq_plots to generate scatter matrix PNG")
    logger.info("=" * 60)

    return df

if __name__ == "__main__":
    main()
