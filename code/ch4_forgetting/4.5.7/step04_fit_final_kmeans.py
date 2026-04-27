#!/usr/bin/env python3
"""
RQ 5.5.7 Step 04: Fit Final K-Means with Optimal K

Purpose: Fit final K-means with optimal K (random_state=42, n_init=50),
extract cluster assignments (100 UIDs with labels) and cluster centers
(K centers x 4 features).

Input:
- data/step01_standardized_features.csv (100 rows, z-scored)
- data/step02_optimal_k.txt (optimal K value)

Output:
- data/step04_cluster_assignments.csv (100 rows: UID, cluster)
- data/step04_cluster_centers.csv (K rows: cluster, 4 z-scored feature centers)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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
        logging.FileHandler(LOG_DIR / "step04_fit_final_kmeans.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Fit final K-means and extract assignments + centers."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 04: Fit Final K-Means with Optimal K")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load inputs
    # -------------------------------------------------------------------------
    features_path = DATA_DIR / "step01_standardized_features.csv"
    optimal_k_path = DATA_DIR / "step02_optimal_k.txt"

    if not features_path.exists():
        logger.error(f"Input file not found: {features_path}")
        sys.exit(1)

    if not optimal_k_path.exists():
        logger.error(f"Input file not found: {optimal_k_path}")
        sys.exit(1)

    # Load features
    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} rows from Step 01")

    # Extract feature matrix and UID
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']
    X = df[feature_cols].values
    uids = df['UID'].values

    # Load optimal K
    with open(optimal_k_path, 'r') as f:
        first_line = f.readline().strip()
        K = int(first_line.split(':')[1].strip())

    logger.info(f"Optimal K from Step 02: {K}")

    # -------------------------------------------------------------------------
    # 2. Fit K-means with optimal K
    # -------------------------------------------------------------------------
    logger.info(f"\nFitting K-means with K={K}...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=50)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    logger.info(f"K-means fitted successfully")

    # -------------------------------------------------------------------------
    # 3. Extract cluster assignments
    # -------------------------------------------------------------------------
    df_assignments = pd.DataFrame({
        'UID': uids,
        'cluster': labels
    })

    # Log cluster sizes
    cluster_sizes = df_assignments['cluster'].value_counts().sort_index()
    logger.info("\nCluster sizes:")
    for cluster_id, count in cluster_sizes.items():
        logger.info(f"  Cluster {cluster_id}: {count} participants")

    # -------------------------------------------------------------------------
    # 4. Extract cluster centers
    # -------------------------------------------------------------------------
    df_centers = pd.DataFrame(
        centers,
        columns=feature_cols
    )
    df_centers.insert(0, 'cluster', range(K))

    logger.info("\nCluster centers (z-score space):")
    for _, row in df_centers.iterrows():
        cluster_id = int(row['cluster'])
        logger.info(f"  Cluster {cluster_id}: "
                   f"S_int={row['Source_intercept']:.3f}, "
                   f"S_slope={row['Source_slope']:.3f}, "
                   f"D_int={row['Destination_intercept']:.3f}, "
                   f"D_slope={row['Destination_slope']:.3f}")

    # -------------------------------------------------------------------------
    # 5. Validate cluster sizes (min 10 participants per cluster)
    # -------------------------------------------------------------------------
    min_size = 10
    logger.info(f"\nValidating cluster sizes (minimum: {min_size})...")

    all_valid = True
    for cluster_id, count in cluster_sizes.items():
        if count < min_size:
            logger.error(f"Cluster {cluster_id} has only {count} participants "
                        f"(minimum: {min_size})")
            all_valid = False
        else:
            logger.info(f"  Cluster {cluster_id}: {count} >= {min_size} - PASS")

    if not all_valid:
        logger.error("Cluster size validation FAILED")
        sys.exit(1)

    logger.info("All clusters >= 10 participants: True")

    # -------------------------------------------------------------------------
    # 6. Validate assignments
    # -------------------------------------------------------------------------
    logger.info("\nValidating cluster assignments...")

    # Check all 100 participants assigned
    if len(df_assignments) != 100:
        logger.error(f"Expected 100 assignments, got {len(df_assignments)}")
        sys.exit(1)
    logger.info("All 100 participants assigned: PASS")

    # Check no duplicate UIDs
    if df_assignments['UID'].duplicated().any():
        logger.error("Duplicate UIDs found in assignments")
        sys.exit(1)
    logger.info("No duplicate UIDs: PASS")

    # Check cluster IDs consecutive from 0
    expected_clusters = set(range(K))
    actual_clusters = set(df_assignments['cluster'].unique())
    if expected_clusters != actual_clusters:
        logger.error(f"Expected clusters {expected_clusters}, got {actual_clusters}")
        sys.exit(1)
    logger.info(f"Cluster IDs consecutive {{0, 1, ..., {K-1}}}: PASS")

    # Check centers in reasonable z-score range
    max_center = df_centers[feature_cols].abs().max().max()
    if max_center > 3:
        logger.warning(f"Cluster center max absolute value: {max_center:.3f} (>3)")
    else:
        logger.info(f"Cluster centers in reasonable z-score range [-3, 3]: PASS")

    # -------------------------------------------------------------------------
    # 7. Save outputs
    # -------------------------------------------------------------------------
    assignments_path = DATA_DIR / "step04_cluster_assignments.csv"
    df_assignments.to_csv(assignments_path, index=False)
    logger.info(f"\nSaved cluster assignments to {assignments_path}")

    centers_path = DATA_DIR / "step04_cluster_centers.csv"
    df_centers.to_csv(centers_path, index=False)
    logger.info(f"Saved cluster centers to {centers_path}")

    # -------------------------------------------------------------------------
    # 8. Final summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 04 COMPLETE")
    logger.info(f"  Fitted K-means with K={K}")
    logger.info(f"  Cluster sizes: {dict(cluster_sizes)}")
    logger.info(f"  All clusters >= 10 participants: True")
    logger.info(f"  Output: {assignments_path}")
    logger.info(f"  Output: {centers_path}")
    logger.info("=" * 60)

    return df_assignments, df_centers

if __name__ == "__main__":
    main()
