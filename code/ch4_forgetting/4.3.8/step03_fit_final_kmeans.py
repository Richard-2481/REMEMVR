#!/usr/bin/env python3
"""Fit Final K-Means Model: Fit final K-means model using optimal K from step02, extract cluster assignments"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_cluster_assignment

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.8
LOG_FILE = RQ_DIR / "logs" / "step03_fit_final_kmeans.log"

# Inputs
INPUT_FILE = RQ_DIR / "data" / "step01_standardized_features.csv"
OPTIMAL_K_FILE = RQ_DIR / "data" / "step02_optimal_k.txt"

# Outputs
ASSIGNMENTS_FILE = RQ_DIR / "data" / "step03_cluster_assignments.csv"
CENTERS_FILE = RQ_DIR / "data" / "step03_cluster_centers.csv"
SIZES_FILE = RQ_DIR / "data" / "step03_cluster_sizes.txt"

# K-means parameters
RANDOM_STATE = 42
N_INIT = 50

# Feature columns (exclude UID)
FEATURE_COLS = [
    'Total_Intercept_Cued_z',
    'Total_Intercept_Free_z',
    'Total_Intercept_Recognition_z',
    'Total_Slope_Cued_z',
    'Total_Slope_Free_z',
    'Total_Slope_Recognition_z'
]

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Fit Final K-Means Model")
        # Load Standardized Features

        log(f"Loading standardized features...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{len(df)} rows, {len(df.columns)} columns")

        # Verify feature columns exist
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Extract feature matrix (exclude UID)
        X = df.values
        UIDs = df['UID'].values

        log(f"Feature matrix: {X.shape[0]} participants, {X.shape[1]} features")
        # Read Optimal K from Step 02

        log(f"Reading optimal K from {OPTIMAL_K_FILE}...")

        with open(OPTIMAL_K_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse optimal K (last line should be "OPTIMAL K: X")
        optimal_k_line = [line for line in lines if line.startswith("OPTIMAL K:")]
        if not optimal_k_line:
            raise ValueError(f"Could not find 'OPTIMAL K:' line in {OPTIMAL_K_FILE}")

        K_optimal = int(optimal_k_line[0].split(':')[1].strip())
        log(f"Optimal K={K_optimal}")
        # Fit Final K-Means Model

        log(f"Fitting final K-means with K={K_optimal}...")
        log(f"Parameters: n_init={N_INIT}, random_state={RANDOM_STATE}")

        kmeans = KMeans(
            n_clusters=K_optimal,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=300
        )
        kmeans.fit(X)

        # Extract cluster assignments (0 to K-1)
        cluster_labels = kmeans.labels_

        # Extract cluster centers (K × 6 matrix)
        cluster_centers = kmeans.cluster_centers_

        log(f"K-means fitting complete")
        log(f"Cluster assignments: {len(cluster_labels)} participants")
        log(f"Cluster centers: {cluster_centers.shape[0]} clusters × {cluster_centers.shape[1]} features")
        # Save Cluster Assignments
        # Output: 100 rows × 2 columns (UID, cluster)
        # Downstream: step04 (quality validation), step05 (bootstrap stability)

        log(f"Saving cluster assignments...")

        df_assignments = pd.DataFrame({
            'UID': UIDs,
            'cluster': cluster_labels
        })

        df_assignments.to_csv(ASSIGNMENTS_FILE, index=False, encoding='utf-8')
        log(f"{ASSIGNMENTS_FILE} ({len(df_assignments)} rows)")
        # Save Cluster Centers
        # Output: K rows × 7 columns (cluster ID + 6 feature means)
        # Downstream: step06 (characterization), plotting pipeline (scatter matrix markers)

        log(f"Saving cluster centers...")

        df_centers = pd.DataFrame(
            cluster_centers,
            columns=FEATURE_COLS
        )
        df_centers.insert(0, 'cluster', range(K_optimal))

        df_centers.to_csv(CENTERS_FILE, index=False, encoding='utf-8')
        log(f"{CENTERS_FILE} ({len(df_centers)} rows)")
        # Compute and Save Cluster Sizes
        # Output: Text report (N per cluster)
        # Downstream: Verify minimum size >= 10 (10% threshold)

        log(f"Computing cluster sizes...")

        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

        log(f"Cluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            pct = (size / len(cluster_labels)) * 100
            log(f"Cluster {cluster_id}: N={size} ({pct:.1f}%)")

        # Write sizes report
        with open(SIZES_FILE, 'w', encoding='utf-8') as f:
            f.write("CLUSTER SIZES REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total participants: {len(cluster_labels)}\n")
            f.write(f"Number of clusters: {K_optimal}\n\n")

            f.write("CLUSTER SIZES:\n")
            f.write("-" * 80 + "\n")

            for cluster_id, size in cluster_sizes.items():
                pct = (size / len(cluster_labels)) * 100
                f.write(f"Cluster {cluster_id}:  N={size:3d}  ({pct:5.1f}%)\n")

            f.write("\n")
            f.write("MINIMUM SIZE THRESHOLD:\n")
            f.write("-" * 80 + "\n")
            min_size = cluster_sizes.min()
            min_threshold = 10
            f.write(f"Minimum cluster size: {min_size}\n")
            f.write(f"Threshold (10%): {min_threshold}\n")

            if min_size >= min_threshold:
                f.write(f"STATUS: PASS (all clusters >= {min_threshold})\n")
            else:
                f.write(f"STATUS: FAIL (cluster {cluster_sizes.idxmin()} has only {min_size} members)\n")

        log(f"{SIZES_FILE}")
        # Run Validation Tool
        # Validates: 100 participants assigned, cluster IDs 0 to K-1, sizes >= 10

        log("Running validate_cluster_assignment...")

        validation_result = validate_cluster_assignment(
            assignments_df=df_assignments,
            n_participants=100,
            min_cluster_size=10,
            cluster_col='cluster'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

        # Check validation passed
        if not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
