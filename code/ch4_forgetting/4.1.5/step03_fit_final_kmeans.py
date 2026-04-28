#!/usr/bin/env python3
"""Fit Final K-means Model with Optimal K: Fit final K-means clustering model with optimal K selected in Step 2 via BIC."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_cluster_assignment

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_final_kmeans.log"


# K-means Parameters

RANDOM_STATE = 42
N_INIT = 50
MIN_CLUSTER_SIZE = 10  # 10% of N=100

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Fit Final K-means Model with Optimal K")
        # Load Input Data

        log("Loading standardized features...")
        standardized_features = pd.read_csv(RQ_DIR / "data/step01_standardized_features.csv")
        log(f"standardized_features ({len(standardized_features)} rows, {len(standardized_features.columns)} cols)")

        # Extract feature matrix (100 x 2)
        X = standardized_features[['Intercept_z', 'Slope_z']].values
        log(f"Feature matrix shape: {X.shape}")
        # Read Optimal K from Step 2

        log("Reading optimal K from Step 2...")
        with open(RQ_DIR / "data/step02_optimal_k.txt", 'r') as f:
            K_initial = int(f.read().strip())
        log(f"Optimal K from Step 2 (BIC minimum): K_initial = {K_initial}")
        # Fit K-means with Remedial Size Enforcement
        #   if any cluster < 10% threshold, refit until all clusters meet threshold

        K_current = K_initial
        remedial_action_taken = False
        remedial_log = []

        log(f"Starting K-means fitting with K={K_current}...")
        log(f"Min cluster size threshold: {MIN_CLUSTER_SIZE} (10% of N=100)")

        while True:
            # Fit K-means with current K
            log(f"Fitting K-means with K={K_current} (random_state={RANDOM_STATE}, n_init={N_INIT})...")
            kmeans = KMeans(n_clusters=K_current, random_state=RANDOM_STATE, n_init=N_INIT)
            cluster_labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_

            # Check cluster sizes
            unique_labels, cluster_counts = np.unique(cluster_labels, return_counts=True)
            log(f"Cluster sizes (K={K_current}): {dict(zip(unique_labels, cluster_counts))}")

            # Check if any cluster is undersized
            min_cluster_size = cluster_counts.min()
            if min_cluster_size < MIN_CLUSTER_SIZE:
                log(f"Undersized cluster detected: min size = {min_cluster_size} < {MIN_CLUSTER_SIZE}")
                if K_current == 1:
                    log(f"Cannot reduce K below 1. Accepting K=1 despite size issue.")
                    remedial_action_taken = True
                    remedial_log.append(f"K={K_current}: Min cluster size {min_cluster_size} < {MIN_CLUSTER_SIZE}, but K=1 is minimum possible.")
                    break
                else:
                    # Reduce K by 1 and refit
                    remedial_action_taken = True
                    remedial_log.append(f"K={K_current}: Min cluster size {min_cluster_size} < {MIN_CLUSTER_SIZE}. Reducing K to {K_current-1}.")
                    log(f"Reducing K from {K_current} to {K_current-1} and refitting...")
                    K_current -= 1
            else:
                log(f"All clusters meet size threshold (min size = {min_cluster_size} >= {MIN_CLUSTER_SIZE})")
                break

        K_final = K_current
        log(f"Final K-means fit complete with K_final = {K_final}")
        # Extract Cluster Assignments and Centers
        # Output 1: Cluster assignments (100 rows: UID, cluster)
        # Output 2: Cluster centers (K rows: cluster, Intercept_z_center, Slope_z_center)

        log("Extracting cluster assignments...")
        cluster_assignments = pd.DataFrame({
            'UID': standardized_features['UID'],
            'cluster': cluster_labels
        })
        log(f"Cluster assignments shape: {cluster_assignments.shape}")

        log("Extracting cluster centers...")
        cluster_centers_df = pd.DataFrame({
            'cluster': np.arange(K_final),
            'Intercept_z_center': cluster_centers[:, 0],
            'Slope_z_center': cluster_centers[:, 1]
        })
        log(f"Cluster centers shape: {cluster_centers_df.shape}")
        # Save Outputs
        # These outputs will be used by: Step 4 (bootstrap stability), Step 6 (characterization), Step 7 (plots)

        log(f"Saving cluster assignments to data/step03_cluster_assignments.csv...")
        cluster_assignments.to_csv(RQ_DIR / "data/step03_cluster_assignments.csv", index=False, encoding='utf-8')
        log(f"step03_cluster_assignments.csv ({len(cluster_assignments)} rows, {len(cluster_assignments.columns)} cols)")

        log(f"Saving cluster centers to data/step03_cluster_centers.csv...")
        cluster_centers_df.to_csv(RQ_DIR / "data/step03_cluster_centers.csv", index=False, encoding='utf-8')
        log(f"step03_cluster_centers.csv ({len(cluster_centers_df)} rows, {len(cluster_centers_df.columns)} cols)")

        # Save remedial action report if K was reduced
        if remedial_action_taken:
            log(f"Saving remedial action report to data/step03_remedial_action.txt...")
            remedial_report = f"Remedial Action Report - K-means Cluster Size Enforcement\n"
            remedial_report += f"=" * 60 + "\n\n"
            remedial_report += f"K_initial (from Step 2 BIC): {K_initial}\n"
            remedial_report += f"K_final (after size enforcement): {K_final}\n\n"
            remedial_report += f"Remedial actions taken:\n"
            for entry in remedial_log:
                remedial_report += f"  - {entry}\n"
            remedial_report += f"\nFinal cluster sizes:\n"
            for label, count in zip(unique_labels, cluster_counts):
                remedial_report += f"  - Cluster {label}: {count} participants\n"

            with open(RQ_DIR / "data/step03_remedial_action.txt", 'w', encoding='utf-8') as f:
                f.write(remedial_report)
            log(f"step03_remedial_action.txt (K reduced from {K_initial} to {K_final})")
        # Run Validation Tool
        # Validates: All participants assigned, cluster IDs consecutive, sizes >= threshold
        # Threshold: min_cluster_size=10 (10% of N=100)

        log("Running validate_cluster_assignment...")
        validation_result = validate_cluster_assignment(
            assignments_df=cluster_assignments,
            n_participants=100,
            min_cluster_size=MIN_CLUSTER_SIZE,
            cluster_col='cluster'
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")
            log(f"Cluster sizes: {validation_result['cluster_sizes']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            raise ValueError(f"Cluster assignment validation failed: {validation_result['message']}")

        log(f"Step 03 complete - K_final={K_final}, all clusters meet size threshold")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
