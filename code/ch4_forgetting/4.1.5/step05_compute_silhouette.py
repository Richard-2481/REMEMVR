#!/usr/bin/env python3
"""compute_silhouette: Compute silhouette coefficient to assess cluster quality using Rousseeuw 1987"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import silhouette_score

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_compute_silhouette.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Compute Silhouette Coefficient")
        # Load Input Data
        #           and cluster assignments from Step 3 (after K-means fitting)

        log("Loading standardized features from Step 1...")
        # Standardized features: UID, Intercept_z, Slope_z (100 participants, mean=0, SD=1)
        features_df = pd.read_csv(RQ_DIR / "data/step01_standardized_features.csv", encoding='utf-8')
        log(f"step01_standardized_features.csv ({len(features_df)} rows, {len(features_df.columns)} cols)")
        log(f"Columns: {features_df.columns.tolist()}")

        log("Loading cluster assignments from Step 3...")
        # Cluster assignments: UID, cluster (100 participants, cluster IDs 0 to K-1)
        assignments_df = pd.read_csv(RQ_DIR / "data/step03_cluster_assignments.csv", encoding='utf-8')
        log(f"step03_cluster_assignments.csv ({len(assignments_df)} rows, {len(assignments_df.columns)} cols)")
        log(f"Columns: {assignments_df.columns.tolist()}")
        # Prepare Data for Silhouette Computation
        # Need: (1) Feature matrix X (100 x 2: Intercept_z, Slope_z)
        #       (2) Cluster labels (100 x 1: cluster IDs)
        # Both must be aligned by participant UID

        log("Merging features with cluster assignments on UID...")
        # Inner merge ensures alignment (all 100 participants present in both files)
        merged_df = features_df.merge(assignments_df, on='UID', how='inner')
        log(f"{len(merged_df)} participants with features + cluster assignments")

        if len(merged_df) != len(features_df):
            raise ValueError(f"UID mismatch: features has {len(features_df)} rows, merged has {len(merged_df)} rows")

        # Extract feature matrix X (N x 2 numpy array)
        X = merged_df[['Intercept_z', 'Slope_z']].values
        log(f"Feature matrix X: shape {X.shape} (100 participants x 2 features)")

        # Extract cluster labels (N x 1 numpy array)
        labels = merged_df['cluster'].values
        n_clusters = len(np.unique(labels))
        log(f"Cluster labels: {len(labels)} participants, {n_clusters} unique clusters")
        log(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        # Compute Silhouette Coefficient
        #   - For each participant i:
        #     a(i) = mean distance to other participants in same cluster (cohesion)
        #     b(i) = mean distance to participants in nearest other cluster (separation)
        #     s(i) = (b(i) - a(i)) / max(a(i), b(i))  (silhouette coefficient)
        #   - Overall silhouette = mean of s(i) across all participants
        #   - 1 = perfect clustering (each point close to own cluster, far from others)
        #   - 0 = overlapping clusters (point on decision boundary)
        #   - -1 = wrong cluster assignment (point closer to other cluster)

        log("Computing silhouette coefficient using Euclidean metric...")
        silhouette_coef = silhouette_score(X=X, labels=labels, metric='euclidean')
        log(f"Silhouette coefficient: {silhouette_coef:.4f}")
        # Interpret Silhouette Score
        # Interpretation thresholds from Rousseeuw (1987) and Kaufman & Rousseeuw (1990)
        # Used in cluster validation literature for K-means and hierarchical clustering

        if silhouette_coef >= 0.70:
            interpretation = "Strong cluster structure found"
            detail = "Clusters are well-separated and cohesive (silhouette >= 0.70)"
        elif silhouette_coef >= 0.50:
            interpretation = "Reasonable cluster structure found"
            detail = "Clusters have moderate separation and cohesion (0.50 <= silhouette < 0.70)"
        elif silhouette_coef >= 0.25:
            interpretation = "Weak cluster structure"
            detail = "Clusters have poor separation, may be artificial (0.25 <= silhouette < 0.50)"
        else:
            interpretation = "No substantial cluster structure"
            detail = "Clustering may be arbitrary, consider K=1 (silhouette < 0.25)"

        log(f"{interpretation}")
        log(f"{detail}")
        # Save Silhouette Score and Interpretation
        # Output: Single text file with silhouette score + interpretation
        # Format: Silhouette: 0.XXXX\nInterpretation: [text]
        # Used by: Step 7 for plot metadata annotation

        output_path = RQ_DIR / "data/step05_silhouette_score.txt"
        log(f"Saving silhouette score and interpretation to {output_path.name}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Silhouette Coefficient: {silhouette_coef:.4f}\n")
            f.write(f"Interpretation: {interpretation}\n")
            f.write(f"Detail: {detail}\n")
            f.write(f"Number of Clusters: {n_clusters}\n")
            f.write(f"Number of Participants: {len(labels)}\n")

        log(f"{output_path.name}")
        # Run Validation Tool
        # Validates: Silhouette coefficient in [-1, 1], finite (not NaN/inf)
        # Why needed: Silhouette computation can fail with degenerate clusters
        #   (e.g., single-participant clusters, identical feature values)

        log("Running validate_numeric_range on silhouette coefficient...")

        # Convert single float to numpy array for validation function
        # (validate_numeric_range expects ndarray or Series)
        silhouette_array = np.array([silhouette_coef])

        validation_result = validate_numeric_range(
            data=silhouette_array,
            min_val=-1.0,  # Theoretical minimum (point in wrong cluster)
            max_val=1.0,   # Theoretical maximum (perfect clustering)
            column_name='silhouette'  # For error messages
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS: Silhouette coefficient in valid range [-1, 1]")
            log(f"Message: {validation_result['message']}")
        else:
            # Validation failed - silhouette out of bounds or NaN/inf
            log(f"FAIL: {validation_result['message']}")
            raise ValueError(f"Silhouette validation failed: {validation_result['message']}")

        log("Step 05 complete - Silhouette coefficient computed and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
