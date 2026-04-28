#!/usr/bin/env python3
"""
Step 04: Validate Clustering Quality

Compute silhouette score, Davies-Bouldin index, and bootstrap stability (Jaccard coefficient).

Input:
  - data/step01_standardized_features.csv (z-scored features)
  - data/step03_cluster_assignments.csv (cluster assignments)

Output:
  - data/step04_cluster_quality_metrics.csv (quality metrics)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_bootstrap_stability

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
INPUT_FEATURES_FILE = RQ_DIR / "data/step01_standardized_features.csv"
INPUT_ASSIGNMENTS_FILE = RQ_DIR / "data/step03_cluster_assignments.csv"
OUTPUT_FILE = RQ_DIR / "data/step04_cluster_quality_metrics.csv"
LOG_FILE = RQ_DIR / "logs/step04_validate_clustering.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def compute_bootstrap_jaccard(X, labels, K, n_iterations=100, sample_frac=0.8):
    """Compute bootstrap Jaccard stability coefficient."""
    np.random.seed(42)
    n_samples = len(X)
    sample_size = int(n_samples * sample_frac)
    jaccard_values = []

    for i in range(n_iterations):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=sample_size, replace=False)
        X_boot = X[idx]

        # Refit K-means on bootstrap sample
        kmeans_boot = KMeans(n_clusters=K, random_state=42+i, n_init=10)
        labels_boot = kmeans_boot.fit_predict(X_boot)

        # Compute Jaccard between original and bootstrap assignments
        # (for samples in bootstrap)
        labels_orig_boot = labels[idx]

        # Simple Jaccard: fraction of sample pairs with same co-clustering status
        n_pairs_same_orig = 0
        n_pairs_same_boot = 0
        n_pairs_same_both = 0
        n_pairs = 0

        for j in range(len(idx)):
            for k in range(j+1, len(idx)):
                same_orig = labels_orig_boot[j] == labels_orig_boot[k]
                same_boot = labels_boot[j] == labels_boot[k]

                if same_orig:
                    n_pairs_same_orig += 1
                if same_boot:
                    n_pairs_same_boot += 1
                if same_orig and same_boot:
                    n_pairs_same_both += 1
                n_pairs += 1

        jaccard = n_pairs_same_both / (n_pairs_same_orig + n_pairs_same_boot - n_pairs_same_both + 1e-10)
        jaccard_values.append(jaccard)

    return jaccard_values

if __name__ == "__main__":
    try:
        log("Step 04: Validate Clustering Quality")

        # Load standardized features
        log(f"Reading {INPUT_FEATURES_FILE}")
        df_features = pd.read_csv(INPUT_FEATURES_FILE)

        feature_cols = [
            'Common_Intercept_z', 'Common_Slope_z',
            'Congruent_Intercept_z', 'Congruent_Slope_z',
            'Incongruent_Intercept_z', 'Incongruent_Slope_z'
        ]
        X = df_features[feature_cols].values

        # Load cluster assignments
        log(f"Reading {INPUT_ASSIGNMENTS_FILE}")
        df_assignments = pd.read_csv(INPUT_ASSIGNMENTS_FILE)
        labels = df_assignments['cluster'].values
        K = len(np.unique(labels))

        log(f"{len(X)} samples, {K} clusters")

        # Compute silhouette score
        log("Computing silhouette score...")
        silhouette = silhouette_score(X, labels)
        log(f"  Silhouette score = {silhouette:.4f}")

        # Compute Davies-Bouldin index
        log("Computing Davies-Bouldin index...")
        davies_bouldin = davies_bouldin_score(X, labels)
        log(f"  Davies-Bouldin index = {davies_bouldin:.4f}")

        # Compute bootstrap Jaccard coefficient
        log("Computing bootstrap Jaccard (100 iterations, 80% sample)...")
        jaccard_values = compute_bootstrap_jaccard(X, labels, K, n_iterations=100, sample_frac=0.8)
        jaccard_mean = np.mean(jaccard_values)
        jaccard_median = np.median(jaccard_values)
        jaccard_min = np.min(jaccard_values)
        log(f"  Bootstrap Jaccard mean = {jaccard_mean:.4f}")
        log(f"  Bootstrap Jaccard median = {jaccard_median:.4f}")
        log(f"  Bootstrap Jaccard min = {jaccard_min:.4f}")

        # Create metrics DataFrame
        metrics = [
            {'metric': 'silhouette', 'value': silhouette, 'threshold': 0.40, 'pass': silhouette >= 0.40},
            {'metric': 'davies_bouldin', 'value': davies_bouldin, 'threshold': 1.5, 'pass': davies_bouldin < 1.5},
            {'metric': 'jaccard_mean', 'value': jaccard_mean, 'threshold': 0.75, 'pass': jaccard_mean > 0.75},
            {'metric': 'jaccard_median', 'value': jaccard_median, 'threshold': 0.75, 'pass': jaccard_median > 0.75},
            {'metric': 'jaccard_min', 'value': jaccard_min, 'threshold': 0.75, 'pass': jaccard_min > 0.75}
        ]

        df_quality = pd.DataFrame(metrics)

        # Save metrics
        log(f"Writing to {OUTPUT_FILE}")
        df_quality.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE}")

        # Report findings
        log("[QUALITY SUMMARY]")
        for _, row in df_quality.iterrows():
            status = "PASS" if row['pass'] else "FAIL"
            log(f"  {row['metric']}: {row['value']:.4f} (threshold={row['threshold']:.2f}) [{status}]")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
