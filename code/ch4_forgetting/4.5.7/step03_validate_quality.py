#!/usr/bin/env python3
"""
RQ 5.5.7 Step 03: Validate Clustering Quality

Purpose: Validate clustering quality using 3 metrics:
- Silhouette score (>=0.40 acceptable, <0.40 weak)
- Davies-Bouldin index (<1.50 acceptable, >=1.50 poor separation)
- Jaccard bootstrap stability (>=0.75 acceptable, B=100 iterations)

Input:
- data/step01_standardized_features.csv (100 rows, z-scored)
- data/step02_optimal_k.txt (optimal K value)

Output:
- data/step03_cluster_validation.csv (3 rows: metric, value, threshold, status)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
        logging.FileHandler(LOG_DIR / "step03_validate_quality.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_jaccard_stability(X, K, n_bootstrap=100, random_state=42):
    """
    Compute Jaccard bootstrap stability for K-means clustering.

    For each bootstrap sample, fit K-means and compute Jaccard similarity
    between original and bootstrap cluster assignments.

    Returns: mean Jaccard, 95% CI (2.5th, 97.5th percentiles), all Jaccard values
    """
    np.random.seed(random_state)
    N = X.shape[0]

    # Fit original K-means
    kmeans_orig = KMeans(n_clusters=K, random_state=random_state, n_init=50)
    labels_orig = kmeans_orig.fit_predict(X)

    jaccard_values = []

    for b in range(n_bootstrap):
        # Resample with replacement
        boot_idx = np.random.choice(N, size=N, replace=True)
        X_boot = X[boot_idx]

        # Fit K-means on bootstrap sample
        kmeans_boot = KMeans(n_clusters=K, random_state=random_state+b+1, n_init=50)
        labels_boot = kmeans_boot.fit_predict(X_boot)

        # Compute Jaccard similarity for each cluster
        # Match clusters by maximum overlap (Hungarian algorithm approximation)
        jaccard_per_cluster = []

        for k in range(K):
            # Original cluster k members (from boot_idx)
            orig_members = set(np.where(labels_orig[boot_idx] == k)[0])

            # Find best matching cluster in bootstrap
            best_jaccard = 0
            for j in range(K):
                boot_members = set(np.where(labels_boot == j)[0])

                if len(orig_members) == 0 and len(boot_members) == 0:
                    jaccard = 1.0
                elif len(orig_members | boot_members) == 0:
                    jaccard = 0.0
                else:
                    intersection = len(orig_members & boot_members)
                    union = len(orig_members | boot_members)
                    jaccard = intersection / union

                best_jaccard = max(best_jaccard, jaccard)

            jaccard_per_cluster.append(best_jaccard)

        # Average Jaccard across clusters for this bootstrap
        jaccard_values.append(np.mean(jaccard_per_cluster))

    jaccard_values = np.array(jaccard_values)
    mean_jaccard = np.mean(jaccard_values)
    ci_low = np.percentile(jaccard_values, 2.5)
    ci_high = np.percentile(jaccard_values, 97.5)

    return mean_jaccard, (ci_low, ci_high), jaccard_values


def main():
    """Validate clustering quality with 3 metrics."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 03: Validate Clustering Quality")
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

    # Extract feature matrix
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']
    X = df[feature_cols].values

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
    logger.info(f"K-means fitted, {K} clusters assigned")

    # -------------------------------------------------------------------------
    # 3. Compute Silhouette score
    # -------------------------------------------------------------------------
    logger.info("\nComputing Silhouette score...")
    silhouette = silhouette_score(X, labels)
    silhouette_threshold = 0.40
    silhouette_status = "PASS" if silhouette >= silhouette_threshold else "FAIL"
    logger.info(f"Silhouette score: {silhouette:.3f} (threshold: {silhouette_threshold})")
    logger.info(f"Silhouette status: {silhouette_status}")

    # -------------------------------------------------------------------------
    # 4. Compute Davies-Bouldin index
    # -------------------------------------------------------------------------
    logger.info("\nComputing Davies-Bouldin index...")
    davies_bouldin = davies_bouldin_score(X, labels)
    db_threshold = 1.50
    db_status = "PASS" if davies_bouldin < db_threshold else "FAIL"
    logger.info(f"Davies-Bouldin index: {davies_bouldin:.3f} (threshold: {db_threshold})")
    logger.info(f"Davies-Bouldin status: {db_status}")

    # -------------------------------------------------------------------------
    # 5. Compute Jaccard bootstrap stability
    # -------------------------------------------------------------------------
    logger.info("\nComputing Jaccard bootstrap stability (B=100)...")
    jaccard_mean, jaccard_ci, jaccard_values = compute_jaccard_stability(
        X, K, n_bootstrap=100, random_state=42
    )
    jaccard_threshold = 0.75
    jaccard_status = "PASS" if jaccard_mean >= jaccard_threshold else "FAIL"
    logger.info(f"Jaccard stability: {jaccard_mean:.3f} "
               f"[95% CI: {jaccard_ci[0]:.3f}, {jaccard_ci[1]:.3f}] "
               f"(threshold: {jaccard_threshold})")
    logger.info(f"Jaccard status: {jaccard_status}")

    # -------------------------------------------------------------------------
    # 6. Assess overall quality
    # -------------------------------------------------------------------------
    logger.info("\n" + "-" * 40)
    logger.info("Overall Quality Assessment")
    logger.info("-" * 40)

    # PASS if Silhouette >= 0.40 OR Jaccard >= 0.75
    # FAIL if BOTH fail
    if silhouette >= silhouette_threshold or jaccard_mean >= jaccard_threshold:
        overall_status = "PASS"
        overall_reason = "At least one criterion met"
    else:
        overall_status = "FAIL"
        overall_reason = "Both Silhouette < 0.40 AND Jaccard < 0.75"

    logger.info(f"Overall quality: {overall_status}")
    logger.info(f"Reason: {overall_reason}")

    if overall_status == "FAIL":
        logger.warning("Clustering quality weak - memory ability appears continuous, "
                      "not categorical. This is a meaningful null finding.")

    # -------------------------------------------------------------------------
    # 7. Create results DataFrame
    # -------------------------------------------------------------------------
    results = [
        {
            'metric': 'Silhouette',
            'value': silhouette,
            'threshold': silhouette_threshold,
            'status': silhouette_status
        },
        {
            'metric': 'Davies-Bouldin',
            'value': davies_bouldin,
            'threshold': db_threshold,
            'status': db_status
        },
        {
            'metric': 'Jaccard',
            'value': jaccard_mean,
            'threshold': jaccard_threshold,
            'status': jaccard_status
        }
    ]

    df_results = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 8. Validate results
    # -------------------------------------------------------------------------
    logger.info("\nValidating metric values...")

    # Silhouette in [0, 1] for K > 1 (actually can be [-1, 1] but typically positive)
    if -1 <= silhouette <= 1:
        logger.info("Silhouette in valid range [-1, 1]: PASS")
    else:
        logger.error(f"Silhouette out of range: {silhouette}")
        sys.exit(1)

    # Davies-Bouldin >= 0
    if davies_bouldin >= 0:
        logger.info("Davies-Bouldin >= 0: PASS")
    else:
        logger.error(f"Davies-Bouldin negative: {davies_bouldin}")
        sys.exit(1)

    # Jaccard in [0, 1]
    if 0 <= jaccard_mean <= 1:
        logger.info("Jaccard in valid range [0, 1]: PASS")
    else:
        logger.error(f"Jaccard out of range: {jaccard_mean}")
        sys.exit(1)

    # All bootstrap Jaccard values in [0, 1]
    if (jaccard_values >= 0).all() and (jaccard_values <= 1).all():
        logger.info("All bootstrap Jaccard values in [0, 1]: PASS")
    else:
        logger.error("Bootstrap Jaccard values out of range")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 9. Save output
    # -------------------------------------------------------------------------
    output_path = DATA_DIR / "step03_cluster_validation.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # -------------------------------------------------------------------------
    # 10. Final summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 03 COMPLETE")
    logger.info(f"  Optimal K: {K}")
    logger.info(f"  Silhouette: {silhouette:.3f} ({silhouette_status})")
    logger.info(f"  Davies-Bouldin: {davies_bouldin:.3f} ({db_status})")
    logger.info(f"  Jaccard: {jaccard_mean:.3f} ({jaccard_status})")
    logger.info(f"  Overall: {overall_status} - {overall_reason}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    return df_results

if __name__ == "__main__":
    main()
