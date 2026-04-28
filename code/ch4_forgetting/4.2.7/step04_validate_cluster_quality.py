#!/usr/bin/env python3
"""
RQ 5.2.7 Step 04: Validate Cluster Quality

Purpose: Assess cluster quality using three metrics:
         1. Silhouette score (cohesion)
         2. Davies-Bouldin index (separation)
         3. Bootstrap Jaccard stability

Input:
  - data/step01_standardized_features.csv (100 rows × 5 cols)
  - data/step03_cluster_assignments.csv (100 rows: UID, cluster)

Output:
  - data/step04_cluster_validation.csv (5 rows: validation metrics)
  - data/step04_validation_summary.txt (quality assessment report)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input files
FEATURES_FILE = DATA_DIR / "step01_standardized_features.csv"
ASSIGNMENTS_FILE = DATA_DIR / "step03_cluster_assignments.csv"

# Z-scored clustering variables
Z_VARS = [
    'Total_Intercept_What_z',
    'Total_Intercept_Where_z',
    'Total_Slope_What_z',
    'Total_Slope_Where_z'
]

# Bootstrap parameters
N_BOOTSTRAP = 100
BOOTSTRAP_SAMPLE_FRAC = 0.80
BOOTSTRAP_SEED = 42

# Quality thresholds
SILHOUETTE_GOOD = 0.50
SILHOUETTE_ACCEPTABLE = 0.40
SILHOUETTE_POOR = 0.25

DB_GOOD = 1.0
DB_ACCEPTABLE = 1.5

JACCARD_STABLE = 0.75
JACCARD_MODERATE = 0.60


def compute_jaccard_index(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Compute Jaccard index between two label assignments.

    Jaccard index = (pairs agreeing) / (total pairs)

    Agreement = both in same cluster OR both in different clusters
    """
    n = len(labels1)
    if n != len(labels2):
        raise ValueError("Label arrays must have same length")

    agreements = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            same_in_1 = labels1[i] == labels1[j]
            same_in_2 = labels2[i] == labels2[j]

            if same_in_1 == same_in_2:
                agreements += 1
            total_pairs += 1

    return agreements / total_pairs if total_pairs > 0 else 0.0


def interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    if score >= SILHOUETTE_GOOD:
        return "Good"
    elif score >= SILHOUETTE_ACCEPTABLE:
        return "Acceptable"
    elif score >= SILHOUETTE_POOR:
        return "Poor"
    else:
        return "Very Poor"


def interpret_db(score: float) -> str:
    """Interpret Davies-Bouldin index."""
    if score < DB_GOOD:
        return "Good"
    elif score < DB_ACCEPTABLE:
        return "Acceptable"
    else:
        return "Poor"


def interpret_jaccard(score: float) -> str:
    """Interpret Jaccard stability."""
    if score > JACCARD_STABLE:
        return "Stable"
    elif score > JACCARD_MODERATE:
        return "Moderate"
    else:
        return "Unstable"


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 04: Validate Cluster Quality")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {FEATURES_FILE}")
    df_features = pd.read_csv(FEATURES_FILE)
    X = df_features.values
    n_samples = len(X)
    print(f"Loaded {n_samples} samples × {X.shape[1]} features")

    print(f"\nLoading: {ASSIGNMENTS_FILE}")
    df_assignments = pd.read_csv(ASSIGNMENTS_FILE)
    labels = df_assignments['cluster'].values
    n_clusters = len(np.unique(labels))
    print(f"Loaded {n_clusters} cluster assignments")

    # 1. Silhouette Score
    print("\n" + "-" * 40)
    print("1. Silhouette Score (Cohesion & Separation)")
    print("-" * 40)

    silhouette = silhouette_score(X, labels)
    silhouette_interp = interpret_silhouette(silhouette)
    print(f"   Silhouette score: {silhouette:.4f}")
    print(f"   Interpretation: {silhouette_interp}")
    print(f"   Thresholds: Good ≥ {SILHOUETTE_GOOD}, Acceptable ≥ {SILHOUETTE_ACCEPTABLE}, Poor ≥ {SILHOUETTE_POOR}")

    # 2. Davies-Bouldin Index
    print("\n" + "-" * 40)
    print("2. Davies-Bouldin Index (Cluster Separation)")
    print("-" * 40)

    db_index = davies_bouldin_score(X, labels)
    db_interp = interpret_db(db_index)
    print(f"   Davies-Bouldin index: {db_index:.4f}")
    print(f"   Interpretation: {db_interp}")
    print(f"   Thresholds: Good < {DB_GOOD}, Acceptable < {DB_ACCEPTABLE}")

    # 3. Bootstrap Stability (Jaccard)
    print("\n" + "-" * 40)
    print("3. Bootstrap Stability (Jaccard Coefficient)")
    print("-" * 40)
    print(f"   Running {N_BOOTSTRAP} bootstrap iterations ({int(BOOTSTRAP_SAMPLE_FRAC*100)}% subsample)...")

    np.random.seed(BOOTSTRAP_SEED)
    jaccard_scores = []

    for i in range(N_BOOTSTRAP):
        # Subsample indices
        sample_size = int(n_samples * BOOTSTRAP_SAMPLE_FRAC)
        sample_idx = np.random.choice(n_samples, size=sample_size, replace=True)

        # Get subsample data
        X_sample = X[sample_idx]
        original_labels_sample = labels[sample_idx]

        # Refit K-means on subsample
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        bootstrap_labels = kmeans.fit_predict(X_sample)

        # Compute Jaccard index
        jaccard = compute_jaccard_index(original_labels_sample, bootstrap_labels)
        jaccard_scores.append(jaccard)

        if (i + 1) % 25 == 0:
            print(f"      Completed {i + 1}/{N_BOOTSTRAP} iterations")

    jaccard_scores = np.array(jaccard_scores)
    jaccard_mean = jaccard_scores.mean()
    jaccard_ci_lower = np.percentile(jaccard_scores, 2.5)
    jaccard_ci_upper = np.percentile(jaccard_scores, 97.5)
    jaccard_interp = interpret_jaccard(jaccard_mean)

    print(f"\n   Bootstrap Jaccard mean: {jaccard_mean:.4f}")
    print(f"   95% CI: [{jaccard_ci_lower:.4f}, {jaccard_ci_upper:.4f}]")
    print(f"   Interpretation: {jaccard_interp}")
    print(f"   Thresholds: Stable > {JACCARD_STABLE}, Moderate > {JACCARD_MODERATE}")

    # Overall assessment
    print("\n" + "=" * 40)
    print("Overall Cluster Quality Assessment")
    print("=" * 40)

    # Decision logic
    if (silhouette >= SILHOUETTE_GOOD and db_index < DB_GOOD and jaccard_mean > JACCARD_STABLE):
        overall = "GOOD"
        recommendation = "Proceed with cluster interpretation and characterization"
    elif (silhouette >= SILHOUETTE_ACCEPTABLE and db_index < DB_ACCEPTABLE and jaccard_mean > JACCARD_MODERATE):
        overall = "ACCEPTABLE"
        recommendation = "Proceed with cluster interpretation - interpret cautiously"
    elif silhouette < SILHOUETTE_POOR:
        overall = "FAIL"
        recommendation = "Cluster structure not supported - run GMM sensitivity or interpret as continuous"
    else:
        overall = "POOR"
        recommendation = "Interpret clusters cautiously - consider alternative K or GMM sensitivity"

    print(f"\n   Silhouette: {silhouette:.3f} ({silhouette_interp})")
    print(f"   Davies-Bouldin: {db_index:.3f} ({db_interp})")
    print(f"   Bootstrap Jaccard: {jaccard_mean:.3f} ({jaccard_interp})")
    print(f"\n   Overall cluster quality: {overall}")
    print(f"   Recommendation: {recommendation}")

    # Save validation results
    results = [
        {'metric': 'silhouette_score', 'value': silhouette, 'interpretation': silhouette_interp},
        {'metric': 'davies_bouldin_index', 'value': db_index, 'interpretation': db_interp},
        {'metric': 'jaccard_mean', 'value': jaccard_mean, 'interpretation': jaccard_interp},
        {'metric': 'jaccard_ci_lower', 'value': jaccard_ci_lower, 'interpretation': '95% CI lower bound'},
        {'metric': 'jaccard_ci_upper', 'value': jaccard_ci_upper, 'interpretation': '95% CI upper bound'},
    ]
    df_results = pd.DataFrame(results)

    output_file = DATA_DIR / "step04_cluster_validation.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Save summary report
    summary_file = DATA_DIR / "step04_validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RQ 5.2.7 Step 04: Cluster Quality Validation Summary\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Overall cluster quality: {overall}\n\n")

        f.write("Validation Metrics:\n")
        f.write("-" * 55 + "\n")
        f.write(f"  Silhouette score:     {silhouette:.4f} ({silhouette_interp})\n")
        f.write(f"    Threshold: Good ≥ {SILHOUETTE_GOOD}, Acceptable ≥ {SILHOUETTE_ACCEPTABLE}\n\n")
        f.write(f"  Davies-Bouldin index: {db_index:.4f} ({db_interp})\n")
        f.write(f"    Threshold: Good < {DB_GOOD}, Acceptable < {DB_ACCEPTABLE}\n\n")
        f.write(f"  Bootstrap Jaccard:    {jaccard_mean:.4f} ({jaccard_interp})\n")
        f.write(f"    95% CI: [{jaccard_ci_lower:.4f}, {jaccard_ci_upper:.4f}]\n")
        f.write(f"    Threshold: Stable > {JACCARD_STABLE}, Moderate > {JACCARD_MODERATE}\n\n")

        f.write("-" * 55 + "\n")
        f.write(f"Recommendation: {recommendation}\n")

        if overall in ["POOR", "FAIL"]:
            f.write("\nNote: Consider running GMM sensitivity analysis or\n")
            f.write("interpreting individual differences as continuous rather\n")
            f.write("than discrete cluster profiles.\n")

    print(f"Saved: {summary_file}")

    print("\n" + "=" * 60)
    print(f"Step 04 COMPLETE: Cluster quality = {overall}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
