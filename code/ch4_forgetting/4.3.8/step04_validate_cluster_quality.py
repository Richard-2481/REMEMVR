#!/usr/bin/env python3
"""
================================================================================
STEP 04: Validate Cluster Quality
================================================================================
RQ: ch5/5.3.8 (Paradigm-Based Clustering)
Purpose: Compute cluster quality metrics (silhouette, Davies-Bouldin, Dunn)
         with validation thresholds

Inputs:
  - data/step01_standardized_features.csv (100 x 6 z-scores)
  - data/step03_cluster_assignments.csv (100 cluster labels)

Outputs:
  - data/step04_cluster_quality_metrics.csv (metric, value, threshold, pass)
  - data/step04_quality_interpretation.txt

Quality Thresholds:
  - Silhouette score >= 0.40 (acceptable separation)
  - Davies-Bouldin index < 1.5 (low within/between cluster ratio)
  - Dunn index > 0 (positive separation)

Note: Threshold failures trigger WARNING (not error) - clustering is
      exploratory and tentative per concept.md
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_validate_cluster_quality.log"

# Quality thresholds
SILHOUETTE_THRESHOLD = 0.40
DAVIES_BOULDIN_THRESHOLD = 1.5

# Logging
def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Dunn Index Computation
def compute_dunn_index(X, labels):
    """
    Compute Dunn index: min inter-cluster distance / max intra-cluster diameter

    Higher Dunn index = better clustering (well-separated, compact clusters)

    Args:
        X: Feature matrix (N x p)
        labels: Cluster labels (N,)

    Returns:
        float: Dunn index
    """
    unique_labels = np.unique(labels)

    # Compute inter-cluster distances (minimum distance between any two clusters)
    inter_cluster_distances = []
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            # Get points in each cluster
            cluster_i = X[labels == label_i]
            cluster_j = X[labels == label_j]
            # Compute pairwise distances between clusters
            distances = cdist(cluster_i, cluster_j, metric='euclidean')
            # Store minimum distance
            inter_cluster_distances.append(distances.min())

    min_inter_cluster_distance = np.min(inter_cluster_distances)

    # Compute intra-cluster diameters (maximum distance within each cluster)
    intra_cluster_diameters = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            # Pairwise distances within cluster
            distances = cdist(cluster_points, cluster_points, metric='euclidean')
            # Maximum distance = diameter
            intra_cluster_diameters.append(distances.max())
        else:
            intra_cluster_diameters.append(0)

    max_intra_cluster_diameter = np.max(intra_cluster_diameters)

    # Dunn index = min inter / max intra
    if max_intra_cluster_diameter == 0:
        return np.inf  # Perfect clustering (all clusters are points)

    dunn_index = min_inter_cluster_distance / max_intra_cluster_diameter
    return dunn_index

# Main Analysis
if __name__ == "__main__":
    try:
        log("="*80)
        log("STEP 04: Validate Cluster Quality")
        log("="*80)
        # Load Data
        log("\nLoading standardized features and cluster assignments...")

        # Load standardized features (z-scores)
        df_features = pd.read_csv(RQ_DIR / "data" / "step01_standardized_features.csv")
        log(f"Standardized features: {df_features.shape}")

        # Load cluster assignments
        df_clusters = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        log(f"Cluster assignments: {df_clusters.shape}")

        # Extract feature matrix (6 z-score columns)
        feature_cols = [col for col in df_features.columns if col.endswith('_z')]
        X = df_features[feature_cols].values
        log(f"Feature matrix: {X.shape} (100 participants x {len(feature_cols)} features)")

        # Extract cluster labels
        labels = df_clusters['cluster'].values
        n_clusters = len(np.unique(labels))
        log(f"Cluster labels: {labels.shape} ({n_clusters} clusters)")
        # Compute Quality Metrics
        log("\nComputing cluster quality metrics...")

        # Silhouette score (mean across participants)
        silhouette = silhouette_score(X, labels)
        log(f"Silhouette score: {silhouette:.4f}")
        log(f"         Threshold: >= {SILHOUETTE_THRESHOLD}")
        silhouette_pass = silhouette >= SILHOUETTE_THRESHOLD
        log(f"         Status: {'PASS' if silhouette_pass else 'WARNING (below threshold)'}")

        # Davies-Bouldin index (within/between cluster ratio)
        davies_bouldin = davies_bouldin_score(X, labels)
        log(f"Davies-Bouldin index: {davies_bouldin:.4f}")
        log(f"         Threshold: < {DAVIES_BOULDIN_THRESHOLD}")
        davies_bouldin_pass = davies_bouldin < DAVIES_BOULDIN_THRESHOLD
        log(f"         Status: {'PASS' if davies_bouldin_pass else 'WARNING (above threshold)'}")

        # Dunn index (min inter-cluster distance / max intra-cluster diameter)
        dunn = compute_dunn_index(X, labels)
        log(f"Dunn index: {dunn:.4f}")
        log(f"         Threshold: > 0 (higher is better)")
        dunn_pass = dunn > 0
        log(f"         Status: {'PASS' if dunn_pass else 'FAIL (negative or zero)'}")
        # Save Metrics Table
        log("\nSaving quality metrics table...")

        metrics_df = pd.DataFrame({
            'metric': ['silhouette', 'davies_bouldin', 'dunn'],
            'value': [silhouette, davies_bouldin, dunn],
            'threshold': [f'>= {SILHOUETTE_THRESHOLD}', f'< {DAVIES_BOULDIN_THRESHOLD}', '> 0'],
            'pass': [silhouette_pass, davies_bouldin_pass, dunn_pass]
        })

        metrics_path = RQ_DIR / "data" / "step04_cluster_quality_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
        log(f"{metrics_path}")
        log(f"        {len(metrics_df)} metrics recorded")
        # Generate Interpretation Report
        log("\nGenerating quality interpretation...")

        interpretation = []
        interpretation.append("CLUSTER QUALITY INTERPRETATION")
        interpretation.append("="*80)
        interpretation.append(f"\nClustering Configuration: K={n_clusters} clusters, N=100 participants")
        interpretation.append(f"Feature Space: 6 dimensions (paradigm-specific intercepts and slopes)")
        interpretation.append("")

        interpretation.append("METRIC SUMMARY:")
        interpretation.append("-"*80)
        interpretation.append(f"Silhouette Score:     {silhouette:.4f}  (threshold >= {SILHOUETTE_THRESHOLD})")
        interpretation.append(f"Davies-Bouldin Index: {davies_bouldin:.4f}  (threshold < {DAVIES_BOULDIN_THRESHOLD})")
        interpretation.append(f"Dunn Index:           {dunn:.4f}  (threshold > 0)")
        interpretation.append("")

        # Overall assessment
        interpretation.append("QUALITY ASSESSMENT:")
        interpretation.append("-"*80)

        # Silhouette interpretation
        if silhouette >= 0.50:
            silhouette_interp = "ACCEPTABLE - Reasonable cluster structure"
        elif silhouette >= 0.40:
            silhouette_interp = "MARGINAL - Weak but detectable cluster structure"
        else:
            silhouette_interp = "POOR - Clusters may overlap substantially"
        interpretation.append(f"Silhouette:     {silhouette_interp}")

        # Davies-Bouldin interpretation
        if davies_bouldin < 1.0:
            db_interp = "GOOD - Clusters well-separated"
        elif davies_bouldin < 1.5:
            db_interp = "ACCEPTABLE - Moderate cluster separation"
        else:
            db_interp = "POOR - High within/between cluster ratio"
        interpretation.append(f"Davies-Bouldin: {db_interp}")

        # Dunn interpretation
        if dunn > 1.0:
            dunn_interp = "EXCELLENT - Well-separated compact clusters"
        elif dunn > 0.5:
            dunn_interp = "GOOD - Reasonable separation and compactness"
        else:
            dunn_interp = "MARGINAL - Limited separation or large intra-cluster spread"
        interpretation.append(f"Dunn:           {dunn_interp}")

        interpretation.append("")

        # Warning if thresholds not met
        if not (silhouette_pass and davies_bouldin_pass):
            interpretation.append("WARNING:")
            interpretation.append("-"*80)
            if not silhouette_pass:
                interpretation.append(f"- Silhouette score {silhouette:.4f} below threshold {SILHOUETTE_THRESHOLD}")
                interpretation.append("  This suggests clusters may not be strongly separated.")
            if not davies_bouldin_pass:
                interpretation.append(f"- Davies-Bouldin index {davies_bouldin:.4f} above threshold {DAVIES_BOULDIN_THRESHOLD}")
                interpretation.append("  This suggests high within-cluster variance relative to between-cluster separation.")
            interpretation.append("")
            interpretation.append("Note: These warnings are INFORMATIONAL for exploratory clustering.")
            interpretation.append("      Results should be interpreted as tentative phenotypes pending validation.")
        else:
            interpretation.append("CONCLUSION:")
            interpretation.append("-"*80)
            interpretation.append("All quality metrics meet thresholds. Clustering shows acceptable structure")
            interpretation.append("for exploratory phenotype identification. Bootstrap stability assessment (Step 05)")
            interpretation.append("will further validate robustness.")

        interpretation_text = "\n".join(interpretation)
        interp_path = RQ_DIR / "data" / "step04_quality_interpretation.txt"
        with open(interp_path, 'w', encoding='utf-8') as f:
            f.write(interpretation_text)
        log(f"{interp_path}")
        # Validate Numeric Ranges
        log("\nValidating metric value ranges...")

        validation_result = validate_numeric_range(
            data=metrics_df['value'].values,
            min_val=-1.0,
            max_val=10.0,
            column_name='value'
        )

        if validation_result['valid']:
            log("All metrics in valid ranges")
            log(f"       Metrics: silhouette={silhouette:.4f}, davies_bouldin={davies_bouldin:.4f}, dunn={dunn:.4f}")
        else:
            log(f"Validation failed")
            raise ValueError(f"Metric values outside valid range")
        # SUCCESS
        log("\n" + "="*80)
        log("Step 04 complete")
        log("="*80)
        log("\nOutputs:")
        log(f"  - {metrics_path}")
        log(f"  - {interp_path}")
        log("\nNext: Step 05 (Bootstrap stability assessment)")

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("\n")
        import traceback
        traceback.print_exc(file=open(LOG_FILE, 'a'))
        traceback.print_exc()
        sys.exit(1)
