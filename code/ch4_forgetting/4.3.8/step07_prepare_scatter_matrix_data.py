#!/usr/bin/env python3
"""
================================================================================
STEP 07: Prepare Scatter Matrix Plot Data
================================================================================
RQ: ch5/5.3.8 (Paradigm-Based Clustering)
Purpose: Prepare plot source CSV for scatter plot matrix visualization
         (6x6 grid of z-scores colored by cluster)

Inputs:
  - data/step01_standardized_features.csv (100 x 6 z-scores)
  - data/step03_cluster_assignments.csv (cluster labels)

Outputs:
  - data/step07_scatter_matrix_data.csv (100 rows x 8 columns)
    Columns: UID + 6 z-score features + cluster label

Purpose:
  - Merge standardized features with cluster assignments
  - Create single DataFrame for scatter plot matrix generation (plotting pipeline)
  - Cluster centers from step03_cluster_centers.csv used for reference markers
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_scatter_matrix_data.log"

# Logging
def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis
if __name__ == "__main__":
    try:
        log("="*80)
        log("STEP 07: Prepare Scatter Matrix Plot Data")
        log("="*80)
        # Load Data
        log("\nLoading standardized features and cluster assignments...")

        # Load standardized features (z-scores)
        df_features = pd.read_csv(RQ_DIR / "data" / "step01_standardized_features.csv")
        log(f"Standardized features: {df_features.shape}")
        log(f"         Columns: {list(df_features.columns)}")

        # Load cluster assignments
        df_clusters = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        log(f"Cluster assignments: {df_clusters.shape}")
        log(f"         Columns: {list(df_clusters.columns)}")
        # Merge on UID
        log("\nMerging features with cluster labels...")

        df_plot_data = df_features.merge(df_clusters, on='UID', how='inner')
        log(f"Plot data: {df_plot_data.shape}")
        log(f"         Columns: {list(df_plot_data.columns)}")

        # Verify no missing data
        if df_plot_data['cluster'].isna().any():
            missing_count = df_plot_data['cluster'].isna().sum()
            log(f"{missing_count} participants missing cluster assignments!")
            raise ValueError(f"Found {missing_count} participants without cluster assignments")

        log(f"All {len(df_plot_data)} participants have cluster assignments")
        # Verify Cluster Coverage
        log("\nChecking cluster coverage...")

        unique_clusters = sorted(df_plot_data['cluster'].unique())
        n_clusters = len(unique_clusters)
        log(f"{n_clusters} unique clusters: {unique_clusters}")

        for cluster_id in unique_clusters:
            count = (df_plot_data['cluster'] == cluster_id).sum()
            log(f"  Cluster {cluster_id}: {count} participants")
        # Save Plot Data
        log("\nSaving scatter matrix plot data...")

        plot_path = RQ_DIR / "data" / "step07_scatter_matrix_data.csv"
        df_plot_data.to_csv(plot_path, index=False, encoding='utf-8')
        log(f"{plot_path}")
        log(f"        {len(df_plot_data)} rows x {len(df_plot_data.columns)} columns")
        # Display Feature Summary
        log("\nFeature statistics by cluster...")

        feature_cols = [col for col in df_plot_data.columns if col.endswith('_z')]
        log(f"\n{'Cluster':<10} {'Feature':<35} {'Mean':>8} {'SD':>7}")
        log("-"*65)

        for cluster_id in unique_clusters:
            cluster_data = df_plot_data[df_plot_data['cluster'] == cluster_id]
            for feature in feature_cols:
                mean_val = cluster_data[feature].mean()
                sd_val = cluster_data[feature].std()
                log(f"{cluster_id:<10} {feature:<35} {mean_val:>8.3f} {sd_val:>7.3f}")
        # Validate Plot Data Completeness
        log("\nValidating plot data completeness...")

        # Expected clusters: 0 to K-1
        expected_clusters = list(range(n_clusters))

        validation_result = validate_plot_data_completeness(
            plot_data=df_plot_data,
            required_domains=[],  # Not domain-based
            required_groups=expected_clusters,
            domain_col=None,
            group_col='cluster'
        )

        if validation_result['valid']:
            log("Plot data completeness validation successful")
            log(f"       All {len(df_plot_data)} participants included")
            log(f"       All {n_clusters} clusters present in data")
        else:
            log(f"Validation error: {validation_result['message']}")
            raise ValueError(validation_result['message'])
        # Reference Information for plotting pipeline
        log("\nAdditional data available for plotting:")
        log(f"       - Cluster centers: data/step03_cluster_centers.csv")
        log(f"       - Cluster sizes:   data/step03_cluster_sizes.txt")
        log(f"       - Cluster profiles: data/step06_cluster_profiles.txt")
        log("")
        log("       plotting pipeline can use these files to:")
        log("       - Add cluster center markers to scatter plots")
        log("       - Label clusters with profile descriptions")
        log("       - Display cluster sizes in legend")
        # SUCCESS
        log("\n" + "="*80)
        log("Step 07 complete")
        log("="*80)
        log("\nOutputs:")
        log(f"  - {plot_path}")
        log(f"    {len(df_plot_data)} participants x {len(feature_cols)} features + cluster label")
        log("\nPlot-Ready Data Structure:")
        log(f"  - UID column for participant tracking")
        log(f"  - {len(feature_cols)} z-score columns for scatter matrix axes")
        log(f"  - cluster column for color coding")
        log("\nNext: plotting pipeline generates scatter plot matrix visualization")

        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("\n")
        import traceback
        traceback.print_exc(file=open(LOG_FILE, 'a'))
        traceback.print_exc()
        sys.exit(1)
