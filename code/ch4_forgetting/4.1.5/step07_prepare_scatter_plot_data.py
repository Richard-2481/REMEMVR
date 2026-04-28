#!/usr/bin/env python3
"""prepare_scatter_plot_data: Prepare scatter plot source data for plotting pipeline visualization. Creates"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_scatter_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: Prepare Scatter Plot Data")
        # Load Input Data
        #           Cluster assignments (100 x 2: UID, cluster)
        #           Cluster centers (K x 3: cluster, Intercept_z_center, Slope_z_center)
        #           Silhouette score (single float value)

        log("Loading standardized features...")
        standardized_features = pd.read_csv(RQ_DIR / "data/step01_standardized_features.csv")
        log(f"standardized_features ({len(standardized_features)} rows, {len(standardized_features.columns)} cols)")

        log("Loading cluster assignments...")
        cluster_assignments = pd.read_csv(RQ_DIR / "data/step03_cluster_assignments.csv")
        log(f"cluster_assignments ({len(cluster_assignments)} rows, {len(cluster_assignments.columns)} cols)")

        log("Loading cluster centers...")
        cluster_centers = pd.read_csv(RQ_DIR / "data/step03_cluster_centers.csv")
        log(f"cluster_centers ({len(cluster_centers)} rows, {len(cluster_centers.columns)} cols)")

        log("Loading silhouette score...")
        with open(RQ_DIR / "data/step05_silhouette_score.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # First line should contain silhouette score (e.g., "0.45" or "Silhouette: 0.45")
            silhouette_line = lines[0].strip()
            # Extract float value (handle "Silhouette: 0.45" or "0.45" formats)
            if ':' in silhouette_line:
                silhouette_score = float(silhouette_line.split(':')[-1].strip())
            else:
                silhouette_score = float(silhouette_line)
        log(f"silhouette_score = {silhouette_score:.3f}")
        # Merge Features with Assignments

        log("Merging standardized features with cluster assignments...")
        # Merge on UID (inner join ensures only matched participants)
        merged_data = standardized_features.merge(cluster_assignments, on='UID', how='inner')
        log(f"Merged data: {len(merged_data)} rows (expected 100)")

        # Validate merge produced expected row count
        if len(merged_data) != 100:
            raise ValueError(f"Merge produced {len(merged_data)} rows, expected 100 (all participants)")
        # Create Scatter Plot Data CSV
        # Output: Participant points for scatter plot (drop UID, keep features + cluster)
        # Columns: Intercept_z, Slope_z, cluster
        # These are the points plotted in 2D space colored by cluster

        log("Creating scatter plot data CSV...")
        scatter_plot_data = merged_data[['Intercept_z', 'Slope_z', 'cluster']].copy()
        scatter_plot_data.to_csv(RQ_DIR / "data/step07_scatter_plot_data.csv", index=False, encoding='utf-8')
        log(f"data/step07_scatter_plot_data.csv ({len(scatter_plot_data)} rows, {len(scatter_plot_data.columns)} cols)")
        # Create Cluster Centers CSV
        # Output: Cluster centroids for overlay markers
        # Columns: Intercept_z_center, Slope_z_center, cluster
        # These are the large markers overlaid on participant points

        log("Creating cluster centers CSV...")
        scatter_plot_centers = cluster_centers[['Intercept_z_center', 'Slope_z_center', 'cluster']].copy()
        scatter_plot_centers.to_csv(RQ_DIR / "data/step07_scatter_plot_centers.csv", index=False, encoding='utf-8')
        log(f"data/step07_scatter_plot_centers.csv ({len(scatter_plot_centers)} rows, {len(scatter_plot_centers.columns)} cols)")
        # Create Plot Metadata YAML
        # Output: Plot configuration for plotting pipeline
        # Contains: silhouette_score, K_final, axis_labels, reference_lines

        log("Creating plot metadata YAML...")
        K_final = len(cluster_centers)  # Number of clusters

        plot_metadata = {
            'silhouette_score': float(silhouette_score),
            'K_final': int(K_final),
            'axis_labels': {
                'x': 'Random Intercept (z-scored)',
                'y': 'Random Slope (z-scored)'
            },
            'reference_lines': {
                'x_zero': True,  # Draw vertical line at x=0 (mean intercept)
                'y_zero': True   # Draw horizontal line at y=0 (mean slope)
            },
            'interpretation': {
                'silhouette': (
                    'Strong (>= 0.50)' if silhouette_score >= 0.50 else
                    'Reasonable (0.25-0.49)' if silhouette_score >= 0.25 else
                    'Weak (< 0.25)'
                )
            }
        }

        with open(RQ_DIR / "data/step07_scatter_plot_metadata.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(plot_metadata, f, default_flow_style=False, sort_keys=False)
        log(f"data/step07_scatter_plot_metadata.yaml (silhouette={silhouette_score:.3f}, K={K_final})")
        # Run Validation Tool
        # Validates: All participants present, cluster IDs match, no NaN
        # Note: For this RQ, we don't have domains/groups - use None for those parameters

        log("Running validate_plot_data_completeness...")

        # NOTE: validate_plot_data_completeness expects domain/group columns
        # For clustering scatter plot, we treat 'cluster' as domain_col
        # Required domains are the cluster IDs (0, 1 for K=2)
        expected_cluster_ids = list(range(K_final))

        validation_result = validate_plot_data_completeness(
            plot_data=scatter_plot_data,
            required_domains=expected_cluster_ids,  # Cluster IDs 0, 1
            required_groups=[],                      # No group variable - empty list instead of None
            domain_col='cluster',                    # Use cluster as domain identifier
            group_col='cluster'                      # Use cluster as placeholder (required)
        )

        # Report validation results
        log(f"valid: {validation_result['valid']}")
        log(f"message: {validation_result['message']}")

        if not validation_result['valid']:
            raise ValueError(f"Plot data validation failed: {validation_result['message']}")

        # Additional manual checks specific to clustering scatter plot
        log("Additional clustering-specific checks...")

        # Check 100 participant rows
        if len(scatter_plot_data) != 100:
            raise ValueError(f"Expected 100 participant rows, got {len(scatter_plot_data)}")
        log(f"Participant rows: {len(scatter_plot_data)} (expected 100) ")

        # Check K_final center rows
        if len(scatter_plot_centers) != K_final:
            raise ValueError(f"Expected {K_final} center rows, got {len(scatter_plot_centers)}")
        log(f"Center rows: {len(scatter_plot_centers)} (expected {K_final}) ")

        # Check cluster IDs match between data and centers
        data_clusters = set(scatter_plot_data['cluster'].unique())
        center_clusters = set(scatter_plot_centers['cluster'].unique())
        if data_clusters != center_clusters:
            raise ValueError(f"Cluster ID mismatch: data={data_clusters}, centers={center_clusters}")
        log(f"Cluster IDs match: {sorted(data_clusters)} ")

        # Check no NaN values
        if scatter_plot_data.isna().any().any():
            raise ValueError("NaN values found in scatter_plot_data")
        if scatter_plot_centers.isna().any().any():
            raise ValueError("NaN values found in scatter_plot_centers")
        log("No NaN values ")

        log("Step 7 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
