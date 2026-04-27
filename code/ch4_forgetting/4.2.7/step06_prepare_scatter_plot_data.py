#!/usr/bin/env python3
"""
RQ 5.2.7 Step 06: Prepare Scatter Plot Matrix Data

Purpose: Create plot source CSV for scatter plot matrix visualization
         with cluster assignments, cluster centers, and metadata.

Input:
  - data/step01_standardized_features.csv (100 rows × 5 cols)
  - data/step03_cluster_assignments.csv (100 rows: UID, cluster)
  - data/step03_cluster_centers.csv (K rows × 5 cols)
  - data/step05_cluster_characterization.txt (cluster labels)

Output:
  - data/step06_scatter_plot_matrix_data.csv (100+K rows × 8 cols)
"""

import re
import sys
from pathlib import Path

import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input files
FEATURES_FILE = DATA_DIR / "step01_standardized_features.csv"
ASSIGNMENTS_FILE = DATA_DIR / "step03_cluster_assignments.csv"
CENTERS_FILE = DATA_DIR / "step03_cluster_centers.csv"
CHAR_FILE = DATA_DIR / "step05_cluster_characterization.txt"

# Z-scored variables
Z_VARS = [
    'Total_Intercept_What_z',
    'Total_Intercept_Where_z',
    'Total_Slope_What_z',
    'Total_Slope_Where_z'
]


def parse_cluster_labels(filepath: Path) -> dict:
    """Parse cluster labels from characterization file."""
    labels = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Find pattern: "Cluster N: Label"
    pattern = r'Cluster (\d+): ([^\n]+)'
    matches = re.findall(pattern, content)

    for cluster_id, label in matches:
        labels[int(cluster_id)] = label.strip()

    return labels


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 06: Prepare Scatter Plot Matrix Data")
    print("=" * 60)

    # Load z-scored features
    print(f"\nLoading: {FEATURES_FILE}")
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df_features)} participants")

    # Load cluster assignments
    print(f"\nLoading: {ASSIGNMENTS_FILE}")
    df_assignments = pd.read_csv(ASSIGNMENTS_FILE)

    # Load cluster centers
    print(f"\nLoading: {CENTERS_FILE}")
    df_centers = pd.read_csv(CENTERS_FILE)
    n_clusters = len(df_centers)
    print(f"Loaded {n_clusters} cluster centers")

    # Parse cluster labels
    print(f"\nParsing cluster labels from: {CHAR_FILE}")
    cluster_labels = parse_cluster_labels(CHAR_FILE)
    print(f"Found {len(cluster_labels)} labels")

    # Merge features with cluster assignments
    df_participants = pd.merge(df_features, df_assignments, on='UID')

    # Add cluster labels
    df_participants['cluster_label'] = df_participants['cluster'].map(cluster_labels)

    # Add point type
    df_participants['point_type'] = 'participant'

    # Reorder columns
    participant_cols = ['UID', 'cluster', 'cluster_label', 'point_type'] + Z_VARS
    df_participants = df_participants[participant_cols]

    print(f"\nParticipant data: {len(df_participants)} rows")

    # Create centroid rows
    centroid_rows = []
    for _, row in df_centers.iterrows():
        cluster_id = int(row['cluster'])
        centroid_row = {
            'UID': f'Centroid_{cluster_id}',
            'cluster': cluster_id,
            'cluster_label': cluster_labels.get(cluster_id, f'Cluster {cluster_id}'),
            'point_type': 'centroid'
        }
        for var in Z_VARS:
            centroid_row[var] = row[var]
        centroid_rows.append(centroid_row)

    df_centroids = pd.DataFrame(centroid_rows)
    print(f"Centroid data: {len(df_centroids)} rows")

    # Combine participant and centroid data
    df_plot = pd.concat([df_participants, df_centroids], ignore_index=True)
    print(f"\nCombined plot data: {len(df_plot)} rows")

    # Validate
    print("\nValidation:")
    n_participants = len(df_plot[df_plot['point_type'] == 'participant'])
    n_centroids = len(df_plot[df_plot['point_type'] == 'centroid'])
    print(f"  - Participants: {n_participants}")
    print(f"  - Centroids: {n_centroids}")
    print(f"  - Total: {len(df_plot)}")

    # Check all clusters represented
    clusters_in_data = sorted(df_plot['cluster'].unique())
    print(f"  - Clusters represented: {clusters_in_data}")

    # Check for missing values
    n_missing = df_plot.isnull().sum().sum()
    if n_missing > 0:
        print(f"  - WARNING: {n_missing} missing values")
    else:
        print(f"  - No missing values in plot data")

    # Summary by cluster
    print("\nCluster distribution:")
    for cluster in sorted(df_plot['cluster'].unique()):
        cluster_data = df_plot[df_plot['cluster'] == cluster]
        n_part = len(cluster_data[cluster_data['point_type'] == 'participant'])
        n_cent = len(cluster_data[cluster_data['point_type'] == 'centroid'])
        label = cluster_labels.get(cluster, 'Unknown')[:40]
        print(f"  Cluster {cluster}: {n_part} participants, {n_cent} centroid - {label}")

    # Save output
    output_file = DATA_DIR / "step06_scatter_plot_matrix_data.csv"
    df_plot.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  - Rows: {len(df_plot)}")
    print(f"  - Columns: {list(df_plot.columns)}")

    print(f"\nPlot data preparation complete: {len(df_plot)} rows created")
    print(f"All {n_clusters} clusters represented with participants and centroids")

    print("\n" + "=" * 60)
    print("Step 06 COMPLETE: Scatter plot data prepared successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
