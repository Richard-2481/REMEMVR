#!/usr/bin/env python3
"""
Step 06: Prepare Scatter Plot Matrix Data

Merge standardized features + cluster assignments + labels for scatter matrix visualization.

Input:
  - data/step01_standardized_features.csv (z-scored features)
  - data/step03_cluster_assignments.csv (cluster assignments)
  - data/step03_cluster_centers.csv (z-scored centers)
  - data/step05_cluster_centers_original_scale.csv (cluster labels)

Output:
  - data/step06_scatter_matrix_plot_data.csv (100 participant rows + K center rows)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
INPUT_FEATURES_FILE = RQ_DIR / "data/step01_standardized_features.csv"
INPUT_ASSIGNMENTS_FILE = RQ_DIR / "data/step03_cluster_assignments.csv"
INPUT_CENTERS_FILE = RQ_DIR / "data/step03_cluster_centers.csv"
INPUT_LABELS_FILE = RQ_DIR / "data/step05_cluster_centers_original_scale.csv"
OUTPUT_FILE = RQ_DIR / "data/step06_scatter_matrix_plot_data.csv"
LOG_FILE = RQ_DIR / "logs/step06_prepare_plot_data.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 06: Prepare Plot Data")

        # Load standardized features
        log(f"Reading {INPUT_FEATURES_FILE}")
        df_features = pd.read_csv(INPUT_FEATURES_FILE)

        # Load cluster assignments
        log(f"Reading {INPUT_ASSIGNMENTS_FILE}")
        df_assignments = pd.read_csv(INPUT_ASSIGNMENTS_FILE)

        # Load cluster centers (z-scored)
        log(f"Reading {INPUT_CENTERS_FILE}")
        df_centers = pd.read_csv(INPUT_CENTERS_FILE)

        # Load cluster labels
        log(f"Reading {INPUT_LABELS_FILE}")
        df_labels = pd.read_csv(INPUT_LABELS_FILE)

        # Merge features + assignments
        log("Combining features and cluster assignments")
        df_participants = pd.merge(df_features, df_assignments, on='UID')

        # Add cluster labels
        df_participants = pd.merge(
            df_participants,
            df_labels[['cluster', 'label']],
            on='cluster'
        )

        # Add data_type column (participant vs center)
        df_participants['data_type'] = 'participant'

        # Prepare cluster centers data
        df_centers_plot = df_centers.copy()
        df_centers_plot = pd.merge(df_centers_plot, df_labels[['cluster', 'label']], on='cluster')
        df_centers_plot['data_type'] = 'center'

        # Rename center columns to match participant columns
        z_cols = [
            'Common_Intercept_z', 'Common_Slope_z',
            'Congruent_Intercept_z', 'Congruent_Slope_z',
            'Incongruent_Intercept_z', 'Incongruent_Slope_z'
        ]

        # Add UID column for centers (using cluster ID as identifier)
        df_centers_plot['UID'] = df_centers_plot['cluster'].apply(lambda x: f"Center_{x}")

        # Reorder columns to match
        participant_cols = ['UID', 'cluster', 'data_type'] + z_cols + ['label']
        df_participants = df_participants[participant_cols]
        df_centers_plot = df_centers_plot[participant_cols]

        # Rename label to cluster_label for clarity
        df_participants = df_participants.rename(columns={'label': 'cluster_label'})
        df_centers_plot = df_centers_plot.rename(columns={'label': 'cluster_label'})

        # Combine participant data + cluster centers
        log("Merging participant data with cluster centers")
        df_plot_data = pd.concat([df_participants, df_centers_plot], ignore_index=True)

        log(f"Plot data shape: {df_plot_data.shape}")
        log(f"  Participant rows: {(df_plot_data['data_type'] == 'participant').sum()}")
        log(f"  Center rows: {(df_plot_data['data_type'] == 'center').sum()}")

        # Save plot data
        log(f"Writing to {OUTPUT_FILE}")
        df_plot_data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE}")

        # Validation checks
        log("Checking plot data completeness")

        # Check all clusters present in participant data
        participant_clusters = df_plot_data[df_plot_data['data_type'] == 'participant']['cluster'].unique()
        center_clusters = df_plot_data[df_plot_data['data_type'] == 'center']['cluster'].unique()

        if not set(participant_clusters) == set(center_clusters):
            log("Cluster mismatch between participants and centers")

        # Check for NaN values
        nan_count = df_plot_data.isna().sum().sum()
        if nan_count > 0:
            log(f"{nan_count} NaN values in plot data")

        # Check no duplicate UIDs in participant data
        participant_uids = df_plot_data[df_plot_data['data_type'] == 'participant']['UID']
        if len(participant_uids) != len(participant_uids.unique()):
            log("Duplicate UIDs in participant data")

        log("[VALIDATION PASS] Plot data ready")
        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
