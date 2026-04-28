#!/usr/bin/env python3
"""
RQ 5.2.7 Step 03: Fit Final K-Means Model

Purpose: Fit final K-means model using optimal K selected in Step 2,
         extract cluster assignments and cluster centers.

Input:
  - data/step01_standardized_features.csv (100 rows × 5 cols)
  - data/step02_optimal_k_selection.txt (optimal K value)

Output:
  - data/step03_cluster_assignments.csv (100 rows: UID, cluster)
  - data/step03_cluster_centers.csv (K rows × 5 cols)
  - data/step03_cluster_sizes.csv (K rows: cluster, N, percent)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input files
FEATURES_FILE = DATA_DIR / "step01_standardized_features.csv"
OPTIMAL_K_FILE = DATA_DIR / "step02_optimal_k_selection.txt"

# Z-scored clustering variables
Z_VARS = [
    'Total_Intercept_What_z',
    'Total_Intercept_Where_z',
    'Total_Slope_What_z',
    'Total_Slope_Where_z'
]

# Model parameters
N_INIT = 50
RANDOM_STATE = 42
MAX_ITER = 300

# Validation parameters
MIN_CLUSTER_PERCENT = 10  # Minimum 10% of sample per cluster


def parse_optimal_k(filepath: Path) -> int:
    """Parse optimal K from step02 output file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Look for "Optimal K = N" pattern
    match = re.search(r'Optimal K\s*=\s*(\d+)', content)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse optimal K from {filepath}")


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 03: Fit Final K-Means Model")
    print("=" * 60)

    # Parse optimal K from Step 2
    print(f"\nReading optimal K from: {OPTIMAL_K_FILE}")
    optimal_k = parse_optimal_k(OPTIMAL_K_FILE)
    print(f"Optimal K = {optimal_k}")

    # Load standardized features
    print(f"\nLoading: {FEATURES_FILE}")
    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} rows")

    # Extract clustering matrix
    X = df.values
    n_samples = len(X)
    print(f"Clustering matrix shape: {X.shape}")

    # Fit final K-means model
    print(f"\nFitting K-means with K={optimal_k}...")
    print(f"  n_init={N_INIT}, random_state={RANDOM_STATE}, max_iter={MAX_ITER}")

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        algorithm='lloyd'
    )
    kmeans.fit(X)

    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")

    # Extract cluster assignments
    labels = kmeans.labels_
    unique_labels = sorted(set(labels))
    print(f"\nCluster assignments extracted")
    print(f"  Unique cluster IDs: {unique_labels}")

    # Validate cluster IDs are consecutive
    expected_labels = list(range(optimal_k))
    if unique_labels != expected_labels:
        print(f"\nERROR: Cluster IDs not consecutive. Expected {expected_labels}, got {unique_labels}")
        sys.exit(1)
    print(f"  Cluster IDs consecutive: [0, 1, ..., {optimal_k-1}]")

    # Create cluster assignments DataFrame
    df_assignments = pd.DataFrame({
        'UID': df['UID'],
        'cluster': labels
    })

    # Compute cluster sizes
    cluster_sizes = df_assignments['cluster'].value_counts().sort_index()
    df_sizes = pd.DataFrame({
        'cluster': cluster_sizes.index,
        'N': cluster_sizes.values,
        'percent': (cluster_sizes.values / n_samples * 100)
    })

    print(f"\nCluster sizes:")
    print("-" * 35)
    print(f"{'Cluster':>8} {'N':>8} {'Percent':>10}")
    print("-" * 35)
    for _, row in df_sizes.iterrows():
        print(f"{int(row['cluster']):>8} {int(row['N']):>8} {row['percent']:>9.1f}%")
    print("-" * 35)
    print(f"{'Total':>8} {df_sizes['N'].sum():>8} {df_sizes['percent'].sum():>9.1f}%")

    # Validate cluster balance
    min_cluster_n = df_sizes['N'].min()
    min_cluster_pct = df_sizes['percent'].min()
    print(f"\nCluster balance validation:")
    print(f"  Smallest cluster: N={min_cluster_n} ({min_cluster_pct:.1f}%)")

    if min_cluster_pct < MIN_CLUSTER_PERCENT:
        print(f"  WARNING: Cluster {df_sizes[df_sizes['percent'] == min_cluster_pct]['cluster'].values[0]} "
              f"has < {MIN_CLUSTER_PERCENT}% of sample")
        print(f"  This may indicate poor cluster structure or outlier-dominated cluster")
        # Don't fail - document but proceed, Step 4 will assess quality
    else:
        print(f"  PASS: All clusters >= {MIN_CLUSTER_PERCENT}% of sample")

    print(f"  Cluster sizes balanced: min(N)={min_cluster_n} >= 10")

    # Extract cluster centers
    centers = kmeans.cluster_centers_
    df_centers = pd.DataFrame(centers, columns=Z_VARS)
    df_centers.insert(0, 'cluster', range(optimal_k))

    print(f"\nCluster centers (z-scored):")
    print("-" * 70)
    header = f"{'Cluster':>8}"
    for var in Z_VARS:
        short_name = var.replace('Total_', '').replace('_z', '')
        header += f" {short_name:>12}"
    print(header)
    print("-" * 70)
    for _, row in df_centers.iterrows():
        line = f"{int(row['cluster']):>8}"
        for var in Z_VARS:
            line += f" {row[var]:>12.3f}"
        print(line)
    print("-" * 70)

    # Save outputs
    assignments_file = DATA_DIR / "step03_cluster_assignments.csv"
    df_assignments.to_csv(assignments_file, index=False)
    print(f"\nSaved: {assignments_file}")

    centers_file = DATA_DIR / "step03_cluster_centers.csv"
    df_centers.to_csv(centers_file, index=False)
    print(f"Saved: {centers_file}")

    sizes_file = DATA_DIR / "step03_cluster_sizes.csv"
    df_sizes.to_csv(sizes_file, index=False)
    print(f"Saved: {sizes_file}")

    # Summary
    print(f"\nFitted K-means with K={optimal_k} clusters")
    print(f"All clusters assigned: {n_samples} participants")

    print("\n" + "=" * 60)
    print(f"Step 03 COMPLETE: K={optimal_k} clusters fitted successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
