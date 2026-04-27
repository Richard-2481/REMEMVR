#!/usr/bin/env python3
"""
RQ 5.2.7 Step 05: Characterize Clusters

Purpose: Compute mean intercept and slope per cluster for each domain,
         assign interpretive labels based on domain-specific patterns.

Input:
  - data/step00_random_effects_from_rq526.csv (100 rows × 5 cols, raw scale)
  - data/step03_cluster_assignments.csv (100 rows: UID, cluster)
  - data/step03_cluster_sizes.csv (K rows: cluster, N, percent)

Output:
  - data/step05_cluster_summary_statistics.csv (K*4 rows: stats per variable)
  - data/step05_cluster_characterization.txt (interpretive descriptions)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input files
RAW_EFFECTS_FILE = DATA_DIR / "step00_random_effects_from_rq526.csv"
ASSIGNMENTS_FILE = DATA_DIR / "step03_cluster_assignments.csv"
SIZES_FILE = DATA_DIR / "step03_cluster_sizes.csv"

# Clustering variables (raw scale)
RAW_VARS = [
    'Total_Intercept_What',
    'Total_Intercept_Where',
    'Total_Slope_What',
    'Total_Slope_Where'
]


def interpret_intercept(value: float) -> str:
    """Interpret intercept value relative to grand mean (0)."""
    if value > 0.3:
        return "High"
    elif value > -0.3:
        return "Average"
    else:
        return "Low"


def interpret_slope(value: float) -> str:
    """Interpret slope value (negative = forgetting)."""
    # Slopes are typically very small (range ~-0.04 to +0.04)
    if value > 0.01:
        return "Improving"  # Unusual - slight gain over time
    elif value > -0.01:
        return "Stable"
    elif value > -0.02:
        return "Slow Decline"
    else:
        return "Fast Decline"


def generate_cluster_label(row: pd.Series) -> str:
    """Generate interpretive label for a cluster based on its pattern."""
    what_int = row['Total_Intercept_What']
    where_int = row['Total_Intercept_Where']
    what_slope = row['Total_Slope_What']
    where_slope = row['Total_Slope_Where']

    # Characterize baseline (intercepts)
    int_what_cat = interpret_intercept(what_int)
    int_where_cat = interpret_intercept(where_int)

    # Characterize forgetting (slopes)
    slope_what_cat = interpret_slope(what_slope)
    slope_where_cat = interpret_slope(where_slope)

    # Generate label
    if int_what_cat == int_where_cat:
        baseline_label = f"{int_what_cat} Baseline"
    else:
        baseline_label = f"{int_what_cat} What / {int_where_cat} Where"

    if slope_what_cat == slope_where_cat:
        slope_label = slope_what_cat
    else:
        slope_label = f"{slope_what_cat} What / {slope_where_cat} Where"

    return f"{baseline_label}, {slope_label}"


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 05: Characterize Clusters")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {RAW_EFFECTS_FILE}")
    df_raw = pd.read_csv(RAW_EFFECTS_FILE)
    print(f"Loaded {len(df_raw)} rows")

    print(f"\nLoading: {ASSIGNMENTS_FILE}")
    df_assignments = pd.read_csv(ASSIGNMENTS_FILE)

    print(f"\nLoading: {SIZES_FILE}")
    df_sizes = pd.read_csv(SIZES_FILE)
    n_clusters = len(df_sizes)

    # Merge raw effects with cluster assignments
    df = pd.merge(df_raw, df_assignments, on='UID')
    print(f"\nMerged data: {len(df)} rows, {n_clusters} clusters")

    # Compute summary statistics per cluster per variable
    print("\nComputing cluster summary statistics...")
    summary_rows = []

    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]

        for var in RAW_VARS:
            summary_rows.append({
                'cluster': cluster,
                'variable': var,
                'mean': cluster_data[var].mean(),
                'SD': cluster_data[var].std(),
                'min': cluster_data[var].min(),
                'max': cluster_data[var].max()
            })

    df_summary = pd.DataFrame(summary_rows)

    # Save summary statistics
    summary_file = DATA_DIR / "step05_cluster_summary_statistics.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    # Compute cluster means for labeling
    cluster_means = df.groupby('cluster')[RAW_VARS].mean()

    # Generate characterizations
    print("\n" + "=" * 60)
    print("Cluster Characterizations")
    print("=" * 60)

    characterizations = []

    for cluster in sorted(df['cluster'].unique()):
        size_row = df_sizes[df_sizes['cluster'] == cluster].iloc[0]
        n = int(size_row['N'])
        pct = size_row['percent']

        means = cluster_means.loc[cluster]
        label = generate_cluster_label(means)

        print(f"\n--- Cluster {cluster}: {label} ---")
        print(f"N = {n} ({pct:.1f}%)")
        print(f"\nMean values (raw scale):")
        print(f"  What Intercept:  {means['Total_Intercept_What']:>8.4f}")
        print(f"  Where Intercept: {means['Total_Intercept_Where']:>8.4f}")
        print(f"  What Slope:      {means['Total_Slope_What']:>8.4f}")
        print(f"  Where Slope:     {means['Total_Slope_Where']:>8.4f}")

        # Generate description
        what_int_cat = interpret_intercept(means['Total_Intercept_What'])
        where_int_cat = interpret_intercept(means['Total_Intercept_Where'])
        what_slope_cat = interpret_slope(means['Total_Slope_What'])
        where_slope_cat = interpret_slope(means['Total_Slope_Where'])

        description_parts = []
        if what_int_cat == where_int_cat:
            description_parts.append(f"shows {what_int_cat.lower()} baseline memory across both domains")
        else:
            description_parts.append(f"shows {what_int_cat.lower()} What memory and {where_int_cat.lower()} Where memory")

        if what_slope_cat == where_slope_cat:
            description_parts.append(f"with {what_slope_cat.lower()} patterns over time")
        else:
            description_parts.append(f"with {what_slope_cat.lower()} What and {where_slope_cat.lower()} Where trajectories")

        description = f"This cluster {'. '.join(description_parts)}."

        characterizations.append({
            'cluster': cluster,
            'label': label,
            'n': n,
            'percent': pct,
            'means': means.to_dict(),
            'description': description,
            'what_int_cat': what_int_cat,
            'where_int_cat': where_int_cat,
            'what_slope_cat': what_slope_cat,
            'where_slope_cat': where_slope_cat
        })

    # Save characterization report
    char_file = DATA_DIR / "step05_cluster_characterization.txt"
    with open(char_file, 'w') as f:
        f.write("RQ 5.2.7 Cluster Characterization Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("Note: Cluster quality assessment = POOR (silhouette = 0.34)\n")
        f.write("Interpret clusters cautiously - boundaries are fuzzy.\n")
        f.write("Bootstrap stability is high (Jaccard = 0.88), indicating\n")
        f.write("consistent participant groupings despite overlap.\n\n")

        for char in characterizations:
            f.write("-" * 55 + "\n")
            f.write(f"Cluster {char['cluster']}: {char['label']}\n")
            f.write(f"N = {char['n']} ({char['percent']:.1f}%)\n")
            f.write("-" * 55 + "\n\n")

            f.write("Mean values (raw theta scale):\n")
            for var, val in char['means'].items():
                short_name = var.replace('Total_', '')
                f.write(f"  {short_name}: {val:>8.4f}\n")
            f.write("\n")

            f.write(f"Interpretation:\n")
            f.write(f"  Baseline (What):  {char['what_int_cat']}\n")
            f.write(f"  Baseline (Where): {char['where_int_cat']}\n")
            f.write(f"  Trajectory (What):  {char['what_slope_cat']}\n")
            f.write(f"  Trajectory (Where): {char['where_slope_cat']}\n\n")

            f.write(f"{char['description']}\n\n")

        # Summary table
        f.write("\n" + "=" * 55 + "\n")
        f.write("Summary Table\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"{'Cluster':>8} {'N':>6} {'%':>6} {'Label':<35}\n")
        f.write("-" * 60 + "\n")
        for char in characterizations:
            f.write(f"{char['cluster']:>8} {char['n']:>6} {char['percent']:>5.1f}% {char['label']:<35}\n")

    print(f"\nSaved: {char_file}")

    print("\n" + "=" * 60)
    print("Step 05 COMPLETE: Clusters characterized successfully")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
