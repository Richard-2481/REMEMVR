#!/usr/bin/env python3
"""
Step 05: Characterize Clusters by Congruence-Specific Patterns

Back-transform cluster centers to original scale, compute summary statistics,
assign interpretive labels.

Input:
  - data/step03_cluster_centers.csv (z-scored centers)
  - data/step00_random_effects_from_rq546.csv (original scale features)
  - data/step03_cluster_assignments.csv (cluster assignments)

Output:
  - data/step05_cluster_centers_original_scale.csv (back-transformed centers with labels)
  - data/step05_cluster_summary_stats.csv (cluster-specific summary statistics)
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
INPUT_CENTERS_FILE = RQ_DIR / "data/step03_cluster_centers.csv"
INPUT_ORIGINAL_FILE = RQ_DIR / "data/step00_random_effects_from_rq546.csv"
INPUT_ASSIGNMENTS_FILE = RQ_DIR / "data/step03_cluster_assignments.csv"
OUTPUT_CENTERS_FILE = RQ_DIR / "data/step05_cluster_centers_original_scale.csv"
OUTPUT_STATS_FILE = RQ_DIR / "data/step05_cluster_summary_stats.csv"
LOG_FILE = RQ_DIR / "logs/step05_characterize_clusters.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 05: Characterize Clusters")

        # Load cluster centers (z-scored)
        log(f"Reading {INPUT_CENTERS_FILE}")
        df_centers_z = pd.read_csv(INPUT_CENTERS_FILE)

        # Load original scale features (for back-transformation means/SDs)
        log(f"Reading {INPUT_ORIGINAL_FILE}")
        df_original = pd.read_csv(INPUT_ORIGINAL_FILE)

        feature_cols = [
            'Common_Intercept', 'Common_Slope',
            'Congruent_Intercept', 'Congruent_Slope',
            'Incongruent_Intercept', 'Incongruent_Slope'
        ]

        # Compute means and SDs from original features
        means = df_original[feature_cols].mean().values
        sds = df_original[feature_cols].std().values

        log("Converting z-scores to original scale")
        log(f"  Means: {means}")
        log(f"  SDs: {sds}")

        # Back-transform cluster centers: original_value = z * SD + mean
        df_centers_orig = df_centers_z.copy()
        z_cols = [
            'Common_Intercept_z', 'Common_Slope_z',
            'Congruent_Intercept_z', 'Congruent_Slope_z',
            'Incongruent_Intercept_z', 'Incongruent_Slope_z'
        ]

        for i, (z_col, orig_col) in enumerate(zip(z_cols, feature_cols)):
            df_centers_orig[orig_col] = df_centers_z[z_col] * sds[i] + means[i]

        # Drop z-scored columns
        df_centers_orig = df_centers_orig[['cluster'] + feature_cols]

        # Load assignments to get cluster sizes
        log(f"Reading {INPUT_ASSIGNMENTS_FILE}")
        df_assignments = pd.read_csv(INPUT_ASSIGNMENTS_FILE)
        cluster_sizes = df_assignments['cluster'].value_counts().to_dict()

        # Add cluster sizes to centers
        df_centers_orig['N'] = df_centers_orig['cluster'].map(cluster_sizes)

        # Generate interpretive labels (simple: rank by mean intercept)
        mean_intercept = df_centers_orig[['Common_Intercept', 'Congruent_Intercept', 'Incongruent_Intercept']].mean(axis=1)
        df_centers_orig['label'] = ['High' if x > 0.1 else 'Low' if x < -0.1 else 'Medium' for x in mean_intercept]

        # Save back-transformed centers
        log(f"Writing to {OUTPUT_CENTERS_FILE}")
        df_centers_orig.to_csv(OUTPUT_CENTERS_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_CENTERS_FILE}")

        # Compute cluster-specific summary statistics
        log("Computing cluster-specific summary statistics")
        df_full = pd.merge(df_assignments, df_original, on='UID')

        summary_rows = []
        for cluster_id in sorted(df_centers_orig['cluster'].unique()):
            df_cluster = df_full[df_full['cluster'] == cluster_id]
            for feature in feature_cols:
                summary_rows.append({
                    'cluster': cluster_id,
                    'feature': feature,
                    'mean': df_cluster[feature].mean(),
                    'SD': df_cluster[feature].std(),
                    'min': df_cluster[feature].min(),
                    'max': df_cluster[feature].max()
                })

        df_summary = pd.DataFrame(summary_rows)

        # Save summary statistics
        log(f"Writing to {OUTPUT_STATS_FILE}")
        df_summary.to_csv(OUTPUT_STATS_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_STATS_FILE}")

        # Report cluster characteristics
        log("")
        for _, row in df_centers_orig.iterrows():
            log(f"  Cluster {int(row['cluster'])}: N={int(row['N'])}, label={row['label']}")
            log(f"    Common: I={row['Common_Intercept']:.4f}, S={row['Common_Slope']:.6f}")
            log(f"    Congruent: I={row['Congruent_Intercept']:.4f}, S={row['Congruent_Slope']:.6f}")
            log(f"    Incongruent: I={row['Incongruent_Intercept']:.4f}, S={row['Incongruent_Slope']:.6f}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
