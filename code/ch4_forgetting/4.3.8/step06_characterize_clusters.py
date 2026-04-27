#!/usr/bin/env python3
"""
================================================================================
STEP 06: Characterize Clusters
================================================================================
RQ: ch5/5.3.8 (Paradigm-Based Clustering)
Purpose: Characterize clusters by computing descriptive statistics per feature
         on original scale (intercepts/slopes) with interpretive labels

Inputs:
  - data/step00_random_effects_wide.csv (100 x 6 original-scale features)
  - data/step03_cluster_assignments.csv (cluster labels)

Outputs:
  - data/step06_cluster_characterization.csv (long format: K x 6 rows)
  - data/step06_cluster_profiles.txt (narrative descriptions)

Characterization:
  - For each cluster: mean, SD, min, max for each of 6 features
  - Interpretive labels based on pattern (e.g., "High performers", "Fast forgetters")
  - Original scale for clinical/practical interpretation
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_cluster_summary_stats

# ============================================================================
# Configuration
# ============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_characterize_clusters.log"

# ============================================================================
# Logging
# ============================================================================
def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# ============================================================================
# Cluster Interpretation Helper
# ============================================================================
def interpret_cluster_pattern(cluster_id, cluster_data):
    """
    Generate interpretive label and description for cluster based on feature patterns.

    Args:
        cluster_id: Cluster identifier (0, 1, 2, ...)
        cluster_data: DataFrame with mean values for 6 features

    Returns:
        tuple: (label, description)
    """
    # Extract mean values
    intercept_free = cluster_data.loc[cluster_data['feature'] == 'Total_Intercept_Free', 'mean'].values[0]
    slope_free = cluster_data.loc[cluster_data['feature'] == 'Total_Slope_Free', 'mean'].values[0]
    intercept_cued = cluster_data.loc[cluster_data['feature'] == 'Total_Intercept_Cued', 'mean'].values[0]
    slope_cued = cluster_data.loc[cluster_data['feature'] == 'Total_Slope_Cued', 'mean'].values[0]
    intercept_recog = cluster_data.loc[cluster_data['feature'] == 'Total_Intercept_Recognition', 'mean'].values[0]
    slope_recog = cluster_data.loc[cluster_data['feature'] == 'Total_Slope_Recognition', 'mean'].values[0]

    # Compute overall performance (average intercept across paradigms)
    avg_intercept = (intercept_free + intercept_cued + intercept_recog) / 3

    # Compute overall forgetting (average slope across paradigms)
    avg_slope = (slope_free + slope_cued + slope_recog) / 3

    # Classify performance level
    if avg_intercept > 40:
        perf_level = "High"
    elif avg_intercept > 30:
        perf_level = "Moderate"
    else:
        perf_level = "Low"

    # Classify forgetting pattern
    if avg_slope > 0.5:
        forgetting = "Minimal forgetting"
    elif avg_slope > -0.5:
        forgetting = "Stable retention"
    elif avg_slope > -2.0:
        forgetting = "Moderate forgetting"
    else:
        forgetting = "Rapid forgetting"

    # Check for paradigm selectivity
    intercepts = [intercept_free, intercept_cued, intercept_recog]
    slopes = [slope_free, slope_cued, slope_recog]
    intercept_range = max(intercepts) - min(intercepts)
    slope_range = max(slopes) - min(slopes)

    # Determine dominant paradigm if selective
    paradigm_selective = False
    dominant_paradigm = None
    if intercept_range > 10:  # Substantial difference in intercepts
        paradigm_selective = True
        max_idx = intercepts.index(max(intercepts))
        paradigms = ['Free', 'Cued', 'Recognition']
        dominant_paradigm = paradigms[max_idx]

    # Generate label
    if paradigm_selective:
        label = f"{perf_level} performers - {dominant_paradigm} dominant"
    else:
        label = f"{perf_level} performers - {forgetting}"

    # Generate description
    description = []
    description.append(f"Cluster {cluster_id}: {label}")
    description.append(f"  Performance: {perf_level} (mean intercept: {avg_intercept:.1f})")
    description.append(f"  Forgetting:  {forgetting} (mean slope: {avg_slope:.2f})")
    description.append(f"  Paradigm intercepts: Free={intercept_free:.1f}, Cued={intercept_cued:.1f}, Recog={intercept_recog:.1f}")
    description.append(f"  Paradigm slopes:     Free={slope_free:.2f}, Cued={slope_cued:.2f}, Recog={slope_recog:.2f}")
    if paradigm_selective:
        description.append(f"  Pattern: Paradigm-selective ({dominant_paradigm} advantage of {intercept_range:.1f} points)")
    else:
        description.append(f"  Pattern: Uniform performance across paradigms")

    return label, "\n".join(description)

# ============================================================================
# Main Analysis
# ============================================================================
if __name__ == "__main__":
    try:
        log("="*80)
        log("STEP 06: Characterize Clusters")
        log("="*80)

        # ====================================================================
        # STEP 1: Load Data
        # ====================================================================
        log("\n[LOAD] Loading random effects and cluster assignments...")

        # Load original-scale random effects
        df_original = pd.read_csv(RQ_DIR / "data" / "step00_random_effects_wide.csv")
        log(f"[LOADED] Original-scale random effects: {df_original.shape}")

        # Load cluster assignments
        df_clusters = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        log(f"[LOADED] Cluster assignments: {df_clusters.shape}")

        # ====================================================================
        # STEP 2: Merge Data
        # ====================================================================
        log("\n[MERGE] Merging features with cluster labels...")

        df_merged = df_original.merge(df_clusters, on='UID', how='inner')
        log(f"[MERGED] Combined dataset: {df_merged.shape}")
        log(f"         {len(df_merged)} participants, {df_merged['cluster'].nunique()} clusters")

        # ====================================================================
        # STEP 3: Compute Cluster Summary Statistics
        # ====================================================================
        log("\n[COMPUTE] Computing summary statistics per cluster...")

        feature_cols = [col for col in df_original.columns if col != 'UID']
        n_clusters = df_merged['cluster'].nunique()
        log(f"[FEATURES] {len(feature_cols)} features to characterize")
        log(f"[CLUSTERS] {n_clusters} clusters to profile")

        # Initialize results list
        characterization_rows = []

        for cluster_id in sorted(df_merged['cluster'].unique()):
            cluster_data = df_merged[df_merged['cluster'] == cluster_id]
            n_members = len(cluster_data)

            log(f"\n[CLUSTER {cluster_id}] N={n_members} participants")

            for feature in feature_cols:
                values = cluster_data[feature].values

                # Compute statistics
                mean_val = values.mean()
                sd_val = values.std()
                min_val = values.min()
                max_val = values.max()

                characterization_rows.append({
                    'cluster': cluster_id,
                    'feature': feature,
                    'mean': mean_val,
                    'SD': sd_val,
                    'min': min_val,
                    'max': max_val,
                    'N': n_members
                })

                log(f"  {feature:30s}: mean={mean_val:6.2f}, SD={sd_val:5.2f}, range=[{min_val:6.2f}, {max_val:6.2f}]")

        # Create characterization DataFrame
        df_characterization = pd.DataFrame(characterization_rows)
        log(f"\n[DONE] Computed statistics for {len(df_characterization)} cluster-feature combinations")

        # ====================================================================
        # STEP 4: Save Characterization Table
        # ====================================================================
        log("\n[SAVE] Saving cluster characterization table...")

        char_path = RQ_DIR / "data" / "step06_cluster_characterization.csv"
        df_characterization.to_csv(char_path, index=False, encoding='utf-8')
        log(f"[SAVED] {char_path}")
        log(f"        {len(df_characterization)} rows (long format: {n_clusters} clusters x {len(feature_cols)} features)")

        # ====================================================================
        # STEP 5: Generate Cluster Profiles (Interpretive Labels)
        # ====================================================================
        log("\n[INTERPRET] Generating cluster profile descriptions...")

        profiles = []
        profiles.append("CLUSTER PROFILES")
        profiles.append("="*80)
        profiles.append(f"\nClustering: K={n_clusters}, N=100 participants")
        profiles.append(f"Features: Paradigm-specific intercepts (immediate performance) and slopes (forgetting rates)")
        profiles.append("")

        cluster_labels = {}
        for cluster_id in sorted(df_merged['cluster'].unique()):
            # Extract cluster data
            cluster_stats = df_characterization[df_characterization['cluster'] == cluster_id]

            # Generate interpretation
            label, description = interpret_cluster_pattern(cluster_id, cluster_stats)
            cluster_labels[cluster_id] = label

            profiles.append("-"*80)
            profiles.append(description)
            profiles.append("")

        # Add summary
        profiles.append("="*80)
        profiles.append("SUMMARY OF PROFILES:")
        profiles.append("-"*80)
        for cluster_id, label in cluster_labels.items():
            n_members = df_characterization[df_characterization['cluster'] == cluster_id]['N'].iloc[0]
            profiles.append(f"Cluster {cluster_id} (N={n_members:2d}): {label}")

        profiles_text = "\n".join(profiles)
        profiles_path = RQ_DIR / "data" / "step06_cluster_profiles.txt"
        with open(profiles_path, 'w', encoding='utf-8') as f:
            f.write(profiles_text)
        log(f"[SAVED] {profiles_path}")

        # ====================================================================
        # STEP 6: Validate Summary Statistics
        # ====================================================================
        log("\n[VALIDATION] Validating cluster summary statistics...")

        # Manual validation: check min <= mean <= max, SD >= 0, N > 0
        validation_passed = True
        validation_errors = []
        
        for idx, row in df_characterization.iterrows():
            cluster_id = row['cluster']
            feature = row['feature']
            min_val = row['min']
            mean_val = row['mean']
            max_val = row['max']
            sd_val = row['SD']
            n_val = row['N']
            
            # Check min <= mean <= max
            if not (min_val <= mean_val <= max_val):
                validation_errors.append(f"Cluster {cluster_id}, {feature}: min={min_val:.3f} mean={mean_val:.3f} max={max_val:.3f}")
                validation_passed = False
            
            # Check SD >= 0
            if sd_val < 0:
                validation_errors.append(f"Cluster {cluster_id}, {feature}: negative SD={sd_val:.3f}")
                validation_passed = False
            
            # Check N > 0
            if n_val <= 0:
                validation_errors.append(f"Cluster {cluster_id}, {feature}: invalid N={n_val}")
                validation_passed = False
        
        if validation_passed:
            log("[PASS] Cluster summary statistics validation successful")
            log(f"       All {len(df_characterization)} rows satisfy: min <= mean <= max, SD >= 0, N > 0")
        else:
            log("[FAIL] Validation errors:")
            for error in validation_errors:
                log(f"       {error}")
            raise ValueError(f"Found {len(validation_errors)} validation errors in cluster summary statistics")

        # ====================================================================
        # SUCCESS
        # ====================================================================
        log("\n" + "="*80)
        log("[SUCCESS] Step 06 complete")
        log("="*80)
        log("\nOutputs:")
        log(f"  - {char_path}")
        log(f"  - {profiles_path}")
        log("\nCluster Labels:")
        for cluster_id, label in cluster_labels.items():
            log(f"  Cluster {cluster_id}: {label}")
        log("\nNext: Step 07 (Prepare scatter matrix plot data)")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("\n[TRACEBACK]")
        import traceback
        traceback.print_exc(file=open(LOG_FILE, 'a'))
        traceback.print_exc()
        sys.exit(1)
