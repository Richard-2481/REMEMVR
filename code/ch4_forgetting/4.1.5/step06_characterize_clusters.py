#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: 06
Step Name: Characterize Clusters
RQ: results/ch5/5.1.5
Generated: 2025-12-02

PURPOSE:
Characterize clusters by computing summary statistics (mean, SD) in raw scale
and assigning interpretive labels based on profile patterns.

EXPECTED INPUTS:
  - data/step00_random_effects_from_rq514.csv (100 x 3: UID, Total_Intercept, Total_Slope)
  - data/step03_cluster_assignments.csv (100 x 2: UID, cluster)

EXPECTED OUTPUTS:
  - data/step06_cluster_characterization.csv (K rows: cluster, N, mean_intercept, sd_intercept, etc.)
  - data/step06_cluster_labels.txt (interpretive labels per cluster)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tool
from tools.validation import validate_cluster_summary_stats

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_characterize_clusters.log"

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 06: Characterize Clusters")

        # =========================================================================
        # STEP 1: Load Input Data
        # =========================================================================

        log("[LOAD] Loading raw-scale random effects from Step 0...")
        random_effects = pd.read_csv(RQ_DIR / "data" / "step00_random_effects_from_rq514.csv")
        log(f"[LOADED] random_effects ({len(random_effects)} rows)")
        log(f"[INFO] Columns: {random_effects.columns.tolist()}")

        log("[LOAD] Loading cluster assignments from Step 3...")
        cluster_assignments = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        log(f"[LOADED] cluster_assignments ({len(cluster_assignments)} rows)")

        # =========================================================================
        # STEP 2: Merge Data
        # =========================================================================

        log("[MERGE] Merging random effects with cluster assignments on UID...")
        merged = pd.merge(random_effects, cluster_assignments, on='UID')
        log(f"[MERGED] {len(merged)} participants with cluster + raw intercept/slope")

        # =========================================================================
        # STEP 3: Compute Cluster Summary Statistics
        # =========================================================================

        log("[ANALYSIS] Computing cluster summary statistics...")

        # Group by cluster and compute statistics
        summary_data = []

        for cluster_id in sorted(merged['cluster'].unique()):
            cluster_data = merged[merged['cluster'] == cluster_id]
            n = len(cluster_data)

            # Intercept statistics (raw scale)
            intercept_min = cluster_data['Total_Intercept'].min()
            intercept_mean = cluster_data['Total_Intercept'].mean()
            intercept_max = cluster_data['Total_Intercept'].max()
            intercept_sd = cluster_data['Total_Intercept'].std()

            # Slope statistics (raw scale)
            slope_min = cluster_data['Total_Slope'].min()
            slope_mean = cluster_data['Total_Slope'].mean()
            slope_max = cluster_data['Total_Slope'].max()
            slope_sd = cluster_data['Total_Slope'].std()

            log(f"[CLUSTER {cluster_id}] N={n}")
            log(f"  Intercept: mean={intercept_mean:.4f}, SD={intercept_sd:.4f}, range=[{intercept_min:.4f}, {intercept_max:.4f}]")
            log(f"  Slope: mean={slope_mean:.4f}, SD={slope_sd:.4f}, range=[{slope_min:.4f}, {slope_max:.4f}]")

            summary_data.append({
                'cluster': cluster_id,
                'N': n,
                'intercept_min': intercept_min,
                'intercept_mean': intercept_mean,
                'intercept_max': intercept_max,
                'intercept_SD': intercept_sd,
                'slope_min': slope_min,
                'slope_mean': slope_mean,
                'slope_max': slope_max,
                'slope_SD': slope_sd
            })

        summary_df = pd.DataFrame(summary_data)

        # =========================================================================
        # STEP 4: Assign Interpretive Labels
        # =========================================================================

        log("[ANALYSIS] Assigning interpretive labels based on profile patterns...")

        # Compute overall medians for comparison
        overall_intercept_median = merged['Total_Intercept'].median()
        overall_slope_median = merged['Total_Slope'].median()

        log(f"[INFO] Overall intercept median: {overall_intercept_median:.4f}")
        log(f"[INFO] Overall slope median: {overall_slope_median:.4f}")

        labels = []
        label_descriptions = []

        for _, row in summary_df.iterrows():
            cluster_id = row['cluster']
            intercept_mean = row['intercept_mean']
            slope_mean = row['slope_mean']

            # Classify based on intercept (baseline performance)
            if intercept_mean > overall_intercept_median:
                baseline_desc = "High baseline"
            else:
                baseline_desc = "Low baseline"

            # Classify based on slope (rate of change)
            # Note: Positive slopes in raw scale indicate increasing theta over time
            # Higher slope = faster improvement (or slower forgetting)
            if slope_mean > overall_slope_median:
                slope_desc = "faster change"
            else:
                slope_desc = "slower change"

            label = f"{baseline_desc}, {slope_desc}"
            labels.append(label)

            description = f"Cluster {cluster_id}: {label}\n"
            description += f"  N participants: {int(row['N'])}\n"
            description += f"  Mean baseline (intercept): {intercept_mean:.4f}\n"
            description += f"  Mean rate of change (slope): {slope_mean:.4f}\n"
            description += f"  Intercept SD: {row['intercept_SD']:.4f}\n"
            description += f"  Slope SD: {row['slope_SD']:.4f}\n"
            label_descriptions.append(description)

            log(f"[LABEL] Cluster {cluster_id}: {label}")

        summary_df['label'] = labels

        # =========================================================================
        # STEP 5: Save Outputs
        # =========================================================================

        log("[SAVE] Saving cluster characterization...")
        summary_df.to_csv(RQ_DIR / "data" / "step06_cluster_characterization.csv", index=False, encoding='utf-8')
        log(f"[SAVED] step06_cluster_characterization.csv ({len(summary_df)} rows)")

        log("[SAVE] Saving cluster labels...")
        labels_text = "Cluster Characterization - Interpretive Labels\n"
        labels_text += "=" * 60 + "\n\n"
        for desc in label_descriptions:
            labels_text += desc + "\n"

        labels_text += "\nInterpretation Guide:\n"
        labels_text += "-" * 40 + "\n"
        labels_text += "Intercept: Baseline ability level (IRT theta scale)\n"
        labels_text += "  - Positive = above average baseline\n"
        labels_text += "  - Negative = below average baseline\n"
        labels_text += "\n"
        labels_text += "Slope: Rate of change over time (theta per day)\n"
        labels_text += "  - Higher slope = faster increase (or slower forgetting)\n"
        labels_text += "  - Lower slope = slower increase (or faster forgetting)\n"

        with open(RQ_DIR / "data" / "step06_cluster_labels.txt", 'w', encoding='utf-8') as f:
            f.write(labels_text)
        log(f"[SAVED] step06_cluster_labels.txt")

        # =========================================================================
        # STEP 6: Run Validation Tool
        # =========================================================================

        log("[VALIDATION] Running validate_cluster_summary_stats...")

        # Validate that sum of N = 100
        total_n = summary_df['N'].sum()
        if total_n != 100:
            raise ValueError(f"Sum of N across clusters ({total_n}) != 100")
        log(f"[VALIDATION] Sum of N = {total_n} (expected 100): PASS")

        # Run the validation function (uses auto-detection for column patterns)
        validation_result = validate_cluster_summary_stats(summary_df)

        if validation_result['valid']:
            log(f"[VALIDATION] PASS - {validation_result['message']}")
        else:
            log(f"[VALIDATION] FAIL - {validation_result['message']}")
            raise ValueError(f"Cluster summary validation failed: {validation_result['message']}")

        log(f"[SUCCESS] Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
