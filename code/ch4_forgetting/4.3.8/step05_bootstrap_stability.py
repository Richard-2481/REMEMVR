#!/usr/bin/env python3
"""
================================================================================
STEP 05: Bootstrap Stability Assessment
================================================================================
RQ: ch5/5.3.8 (Paradigm-Based Clustering)
Purpose: Assess cluster stability via bootstrap resampling (100 iterations)
         with Jaccard index comparison to original clustering

Inputs:
  - data/step01_standardized_features.csv (100 x 6 z-scores)
  - data/step03_cluster_assignments.csv (original cluster labels)
  - data/step02_optimal_k.txt (K value)

Outputs:
  - data/step05_bootstrap_stability.csv (100 iterations x Jaccard values)
  - data/step05_stability_summary.txt

Bootstrap Protocol:
  - 100 iterations
  - 80% subsample (N=80) without replacement per iteration
  - Fit K-means on subsample with same K
  - Compute Jaccard index (overlap with original clustering)
  - Target: Mean Jaccard >= 0.75 for stable clustering

Note: Mean Jaccard < 0.75 triggers WARNING (not error) - clustering is
      exploratory and tentative per concept.md
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_bootstrap_stability

# ============================================================================
# Configuration
# ============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_bootstrap_stability.log"

# Bootstrap parameters
N_ITERATIONS = 100
SUBSAMPLE_SIZE = 80  # 80% of 100 participants
RANDOM_STATE_BASE = 42
JACCARD_THRESHOLD = 0.75

# ============================================================================
# Logging
# ============================================================================
def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# ============================================================================
# Jaccard Index Computation
# ============================================================================
def compute_jaccard_for_subsample(original_labels, subsample_labels, subsample_indices):
    """
    Compute Jaccard index between original and subsample cluster assignments.

    Jaccard = (# pairs in same cluster in both) / (# pairs in same cluster in either)

    Args:
        original_labels: Original cluster labels (full sample)
        subsample_labels: Bootstrap cluster labels (subsample)
        subsample_indices: Indices of subsampled participants

    Returns:
        float: Jaccard index [0, 1]
    """
    # Extract original labels for subsampled participants
    original_sub = original_labels[subsample_indices]

    # Use sklearn jaccard_score (treating each pair as a sample)
    # Need to convert cluster labels to binary matrix (same cluster = 1, different = 0)

    # Alternative approach: manually compute Jaccard from pairwise same-cluster relationships
    n = len(subsample_indices)
    same_in_original = 0
    same_in_bootstrap = 0
    same_in_both = 0

    for i in range(n):
        for j in range(i+1, n):
            orig_same = (original_sub[i] == original_sub[j])
            boot_same = (subsample_labels[i] == subsample_labels[j])

            if orig_same:
                same_in_original += 1
            if boot_same:
                same_in_bootstrap += 1
            if orig_same and boot_same:
                same_in_both += 1

    # Jaccard = intersection / union
    union = same_in_original + same_in_bootstrap - same_in_both
    if union == 0:
        return 1.0  # Perfect agreement (no pairs in same cluster in either)

    jaccard = same_in_both / union
    return jaccard

# ============================================================================
# Main Analysis
# ============================================================================
if __name__ == "__main__":
    try:
        log("="*80)
        log("STEP 05: Bootstrap Stability Assessment")
        log("="*80)

        # ====================================================================
        # STEP 1: Load Data
        # ====================================================================
        log("\n[LOAD] Loading standardized features and cluster assignments...")

        # Load standardized features
        df_features = pd.read_csv(RQ_DIR / "data" / "step01_standardized_features.csv")
        log(f"[LOADED] Standardized features: {df_features.shape}")

        # Load original cluster assignments
        df_clusters = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        log(f"[LOADED] Original cluster assignments: {df_clusters.shape}")

        # Read optimal K from file
        optimal_k_path = RQ_DIR / "data" / "step02_optimal_k.txt"
        with open(optimal_k_path, 'r') as f:
            text = f.read()
            # Parse "OPTIMAL K: 3" from last line
            optimal_k = int(text.strip().split('\n')[-1].split(':')[-1].strip())
        log(f"[LOADED] Optimal K: {optimal_k}")

        # Extract feature matrix
        feature_cols = [col for col in df_features.columns if col.endswith('_z')]
        X = df_features[feature_cols].values
        log(f"[EXTRACT] Feature matrix: {X.shape}")

        # Extract original labels
        original_labels = df_clusters['cluster'].values
        log(f"[EXTRACT] Original labels: {original_labels.shape}")

        # ====================================================================
        # STEP 2: Bootstrap Resampling
        # ====================================================================
        log(f"\n[BOOTSTRAP] Running {N_ITERATIONS} bootstrap iterations...")
        log(f"            Subsample size: {SUBSAMPLE_SIZE} participants (80%)")

        jaccard_values = []
        np.random.seed(RANDOM_STATE_BASE)

        for i in range(N_ITERATIONS):
            # Sample 80% of participants without replacement
            subsample_indices = np.random.choice(
                len(X),
                size=SUBSAMPLE_SIZE,
                replace=False
            )
            X_subsample = X[subsample_indices]

            # Fit K-means on subsample
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=RANDOM_STATE_BASE + i,
                n_init=50
            )
            subsample_labels = kmeans.fit_predict(X_subsample)

            # Compute Jaccard index
            jaccard = compute_jaccard_for_subsample(
                original_labels,
                subsample_labels,
                subsample_indices
            )
            jaccard_values.append(jaccard)

            # Progress logging (every 10 iterations)
            if (i + 1) % 10 == 0:
                log(f"[ITERATION {i+1:3d}] Jaccard: {jaccard:.4f}")

        log(f"[DONE] Completed {N_ITERATIONS} bootstrap iterations")

        # ====================================================================
        # STEP 3: Compute Summary Statistics
        # ====================================================================
        log("\n[SUMMARY] Computing bootstrap statistics...")

        jaccard_array = np.array(jaccard_values)
        mean_jaccard = jaccard_array.mean()
        std_jaccard = jaccard_array.std()
        ci_lower = np.percentile(jaccard_array, 2.5)
        ci_upper = np.percentile(jaccard_array, 97.5)

        log(f"[STAT] Mean Jaccard: {mean_jaccard:.4f}")
        log(f"[STAT] SD Jaccard:   {std_jaccard:.4f}")
        log(f"[STAT] 95% CI:       [{ci_lower:.4f}, {ci_upper:.4f}]")
        log(f"[STAT] Range:        [{jaccard_array.min():.4f}, {jaccard_array.max():.4f}]")

        # Check threshold
        stability_pass = mean_jaccard >= JACCARD_THRESHOLD
        log(f"\n[THRESHOLD] Target: Mean Jaccard >= {JACCARD_THRESHOLD}")
        log(f"            Status: {'PASS' if stability_pass else 'WARNING (below threshold)'}")

        # ====================================================================
        # STEP 4: Save Bootstrap Results
        # ====================================================================
        log("\n[SAVE] Saving bootstrap results...")

        bootstrap_df = pd.DataFrame({
            'iteration': range(1, N_ITERATIONS + 1),
            'jaccard': jaccard_values
        })

        bootstrap_path = RQ_DIR / "data" / "step05_bootstrap_stability.csv"
        bootstrap_df.to_csv(bootstrap_path, index=False, encoding='utf-8')
        log(f"[SAVED] {bootstrap_path}")
        log(f"        {len(bootstrap_df)} bootstrap iterations recorded")

        # ====================================================================
        # STEP 5: Generate Stability Summary Report
        # ====================================================================
        log("\n[REPORT] Generating stability summary...")

        summary = []
        summary.append("BOOTSTRAP STABILITY ASSESSMENT")
        summary.append("="*80)
        summary.append(f"\nClustering Configuration: K={optimal_k} clusters, N=100 participants")
        summary.append(f"Bootstrap Protocol: {N_ITERATIONS} iterations, {SUBSAMPLE_SIZE} participants per subsample (80%)")
        summary.append("")

        summary.append("JACCARD INDEX DISTRIBUTION:")
        summary.append("-"*80)
        summary.append(f"Mean:        {mean_jaccard:.4f}")
        summary.append(f"SD:          {std_jaccard:.4f}")
        summary.append(f"95% CI:      [{ci_lower:.4f}, {ci_upper:.4f}]")
        summary.append(f"Range:       [{jaccard_array.min():.4f}, {jaccard_array.max():.4f}]")
        summary.append("")

        summary.append("INTERPRETATION:")
        summary.append("-"*80)
        if mean_jaccard >= 0.85:
            stability_interp = "EXCELLENT - Clustering is highly stable"
        elif mean_jaccard >= 0.75:
            stability_interp = "GOOD - Clustering is acceptably stable"
        elif mean_jaccard >= 0.65:
            stability_interp = "MARGINAL - Clustering shows moderate stability"
        else:
            stability_interp = "POOR - Clustering is unstable (sensitive to sampling)"

        summary.append(f"Stability: {stability_interp}")
        summary.append("")

        if not stability_pass:
            summary.append("WARNING:")
            summary.append("-"*80)
            summary.append(f"Mean Jaccard {mean_jaccard:.4f} is below threshold {JACCARD_THRESHOLD}")
            summary.append("This suggests cluster assignments are sensitive to sample composition.")
            summary.append("Phenotypes should be interpreted as tentative and exploratory.")
            summary.append("")
            summary.append("Possible Reasons:")
            summary.append("  - Small sample size (N=100) limits stability")
            summary.append("  - Clusters may overlap in feature space")
            summary.append("  - Some participants near cluster boundaries")
            summary.append("")
            summary.append("Recommendation:")
            summary.append("  - Report clusters as exploratory phenotypes")
            summary.append("  - Cross-validate with external outcomes if available")
            summary.append("  - Consider alternative K values for robustness check")
        else:
            summary.append("CONCLUSION:")
            summary.append("-"*80)
            summary.append("Bootstrap stability meets threshold. Cluster assignments are robust")
            summary.append("to resampling, supporting validity of identified phenotypes for")
            summary.append("exploratory analysis and hypothesis generation.")

        summary_text = "\n".join(summary)
        summary_path = RQ_DIR / "data" / "step05_stability_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        log(f"[SAVED] {summary_path}")

        # ====================================================================
        # STEP 6: Validate Bootstrap Results
        # ====================================================================
        log("\n[VALIDATION] Validating bootstrap stability...")

        validation_result = validate_bootstrap_stability(
            stability_df=bootstrap_df,
            min_jaccard_threshold=JACCARD_THRESHOLD
        )


        # Check validation result
        # Note: valid=False could mean out-of-bounds (error) OR below-threshold (warning)
        import math
        if math.isnan(validation_result['mean_jaccard']):
            # NaN means out-of-bounds (TRUE ERROR)
            log(f"[FAIL] Validation error: {validation_result['message']}")
            raise ValueError(validation_result['message'])
        else:
            # Mean computed successfully - check threshold
            log(f"[PASS] Bootstrap stability computed: Mean Jaccard = {validation_result['mean_jaccard']:.4f}")
            log(f"       95% CI: [{validation_result['ci_lower']:.4f}, {validation_result['ci_upper']:.4f}]")
            
            if validation_result['valid']:
                log(f"       Status: PASS (above threshold {JACCARD_THRESHOLD})")
            else:
                log(f"       Status: WARNING (below threshold {JACCARD_THRESHOLD})")
                log("\n[WARNING] Bootstrap stability below threshold - tentative clustering")
                log("          This is NOT an error - clustering is exploratory")

        # SUCCESS
        # ====================================================================
        log("\n" + "="*80)
        log("[SUCCESS] Step 05 complete")
        log("="*80)
        log("\nOutputs:")
        log(f"  - {bootstrap_path}")
        log(f"  - {summary_path}")
        log("\nNext: Step 06 (Characterize cluster profiles)")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("\n[TRACEBACK]")
        import traceback
        traceback.print_exc(file=open(LOG_FILE, 'a'))
        traceback.print_exc()
        sys.exit(1)
