#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: 04
Step Name: Bootstrap Stability Validation
RQ: results/ch5/5.1.5
Generated: 2025-12-02

PURPOSE:
Bootstrap resampling (B=100) with Jaccard coefficient to assess cluster stability.
Tests whether the K=2 clustering solution is robust to sampling variation.
Jaccard >= 0.75 = Stable, 0.60-0.74 = Questionable, < 0.60 = Unstable.

EXPECTED INPUTS:
  - data/step01_standardized_features.csv (100 x 3: UID, Intercept_z, Slope_z)
  - data/step03_cluster_assignments.csv (100 x 2: UID, cluster)
  - data/step02_optimal_k.txt (K=2)

EXPECTED OUTPUTS:
  - data/step04_bootstrap_jaccard.csv (100 rows: iteration, jaccard)
  - data/step04_stability_summary.txt (mean Jaccard, 95% CI, classification)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tool
from tools.validation import validate_bootstrap_stability

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_bootstrap_stability.log"

RANDOM_STATE = 42
N_INIT = 50
N_BOOTSTRAP = 100

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Jaccard Computation with Cluster Relabeling
# =============================================================================

def compute_jaccard_with_relabeling(labels_original: np.ndarray, labels_bootstrap: np.ndarray,
                                    sample_indices: np.ndarray) -> float:
    """
    Compute Jaccard similarity between original and bootstrap clustering.

    Uses Hungarian algorithm-style matching to find optimal cluster label alignment.
    Only compares samples that appear in the bootstrap sample.

    Returns Jaccard coefficient (mean over classes for multi-class).
    """
    # Get original labels for bootstrap samples
    original_subset = labels_original[sample_indices]

    # Handle K=2 case with binary relabeling
    n_clusters_original = len(np.unique(labels_original))
    n_clusters_bootstrap = len(np.unique(labels_bootstrap))

    if n_clusters_original == 2 and n_clusters_bootstrap == 2:
        # Try both labelings and take max Jaccard
        jaccard_direct = jaccard_score(original_subset, labels_bootstrap, average='macro')
        labels_flipped = 1 - labels_bootstrap  # Flip 0<->1
        jaccard_flipped = jaccard_score(original_subset, labels_flipped, average='macro')
        return max(jaccard_direct, jaccard_flipped)
    else:
        # For K>2, use direct comparison (more complex alignment needed for production)
        return jaccard_score(original_subset, labels_bootstrap, average='macro')

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 04: Bootstrap Stability Validation")

        # =========================================================================
        # STEP 1: Load Input Data
        # =========================================================================

        log("[LOAD] Loading standardized features...")
        standardized_features = pd.read_csv(RQ_DIR / "data" / "step01_standardized_features.csv")
        X = standardized_features[['Intercept_z', 'Slope_z']].values
        log(f"[LOADED] Feature matrix shape: {X.shape}")

        log("[LOAD] Loading original cluster assignments...")
        cluster_assignments = pd.read_csv(RQ_DIR / "data" / "step03_cluster_assignments.csv")
        original_labels = cluster_assignments['cluster'].values
        log(f"[LOADED] Original cluster labels shape: {original_labels.shape}")

        log("[LOAD] Reading K_final from Step 2...")
        with open(RQ_DIR / "data" / "step02_optimal_k.txt", 'r') as f:
            K_final = int(f.read().strip())
        log(f"[INFO] K_final = {K_final}")

        N = len(X)
        log(f"[INFO] N = {N} participants")

        # =========================================================================
        # STEP 2: Bootstrap Resampling with Jaccard Computation
        # =========================================================================

        log(f"[ANALYSIS] Running {N_BOOTSTRAP} bootstrap iterations...")

        jaccard_coefficients = []

        for b in range(N_BOOTSTRAP):
            # Create bootstrap sample (resample with replacement)
            np.random.seed(RANDOM_STATE + b)
            sample_indices = np.random.choice(N, size=N, replace=True)
            X_bootstrap = X[sample_indices]

            # Fit K-means on bootstrap sample
            kmeans = KMeans(n_clusters=K_final, random_state=RANDOM_STATE + b, n_init=N_INIT)
            bootstrap_labels = kmeans.fit_predict(X_bootstrap)

            # Compute Jaccard coefficient with optimal relabeling
            jaccard = compute_jaccard_with_relabeling(original_labels, bootstrap_labels, sample_indices)
            jaccard_coefficients.append(jaccard)

            if (b + 1) % 20 == 0:
                log(f"[PROGRESS] Completed {b + 1}/{N_BOOTSTRAP} iterations (current Jaccard: {jaccard:.4f})")

        log(f"[DONE] Bootstrap resampling complete")

        # =========================================================================
        # STEP 3: Compute Summary Statistics
        # =========================================================================

        jaccard_array = np.array(jaccard_coefficients)
        mean_jaccard = jaccard_array.mean()
        ci_lower = np.percentile(jaccard_array, 2.5)
        ci_upper = np.percentile(jaccard_array, 97.5)

        log(f"[SUMMARY] Mean Jaccard: {mean_jaccard:.4f}")
        log(f"[SUMMARY] 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Classify stability
        if mean_jaccard >= 0.75:
            stability_class = "Stable"
            recommendation = "Proceed with confidence"
        elif mean_jaccard >= 0.60:
            stability_class = "Questionable"
            recommendation = "Report with caution"
        else:
            stability_class = "Unstable"
            recommendation = "Consider reducing K"

        log(f"[CLASSIFICATION] Stability: {stability_class} ({recommendation})")

        # =========================================================================
        # STEP 4: Save Outputs
        # =========================================================================

        log("[SAVE] Saving bootstrap Jaccard coefficients...")
        jaccard_df = pd.DataFrame({
            'iteration': list(range(1, N_BOOTSTRAP + 1)),
            'jaccard': jaccard_coefficients
        })
        jaccard_df.to_csv(RQ_DIR / "data" / "step04_bootstrap_jaccard.csv", index=False, encoding='utf-8')
        log(f"[SAVED] step04_bootstrap_jaccard.csv ({len(jaccard_df)} rows)")

        log("[SAVE] Saving stability summary...")
        summary = f"""Bootstrap Stability Validation (B={N_BOOTSTRAP} iterations)
{'=' * 60}

K_final: {K_final}
N participants: {N}

Summary Statistics:
  Mean Jaccard: {mean_jaccard:.4f}
  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]

Stability Classification: {stability_class}
Recommendation: {recommendation}

Threshold Reference:
  Jaccard >= 0.75: Stable (proceed with confidence)
  Jaccard 0.60-0.74: Questionable (report with caution)
  Jaccard < 0.60: Unstable (consider reducing K)
"""
        with open(RQ_DIR / "data" / "step04_stability_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        log(f"[SAVED] step04_stability_summary.txt")

        # =========================================================================
        # STEP 5: Run Validation Tool
        # =========================================================================

        log("[VALIDATION] Running validate_bootstrap_stability...")
        validation_result = validate_bootstrap_stability(
            stability_df=jaccard_df,
            min_jaccard_threshold=0.75,
            jaccard_col='jaccard'
        )

        if validation_result['valid']:
            log(f"[VALIDATION] PASS - {validation_result['message']}")
        else:
            log(f"[VALIDATION] WARNING - {validation_result['message']}")
            # Note: Don't raise error for stability warnings, just report

        log(f"[VALIDATION] Mean Jaccard: {validation_result['mean_jaccard']:.4f}")
        log(f"[VALIDATION] 95% CI: [{validation_result['ci_lower']:.4f}, {validation_result['ci_upper']:.4f}]")

        log(f"[SUCCESS] Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
