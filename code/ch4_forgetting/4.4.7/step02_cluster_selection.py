#!/usr/bin/env python3
"""
Step 02: Cluster Model Selection (K=1 to K=6 via BIC)

Test K=1 to K=6 cluster solutions, compute BIC for each,
select optimal K as BIC minimum.

Input: data/step01_standardized_features.csv (100 rows × 7 columns, z-scored)
Output:
  - data/step02_cluster_selection.csv (6 rows: K, inertia, BIC)
  - data/step02_optimal_k.txt (single integer: optimal K)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_cluster_assignment

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = RQ_DIR / "data/step01_standardized_features.csv"
OUTPUT_BIC_FILE = RQ_DIR / "data/step02_cluster_selection.csv"
OUTPUT_K_FILE = RQ_DIR / "data/step02_optimal_k.txt"
LOG_FILE = RQ_DIR / "logs/step02_cluster_selection.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 02: Cluster Model Selection (BIC)")

        # Load standardized features
        log(f"Reading {INPUT_FILE}")
        df_standardized = pd.read_csv(INPUT_FILE)
        log(f"{len(df_standardized)} rows, {len(df_standardized.columns)} columns")

        # Extract z-scored feature columns
        feature_cols = [
            'Common_Intercept_z', 'Common_Slope_z',
            'Congruent_Intercept_z', 'Congruent_Slope_z',
            'Incongruent_Intercept_z', 'Incongruent_Slope_z'
        ]

        X = df_standardized[feature_cols].values
        N = X.shape[0]
        log(f"Clustering on {N} samples, {X.shape[1]} features")

        # Test K=1 to K=6
        K_range = range(1, 7)
        results = []

        log("Testing K=1 to K=6")
        for K in K_range:
            log(f"  Fitting K={K}...")
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=50)
            kmeans.fit(X)

            inertia = kmeans.inertia_

            # BIC formula: N * log(inertia / N) + K * log(N)
            # Note: For K=1, inertia is total variance
            bic = N * np.log(inertia / N) + K * np.log(N)

            results.append({
                'K': K,
                'inertia': inertia,
                'BIC': bic
            })
            log(f"    K={K}: inertia={inertia:.4f}, BIC={bic:.4f}")

        # Create DataFrame
        df_bic = pd.DataFrame(results)

        # Find optimal K (minimum BIC)
        optimal_idx = df_bic['BIC'].idxmin()
        optimal_k = int(df_bic.loc[optimal_idx, 'K'])
        log(f"Optimal K = {optimal_k} (BIC = {df_bic.loc[optimal_idx, 'BIC']:.4f})")

        # Verify inertia is monotonically decreasing
        inertia_diffs = np.diff(df_bic['inertia'].values)
        if not np.all(inertia_diffs <= 0):
            log("Inertia not monotonically decreasing (unexpected!)")

        # Save BIC table
        log(f"Writing BIC table to {OUTPUT_BIC_FILE}")
        df_bic.to_csv(OUTPUT_BIC_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_BIC_FILE}")

        # Save optimal K
        log(f"Writing optimal K to {OUTPUT_K_FILE}")
        with open(OUTPUT_K_FILE, 'w', encoding='utf-8') as f:
            f.write(str(optimal_k))
        log(f"{OUTPUT_K_FILE}")

        # Validation: Check BIC selection quality
        log("Validating BIC model selection")

        # Check all K values present
        if len(df_bic) != 6:
            raise ValueError(f"Expected 6 rows (K=1-6), got {len(df_bic)}")

        # Check BIC has minimum within range (not at boundary)
        if optimal_k == 1 or optimal_k == 6:
            log("Optimal K at boundary (K=1 or K=6) - may need wider search range")

        # Check no NaN values
        nan_count = df_bic.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in BIC table")

        log("[VALIDATION PASS] BIC model selection valid")
        log(f"Step 02 complete - Optimal K = {optimal_k}")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
