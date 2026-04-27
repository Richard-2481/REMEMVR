#!/usr/bin/env python3
"""
RQ 5.5.7 Step 02: K-Means Model Selection (K=1 to K=6)

Purpose: Test K=1 to K=6 using K-means clustering, compute inertia and BIC
for each K, select optimal K as BIC minimum (or K-1 if BIC minimum at K=6).

Input:
- data/step01_standardized_features.csv (100 rows, z-scored)

Output:
- data/step02_cluster_selection.csv (6 rows: K, inertia, BIC)
- data/step02_optimal_k.txt (optimal K with justification)

BIC Formula: BIC = inertia + K * log(N) * D
where N=100 (sample size), D=4 (features)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "step02_cluster_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Perform K-means model selection using BIC."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.7 Step 02: K-Means Model Selection (K=1 to K=6)")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load input data
    # -------------------------------------------------------------------------
    input_path = DATA_DIR / "step01_standardized_features.csv"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run Step 01 first")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from Step 01")

    # Extract feature matrix (exclude UID)
    feature_cols = ['Source_intercept', 'Source_slope',
                    'Destination_intercept', 'Destination_slope']
    X = df[feature_cols].values

    N = X.shape[0]  # 100 samples
    D = X.shape[1]  # 4 features
    logger.info(f"Feature matrix: {N} samples x {D} features")

    # -------------------------------------------------------------------------
    # 2. Fit K-means for K=1 to K=6
    # -------------------------------------------------------------------------
    logger.info("Fitting K-means for K=1 to K=6...")

    k_range = [1, 2, 3, 4, 5, 6]
    results = []

    for K in k_range:
        # Fit K-means
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=50)
        kmeans.fit(X)

        # Extract inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_

        # Compute BIC = inertia + K * log(N) * D
        # This is a simplified BIC for clustering
        bic = inertia + K * np.log(N) * D

        results.append({
            'K': K,
            'inertia': inertia,
            'BIC': bic
        })

        logger.info(f"  K={K}: inertia={inertia:.2f}, BIC={bic:.2f}")

    # -------------------------------------------------------------------------
    # 3. Create results DataFrame
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 4. Validate results
    # -------------------------------------------------------------------------
    logger.info("\nValidating results...")

    # Check inertia decreases monotonically
    inertia_values = df_results['inertia'].values
    is_monotonic = all(inertia_values[i] >= inertia_values[i+1]
                       for i in range(len(inertia_values)-1))

    if is_monotonic:
        logger.info("Inertia decreases monotonically: PASS")
    else:
        logger.warning("Inertia not strictly monotonically decreasing")

    # Check all values positive
    if (df_results['inertia'] >= 0).all() and (df_results['BIC'] >= 0).all():
        logger.info("All inertia and BIC values non-negative: PASS")
    else:
        logger.error("Found negative inertia or BIC values")
        sys.exit(1)

    # Check K values consecutive
    if list(df_results['K']) == k_range:
        logger.info("K values consecutive {1,2,3,4,5,6}: PASS")
    else:
        logger.error("K values not consecutive")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 5. Select optimal K
    # -------------------------------------------------------------------------
    bic_values = df_results['BIC'].values
    bic_min_idx = np.argmin(bic_values)
    bic_min_k = df_results.iloc[bic_min_idx]['K']
    bic_min_value = df_results.iloc[bic_min_idx]['BIC']

    logger.info(f"\nBIC minimum at K={bic_min_k} (BIC={bic_min_value:.2f})")

    # If BIC minimum at K=6, select K=5 (avoid boundary)
    if bic_min_k == 6:
        optimal_k = 5
        justification = (f"BIC minimum at boundary K=6, selected K=5 to avoid boundary. "
                        f"BIC at K=5: {df_results[df_results['K']==5]['BIC'].values[0]:.2f}, "
                        f"BIC at K=6: {bic_min_value:.2f}")
        logger.warning("BIC minimum at boundary (K=6), selecting K=5")
    else:
        optimal_k = int(bic_min_k)
        justification = f"K={optimal_k} selected as BIC minimum (BIC={bic_min_value:.2f})"

    logger.info(f"Optimal K selected: {optimal_k}")
    logger.info(f"Justification: {justification}")

    # -------------------------------------------------------------------------
    # 6. Save outputs
    # -------------------------------------------------------------------------
    # Save cluster selection CSV
    csv_path = DATA_DIR / "step02_cluster_selection.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved cluster selection to {csv_path}")

    # Save optimal K text file
    txt_path = DATA_DIR / "step02_optimal_k.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Optimal K: {optimal_k}\n")
        f.write(f"Justification: {justification}\n\n")
        f.write("BIC values for all K:\n")
        for _, row in df_results.iterrows():
            f.write(f"  K={int(row['K'])}: BIC={row['BIC']:.2f}\n")

    logger.info(f"Saved optimal K to {txt_path}")

    # -------------------------------------------------------------------------
    # 7. Print BIC summary table
    # -------------------------------------------------------------------------
    logger.info("\nBIC Summary Table:")
    logger.info("-" * 40)
    logger.info(f"{'K':<5} {'Inertia':<12} {'BIC':<12} {'Selected'}")
    logger.info("-" * 40)
    for _, row in df_results.iterrows():
        K = int(row['K'])
        selected = "*" if K == optimal_k else ""
        logger.info(f"{K:<5} {row['inertia']:<12.2f} {row['BIC']:<12.2f} {selected}")
    logger.info("-" * 40)

    # -------------------------------------------------------------------------
    # 8. Final summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Step 02 COMPLETE")
    logger.info(f"  Tested K=1 to K=6")
    logger.info(f"  BIC minimum at K={bic_min_k}")
    logger.info(f"  Selected optimal K={optimal_k}")
    logger.info(f"  Output: {csv_path}")
    logger.info(f"  Output: {txt_path}")
    logger.info("=" * 60)

    return optimal_k

if __name__ == "__main__":
    main()
