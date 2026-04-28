#!/usr/bin/env python3
"""
RQ 5.2.7 Step 02: K-Means Model Selection

Purpose: Test K=1 to K=6 cluster solutions using K-means algorithm,
         compute inertia and BIC for each K, select optimal K.

Input:
  - data/step01_standardized_features.csv (100 rows × 5 cols)

Output:
  - data/step02_cluster_selection.csv (6 rows: K=1-6 with inertia and BIC)
  - data/step02_optimal_k_selection.txt (selected K with justification)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Setup paths
SCRIPT_DIR = Path(__file__).parent
RQ_DIR = SCRIPT_DIR.parent
DATA_DIR = RQ_DIR / "data"

# Input file
INPUT_FILE = DATA_DIR / "step01_standardized_features.csv"

# Z-scored clustering variables
Z_VARS = [
    'Total_Intercept_What_z',
    'Total_Intercept_Where_z',
    'Total_Slope_What_z',
    'Total_Slope_Where_z'
]

# Model selection parameters
K_RANGE = range(1, 7)  # K=1, 2, 3, 4, 5, 6
N_INIT = 50  # Number of random initializations per K
RANDOM_STATE = 42
MAX_ITER = 300


def compute_bic(inertia: float, n_samples: int, n_features: int, k: int) -> float:
    """
    Compute Bayesian Information Criterion for K-means.

    BIC = n * log(inertia / n) + k * log(n) * p

    Where:
    - n = number of samples
    - p = number of features
    - k = number of clusters
    - inertia = within-cluster sum of squares
    """
    # Avoid log(0) if inertia is 0
    if inertia <= 0:
        return np.inf

    bic = n_samples * np.log(inertia / n_samples) + k * np.log(n_samples) * n_features
    return bic


def main():
    print("=" * 60)
    print("RQ 5.2.7 Step 02: K-Means Model Selection")
    print("=" * 60)

    # Load standardized features
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")

    # Extract clustering matrix (exclude UID column)
    X = df.values
    n_samples, n_features = X.shape
    print(f"Clustering matrix shape: {n_samples} samples × {n_features} features")

    # Test K=1 to K=6
    print("\nTesting K=1 to K=6 models...")
    print(f"  n_init={N_INIT}, random_state={RANDOM_STATE}, max_iter={MAX_ITER}")

    results = []
    for k in K_RANGE:
        print(f"\n  K={k}:", end=" ")

        # Fit K-means
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            algorithm='lloyd'
        )
        kmeans.fit(X)

        inertia = kmeans.inertia_
        bic = compute_bic(inertia, n_samples, n_features, k)

        results.append({
            'K': k,
            'inertia': inertia,
            'BIC': bic
        })

        print(f"inertia={inertia:.2f}, BIC={bic:.2f}")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Validate monotonicity of inertia (must decrease with K)
    print("\n\nValidation:")
    inertia_monotonic = all(
        df_results.loc[i, 'inertia'] >= df_results.loc[i+1, 'inertia']
        for i in range(len(df_results) - 1)
    )
    if inertia_monotonic:
        print("  - PASS: Inertia decreases monotonically with K")
    else:
        print("  - WARNING: Inertia does NOT decrease monotonically (unusual)")

    # Find optimal K
    min_bic_idx = df_results['BIC'].idxmin()
    optimal_k = df_results.loc[min_bic_idx, 'K']
    min_bic = df_results.loc[min_bic_idx, 'BIC']

    print(f"\nBIC minimum at K={optimal_k}, BIC={min_bic:.2f}")

    # Check parsimony rule: if multiple K have ΔBIC < 2, select smallest
    df_results['delta_BIC'] = df_results['BIC'] - min_bic
    near_optimal = df_results[df_results['delta_BIC'] < 2]

    if len(near_optimal) > 1:
        # Multiple K values have similar BIC - apply parsimony
        parsimonious_k = near_optimal['K'].min()
        print(f"\nParsimony rule applied: K={list(near_optimal['K'])} have ΔBIC < 2")
        print(f"Selecting K={parsimonious_k} (most parsimonious)")
        optimal_k = parsimonious_k
    else:
        print(f"\nClear BIC minimum at K={optimal_k}")

    # Print BIC comparison table
    print("\nBIC Comparison Table:")
    print("-" * 40)
    print(f"{'K':>4} {'Inertia':>12} {'BIC':>12} {'ΔBIC':>10}")
    print("-" * 40)
    for _, row in df_results.iterrows():
        marker = " ***" if row['K'] == optimal_k else ""
        print(f"{row['K']:>4} {row['inertia']:>12.2f} {row['BIC']:>12.2f} {row['delta_BIC']:>10.2f}{marker}")
    print("-" * 40)

    # Save cluster selection results
    output_file = DATA_DIR / "step02_cluster_selection.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Save optimal K selection report
    report_file = DATA_DIR / "step02_optimal_k_selection.txt"
    with open(report_file, 'w') as f:
        f.write("RQ 5.2.7 Step 02: Optimal K Selection\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Optimal K = {optimal_k}\n\n")

        if len(near_optimal) > 1:
            f.write(f"Justification: Parsimony rule applied\n")
            f.write(f"  K values with ΔBIC < 2: {list(near_optimal['K'].values)}\n")
            f.write(f"  Selected K={optimal_k} as most parsimonious model\n\n")
        else:
            f.write(f"Justification: Clear BIC minimum\n")
            f.write(f"  K={optimal_k} has minimum BIC = {min_bic:.2f}\n")
            f.write(f"  Next best K has ΔBIC = {df_results[df_results['K'] != optimal_k]['delta_BIC'].min():.2f}\n\n")

        f.write("BIC Comparison Table:\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'K':>4} {'Inertia':>12} {'BIC':>12} {'ΔBIC':>10}\n")
        f.write("-" * 45 + "\n")
        for _, row in df_results.iterrows():
            marker = " <-- optimal" if row['K'] == optimal_k else ""
            f.write(f"{row['K']:>4} {row['inertia']:>12.2f} {row['BIC']:>12.2f} {row['delta_BIC']:>10.2f}{marker}\n")
        f.write("-" * 45 + "\n")

        f.write(f"\nModel Selection Parameters:\n")
        f.write(f"  - n_samples: {n_samples}\n")
        f.write(f"  - n_features: {n_features}\n")
        f.write(f"  - n_init: {N_INIT}\n")
        f.write(f"  - random_state: {RANDOM_STATE}\n")

    print(f"Saved: {report_file}")

    print("\n" + "=" * 60)
    print(f"Step 02 COMPLETE: Optimal K selected: K={optimal_k}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
