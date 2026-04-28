#!/usr/bin/env python3
"""test_k_clusters: Test K-means clustering for K=1 to K=10 clusters (EXTENDED from K=1-6), compute BIC"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from scipy.cluster.vq import kmeans2

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_test_k_clusters.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# K-means BIC Computation

def compute_bic_for_kmeans(X, K, n_init=50, random_state=42):
    """
    Fit K-means with K clusters and compute BIC.

    BIC formula: N * log(inertia/N) + K * log(N)
    where N = number of samples, inertia = within-cluster sum of squared distances

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (N x D)
    K : int
        Number of clusters
    n_init : int
        Number of initializations (best result kept)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    inertia : float
        Within-cluster sum of squared distances
    bic : float
        Bayesian Information Criterion value
    """
    N = X.shape[0]

    # Handle K=1 case (all points in one cluster)
    if K == 1:
        # Inertia = sum of squared distances from global mean
        centroid = X.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(X - centroid, axis=1)
        inertia = np.sum(distances ** 2)
        bic = N * np.log(inertia / N) + K * np.log(N)
        return inertia, bic

    # For K > 1, use scipy.cluster.vq.kmeans2
    # Run multiple initializations and keep best result (lowest inertia)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    for seed_offset in range(n_init):
        try:
            # kmeans2 uses different random seed each iteration
            np.random.seed(random_state + seed_offset)
            centroids, labels = kmeans2(X, K, minit='points', iter=300)

            # Compute inertia for this initialization
            inertia = 0.0
            for k in range(K):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centroid = centroids[k]
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    inertia += np.sum(distances ** 2)

            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        except Exception as e:
            # Handle convergence failures gracefully (continue with other inits)
            log(f"K-means K={K} seed={random_state + seed_offset} failed: {e}")
            continue

    if best_inertia == np.inf:
        raise ValueError(f"All {n_init} initializations failed for K={K}")

    # Compute BIC using best inertia
    inertia = best_inertia
    bic = N * np.log(inertia / N) + K * np.log(N)

    return inertia, bic

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Test K Clusters")
        # Load Input Data

        log("Loading standardized features from Step 1...")
        input_path = RQ_DIR / "data" / "step01_standardized_features.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file missing: {input_path}")

        df_features = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df_features)} rows, {len(df_features.columns)} cols)")

        # Validate required columns present
        required_cols = ['Intercept_z', 'Slope_z']
        if not all(col in df_features.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {df_features.columns.tolist()}")

        # Extract feature matrix (N x 2)
        X = df_features[required_cols].values
        N = X.shape[0]
        log(f"Feature matrix shape: {X.shape} (N={N} participants, D=2 features)")

        # Check for NaN values (not tolerated in clustering)
        if np.isnan(X).any():
            raise ValueError("NaN values found in feature matrix - cannot perform K-means")
        # Run K-means for K=1 to K=10 and Compute BIC

        log("Testing K-means for K=1 to K=10 (EXTENDED range)...")

        k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random_state = 42
        n_init = 50

        results = []

        for K in k_range:
            log(f"Fitting K={K} clusters (n_init={n_init})...")

            # Fit K-means and compute BIC
            inertia, bic = compute_bic_for_kmeans(
                X=X,
                K=K,
                n_init=n_init,
                random_state=random_state
            )

            log(f"K={K}: inertia={inertia:.4f}, BIC={bic:.4f}")

            # Record results
            results.append({
                'K': K,
                'inertia': inertia,
                'BIC': bic
            })

        # Create results DataFrame
        df_results = pd.DataFrame(results)
        log(f"K-means testing complete for {len(k_range)} K values")
        # Select Optimal K via BIC Minimum (with Elbow Fallback)
        # BIC interpretation: Lower BIC = better model
        # Optimal K: argmin(BIC) - K with lowest BIC value
        # FALLBACK: If BIC minimum is at boundary (K=10), use elbow method instead
        # Elbow method: Find K where second derivative of inertia is maximized

        log("Selecting optimal K via BIC minimum...")

        bic_optimal_idx = df_results['BIC'].idxmin()
        bic_optimal_k = int(df_results.loc[bic_optimal_idx, 'K'])
        bic_optimal_value = df_results.loc[bic_optimal_idx, 'BIC']

        log(f"BIC minimum at K={bic_optimal_k} (BIC = {bic_optimal_value:.4f})")

        # Check if BIC is at boundary (K=max tested)
        max_k_tested = max(k_range)
        if bic_optimal_k == max_k_tested:
            log(f"BIC minimum at boundary K={max_k_tested} - using elbow method instead")

            # Elbow method: Second derivative of inertia curve
            # Maximum second derivative indicates sharpest "bend" in the curve
            inertia_values = df_results['inertia'].values
            first_deriv = np.diff(inertia_values)  # Rate of decrease
            second_deriv = np.diff(first_deriv)    # Acceleration (positive = elbow)

            # Elbow at K where second derivative is maximum
            # second_deriv[i] corresponds to K=i+2 (since diff reduces length by 1 twice)
            elbow_idx = np.argmax(second_deriv)
            elbow_k = elbow_idx + 2  # Convert index to K value

            log(f"Second derivative analysis:")
            for i, d in enumerate(second_deriv):
                k_value = i + 2
                marker = " <-- ELBOW" if k_value == elbow_k else ""
                log(f"  K={k_value}: second_deriv={d:.4f}{marker}")

            optimal_k = elbow_k
            optimal_method = "elbow"
            log(f"K_optimal = {optimal_k} (via elbow method due to BIC boundary)")
        else:
            optimal_k = bic_optimal_k
            optimal_method = "BIC"
            log(f"K_optimal = {optimal_k} (via BIC minimum)")

        optimal_bic = df_results.loc[df_results['K'] == optimal_k, 'BIC'].values[0]

        # Log full BIC comparison table
        log("[BIC COMPARISON]")
        for _, row in df_results.iterrows():
            marker = " <-- OPTIMAL" if row['K'] == optimal_k else ""
            log(f"  K={int(row['K'])}: BIC={row['BIC']:.4f}{marker}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 (fit final K-means with optimal K)

        # Save cluster selection results (K, inertia, BIC for all 6 K values)
        output_path_results = RQ_DIR / "data" / "step02_cluster_selection.csv"
        log(f"Saving cluster selection results to {output_path_results.name}...")
        df_results.to_csv(output_path_results, index=False, encoding='utf-8')
        log(f"{output_path_results.name} ({len(df_results)} rows, {len(df_results.columns)} cols)")

        # Save optimal K to text file (single integer)
        output_path_k = RQ_DIR / "data" / "step02_optimal_k.txt"
        log(f"Saving optimal K to {output_path_k.name}...")
        with open(output_path_k, 'w', encoding='utf-8') as f:
            f.write(f"{optimal_k}\n")
        log(f"{output_path_k.name} (K_optimal = {optimal_k})")
        # Run Validation Tool
        # Validates: Inertia values >= 0 (positive)
        # Threshold: min_val=0.0, max_val=inf

        log("Running validate_numeric_range on inertia values...")

        validation_result = validate_numeric_range(
            data=df_results['inertia'].values,
            min_val=0.0,
            max_val=np.inf,
            column_name='inertia'
        )

        if not validation_result['valid']:
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log(f"Inertia range validation: {validation_result['message']}")

        # Additional validation checks (from recipe criteria)
        log("Checking inertia monotonicity...")
        inertia_values = df_results['inertia'].values
        is_monotonic = all(inertia_values[i] >= inertia_values[i+1] for i in range(len(inertia_values)-1))

        if not is_monotonic:
            log("Inertia not strictly monotonically decreasing")
            log(f"Inertia values: {inertia_values}")
        else:
            log("Inertia monotonically decreasing: PASS")

        log("Checking BIC finite values...")
        bic_values = df_results['BIC'].values
        if np.isnan(bic_values).any() or np.isinf(bic_values).any():
            raise ValueError("BIC values contain NaN or inf - validation failed")
        else:
            log("BIC finite values: PASS")

        log("Checking all 10 K values tested...")
        if len(df_results) != 10:
            raise ValueError(f"Expected 10 K values tested, found {len(df_results)}")
        if not all(df_results['K'].values == np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])):
            raise ValueError(f"K values not as expected. Found: {df_results['K'].values}")
        log("All 10 K values tested: PASS")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
