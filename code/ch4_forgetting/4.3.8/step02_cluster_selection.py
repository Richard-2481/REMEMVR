#!/usr/bin/env python3
"""Cluster Selection (K=1-6): Fit K-means models for K=1-6, compute BIC for each, select optimal K via BIC"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.8
LOG_FILE = RQ_DIR / "logs" / "step02_cluster_selection.log"

# Input from step01
INPUT_FILE = RQ_DIR / "data" / "step01_standardized_features.csv"

# Outputs
OUTPUT_FILE = RQ_DIR / "data" / "step02_cluster_selection.csv"
OPTIMAL_K_FILE = RQ_DIR / "data" / "step02_optimal_k.txt"
ELBOW_DATA_FILE = RQ_DIR / "data" / "step02_elbow_plot_data.csv"

# K-means parameters
K_RANGE = range(1, 7)  # K=1 to K=6
RANDOM_STATE = 42
N_INIT = 50

# Feature columns (exclude UID)
FEATURE_COLS = [
    'Total_Intercept_Cued_z',
    'Total_Intercept_Free_z',
    'Total_Intercept_Recognition_z',
    'Total_Slope_Cued_z',
    'Total_Slope_Free_z',
    'Total_Slope_Recognition_z'
]

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Cluster Selection (K=1-6)")
        # Load Standardized Features

        log(f"Loading standardized features...")
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        log(f"{len(df)} rows, {len(df.columns)} columns")

        # Verify feature columns exist
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Extract feature matrix (exclude UID)
        X = df.values
        N = X.shape[0]  # Number of participants (100)
        p = X.shape[1]  # Number of features (6)

        log(f"Feature matrix: N={N} participants, p={p} features")
        # Fit K-Means for K=1 to K=6, Compute BIC
        # Formula: BIC = N * log(inertia/N) + K * p * log(N)

        log("Fitting K-means for K=1 to K=6...")

        results = []

        for K in K_RANGE:
            log(f"Fitting K={K} (n_init={N_INIT}, random_state={RANDOM_STATE})...")

            # Fit K-means
            kmeans = KMeans(
                n_clusters=K,
                random_state=RANDOM_STATE,
                n_init=N_INIT,
                max_iter=300
            )
            kmeans.fit(X)

            # Extract inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_

            # Compute BIC
            # BIC = N * log(inertia / N) + K * p * log(N)
            # Note: Original spec says "K * p * log(N)" but standard BIC for K-means is
            # "K * log(N)" (parameters = K cluster centers, each a scalar for dimensionality)
            # However, following spec exactly: K * p * log(N) where p=6 features
            BIC = N * np.log(inertia / N) + K * p * np.log(N)

            log(f"K={K}: inertia={inertia:.4f}, BIC={BIC:.4f}")

            results.append({
                'K': K,
                'inertia': inertia,
                'BIC': BIC
            })

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        log(f"K-means fitting complete for {len(df_results)} K values")
        # Select Optimal K via BIC Minimum
        # Rule: K with minimum BIC
        # Parsimony rule: If BIC[K+1] - BIC[K] < 2, prefer simpler K

        log("Selecting optimal K via BIC minimum...")

        # Find K with minimum BIC
        idx_min = df_results['BIC'].idxmin()
        K_min = df_results.loc[idx_min, 'K']
        BIC_min = df_results.loc[idx_min, 'BIC']

        log(f"Minimum BIC at K={K_min} (BIC={BIC_min:.4f})")

        # Apply parsimony rule: check if simpler model is acceptable
        # If K_min > 1 and BIC[K_min] - BIC[K_min-1] < 2, prefer K_min-1
        if K_min > 1:
            BIC_prev = df_results.loc[df_results['K'] == K_min - 1, 'BIC'].values[0]
            delta_BIC = BIC_min - BIC_prev

            if delta_BIC < 2:
                log(f"Parsimony rule: BIC[{K_min}] - BIC[{K_min-1}] = {delta_BIC:.4f} < 2")
                log(f"Prefer simpler model K={K_min-1}")
                K_optimal = K_min - 1
            else:
                log(f"Parsimony rule: BIC[{K_min}] - BIC[{K_min-1}] = {delta_BIC:.4f} >= 2")
                log(f"Keep K={K_min}")
                K_optimal = K_min
        else:
            K_optimal = K_min

        BIC_optimal = df_results.loc[df_results['K'] == K_optimal, 'BIC'].values[0]
        log(f"Optimal K={K_optimal} (BIC={BIC_optimal:.4f})")
        # Save Outputs
        # Output 1: Cluster selection table (K, inertia, BIC)
        # Output 2: Optimal K report
        # Output 3: Elbow plot data (same as table)

        log(f"Saving cluster selection table...")
        df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_results)} rows)")

        # Write optimal K report
        log(f"Saving optimal K report...")
        with open(OPTIMAL_K_FILE, 'w', encoding='utf-8') as f:
            f.write("OPTIMAL K SELECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"BIC VALUES:\n")
            f.write("-" * 80 + "\n")
            for _, row in df_results.iterrows():
                f.write(f"K={int(row['K'])}:  BIC={row['BIC']:10.4f}  inertia={row['inertia']:10.4f}\n")

            f.write("\n")
            f.write(f"SELECTION RATIONALE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Minimum BIC at K={K_min} (BIC={BIC_min:.4f})\n")

            if K_optimal != K_min:
                BIC_prev = df_results.loc[df_results['K'] == K_optimal, 'BIC'].values[0]
                delta_BIC = BIC_min - BIC_prev
                f.write(f"Parsimony rule applied: BIC[{K_min}] - BIC[{K_optimal}] = {delta_BIC:.4f} < 2\n")
                f.write(f"Selected simpler model K={K_optimal}\n")
            else:
                f.write(f"Selected K={K_optimal} (minimum BIC)\n")

            f.write("\n")
            f.write(f"OPTIMAL K: {K_optimal}\n")

        log(f"{OPTIMAL_K_FILE}")

        # Save elbow plot data (same as cluster selection table)
        log(f"Saving elbow plot data...")
        df_results.to_csv(ELBOW_DATA_FILE, index=False, encoding='utf-8')
        log(f"{ELBOW_DATA_FILE}")
        # Run Validation Tool
        # Validates: 6 rows, 3 columns (K, inertia, BIC)

        log("Running validate_dataframe_structure...")

        validation_result = validate_dataframe_structure(
            df=df_results,
            expected_rows=6,
            expected_columns=['K', 'inertia', 'BIC'],
            column_types=None
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

        # Check validation passed
        if not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")

        # Additional validation: inertia decreases monotonically
        inertia_values = df_results['inertia'].values
        if not all(inertia_values[i] > inertia_values[i+1] for i in range(len(inertia_values)-1)):
            log("Inertia does NOT decrease monotonically (unexpected)")
        else:
            log("Inertia decreases monotonically (expected)")

        # Additional validation: BIC minimum in [2, 6] (K=1 not meaningful)
        if K_optimal < 2 or K_optimal > 6:
            log(f"Optimal K={K_optimal} outside expected range [2, 6]")
        else:
            log(f"Optimal K={K_optimal} in expected range [2, 6]")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
