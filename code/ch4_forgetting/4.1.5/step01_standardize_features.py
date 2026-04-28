#!/usr/bin/env python3
"""Standardize Clustering Features: Standardize Total_Intercept and Total_Slope to z-scores (mean=0, SD=1) for equal"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_standardize_features.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Standardize Clustering Features")
        # Load Input Data
        #          from best-fitting LMM trajectory model

        log("Loading random effects from Step 0...")
        input_path = RQ_DIR / "data" / "step00_random_effects_from_rq514.csv"

        random_effects = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(random_effects)} rows, {len(random_effects.columns)} cols)")
        log(f"Columns: {random_effects.columns.tolist()}")

        # Check for NaN values (pre-standardization)
        if random_effects[['Total_Intercept', 'Total_Slope']].isnull().any().any():
            n_missing_intercept = random_effects['Total_Intercept'].isnull().sum()
            n_missing_slope = random_effects['Total_Slope'].isnull().sum()
            raise ValueError(f"NaN values detected before standardization: Total_Intercept={n_missing_intercept}, Total_Slope={n_missing_slope}")

        log(f"Pre-standardization stats:")
        log(f"  Total_Intercept: mean={random_effects['Total_Intercept'].mean():.4f}, SD={random_effects['Total_Intercept'].std():.4f}")
        log(f"  Total_Slope: mean={random_effects['Total_Slope'].mean():.4f}, SD={random_effects['Total_Slope'].std():.4f}")
        # Run Analysis Tool (scipy.stats.zscore)
        # Formula: z = (X - mean(X)) / std(X)

        log("Computing z-scores for Total_Intercept and Total_Slope...")

        # Extract raw features as numpy arrays
        intercept_raw = random_effects['Total_Intercept'].values
        slope_raw = random_effects['Total_Slope'].values

        # Compute z-scores using scipy.stats.zscore
        # Parameters:
        #   axis=0: Standardize along column axis (each column independently)
        #   ddof=0: Use population standard deviation (N denominator, not N-1)
        #   nan_policy='propagate': Propagate NaN values (will fail if NaN present)
        intercept_z = zscore(intercept_raw, axis=0, ddof=0, nan_policy='propagate')
        slope_z = zscore(slope_raw, axis=0, ddof=0, nan_policy='propagate')

        log("Z-score computation complete")

        # Verify standardization quality
        mean_intercept_z = np.mean(intercept_z)
        sd_intercept_z = np.std(intercept_z, ddof=0)
        mean_slope_z = np.mean(slope_z)
        sd_slope_z = np.std(slope_z, ddof=0)

        log(f"Post-standardization stats:")
        log(f"  Intercept_z: mean={mean_intercept_z:.6f}, SD={sd_intercept_z:.6f}")
        log(f"  Slope_z: mean={mean_slope_z:.6f}, SD={sd_slope_z:.6f}")

        # Quick sanity check (validation tool will do comprehensive check)
        if abs(mean_intercept_z) > 0.01 or abs(mean_slope_z) > 0.01:
            log(f"Mean not close to 0 (Intercept_z: {mean_intercept_z:.6f}, Slope_z: {mean_slope_z:.6f})")
        if not (0.95 < sd_intercept_z < 1.05) or not (0.95 < sd_slope_z < 1.05):
            log(f"SD not close to 1 (Intercept_z: {sd_intercept_z:.6f}, Slope_z: {sd_slope_z:.6f})")
        # Save Analysis Outputs
        # These outputs will be used by: Step 2 (K-means clustering)

        log("Creating standardized features DataFrame...")

        # Create DataFrame with UID and z-scored features
        standardized_features = pd.DataFrame({
            'UID': random_effects['UID'],
            'Intercept_z': intercept_z,
            'Slope_z': slope_z
        })

        output_path = RQ_DIR / "data" / "step01_standardized_features.csv"
        standardized_features.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(standardized_features)} rows, {len(standardized_features.columns)} cols)")
        log(f"Columns: {standardized_features.columns.tolist()}")
        # Run Validation Tool
        # Validates: Mean ~ 0 (|mean| < tolerance), SD ~ 1 (|SD - 1| < tolerance)
        # Threshold: tolerance=0.01 (strict validation for standardization quality)

        log("Running validate_standardization...")

        validation_result = validate_standardization(
            df=standardized_features,
            column_names=['Intercept_z', 'Slope_z'],
            tolerance=0.01  # Strict tolerance: mean within ±0.01, SD within 0.99-1.01
        )

        # Report validation results
        if validation_result['valid']:
            log("Standardization quality PASS")
            log(f"Mean values: {validation_result['mean_values']}")
            log(f"SD values: {validation_result['sd_values']}")
        else:
            log(f"Standardization quality FAIL")
            log(f"{validation_result['message']}")
            raise ValueError(f"Standardization validation failed: {validation_result['message']}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
