#!/usr/bin/env python3
"""
Step 01: Standardize Clustering Features to Z-Scores

Standardize all 6 clustering features (intercepts + slopes) to mean=0, SD=1
for equal weighting in K-means clustering.

Input: data/step00_random_effects_from_rq546.csv (100 rows × 7 columns)
Output: data/step01_standardized_features.csv (100 rows × 7 columns, z-scored)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = RQ_DIR / "data/step00_random_effects_from_rq546.csv"
OUTPUT_FILE = RQ_DIR / "data/step01_standardized_features.csv"
LOG_FILE = RQ_DIR / "logs/step01_standardize_features.log"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 01: Standardize Features")

        # Load wide-format random effects
        log(f"[LOAD] Reading {INPUT_FILE}")
        df_features = pd.read_csv(INPUT_FILE)
        log(f"[LOADED] {len(df_features)} rows, {len(df_features.columns)} columns")

        # Extract feature columns (all except UID)
        feature_cols = [
            'Common_Intercept', 'Common_Slope',
            'Congruent_Intercept', 'Congruent_Slope',
            'Incongruent_Intercept', 'Incongruent_Slope'
        ]

        log(f"[INFO] Standardizing {len(feature_cols)} features")

        # Extract features as numpy array
        X = df_features[feature_cols].values

        # Standardize to z-scores (mean=0, SD=1)
        log("[STANDARDIZE] Applying StandardScaler (mean=0, SD=1)")
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_z = scaler.fit_transform(X)

        log(f"[INFO] Before standardization:")
        log(f"  Mean: {X.mean(axis=0)}")
        log(f"  SD: {X.std(axis=0)}")
        log(f"[INFO] After standardization:")
        log(f"  Mean: {X_z.mean(axis=0)}")
        log(f"  SD: {X_z.std(axis=0)}")

        # Create output DataFrame with z-scored features
        df_standardized = pd.DataFrame({
            'UID': df_features['UID'],
            'Common_Intercept_z': X_z[:, 0],
            'Common_Slope_z': X_z[:, 1],
            'Congruent_Intercept_z': X_z[:, 2],
            'Congruent_Slope_z': X_z[:, 3],
            'Incongruent_Intercept_z': X_z[:, 4],
            'Incongruent_Slope_z': X_z[:, 5]
        })

        # Verify no NaN values
        nan_count = df_standardized.isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values after standardization")

        # Save to CSV
        log(f"[SAVE] Writing to {OUTPUT_FILE}")
        df_standardized.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"[SAVED] {OUTPUT_FILE}")

        # Validate standardization
        log("[VALIDATION] Validating standardization")
        z_cols = [f"{col}_z" for col in feature_cols]
        validation_result = validate_standardization(
            df=df_standardized,
            column_names=z_cols,
            tolerance=0.1
        )

        if not validation_result['valid']:
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log(f"[VALIDATION PASS] {validation_result['message']}")
        log("[SUCCESS] Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK]")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
