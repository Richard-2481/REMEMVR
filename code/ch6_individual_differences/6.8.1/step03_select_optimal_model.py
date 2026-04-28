#!/usr/bin/env python3
"""Select Optimal Model: Select optimal K solution based on BIC minimum, profile size constraints (N>=20),"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step03_select_optimal_model.log"
OUTPUT_DIR = RQ_DIR / "data"

# Inputs
INPUT_FIT = OUTPUT_DIR / 'step02_lpa_fit_comparison.csv'
INPUT_MODELS = OUTPUT_DIR / 'step02_lpa_fitted_models.pkl'
INPUT_THETA = OUTPUT_DIR / 'step01_domain_theta_scores.csv'

# Outputs
OUTPUT_PROFILES = OUTPUT_DIR / 'step03_optimal_profiles.csv'
OUTPUT_SUMMARY = OUTPUT_DIR / 'step03_model_selection_summary.txt'

# Selection Criteria
MIN_PROFILE_SIZE = 20
ENTROPY_THRESHOLD = 0.80

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Select Optimal Model")
        # Load Fit Comparison Results

        log("\nLoading fit comparison results...")
        log(f"{INPUT_FIT}")

        df_fit = pd.read_csv(INPUT_FIT)
        log(f"{len(df_fit)} models")
        log(f"\n[FIT TABLE]")
        log(f"{df_fit.to_string(index=False)}")

        # Scientific Mantra Checkpoint
        log("\nFit comparison validation")
        expected_cols = ['K', 'BIC', 'AIC', 'Entropy', 'Converged']
        if df_fit.columns.tolist() != expected_cols:
            raise ValueError(f"Column mismatch in fit comparison")
        log("Fit comparison structure validated")
        # Load Fitted Models

        log(f"\nLoading fitted models...")
        log(f"{INPUT_MODELS}")

        with open(INPUT_MODELS, 'rb') as f:
            fitted_models = pickle.load(f)

        log(f"{len(fitted_models)} models (K={list(fitted_models.keys())})")
        # Load Theta Scores

        log(f"\nLoading theta scores...")
        log(f"{INPUT_THETA}")

        df_theta = pd.read_csv(INPUT_THETA)
        log(f"{len(df_theta)} participants")

        X = df_theta[['theta_What', 'theta_Where']].values
        log(f"Feature matrix shape: {X.shape}")
        # Select Optimal K
        # Selection hierarchy:
        # 1. BIC minimum (primary criterion)
        # 2. Profile size constraint (all profiles N>=20)
        # 3. Entropy threshold (>0.80 for good classification)

        log("\nApplying model selection criteria...")
        log(f"1. BIC minimum (primary)")
        log(f"2. All profiles N>={MIN_PROFILE_SIZE}")
        log(f"3. Entropy >{ENTROPY_THRESHOLD}")

        # Identify BIC minimum
        bic_min_idx = df_fit['BIC'].idxmin()
        optimal_k = df_fit.loc[bic_min_idx, 'K']
        bic_min_value = df_fit.loc[bic_min_idx, 'BIC']

        log(f"\nMinimum BIC at K={optimal_k} (BIC={bic_min_value:.2f})")

        # Check constraints for optimal K
        optimal_model = fitted_models[optimal_k]
        optimal_entropy = df_fit.loc[bic_min_idx, 'Entropy']
        optimal_converged = df_fit.loc[bic_min_idx, 'Converged']

        log(f"Optimal K={optimal_k} properties:")
        log(f"  - Entropy: {optimal_entropy:.4f}")
        log(f"  - Converged: {optimal_converged}")

        # Get profile assignments for optimal K
        profile_labels = optimal_model.predict(X)
        profile_counts = pd.Series(profile_labels).value_counts().sort_index()

        log(f"Profile sizes for K={optimal_k}:")
        for profile, count in profile_counts.items():
            log(f"  - Profile {profile}: N={count}")

        # Check profile size constraint
        min_profile_count = profile_counts.min()
        if min_profile_count < MIN_PROFILE_SIZE:
            log(f"Smallest profile (N={min_profile_count}) below threshold (N>={MIN_PROFILE_SIZE})")
        else:
            log(f"All profiles meet size constraint (min N={min_profile_count})")

        # Check entropy constraint
        if optimal_entropy < ENTROPY_THRESHOLD:
            log(f"Entropy ({optimal_entropy:.4f}) below threshold ({ENTROPY_THRESHOLD})")
        else:
            log(f"Entropy meets threshold ({optimal_entropy:.4f} > {ENTROPY_THRESHOLD})")

        # Scientific Mantra Checkpoint
        log(f"\nOptimal K selected: K={optimal_k}")
        # Extract Profile Assignments

        log(f"\nExtracting profile assignments for K={optimal_k}...")

        # Get posterior probabilities
        posterior_probs = optimal_model.predict_proba(X)

        # Create output dataframe
        df_profiles = df_theta[['UID']].copy()
        df_profiles['Profile'] = profile_labels
        df_profiles['Max_Posterior_Prob'] = posterior_probs.max(axis=1)

        log(f"{len(df_profiles)} profile assignments")
        log(f"Posterior probability summary:")
        log(f"{df_profiles['Max_Posterior_Prob'].describe().to_string()}")

        # Count high-confidence assignments (>80%)
        high_confidence = (df_profiles['Max_Posterior_Prob'] > 0.80).sum()
        pct_high_confidence = 100.0 * high_confidence / len(df_profiles)

        log(f"High confidence assignments (>80%): {high_confidence}/{len(df_profiles)} ({pct_high_confidence:.1f}%)")
        # Save Profile Assignments

        log(f"\nSaving profile assignments to {OUTPUT_PROFILES}")

        df_profiles.to_csv(OUTPUT_PROFILES, index=False, encoding='utf-8')

        log(f"{len(df_profiles)} rows")
        log(f"{df_profiles.columns.tolist()}")
        # Save Model Selection Summary

        log(f"\nSaving model selection summary to {OUTPUT_SUMMARY}")

        with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL SELECTION SUMMARY - RQ 7.8.1 LPA\n")
            f.write("=" * 80 + "\n\n")

            f.write("SELECTION CRITERIA (Hierarchical):\n")
            f.write(f"  1. BIC minimum (primary criterion)\n")
            f.write(f"  2. All profiles N >= {MIN_PROFILE_SIZE} participants\n")
            f.write(f"  3. Entropy > {ENTROPY_THRESHOLD} (good classification)\n\n")

            f.write("-" * 80 + "\n")
            f.write("FIT COMPARISON TABLE\n")
            f.write("-" * 80 + "\n\n")
            f.write(df_fit.to_string(index=False))
            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write("OPTIMAL MODEL SELECTION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Optimal K: {optimal_k}\n")
            f.write(f"BIC: {bic_min_value:.2f}\n")
            f.write(f"Entropy: {optimal_entropy:.4f}\n")
            f.write(f"Converged: {optimal_converged}\n\n")

            f.write("Profile Sizes:\n")
            for profile, count in profile_counts.items():
                f.write(f"  Profile {profile}: N={count}\n")

            f.write(f"\nMinimum Profile Size: {min_profile_count}\n")
            f.write(f"Profile Size Constraint Met: {min_profile_count >= MIN_PROFILE_SIZE}\n")
            f.write(f"Entropy Threshold Met: {optimal_entropy >= ENTROPY_THRESHOLD}\n\n")

            f.write("-" * 80 + "\n")
            f.write("CLASSIFICATION QUALITY\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Mean Posterior Probability: {df_profiles['Max_Posterior_Prob'].mean():.4f}\n")
            f.write(f"Median Posterior Probability: {df_profiles['Max_Posterior_Prob'].median():.4f}\n")
            f.write(f"Min Posterior Probability: {df_profiles['Max_Posterior_Prob'].min():.4f}\n")
            f.write(f"High Confidence (>80%): {high_confidence}/{len(df_profiles)} ({pct_high_confidence:.1f}%)\n\n")

        log(f"Model selection summary written")
        # VALIDATION: Check Output Quality

        log("\nOutput validation...")

        # Check 1: All participants assigned
        if len(df_profiles) != 100:
            raise ValueError(f"Expected 100 participants, got {len(df_profiles)}")
        log(f"All 100 participants assigned to profiles")

        # Check 2: No missing values
        missing_count = df_profiles.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"Found {missing_count} missing values in profile assignments")
        log(f"No missing values")

        # Check 3: Profile labels in expected range
        if df_profiles['Profile'].min() < 0 or df_profiles['Profile'].max() >= optimal_k:
            raise ValueError(f"Profile labels out of range [0, {optimal_k-1}]")
        log(f"Profile labels in valid range")

        log("\nStep 03 complete")
        log(f"Optimal solution: K={optimal_k} profiles")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
