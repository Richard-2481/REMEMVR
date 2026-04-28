#!/usr/bin/env python3
"""Fit LPA Models: Fit Latent Profile Analysis models for K=1, 2, 3, 4 profiles using sklearn"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import traceback
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lpa_models.log"
OUTPUT_DIR = RQ_DIR / "data"

# Input
INPUT_FILE = OUTPUT_DIR / 'step01_domain_theta_scores.csv'

# Outputs
OUTPUT_MODELS = OUTPUT_DIR / 'step02_lpa_fitted_models.pkl'
OUTPUT_FIT = OUTPUT_DIR / 'step02_lpa_fit_comparison.csv'
OUTPUT_DIAGNOSTICS = OUTPUT_DIR / 'step02_lpa_convergence_diagnostics.txt'

# LPA Parameters
K_RANGE = [1, 2, 3, 4]
N_INIT = 100  # Multiple random starts for stability
MAX_ITER = 1000
RANDOM_STATE = 42
COVARIANCE_TYPE = 'full'  # Full covariance (most flexible)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def compute_entropy(posterior_probs):
    """
    Compute entropy for classification quality.

    Formula: E = 1 - ( -sum(p_ik * log(p_ik)) / (N * log(K)) )

    Where:
    - p_ik = posterior probability of participant i in class k
    - N = number of participants
    - K = number of classes

    Range: [0, 1] where 1 = perfect classification
    Threshold: > 0.80 indicates good classification
    """
    n_participants, n_classes = posterior_probs.shape

    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    posterior_probs = np.clip(posterior_probs, epsilon, 1.0)

    # Compute entropy
    entropy_sum = -np.sum(posterior_probs * np.log(posterior_probs))
    max_entropy = n_participants * np.log(n_classes)

    # Normalized entropy (higher = better classification)
    if max_entropy == 0:  # K=1 case
        return 1.0

    entropy = 1.0 - (entropy_sum / max_entropy)

    return entropy

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit LPA Models")
        # Load Standardized Theta Scores

        log("\nLoading standardized theta scores...")
        log(f"{INPUT_FILE}")

        df_theta = pd.read_csv(INPUT_FILE)
        log(f"{len(df_theta)} rows, {len(df_theta.columns)} columns")
        log(f"{df_theta.columns.tolist()}")

        # Scientific Mantra Checkpoint
        log("\nInput validation")
        expected_cols = ['UID', 'theta_What', 'theta_Where']
        if df_theta.columns.tolist() != expected_cols:
            raise ValueError(f"Column mismatch. Expected: {expected_cols}")
        if len(df_theta) != 100:
            log(f"Expected 100 rows, got {len(df_theta)}")
        log(f"Input structure validated")

        # Extract feature matrix (exclude UID)
        X = df_theta[['theta_What', 'theta_Where']].values
        log(f"Feature matrix shape: {X.shape}")
        log(f"Feature means: {X.mean(axis=0)}")
        log(f"Feature SDs: {X.std(axis=0)}")
        # Fit LPA Models for K=1, 2, 3, 4

        log(f"\nFitting LPA models for K={K_RANGE}")
        log(f"n_init={N_INIT}, max_iter={MAX_ITER}, covariance_type='{COVARIANCE_TYPE}'")

        fitted_models = {}
        fit_results = []

        for k in K_RANGE:
            log(f"\n[FIT K={k}] Fitting {k}-profile model...")

            # Initialize and fit model
            model = GaussianMixture(
                n_components=k,
                covariance_type=COVARIANCE_TYPE,
                n_init=N_INIT,
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
                verbose=0
            )

            model.fit(X)

            # Check convergence
            converged = model.converged_
            n_iter = model.n_iter_

            log(f"[FIT K={k}] Converged: {converged}, Iterations: {n_iter}")

            if not converged:
                log(f"Model K={k} did NOT converge after {MAX_ITER} iterations")

            # Compute fit indices
            bic = model.bic(X)
            aic = model.aic(X)
            log_likelihood = model.score(X) * len(X)  # score() returns mean log-likelihood

            # Compute entropy (classification quality)
            posterior_probs = model.predict_proba(X)
            entropy = compute_entropy(posterior_probs)

            log(f"[FIT K={k}] BIC: {bic:.2f}, AIC: {aic:.2f}, Entropy: {entropy:.4f}")
            log(f"[FIT K={k}] Log-Likelihood: {log_likelihood:.2f}")

            # Store model and results
            fitted_models[k] = model
            fit_results.append({
                'K': k,
                'BIC': bic,
                'AIC': aic,
                'Entropy': entropy,
                'Converged': converged,
                'N_Iter': n_iter,
                'LogLikelihood': log_likelihood
            })

        # Scientific Mantra Checkpoint
        log("\nModel fitting validation")
        all_converged = all(r['Converged'] for r in fit_results)
        if not all_converged:
            log("Not all models converged - results may be unreliable")
        else:
            log("All models converged successfully")
        # Save Fitted Models (Pickle)

        log(f"\nSaving fitted models to {OUTPUT_MODELS}")

        with open(OUTPUT_MODELS, 'wb') as f:
            pickle.dump(fitted_models, f)

        log(f"{len(fitted_models)} models saved")
        # Save Fit Comparison (CSV)

        log(f"\nSaving fit comparison to {OUTPUT_FIT}")

        df_fit = pd.DataFrame(fit_results)
        df_fit = df_fit[['K', 'BIC', 'AIC', 'Entropy', 'Converged']]  # Select key columns

        df_fit.to_csv(OUTPUT_FIT, index=False, encoding='utf-8')

        log(f"Fit comparison table:")
        log(f"{df_fit.to_string(index=False)}")

        # Identify BIC minimum
        bic_min_k = df_fit.loc[df_fit['BIC'].idxmin(), 'K']
        log(f"\nBIC minimum at K={bic_min_k}")
        # Save Convergence Diagnostics (Text)

        log(f"\nSaving convergence diagnostics to {OUTPUT_DIAGNOSTICS}")

        with open(OUTPUT_DIAGNOSTICS, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LPA CONVERGENCE DIAGNOSTICS - RQ 7.8.1\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"N Participants: {len(X)}\n")
            f.write(f"N Features: {X.shape[1]} (theta_What, theta_Where)\n")
            f.write(f"K Range: {K_RANGE}\n")
            f.write(f"Random Starts (n_init): {N_INIT}\n")
            f.write(f"Max Iterations: {MAX_ITER}\n")
            f.write(f"Covariance Type: {COVARIANCE_TYPE}\n")
            f.write(f"Random State: {RANDOM_STATE}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MODEL FIT SUMMARY\n")
            f.write("-" * 80 + "\n\n")

            for result in fit_results:
                f.write(f"K = {result['K']}\n")
                f.write(f"  Converged: {result['Converged']}\n")
                f.write(f"  Iterations: {result['N_Iter']}\n")
                f.write(f"  BIC: {result['BIC']:.2f}\n")
                f.write(f"  AIC: {result['AIC']:.2f}\n")
                f.write(f"  Entropy: {result['Entropy']:.4f}\n")
                f.write(f"  Log-Likelihood: {result['LogLikelihood']:.2f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MODEL SELECTION GUIDANCE\n")
            f.write("-" * 80 + "\n\n")
            f.write("BIC: Lower is better (balances fit + parsimony)\n")
            f.write("AIC: Lower is better (less penalizes complexity than BIC)\n")
            f.write("Entropy: Higher is better (>0.80 = good classification)\n")
            f.write("Minimum profile size: Each profile should have N>=20 participants\n\n")

            f.write(f"BIC minimum: K={bic_min_k}\n")

            # Entropy quality
            high_entropy_ks = [r['K'] for r in fit_results if r['Entropy'] > 0.80]
            f.write(f"High entropy (>0.80): K={high_entropy_ks}\n")

        log(f"Convergence diagnostics written")
        # VALIDATION: Check Model Quality

        log("\nModel quality checks...")

        # Check 1: All models converged
        if not all_converged:
            log("Some models did not converge")
        else:
            log("All models converged")

        # Check 2: BIC pattern (should decrease then increase)
        bic_values = df_fit['BIC'].values
        bic_decreasing_first = bic_values[0] > bic_values[1]

        if bic_decreasing_first:
            log("BIC decreases from K=1 to K=2 (expected pattern)")
        else:
            log("BIC does not decrease from K=1 to K=2 (unusual)")

        # Check 3: Entropy quality
        high_entropy_count = len(high_entropy_ks)
        log(f"{high_entropy_count} models with entropy >0.80")

        log("\nStep 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
