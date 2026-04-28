#!/usr/bin/env python3
"""Classification Quality: Assess LPA classification quality using multiple metrics: entropy, silhouette"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import traceback
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step06_classification_quality.log"
OUTPUT_DIR = RQ_DIR / "data"

# Inputs
INPUT_MODELS = OUTPUT_DIR / 'step02_lpa_fitted_models.pkl'
INPUT_PROFILES = OUTPUT_DIR / 'step03_optimal_profiles.csv'
INPUT_THETA = OUTPUT_DIR / 'step01_domain_theta_scores.csv'

# Outputs
OUTPUT_QUALITY = OUTPUT_DIR / 'step06_classification_quality.csv'
OUTPUT_DIAGNOSTICS = OUTPUT_DIR / 'step06_model_diagnostics.txt'

# Bootstrap Parameters
N_BOOTSTRAP = 100
RANDOM_STATE = 42
JACCARD_THRESHOLD = 0.75

# Quality Thresholds
ENTROPY_THRESHOLD = 0.80
SILHOUETTE_THRESHOLD = 0.25

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

    Range: [0, 1] where 1 = perfect classification
    """
    n_participants, n_classes = posterior_probs.shape

    epsilon = 1e-10
    posterior_probs = np.clip(posterior_probs, epsilon, 1.0)

    entropy_sum = -np.sum(posterior_probs * np.log(posterior_probs))
    max_entropy = n_participants * np.log(n_classes)

    if max_entropy == 0:
        return 1.0

    entropy = 1.0 - (entropy_sum / max_entropy)

    return entropy

def jaccard_index(labels1, labels2):
    """
    Compute Jaccard index for agreement between two label sets.

    Formula: J = (# pairs in same cluster in both) / (# pairs in same cluster in at least one)

    Range: [0, 1] where 1 = perfect agreement
    """
    n = len(labels1)

    # Count pairs in same cluster in both labelings
    same_in_both = 0
    same_in_at_least_one = 0

    for i in range(n):
        for j in range(i+1, n):
            same_in_1 = (labels1[i] == labels1[j])
            same_in_2 = (labels2[i] == labels2[j])

            if same_in_1 and same_in_2:
                same_in_both += 1

            if same_in_1 or same_in_2:
                same_in_at_least_one += 1

    if same_in_at_least_one == 0:
        return 1.0

    jaccard = same_in_both / same_in_at_least_one

    return jaccard

def interpret_metric(metric_name, value):
    """
    Provide interpretation for quality metric values.
    """
    interpretations = {
        'Entropy': "Excellent" if value > 0.90 else ("Good" if value > 0.80 else ("Fair" if value > 0.70 else "Poor")),
        'Silhouette': "Good" if value > 0.50 else ("Fair" if value > 0.25 else "Poor"),
        'Davies_Bouldin': "Good" if value < 1.0 else ("Fair" if value < 1.5 else "Poor"),
        'Calinski_Harabasz': "Good" if value > 100 else ("Fair" if value > 50 else "Poor"),
        'Bootstrap_Jaccard': "Stable" if value > 0.75 else ("Moderate" if value > 0.60 else "Unstable"),
        'High_Confidence_Pct': "Excellent" if value > 90 else ("Good" if value > 80 else ("Fair" if value > 70 else "Poor"))
    }

    return interpretations.get(metric_name, "N/A")

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Classification Quality")
        # Load Fitted Models

        log("\nLoading fitted models...")
        log(f"{INPUT_MODELS}")

        with open(INPUT_MODELS, 'rb') as f:
            fitted_models = pickle.load(f)

        log(f"{len(fitted_models)} models")
        # Load Profile Assignments

        log(f"\nLoading profile assignments...")
        log(f"{INPUT_PROFILES}")

        df_profiles = pd.read_csv(INPUT_PROFILES)
        log(f"{len(df_profiles)} participants")

        optimal_k = df_profiles['Profile'].nunique()
        log(f"Optimal K={optimal_k} profiles")
        # Load Theta Scores

        log(f"\nLoading theta scores...")
        log(f"{INPUT_THETA}")

        df_theta = pd.read_csv(INPUT_THETA)
        log(f"{len(df_theta)} participants")

        X = df_theta[['theta_What', 'theta_Where']].values
        labels = df_profiles['Profile'].values

        # Scientific Mantra Checkpoint
        log("\nInput validation")
        log(f"Data loaded successfully")
        # Compute Classification Quality Metrics

        log("\nComputing classification quality metrics...")

        optimal_model = fitted_models[optimal_k]

        # Metric 1: Entropy
        log("\n[METRIC 1] Entropy (classification certainty)")
        posterior_probs = optimal_model.predict_proba(X)
        entropy = compute_entropy(posterior_probs)
        log(f"Entropy: {entropy:.4f} (threshold: >{ENTROPY_THRESHOLD})")
        log(f"{interpret_metric('Entropy', entropy)}")

        # Metric 2: Silhouette Coefficient
        log("\n[METRIC 2] Silhouette Coefficient (cluster separation)")
        if optimal_k > 1:
            silhouette = silhouette_score(X, labels, metric='euclidean')
            log(f"Silhouette: {silhouette:.4f} (threshold: >{SILHOUETTE_THRESHOLD})")
            log(f"{interpret_metric('Silhouette', silhouette)}")
        else:
            silhouette = np.nan
            log(f"Silhouette not applicable for K=1")

        # Metric 3: Davies-Bouldin Index
        log("\n[METRIC 3] Davies-Bouldin Index (cluster compactness)")
        if optimal_k > 1:
            davies_bouldin = davies_bouldin_score(X, labels)
            log(f"Davies-Bouldin: {davies_bouldin:.4f} (threshold: <1.5, lower=better)")
            log(f"{interpret_metric('Davies_Bouldin', davies_bouldin)}")
        else:
            davies_bouldin = np.nan
            log(f"Davies-Bouldin not applicable for K=1")

        # Metric 4: Calinski-Harabasz Index
        log("\n[METRIC 4] Calinski-Harabasz Index (variance ratio)")
        if optimal_k > 1:
            calinski = calinski_harabasz_score(X, labels)
            log(f"Calinski-Harabasz: {calinski:.2f} (higher=better)")
            log(f"{interpret_metric('Calinski_Harabasz', calinski)}")
        else:
            calinski = np.nan
            log(f"Calinski-Harabasz not applicable for K=1")

        # Metric 5: Average Posterior Probability
        log("\n[METRIC 5] Average Posterior Probability")
        avg_posterior = posterior_probs.max(axis=1).mean()
        log(f"Mean max posterior: {avg_posterior:.4f}")

        # Metric 6: High Confidence Percentage
        log("\n[METRIC 6] High Confidence Assignments (>80%)")
        high_confidence_count = (posterior_probs.max(axis=1) > 0.80).sum()
        high_confidence_pct = 100.0 * high_confidence_count / len(posterior_probs)
        log(f"High confidence: {high_confidence_count}/{len(posterior_probs)} ({high_confidence_pct:.1f}%)")
        log(f"{interpret_metric('High_Confidence_Pct', high_confidence_pct)}")

        # Scientific Mantra Checkpoint
        log("\nMetric computation validation")
        log(f"All metrics computed")
        # Bootstrap Stability Assessment

        log(f"\nAssessing classification stability...")
        log(f"n_bootstrap={N_BOOTSTRAP}, random_state={RANDOM_STATE}")

        np.random.seed(RANDOM_STATE)

        jaccard_scores = []

        for i in range(N_BOOTSTRAP):
            if (i + 1) % 25 == 0:
                log(f"Iteration {i+1}/{N_BOOTSTRAP}...")

            # Resample participants with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]

            # Fit model on bootstrap sample
            boot_model = GaussianMixture(
                n_components=optimal_k,
                covariance_type='full',
                n_init=10,  # Fewer inits for speed
                max_iter=100,
                random_state=RANDOM_STATE + i,
                verbose=0
            )
            boot_model.fit(X_boot)

            # Predict labels for original data
            boot_labels = boot_model.predict(X)

            # Compute Jaccard index (agreement with original labels)
            jaccard = jaccard_index(labels, boot_labels)
            jaccard_scores.append(jaccard)

        # Bootstrap summary statistics
        mean_jaccard = np.mean(jaccard_scores)
        median_jaccard = np.median(jaccard_scores)
        min_jaccard = np.min(jaccard_scores)

        log(f"\nStability results:")
        log(f"Mean Jaccard: {mean_jaccard:.4f}")
        log(f"Median Jaccard: {median_jaccard:.4f}")
        log(f"Min Jaccard: {min_jaccard:.4f}")
        log(f"{interpret_metric('Bootstrap_Jaccard', mean_jaccard)}")

        # Scientific Mantra Checkpoint
        log("\nBootstrap validation")
        log(f"Bootstrap stability assessed ({N_BOOTSTRAP} iterations)")
        # Save Classification Quality Metrics

        log(f"\nSaving classification quality metrics to {OUTPUT_QUALITY}")

        quality_metrics = [
            {'Metric': 'Entropy', 'Value': entropy, 'Interpretation': interpret_metric('Entropy', entropy)},
            {'Metric': 'Silhouette_Coefficient', 'Value': silhouette, 'Interpretation': interpret_metric('Silhouette', silhouette) if not np.isnan(silhouette) else 'N/A'},
            {'Metric': 'Davies_Bouldin_Index', 'Value': davies_bouldin, 'Interpretation': interpret_metric('Davies_Bouldin', davies_bouldin) if not np.isnan(davies_bouldin) else 'N/A'},
            {'Metric': 'Calinski_Harabasz_Index', 'Value': calinski, 'Interpretation': interpret_metric('Calinski_Harabasz', calinski) if not np.isnan(calinski) else 'N/A'},
            {'Metric': 'Avg_Posterior_Probability', 'Value': avg_posterior, 'Interpretation': f"{avg_posterior:.1%} certainty"},
            {'Metric': 'High_Confidence_Pct', 'Value': high_confidence_pct, 'Interpretation': interpret_metric('High_Confidence_Pct', high_confidence_pct)},
            {'Metric': 'Bootstrap_Jaccard_Mean', 'Value': mean_jaccard, 'Interpretation': interpret_metric('Bootstrap_Jaccard', mean_jaccard)}
        ]

        df_quality = pd.DataFrame(quality_metrics)
        df_quality.to_csv(OUTPUT_QUALITY, index=False, encoding='utf-8')

        log(f"{len(df_quality)} metrics")
        log(f"\n[QUALITY TABLE]")
        log(f"{df_quality.to_string(index=False)}")
        # Save Model Diagnostics Report

        log(f"\nSaving model diagnostics to {OUTPUT_DIAGNOSTICS}")

        with open(OUTPUT_DIAGNOSTICS, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL DIAGNOSTICS REPORT - RQ 7.8.1 LPA\n")
            f.write("=" * 80 + "\n\n")

            f.write("OPTIMAL MODEL:\n")
            f.write(f"  K = {optimal_k} profiles\n")
            f.write(f"  N = {len(X)} participants\n")
            f.write(f"  Features: theta_What, theta_Where\n\n")

            f.write("-" * 80 + "\n")
            f.write("CLASSIFICATION QUALITY METRICS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Entropy: {entropy:.4f}\n")
            f.write(f"  Interpretation: {interpret_metric('Entropy', entropy)}\n")
            f.write(f"  Threshold: >{ENTROPY_THRESHOLD} for good classification\n\n")

            if not np.isnan(silhouette):
                f.write(f"Silhouette Coefficient: {silhouette:.4f}\n")
                f.write(f"  Interpretation: {interpret_metric('Silhouette', silhouette)}\n")
                f.write(f"  Range: [-1, 1], >0.25 indicates reasonable separation\n\n")

            if not np.isnan(davies_bouldin):
                f.write(f"Davies-Bouldin Index: {davies_bouldin:.4f}\n")
                f.write(f"  Interpretation: {interpret_metric('Davies_Bouldin', davies_bouldin)}\n")
                f.write(f"  Range: [0, inf], <1.5 indicates good separation (lower=better)\n\n")

            if not np.isnan(calinski):
                f.write(f"Calinski-Harabasz Index: {calinski:.2f}\n")
                f.write(f"  Interpretation: {interpret_metric('Calinski_Harabasz', calinski)}\n")
                f.write(f"  Range: [0, inf], higher=better\n\n")

            f.write(f"Average Posterior Probability: {avg_posterior:.4f}\n")
            f.write(f"  Interpretation: {avg_posterior:.1%} mean classification certainty\n\n")

            f.write(f"High Confidence Assignments (>80%): {high_confidence_count}/{len(posterior_probs)} ({high_confidence_pct:.1f}%)\n")
            f.write(f"  Interpretation: {interpret_metric('High_Confidence_Pct', high_confidence_pct)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("BOOTSTRAP STABILITY ASSESSMENT\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Bootstrap Iterations: {N_BOOTSTRAP}\n")
            f.write(f"Stability Metric: Jaccard Index (classification agreement)\n\n")

            f.write(f"Mean Jaccard: {mean_jaccard:.4f}\n")
            f.write(f"Median Jaccard: {median_jaccard:.4f}\n")
            f.write(f"Min Jaccard: {min_jaccard:.4f}\n\n")

            f.write(f"Interpretation: {interpret_metric('Bootstrap_Jaccard', mean_jaccard)}\n")
            f.write(f"Threshold: >{JACCARD_THRESHOLD} indicates stable classifications\n\n")

            f.write("-" * 80 + "\n")
            f.write("OVERALL QUALITY ASSESSMENT\n")
            f.write("-" * 80 + "\n\n")

            # Summary assessment
            quality_pass_count = 0
            total_checks = 0

            if entropy > ENTROPY_THRESHOLD:
                f.write("Entropy exceeds threshold (good classification certainty)\n")
                quality_pass_count += 1
            else:
                f.write("Entropy below threshold (poor classification certainty)\n")
            total_checks += 1

            if not np.isnan(silhouette) and silhouette > SILHOUETTE_THRESHOLD:
                f.write("Silhouette exceeds threshold (reasonable cluster separation)\n")
                quality_pass_count += 1
                total_checks += 1
            elif not np.isnan(silhouette):
                f.write("Silhouette below threshold (poor cluster separation)\n")
                total_checks += 1

            if mean_jaccard > JACCARD_THRESHOLD:
                f.write("Bootstrap Jaccard exceeds threshold (stable classifications)\n")
                quality_pass_count += 1
            else:
                f.write("Bootstrap Jaccard below threshold (unstable classifications)\n")
            total_checks += 1

            f.write(f"\nOverall: {quality_pass_count}/{total_checks} quality checks passed\n")

        log(f"Model diagnostics written")
        # VALIDATION: Final Checks

        log("\nFinal validation...")

        # Check 1: Entropy in valid range
        if not (0 <= entropy <= 1):
            raise ValueError(f"Entropy out of range: {entropy}")
        log(f"Entropy in valid range [0, 1]")

        # Check 2: Bootstrap Jaccard scores reasonable
        if mean_jaccard < 0 or mean_jaccard > 1:
            raise ValueError(f"Invalid Jaccard index: {mean_jaccard}")
        log(f"Jaccard index in valid range [0, 1]")

        log("\nStep 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
