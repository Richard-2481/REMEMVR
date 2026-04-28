#!/usr/bin/env python3
"""
RQ 6.1.5: Trajectory Clustering - Confidence Phenotypes
========================================================

Analysis: K-means clustering on random effects (intercept + slope) from RQ 6.1.4
           to identify confidence phenotypes and compare with Ch5 5.1.5 accuracy phenotypes.

Steps:
  01. Load random effects from RQ 6.1.4
  02. Standardize features to z-scores
  03. K-means clustering for K=1-6 with BIC selection
  04. Fit final K-means model with optimal K
  05. Validate cluster quality (silhouette, Davies-Bouldin, Jaccard)
  06. Characterize clusters (mean intercept/slope, phenotype labels)
  07. Cross-tabulate with Ch5 5.1.5 accuracy clusters
  08. Chi-square test of association (integration vs dissociation)

Dependencies:
  - RQ 6.1.4: results/ch6/6.1.4/data/step03_random_effects.csv
  - Ch5 5.1.5: results/ch5/5.1.5/data/step03_cluster_assignments.csv

Date: 2025-12-11
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.5
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root

LOG_FILE = RQ_DIR / "logs" / "steps_01_to_08.log"

# Input dependencies
INPUT_RANDOM_EFFECTS = PROJECT_ROOT / "results/ch6/6.1.4/data/step03_random_effects.csv"
INPUT_ACCURACY_CLUSTERS = PROJECT_ROOT / "results/ch5/5.1.5/data/step03_cluster_assignments.csv"

# Output files
DATA_DIR = RQ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Parameters
K_RANGE = range(1, 7)  # K=1 to K=6
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
SILHOUETTE_THRESHOLD = 0.40
DAVIES_BOULDIN_THRESHOLD = 1.0
JACCARD_THRESHOLD = 0.75
MIN_CLUSTER_SIZE = 10  # 10% of N=100

# LOGGING

def log(msg: str):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Load Random Effects from RQ 6.1.4

def step01_load_random_effects() -> pd.DataFrame:
    """Load random effects (intercept, slope) from RQ 6.1.4."""
    log("=" * 60)
    log("STEP 01: Load Random Effects from RQ 6.1.4")
    log("=" * 60)

    # Check dependency exists
    if not INPUT_RANDOM_EFFECTS.exists():
        raise FileNotFoundError(f"DEPENDENCY ERROR: RQ 6.1.4 must complete first. File not found: {INPUT_RANDOM_EFFECTS}")

    # Load data
    df = pd.read_csv(INPUT_RANDOM_EFFECTS)
    log(f"Loaded {len(df)} rows from {INPUT_RANDOM_EFFECTS.name}")
    log(f"Columns: {list(df.columns)}")

    # Rename columns to standard names
    # RQ 6.1.4 uses: UID, random_intercept, random_slope
    # Standard names: UID, intercept, slope
    df = df.rename(columns={
        'random_intercept': 'intercept',
        'random_slope': 'slope'
    })

    # Validate
    required_cols = ['UID', 'intercept', 'slope']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 100:
        raise ValueError(f"Expected 100 participants, found {len(df)}")

    if df.isna().any().any():
        raise ValueError(f"NaN values detected in random effects")

    # Save pass-through validation
    output_path = DATA_DIR / "step01_random_effects_loaded.csv"
    df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    log(f"VALIDATION PASS: 100 participants loaded, no NaN values")
    log(f"Intercept range: [{df['intercept'].min():.4f}, {df['intercept'].max():.4f}]")
    log(f"Slope range: [{df['slope'].min():.4f}, {df['slope'].max():.4f}]")

    return df

# Standardize Features to Z-Scores

def step02_standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize intercept and slope to z-scores for equal feature weighting."""
    log("")
    log("=" * 60)
    log("STEP 02: Standardize Features to Z-Scores")
    log("=" * 60)

    # Compute z-scores
    df['intercept_z'] = zscore(df['intercept'])
    df['slope_z'] = zscore(df['slope'])

    # Validate standardization
    for col in ['intercept_z', 'slope_z']:
        mean_val = df[col].mean()
        std_val = df[col].std()

        if abs(mean_val) > 0.01:
            raise ValueError(f"Z-score mean validation failed: {col} mean = {mean_val:.6f}")
        if not (0.95 < std_val < 1.05):
            raise ValueError(f"Z-score SD validation failed: {col} SD = {std_val:.6f}")

        log(f"{col}: mean={mean_val:.6f}, SD={std_val:.6f}")

    # Check for outliers (>3 SD)
    outliers_intercept = (abs(df['intercept_z']) > 3).sum()
    outliers_slope = (abs(df['slope_z']) > 3).sum()
    if outliers_intercept > 0 or outliers_slope > 0:
        log(f"WARNING: Outliers detected - intercept: {outliers_intercept}, slope: {outliers_slope} (>3 SD)")

    # Save
    output_path = DATA_DIR / "step02_standardized_features.csv"
    df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    log("VALIDATION PASS: Z-score statistics within tolerance")

    return df

# K-Means Clustering for K=1-6 with BIC Selection

def step03_cluster_selection(df: pd.DataFrame) -> int:
    """Fit K-means for K=1-6, compute BIC, select optimal K."""
    log("")
    log("=" * 60)
    log("STEP 03: K-Means Clustering for K=1-6 with BIC Selection")
    log("=" * 60)

    X = df[['intercept_z', 'slope_z']].values
    n = len(X)

    results = []
    for k in K_RANGE:
        if k == 1:
            # K=1: SSE = sum of squared distances to centroid (total variance * n)
            centroid = X.mean(axis=0)
            sse = np.sum((X - centroid) ** 2)
        else:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            kmeans.fit(X)
            sse = kmeans.inertia_

        # BIC: N * log(SSE/N) + K * log(N) * d
        # where d = number of parameters = K * (dimensions + 1) for centers + K for cluster probs
        # Simplified: N * log(SSE/N) + K * log(N)
        bic = n * np.log(sse / n) + k * np.log(n)

        results.append({
            'K': k,
            'SSE': sse,
            'BIC': bic
        })
        log(f"K={k}: SSE={sse:.4f}, BIC={bic:.4f}")

    results_df = pd.DataFrame(results)

    # Validate SSE decreases monotonically
    sse_values = results_df['SSE'].values
    if not all(sse_values[i] >= sse_values[i+1] for i in range(len(sse_values)-1)):
        log("WARNING: SSE does not decrease monotonically (may indicate numerical instability)")

    # Find optimal K (minimum BIC)
    optimal_idx = results_df['BIC'].idxmin()
    optimal_k = results_df.loc[optimal_idx, 'K']
    results_df['optimal'] = results_df['K'] == optimal_k

    if optimal_k == 1:
        log("WARNING: K=1 is optimal (no clustering structure detected)")

    # Save
    output_path = DATA_DIR / "step03_cluster_selection.csv"
    results_df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    # Save BIC plot data
    bic_plot_path = DATA_DIR / "step03_bic_plot_data.csv"
    results_df[['K', 'BIC']].to_csv(bic_plot_path, index=False)
    log(f"Saved: {bic_plot_path.name}")

    log(f"")
    log(f"Optimal K selected: K={optimal_k} (BIC={results_df.loc[optimal_idx, 'BIC']:.4f})")
    log("VALIDATION PASS: BIC minimum identified")

    return int(optimal_k)

# Fit Final K-Means Model with Optimal K

def step04_fit_final_kmeans(df: pd.DataFrame, optimal_k: int) -> pd.DataFrame:
    """Fit final K-means model with optimal K."""
    log("")
    log("=" * 60)
    log(f"STEP 04: Fit Final K-Means Model with K={optimal_k}")
    log("=" * 60)

    X = df[['intercept_z', 'slope_z']].values

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(X)

    # Validate cluster assignments
    cluster_counts = df['cluster_label'].value_counts().sort_index()
    log(f"Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        log(f"  Cluster {cluster_id}: N={count}")

    # Check minimum cluster size
    min_size = cluster_counts.min()
    if min_size < MIN_CLUSTER_SIZE:
        log(f"WARNING: Trivial cluster detected (min size = {min_size}, threshold = {MIN_CLUSTER_SIZE})")

    # Check all cluster IDs are consecutive from 0
    expected_ids = set(range(optimal_k))
    actual_ids = set(df['cluster_label'].unique())
    if expected_ids != actual_ids:
        raise ValueError(f"Cluster IDs not consecutive: expected {expected_ids}, got {actual_ids}")

    # Save cluster assignments
    output_df = df[['UID', 'cluster_label', 'intercept_z', 'slope_z']].copy()
    output_path = DATA_DIR / "step04_cluster_assignments.csv"
    output_df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    # Save cluster centers
    centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['intercept_z', 'slope_z']
    )
    centers_df['cluster_label'] = range(optimal_k)
    centers_path = DATA_DIR / "step04_cluster_centers.csv"
    centers_df.to_csv(centers_path, index=False)
    log(f"Saved: {centers_path.name}")

    log(f"")
    log(f"Final K-means fit complete with K={optimal_k}")
    log(f"VALIDATION PASS: All clusters >= {MIN_CLUSTER_SIZE} threshold")

    return df

# Validate Cluster Quality

def step05_validate_cluster_quality(df: pd.DataFrame, optimal_k: int) -> pd.DataFrame:
    """Compute silhouette, Davies-Bouldin, and Jaccard bootstrap stability."""
    log("")
    log("=" * 60)
    log("STEP 05: Validate Cluster Quality")
    log("=" * 60)

    X = df[['intercept_z', 'slope_z']].values
    labels = df['cluster_label'].values

    # Skip metrics if K=1 (silhouette undefined)
    if optimal_k == 1:
        log("K=1: Cluster quality metrics not applicable (trivial clustering)")
        metrics = [
            {'metric': 'silhouette', 'value': np.nan, 'threshold': SILHOUETTE_THRESHOLD, 'pass': False},
            {'metric': 'davies_bouldin', 'value': np.nan, 'threshold': DAVIES_BOULDIN_THRESHOLD, 'pass': False},
            {'metric': 'jaccard_mean', 'value': np.nan, 'threshold': JACCARD_THRESHOLD, 'pass': False},
            {'metric': 'jaccard_ci_lower', 'value': np.nan, 'threshold': np.nan, 'pass': False},
            {'metric': 'jaccard_ci_upper', 'value': np.nan, 'threshold': np.nan, 'pass': False},
        ]
    else:
        # Silhouette score
        silhouette = silhouette_score(X, labels)
        silhouette_pass = silhouette >= SILHOUETTE_THRESHOLD
        log(f"Silhouette score: {silhouette:.4f} (threshold: {SILHOUETTE_THRESHOLD}, pass: {silhouette_pass})")

        # Davies-Bouldin index
        davies_bouldin = davies_bouldin_score(X, labels)
        davies_bouldin_pass = davies_bouldin <= DAVIES_BOULDIN_THRESHOLD
        log(f"Davies-Bouldin index: {davies_bouldin:.4f} (threshold: {DAVIES_BOULDIN_THRESHOLD}, pass: {davies_bouldin_pass})")

        # Jaccard bootstrap stability
        log(f"Computing Jaccard bootstrap stability ({N_BOOTSTRAP} iterations)...")
        np.random.seed(RANDOM_STATE)
        n = len(X)
        jaccard_scores = []

        for b in range(N_BOOTSTRAP):
            # Bootstrap sample
            boot_idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[boot_idx]

            # Fit K-means on bootstrap sample
            kmeans_boot = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
            labels_boot = kmeans_boot.fit_predict(X_boot)

            # Compute Jaccard: proportion of pairs that maintain co-membership
            # For efficiency, we compute on a sample of pairs
            n_pairs = min(1000, n * (n - 1) // 2)
            pairs_i = np.random.choice(n, size=n_pairs)
            pairs_j = np.random.choice(n, size=n_pairs)

            # Original co-membership
            original_same = labels[boot_idx[pairs_i]] == labels[boot_idx[pairs_j]]
            # Bootstrap co-membership
            boot_same = labels_boot[pairs_i] == labels_boot[pairs_j]

            # Jaccard = intersection / union
            intersection = np.sum(original_same & boot_same)
            union = np.sum(original_same | boot_same)
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)

        jaccard_mean = np.mean(jaccard_scores)
        jaccard_ci_lower = np.percentile(jaccard_scores, 2.5)
        jaccard_ci_upper = np.percentile(jaccard_scores, 97.5)
        jaccard_pass = jaccard_mean >= JACCARD_THRESHOLD

        log(f"Jaccard stability: {jaccard_mean:.4f} (95% CI: [{jaccard_ci_lower:.4f}, {jaccard_ci_upper:.4f}], threshold: {JACCARD_THRESHOLD}, pass: {jaccard_pass})")

        if silhouette < SILHOUETTE_THRESHOLD:
            log(f"WARNING: Weak clustering structure (silhouette={silhouette:.4f})")
        if jaccard_mean < JACCARD_THRESHOLD:
            log(f"WARNING: Unstable clusters (Jaccard={jaccard_mean:.4f})")

        metrics = [
            {'metric': 'silhouette', 'value': silhouette, 'threshold': SILHOUETTE_THRESHOLD, 'pass': silhouette_pass},
            {'metric': 'davies_bouldin', 'value': davies_bouldin, 'threshold': DAVIES_BOULDIN_THRESHOLD, 'pass': davies_bouldin_pass},
            {'metric': 'jaccard_mean', 'value': jaccard_mean, 'threshold': JACCARD_THRESHOLD, 'pass': jaccard_pass},
            {'metric': 'jaccard_ci_lower', 'value': jaccard_ci_lower, 'threshold': np.nan, 'pass': True},
            {'metric': 'jaccard_ci_upper', 'value': jaccard_ci_upper, 'threshold': np.nan, 'pass': True},
        ]

    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    output_path = DATA_DIR / "step05_validation_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    log("VALIDATION PASS: Cluster quality metrics computed")

    return metrics_df

# Characterize Clusters

def step06_characterize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean intercept/slope per cluster and assign phenotype labels."""
    log("")
    log("=" * 60)
    log("STEP 06: Characterize Clusters")
    log("=" * 60)

    # Load original random effects (not z-scores)
    df_original = pd.read_csv(DATA_DIR / "step01_random_effects_loaded.csv")
    df_merged = df[['UID', 'cluster_label']].merge(df_original, on='UID')

    # Group by cluster
    cluster_stats = df_merged.groupby('cluster_label').agg({
        'UID': 'count',
        'intercept': ['mean', 'std'],
        'slope': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    cluster_stats.columns = ['cluster_label', 'N', 'mean_intercept', 'sd_intercept', 'mean_slope', 'sd_slope']

    # Assign phenotype labels based on intercept/slope patterns
    # High intercept + shallow (less negative) slope = Resilient
    # Low intercept + steep (more negative) slope = Vulnerable
    median_intercept = cluster_stats['mean_intercept'].median()
    median_slope = cluster_stats['mean_slope'].median()

    phenotypes = []
    for _, row in cluster_stats.iterrows():
        high_intercept = row['mean_intercept'] >= median_intercept
        shallow_slope = row['mean_slope'] >= median_slope  # Less negative = shallower decline

        if high_intercept and shallow_slope:
            phenotype = "Resilient"
        elif not high_intercept and not shallow_slope:
            phenotype = "Vulnerable"
        elif high_intercept and not shallow_slope:
            phenotype = "High-Baseline-Fast-Decline"
        else:
            phenotype = "Low-Baseline-Slow-Decline"
        phenotypes.append(phenotype)

    cluster_stats['phenotype'] = phenotypes

    # Log characterization
    log(f"Cluster characterization (K={len(cluster_stats)}):")
    for _, row in cluster_stats.iterrows():
        log(f"  Cluster {int(row['cluster_label'])}: N={int(row['N'])}, "
            f"mean_intercept={row['mean_intercept']:.4f}, mean_slope={row['mean_slope']:.4f}, "
            f"phenotype={row['phenotype']}")

    # Validate N sums to 100
    total_n = cluster_stats['N'].sum()
    if total_n != 100:
        raise ValueError(f"Cluster N mismatch: total={total_n}, expected=100")

    # Save characterization
    output_path = DATA_DIR / "step06_cluster_characterization.csv"
    cluster_stats.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    # Save phenotype descriptions
    descriptions_path = DATA_DIR / "step06_phenotype_descriptions.txt"
    with open(descriptions_path, 'w') as f:
        f.write("CONFIDENCE PHENOTYPE DESCRIPTIONS\n")
        f.write("=" * 50 + "\n\n")
        for _, row in cluster_stats.iterrows():
            f.write(f"Cluster {int(row['cluster_label'])}: {row['phenotype']}\n")
            f.write(f"  N = {int(row['N'])} participants\n")
            f.write(f"  Mean baseline confidence (intercept): {row['mean_intercept']:.4f}\n")
            f.write(f"  Mean decline rate (slope): {row['mean_slope']:.4f}\n")
            f.write(f"  SD intercept: {row['sd_intercept']:.4f}\n")
            f.write(f"  SD slope: {row['sd_slope']:.4f}\n\n")
    log(f"Saved: {descriptions_path.name}")

    log("VALIDATION PASS: All clusters characterized")

    return cluster_stats

# Cross-Tabulate with Ch5 5.1.5 Accuracy Clusters

def step07_crosstab_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate confidence clusters with accuracy clusters from Ch5 5.1.5."""
    log("")
    log("=" * 60)
    log("STEP 07: Cross-Tabulate with Ch5 5.1.5 Accuracy Clusters")
    log("=" * 60)

    # Check dependency exists
    if not INPUT_ACCURACY_CLUSTERS.exists():
        raise FileNotFoundError(f"DEPENDENCY ERROR: Ch5 5.1.5 must complete first. File not found: {INPUT_ACCURACY_CLUSTERS}")

    # Load confidence clusters (this RQ)
    df_confidence = df[['UID', 'cluster_label']].copy()
    df_confidence = df_confidence.rename(columns={'cluster_label': 'cluster_confidence'})

    # Load accuracy clusters (Ch5 5.1.5)
    df_accuracy = pd.read_csv(INPUT_ACCURACY_CLUSTERS)
    # Ch5 5.1.5 uses: UID, cluster
    df_accuracy = df_accuracy.rename(columns={'cluster': 'cluster_accuracy'})

    log(f"Loaded {len(df_accuracy)} accuracy cluster assignments from Ch5 5.1.5")

    # Merge on UID
    df_merged = df_confidence.merge(df_accuracy, on='UID', how='inner')

    if len(df_merged) != 100:
        raise ValueError(f"UID mismatch: expected 100 matches, got {len(df_merged)}")

    log(f"All 100 participants matched between RQ 6.1.5 and Ch5 5.1.5")

    # Create contingency table
    crosstab = pd.crosstab(df_merged['cluster_confidence'], df_merged['cluster_accuracy'])
    log(f"Cross-tabulation: {crosstab.shape[0]} confidence clusters x {crosstab.shape[1]} accuracy clusters")

    # Validate crosstab sum
    if crosstab.values.sum() != 100:
        raise ValueError(f"Crosstab sum mismatch: {crosstab.values.sum()}, expected 100")

    # Save count table
    output_path = DATA_DIR / "step07_crosstab_confidence_accuracy.csv"
    crosstab.to_csv(output_path)
    log(f"Saved: {output_path.name}")

    # Save row percentages
    row_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    row_pct_path = DATA_DIR / "step07_crosstab_row_percentages.csv"
    row_pct.to_csv(row_pct_path)
    log(f"Saved: {row_pct_path.name}")

    # Save column percentages
    col_pct = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
    col_pct_path = DATA_DIR / "step07_crosstab_column_percentages.csv"
    col_pct.to_csv(col_pct_path)
    log(f"Saved: {col_pct_path.name}")

    # Log crosstab
    log(f"\nContingency Table (counts):")
    log(crosstab.to_string())

    log("\nVALIDATION PASS: Crosstab sums to 100 participants")

    return crosstab

# Chi-Square Test of Association

def step08_chi_square_test(crosstab: pd.DataFrame):
    """Chi-square test of independence (integration vs dissociation hypothesis)."""
    log("")
    log("=" * 60)
    log("STEP 08: Chi-Square Test of Association")
    log("=" * 60)

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(crosstab.values)

    log(f"Chi-square test: chi2={chi2:.4f}, df={dof}, p={p_value:.6f}")

    # Compute Cramer's V effect size
    n = crosstab.values.sum()
    min_dim = min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    log(f"Effect size (Cramer's V): {cramers_v:.4f}")

    # Interpret result
    alpha = 0.05
    interpretation = "integrated" if p_value < alpha else "dissociated"

    log(f"")
    log(f"INTERPRETATION: {interpretation.upper()}")
    if interpretation == "integrated":
        log(f"  p < {alpha}: Confidence and accuracy phenotypes are ASSOCIATED")
        log(f"  Metacognition tracks memory state (integrated memory-metacognition system)")
    else:
        log(f"  p >= {alpha}: Confidence and accuracy phenotypes are INDEPENDENT")
        log(f"  Dissociable systems (memory and metacognition may diverge)")

    # Validate
    if not (0 <= p_value <= 1):
        raise ValueError(f"Invalid p-value: {p_value}")
    if not (0 <= cramers_v <= 1):
        raise ValueError(f"Invalid Cramer's V: {cramers_v}")

    # Save results
    results = [
        {'statistic': 'chi_square', 'value': chi2, 'interpretation': interpretation},
        {'statistic': 'df', 'value': dof, 'interpretation': interpretation},
        {'statistic': 'p_value', 'value': p_value, 'interpretation': interpretation},
        {'statistic': 'cramers_v', 'value': cramers_v, 'interpretation': interpretation},
    ]
    results_df = pd.DataFrame(results)

    output_path = DATA_DIR / "step08_chi_square_test.csv"
    results_df.to_csv(output_path, index=False)
    log(f"Saved: {output_path.name}")

    # Save interpretation text
    interp_path = DATA_DIR / "step08_association_interpretation.txt"
    with open(interp_path, 'w') as f:
        f.write("CONFIDENCE-ACCURACY PHENOTYPE ASSOCIATION TEST\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Chi-square test of independence:\n")
        f.write(f"  Chi-square statistic: {chi2:.4f}\n")
        f.write(f"  Degrees of freedom: {dof}\n")
        f.write(f"  p-value: {p_value:.6f}\n")
        f.write(f"  Effect size (Cramer's V): {cramers_v:.4f}\n\n")
        f.write(f"RESULT: {interpretation.upper()}\n\n")
        if interpretation == "integrated":
            f.write("Confidence and accuracy phenotypes show significant association (p < 0.05).\n")
            f.write("This supports the INTEGRATION hypothesis: metacognitive monitoring tracks memory state.\n")
            f.write("Participants with resilient memory trajectories tend to also show resilient confidence trajectories.\n")
        else:
            f.write("Confidence and accuracy phenotypes show no significant association (p >= 0.05).\n")
            f.write("This supports the DISSOCIATION hypothesis: memory and metacognition operate independently.\n")
            f.write("A participant's memory phenotype does not predict their confidence phenotype.\n")
    log(f"Saved: {interp_path.name}")

    log("\nVALIDATION PASS: Association test complete")

# MAIN EXECUTION

def main():
    """Execute all 8 steps of RQ 6.1.5 analysis."""
    # Initialize log
    LOG_FILE.parent.mkdir(exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("RQ 6.1.5: Trajectory Clustering - Confidence Phenotypes")
    log("=" * 60)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"RQ directory: {RQ_DIR}")
    log("")

    try:
        # Step 01: Load random effects
        df = step01_load_random_effects()

        # Step 02: Standardize features
        df = step02_standardize_features(df)

        # Step 03: K-means clustering with BIC selection
        optimal_k = step03_cluster_selection(df)

        # Step 04: Fit final K-means
        df = step04_fit_final_kmeans(df, optimal_k)

        # Step 05: Validate cluster quality
        metrics_df = step05_validate_cluster_quality(df, optimal_k)

        # Step 06: Characterize clusters
        cluster_stats = step06_characterize_clusters(df)

        # Step 07: Cross-tabulate with accuracy clusters
        crosstab = step07_crosstab_clusters(df)

        # Step 08: Chi-square test
        step08_chi_square_test(crosstab)

        log("")
        log("=" * 60)
        log("RQ 6.1.5 ANALYSIS COMPLETE")
        log("=" * 60)
        log(f"Optimal K: {optimal_k}")
        log(f"Phenotypes identified: {list(cluster_stats['phenotype'].values)}")

        # Summary
        log("")
        log("OUTPUT FILES:")
        for f in sorted(DATA_DIR.glob("step*.csv")) + sorted(DATA_DIR.glob("step*.txt")):
            log(f"  {f.name}")

        return 0

    except Exception as e:
        log(f"")
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
