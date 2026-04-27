#!/usr/bin/env python3
"""
RQ 6.8.4: Source-Destination Confidence Clustering
===================================================
K-means clustering on source-destination confidence random effects.
Tests whether exceptional clustering quality from Ch5 5.5.7 (Silhouette=0.417) replicates.

Steps:
  00: Reshape random effects (200 rows -> 100 rows x 4 features)
  01: Standardize features (z-scores)
  02: K-means cluster selection (BIC for K=1-6)
  03: Fit final K-means with optimal K
  04: Validate clustering quality (Silhouette, Davies-Bouldin, Jaccard)
  05: Characterize clusters (phenotypes)
  06: Cross-tabulate with Ch5 5.5.7 accuracy clusters
  07: Prepare visualization data (PCA projection)

Dependencies:
  - RQ 6.8.3: results/ch6/6.8.3/data/step03_random_effects.csv
  - Ch5 5.5.7: results/ch5/5.5.7/data/step04_cluster_assignments.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.8.4
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR project root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_07.log"

# Dependency paths
RANDOM_EFFECTS_PATH = PROJECT_ROOT / "results" / "ch6" / "6.8.3" / "data" / "step03_random_effects.csv"
CH5_CLUSTERS_PATH = PROJECT_ROOT / "results" / "ch5" / "5.5.7" / "data" / "step04_cluster_assignments.csv"
CH5_VALIDATION_PATH = PROJECT_ROOT / "results" / "ch5" / "5.5.7" / "data" / "step03_cluster_validation.csv"

RANDOM_STATE = 42

def log(msg):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# =============================================================================
# STEP 00: Reshape Random Effects Data (Long -> Wide)
# =============================================================================

def step00_reshape_random_effects():
    """
    Reshape random effects from 200 rows (100 UID x 2 locations) to
    100 rows x 4 features (Source_intercept, Source_slope, Dest_intercept, Dest_slope).
    """
    log("\n" + "="*70)
    log("STEP 00: Reshape Random Effects Data (Long -> Wide)")
    log("="*70)

    # Check dependency
    if not RANDOM_EFFECTS_PATH.exists():
        raise FileNotFoundError(f"Random effects file not found: {RANDOM_EFFECTS_PATH}")

    # Load random effects
    log(f"\nLoading random effects from: {RANDOM_EFFECTS_PATH}")
    df = pd.read_csv(RANDOM_EFFECTS_PATH)
    log(f"  Loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Pivot to wide format
    log("\nPivoting to wide format...")

    # Source random effects
    source = df[df['location_type'] == 'Source'][['UID', 'random_intercept', 'random_slope']].copy()
    source = source.rename(columns={
        'random_intercept': 'Source_intercept',
        'random_slope': 'Source_slope'
    })

    # Destination random effects
    dest = df[df['location_type'] == 'Destination'][['UID', 'random_intercept', 'random_slope']].copy()
    dest = dest.rename(columns={
        'random_intercept': 'Destination_intercept',
        'random_slope': 'Destination_slope'
    })

    # Merge
    df_wide = pd.merge(source, dest, on='UID', how='inner')

    log(f"\nReshaped {len(df)} rows -> {len(df_wide)} rows (2 locations -> 4 features per participant)")
    log(f"  Columns: {list(df_wide.columns)}")

    # Validate
    assert len(df_wide) == 100, f"Expected 100 rows, got {len(df_wide)}"
    assert df_wide.isna().sum().sum() == 0, "NaN values detected"

    # Save
    output_path = RQ_DIR / "data" / "step00_clustering_input.csv"
    df_wide.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    log(f"\nAll 100 participants matched between source and destination")
    return df_wide

# =============================================================================
# STEP 01: Standardize Features
# =============================================================================

def step01_standardize_features(df_raw):
    """
    Z-score standardize all 4 features for K-means clustering.
    """
    log("\n" + "="*70)
    log("STEP 01: Standardize Features")
    log("="*70)

    feature_cols = ['Source_intercept', 'Source_slope', 'Destination_intercept', 'Destination_slope']

    # Standardize
    scaler = StandardScaler()
    df_z = df_raw.copy()
    df_z[feature_cols] = scaler.fit_transform(df_raw[feature_cols])

    # Rename columns
    df_z = df_z.rename(columns={
        'Source_intercept': 'Source_intercept_z',
        'Source_slope': 'Source_slope_z',
        'Destination_intercept': 'Destination_intercept_z',
        'Destination_slope': 'Destination_slope_z'
    })

    # Verify standardization
    log("\nStandardization verification:")
    for col in df_z.columns:
        if col.endswith('_z'):
            mean_val = df_z[col].mean()
            std_val = df_z[col].std()
            log(f"  {col}: mean={mean_val:.4f}, SD={std_val:.4f}")

    # Save
    output_path = RQ_DIR / "data" / "step01_standardized_features.csv"
    df_z.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    log(f"\nStandardized 4 features: mean approximately 0, SD approximately 1")
    return df_z

# =============================================================================
# STEP 02: K-Means Cluster Selection (BIC)
# =============================================================================

def step02_cluster_selection(df_z):
    """
    Select optimal K using BIC for K=1 to K=6.
    """
    log("\n" + "="*70)
    log("STEP 02: K-Means Cluster Selection (BIC)")
    log("="*70)

    feature_cols = [c for c in df_z.columns if c.endswith('_z')]
    X = df_z[feature_cols].values
    N = len(X)
    n_features = len(feature_cols)

    results = []
    for k in range(1, 7):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        inertia = kmeans.inertia_

        # BIC = n * log(SSE/n) + k * n_features * log(n)
        # Where SSE = inertia (within-cluster sum of squares)
        bic = N * np.log(inertia / N) + k * n_features * np.log(N)

        results.append({
            'K': k,
            'inertia': inertia,
            'BIC': bic
        })

        log(f"  K={k}: inertia={inertia:.2f}, BIC={bic:.2f}")

    df_results = pd.DataFrame(results)

    # Find optimal K (minimum BIC)
    optimal_k = df_results.loc[df_results['BIC'].idxmin(), 'K']
    df_results['optimal'] = df_results['K'] == optimal_k

    log(f"\nBIC minimum at K = {optimal_k}")
    log(f"Optimal K selected: {optimal_k}")

    # Save
    output_path = RQ_DIR / "data" / "step02_cluster_selection.csv"
    df_results.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return int(optimal_k), df_results

# =============================================================================
# STEP 03: Fit Final K-Means Clustering
# =============================================================================

def step03_fit_final_kmeans(df_z, optimal_k):
    """
    Fit K-means with optimal K, extract cluster assignments.
    """
    log("\n" + "="*70)
    log(f"STEP 03: Fit Final K-Means Clustering (K={optimal_k})")
    log("="*70)

    feature_cols = [c for c in df_z.columns if c.endswith('_z')]
    X = df_z[feature_cols].values

    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)

    # Create assignments DataFrame
    df_assignments = pd.DataFrame({
        'UID': df_z['UID'],
        'cluster': labels
    })

    # Cluster sizes
    cluster_sizes = df_assignments['cluster'].value_counts().sort_index()
    log(f"\nCluster sizes:")
    for c, size in cluster_sizes.items():
        pct = size / len(df_assignments) * 100
        log(f"  Cluster {c}: {size} participants ({pct:.1f}%)")

    # Check minimum cluster size (10% of N=100)
    min_size = cluster_sizes.min()
    if min_size < 10:
        log(f"  WARNING: Cluster size {min_size} < 10 (10% threshold)")
    else:
        log(f"  All clusters >= 10% of N")

    # Save
    output_path = RQ_DIR / "data" / "step03_cluster_assignments.csv"
    df_assignments.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    log(f"\nFitted K-means with K = {optimal_k} clusters")
    return df_assignments, kmeans

# =============================================================================
# STEP 04: Validate Clustering Quality
# =============================================================================

def step04_validate_clustering_quality(df_z, df_assignments):
    """
    Compute clustering quality metrics: Silhouette, Davies-Bouldin, Jaccard bootstrap.
    """
    log("\n" + "="*70)
    log("STEP 04: Validate Clustering Quality")
    log("="*70)

    feature_cols = [c for c in df_z.columns if c.endswith('_z')]
    X = df_z[feature_cols].values
    labels = df_assignments['cluster'].values

    # Silhouette coefficient
    silhouette = silhouette_score(X, labels)

    # Davies-Bouldin index
    db_index = davies_bouldin_score(X, labels)

    # Jaccard bootstrap stability
    log("\nRunning Jaccard bootstrap stability (100 iterations)...")
    jaccard_scores = []
    n_bootstrap = 100
    n_samples = len(X)
    optimal_k = len(np.unique(labels))

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[idx]

        # Fit K-means on bootstrap
        kmeans_boot = KMeans(n_clusters=optimal_k, random_state=i, n_init=5)
        labels_boot = kmeans_boot.fit_predict(X_boot)

        # Compute Jaccard similarity between original and bootstrap labels
        # For each pair of points, check if they're in same cluster in both
        same_original = (labels[idx, None] == labels[idx]).flatten()
        same_boot = (labels_boot[:, None] == labels_boot).flatten()

        # Jaccard = intersection / union
        intersection = np.sum(same_original & same_boot)
        union = np.sum(same_original | same_boot)
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)

    jaccard_stability = np.mean(jaccard_scores)

    # Create validation DataFrame
    results = [
        {'metric': 'Silhouette', 'value': silhouette, 'threshold': 0.40, 'pass': silhouette >= 0.40},
        {'metric': 'Davies_Bouldin', 'value': db_index, 'threshold': 1.0, 'pass': db_index < 1.0},
        {'metric': 'Jaccard', 'value': jaccard_stability, 'threshold': 0.70, 'pass': jaccard_stability > 0.70}
    ]
    df_validation = pd.DataFrame(results)

    log("\nClustering Quality Metrics:")
    log(f"  Silhouette coefficient: {silhouette:.4f} (threshold >= 0.40) -> {'PASS' if silhouette >= 0.40 else 'FAIL'}")
    log(f"  Davies-Bouldin index: {db_index:.4f} (threshold < 1.0) -> {'PASS' if db_index < 1.0 else 'FAIL'}")
    log(f"  Jaccard stability: {jaccard_stability:.4f} (threshold > 0.70) -> {'PASS' if jaccard_stability > 0.70 else 'FAIL'}")

    # Compare to Ch5 5.5.7 threshold
    ch5_silhouette = 0.417
    log(f"\n  Ch5 5.5.7 Silhouette: {ch5_silhouette:.4f}")
    log(f"  This RQ Silhouette: {silhouette:.4f}")
    if silhouette >= 0.40:
        log(f"  -> THRESHOLD MET: Matches Ch5 5.5.7 quality!")
    else:
        log(f"  -> THRESHOLD NOT MET: Below Ch5 5.5.7 quality")

    # Save
    output_path = RQ_DIR / "data" / "step04_validation.csv"
    df_validation.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return df_validation, silhouette

# =============================================================================
# STEP 05: Characterize Clusters
# =============================================================================

def step05_characterize_clusters(df_raw, df_assignments):
    """
    Compute cluster phenotype characterizations.
    """
    log("\n" + "="*70)
    log("STEP 05: Characterize Clusters")
    log("="*70)

    # Merge raw features with cluster assignments
    df = pd.merge(df_raw, df_assignments, on='UID')

    feature_cols = ['Source_intercept', 'Source_slope', 'Destination_intercept', 'Destination_slope']

    results = []
    for cluster in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster]
        row = {'cluster': cluster, 'N': len(subset)}

        for col in feature_cols:
            row[f'{col}_mean'] = subset[col].mean()
            row[f'{col}_SD'] = subset[col].std()
            row[f'{col}_min'] = subset[col].min()
            row[f'{col}_max'] = subset[col].max()

        # Assign phenotype label based on mean values
        src_int = row['Source_intercept_mean']
        src_slope = row['Source_slope_mean']
        dst_int = row['Destination_intercept_mean']
        dst_slope = row['Destination_slope_mean']

        # Create phenotype description
        src_baseline = "High" if src_int > 0 else "Low"
        src_trend = "Resilient" if src_slope > 0 else "Declining"
        dst_baseline = "High" if dst_int > 0 else "Low"
        dst_trend = "Resilient" if dst_slope > 0 else "Declining"

        phenotype = f"{src_baseline}Src-{src_trend}, {dst_baseline}Dst-{dst_trend}"
        row['phenotype'] = phenotype

        results.append(row)

    df_char = pd.DataFrame(results)

    log("\nCluster Characterizations:")
    for _, row in df_char.iterrows():
        log(f"\n  Cluster {row['cluster']} (N={row['N']}):")
        log(f"    Source:      intercept={row['Source_intercept_mean']:.3f}, slope={row['Source_slope_mean']:.3f}")
        log(f"    Destination: intercept={row['Destination_intercept_mean']:.3f}, slope={row['Destination_slope_mean']:.3f}")
        log(f"    Phenotype: {row['phenotype']}")

    # Save
    output_path = RQ_DIR / "data" / "step05_cluster_characterization.csv"
    df_char.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    log(f"\nCharacterized {len(df_char)} clusters")
    return df_char

# =============================================================================
# STEP 06: Cross-Tabulate with Ch5 5.5.7 Accuracy Clusters
# =============================================================================

def step06_crosstab_ch5(df_assignments):
    """
    Cross-tabulate confidence clusters with Ch5 5.5.7 accuracy clusters.
    Chi-square test of association with dual p-values (D068).
    """
    log("\n" + "="*70)
    log("STEP 06: Cross-Tabulate with Ch5 5.5.7 Accuracy Clusters")
    log("="*70)

    # Load Ch5 5.5.7 accuracy clusters
    if not CH5_CLUSTERS_PATH.exists():
        log(f"WARNING: Ch5 5.5.7 file not found: {CH5_CLUSTERS_PATH}")
        log("Skipping cross-tabulation")
        return None, None

    log(f"\nLoading Ch5 5.5.7 accuracy clusters from: {CH5_CLUSTERS_PATH}")
    df_acc = pd.read_csv(CH5_CLUSTERS_PATH)
    df_acc = df_acc.rename(columns={'cluster': 'cluster_accuracy'})

    # Rename confidence clusters
    df_conf = df_assignments.rename(columns={'cluster': 'cluster_confidence'})

    # Merge
    df_merged = pd.merge(df_conf, df_acc, on='UID', how='inner')
    log(f"  Merged: {len(df_merged)} participants matched")

    # Cross-tabulation
    crosstab = pd.crosstab(df_merged['cluster_confidence'], df_merged['cluster_accuracy'], margins=True)
    log(f"\nCross-tabulation (Confidence x Accuracy):\n{crosstab}")

    # Chi-square test
    contingency = pd.crosstab(df_merged['cluster_confidence'], df_merged['cluster_accuracy'])
    chi2, p_uncorrected, dof, expected = stats.chi2_contingency(contingency)
    p_bonferroni = min(p_uncorrected * 1, 1.0)  # Only 1 test in this RQ

    log(f"\nChi-square test of association:")
    log(f"  X² = {chi2:.2f}")
    log(f"  df = {dof}")
    log(f"  p_uncorrected = {p_uncorrected:.4e}")
    log(f"  p_bonferroni = {p_bonferroni:.4e}")

    if p_uncorrected < 0.05:
        log(f"  -> SIGNIFICANT: Confidence and accuracy phenotypes are ASSOCIATED")
    else:
        log(f"  -> NOT SIGNIFICANT: No association between confidence and accuracy phenotypes")

    # Save crosstab
    crosstab_path = RQ_DIR / "data" / "step06_crosstab.csv"
    crosstab.to_csv(crosstab_path)
    log(f"\nSaved: {crosstab_path}")

    # Save chi-square results
    chi_df = pd.DataFrame([{
        'chi_square': chi2,
        'df': dof,
        'p_uncorrected': p_uncorrected,
        'p_bonferroni': p_bonferroni,
        'significant_uncorrected': p_uncorrected < 0.05,
        'significant_bonferroni': p_bonferroni < 0.05
    }])
    chi_path = RQ_DIR / "data" / "step06_chi_square.csv"
    chi_df.to_csv(chi_path, index=False)
    log(f"Saved: {chi_path}")

    log(f"\nChi-square test: X2 = {chi2:.2f}, df = {dof}, p = {p_uncorrected:.4e}")
    return crosstab, chi_df

# =============================================================================
# STEP 07: Prepare Visualization Data (PCA Projection)
# =============================================================================

def step07_prepare_visualization(df_z, df_assignments):
    """
    PCA projection to 2D for cluster visualization.
    """
    log("\n" + "="*70)
    log("STEP 07: Prepare Visualization Data (PCA Projection)")
    log("="*70)

    feature_cols = [c for c in df_z.columns if c.endswith('_z')]
    X = df_z[feature_cols].values

    # PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    # Variance explained
    var_explained = pca.explained_variance_ratio_ * 100
    log(f"\nPCA projection complete: 4D -> 2D")
    log(f"  PC1 explains {var_explained[0]:.1f}% variance")
    log(f"  PC2 explains {var_explained[1]:.1f}% variance")
    log(f"  Total: {sum(var_explained):.1f}%")

    # Create visualization DataFrame
    df_viz = pd.DataFrame({
        'UID': df_z['UID'],
        'PC1': pcs[:, 0],
        'PC2': pcs[:, 1],
        'cluster': df_assignments['cluster']
    })

    # Save
    output_path = RQ_DIR / "data" / "step07_cluster_scatter_data.csv"
    df_viz.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return df_viz, var_explained

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all steps."""
    log("\n" + "="*70)
    log("RQ 6.8.4: Source-Destination Confidence Clustering")
    log("="*70)
    log(f"RQ_DIR: {RQ_DIR}")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # Step 00: Reshape random effects
    df_raw = step00_reshape_random_effects()

    # Step 01: Standardize features
    df_z = step01_standardize_features(df_raw)

    # Step 02: Cluster selection (BIC)
    optimal_k, df_selection = step02_cluster_selection(df_z)

    # Step 03: Fit final K-means
    df_assignments, kmeans = step03_fit_final_kmeans(df_z, optimal_k)

    # Step 04: Validate clustering quality
    df_validation, silhouette = step04_validate_clustering_quality(df_z, df_assignments)

    # Step 05: Characterize clusters
    df_char = step05_characterize_clusters(df_raw, df_assignments)

    # Step 06: Cross-tabulate with Ch5 5.5.7
    crosstab, chi_df = step06_crosstab_ch5(df_assignments)

    # Step 07: Prepare visualization
    df_viz, var_explained = step07_prepare_visualization(df_z, df_assignments)

    log("\n" + "="*70)
    log("ALL STEPS COMPLETE")
    log("="*70)

    # Final summary
    log("\nFiles Created:")
    log(f"  data/step00_clustering_input.csv (100 rows x 4 features)")
    log(f"  data/step01_standardized_features.csv (100 rows x 4 z-scores)")
    log(f"  data/step02_cluster_selection.csv (BIC for K=1-6)")
    log(f"  data/step03_cluster_assignments.csv (100 UIDs x cluster)")
    log(f"  data/step04_validation.csv (quality metrics)")
    log(f"  data/step05_cluster_characterization.csv (phenotypes)")
    log(f"  data/step06_crosstab.csv (confidence x accuracy)")
    log(f"  data/step06_chi_square.csv (association test)")
    log(f"  data/step07_cluster_scatter_data.csv (PCA for plots)")

    log(f"\nKEY RESULTS:")
    log(f"  Optimal K: {optimal_k}")
    log(f"  Silhouette: {silhouette:.4f} (threshold >= 0.40)")
    log(f"  Ch5 5.5.7 Silhouette: 0.417")
    if silhouette >= 0.40:
        log(f"  -> HYPOTHESIS SUPPORTED: Exceptional clustering quality replicates in confidence!")
    else:
        log(f"  -> HYPOTHESIS NOT SUPPORTED: Clustering quality below threshold")

if __name__ == "__main__":
    main()
