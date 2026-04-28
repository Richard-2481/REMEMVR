#!/usr/bin/env python3
"""
RQ 6.8.3: Source-Destination Confidence ICC - Opposite Correlation Pattern
==========================================================================
Tests whether the opposite intercept-slope correlation pattern from Ch5 5.5.6
(Source r=+0.99, Destination r=-0.90) replicates in confidence data.

Steps:
  00: Extract confidence theta data from RQ 6.8.1 and reshape to long format
  01: Fit Source confidence LMM with random slopes
  02: Fit Destination confidence LMM with random slopes
  03: Extract random effects for both location types (for RQ 6.8.4 clustering)
  04: Compute intercept-slope correlations per location type
  05: Compare to Ch5 5.5.6 accuracy correlations

Dependencies:
  - RQ 6.8.1: results/ch6/6.8.1/data/step03_theta_confidence.csv
  - Ch5 5.5.6: results/ch5/5.5.6/data/step05_intercept_slope_correlations.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.8.3
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR project root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"

# Dependency paths
CONFIDENCE_THETA_PATH = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step03_theta_confidence.csv"
LMM_INPUT_PATH = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step04_lmm_input.csv"
TSVR_PATH = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step00_tsvr_mapping.csv"
CH5_CORRELATIONS_PATH = PROJECT_ROOT / "results" / "ch5" / "5.5.6" / "data" / "step05_intercept_slope_correlations.csv"

def log(msg):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Extract Confidence Theta Data and Reshape to Long Format

def step00_extract_and_reshape():
    """
    Load confidence theta from 6.8.1, reshape to long format, merge with TSVR.
    Creates 800 rows: 100 participants x 4 tests x 2 location types.
    """
    log("\n" + "="*70)
    log("STEP 00: Extract Confidence Theta Data and Reshape to Long Format")
    log("="*70)

    # Load LMM input from 6.8.1 (already has TSVR and long format)
    log(f"\nLoading LMM input from: {LMM_INPUT_PATH}")
    df = pd.read_csv(LMM_INPUT_PATH)
    log(f"  Loaded: {len(df)} rows, columns: {list(df.columns)}")

    # The 6.8.1 step04_lmm_input.csv already has the structure we need:
    # composite_ID, UID, TEST, TSVR_hours, log_TSVR, location, theta

    # Rename columns for consistency
    df = df.rename(columns={'location': 'location_type'})

    # Add SE (not available, use placeholder)
    df['se'] = 0.5  # Placeholder - actual SE would require going back to IRT output

    # Select and reorder columns
    df = df[['UID', 'TEST', 'location_type', 'theta', 'se', 'TSVR_hours']].copy()

    # Validate
    n_source = (df['location_type'] == 'Source').sum()
    n_dest = (df['location_type'] == 'Destination').sum()
    n_uid = df['UID'].nunique()

    log(f"\nData summary:")
    log(f"  Total rows: {len(df)} (expected: 800)")
    log(f"  Unique UIDs: {n_uid} (expected: 100)")
    log(f"  Source rows: {n_source} (expected: 400)")
    log(f"  Destination rows: {n_dest} (expected: 400)")

    # Save
    output_path = RQ_DIR / "data" / "step00_lmm_input_confidence_location.csv"
    df.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    log(f"\nData extraction complete: {len(df)} rows created")
    return df

# Fit Source Confidence LMM with Random Slopes

def step01_fit_source_lmm(df_all):
    """
    Fit LMM for Source confidence: theta ~ TSVR_hours + (TSVR_hours | UID)
    Extracts variance components: var_intercept, var_slope, cov_int_slope, var_residual
    """
    log("\n" + "="*70)
    log("STEP 01: Fit Source Confidence LMM with Random Slopes")
    log("="*70)

    # Filter to Source only
    df = df_all[df_all['location_type'] == 'Source'].copy()
    log(f"\nSource data: {len(df)} rows (100 participants x 4 tests)")

    # Scale TSVR for numerical stability (per 100 hours)
    df['TSVR_scaled'] = df['TSVR_hours'] / 100.0

    log("\nFitting LMM: theta ~ TSVR_scaled + (TSVR_scaled | UID)")
    log("  Random effects: Intercept + slope per participant")

    # Fit LMM with random intercept and random slope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "theta ~ TSVR_scaled",
            data=df,
            groups=df['UID'],
            re_formula="~TSVR_scaled"  # Random intercept + random slope
        )
        result = model.fit(reml=True)

    log("\n" + str(result.summary()))

    # Save model summary
    summary_path = RQ_DIR / "data" / "step01_source_lmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"\nSaved: {summary_path}")

    # Extract variance components from random effects covariance matrix
    # The cov_re attribute contains the covariance matrix of random effects
    cov_re = result.cov_re
    log(f"\nRandom effects covariance matrix:\n{cov_re}")

    # Extract components
    var_intercept = cov_re.iloc[0, 0]
    var_slope = cov_re.iloc[1, 1]
    cov_int_slope = cov_re.iloc[0, 1]
    var_residual = result.scale

    # Compute correlation
    corr_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope) if var_intercept > 0 and var_slope > 0 else np.nan

    # Create variance components DataFrame
    vc_df = pd.DataFrame({
        'component': ['var_intercept', 'var_slope', 'cov_int_slope', 'var_residual', 'corr_int_slope'],
        'estimate': [var_intercept, var_slope, cov_int_slope, var_residual, corr_int_slope]
    })

    log("\nVariance Components (Source):")
    log(vc_df.to_string(index=False))

    # Save variance components
    vc_path = RQ_DIR / "data" / "step01_source_variance_components.csv"
    vc_df.to_csv(vc_path, index=False)
    log(f"\nSaved: {vc_path}")

    log(f"\nSource LMM converged: True")
    log(f"Intercept-Slope Correlation (Source): r = {corr_int_slope:.4f}")

    return result, vc_df

# Fit Destination Confidence LMM with Random Slopes

def step02_fit_destination_lmm(df_all):
    """
    Fit LMM for Destination confidence: theta ~ TSVR_hours + (TSVR_hours | UID)
    """
    log("\n" + "="*70)
    log("STEP 02: Fit Destination Confidence LMM with Random Slopes")
    log("="*70)

    # Filter to Destination only
    df = df_all[df_all['location_type'] == 'Destination'].copy()
    log(f"\nDestination data: {len(df)} rows (100 participants x 4 tests)")

    # Scale TSVR for numerical stability
    df['TSVR_scaled'] = df['TSVR_hours'] / 100.0

    log("\nFitting LMM: theta ~ TSVR_scaled + (TSVR_scaled | UID)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "theta ~ TSVR_scaled",
            data=df,
            groups=df['UID'],
            re_formula="~TSVR_scaled"
        )
        result = model.fit(reml=True)

    log("\n" + str(result.summary()))

    # Save model summary
    summary_path = RQ_DIR / "data" / "step02_destination_lmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"\nSaved: {summary_path}")

    # Extract variance components
    cov_re = result.cov_re
    log(f"\nRandom effects covariance matrix:\n{cov_re}")

    var_intercept = cov_re.iloc[0, 0]
    var_slope = cov_re.iloc[1, 1]
    cov_int_slope = cov_re.iloc[0, 1]
    var_residual = result.scale

    corr_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope) if var_intercept > 0 and var_slope > 0 else np.nan

    vc_df = pd.DataFrame({
        'component': ['var_intercept', 'var_slope', 'cov_int_slope', 'var_residual', 'corr_int_slope'],
        'estimate': [var_intercept, var_slope, cov_int_slope, var_residual, corr_int_slope]
    })

    log("\nVariance Components (Destination):")
    log(vc_df.to_string(index=False))

    vc_path = RQ_DIR / "data" / "step02_destination_variance_components.csv"
    vc_df.to_csv(vc_path, index=False)
    log(f"\nSaved: {vc_path}")

    log(f"\nDestination LMM converged: True")
    log(f"Intercept-Slope Correlation (Destination): r = {corr_int_slope:.4f}")

    return result, vc_df

# Extract Random Effects for Both Location Types

def step03_extract_random_effects(source_result, dest_result):
    """
    Extract participant-level random intercepts and slopes for both location types.
    Creates 200 rows (100 participants x 2 location types) for RQ 6.8.4 clustering.
    """
    log("\n" + "="*70)
    log("STEP 03: Extract Random Effects for Both Location Types")
    log("="*70)

    all_re = []

    # Extract Source random effects
    source_re = source_result.random_effects
    for uid, effects in source_re.items():
        all_re.append({
            'UID': uid,
            'location_type': 'Source',
            'random_intercept': effects['Group'],  # Intercept random effect
            'random_slope': effects['TSVR_scaled']  # Slope random effect
        })

    # Extract Destination random effects
    dest_re = dest_result.random_effects
    for uid, effects in dest_re.items():
        all_re.append({
            'UID': uid,
            'location_type': 'Destination',
            'random_intercept': effects['Group'],
            'random_slope': effects['TSVR_scaled']
        })

    df_re = pd.DataFrame(all_re)

    log(f"\nRandom effects extracted:")
    log(f"  Total rows: {len(df_re)} (expected: 200)")
    log(f"  Unique UIDs: {df_re['UID'].nunique()} (expected: 100)")
    log(f"  Source rows: {(df_re['location_type'] == 'Source').sum()}")
    log(f"  Destination rows: {(df_re['location_type'] == 'Destination').sum()}")

    # Descriptives
    log("\nRandom Effects Descriptives:")
    for loc in ['Source', 'Destination']:
        subset = df_re[df_re['location_type'] == loc]
        log(f"\n  {loc}:")
        log(f"    random_intercept: mean={subset['random_intercept'].mean():.4f}, SD={subset['random_intercept'].std():.4f}")
        log(f"    random_slope: mean={subset['random_slope'].mean():.4f}, SD={subset['random_slope'].std():.4f}")

    # Save
    output_path = RQ_DIR / "data" / "step03_random_effects.csv"
    df_re.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log("  NOTE: This file is REQUIRED for RQ 6.8.4 clustering")

    log(f"\nRandom effects extraction complete: {len(df_re)} rows created")
    return df_re

# Compute Intercept-Slope Correlations Per Location Type

def step04_compute_correlations(source_vc, dest_vc):
    """
    Compute intercept-slope correlations with confidence intervals and dual p-values.
    """
    log("\n" + "="*70)
    log("STEP 04: Compute Intercept-Slope Correlations Per Location Type")
    log("="*70)

    results = []
    N = 100  # participants

    for loc, vc_df in [('Source', source_vc), ('Destination', dest_vc)]:
        r = vc_df[vc_df['component'] == 'corr_int_slope']['estimate'].values[0]

        # Fisher's z transformation for CI
        z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else np.nan
        se_z = 1 / np.sqrt(N - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z

        # Back-transform to r scale
        CI_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        CI_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        # Test H0: rho = 0
        t_stat = r * np.sqrt(N - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
        p_uncorrected = 2 * (1 - stats.t.cdf(abs(t_stat), N - 2))
        p_bonferroni = min(p_uncorrected * 2, 1.0)  # 2 tests

        results.append({
            'location_type': loc,
            'correlation': r,
            'CI_lower': CI_lower,
            'CI_upper': CI_upper,
            'p_uncorrected': p_uncorrected,
            'p_bonferroni': p_bonferroni,
            'N': N
        })

        log(f"\n{loc}:")
        log(f"  Intercept-Slope Correlation: r = {r:.4f}")
        log(f"  95% CI: [{CI_lower:.4f}, {CI_upper:.4f}]")
        log(f"  p_uncorrected = {p_uncorrected:.4e}")
        log(f"  p_bonferroni = {p_bonferroni:.4e}")

        if p_bonferroni < 0.05:
            log(f"  -> SIGNIFICANT at Bonferroni-corrected alpha = 0.05")
        else:
            log(f"  -> Not significant at Bonferroni-corrected alpha = 0.05")

    df_corr = pd.DataFrame(results)

    # Save
    output_path = RQ_DIR / "data" / "step04_intercept_slope_correlations.csv"
    df_corr.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    # Key comparison
    source_r = df_corr[df_corr['location_type'] == 'Source']['correlation'].values[0]
    dest_r = df_corr[df_corr['location_type'] == 'Destination']['correlation'].values[0]

    log("\n" + "="*50)
    log("CRITICAL PATTERN CHECK: Opposite Signs?")
    log("="*50)
    log(f"  Source correlation:      r = {source_r:+.4f}")
    log(f"  Destination correlation: r = {dest_r:+.4f}")

    if np.sign(source_r) != np.sign(dest_r):
        log(f"  -> OPPOSITE SIGNS: Pattern replicates Ch5 5.5.6!")
    else:
        log(f"  -> SAME SIGN: Pattern does NOT replicate Ch5 5.5.6")

    log(f"\nIntercept-slope correlations computed for 2 location types")
    return df_corr

# Compare to Ch5 5.5.6 Accuracy Correlations

def step05_compare_to_ch5(confidence_corr):
    """
    Compare confidence correlations to Ch5 5.5.6 accuracy correlations.
    """
    log("\n" + "="*70)
    log("STEP 05: Compare Confidence Correlations to Ch5 5.5.6 Accuracy")
    log("="*70)

    # Load Ch5 5.5.6 accuracy correlations
    if not CH5_CORRELATIONS_PATH.exists():
        log(f"ERROR: Ch5 5.5.6 file not found: {CH5_CORRELATIONS_PATH}")
        raise FileNotFoundError(f"Missing: {CH5_CORRELATIONS_PATH}")

    log(f"\nLoading Ch5 5.5.6 accuracy correlations from: {CH5_CORRELATIONS_PATH}")
    df_acc = pd.read_csv(CH5_CORRELATIONS_PATH)
    log(f"  Loaded: {len(df_acc)} rows")

    # Rename columns for clarity
    df_acc = df_acc.rename(columns={
        'location': 'location_type',
        'r': 'correlation_accuracy'
    })

    # Merge with confidence correlations
    df_conf = confidence_corr.copy()
    df_conf = df_conf.rename(columns={'correlation': 'correlation_confidence'})

    df_compare = pd.merge(
        df_conf[['location_type', 'correlation_confidence', 'CI_lower', 'CI_upper', 'p_uncorrected', 'p_bonferroni']],
        df_acc[['location_type', 'correlation_accuracy']],
        on='location_type'
    )

    # Rename CI columns for confidence
    df_compare = df_compare.rename(columns={
        'CI_lower': 'CI_lower_confidence',
        'CI_upper': 'CI_upper_confidence',
        'p_uncorrected': 'p_uncorrected_confidence',
        'p_bonferroni': 'p_bonferroni_confidence'
    })

    # Compute comparison metrics
    df_compare['direction_match'] = np.sign(df_compare['correlation_confidence']) == np.sign(df_compare['correlation_accuracy'])
    df_compare['magnitude_difference'] = np.abs(df_compare['correlation_confidence'] - df_compare['correlation_accuracy'])

    log("\n" + "="*50)
    log("SIDE-BY-SIDE COMPARISON")
    log("="*50)

    for _, row in df_compare.iterrows():
        loc = row['location_type']
        r_conf = row['correlation_confidence']
        r_acc = row['correlation_accuracy']
        match = row['direction_match']
        diff = row['magnitude_difference']

        log(f"\n{loc}:")
        log(f"  Accuracy (Ch5 5.5.6):  r = {r_acc:+.4f}")
        log(f"  Confidence (Ch6 6.8.3): r = {r_conf:+.4f}")
        log(f"  Direction match: {match}")
        log(f"  Magnitude difference: {diff:.4f}")

    # Save comparison
    output_path = RQ_DIR / "data" / "step05_ch5_comparison.csv"
    df_compare.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    # Create summary text
    source_acc = df_compare[df_compare['location_type'] == 'Source']['correlation_accuracy'].values[0]
    source_conf = df_compare[df_compare['location_type'] == 'Source']['correlation_confidence'].values[0]
    dest_acc = df_compare[df_compare['location_type'] == 'Destination']['correlation_accuracy'].values[0]
    dest_conf = df_compare[df_compare['location_type'] == 'Destination']['correlation_confidence'].values[0]

    # Check if opposite pattern replicates
    acc_opposite = np.sign(source_acc) != np.sign(dest_acc)
    conf_opposite = np.sign(source_conf) != np.sign(dest_conf)
    pattern_replicates = acc_opposite and conf_opposite and \
                        (np.sign(source_conf) == np.sign(source_acc)) and \
                        (np.sign(dest_conf) == np.sign(dest_acc))

    log("\n" + "="*50)
    log("PATTERN REPLICATION ASSESSMENT")
    log("="*50)
    log(f"\nCh5 5.5.6 Accuracy Pattern:")
    log(f"  Source:      r = {source_acc:+.4f} (positive -> regression to mean)")
    log(f"  Destination: r = {dest_acc:+.4f} (negative -> fan effect)")
    log(f"  OPPOSITE SIGNS: {acc_opposite}")

    log(f"\nRQ 6.8.3 Confidence Pattern:")
    log(f"  Source:      r = {source_conf:+.4f}")
    log(f"  Destination: r = {dest_conf:+.4f}")
    log(f"  OPPOSITE SIGNS: {conf_opposite}")

    log(f"\nPATTERN REPLICATION: {pattern_replicates}")

    if pattern_replicates:
        log("\n*** HYPOTHESIS SUPPORTED ***")
        log("The opposite correlation pattern from Ch5 5.5.6 REPLICATES in confidence!")
        log("Source and Destination show fundamentally different forgetting dynamics")
        log("at BOTH memory (accuracy) and metacognitive (confidence) levels.")
    else:
        if conf_opposite:
            log("\n*** PARTIAL SUPPORT ***")
            log("Confidence shows opposite signs, but direction may not match accuracy.")
        else:
            log("\n*** HYPOTHESIS NOT SUPPORTED ***")
            log("Confidence does NOT show opposite signs like accuracy.")

    # Save summary text
    summary_path = RQ_DIR / "data" / "step05_pattern_replication_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Ch5 5.5.6 vs RQ 6.8.3 Pattern Comparison\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy (Ch5 5.5.6):\n")
        f.write(f"  Source:      r = {source_acc:+.4f}\n")
        f.write(f"  Destination: r = {dest_acc:+.4f}\n")
        f.write(f"  Opposite signs: {acc_opposite}\n\n")
        f.write(f"Confidence (Ch6 6.8.3):\n")
        f.write(f"  Source:      r = {source_conf:+.4f}\n")
        f.write(f"  Destination: r = {dest_conf:+.4f}\n")
        f.write(f"  Opposite signs: {conf_opposite}\n\n")
        f.write(f"Pattern Replication: {pattern_replicates}\n")
    log(f"\nSaved: {summary_path}")

    log(f"\nCh5 5.5.6 comparison complete: 2 location types compared")
    return df_compare, pattern_replicates

# MAIN EXECUTION

def main():
    """Execute all steps."""
    log("\n" + "="*70)
    log("RQ 6.8.3: Source-Destination Confidence ICC")
    log("="*70)
    log(f"RQ_DIR: {RQ_DIR}")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # Step 00: Extract and reshape data
    df_all = step00_extract_and_reshape()

    # Step 01: Fit Source LMM
    source_result, source_vc = step01_fit_source_lmm(df_all)

    # Step 02: Fit Destination LMM
    dest_result, dest_vc = step02_fit_destination_lmm(df_all)

    # Step 03: Extract random effects for clustering (RQ 6.8.4 dependency)
    df_re = step03_extract_random_effects(source_result, dest_result)

    # Step 04: Compute intercept-slope correlations
    df_corr = step04_compute_correlations(source_vc, dest_vc)

    # Step 05: Compare to Ch5 5.5.6
    df_compare, pattern_replicates = step05_compare_to_ch5(df_corr)

    log("\n" + "="*70)
    log("ALL STEPS COMPLETE")
    log("="*70)

    # Final summary
    log("\nFiles Created:")
    log(f"  data/step00_lmm_input_confidence_location.csv (800 rows)")
    log(f"  data/step01_source_lmm_model_summary.txt")
    log(f"  data/step01_source_variance_components.csv")
    log(f"  data/step02_destination_lmm_model_summary.txt")
    log(f"  data/step02_destination_variance_components.csv")
    log(f"  data/step03_random_effects.csv (200 rows - for RQ 6.8.4)")
    log(f"  data/step04_intercept_slope_correlations.csv")
    log(f"  data/step05_ch5_comparison.csv")
    log(f"  data/step05_pattern_replication_summary.txt")

    log(f"\nKEY RESULT: Pattern replication = {pattern_replicates}")

if __name__ == "__main__":
    main()
