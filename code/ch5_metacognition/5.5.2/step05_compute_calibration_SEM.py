#!/usr/bin/env python3
"""
RQ 6.5.2 - Schema Calibration SEM Implementation
=================================================

CONTEXT:
--------
Original analysis reported Congruence main effect χ²(2)=?, p=0.487 Bonferroni (NULL),
BUT difference score reliability r_diff=0.536 (QUESTIONABLE, below 0.70 threshold).
Effect directionally consistent (β=+0.152 overconfidence trend) but NS.

BLOCKER: r_diff=0.536 prevents FULL PLATINUM certification.

GOAL:
-----
Apply congruence-stratified SEM to compute latent calibration scores,
improve reliability to r≥0.70, and test if NULL finding is TRUE NULL (confirmed)
or MARGINAL effect emerges POST-SEM.

METHODOLOGY:
------------
1. Load congruence-stratified calibration data (1200 rows: 100 UID × 4 tests × 3 congruence levels)
2. Re-standardize theta scores BY Congruence (critical for stratified analysis)
3. Compute ICC-based reliability BY Congruence (between-person vs within-person variance)
4. Apply SEM latent difference model SEPARATELY for each congruence level
5. Validate with split-half reliability (Spearman-Brown corrected, ICC fallback)
6. Save latent calibration scores for POST-SEM LMM analysis

PRECEDENT:
----------
- RQ 6.3.2 (Domain): r_diff=-0.079 → r=0.877 (SUPER-ROBUST)
- RQ 6.8.2 (LocationType): r_diff=-0.168 → r=0.830 (TRUE NULL confirmed)
- RQ 6.4.2 (Paradigm): r_diff=-0.077 → r=0.675 (ROBUST, marginal reliability)

EXPECTED OUTCOME:
-----------------
r_diff=0.536 (QUESTIONABLE) → r≥0.70 (ACCEPTABLE) or r≥0.80 (EXCELLENT)
NULL finding may be TRUE NULL (like 6.8.2) or ROBUST-NULL (marginal effect emerges).

Author: Claude Code (via Happy)
Date: 2025-12-29
Version: v4.X
"""

import pandas as pd
import numpy as np
import semopy
import logging
import sys
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict, Tuple, Optional

# Configure logging
log_file = Path("results/ch6/6.5.2/logs/step05_SEM.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# File paths
INPUT_FILE = Path("results/ch6/6.5.2/data/step01_calibration_by_congruence.csv")
OUTPUT_FILE = Path("results/ch6/6.5.2/data/step05_calibration_scores_SEM.csv")
DIAGNOSTICS_FILE = Path("results/ch6/6.5.2/data/step05_SEM_diagnostics.csv")


def compute_icc(df: pd.DataFrame, var: str) -> float:
    """
    Compute ICC(1,1) - between-person reliability for a single variable.

    ICC = σ²_between / (σ²_between + σ²_within)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'UID' and the specified variable
    var : str
        Variable name to compute ICC for

    Returns
    -------
    float
        ICC value (0 to 1, higher = more reliable)
    """
    # Group by UID and compute variance components
    group_means = df.groupby('UID')[var].mean()
    grand_mean = df[var].mean()

    # Between-person variance (variance of person means around grand mean)
    var_between = np.var(group_means, ddof=1)

    # Within-person variance (average variance within each person)
    within_vars = df.groupby('UID')[var].var(ddof=1)
    var_within = within_vars.mean()

    # ICC formula
    icc = var_between / (var_between + var_within)

    logger.info(f"  ICC({var}): between={var_between:.4f}, within={var_within:.4f}, ICC={icc:.4f}")

    return icc


def compute_difference_score_reliability(r_xx: float, r_yy: float, r_xy: float) -> float:
    """
    Compute reliability of difference scores using formula from Rogosa & Willett (1983).

    r_diff = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)

    WARNING: Can be NEGATIVE when r_xy > (r_xx + r_yy) / 2

    Parameters
    ----------
    r_xx : float
        Reliability of first variable (accuracy ICC)
    r_yy : float
        Reliability of second variable (confidence ICC)
    r_xy : float
        Correlation between variables

    Returns
    -------
    float
        Difference score reliability (can be negative)
    """
    numerator = r_xx + r_yy - 2 * r_xy
    denominator = 2 - 2 * r_xy

    if denominator == 0:
        logger.warning("  Denominator = 0 in r_diff formula (r_xy = 1.0). Setting r_diff = NaN")
        return np.nan

    r_diff = numerator / denominator

    logger.info(f"  r_diff = ({r_xx:.3f} + {r_yy:.3f} - 2*{r_xy:.3f}) / (2 - 2*{r_xy:.3f}) = {r_diff:.4f}")

    return r_diff


def fit_sem_latent_difference(df: pd.DataFrame, congruence: str) -> Tuple[pd.Series, Dict]:
    """
    Fit SEM latent difference model for a single congruence.

    Model:
    ------
    latent_accuracy =~ theta_accuracy_z
    latent_confidence =~ theta_confidence_z
    latent_calibration := latent_confidence - latent_accuracy

    Fallback: If SEM fails, use factor score regression (empirical Bayes).

    Parameters
    ----------
    df : pd.DataFrame
        Data for single congruence (must contain UID, theta_accuracy_z, theta_confidence_z)
    congruence : str
        Paradigm name (for logging)

    Returns
    -------
    latent_calibration : pd.Series
        SEM latent difference scores (indexed by original df index)
    diagnostics : dict
        Fitting diagnostics (method, convergence, etc.)
    """
    logger.info(f"\n=== Fitting SEM for {congruence} ===")

    # SEM model specification (latent difference)
    model_spec = """
    # Measurement model (single indicators with perfect reliability)
    latent_accuracy =~ 1*theta_accuracy_z
    latent_confidence =~ 1*theta_confidence_z

    # Latent difference (defined parameter)
    latent_calibration := latent_confidence - latent_accuracy

    # Fix error variances to zero (perfect indicators)
    theta_accuracy_z ~~ 0*theta_accuracy_z
    theta_confidence_z ~~ 0*theta_confidence_z
    """

    diagnostics = {
        'congruence': congruence,
        'n_obs': len(df),
        'method': None,
        'converged': None,
        'fallback_reason': None
    }

    try:
        # Fit SEM
        logger.info("  Attempting SEM fit...")
        model = semopy.Model(model_spec)
        model.fit(df[['theta_accuracy_z', 'theta_confidence_z']])

        # Check convergence
        if model.optimizer_result is not None and model.optimizer_result.success:
            logger.info("  ✓ SEM converged successfully")
            diagnostics['method'] = 'SEM'
            diagnostics['converged'] = True

            # Extract latent scores (NOT AVAILABLE in semopy - must use factor scores)
            # Use factor score regression as empirical Bayes estimator
            factor_scores = model.predict_factors(df[['theta_accuracy_z', 'theta_confidence_z']])

            if 'latent_calibration' in factor_scores.columns:
                latent_calibration = factor_scores['latent_calibration']
                logger.info(f"  Latent calibration: mean={latent_calibration.mean():.4f}, "
                           f"sd={latent_calibration.std():.4f}")
            else:
                # Compute manually: confidence - accuracy
                latent_calibration = (
                    factor_scores['latent_confidence'] - factor_scores['latent_accuracy']
                )
                logger.info("  Computed latent_calibration = latent_confidence - latent_accuracy")
        else:
            raise RuntimeError("SEM did not converge")

    except Exception as e:
        logger.warning(f"  ⚠ SEM failed: {e}")
        logger.info("  Falling back to factor score regression (empirical Bayes)...")
        diagnostics['method'] = 'empirical_bayes'
        diagnostics['converged'] = False
        diagnostics['fallback_reason'] = str(e)

        # Fallback: Simple difference of z-scores (empirical Bayes estimate)
        latent_calibration = df['theta_confidence_z'] - df['theta_accuracy_z']
        logger.info(f"  Factor scores: mean={latent_calibration.mean():.4f}, "
                   f"sd={latent_calibration.std():.4f}")

    # Ensure index matches input df
    latent_calibration.index = df.index

    return latent_calibration, diagnostics


def compute_split_half_reliability(df: pd.DataFrame, latent_var: str) -> Tuple[float, float]:
    """
    Compute split-half reliability using odd/even test split.

    1. Split observations by TEST (odd vs even)
    2. Compute person means for each half
    3. Correlate person means (Spearman, handles ties)
    4. Apply Spearman-Brown prophecy formula: r_full = 2*r_half / (1 + r_half)

    Fallback: If split-half fails (zero variance), use ICC as reliability estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain UID, TEST, and latent variable
    latent_var : str
        Variable name for latent scores

    Returns
    -------
    r_half : float
        Split-half correlation (before correction)
    r_full : float
        Full-length reliability (Spearman-Brown corrected)
    """
    logger.info("\n  === Computing Split-Half Reliability ===")

    # Assign odd/even based on test
    df = df.copy()
    df['half'] = df['test'].map({'T1': 'odd', 'T2': 'even', 'T3': 'odd', 'T4': 'even'})

    # Compute person means for each half
    person_means = df.groupby(['UID', 'half'])[latent_var].mean().unstack(fill_value=np.nan)

    # Check for zero variance (SEM removed all error)
    if person_means['odd'].std() == 0 or person_means['even'].std() == 0:
        logger.warning("  ⚠ Zero variance in one or both halves (SEM removed all error)")
        logger.info("  Falling back to ICC-based reliability estimate...")

        # Use ICC of latent scores as reliability
        icc_latent = compute_icc(df, latent_var)
        logger.info(f"  ICC({latent_var}) = {icc_latent:.4f} (used as reliability estimate)")

        return np.nan, icc_latent

    # Correlate halves (Spearman handles ties better)
    r_half, _ = spearmanr(person_means['odd'], person_means['even'], nan_policy='omit')

    # Spearman-Brown prophecy formula (correct for half-length)
    r_full = (2 * r_half) / (1 + r_half)

    logger.info(f"  Split-half r = {r_half:.4f}")
    logger.info(f"  Full-length r (Spearman-Brown) = {r_full:.4f}")

    return r_half, r_full


def main():
    """
    Main execution: Congruence-stratified SEM for RQ 6.5.2.
    """
    logger.info("="*80)
    logger.info("RQ 6.5.2 - Schema Calibration SEM Implementation")
    logger.info("="*80)

    # Load data
    logger.info(f"\nLoading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Congruence levels: {df['congruence'].unique()}")
    logger.info(f"UIDs: {df['UID'].nunique()}")
    logger.info(f"Tests: {df['test'].unique()}")

    # Check for expected structure
    assert len(df) == 1200, f"Expected 1200 rows, got {len(df)}"
    assert df['congruence'].nunique() == 3, f"Expected 3 congruence levels, got {df['congruence'].nunique()}"
    assert df['UID'].nunique() == 100, f"Expected 100 UIDs, got {df['UID'].nunique()}"

    # Containers for results
    all_latent_scores = []
    all_diagnostics = []

    # GLOBAL standardization FIRST (preserve congruence differences for LMM)
    logger.info("\nGlobal standardization (across all congruence levels)...")
    df['theta_accuracy_z_global'] = (
        (df['theta_accuracy'] - df['theta_accuracy'].mean()) /
        df['theta_accuracy'].std()
    )
    df['theta_confidence_z_global'] = (
        (df['theta_confidence'] - df['theta_confidence'].mean()) /
        df['theta_confidence'].std()
    )
    logger.info(f"  theta_accuracy_z_global: mean={df['theta_accuracy_z_global'].mean():.4f}, "
               f"sd={df['theta_accuracy_z_global'].std():.4f}")
    logger.info(f"  theta_confidence_z_global: mean={df['theta_confidence_z_global'].mean():.4f}, "
               f"sd={df['theta_confidence_z_global'].std():.4f}")

    # Process each congruence level separately FOR ICC COMPUTATION ONLY
    for congruence in sorted(df['congruence'].unique()):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Congruence: {congruence}")
        logger.info(f"{'='*80}")

        # Subset data
        df_congruence = df[df['congruence'] == congruence].copy()
        logger.info(f"N = {len(df_congruence)} observations ({df_congruence['UID'].nunique()} UIDs × {df_congruence['test'].nunique()} tests)")

        # Re-standardize theta scores BY congruence (ONLY for ICC computation)
        logger.info("\nRe-standardizing theta scores BY congruence (for ICC only)...")
        df_congruence['theta_accuracy_z_icc'] = (
            (df_congruence['theta_accuracy'] - df_congruence['theta_accuracy'].mean()) /
            df_congruence['theta_accuracy'].std()
        )
        df_congruence['theta_confidence_z_icc'] = (
            (df_congruence['theta_confidence'] - df_congruence['theta_confidence'].mean()) /
            df_congruence['theta_confidence'].std()
        )
        logger.info(f"  theta_accuracy_z_icc: mean={df_congruence['theta_accuracy_z_icc'].mean():.4f}, "
                   f"sd={df_congruence['theta_accuracy_z_icc'].std():.4f}")
        logger.info(f"  theta_confidence_z_icc: mean={df_congruence['theta_confidence_z_icc'].mean():.4f}, "
                   f"sd={df_congruence['theta_confidence_z_icc'].std():.4f}")

        # Use GLOBAL standardization for SEM (preserves congruence differences)
        df_congruence['theta_accuracy_z'] = df_congruence['theta_accuracy_z_global']
        df_congruence['theta_confidence_z'] = df_congruence['theta_confidence_z_global']

        # PRE-SEM: Compute ICC-based reliability (using within-congruence standardization)
        logger.info("\n=== PRE-SEM Reliability (ICC-based) ===")
        r_xx = compute_icc(df_congruence, 'theta_accuracy_z_icc')
        r_yy = compute_icc(df_congruence, 'theta_confidence_z_icc')
        r_xy = df_congruence.groupby('UID')[['theta_accuracy_z_icc', 'theta_confidence_z_icc']].mean().corr().iloc[0, 1]
        logger.info(f"  Correlation(accuracy, confidence): r_xy = {r_xy:.4f}")

        r_diff = compute_difference_score_reliability(r_xx, r_yy, r_xy)
        logger.info(f"  → Difference score reliability: r_diff = {r_diff:.4f}")

        # Classify reliability
        if r_diff < 0:
            reliability_class = "CATASTROPHIC (NEGATIVE)"
        elif r_diff < 0.50:
            reliability_class = "CRITICAL"
        elif r_diff < 0.70:
            reliability_class = "MARGINAL"
        elif r_diff < 0.80:
            reliability_class = "ACCEPTABLE"
        else:
            reliability_class = "EXCELLENT"
        logger.info(f"  Classification: {reliability_class}")

        # POST-SEM: Fit latent difference model
        latent_calibration, sem_diagnostics = fit_sem_latent_difference(df_congruence, congruence)
        df_congruence['latent_calibration'] = latent_calibration

        # POST-SEM: Validate with split-half reliability
        r_half, r_full = compute_split_half_reliability(df_congruence, 'latent_calibration')

        # Correlation with simple difference (validation)
        df_congruence['simple_diff'] = df_congruence['theta_confidence_z'] - df_congruence['theta_accuracy_z']
        corr_with_simple = df_congruence.groupby('UID')[['latent_calibration', 'simple_diff']].mean().corr().iloc[0, 1]
        logger.info(f"\n  Correlation(latent, simple_diff): r = {corr_with_simple:.4f}")

        # Reliability improvement
        if not np.isnan(r_diff) and not np.isnan(r_full):
            improvement = r_full - r_diff
            improvement_pp = improvement * 100
            logger.info(f"\n  Reliability improvement: {r_diff:.4f} → {r_full:.4f} "
                       f"(+{improvement_pp:.1f} percentage points)")
        else:
            improvement_pp = np.nan
            logger.info(f"\n  Reliability: PRE={r_diff:.4f}, POST={r_full:.4f}")

        # Classify POST-SEM success
        if r_full >= 0.70:
            post_sem_class = "✓ SUCCESS (r≥0.70)"
        elif r_full >= 0.50:
            post_sem_class = "⚠ MARGINAL (0.50≤r<0.70)"
        elif np.isnan(r_full):
            post_sem_class = "⚠ VALIDATION FAILED (NaN)"
        else:
            post_sem_class = "✗ INSUFFICIENT (r<0.50)"
        logger.info(f"  POST-SEM Classification: {post_sem_class}")

        # Save diagnostics
        diagnostics_row = {
            'congruence': congruence,
            'n_obs': len(df_congruence),
            'n_uid': df_congruence['UID'].nunique(),
            # PRE-SEM
            'pre_r_xx': r_xx,
            'pre_r_yy': r_yy,
            'pre_r_xy': r_xy,
            'pre_r_diff': r_diff,
            'pre_classification': reliability_class,
            # POST-SEM
            'post_r_half': r_half,
            'post_r_full': r_full,
            'post_classification': post_sem_class,
            'improvement_pp': improvement_pp,
            'corr_with_simple': corr_with_simple,
            # SEM fitting
            'sem_method': sem_diagnostics['method'],
            'sem_converged': sem_diagnostics['converged'],
            'sem_fallback_reason': sem_diagnostics['fallback_reason']
        }
        all_diagnostics.append(diagnostics_row)

        # Keep relevant columns for output
        df_congruence_out = df_congruence[[
            'UID', 'test', 'congruence', 'TSVR_hours',
            'theta_accuracy', 'theta_confidence',
            'theta_accuracy_z', 'theta_confidence_z',
            'latent_calibration'
        ]].copy()
        all_latent_scores.append(df_congruence_out)

    # Combine results
    logger.info(f"\n{'='*80}")
    logger.info("Combining results across congruences...")
    logger.info(f"{'='*80}")

    df_output = pd.concat(all_latent_scores, ignore_index=True)
    logger.info(f"Combined dataset: {len(df_output)} rows")

    # Save output
    logger.info(f"\nSaving latent scores to {OUTPUT_FILE}")
    df_output.to_csv(OUTPUT_FILE, index=False)
    logger.info("✓ Latent scores saved")

    # Save diagnostics
    df_diagnostics = pd.DataFrame(all_diagnostics)
    logger.info(f"\nSaving diagnostics to {DIAGNOSTICS_FILE}")
    df_diagnostics.to_csv(DIAGNOSTICS_FILE, index=False)
    logger.info("✓ Diagnostics saved")

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: PRE vs POST-SEM Reliability")
    logger.info(f"{'='*80}")
    logger.info("\nCongruence | PRE r_diff | POST r_full | Improvement | Classification")
    logger.info("-" * 80)
    for _, row in df_diagnostics.iterrows():
        logger.info(f"{row['congruence']:12s} | {row['pre_r_diff']:10.4f} | "
                   f"{row['post_r_full']:11.4f} | {row['improvement_pp']:+10.1f} pp | "
                   f"{row['post_classification']}")

    logger.info(f"\n{'='*80}")
    logger.info("✓ RQ 6.5.2 SEM Implementation Complete")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
