#!/usr/bin/env python3
"""Correlation Analysis (IRT vs CTT per Domain): Compute Pearson correlations between IRT theta scores and CTT mean scores for"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def fisher_z_transform(r: float) -> float:
    """
    Fisher z-transformation for correlation coefficient.

    z = 0.5 * ln((1+r)/(1-r))
    """
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z: float) -> float:
    """
    Inverse Fisher z-transformation.

    r = (e^(2z) - 1) / (e^(2z) + 1)
    """
    e2z = np.exp(2 * z)
    return (e2z - 1) / (e2z + 1)

def compute_correlation_ci(r: float, n: int, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for Pearson correlation using Fisher z-transform.

    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    ci_level : float
        Confidence level (default 0.95)

    Returns
    -------
    Tuple[float, float]
        (CI_lower, CI_upper)
    """
    # Fisher z-transform
    z = fisher_z_transform(r)

    # Standard error in z-space
    se_z = 1 / np.sqrt(n - 3)

    # Z-score for CI level
    alpha = 1 - ci_level
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # CI in z-space
    z_lower = z - z_critical * se_z
    z_upper = z + z_critical * se_z

    # Transform back to r-space
    r_lower = inverse_fisher_z(z_lower)
    r_upper = inverse_fisher_z(z_upper)

    return r_lower, r_upper

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Holm-Bonferroni correction to p-values.

    Sequentially rejective procedure (less conservative than standard Bonferroni).

    Parameters
    ----------
    p_values : List[float]
        Uncorrected p-values
    alpha : float
        Family-wise error rate (default 0.05)

    Returns
    -------
    List[float]
        Corrected p-values (same length and order as input)
    """
    m = len(p_values)

    # Create list of (index, p_value) tuples
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]

    # Sort by p-value (ascending)
    indexed_pvals.sort(key=lambda x: x[1])

    # Compute corrected p-values
    corrected = [0.0] * m
    for k, (orig_idx, p) in enumerate(indexed_pvals):
        # Holm correction: min(1, p * (m - k))
        # where k is rank (0-indexed, so k=0 is smallest p-value)
        corrected[orig_idx] = min(1.0, p * (m - k))

    return corrected

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Correlation Analysis (IRT vs CTT per Domain)")
        # Load Input Data

        log("Loading input data...")

        # Load IRT theta scores (wide format: 3 domain columns)
        irt_theta = pd.read_csv(RQ_DIR / "data/step00_irt_theta_loaded.csv")
        log(f"step00_irt_theta_loaded.csv ({len(irt_theta)} rows, {len(irt_theta.columns)} cols)")

        # Load CTT scores (long format: 1 score column with domain indicator)
        ctt_scores = pd.read_csv(RQ_DIR / "data/step01_ctt_scores.csv")
        log(f"step01_ctt_scores.csv ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        # Reshape IRT Theta to Long Format (When EXCLUDED)
        #               to long format (domain column + IRT_score column)

        log("Reshaping IRT theta to long format (When excluded)...")

        # Reshape: composite_ID stays as identifier, melt domain columns (NO theta_when)
        irt_long = pd.melt(
            irt_theta,
            id_vars=['composite_ID'],
            value_vars=['theta_what', 'theta_where'],  # NO theta_when
            var_name='theta_column',
            value_name='IRT_score'
        )

        # Map theta column names to domain labels (lowercase to match CTT)
        domain_mapping = {
            'theta_what': 'what',
            'theta_where': 'where'
            # 'theta_when': 'when' - EXCLUDED
        }
        irt_long['domain'] = irt_long['theta_column'].map(domain_mapping)
        irt_long = irt_long.drop(columns=['theta_column'])

        log(f"IRT long format: {len(irt_long)} rows (When excluded)")
        # Merge IRT and CTT on composite_ID + domain

        log("Merging IRT and CTT scores...")

        # Merge on composite_ID and domain
        merged = pd.merge(
            irt_long,
            ctt_scores[['composite_ID', 'domain', 'CTT_score']],
            on=['composite_ID', 'domain'],
            how='inner'
        )

        log(f"{len(merged)} rows with both IRT and CTT scores")

        # Sanity check: Should have 800 rows (400 observations x 2 domains - When excluded)
        if len(merged) != 800:
            log(f"Expected 800 rows, got {len(merged)}")
        # Compute Pearson Correlations per Domain (When EXCLUDED)

        log("Computing Pearson correlations per domain (When excluded)...")

        correlations = []

        for domain in ['what', 'where']:  # NO 'when'
            # Filter to domain
            domain_data = merged[merged['domain'] == domain].copy()
            n = len(domain_data)

            # Compute Pearson correlation
            r, p_uncorrected = stats.pearsonr(
                domain_data['IRT_score'],
                domain_data['CTT_score']
            )

            # Compute 95% CI using Fisher z-transform
            ci_lower, ci_upper = compute_correlation_ci(r, n, ci_level=0.95)

            # Test thresholds
            threshold_0_70 = r > 0.70
            threshold_0_90 = r > 0.90

            correlations.append({
                'domain': domain,
                'r': r,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'n': n,
                'threshold_0.70': threshold_0_70,
                'threshold_0.90': threshold_0_90
            })

            log(f"{domain}: r={r:.3f}, p={p_uncorrected:.4f}, n={n}")
        # Compute Overall Correlation (All Domains Pooled - When excluded)

        log("Computing overall correlation (all domains pooled)...")

        r_overall, p_overall = stats.pearsonr(
            merged['IRT_score'],
            merged['CTT_score']
        )

        n_overall = len(merged)
        ci_lower_overall, ci_upper_overall = compute_correlation_ci(
            r_overall, n_overall, ci_level=0.95
        )

        correlations.append({
            'domain': 'Overall',
            'r': r_overall,
            'CI_lower': ci_lower_overall,
            'CI_upper': ci_upper_overall,
            'p_uncorrected': p_overall,
            'n': n_overall,
            'threshold_0.70': r_overall > 0.70,
            'threshold_0.90': r_overall > 0.90
        })

        log(f"Overall: r={r_overall:.3f}, p={p_overall:.4f}, n={n_overall}")
        # Apply Holm-Bonferroni Correction (m=3 tests - When excluded)

        log("Applying Holm-Bonferroni correction (m=3 tests - When excluded)...")

        # Convert to DataFrame for easier manipulation
        corr_df = pd.DataFrame(correlations)

        # Extract uncorrected p-values
        p_uncorrected_list = corr_df['p_uncorrected'].tolist()

        # Apply Holm-Bonferroni correction
        p_holm_list = holm_bonferroni_correction(p_uncorrected_list, alpha=0.05)

        # Add corrected p-values to DataFrame
        corr_df['p_holm'] = p_holm_list

        log("Holm-Bonferroni correction applied")

        # Report dual p-values (Decision D068)
        for _, row in corr_df.iterrows():
            log(f"[D068] {row['domain']}: p_uncorrected={row['p_uncorrected']:.4f}, p_holm={row['p_holm']:.4f}")
        # Save Correlation Results
        # These outputs will be used by: Step 7 (scatterplot data preparation)

        log("Saving correlation results...")

        # Reorder columns for clarity
        corr_df = corr_df[[
            'domain', 'r', 'CI_lower', 'CI_upper',
            'p_uncorrected', 'p_holm', 'n',
            'threshold_0.70', 'threshold_0.90'
        ]]

        # Save to data/ folder (per folder conventions, statistical outputs go to data/)
        output_path = RQ_DIR / "data/step02_correlations.csv"
        corr_df.to_csv(output_path, index=False, encoding='utf-8')

        log(f"step02_correlations.csv ({len(corr_df)} rows, {len(corr_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Decision D068 dual p-value reporting (uncorrected + corrected)
        # Threshold: D068 compliance required (both p_uncorrected and p_holm present)

        log("Running validate_correlation_test_d068...")

        validation_result = validate_correlation_test_d068(
            correlation_df=corr_df,
            required_cols=['p_uncorrected', 'p_holm']
        )

        # Report validation results
        log(f"valid: {validation_result['valid']}")
        log(f"d068_compliant: {validation_result['d068_compliant']}")
        log(f"message: {validation_result['message']}")

        # Check if validation passed
        if not validation_result['valid']:
            missing = validation_result.get('missing_cols', [])
            raise ValueError(f"Validation failed: {validation_result['message']} (missing: {missing})")

        # Additional validation criteria from 4_analysis.yaml
        log("Checking additional criteria...")

        # r values in [-1, 1]
        if not all(corr_df['r'].between(-1, 1)):
            raise ValueError("Correlation coefficients outside [-1, 1] bounds")
        log("r values in [-1, 1]: PASS")

        # CI_lower < r < CI_upper
        ci_check = all((corr_df['CI_lower'] < corr_df['r']) & (corr_df['r'] < corr_df['CI_upper']))
        if not ci_check:
            raise ValueError("Confidence intervals do not bracket point estimates")
        log("CI_lower < r < CI_upper: PASS")

        # p_holm >= p_uncorrected
        p_check = all(corr_df['p_holm'] >= corr_df['p_uncorrected'])
        if not p_check:
            raise ValueError("Corrected p-values smaller than uncorrected (impossible)")
        log("p_holm >= p_uncorrected: PASS")

        # Exactly 3 rows (When excluded)
        if len(corr_df) != 3:
            raise ValueError(f"Expected 3 rows (What, Where, Overall - When excluded), got {len(corr_df)}")
        log("Exactly 3 rows (When excluded): PASS")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
