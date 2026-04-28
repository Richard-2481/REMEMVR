#!/usr/bin/env python3
"""
Step 04: Compute 6 Planned Contrasts with Bonferroni Correction
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Test 6 planned contrasts comparing consolidation benefits
(Late slope - Early slope) across paradigms with Bonferroni correction
(alpha = 0.0083) and dual p-value reporting per Decision D068.
"""

import sys
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step04_compute_planned_contrasts.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_contrasts(result, slopes_df):
    """
    Compute planned contrasts for piecewise LMM.

    Contrasts:
    1-3: Consolidation benefit within each paradigm (Late slope - Early slope)
    4-6: Paradigm differences in consolidation benefit
    """
    # Get fixed effects and covariance matrix
    fe_params = result.fe_params
    fe_cov = result.cov_params()

    # Build slopes dictionary for reference
    slopes = {}
    for _, row in slopes_df.iterrows():
        key = (row['Segment'], row['paradigm'])
        slopes[key] = row['slope']

    # Define contrasts
    # Contrast 1-3: Consolidation benefit = Late slope - Early slope
    # (Positive means slower forgetting in Late = consolidation helped stabilize Early)
    # Actually, looking at results, Early has faster decay, so Late-Early > 0 means
    # Late is less steep (less negative) than Early

    contrasts_data = []

    # --- Contrast 1: IFR consolidation benefit ---
    # Late IFR - Early IFR
    early_ifr = slopes[('Early', 'IFR')]
    late_ifr = slopes[('Late', 'IFR')]
    estimate = late_ifr - early_ifr

    # The contrast is: (Days_within + Days_within:Segment[Late]) - Days_within
    #                = Days_within:Segment[Late]
    # So SE is just the SE of that coefficient
    contrast_coef = 'Days_within:Segment[T.Late]'
    se = np.sqrt(fe_cov.loc[contrast_coef, contrast_coef])
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'IFR_consolidation_benefit',
        'description': 'Late IFR slope - Early IFR slope',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # --- Contrast 2: ICR consolidation benefit ---
    # Late ICR - Early ICR
    early_icr = slopes[('Early', 'ICR')]
    late_icr = slopes[('Late', 'ICR')]
    estimate = late_icr - early_icr

    # For ICR, the contrast involves multiple coefficients
    # Late ICR slope = Days_within + Days_within:Segment[Late] + Days_within:paradigm+ interaction
    # Early ICR slope = Days_within + Days_within:paradigm# Difference = Days_within:Segment[Late] + Days_within:Segment[Late]:paradigmcoefs = ['Days_within:Segment[T.Late]', 'Days_within:Segment[T.Late]:paradigm_code[T.ICR]']
    var = sum(fe_cov.loc[c1, c2] for c1 in coefs for c2 in coefs if c1 in fe_cov.index and c2 in fe_cov.columns)
    se = np.sqrt(var)
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'ICR_consolidation_benefit',
        'description': 'Late ICR slope - Early ICR slope',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # --- Contrast 3: IRE consolidation benefit ---
    early_ire = slopes[('Early', 'IRE')]
    late_ire = slopes[('Late', 'IRE')]
    estimate = late_ire - early_ire

    coefs = ['Days_within:Segment[T.Late]', 'Days_within:Segment[T.Late]:paradigm_code[T.IRE]']
    var = sum(fe_cov.loc[c1, c2] for c1 in coefs for c2 in coefs if c1 in fe_cov.index and c2 in fe_cov.columns)
    se = np.sqrt(var)
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'IRE_consolidation_benefit',
        'description': 'Late IRE slope - Early IRE slope',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # --- Contrast 4-6: Paradigm differences in consolidation benefit ---
    # These test whether the consolidation benefit differs across paradigms

    # Contrast 4: IFR vs ICR benefit difference
    # (Late IFR - Early IFR) - (Late ICR - Early ICR)
    ifr_benefit = late_ifr - early_ifr
    icr_benefit = late_icr - early_icr
    estimate = ifr_benefit - icr_benefit

    # This equals -Days_within:Segment[Late]:paradigmcontrast_coef = 'Days_within:Segment[T.Late]:paradigm_code[T.ICR]'
    se = np.sqrt(fe_cov.loc[contrast_coef, contrast_coef])
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'IFR_vs_ICR_benefit_difference',
        'description': '(IFR consolidation benefit) - (ICR consolidation benefit)',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # Contrast 5: IFR vs IRE benefit difference
    ire_benefit = late_ire - early_ire
    estimate = ifr_benefit - ire_benefit

    contrast_coef = 'Days_within:Segment[T.Late]:paradigm_code[T.IRE]'
    se = np.sqrt(fe_cov.loc[contrast_coef, contrast_coef])
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'IFR_vs_IRE_benefit_difference',
        'description': '(IFR consolidation benefit) - (IRE consolidation benefit)',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # Contrast 6: ICR vs IRE benefit difference
    estimate = icr_benefit - ire_benefit

    # This is Days_within:Segment[Late]:paradigm- Days_within:Segment[Late]:paradigmcoefs = ['Days_within:Segment[T.Late]:paradigm_code[T.ICR]',
             'Days_within:Segment[T.Late]:paradigm_code[T.IRE]']
    # Var(A - B) = Var(A) + Var(B) - 2*Cov(A,B)
    var = (fe_cov.loc[coefs[0], coefs[0]] +
           fe_cov.loc[coefs[1], coefs[1]] -
           2 * fe_cov.loc[coefs[0], coefs[1]])
    se = np.sqrt(var)
    z = estimate / se
    p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

    contrasts_data.append({
        'contrast_name': 'ICR_vs_IRE_benefit_difference',
        'description': '(ICR consolidation benefit) - (IRE consolidation benefit)',
        'estimate': estimate,
        'SE': se,
        'z_statistic': z,
        'p_uncorrected': p_uncorrected
    })

    # Create DataFrame
    contrasts_df = pd.DataFrame(contrasts_data)

    # Add Bonferroni correction (6 comparisons)
    n_comparisons = 6
    alpha_bonferroni = 0.05 / n_comparisons  # 0.0083

    contrasts_df['p_bonferroni'] = np.minimum(contrasts_df['p_uncorrected'] * n_comparisons, 1.0)
    contrasts_df['alpha_bonferroni'] = alpha_bonferroni
    contrasts_df['significant'] = contrasts_df['p_bonferroni'] < alpha_bonferroni

    # Add 95% CIs
    contrasts_df['CI_lower'] = contrasts_df['estimate'] - 1.96 * contrasts_df['SE']
    contrasts_df['CI_upper'] = contrasts_df['estimate'] + 1.96 * contrasts_df['SE']

    return contrasts_df


def compute_effect_sizes(contrasts_df, slopes_df):
    """Compute Cohen's d effect sizes for contrasts."""
    effect_sizes = []

    # For consolidation benefit contrasts, use pooled SD of slopes
    # Rough approximation: average SE as proxy for within-group SD
    avg_se = slopes_df['SE'].mean()

    for _, row in contrasts_df.iterrows():
        # Cohen's d = estimate / SD
        # Using SE as proxy (this is an approximation)
        cohens_d = abs(row['estimate']) / row['SE']

        # Interpretation thresholds (Cohen, 1988)
        if cohens_d < 0.2:
            interp = "negligible"
        elif cohens_d < 0.5:
            interp = "small"
        elif cohens_d < 0.8:
            interp = "medium"
        else:
            interp = "large"

        effect_sizes.append({
            'contrast_name': row['contrast_name'],
            'effect_size': cohens_d,
            'effect_type': 'cohens_d_approx',
            'interpretation': interp
        })

    return pd.DataFrame(effect_sizes)


def main():
    """Compute planned contrasts with Bonferroni correction."""
    logger.info("=" * 60)
    logger.info("Step 04: Compute Planned Contrasts")
    logger.info("=" * 60)

    # Define paths
    model_file = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
    slopes_file = RQ_DIR / "data" / "step03_segment_paradigm_slopes.csv"
    contrasts_file = RQ_DIR / "data" / "step04_planned_contrasts.csv"
    effect_sizes_file = RQ_DIR / "data" / "step04_effect_sizes.csv"

    # --- Load model and slopes ---
    logger.info(f"Loading model from: {model_file}")
    with open(model_file, 'rb') as f:
        result = pickle.load(f)

    logger.info(f"Loading slopes from: {slopes_file}")
    slopes_df = pd.read_csv(slopes_file)

    # --- Compute contrasts ---
    logger.info("\nComputing 6 planned contrasts...")
    contrasts_df = compute_contrasts(result, slopes_df)

    # --- Display contrasts ---
    logger.info("\n" + "=" * 60)
    logger.info("PLANNED CONTRASTS")
    logger.info("=" * 60)

    display_cols = ['contrast_name', 'estimate', 'SE', 'z_statistic',
                    'p_uncorrected', 'p_bonferroni', 'significant']
    logger.info(f"\n{contrasts_df[display_cols].to_string(index=False)}")

    # --- Compute effect sizes ---
    logger.info("\nComputing effect sizes...")
    effect_sizes_df = compute_effect_sizes(contrasts_df, slopes_df)

    logger.info("\n" + "=" * 60)
    logger.info("EFFECT SIZES")
    logger.info("=" * 60)
    logger.info(f"\n{effect_sizes_df.to_string(index=False)}")

    # --- Validation ---
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check row count
    if len(contrasts_df) != 6:
        logger.error(f"CRITICAL: Expected 6 contrasts, got {len(contrasts_df)}")
        sys.exit(1)
    logger.info("VALIDATION - PASS: 6 contrasts computed")

    # Check dual p-values (Decision D068)
    if 'p_uncorrected' not in contrasts_df.columns:
        logger.error("CRITICAL: Missing p_uncorrected column (Decision D068 violation)")
        sys.exit(1)
    if 'p_bonferroni' not in contrasts_df.columns:
        logger.error("CRITICAL: Missing p_bonferroni column (Decision D068 violation)")
        sys.exit(1)
    logger.info("VALIDATION - PASS: Dual p-values present (uncorrected + bonferroni)")

    # Check alpha
    if not np.allclose(contrasts_df['alpha_bonferroni'].unique(), [0.0083333333], rtol=0.01):
        logger.warning("WARNING: alpha_bonferroni may not be exactly 0.0083")
    logger.info(f"Bonferroni alpha = 0.0083 for {len(contrasts_df)} contrasts")

    # Check p_bonferroni >= p_uncorrected
    if (contrasts_df['p_bonferroni'] < contrasts_df['p_uncorrected']).any():
        logger.error("CRITICAL: p_bonferroni < p_uncorrected (impossible)")
        sys.exit(1)
    logger.info("VALIDATION - PASS: p_bonferroni >= p_uncorrected")

    # Check for NaN
    if contrasts_df['estimate'].isna().any():
        logger.error("CRITICAL: NaN in contrast estimates")
        sys.exit(1)
    logger.info("VALIDATION - PASS: No NaN in estimates")

    logger.info("Effect sizes computed for all 6 contrasts")

    # --- Save outputs ---
    contrasts_df.to_csv(contrasts_file, index=False)
    logger.info(f"\nContrasts saved: {contrasts_file}")

    effect_sizes_df.to_csv(effect_sizes_file, index=False)
    logger.info(f"Effect sizes saved: {effect_sizes_file}")

    # --- Interpretation ---
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    # Consolidation benefits
    logger.info("\nConsolidation benefits (Late slope - Early slope):")
    logger.info("(Positive = slower forgetting in Late vs Early)")
    for _, row in contrasts_df.iloc[:3].iterrows():
        sig_str = "***" if row['significant'] else ""
        logger.info(f"  {row['contrast_name']}: {row['estimate']:.4f} (p={row['p_bonferroni']:.4f}) {sig_str}")

    # Paradigm differences
    logger.info("\nParadigm differences in consolidation benefit:")
    for _, row in contrasts_df.iloc[3:].iterrows():
        sig_str = "***" if row['significant'] else ""
        logger.info(f"  {row['contrast_name']}: {row['estimate']:.4f} (p={row['p_bonferroni']:.4f}) {sig_str}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 04 COMPLETE")
    logger.info("=" * 60)

    n_significant = contrasts_df['significant'].sum()
    logger.info(f"{n_significant}/6 contrasts significant after Bonferroni correction")

    return contrasts_df, effect_sizes_df


if __name__ == "__main__":
    main()
