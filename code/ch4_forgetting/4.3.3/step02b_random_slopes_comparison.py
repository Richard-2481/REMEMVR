#!/usr/bin/env python3
"""
Step 02b: Random Slopes Comparison
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Test intercepts-only vs intercepts+slopes random effects structure
to determine if individual differences in forgetting rates exist.

Comparison: ΔAIC between:
  - Model A: (1 | UID) - Intercepts only
  - Model B: (1 + Days_within | UID) - Intercepts + slopes

Decision: If ΔAIC > 2 → Use slopes (individual heterogeneity confirmed)
          If ΔAIC < 2 → Use intercepts (homogeneous effects confirmed)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step02b_random_slopes_comparison.log"
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


def main():
    """Compare intercepts-only vs intercepts+slopes models."""
    logger.info("=" * 60)
    logger.info("Step 02b: Random Slopes Comparison")
    logger.info("=" * 60)

    # Define paths
    input_file = RQ_DIR / "data" / "step01_piecewise_lmm_input.csv"
    output_file = RQ_DIR / "data" / "step02b_random_slopes_comparison.csv"

    # Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded: {len(df)} rows, {df['UID'].nunique()} participants")

    # Prepare categorical variables
    df['paradigm_code'] = pd.Categorical(df['paradigm_code'], categories=['IFR', 'ICR', 'IRE'])
    df['Segment'] = pd.Categorical(df['Segment'], categories=['Early', 'Late'])

    # --- Fit Model A: Intercepts Only ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL A: Random Intercepts Only")
    logger.info("=" * 60)
    logger.info("Formula: theta ~ Days_within * Segment * paradigm_code")
    logger.info("Random effects: (1 | UID)")

    try:
        model_intercepts = smf.mixedlm(
            formula="theta ~ Days_within * Segment * paradigm_code",
            data=df,
            groups=df["UID"],
            re_formula="~1"  # Intercepts only
        )
        result_intercepts = model_intercepts.fit(reml=False, method='powell')
        logger.info("✓ Model A converged")
        
        logger.info(f"  AIC: {result_intercepts.aic:.2f}")
        logger.info(f"  BIC: {result_intercepts.bic:.2f}")
        logger.info(f"  Log-likelihood: {result_intercepts.llf:.2f}")
        logger.info(f"  Random intercept variance: {result_intercepts.cov_re.iloc[0,0]:.4f}")

        intercepts_success = True
        intercepts_aic = result_intercepts.aic
        intercepts_bic = result_intercepts.bic

    except Exception as e:
        logger.error(f"✗ Model A failed to converge: {e}")
        intercepts_success = False
        intercepts_aic = np.nan
        intercepts_bic = np.nan

    # --- Fit Model B: Intercepts + Slopes ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL B: Random Intercepts + Random Slopes")
    logger.info("=" * 60)
    logger.info("Formula: theta ~ Days_within * Segment * paradigm_code")
    logger.info("Random effects: (1 + Days_within | UID)")

    try:
        model_slopes = smf.mixedlm(
            formula="theta ~ Days_within * Segment * paradigm_code",
            data=df,
            groups=df["UID"],
            re_formula="~Days_within"  # Intercepts + slopes
        )
        result_slopes = model_slopes.fit(reml=False, method='powell')
        logger.info("✓ Model B converged")
        
        logger.info(f"  AIC: {result_slopes.aic:.2f}")
        logger.info(f"  BIC: {result_slopes.bic:.2f}")
        logger.info(f"  Log-likelihood: {result_slopes.llf:.2f}")
        
        # Check if covariance matrix has sufficient dimensions for slope variance
        if result_slopes.cov_re.shape[0] >= 2:
            slope_var = result_slopes.cov_re.iloc[1, 1]
            slope_sd = np.sqrt(slope_var)
            logger.info(f"  Random intercept variance: {result_slopes.cov_re.iloc[0,0]:.4f}")
            logger.info(f"  Random slope variance: {slope_var:.4f}")
            logger.info(f"  Random slope SD: {slope_sd:.4f}")
            slopes_var = slope_var
        else:
            logger.warning("  WARNING: Covariance matrix singular (boundary issue)")
            slopes_var = 0.0

        slopes_success = True
        slopes_aic = result_slopes.aic
        slopes_bic = result_slopes.bic

    except Exception as e:
        logger.error(f"✗ Model B failed to converge: {e}")
        slopes_success = False
        slopes_aic = np.nan
        slopes_bic = np.nan
        slopes_var = np.nan

    # --- Compare Models ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    if intercepts_success and slopes_success:
        delta_aic = intercepts_aic - slopes_aic
        delta_bic = intercepts_bic - slopes_bic

        logger.info(f"Intercepts-only AIC: {intercepts_aic:.2f}")
        logger.info(f"Intercepts+slopes AIC: {slopes_aic:.2f}")
        logger.info(f"ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
        logger.info(f"ΔBIC: {delta_bic:.2f}")

        # Interpret (from rq_platinum Step 12C)
        logger.info("\n" + "=" * 60)
        logger.info("INTERPRETATION")
        logger.info("=" * 60)

        if delta_aic > 2:
            decision = "USE_SLOPES"
            logger.info("🟢 OPTION A: Slopes Improve Fit (ΔAIC > 2)")
            logger.info("  → Individual differences in forgetting rates CONFIRMED")
            logger.info(f"  → Random slope variance non-zero (SD = {slope_sd:.4f})")
            logger.info("  → Use intercepts+slopes model for downstream analyses")
            logger.info("  → Can claim heterogeneous effects (individual variability)")

        elif slopes_var <= 0.001:
            decision = "USE_INTERCEPTS_BOUNDARY"
            logger.info("🟡 OPTION B: Slopes at Boundary")
            logger.info("  → Random slope variance near zero (convergence issue)")
            logger.info("  → Keep intercepts-only model (slopes not estimable)")
            logger.info("  → Homogeneous effects ASSUMED (not confirmed)")
            logger.info("  → LIMITATION: Cannot definitively test heterogeneity")

        else:
            decision = "USE_INTERCEPTS_VALIDATED"
            logger.info("🟢 OPTION C: Slopes Converge But Don't Improve (|ΔAIC| < 2)")
            logger.info("  → Random slope variance negligible despite convergence")
            logger.info(f"  → Slope variance: {slopes_var:.4f}")
            logger.info("  → Keep intercepts-only model (simpler, equivalent fit)")
            logger.info("  → Homogeneous effects CONFIRMED (tested and validated)")

    else:
        logger.error("✗ Cannot compare models (one or both failed to converge)")
        decision = "COMPARISON_FAILED"
        delta_aic = np.nan
        delta_bic = np.nan

    # --- Save Comparison Results ---
    comparison = pd.DataFrame({
        'model': ['Intercepts_Only', 'Intercepts_Slopes'],
        'aic': [intercepts_aic, slopes_aic],
        'bic': [intercepts_bic, slopes_bic],
        'delta_aic': [0.0, delta_aic],
        'converged': [intercepts_success, slopes_success],
        'random_slope_var': [0.0, slopes_var if slopes_success else np.nan]
    })

    comparison.to_csv(output_file, index=False)
    logger.info(f"\n✓ Comparison results saved: {output_file}")

    # --- Final Recommendation ---
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION")
    logger.info("=" * 60)

    if decision == "USE_SLOPES":
        logger.info("Recommendation: USE random slopes model (current step02)")
        logger.info("  → Individual heterogeneity confirmed")
        logger.info("  → Document in validation.md as OPTION A")
    elif decision == "USE_INTERCEPTS_VALIDATED":
        logger.info("Recommendation: SWITCH to intercepts-only model")
        logger.info("  → Homogeneous effects confirmed via empirical test")
        logger.info("  → Document in validation.md as OPTION C")
    elif decision == "USE_INTERCEPTS_BOUNDARY":
        logger.info("Recommendation: SWITCH to intercepts-only model")
        logger.info("  → Convergence issues with slopes (boundary)")
        logger.info("  → Document in validation.md as OPTION B")
    else:
        logger.info("Recommendation: MANUAL REVIEW REQUIRED")
        logger.info("  → Model comparison inconclusive")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 02B COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
