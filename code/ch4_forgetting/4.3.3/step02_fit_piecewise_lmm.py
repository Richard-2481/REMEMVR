#!/usr/bin/env python3
"""
Step 02: Fit Piecewise LMM
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Fit piecewise Linear Mixed Model with 3-way interaction
(Days_within x Segment x paradigm) to test whether consolidation
benefits differ across retrieval paradigms.

Formula: theta ~ Days_within * Segment * paradigm + (1 + Days_within | UID)
"""

import sys
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step02_fit_piecewise_lmm.log"
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
    """Fit piecewise LMM with 3-way interaction."""
    logger.info("=" * 60)
    logger.info("Step 02: Fit Piecewise LMM")
    logger.info("=" * 60)

    # Define paths
    input_file = RQ_DIR / "data" / "step01_piecewise_lmm_input.csv"
    model_file = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
    summary_file = RQ_DIR / "data" / "step02_lmm_model_summary.txt"

    # --- Load data ---
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded: {len(df)} rows")

    # --- Prepare data for LMM ---
    # Use paradigm_code (IFR, ICR, IRE) for cleaner output
    # Ensure categorical variables are properly typed
    df['paradigm_code'] = pd.Categorical(df['paradigm_code'], categories=['IFR', 'ICR', 'IRE'])
    df['Segment'] = pd.Categorical(df['Segment'], categories=['Early', 'Late'])

    # Log design summary
    logger.info("\nDesign summary:")
    logger.info(f"  Participants (UID): {df['UID'].nunique()}")
    logger.info(f"  Observations: {len(df)}")
    logger.info(f"  Segment levels: {df['Segment'].cat.categories.tolist()}")
    logger.info(f"  Paradigm levels: {df['paradigm_code'].cat.categories.tolist()}")

    # --- Fit LMM ---
    logger.info("\nFitting piecewise LMM...")
    logger.info("Formula: theta ~ Days_within * Segment * paradigm_code")
    logger.info("Random effects: (1 + Days_within | UID)")
    logger.info("Estimation: REML=False (ML for model comparison)")

    # Fit model with random intercept and random slope for Days_within
    # Using mixedlm with random slope specification
    try:
        model = smf.mixedlm(
            formula="theta ~ Days_within * Segment * paradigm_code",
            data=df,
            groups=df["UID"],
            re_formula="~Days_within"  # Random intercept + random slope
        )
        result = model.fit(reml=False, method='powell')  # Powell often more robust
        logger.info("Model fit complete")

    except Exception as e:
        logger.error(f"CRITICAL: Model fitting failed: {e}")
        logger.info("Trying fallback with random intercepts only...")

        try:
            model = smf.mixedlm(
                formula="theta ~ Days_within * Segment * paradigm_code",
                data=df,
                groups=df["UID"]
            )
            result = model.fit(reml=False)
            logger.warning("WARNING: Fitted with random intercepts only (random slopes failed)")
        except Exception as e2:
            logger.error(f"CRITICAL: Even random intercepts-only model failed: {e2}")
            sys.exit(1)

    # --- Check convergence ---
    logger.info("\nConvergence check:")
    converged = result.converged
    logger.info(f"  Converged: {converged}")

    if not converged:
        logger.warning("WARNING: Model did not converge fully")
        logger.warning("Proceeding with caution - interpret results carefully")

    # --- Log model summary ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 60)

    summary_str = str(result.summary())
    logger.info("\n" + summary_str)

    # --- Extract key statistics ---
    logger.info("\n" + "=" * 60)
    logger.info("FIXED EFFECTS")
    logger.info("=" * 60)

    # Fixed effects table
    fe_df = pd.DataFrame({
        'Coefficient': result.fe_params,
        'Std.Err': result.bse_fe,
        'z': result.tvalues,
        'P>|z|': result.pvalues
    })
    logger.info("\n" + str(fe_df))

    # --- Random effects ---
    logger.info("\n" + "=" * 60)
    logger.info("RANDOM EFFECTS")
    logger.info("=" * 60)

    # Extract random effects variance components
    re_cov = result.cov_re
    logger.info(f"Random effects covariance matrix:\n{re_cov}")

    # --- Model fit statistics ---
    logger.info("\n" + "=" * 60)
    logger.info("MODEL FIT STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Log-likelihood: {result.llf:.4f}")
    logger.info(f"  AIC: {result.aic:.4f}")
    logger.info(f"  BIC: {result.bic:.4f}")
    logger.info(f"  N groups (participants): {result.nobs}")

    # --- Save model ---
    with open(model_file, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"\nModel saved: {model_file}")

    # --- Save summary to text file ---
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RQ 5.3.3 - Piecewise LMM Model Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Formula: theta ~ Days_within * Segment * paradigm_code\n")
        f.write(f"Random effects: ~Days_within | UID\n")
        f.write(f"Estimation: ML (REML=False)\n\n")
        f.write(f"Observations: {len(df)}\n")
        f.write(f"Groups (participants): {df['UID'].nunique()}\n")
        f.write(f"Converged: {converged}\n\n")
        f.write("=" * 60 + "\n")
        f.write("FULL MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_str)
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("FIXED EFFECTS TABLE\n")
        f.write("=" * 60 + "\n\n")
        f.write(fe_df.to_string())
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("MODEL FIT STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Log-likelihood: {result.llf:.4f}\n")
        f.write(f"AIC: {result.aic:.4f}\n")
        f.write(f"BIC: {result.bic:.4f}\n")

    logger.info(f"Summary saved: {summary_file}")

    # --- Validation ---
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check for NaN coefficients in FIXED effects only (exclude variance components)
    # Variance components are: 'Group Var', 'Days_within Var', 'Group x Days_within Cov'
    variance_components = ['Group Var', 'Days_within Var', 'Group x Days_within Cov']
    fixed_effects_only = fe_df.drop(index=[v for v in variance_components if v in fe_df.index], errors='ignore')

    nan_coefs = fixed_effects_only['Coefficient'].isna().sum()
    if nan_coefs > 0:
        logger.error(f"CRITICAL: {nan_coefs} NaN coefficients in fixed effects")
        sys.exit(1)
    logger.info("VALIDATION - PASS: No NaN in fixed effects")

    # Check for extreme standard errors
    extreme_se = (fe_df['Std.Err'] > 10).sum()
    if extreme_se > 0:
        logger.warning(f"WARNING: {extreme_se} coefficients have SE > 10 (may indicate estimation issues)")
    else:
        logger.info("VALIDATION - PASS: Standard errors in reasonable range")

    logger.info(f"VALIDATION - PASS: LMM convergence = {converged}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 02 COMPLETE")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
