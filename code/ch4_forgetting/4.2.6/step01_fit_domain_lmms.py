#!/usr/bin/env python3
"""
RQ 5.2.6 Step 01: Fit Domain-Stratified LMMs with Random Slopes

Purpose:
    Fit separate Linear Mixed Models for What and Where domains with random
    intercepts and slopes to estimate domain-specific variance components.

Input:
    - data/step00_lmm_input_filtered.csv (800 rows: 100 UID x 4 tests x 2 domains)

Output:
    - data/step01_model_metadata_what.yaml
    - data/step01_model_metadata_where.yaml
    - data/step01_fitted_models.pkl (Dict with 2 fitted MixedLM objects)

Model Specification:
    - Fixed effects: theta ~ log_TSVR (log-transformed time, per RQ 5.2.1 model selection)
    - Random effects: (log_TSVR | UID) - random intercept + random slope
    - REML=False for model comparison consistency

CRITICAL: Random slopes on log_TSVR not TSVR_hours
    Per Session 2025-12-03 06:00 model correction, random slopes must align with
    the dominant fixed effects time transformation. RQ 5.2.1 established Log model
    as best fit (AIC weight 61.9%), so random slopes should be on log_TSVR.

Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import pickle
import yaml
import warnings
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step01_fit_domain_lmms.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def fit_domain_lmm(df_domain: pd.DataFrame, domain_name: str) -> tuple:
    """
    Fit LMM for a single domain with random intercepts and slopes.

    Returns:
        (fitted_model, metadata_dict)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fitting LMM for domain: {domain_name}")
    logger.info(f"{'='*60}")

    n_obs = len(df_domain)
    n_groups = df_domain['UID'].nunique()
    logger.info(f"  Observations: {n_obs}")
    logger.info(f"  Groups (participants): {n_groups}")

    # Create log-transformed time variable
    # Add small constant to avoid log(0) for T1
    df_domain = df_domain.copy()
    df_domain['log_TSVR'] = np.log(df_domain['TSVR_hours'] + 1)

    logger.info(f"  log_TSVR range: [{df_domain['log_TSVR'].min():.3f}, {df_domain['log_TSVR'].max():.3f}]")

    # Model specification: theta ~ log_TSVR + (log_TSVR | UID)
    # CRITICAL: Random slopes on log_TSVR to match fixed effects transformation
    formula = "theta ~ log_TSVR"

    metadata = {
        'domain': domain_name,
        'n_obs': int(n_obs),
        'n_groups': int(n_groups),
        'formula': formula,
        're_formula': '~log_TSVR',
        'time_variable': 'log_TSVR',
        'reml': False
    }

    # Try fitting with full random structure first
    try:
        logger.info(f"  Attempting fit with full random structure (intercept + slope correlated)")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = smf.mixedlm(
                formula,
                df_domain,
                groups=df_domain['UID'],
                re_formula='~log_TSVR'
            )
            result = model.fit(reml=False, method='lbfgs')

            # Check for convergence warnings
            convergence_warnings = [str(warning.message) for warning in w
                                     if 'converg' in str(warning.message).lower()]
            if convergence_warnings:
                logger.warning(f"  Convergence warnings: {convergence_warnings}")

        # Check convergence
        converged = result.converged
        logger.info(f"  Model converged: {converged}")

        if converged:
            metadata['random_structure'] = 'Full'
            metadata['optimizer'] = 'lbfgs'
            metadata['converged'] = True
            metadata['log_likelihood'] = float(result.llf)
            metadata['aic'] = float(result.aic)
            metadata['bic'] = float(result.bic)
            logger.info(f"  Log-likelihood: {result.llf:.2f}")
            logger.info(f"  AIC: {result.aic:.2f}")
            logger.info(f"  BIC: {result.bic:.2f}")
            return result, metadata

    except Exception as e:
        logger.warning(f"  Full model failed: {str(e)}")

    # Fallback 1: Try with uncorrelated random effects
    try:
        logger.info(f"  Attempting fit with uncorrelated random effects")

        model = smf.mixedlm(
            formula,
            df_domain,
            groups=df_domain['UID'],
            re_formula='~log_TSVR'
        )
        # Use variance components structure (uncorrelated)
        result = model.fit(reml=False, method='lbfgs', free=None)

        converged = result.converged
        logger.info(f"  Model converged: {converged}")

        if converged:
            metadata['random_structure'] = 'Uncorrelated'
            metadata['optimizer'] = 'lbfgs'
            metadata['converged'] = True
            metadata['log_likelihood'] = float(result.llf)
            metadata['aic'] = float(result.aic)
            metadata['bic'] = float(result.bic)
            logger.info(f"  Log-likelihood: {result.llf:.2f}")
            logger.info(f"  AIC: {result.aic:.2f}")
            logger.info(f"  BIC: {result.bic:.2f}")
            return result, metadata

    except Exception as e:
        logger.warning(f"  Uncorrelated model failed: {str(e)}")

    # Fallback 2: Random intercepts only
    try:
        logger.info(f"  Attempting fit with random intercepts only")

        model = smf.mixedlm(
            formula,
            df_domain,
            groups=df_domain['UID'],
            re_formula='~1'
        )
        result = model.fit(reml=False, method='lbfgs')

        converged = result.converged
        logger.info(f"  Model converged: {converged}")

        if converged:
            metadata['random_structure'] = 'Intercept-only'
            metadata['optimizer'] = 'lbfgs'
            metadata['converged'] = True
            metadata['log_likelihood'] = float(result.llf)
            metadata['aic'] = float(result.aic)
            metadata['bic'] = float(result.bic)
            metadata['note'] = 'Random slopes not supported; variance decomposition limited to intercepts'
            logger.info(f"  Log-likelihood: {result.llf:.2f}")
            logger.info(f"  AIC: {result.aic:.2f}")
            logger.info(f"  BIC: {result.bic:.2f}")
            logger.warning(f"  NOTE: Random slopes not supported - slope ICC will be NA")
            return result, metadata

    except Exception as e:
        logger.error(f"  Intercept-only model also failed: {str(e)}")

    # All attempts failed
    metadata['converged'] = False
    metadata['error'] = 'All model specifications failed to converge'
    logger.error(f"  CRITICAL: All model specifications failed for {domain_name}")
    return None, metadata


def main():
    """Fit domain-stratified LMMs for variance decomposition."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 01: Fit Domain-Stratified LMMs")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load filtered data from Step 00
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.2.6/data/step00_lmm_input_filtered.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Input file not found: {input_path}")
        logger.error("Step 00 must complete before Step 01")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded input data: {len(df)} rows from {input_path}")

    # Validate expected structure
    domains = df['domain'].unique()
    logger.info(f"Domains in data: {list(domains)}")

    if 'when' in [d.lower() for d in domains]:
        logger.error("When domain found in data - should have been filtered in Step 00!")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 2. Fit LMM for each domain
    # ---------------------------------------------------------------------
    fitted_models = {}
    all_metadata = {}

    for domain in ['what', 'where']:
        df_domain = df[df['domain'].str.lower() == domain].copy()

        if len(df_domain) == 0:
            logger.error(f"No data for domain: {domain}")
            sys.exit(1)

        result, metadata = fit_domain_lmm(df_domain, domain.capitalize())
        fitted_models[domain.capitalize()] = result
        all_metadata[domain.capitalize()] = metadata

        # Save individual domain metadata
        metadata_path = Path(f"results/ch5/5.2.6/data/step01_model_metadata_{domain}.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        logger.info(f"Saved metadata to: {metadata_path}")

    # ---------------------------------------------------------------------
    # 3. Validate all models converged
    # ---------------------------------------------------------------------
    failed_domains = [d for d, m in all_metadata.items() if not m.get('converged', False)]
    if failed_domains:
        logger.error(f"CRITICAL: Models failed to converge for domains: {failed_domains}")
        logger.error("Cannot proceed to variance decomposition without fitted models")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 4. Save fitted models
    # ---------------------------------------------------------------------
    models_path = Path("results/ch5/5.2.6/data/step01_fitted_models.pkl")
    with open(models_path, 'wb') as f:
        pickle.dump(fitted_models, f)
    logger.info(f"\nSaved fitted models to: {models_path}")

    # ---------------------------------------------------------------------
    # 5. Print model summaries
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("MODEL SUMMARIES")
    logger.info("=" * 60)

    for domain, result in fitted_models.items():
        if result is not None:
            logger.info(f"\n--- {domain} Domain ---")
            logger.info(f"Random structure: {all_metadata[domain]['random_structure']}")

            # Fixed effects
            logger.info(f"\nFixed Effects:")
            logger.info(f"  Intercept: {result.fe_params['Intercept']:.4f} (SE: {result.bse_fe['Intercept']:.4f})")
            logger.info(f"  log_TSVR:  {result.fe_params['log_TSVR']:.4f} (SE: {result.bse_fe['log_TSVR']:.4f})")

            # Random effects variances
            logger.info(f"\nRandom Effects Variances:")
            cov_re = result.cov_re
            logger.info(f"  cov_re shape: {cov_re.shape}")
            if cov_re.shape == (2, 2):
                logger.info(f"  Group Var (intercept): {cov_re.iloc[0, 0]:.6f}")
                logger.info(f"  log_TSVR Var (slope):  {cov_re.iloc[1, 1]:.6f}")
                logger.info(f"  Cov(intercept, slope): {cov_re.iloc[0, 1]:.6f}")
            elif cov_re.shape == (1, 1):
                logger.info(f"  Group Var (intercept): {cov_re.iloc[0, 0]:.6f}")
                logger.info(f"  (No random slopes)")

            # Residual variance
            logger.info(f"  Scale (residual): {result.scale:.6f}")

    # ---------------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 01 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models fitted: {len(fitted_models)}")
    for domain, meta in all_metadata.items():
        logger.info(f"  {domain}: {meta['random_structure']} structure, AIC={meta.get('aic', 'N/A'):.2f}")
    logger.info("Ready for Step 02: Extract Variance Components")


if __name__ == "__main__":
    main()
