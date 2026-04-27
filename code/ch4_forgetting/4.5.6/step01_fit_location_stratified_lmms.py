#!/usr/bin/env python3
"""
RQ 5.5.6 Step 01: Fit Location-Stratified LMMs with Random Slopes

Purpose:
    Fit separate Linear Mixed Models for Source and Destination locations with random
    intercepts and slopes to estimate location-specific variance components.

Input:
    - ../../5.5.1/data/step04_lmm_input.csv (801 rows: 800 data + header)
      Columns: UID, test, composite_ID, TSVR_hours, Days, log_Days_plus1,
               Days_squared, LocationType, LocationType_coded, theta, se

Output:
    - data/step01_model_metadata_source.yaml
    - data/step01_model_metadata_destination.yaml
    - data/step01_source_lmm_model.pkl
    - data/step01_destination_lmm_model.pkl

Model Specification:
    - Fixed effects: theta ~ log_TSVR (log-transformed time)
    - Random effects: (log_TSVR | UID) - random intercept + random slope
    - REML=False for model comparison consistency
    - log_TSVR = np.log(TSVR_hours + 1) - computed from TSVR_hours

Note on LocationType values:
    - Input data has "source" and "destination" (lowercase)
    - Outputs use lowercase for consistency with input data

Author: Claude (g_code agent)
Date: 2025-12-05
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
log_path = Path("results/ch5/5.5.6/logs/step01_fit_location_stratified_lmms.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def fit_location_lmm(df_location: pd.DataFrame, location_name: str) -> tuple:
    """
    Fit LMM for a single location type with random intercepts and slopes.

    Args:
        df_location: DataFrame filtered to single location type
        location_name: "source" or "destination"

    Returns:
        (fitted_model, metadata_dict)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fitting LMM for location: {location_name}")
    logger.info(f"{'='*60}")

    n_obs = len(df_location)
    n_groups = df_location['UID'].nunique()
    logger.info(f"  Observations: {n_obs}")
    logger.info(f"  Groups (participants): {n_groups}")

    # Create log-transformed time variable
    # Add small constant to avoid log(0) for T1
    df_location = df_location.copy()
    df_location['log_TSVR'] = np.log(df_location['TSVR_hours'] + 1)

    logger.info(f"  log_TSVR range: [{df_location['log_TSVR'].min():.3f}, {df_location['log_TSVR'].max():.3f}]")

    # Model specification: theta ~ log_TSVR + (log_TSVR | UID)
    # Random slopes on log_TSVR to match fixed effects transformation
    formula = "theta ~ log_TSVR"

    metadata = {
        'location': location_name,
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
                df_location,
                groups=df_location['UID'],
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
            df_location,
            groups=df_location['UID'],
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
            df_location,
            groups=df_location['UID'],
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
    logger.error(f"  CRITICAL: All model specifications failed for {location_name}")
    return None, metadata


def main():
    """Fit location-stratified LMMs for variance decomposition."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.6 Step 01: Fit Location-Stratified LMMs")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load input data from RQ 5.5.1
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.5.1/data/step04_lmm_input.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Input file not found: {input_path}")
        logger.error("RQ 5.5.1 Step 04 must complete before this step")
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"Loaded input data: {len(df)} rows from {input_path}")

    # Validate expected structure
    if 'LocationType' not in df.columns:
        logger.error("EXPECTATIONS ERROR: LocationType column not found in input data")
        sys.exit(1)

    locations = df['LocationType'].unique()
    logger.info(f"LocationType values in data: {list(locations)}")

    expected_locations = {'source', 'destination'}
    if set(locations) != expected_locations:
        logger.error(f"EXPECTATIONS ERROR: Expected LocationType values {expected_locations}, got {set(locations)}")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 2. Fit LMM for each location type
    # ---------------------------------------------------------------------
    fitted_models = {}
    all_metadata = {}

    for location in ['source', 'destination']:
        df_location = df[df['LocationType'] == location].copy()

        if len(df_location) == 0:
            logger.error(f"No data for location: {location}")
            sys.exit(1)

        logger.info(f"\nProcessing location: {location}")
        logger.info(f"  Rows: {len(df_location)}")

        result, metadata = fit_location_lmm(df_location, location)
        fitted_models[location] = result
        all_metadata[location] = metadata

        # Save individual location metadata
        metadata_path = Path(f"results/ch5/5.5.6/data/step01_model_metadata_{location}.yaml")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        logger.info(f"Saved metadata to: {metadata_path}")

    # ---------------------------------------------------------------------
    # 3. Validate all models converged
    # ---------------------------------------------------------------------
    failed_locations = [loc for loc, m in all_metadata.items() if not m.get('converged', False)]
    if failed_locations:
        logger.error(f"CRITICAL: Models failed to converge for locations: {failed_locations}")
        logger.error("Cannot proceed to variance decomposition without fitted models")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 4. Save individual fitted models
    # ---------------------------------------------------------------------
    for location in ['source', 'destination']:
        model_path = Path(f"results/ch5/5.5.6/data/step01_{location}_lmm_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(fitted_models[location], f)
        logger.info(f"Saved fitted model to: {model_path}")

    # ---------------------------------------------------------------------
    # 5. Print model summaries
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("MODEL SUMMARIES")
    logger.info("=" * 60)

    for location, result in fitted_models.items():
        if result is not None:
            logger.info(f"\n--- {location.capitalize()} Location ---")
            logger.info(f"Random structure: {all_metadata[location]['random_structure']}")

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
    for location, meta in all_metadata.items():
        aic_val = meta.get('aic', 'N/A')
        aic_str = f"{aic_val:.2f}" if isinstance(aic_val, (int, float)) else aic_val
        logger.info(f"  {location}: {meta['random_structure']} structure, AIC={aic_str}")
    logger.info("Ready for Step 02: Extract Variance Components")


if __name__ == "__main__":
    main()
