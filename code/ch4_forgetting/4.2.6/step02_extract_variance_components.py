#!/usr/bin/env python3
"""
RQ 5.2.6 Step 02: Extract Variance Components per Domain

Purpose:
    Extract variance components (var_intercept, var_slope, cov_int_slope,
    var_residual) from fitted domain-stratified LMMs.

Input:
    - data/step01_fitted_models.pkl (Dict with 2 fitted MixedLM objects)
    - data/step01_model_metadata_what.yaml
    - data/step01_model_metadata_where.yaml

Output:
    - data/step02_variance_components.csv (10 rows: 5 components x 2 domains)

Variance Components:
    - var_intercept: Between-person variance in baseline theta (Day 0)
    - var_slope: Between-person variance in forgetting rate (theta change per log-hour)
    - cov_int_slope: Covariance between intercept and slope
    - var_residual: Within-person variance (measurement error + unexplained)
    - total_variance: Sum of all variance components

Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import pickle
import yaml

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step02_extract_variance_components.log")
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


def extract_variance_components(result, metadata: dict, domain: str) -> pd.DataFrame:
    """Extract variance components from a fitted MixedLM model."""

    logger.info(f"\nExtracting variance components for {domain} domain")

    random_structure = metadata.get('random_structure', 'Unknown')
    logger.info(f"  Random structure: {random_structure}")

    components = []

    # Get random effects covariance matrix
    cov_re = result.cov_re

    # Extract var_intercept (always present)
    var_intercept = float(cov_re.iloc[0, 0])
    components.append({
        'domain': domain,
        'component': 'var_intercept',
        'value': var_intercept,
        'interpretation': 'Between-person variance in baseline ability (Day 0)'
    })
    logger.info(f"  var_intercept: {var_intercept:.6f}")

    # Extract var_slope (if random slopes fit)
    if random_structure in ['Full', 'Uncorrelated'] and cov_re.shape[0] >= 2:
        var_slope = float(cov_re.iloc[1, 1])
        logger.info(f"  var_slope: {var_slope:.6f}")
    else:
        var_slope = np.nan
        logger.info(f"  var_slope: NA (intercept-only model)")

    components.append({
        'domain': domain,
        'component': 'var_slope',
        'value': var_slope,
        'interpretation': 'Between-person variance in forgetting rate (per log-hour)'
    })

    # Extract cov_int_slope (if full random structure)
    if random_structure == 'Full' and cov_re.shape[0] >= 2:
        cov_int_slope = float(cov_re.iloc[0, 1])
        logger.info(f"  cov_int_slope: {cov_int_slope:.6f}")
    else:
        cov_int_slope = 0.0 if random_structure == 'Uncorrelated' else np.nan
        logger.info(f"  cov_int_slope: {cov_int_slope} ({random_structure} structure)")

    components.append({
        'domain': domain,
        'component': 'cov_int_slope',
        'value': cov_int_slope,
        'interpretation': 'Covariance between baseline and forgetting rate'
    })

    # Extract var_residual (scale parameter)
    var_residual = float(result.scale)
    components.append({
        'domain': domain,
        'component': 'var_residual',
        'value': var_residual,
        'interpretation': 'Within-person variance (measurement error + unexplained)'
    })
    logger.info(f"  var_residual: {var_residual:.6f}")

    # Compute total variance (for variance decomposition)
    if pd.isna(var_slope):
        total_variance = var_intercept + var_residual
    else:
        total_variance = var_intercept + var_slope + var_residual

    components.append({
        'domain': domain,
        'component': 'total_variance',
        'value': total_variance,
        'interpretation': 'Sum of all variance components'
    })
    logger.info(f"  total_variance: {total_variance:.6f}")

    return pd.DataFrame(components)


def main():
    """Extract variance components from fitted domain LMMs."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 02: Extract Variance Components per Domain")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load fitted models from Step 01
    # ---------------------------------------------------------------------
    models_path = Path("results/ch5/5.2.6/data/step01_fitted_models.pkl")

    if not models_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Models file not found: {models_path}")
        logger.error("Step 01 must complete before Step 02")
        sys.exit(1)

    with open(models_path, 'rb') as f:
        fitted_models = pickle.load(f)

    logger.info(f"Loaded fitted models: {list(fitted_models.keys())}")

    # ---------------------------------------------------------------------
    # 2. Load metadata for each domain
    # ---------------------------------------------------------------------
    all_metadata = {}
    for domain in ['What', 'Where']:
        metadata_path = Path(f"results/ch5/5.2.6/data/step01_model_metadata_{domain.lower()}.yaml")
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            sys.exit(1)

        with open(metadata_path, 'r') as f:
            all_metadata[domain] = yaml.safe_load(f)

        logger.info(f"Loaded metadata for {domain}: {all_metadata[domain]['random_structure']} structure")

    # ---------------------------------------------------------------------
    # 3. Extract variance components for each domain
    # ---------------------------------------------------------------------
    variance_dfs = []

    for domain in ['What', 'Where']:
        result = fitted_models[domain]
        metadata = all_metadata[domain]

        if result is None:
            logger.error(f"No fitted model for {domain} domain")
            sys.exit(1)

        df_var = extract_variance_components(result, metadata, domain)
        variance_dfs.append(df_var)

    variance_components = pd.concat(variance_dfs, ignore_index=True)

    # ---------------------------------------------------------------------
    # 4. Validate variance components
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking variance component constraints")
    logger.info("=" * 60)

    # Check for Heywood cases (negative variances)
    variance_cols = ['var_intercept', 'var_slope', 'var_residual']
    for component in variance_cols:
        comp_data = variance_components[variance_components['component'] == component]
        for _, row in comp_data.iterrows():
            if pd.notna(row['value']) and row['value'] < 0:
                logger.error(f"HEYWOOD CASE: {row['domain']} {component} = {row['value']:.6f} (negative variance)")
                sys.exit(1)

    logger.info("  All variance components non-negative: PASS")

    # Check for missing critical components
    for domain in ['What', 'Where']:
        domain_data = variance_components[variance_components['domain'] == domain]
        if len(domain_data) != 5:
            logger.error(f"Missing components for {domain}: expected 5, got {len(domain_data)}")
            sys.exit(1)

    logger.info("  All domains have 5 components: PASS")
    logger.info(f"  Total rows: {len(variance_components)} (expected 10)")

    # ---------------------------------------------------------------------
    # 5. Save variance components
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step02_variance_components.csv")
    variance_components.to_csv(output_path, index=False)
    logger.info(f"\nSaved variance components to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary table
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VARIANCE COMPONENTS SUMMARY")
    logger.info("=" * 60)

    for domain in ['What', 'Where']:
        domain_data = variance_components[variance_components['domain'] == domain]
        logger.info(f"\n{domain} Domain:")
        for _, row in domain_data.iterrows():
            if pd.isna(row['value']):
                val_str = "NA"
            else:
                val_str = f"{row['value']:.6f}"
            logger.info(f"  {row['component']}: {val_str}")

    # ---------------------------------------------------------------------
    # 7. Variance decomposition preview
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VARIANCE DECOMPOSITION PREVIEW (for ICC computation)")
    logger.info("=" * 60)

    for domain in ['What', 'Where']:
        domain_data = variance_components[variance_components['domain'] == domain]

        var_int = domain_data[domain_data['component'] == 'var_intercept']['value'].values[0]
        var_slope = domain_data[domain_data['component'] == 'var_slope']['value'].values[0]
        var_res = domain_data[domain_data['component'] == 'var_residual']['value'].values[0]

        logger.info(f"\n{domain} Domain:")
        logger.info(f"  var_intercept:  {var_int:.4f} ({var_int/(var_int+var_res)*100:.1f}% of simple total)")
        if pd.notna(var_slope):
            logger.info(f"  var_slope:      {var_slope:.4f} ({var_slope/(var_slope+var_res)*100:.1f}% of simple total)")
        else:
            logger.info(f"  var_slope:      NA")
        logger.info(f"  var_residual:   {var_res:.4f}")

    # ---------------------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 02 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Extracted variance components for {len(fitted_models)} domains")
    logger.info(f"Output: {len(variance_components)} rows (5 components x 2 domains)")
    logger.info("Ready for Step 03: Compute ICC Estimates")


if __name__ == "__main__":
    main()
