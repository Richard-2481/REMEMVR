#!/usr/bin/env python3
"""
RQ 5.5.6 Step 02: Extract Variance Components per Location

Purpose:
    Extract variance components (var_intercept, var_slope, cov_int_slope,
    var_residual, correlation_int_slope) from fitted location-stratified LMMs.

Input:
    - data/step01_source_lmm_model.pkl (Fitted Source LMM)
    - data/step01_destination_lmm_model.pkl (Fitted Destination LMM)
    - data/step01_model_metadata_source.yaml
    - data/step01_model_metadata_destination.yaml

Output:
    - data/step02_variance_components.csv (10 rows: 5 components x 2 locations)

Variance Components:
    - var_intercept: Between-person variance in baseline theta
    - var_slope: Between-person variance in forgetting rate (theta change per time unit)
    - cov_int_slope: Covariance between intercept and slope
    - var_residual: Within-person variance (measurement error + unexplained)
    - correlation_int_slope: Correlation between intercepts and slopes

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

# Set up logging
log_path = Path("results/ch5/5.5.6/logs/step02_extract_variance_components.log")
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


def extract_variance_components(result, metadata: dict, location: str) -> pd.DataFrame:
    """Extract variance components from a fitted MixedLM model.

    Args:
        result: statsmodels MixedLMResults object
        metadata: Model metadata dict
        location: Location type ('Source' or 'Destination')

    Returns:
        DataFrame with 5 rows (5 variance components)
    """

    logger.info(f"\nExtracting variance components for {location} location")

    random_structure = metadata.get('random_structure', 'Unknown')
    logger.info(f"  Random structure: {random_structure}")

    components = []

    # Get random effects covariance matrix
    cov_re = result.cov_re

    # Extract var_intercept (always present)
    var_intercept = float(cov_re.iloc[0, 0])
    components.append({
        'location': location,
        'component': 'var_intercept',
        'value': var_intercept
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
        'location': location,
        'component': 'var_slope',
        'value': var_slope
    })

    # Extract cov_int_slope (if full random structure)
    if random_structure == 'Full' and cov_re.shape[0] >= 2:
        cov_int_slope = float(cov_re.iloc[0, 1])
        logger.info(f"  cov_int_slope: {cov_int_slope:.6f}")
    else:
        cov_int_slope = 0.0 if random_structure == 'Uncorrelated' else np.nan
        logger.info(f"  cov_int_slope: {cov_int_slope} ({random_structure} structure)")

    components.append({
        'location': location,
        'component': 'cov_int_slope',
        'value': cov_int_slope
    })

    # Extract var_residual (scale parameter)
    var_residual = float(result.scale)
    components.append({
        'location': location,
        'component': 'var_residual',
        'value': var_residual
    })
    logger.info(f"  var_residual: {var_residual:.6f}")

    # Compute correlation_int_slope
    if random_structure == 'Full' and cov_re.shape[0] >= 2 and var_intercept > 0 and var_slope > 0:
        correlation_int_slope = cov_int_slope / (np.sqrt(var_intercept) * np.sqrt(var_slope))
        logger.info(f"  correlation_int_slope: {correlation_int_slope:.6f}")
    else:
        correlation_int_slope = np.nan
        logger.info(f"  correlation_int_slope: NA (cannot compute from {random_structure} structure)")

    components.append({
        'location': location,
        'component': 'correlation_int_slope',
        'value': correlation_int_slope
    })

    return pd.DataFrame(components)


def main():
    """Extract variance components from fitted location LMMs."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.6 Step 02: Extract Variance Components per Location")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load fitted models from Step 01
    # ---------------------------------------------------------------------
    source_model_path = Path("results/ch5/5.5.6/data/step01_source_lmm_model.pkl")
    destination_model_path = Path("results/ch5/5.5.6/data/step01_destination_lmm_model.pkl")

    if not source_model_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Source model file not found: {source_model_path}")
        logger.error("Step 01 must complete before Step 02")
        sys.exit(1)

    if not destination_model_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Destination model file not found: {destination_model_path}")
        logger.error("Step 01 must complete before Step 02")
        sys.exit(1)

    # Load Source model
    with open(source_model_path, 'rb') as f:
        source_model = pickle.load(f)
    logger.info(f"Loaded Source model from {source_model_path}")

    # Load Destination model
    with open(destination_model_path, 'rb') as f:
        destination_model = pickle.load(f)
    logger.info(f"Loaded Destination model from {destination_model_path}")

    # ---------------------------------------------------------------------
    # 2. Load metadata for each location
    # ---------------------------------------------------------------------
    all_metadata = {}
    for location_lower in ['source', 'destination']:
        metadata_path = Path(f"results/ch5/5.5.6/data/step01_model_metadata_{location_lower}.yaml")
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            sys.exit(1)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata[location_lower.capitalize()] = yaml.safe_load(f)

        logger.info(f"Loaded metadata for {location_lower.capitalize()}: {all_metadata[location_lower.capitalize()]['random_structure']} structure")

    # ---------------------------------------------------------------------
    # 3. Extract variance components for each location
    # ---------------------------------------------------------------------
    variance_dfs = []

    # Extract for Source
    df_source = extract_variance_components(source_model, all_metadata['Source'], 'Source')
    variance_dfs.append(df_source)

    # Extract for Destination
    df_destination = extract_variance_components(destination_model, all_metadata['Destination'], 'Destination')
    variance_dfs.append(df_destination)

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
                logger.error(f"HEYWOOD CASE: {row['location']} {component} = {row['value']:.6f} (negative variance)")
                sys.exit(1)

    logger.info("  [PASS] All variance components non-negative")

    # Check var_intercept > 0 (strictly positive, required)
    intercept_data = variance_components[variance_components['component'] == 'var_intercept']
    for _, row in intercept_data.iterrows():
        if row['value'] <= 0:
            logger.error(f"VALIDATION ERROR: {row['location']} var_intercept = {row['value']:.6f} (must be > 0)")
            sys.exit(1)

    logger.info("  [PASS] var_intercept > 0 for both locations")

    # Check var_residual > 0 (strictly positive, required)
    residual_data = variance_components[variance_components['component'] == 'var_residual']
    for _, row in residual_data.iterrows():
        if row['value'] <= 0:
            logger.error(f"VALIDATION ERROR: {row['location']} var_residual = {row['value']:.6f} (must be > 0)")
            sys.exit(1)

    logger.info("  [PASS] var_residual > 0 for both locations")

    # Check correlation_int_slope in [-1, 1]
    corr_data = variance_components[variance_components['component'] == 'correlation_int_slope']
    for _, row in corr_data.iterrows():
        if pd.notna(row['value']) and (row['value'] < -1 or row['value'] > 1):
            logger.error(f"VALIDATION ERROR: {row['location']} correlation_int_slope = {row['value']:.6f} (must be in [-1, 1])")
            sys.exit(1)

    logger.info("  [PASS] correlation_int_slope in [-1, 1] for all locations")

    # Check for missing critical components
    for location in ['Source', 'Destination']:
        location_data = variance_components[variance_components['location'] == location]
        if len(location_data) != 5:
            logger.error(f"Missing components for {location}: expected 5, got {len(location_data)}")
            sys.exit(1)

    logger.info("  [PASS] All locations have 5 components")
    logger.info(f"  Total rows: {len(variance_components)} (expected 10)")

    # Check for NaN or inf values
    if variance_components['value'].isnull().any():
        nan_rows = variance_components[variance_components['value'].isnull()]
        logger.warning(f"  WARNING: {len(nan_rows)} NaN values found (acceptable for intercept-only models):")
        for _, row in nan_rows.iterrows():
            logger.warning(f"    {row['location']} - {row['component']}: NaN")

    if np.isinf(variance_components['value']).any():
        logger.error("VALIDATION ERROR: Infinite values detected in variance components")
        sys.exit(1)

    logger.info("  [PASS] No infinite values")

    # ---------------------------------------------------------------------
    # 5. Save variance components
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.5.6/data/step02_variance_components.csv")
    variance_components.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved variance components to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary table
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VARIANCE COMPONENTS SUMMARY")
    logger.info("=" * 60)

    for location in ['Source', 'Destination']:
        location_data = variance_components[variance_components['location'] == location]
        logger.info(f"\n{location} Location:")
        for _, row in location_data.iterrows():
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

    for location in ['Source', 'Destination']:
        location_data = variance_components[variance_components['location'] == location]

        var_int = location_data[location_data['component'] == 'var_intercept']['value'].values[0]
        var_slope = location_data[location_data['component'] == 'var_slope']['value'].values[0]
        var_res = location_data[location_data['component'] == 'var_residual']['value'].values[0]

        logger.info(f"\n{location} Location:")
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
    logger.info(f"Extracted variance components for 2 locations (Source, Destination)")
    logger.info(f"Output: {len(variance_components)} rows (5 components x 2 locations)")
    logger.info("Ready for Step 03: Compute ICC Estimates")


if __name__ == "__main__":
    main()
