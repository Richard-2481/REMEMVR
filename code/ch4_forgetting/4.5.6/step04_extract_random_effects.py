#!/usr/bin/env python3
"""
RQ 5.5.6 Step 04: Extract Individual Random Effects per Location

Purpose:
    Extract individual-specific random intercepts and slopes for all 100
    participants across both locations (Source, Destination).

Input:
    - data/step01_source_lmm_model.pkl (Fitted Source LMM)
    - data/step01_destination_lmm_model.pkl (Fitted Destination LMM)
    - data/step01_model_metadata_source.yaml
    - data/step01_model_metadata_destination.yaml

Output:
    - data/step04_random_effects.csv (200 rows: 100 UID x 2 locations)
      Columns: UID, location, random_intercept, random_slope

CRITICAL: This file is REQUIRED INPUT for RQ 5.5.7 (location-based clustering).
RQ 5.5.7 cannot proceed without this file.

Date: 2025-12-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import pickle
import yaml
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Set up logging
log_path = Path("results/ch5/5.5.6/logs/step04_extract_random_effects.log")
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


def extract_random_effects_from_model(result: MixedLMResults, metadata: dict, location: str) -> pd.DataFrame:
    """
    Extract random effects (intercepts and slopes) for each participant from a fitted model.

    Args:
        result: Fitted MixedLM model from statsmodels
        metadata: Model metadata dict (contains random_structure info)
        location: Location name (Source or Destination)

    Returns:
        DataFrame with columns: UID, location, random_intercept, random_slope
    """
    logger.info(f"\nExtracting random effects for {location} location")

    random_structure = metadata.get('random_structure', 'Unknown')
    logger.info(f"  Random structure: {random_structure}")
    logger.info(f"  Expected n_groups: 100")

    # Get random effects from model
    # For statsmodels MixedLM, random_effects is a dict {group: effects}
    random_effects_dict = result.random_effects

    rows = []
    for uid, effects in random_effects_dict.items():
        row = {
            'UID': uid,
            'location': location,
        }

        # Effects can be a pandas Series or array
        if isinstance(effects, pd.Series):
            effects_values = effects.values
            effects_index = effects.index.tolist()
        else:
            effects_values = np.array(effects)
            effects_index = ['Group', 'log_TSVR'] if len(effects_values) > 1 else ['Group']

        # Extract intercept (always present)
        row['random_intercept'] = float(effects_values[0])

        # Extract slope (if random slopes fit)
        if random_structure in ['Full', 'Uncorrelated'] and len(effects_values) >= 2:
            row['random_slope'] = float(effects_values[1])
        else:
            row['random_slope'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"  Extracted {len(df)} participants")
    logger.info(f"  random_intercept range: [{df['random_intercept'].min():.4f}, {df['random_intercept'].max():.4f}]")

    if pd.notna(df['random_slope'].iloc[0]):
        logger.info(f"  random_slope range: [{df['random_slope'].min():.4f}, {df['random_slope'].max():.4f}]")
    else:
        logger.info(f"  random_slope: NA (intercept-only model)")

    return df


def main():
    """Extract individual random effects for each location."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.6 Step 04: Extract Individual Random Effects per Location")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load fitted models from Step 01
    # ---------------------------------------------------------------------
    models = {}
    for location in ['source', 'destination']:
        model_path = Path(f"results/ch5/5.5.6/data/step01_{location}_lmm_model.pkl")

        if not model_path.exists():
            logger.error(f"EXPECTATIONS ERROR: Model file not found: {model_path}")
            logger.error("Step 01 must complete before Step 04")
            sys.exit(1)

        # Load model using MixedLMResults.load() method (correct way per REMEMVR Data Conventions)
        models[location] = MixedLMResults.load(str(model_path))

    logger.info(f"Loaded fitted models: {list(models.keys())}")

    # ---------------------------------------------------------------------
    # 2. Load metadata for each location
    # ---------------------------------------------------------------------
    all_metadata = {}
    for location in ['source', 'destination']:
        metadata_path = Path(f"results/ch5/5.5.6/data/step01_model_metadata_{location}.yaml")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            all_metadata[location] = yaml.safe_load(f)
        logger.info(f"Loaded metadata for {location}: {all_metadata[location]['random_structure']} structure")

    # ---------------------------------------------------------------------
    # 3. Extract random effects for each location
    # ---------------------------------------------------------------------
    random_effects_dfs = []

    for location in ['source', 'destination']:
        result = models[location]
        metadata = all_metadata[location]

        if result is None:
            logger.error(f"No fitted model for {location} location")
            sys.exit(1)

        # Capitalize location name for output (Source, Destination)
        location_name = location.capitalize()
        df_re = extract_random_effects_from_model(result, metadata, location_name)
        random_effects_dfs.append(df_re)

    random_effects = pd.concat(random_effects_dfs, ignore_index=True)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking random effects structure")
    logger.info("=" * 60)

    # Check expected row count: 100 participants x 2 locations = 200
    expected_rows = 100 * 2
    if len(random_effects) != expected_rows:
        logger.error(f"Expected {expected_rows} rows (100 UID x 2 locations), got {len(random_effects)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(random_effects)} (expected {expected_rows}): PASS")

    # Check all participants present
    n_unique_uid = random_effects['UID'].nunique()
    if n_unique_uid != 100:
        logger.error(f"Expected 100 unique UIDs, got {n_unique_uid}")
        sys.exit(1)
    logger.info(f"  Unique UIDs: {n_unique_uid}: PASS")

    # Check all locations present
    locations_present = set(random_effects['location'].unique())
    expected_locations = {'Source', 'Destination'}
    if locations_present != expected_locations:
        logger.error(f"Expected locations {expected_locations}, got {locations_present}")
        sys.exit(1)
    logger.info(f"  Locations present: {locations_present}: PASS")

    # Check each participant has all locations
    uid_location_counts = random_effects.groupby('UID')['location'].nunique()
    incomplete_uids = uid_location_counts[uid_location_counts != 2]
    if len(incomplete_uids) > 0:
        logger.error(f"Incomplete location coverage for UIDs: {list(incomplete_uids.index)}")
        sys.exit(1)
    logger.info(f"  All UIDs have both locations: PASS")

    # Check no NaN in random_intercept
    nan_intercept = random_effects['random_intercept'].isna().sum()
    if nan_intercept > 0:
        logger.error(f"Missing random_intercept values: {nan_intercept}")
        sys.exit(1)
    logger.info(f"  No missing random_intercept values: PASS")

    # Check no NaN in random_slope (both models should have Full structure)
    nan_slope = random_effects['random_slope'].isna().sum()
    if nan_slope > 0:
        logger.error(f"Missing random_slope values: {nan_slope}")
        logger.error("Both models should have Full random structure with slopes")
        sys.exit(1)
    logger.info(f"  No missing random_slope values: PASS")

    # Check no duplicate UID-location pairs
    duplicate_check = random_effects.groupby(['UID', 'location']).size()
    duplicates = duplicate_check[duplicate_check > 1]
    if len(duplicates) > 0:
        logger.error(f"Found {len(duplicates)} duplicate UID-location pairs")
        sys.exit(1)
    logger.info(f"  No duplicate UID-location pairs: PASS")

    # Check for inf values
    inf_check = np.isinf(random_effects[['random_intercept', 'random_slope']]).any()
    if inf_check.any():
        logger.error("Found infinite values in random effects")
        sys.exit(1)
    logger.info(f"  No infinite values: PASS")

    # ---------------------------------------------------------------------
    # 5. Save random effects
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.5.6/data/step04_random_effects.csv")
    random_effects.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved random effects to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary statistics
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("RANDOM EFFECTS SUMMARY")
    logger.info("=" * 60)

    for location in ['Source', 'Destination']:
        location_data = random_effects[random_effects['location'] == location]
        logger.info(f"\n{location} Location (n={len(location_data)}):")
        logger.info(f"  random_intercept: mean={location_data['random_intercept'].mean():.4f}, SD={location_data['random_intercept'].std():.4f}")
        if pd.notna(location_data['random_slope'].iloc[0]):
            logger.info(f"  random_slope: mean={location_data['random_slope'].mean():.4f}, SD={location_data['random_slope'].std():.4f}")
        else:
            logger.info(f"  random_slope: NA (intercept-only model)")

    # Correlation between locations (intercepts)
    source_int = random_effects[random_effects['location'] == 'Source'].set_index('UID')['random_intercept']
    dest_int = random_effects[random_effects['location'] == 'Destination'].set_index('UID')['random_intercept']

    # Align indices
    common_uids = source_int.index.intersection(dest_int.index)
    cross_location_r = source_int[common_uids].corr(dest_int[common_uids])
    logger.info(f"\nCross-location intercept correlation (Source-Destination): r={cross_location_r:.3f}")

    # If slopes available
    source_slope = random_effects[random_effects['location'] == 'Source'].set_index('UID')['random_slope']
    dest_slope = random_effects[random_effects['location'] == 'Destination'].set_index('UID')['random_slope']
    if pd.notna(source_slope.iloc[0]) and pd.notna(dest_slope.iloc[0]):
        slope_r = source_slope[common_uids].corr(dest_slope[common_uids])
        logger.info(f"Cross-location slope correlation (Source-Destination): r={slope_r:.3f}")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 04 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Extracted random effects for {len(random_effects)} observations")
    logger.info(f"  100 participants x 2 locations = {len(random_effects)} rows")
    logger.info("CRITICAL: Output file ready for RQ 5.5.7 (location-based clustering)")
    logger.info("Ready for Step 05: Test Intercept-Slope Correlations")


if __name__ == "__main__":
    main()
