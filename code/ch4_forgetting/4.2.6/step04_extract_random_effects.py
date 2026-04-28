#!/usr/bin/env python3
"""
RQ 5.2.6 Step 04: Extract Individual Random Effects per Domain

Purpose:
    Extract individual-specific random intercepts and slopes for all 100
    participants across both domains (What, Where).

Input:
    - data/step01_fitted_models.pkl (Dict with 2 fitted MixedLM objects)
    - data/step01_model_metadata_what.yaml
    - data/step01_model_metadata_where.yaml

Output:
    - data/step04_random_effects.csv (200 rows: 100 UID x 2 domains)

Note: This file is REQUIRED INPUT for RQ 5.2.7 (domain-based clustering).

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
log_path = Path("results/ch5/5.2.6/logs/step04_extract_random_effects.log")
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


def extract_random_effects_from_model(result, metadata: dict, domain: str) -> pd.DataFrame:
    """
    Extract random effects (intercepts and slopes) for each participant from a fitted model.

    Returns:
        DataFrame with columns: UID, domain, Total_Intercept, Total_Slope, intercept_se, slope_se
    """
    logger.info(f"\nExtracting random effects for {domain} domain")

    random_structure = metadata.get('random_structure', 'Unknown')
    logger.info(f"  Random structure: {random_structure}")

    # Get random effects from model
    # For statsmodels MixedLM, random_effects is a dict {group: effects}
    random_effects_dict = result.random_effects

    rows = []
    for uid, effects in random_effects_dict.items():
        row = {
            'UID': uid,
            'domain': domain,
        }

        # Effects can be a pandas Series or array
        if isinstance(effects, pd.Series):
            effects_values = effects.values
            effects_index = effects.index.tolist()
        else:
            effects_values = np.array(effects)
            effects_index = ['Group', 'log_TSVR'] if len(effects_values) > 1 else ['Group']

        # Extract intercept (always present)
        row['Total_Intercept'] = float(effects_values[0])

        # Extract slope (if random slopes fit)
        if random_structure in ['Full', 'Uncorrelated'] and len(effects_values) >= 2:
            row['Total_Slope'] = float(effects_values[1])
        else:
            row['Total_Slope'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Get standard errors from model (if available)
    # Note: statsmodels doesn't always provide SE for BLUPs directly
    # We'll estimate from the variance components
    cov_re = result.cov_re

    # Approximate SE from variance components
    var_intercept = float(cov_re.iloc[0, 0])
    df['intercept_se'] = np.sqrt(var_intercept)

    if random_structure in ['Full', 'Uncorrelated'] and cov_re.shape[0] >= 2:
        var_slope = float(cov_re.iloc[1, 1])
        df['slope_se'] = np.sqrt(var_slope)
    else:
        df['slope_se'] = np.nan

    logger.info(f"  Extracted {len(df)} participants")
    logger.info(f"  Total_Intercept range: [{df['Total_Intercept'].min():.4f}, {df['Total_Intercept'].max():.4f}]")

    if pd.notna(df['Total_Slope'].iloc[0]):
        logger.info(f"  Total_Slope range: [{df['Total_Slope'].min():.4f}, {df['Total_Slope'].max():.4f}]")
    else:
        logger.info(f"  Total_Slope: NA (intercept-only model)")

    return df


def main():
    """Extract individual random effects for each domain."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 04: Extract Individual Random Effects per Domain")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load fitted models from Step 01
    # ---------------------------------------------------------------------
    models_path = Path("results/ch5/5.2.6/data/step01_fitted_models.pkl")

    if not models_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Models file not found: {models_path}")
        logger.error("Step 01 must complete before Step 04")
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
        with open(metadata_path, 'r') as f:
            all_metadata[domain] = yaml.safe_load(f)
        logger.info(f"Loaded metadata for {domain}: {all_metadata[domain]['random_structure']} structure")

    # ---------------------------------------------------------------------
    # 3. Extract random effects for each domain
    # ---------------------------------------------------------------------
    random_effects_dfs = []

    for domain in ['What', 'Where']:
        result = fitted_models[domain]
        metadata = all_metadata[domain]

        if result is None:
            logger.error(f"No fitted model for {domain} domain")
            sys.exit(1)

        df_re = extract_random_effects_from_model(result, metadata, domain)
        random_effects_dfs.append(df_re)

    random_effects = pd.concat(random_effects_dfs, ignore_index=True)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking random effects structure")
    logger.info("=" * 60)

    # Check expected row count: 100 participants x 2 domains = 200
    expected_rows = 100 * 2
    if len(random_effects) != expected_rows:
        logger.error(f"Expected {expected_rows} rows (100 UID x 2 domains), got {len(random_effects)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(random_effects)} (expected {expected_rows}): PASS")

    # Check all participants present
    n_unique_uid = random_effects['UID'].nunique()
    if n_unique_uid != 100:
        logger.error(f"Expected 100 unique UIDs, got {n_unique_uid}")
        sys.exit(1)
    logger.info(f"  Unique UIDs: {n_unique_uid}: PASS")

    # Check all domains present
    domains_present = set(random_effects['domain'].unique())
    expected_domains = {'What', 'Where'}
    if domains_present != expected_domains:
        logger.error(f"Expected domains {expected_domains}, got {domains_present}")
        sys.exit(1)
    logger.info(f"  Domains present: {domains_present}: PASS")

    # Check each participant has all domains
    uid_domain_counts = random_effects.groupby('UID')['domain'].nunique()
    incomplete_uids = uid_domain_counts[uid_domain_counts != 2]
    if len(incomplete_uids) > 0:
        logger.error(f"Incomplete domain coverage for UIDs: {list(incomplete_uids.index)}")
        sys.exit(1)
    logger.info(f"  All UIDs have both domains: PASS")

    # Check no NaN in Total_Intercept
    nan_intercept = random_effects['Total_Intercept'].isna().sum()
    if nan_intercept > 0:
        logger.error(f"Missing Total_Intercept values: {nan_intercept}")
        sys.exit(1)
    logger.info(f"  No missing Total_Intercept values: PASS")

    # Check intercept_se > 0
    if (random_effects['intercept_se'] <= 0).any():
        logger.error("intercept_se contains non-positive values")
        sys.exit(1)
    logger.info(f"  All intercept_se > 0: PASS")

    # ---------------------------------------------------------------------
    # 5. Save random effects
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step04_random_effects.csv")
    random_effects.to_csv(output_path, index=False)
    logger.info(f"\nSaved random effects to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary statistics
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("RANDOM EFFECTS SUMMARY")
    logger.info("=" * 60)

    for domain in ['What', 'Where']:
        domain_data = random_effects[random_effects['domain'] == domain]
        logger.info(f"\n{domain} Domain (n={len(domain_data)}):")
        logger.info(f"  Total_Intercept: mean={domain_data['Total_Intercept'].mean():.4f}, SD={domain_data['Total_Intercept'].std():.4f}")
        if pd.notna(domain_data['Total_Slope'].iloc[0]):
            logger.info(f"  Total_Slope: mean={domain_data['Total_Slope'].mean():.4f}, SD={domain_data['Total_Slope'].std():.4f}")
        else:
            logger.info(f"  Total_Slope: NA (intercept-only model)")

    # Correlation between domains (intercepts)
    what_int = random_effects[random_effects['domain'] == 'What'].set_index('UID')['Total_Intercept']
    where_int = random_effects[random_effects['domain'] == 'Where'].set_index('UID')['Total_Intercept']

    # Align indices
    common_uids = what_int.index.intersection(where_int.index)
    cross_domain_r = what_int[common_uids].corr(where_int[common_uids])
    logger.info(f"\nCross-domain intercept correlation (What-Where): r={cross_domain_r:.3f}")

    # If slopes available
    what_slope = random_effects[random_effects['domain'] == 'What'].set_index('UID')['Total_Slope']
    where_slope = random_effects[random_effects['domain'] == 'Where'].set_index('UID')['Total_Slope']
    if pd.notna(what_slope.iloc[0]) and pd.notna(where_slope.iloc[0]):
        slope_r = what_slope[common_uids].corr(where_slope[common_uids])
        logger.info(f"Cross-domain slope correlation (What-Where): r={slope_r:.3f}")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 04 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Extracted random effects for {len(random_effects)} observations")
    logger.info(f"  100 participants x 2 domains = {len(random_effects)} rows")
    logger.info("Output file ready for RQ 5.2.7 (domain-based clustering)")
    logger.info("Ready for Step 05: Test Intercept-Slope Correlations")


if __name__ == "__main__":
    main()
