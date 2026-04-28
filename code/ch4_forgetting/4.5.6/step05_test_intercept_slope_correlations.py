#!/usr/bin/env python3
"""
RQ 5.5.6 Step 05: Test Intercept-Slope Correlations per Location

Purpose:
    Test whether baseline location memory ability (intercept) is correlated with
    forgetting rate (slope) within each location type (Source vs Destination).
    Negative correlation indicates high performers maintain advantage over time.

Input:
    - data/step04_random_effects.csv (200 rows: 100 UID x 2 locations)

Output:
    - data/step05_intercept_slope_correlations.csv (2 rows: Source, Destination)

Decision D068 Compliance:
    Reports DUAL p-values (p_uncorrected and p_bonferroni) for all correlation tests.
    Bonferroni correction: alpha = 0.05 / 2 locations = 0.025

Statistical Method:
    - Pearson correlation between random intercepts and random slopes
    - Separate test for each location (Source, Destination)
    - t-statistic: t = r * sqrt((N-2) / (1 - r^2))
    - df = N - 2 = 98 (for N=100 per location)
    - Two-tailed test (testing for any correlation, not directional)

Interpretation:
    - Negative correlation: High performers maintain advantage (baseline predicts persistence)
    - Positive correlation: High performers decline faster (regression to mean)
    - Near-zero correlation: Baseline and forgetting rate independent

Date: 2025-12-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from scipy import stats

# Set up logging
log_path = Path("results/ch5/5.5.6/logs/step05_test_intercept_slope_correlations.log")
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


def interpret_correlation(r: float, p_bonferroni: float, alpha: float = 0.05) -> str:
    """
    Interpret correlation direction and significance.

    Args:
        r: Pearson correlation coefficient
        p_bonferroni: Bonferroni-corrected p-value
        alpha: Family-wise significance threshold (default 0.05)

    Returns:
        Interpretation string
    """
    if p_bonferroni < alpha:
        if r < -0.1:
            return "Significant negative: high performers maintain advantage over time"
        elif r > 0.1:
            return "Significant positive: high performers decline faster (regression to mean)"
        else:
            return "Significant but near-zero: baseline and forgetting rate weakly related"
    else:
        return "Not significant: no reliable relationship between baseline and forgetting rate"


def main():
    """Test intercept-slope correlations for each location type."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.6 Step 05: Test Intercept-Slope Correlations per Location")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load random effects from Step 04
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.5.6/data/step04_random_effects.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Random effects file not found: {input_path}")
        logger.error("Step 04 must complete before Step 05")
        sys.exit(1)

    random_effects = pd.read_csv(input_path)
    logger.info(f"Loaded random effects: {len(random_effects)} rows from {input_path}")
    logger.info(f"Columns: {list(random_effects.columns)}")

    # Validate expected structure
    expected_rows = 200
    if len(random_effects) != expected_rows:
        logger.error(f"Expected {expected_rows} rows (100 UID x 2 locations), got {len(random_effects)}")
        sys.exit(1)

    expected_cols = ['UID', 'location', 'random_intercept', 'random_slope']
    missing_cols = [col for col in expected_cols if col not in random_effects.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    logger.info(f"  Input validation: PASS (200 rows, 4 columns)")

    # ---------------------------------------------------------------------
    # 2. Set up Bonferroni correction
    # ---------------------------------------------------------------------
    n_tests = 2  # 2 locations (Source, Destination)
    family_alpha = 0.05  # Family-wise alpha
    bonferroni_alpha = family_alpha / n_tests

    logger.info(f"\nBonferroni correction:")
    logger.info(f"  Number of tests: {n_tests}")
    logger.info(f"  Family-wise alpha: {family_alpha}")
    logger.info(f"  Per-test alpha (Bonferroni): {bonferroni_alpha:.4f}")

    # ---------------------------------------------------------------------
    # 3. Test correlation for each location
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("INTERCEPT-SLOPE CORRELATION TESTS")
    logger.info("=" * 60)

    correlation_rows = []

    for location in ['Source', 'Destination']:
        logger.info(f"\n{location} Location:")

        location_data = random_effects[random_effects['location'] == location]
        n = len(location_data)

        if n != 100:
            logger.error(f"Expected 100 participants for {location}, got {n}")
            sys.exit(1)

        intercepts = location_data['random_intercept'].values
        slopes = location_data['random_slope'].values

        # Pearson correlation
        r, p_uncorrected = stats.pearsonr(intercepts, slopes)

        # Compute t-statistic and df
        # t = r * sqrt((N-2) / (1 - r^2))
        df = n - 2
        if abs(r) < 1.0:  # Avoid division by zero for perfect correlation
            t_statistic = r * np.sqrt(df / (1 - r**2))
        else:
            t_statistic = np.inf if r > 0 else -np.inf

        # Bonferroni correction
        p_bonferroni = min(p_uncorrected * n_tests, 1.0)  # Cap at 1.0

        # Significance based on Bonferroni-corrected p-value
        significant_bonferroni = p_bonferroni < bonferroni_alpha

        # Interpretation
        interpretation = interpret_correlation(r, p_bonferroni, family_alpha)

        logger.info(f"  N = {n}")
        logger.info(f"  r = {r:.4f}")
        logger.info(f"  t-statistic = {t_statistic:.4f}")
        logger.info(f"  df = {df}")
        logger.info(f"  p_uncorrected = {p_uncorrected:.4f}")
        logger.info(f"  p_bonferroni = {p_bonferroni:.4f}")
        logger.info(f"  Significant (Bonferroni alpha={bonferroni_alpha:.4f}): {significant_bonferroni}")
        logger.info(f"  Interpretation: {interpretation}")

        # Decision D068: Must report both p-values
        logger.info(f"  Decision D068: Dual p-values reported (uncorrected + Bonferroni)")

        correlation_rows.append({
            'location': location,
            'r': r,
            'N': n,
            't_statistic': t_statistic,
            'df': df,
            'p_uncorrected': p_uncorrected,
            'p_bonferroni': p_bonferroni,
            'significant_bonferroni': significant_bonferroni
        })

    correlation_df = pd.DataFrame(correlation_rows)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking correlation results")
    logger.info("=" * 60)

    # Check expected row count
    if len(correlation_df) != 2:  # 2 locations
        logger.error(f"Expected 2 rows (2 locations), got {len(correlation_df)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(correlation_df)}: PASS")

    # Check both p-value columns present (Decision D068)
    required_cols = ['location', 'r', 'N', 't_statistic', 'df', 'p_uncorrected', 'p_bonferroni', 'significant_bonferroni']
    for col in required_cols:
        if col not in correlation_df.columns:
            logger.error(f"Decision D068/Spec violation: missing column '{col}'")
            sys.exit(1)
    logger.info(f"  All required columns present: PASS")

    # Check r bounds [-1, 1]
    for _, row in correlation_df.iterrows():
        if pd.notna(row['r']):
            if row['r'] < -1 or row['r'] > 1:
                logger.error(f"Correlation out of bounds: {row['location']} r = {row['r']}")
                sys.exit(1)
    logger.info(f"  All r in [-1, 1]: PASS")

    # Check N = 100 for both locations
    if not all(correlation_df['N'] == 100):
        logger.error(f"Expected N=100 for all locations, got {correlation_df['N'].tolist()}")
        sys.exit(1)
    logger.info(f"  All N = 100: PASS")

    # Check df = 98 for both locations
    if not all(correlation_df['df'] == 98):
        logger.error(f"Expected df=98 for all locations, got {correlation_df['df'].tolist()}")
        sys.exit(1)
    logger.info(f"  All df = 98: PASS")

    # Check p-value bounds [0, 1]
    for _, row in correlation_df.iterrows():
        for col in ['p_uncorrected', 'p_bonferroni']:
            if pd.notna(row[col]):
                if row[col] < 0 or row[col] > 1:
                    logger.error(f"p-value out of bounds: {row['location']} {col} = {row[col]}")
                    sys.exit(1)
    logger.info(f"  All p-values in [0, 1]: PASS")

    # Check p_bonferroni >= p_uncorrected (correction can only increase p-value)
    for _, row in correlation_df.iterrows():
        if row['p_bonferroni'] < row['p_uncorrected'] - 1e-10:  # Small tolerance for floating point
            logger.error(f"p_bonferroni < p_uncorrected for {row['location']}: {row['p_bonferroni']} < {row['p_uncorrected']}")
            sys.exit(1)
    logger.info(f"  All p_bonferroni >= p_uncorrected: PASS")

    # Check significant_bonferroni is boolean
    if not all(correlation_df['significant_bonferroni'].apply(lambda x: isinstance(x, (bool, np.bool_)))):
        logger.error(f"significant_bonferroni must be boolean type")
        sys.exit(1)
    logger.info(f"  significant_bonferroni is boolean: PASS")

    # ---------------------------------------------------------------------
    # 5. Save correlation results
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.5.6/data/step05_intercept_slope_correlations.csv")
    correlation_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved correlation results to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary and interpretation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    # Check for significant correlations
    for _, row in correlation_df.iterrows():
        if row['significant_bonferroni']:
            logger.info(f"  {row['location']}: SIGNIFICANT (r={row['r']:.3f}, p_bonf={row['p_bonferroni']:.4f})")
            if row['r'] < 0:
                logger.info(f"    -> High baseline performers maintain advantage over time")
            elif row['r'] > 0:
                logger.info(f"    -> High baseline performers decline faster (regression to mean)")
        else:
            logger.info(f"  {row['location']}: Not significant (r={row['r']:.3f}, p_bonf={row['p_bonferroni']:.4f})")

    # Theoretical note
    logger.info("\n" + "-" * 60)
    logger.info("THEORETICAL NOTE:")
    logger.info("-" * 60)
    logger.info("Intercept-slope correlations quantify whether baseline ability predicts")
    logger.info("forgetting rate. Negative correlations suggest high performers maintain")
    logger.info("their advantage over time (common in memory research).")
    logger.info("\nIMPORTANT: With only 4 timepoints per person, slope estimates are unreliable.")
    logger.info("The variance decomposition analysis (RQ 5.5.6) found ICC_slope ~0.01,")
    logger.info("indicating most slope variance is residual noise rather than true individual")
    logger.info("differences. Interpret these correlations with caution.")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 05 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Tested intercept-slope correlations for {len(correlation_df)} location types")
    logger.info("Decision D068 compliance: Both p_uncorrected and p_bonferroni reported")
    logger.info(f"Output: {output_path}")
    logger.info("Ready for Step 06: Compare ICC Across Locations")


if __name__ == "__main__":
    main()
