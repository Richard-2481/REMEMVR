#!/usr/bin/env python3
"""
RQ 5.2.6 Step 05: Test Intercept-Slope Correlations per Domain

Purpose:
    Test whether baseline ability (intercept) is correlated with forgetting rate
    (slope) within each domain. Negative correlation indicates high performers
    maintain advantage over time (Fan Effect).

Input:
    - data/step04_random_effects.csv (200 rows: 100 UID x 2 domains)

Output:
    - data/step05_intercept_slope_correlations.csv (2 rows: one per domain)

Decision D068 Compliance:
    Reports DUAL p-values (p_uncorrected and p_bonferroni) for all correlation tests.
    Bonferroni correction: alpha = 0.01 / 2 domains = 0.005

Interpretation:
    - Negative correlation: High performers maintain advantage (baseline predicts persistence)
    - Positive correlation: High performers decline faster (regression to mean)
    - Near-zero correlation: Baseline and forgetting rate independent

Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from scipy import stats

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step05_test_intercept_slope_correlations.log")
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


def interpret_correlation(r: float, p_bonferroni: float, alpha: float = 0.01) -> str:
    """
    Interpret correlation direction and significance.
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
    """Test intercept-slope correlations for each domain."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 05: Test Intercept-Slope Correlations per Domain")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load random effects from Step 04
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.2.6/data/step04_random_effects.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Random effects file not found: {input_path}")
        logger.error("Step 04 must complete before Step 05")
        sys.exit(1)

    random_effects = pd.read_csv(input_path)
    logger.info(f"Loaded random effects: {len(random_effects)} rows from {input_path}")

    # ---------------------------------------------------------------------
    # 2. Set up Bonferroni correction
    # ---------------------------------------------------------------------
    n_tests = 2  # 2 domains (What, Where) - When excluded
    family_alpha = 0.01  # Family-wise alpha
    bonferroni_alpha = family_alpha / n_tests

    logger.info(f"\nBonferroni correction:")
    logger.info(f"  Number of tests: {n_tests}")
    logger.info(f"  Family-wise alpha: {family_alpha}")
    logger.info(f"  Per-test alpha: {bonferroni_alpha:.4f}")

    # ---------------------------------------------------------------------
    # 3. Test correlation for each domain
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("INTERCEPT-SLOPE CORRELATION TESTS")
    logger.info("=" * 60)

    correlation_rows = []

    for domain in ['What', 'Where']:
        logger.info(f"\n{domain} Domain:")

        domain_data = random_effects[random_effects['domain'] == domain]
        n = len(domain_data)

        # Check if slopes are available
        if domain_data['Total_Slope'].isna().all():
            logger.warning(f"  No slope data available (intercept-only model)")
            correlation_rows.append({
                'domain': domain,
                'r': np.nan,
                'p_uncorrected': np.nan,
                'p_bonferroni': np.nan,
                'n': n,
                'interpretation': "NA (intercept-only model)"
            })
            continue

        intercepts = domain_data['Total_Intercept'].values
        slopes = domain_data['Total_Slope'].values

        # Pearson correlation
        r, p_uncorrected = stats.pearsonr(intercepts, slopes)

        # Bonferroni correction
        p_bonferroni = min(p_uncorrected * n_tests, 1.0)  # Cap at 1.0

        # Interpretation
        interpretation = interpret_correlation(r, p_bonferroni, family_alpha)

        logger.info(f"  n = {n}")
        logger.info(f"  r = {r:.4f}")
        logger.info(f"  p_uncorrected = {p_uncorrected:.4f}")
        logger.info(f"  p_bonferroni = {p_bonferroni:.4f}")
        logger.info(f"  Interpretation: {interpretation}")

        # Decision D068: Must report both p-values
        logger.info(f"  Decision D068: Dual p-values reported (uncorrected + Bonferroni)")

        correlation_rows.append({
            'domain': domain,
            'r': r,
            'p_uncorrected': p_uncorrected,
            'p_bonferroni': p_bonferroni,
            'n': n,
            'interpretation': interpretation
        })

    correlation_df = pd.DataFrame(correlation_rows)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking correlation results")
    logger.info("=" * 60)

    # Check expected row count
    if len(correlation_df) != 2:  # 2 domains
        logger.error(f"Expected 2 rows (2 domains), got {len(correlation_df)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(correlation_df)}: PASS")

    # Check both p-value columns present (Decision D068)
    required_cols = ['r', 'p_uncorrected', 'p_bonferroni']
    for col in required_cols:
        if col not in correlation_df.columns:
            logger.error(f"Decision D068 violation: missing column '{col}'")
            sys.exit(1)
    logger.info(f"  Decision D068 columns present: PASS")

    # Check r bounds [-1, 1]
    for _, row in correlation_df.iterrows():
        if pd.notna(row['r']):
            if row['r'] < -1 or row['r'] > 1:
                logger.error(f"Correlation out of bounds: {row['domain']} r = {row['r']}")
                sys.exit(1)
    logger.info(f"  All r in [-1, 1]: PASS")

    # Check p-value bounds [0, 1]
    for _, row in correlation_df.iterrows():
        for col in ['p_uncorrected', 'p_bonferroni']:
            if pd.notna(row[col]):
                if row[col] < 0 or row[col] > 1:
                    logger.error(f"p-value out of bounds: {row['domain']} {col} = {row[col]}")
                    sys.exit(1)
    logger.info(f"  All p-values in [0, 1]: PASS")

    # ---------------------------------------------------------------------
    # 5. Save correlation results
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step05_intercept_slope_correlations.csv")
    correlation_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved correlation results to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary and interpretation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    # Check for negative correlations (Fan Effect)
    for _, row in correlation_df.iterrows():
        if pd.notna(row['r']) and row['r'] < 0:
            if row['p_bonferroni'] < family_alpha:
                logger.info(f"  {row['domain']}: SIGNIFICANT NEGATIVE (r={row['r']:.3f}, p_bonf={row['p_bonferroni']:.4f})")
                logger.info(f"    -> High baseline performers maintain advantage over time")
            else:
                logger.info(f"  {row['domain']}: Negative trend but not significant (r={row['r']:.3f}, p_bonf={row['p_bonferroni']:.4f})")

    # Theoretical note
    logger.info("\n" + "-" * 60)
    logger.info("THEORETICAL NOTE:")
    logger.info("-" * 60)
    logger.info("Negative intercept-slope correlation is expected in learning/memory research.")
    logger.info("It reflects the 'Fan Effect': high performers start high and forget slowly,")
    logger.info("while low performers start low and may forget more quickly.")
    logger.info("However, with only 4 timepoints, slope estimates are unreliable (ICC_slope ~0.01).")
    logger.info("Interpret with caution - correlation may be attenuated by measurement error.")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 05 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Tested intercept-slope correlations for {len(correlation_df)} domains")
    logger.info("Decision D068 compliance: Both p_uncorrected and p_bonferroni reported")
    logger.info("Ready for Step 06: Compare ICC Across Domains")


if __name__ == "__main__":
    main()
