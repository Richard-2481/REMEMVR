#!/usr/bin/env python3
"""
RQ 5.2.6 Step 03: Compute ICC Estimates per Domain

Purpose:
    Compute Intraclass Correlation Coefficients (ICC) for intercepts and slopes
    to quantify between-person versus within-person variance.

Input:
    - data/step02_variance_components.csv (10 rows: 5 components x 2 domains)

Output:
    - data/step03_icc_estimates.csv (6 rows: 3 ICC types x 2 domains)

ICC Types:
    - ICC_intercept: var_intercept / (var_intercept + var_residual)
    - ICC_slope_simple: var_slope / (var_slope + var_residual)
    - ICC_slope_conditional: ICC at specific timepoint accounting for intercept-slope correlation

Interpretation Thresholds (per 1_concept.md):
    - ICC < 0.20: Low (mostly measurement noise)
    - 0.20 <= ICC < 0.40: Moderate (mixed trait and state variance)
    - ICC >= 0.40: Substantial (trait-like individual differences dominate)

IMPORTANT NOTE ON ICC_slope:
    Per Session 2025-12-03 14:30 ICC investigation, with only 4 timepoints per participant,
    individual slopes are estimated with massive uncertainty (~93% shrinkage from sparse design).
    Low ICC_slope values may reflect design limitations rather than absence of true differences.
    The thesis should interpret ICC_slope cautiously.

Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step03_compute_icc_estimates.log")
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


def interpret_icc(icc_value: float) -> tuple:
    """
    Interpret ICC value according to thresholds from 1_concept.md.

    Returns:
        (interpretation, threshold_used)
    """
    if pd.isna(icc_value):
        return "NA", "NA"
    elif icc_value < 0.20:
        return "Low", "<0.20"
    elif icc_value < 0.40:
        return "Moderate", "0.20-0.40"
    else:
        return "Substantial", ">=0.40"


def compute_icc_conditional(var_int: float, var_slope: float, cov_int_slope: float,
                            var_residual: float, timepoint: float) -> float:
    """
    Compute ICC at a specific timepoint, accounting for intercept-slope correlation.

    Formula:
        var_total(t) = var_int + 2*cov_int_slope*t + var_slope*t^2 + var_residual
        ICC(t) = (var_int + 2*cov_int_slope*t + var_slope*t^2) / var_total(t)

    For log-transformed time, timepoint should be log(hours + 1).
    """
    if pd.isna(var_slope):
        return np.nan

    # Between-person variance at timepoint
    var_between = var_int + 2 * cov_int_slope * timepoint + var_slope * (timepoint ** 2)

    # Total variance at timepoint
    var_total = var_between + var_residual

    if var_total <= 0:
        return np.nan

    icc = var_between / var_total

    # Bound ICC to [0, 1]
    return max(0.0, min(1.0, icc))


def main():
    """Compute ICC estimates for each domain."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 03: Compute ICC Estimates per Domain")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load variance components from Step 02
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.2.6/data/step02_variance_components.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Variance components file not found: {input_path}")
        logger.error("Step 02 must complete before Step 03")
        sys.exit(1)

    variance_df = pd.read_csv(input_path)
    logger.info(f"Loaded variance components: {len(variance_df)} rows from {input_path}")

    # ---------------------------------------------------------------------
    # 2. Define timepoint for conditional ICC
    # ---------------------------------------------------------------------
    # For log_TSVR, Day 6 ~ 144 hours -> log(145) ≈ 4.98
    # But we also want interpretable at multiple timepoints
    timepoint_day6 = np.log(144 + 1)  # log(145) for Day 6
    logger.info(f"Conditional ICC timepoint: log(145) = {timepoint_day6:.3f} (Day 6)")

    # ---------------------------------------------------------------------
    # 3. Compute ICC for each domain
    # ---------------------------------------------------------------------
    icc_rows = []

    for domain in ['What', 'Where']:
        logger.info(f"\nComputing ICC for {domain} domain")

        domain_data = variance_df[variance_df['domain'] == domain]

        # Get variance components
        var_int = domain_data[domain_data['component'] == 'var_intercept']['value'].values[0]
        var_slope = domain_data[domain_data['component'] == 'var_slope']['value'].values[0]
        cov_int_slope = domain_data[domain_data['component'] == 'cov_int_slope']['value'].values[0]
        var_res = domain_data[domain_data['component'] == 'var_residual']['value'].values[0]

        logger.info(f"  var_intercept: {var_int:.6f}")
        logger.info(f"  var_slope: {var_slope:.6f}" if pd.notna(var_slope) else "  var_slope: NA")
        logger.info(f"  cov_int_slope: {cov_int_slope:.6f}" if pd.notna(cov_int_slope) else "  cov_int_slope: NA")
        logger.info(f"  var_residual: {var_res:.6f}")

        # ICC for intercept (baseline ability)
        icc_intercept = var_int / (var_int + var_res)
        interp, thresh = interpret_icc(icc_intercept)
        icc_rows.append({
            'domain': domain,
            'icc_type': 'intercept',
            'icc_value': icc_intercept,
            'interpretation': interp,
            'threshold_used': thresh
        })
        logger.info(f"  ICC_intercept: {icc_intercept:.4f} ({interp})")

        # ICC for slope (simple - no timepoint adjustment)
        if pd.notna(var_slope) and var_slope > 0:
            icc_slope_simple = var_slope / (var_slope + var_res)
        else:
            icc_slope_simple = np.nan
        interp, thresh = interpret_icc(icc_slope_simple)
        icc_rows.append({
            'domain': domain,
            'icc_type': 'slope_simple',
            'icc_value': icc_slope_simple,
            'interpretation': interp,
            'threshold_used': thresh
        })
        icc_slope_str = f"{icc_slope_simple:.4f}" if pd.notna(icc_slope_simple) else "NA"
        logger.info(f"  ICC_slope_simple: {icc_slope_str} ({interp})")

        # ICC conditional at Day 6 (accounts for intercept-slope covariance)
        icc_slope_cond = compute_icc_conditional(var_int, var_slope, cov_int_slope, var_res, timepoint_day6)
        interp, thresh = interpret_icc(icc_slope_cond)
        icc_rows.append({
            'domain': domain,
            'icc_type': 'slope_conditional',
            'icc_value': icc_slope_cond,
            'interpretation': interp,
            'threshold_used': thresh
        })
        icc_cond_str = f"{icc_slope_cond:.4f}" if pd.notna(icc_slope_cond) else "NA"
        logger.info(f"  ICC_slope_conditional (Day 6): {icc_cond_str} ({interp})")

    icc_df = pd.DataFrame(icc_rows)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking ICC constraints")
    logger.info("=" * 60)

    # Check ICC bounds [0, 1]
    for _, row in icc_df.iterrows():
        if pd.notna(row['icc_value']):
            if row['icc_value'] < 0 or row['icc_value'] > 1:
                logger.error(f"ICC OUT OF BOUNDS: {row['domain']} {row['icc_type']} = {row['icc_value']}")
                sys.exit(1)

    logger.info("  All ICC values in [0, 1]: PASS")
    logger.info(f"  Total rows: {len(icc_df)} (expected 6)")

    # Check expected row count
    if len(icc_df) != 6:  # 3 ICC types x 2 domains
        logger.error(f"Expected 6 ICC rows (3 types x 2 domains), got {len(icc_df)}")
        sys.exit(1)

    logger.info("  Row count correct: PASS")

    # ---------------------------------------------------------------------
    # 5. Save ICC estimates
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step03_icc_estimates.csv")
    icc_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved ICC estimates to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary and interpretation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("ICC ESTIMATES SUMMARY")
    logger.info("=" * 60)

    for domain in ['What', 'Where']:
        domain_icc = icc_df[icc_df['domain'] == domain]
        logger.info(f"\n{domain} Domain:")
        for _, row in domain_icc.iterrows():
            val = row['icc_value']
            if pd.isna(val):
                val_str = "NA"
            else:
                val_str = f"{val:.4f}"
            logger.info(f"  {row['icc_type']}: {val_str} ({row['interpretation']})")

    # Highlight key findings
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    # ICC intercept comparison
    what_int = icc_df[(icc_df['domain'] == 'What') & (icc_df['icc_type'] == 'intercept')]['icc_value'].values[0]
    where_int = icc_df[(icc_df['domain'] == 'Where') & (icc_df['icc_type'] == 'intercept')]['icc_value'].values[0]
    logger.info(f"ICC_intercept: What={what_int:.3f}, Where={where_int:.3f}")
    if what_int >= 0.40 and where_int >= 0.40:
        logger.info("  -> SUBSTANTIAL between-person variance in baseline for BOTH domains")
    elif what_int >= 0.40 or where_int >= 0.40:
        logger.info("  -> SUBSTANTIAL between-person variance in at least one domain")

    # ICC slope comparison
    what_slope = icc_df[(icc_df['domain'] == 'What') & (icc_df['icc_type'] == 'slope_simple')]['icc_value'].values[0]
    where_slope = icc_df[(icc_df['domain'] == 'Where') & (icc_df['icc_type'] == 'slope_simple')]['icc_value'].values[0]

    if pd.notna(what_slope) and pd.notna(where_slope):
        logger.info(f"ICC_slope_simple: What={what_slope:.4f}, Where={where_slope:.4f}")
        if what_slope < 0.05 and where_slope < 0.05:
            logger.info("  -> LOW ICC_slope in both domains")
            logger.info("  -> CAUTION: May reflect 4-timepoint design limitation (see ICC investigation)")
            logger.info("  -> Cannot reliably claim absence of individual slope differences")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 03 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Computed {len(icc_df)} ICC estimates (3 types x 2 domains)")
    logger.info("Ready for Step 04: Extract Individual Random Effects")


if __name__ == "__main__":
    main()
