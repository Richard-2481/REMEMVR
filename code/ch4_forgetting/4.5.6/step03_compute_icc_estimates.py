#!/usr/bin/env python3
"""
RQ 5.5.6 Step 03: Compute ICC Estimates per Location

Purpose:
    Compute Intraclass Correlation Coefficients (ICC) for intercepts and slopes
    to quantify proportion of variance attributable to between-person differences.

Input:
    - data/step02_variance_components.csv (10 rows: 5 components x 2 locations)

Output:
    - data/step03_icc_estimates.csv (6 rows: 3 ICC types x 2 locations)

ICC Types:
    - ICC_intercept: var_intercept / (var_intercept + var_residual)
    - ICC_slope_simple: var_slope / (var_slope + var_residual)
    - ICC_slope_conditional: ICC at Day 6 timepoint accounting for intercept-slope correlation

Interpretation Thresholds (Cicchetti 1994):
    - Poor: < 0.40
    - Fair: 0.40-0.59
    - Good: 0.60-0.74
    - Excellent: >= 0.75

Author: Claude (g_code agent)
Date: 2025-12-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
log_path = Path("results/ch5/5.5.6/logs/step03_compute_icc_estimates.log")
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


def interpret_icc(icc_value: float) -> str:
    """
    Interpret ICC value according to Cicchetti (1994) thresholds.

    Returns:
        Interpretation category (Poor, Fair, Good, Excellent)
    """
    if pd.isna(icc_value):
        return "NA"
    elif icc_value < 0.40:
        return "Poor"
    elif icc_value < 0.60:
        return "Fair"
    elif icc_value < 0.75:
        return "Good"
    else:
        return "Excellent"


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
    """Compute ICC estimates for each location."""

    logger.info("=" * 60)
    logger.info("RQ 5.5.6 Step 03: Compute ICC Estimates per Location")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load variance components from Step 02
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.5.6/data/step02_variance_components.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Variance components file not found: {input_path}")
        logger.error("Step 02 must complete before Step 03")
        sys.exit(1)

    variance_df = pd.read_csv(input_path)
    logger.info(f"Loaded variance components: {len(variance_df)} rows from {input_path}")

    # ---------------------------------------------------------------------
    # 2. Define timepoint for conditional ICC
    # ---------------------------------------------------------------------
    # For log_TSVR, Day 6 ~ 144 hours -> log(145) ~ 4.98
    timepoint_day6 = np.log(144 + 1)  # log(145) for Day 6
    logger.info(f"Conditional ICC timepoint: log(145) = {timepoint_day6:.3f} (Day 6)")

    # ---------------------------------------------------------------------
    # 3. Compute ICC for each location
    # ---------------------------------------------------------------------
    icc_rows = []

    for location in ['Source', 'Destination']:
        logger.info(f"\nComputing ICC for {location} location")

        location_data = variance_df[variance_df['location'] == location]

        # Get variance components
        var_int = location_data[location_data['component'] == 'var_intercept']['value'].values[0]
        var_slope = location_data[location_data['component'] == 'var_slope']['value'].values[0]
        cov_int_slope = location_data[location_data['component'] == 'cov_int_slope']['value'].values[0]
        var_res = location_data[location_data['component'] == 'var_residual']['value'].values[0]

        logger.info(f"  var_intercept: {var_int:.6f}")
        logger.info(f"  var_slope: {var_slope:.6f}")
        logger.info(f"  cov_int_slope: {cov_int_slope:.6f}")
        logger.info(f"  var_residual: {var_res:.6f}")

        # ICC for intercept (baseline ability)
        icc_intercept = var_int / (var_int + var_res)
        interp = interpret_icc(icc_intercept)
        icc_rows.append({
            'location': location,
            'icc_type': 'ICC_intercept',
            'value': icc_intercept,
            'interpretation': interp
        })
        logger.info(f"  ICC_intercept: {icc_intercept:.4f} ({interp})")

        # ICC for slope (simple - no timepoint adjustment)
        icc_slope_simple = var_slope / (var_slope + var_res)
        interp = interpret_icc(icc_slope_simple)
        icc_rows.append({
            'location': location,
            'icc_type': 'ICC_slope_simple',
            'value': icc_slope_simple,
            'interpretation': interp
        })
        logger.info(f"  ICC_slope_simple: {icc_slope_simple:.4f} ({interp})")

        # ICC conditional at Day 6 (accounts for intercept-slope covariance)
        icc_slope_cond = compute_icc_conditional(var_int, var_slope, cov_int_slope, var_res, timepoint_day6)
        interp = interpret_icc(icc_slope_cond)
        icc_rows.append({
            'location': location,
            'icc_type': 'ICC_slope_conditional',
            'value': icc_slope_cond,
            'interpretation': interp
        })
        logger.info(f"  ICC_slope_conditional (Day 6): {icc_slope_cond:.4f} ({interp})")

    icc_df = pd.DataFrame(icc_rows)

    # ---------------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking ICC constraints")
    logger.info("=" * 60)

    # Check ICC bounds [0, 1]
    for _, row in icc_df.iterrows():
        if pd.notna(row['value']):
            if row['value'] < 0 or row['value'] > 1:
                logger.error(f"ICC OUT OF BOUNDS: {row['location']} {row['icc_type']} = {row['value']}")
                sys.exit(1)

    logger.info("  All ICC values in [0, 1]: PASS")
    logger.info(f"  Total rows: {len(icc_df)} (expected 6)")

    # Check expected row count
    if len(icc_df) != 6:  # 3 ICC types x 2 locations
        logger.error(f"Expected 6 ICC rows (3 types x 2 locations), got {len(icc_df)}")
        sys.exit(1)

    logger.info("  Row count correct: PASS")

    # Check for NaN values
    nan_count = icc_df['value'].isna().sum()
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN ICC values")
        sys.exit(1)

    logger.info("  No NaN values: PASS")

    # ---------------------------------------------------------------------
    # 5. Save ICC estimates
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.5.6/data/step03_icc_estimates.csv")
    icc_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved ICC estimates to: {output_path}")

    # ---------------------------------------------------------------------
    # 6. Summary and interpretation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("ICC ESTIMATES SUMMARY")
    logger.info("=" * 60)

    for location in ['Source', 'Destination']:
        location_icc = icc_df[icc_df['location'] == location]
        logger.info(f"\n{location} Location:")
        for _, row in location_icc.iterrows():
            val = row['value']
            logger.info(f"  {row['icc_type']}: {val:.4f} ({row['interpretation']})")

    # Highlight key findings
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    # ICC intercept comparison
    source_int = icc_df[(icc_df['location'] == 'Source') & (icc_df['icc_type'] == 'ICC_intercept')]['value'].values[0]
    dest_int = icc_df[(icc_df['location'] == 'Destination') & (icc_df['icc_type'] == 'ICC_intercept')]['value'].values[0]
    logger.info(f"ICC_intercept: Source={source_int:.3f}, Destination={dest_int:.3f}")

    if source_int >= 0.40 and dest_int >= 0.40:
        logger.info("  -> Fair or better reliability for BOTH locations")
    elif source_int >= 0.40 or dest_int >= 0.40:
        logger.info("  -> Fair or better reliability for at least one location")

    # ICC slope comparison
    source_slope = icc_df[(icc_df['location'] == 'Source') & (icc_df['icc_type'] == 'ICC_slope_simple')]['value'].values[0]
    dest_slope = icc_df[(icc_df['location'] == 'Destination') & (icc_df['icc_type'] == 'ICC_slope_simple')]['value'].values[0]

    logger.info(f"ICC_slope_simple: Source={source_slope:.4f}, Destination={dest_slope:.4f}")
    if source_slope < 0.05 and dest_slope < 0.05:
        logger.info("  -> LOW ICC_slope in both locations (near-zero variance)")

    # ---------------------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 03 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Computed {len(icc_df)} ICC estimates (3 types x 2 locations)")
    logger.info("Ready for Step 04: Extract Individual Random Effects")


if __name__ == "__main__":
    main()
