#!/usr/bin/env python3
"""
Step 03: Extract 6 Segment-Paradigm Slopes
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Extract 6 segment-paradigm-specific slopes (Early IFR, Early ICR,
Early IRE, Late IFR, Late ICR, Late IRE) via linear combinations of fixed
effects with delta method standard errors.
"""

import sys
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step03_extract_segment_slopes.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_slopes_from_piecewise_lmm(result):
    """
    Extract segment-paradigm slopes from piecewise LMM.

    The model has reference categories:
    - Segment: Early (reference)
    - Paradigm: IFR (reference)

    To get each slope, we need to sum the appropriate coefficients:

    Early IFR slope = Days_within
    Early ICR slope = Days_within + Days_within:paradigmEarly IRE slope = Days_within + Days_within:paradigmLate IFR slope = Days_within + Days_within:Segment[Late]
    Late ICR slope = Days_within + Days_within:Segment[Late] + Days_within:paradigm+ Days_within:Segment[Late]:paradigmLate IRE slope = Days_within + Days_within:Segment[Late] + Days_within:paradigm+ Days_within:Segment[Late]:paradigm"""
    # Get fixed effects
    fe_params = result.fe_params
    fe_cov = result.cov_params()

    # Map coefficient names (statsmodels uses T.* notation for categorical levels)
    coef_names = {
        'Days_within': 'Days_within',
        'Days_within:Segment[Late]': 'Days_within:Segment[T.Late]',
        'Days_within:paradigm': 'Days_within:paradigm_code[T.ICR]',
        'Days_within:paradigm': 'Days_within:paradigm_code[T.IRE]',
        'Days_within:Segment[Late]:paradigm': 'Days_within:Segment[T.Late]:paradigm_code[T.ICR]',
        'Days_within:Segment[Late]:paradigm': 'Days_within:Segment[T.Late]:paradigm_code[T.IRE]'
    }

    # Check what coefficients exist
    available_coefs = list(fe_params.index)
    logger.info(f"Available coefficients: {available_coefs}")

    # Extract coefficient values
    def get_coef(name):
        actual_name = coef_names.get(name, name)
        if actual_name in fe_params.index:
            return fe_params[actual_name]
        else:
            logger.warning(f"Coefficient {name} ({actual_name}) not found, using 0")
            return 0.0

    # Define contrast vectors for delta method
    # Order of coefficients in the model (fixed effects only)
    fe_names = [n for n in fe_params.index if 'Var' not in n and 'Cov' not in n]
    logger.info(f"Fixed effect names: {fe_names}")

    # Create contrast matrix for each slope
    slopes_data = []

    # Define the 6 slopes
    slope_definitions = [
        ('Early', 'IFR', ['Days_within']),
        ('Early', 'ICR', ['Days_within', 'Days_within:paradigm_code[T.ICR]']),
        ('Early', 'IRE', ['Days_within', 'Days_within:paradigm_code[T.IRE]']),
        ('Late', 'IFR', ['Days_within', 'Days_within:Segment[T.Late]']),
        ('Late', 'ICR', ['Days_within', 'Days_within:Segment[T.Late]',
                        'Days_within:paradigm_code[T.ICR]',
                        'Days_within:Segment[T.Late]:paradigm_code[T.ICR]']),
        ('Late', 'IRE', ['Days_within', 'Days_within:Segment[T.Late]',
                        'Days_within:paradigm_code[T.IRE]',
                        'Days_within:Segment[T.Late]:paradigm_code[T.IRE]'])
    ]

    for segment, paradigm, coef_list in slope_definitions:
        # Calculate slope as sum of coefficients
        slope = sum(fe_params.get(c, 0) for c in coef_list)

        # Calculate SE using delta method (variance of linear combination)
        # Var(sum) = sum of variances + 2*sum of covariances
        var = 0.0
        for i, c1 in enumerate(coef_list):
            if c1 in fe_cov.index:
                var += fe_cov.loc[c1, c1]
                for c2 in coef_list[i+1:]:
                    if c2 in fe_cov.columns:
                        var += 2 * fe_cov.loc[c1, c2]

        se = np.sqrt(var) if var > 0 else np.nan

        # Calculate z-statistic and p-value
        z = slope / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

        # Calculate 95% CI
        ci_lower = slope - 1.96 * se if not np.isnan(se) else np.nan
        ci_upper = slope + 1.96 * se if not np.isnan(se) else np.nan

        # Interpretation
        if not np.isnan(p) and p < 0.05:
            if slope < 0:
                interp = "Significant decline (forgetting)"
            else:
                interp = "Significant improvement"
        else:
            interp = "No significant change"

        slopes_data.append({
            'Segment': segment,
            'paradigm': paradigm,
            'slope': slope,
            'SE': se,
            'z_statistic': z,
            'p_value': p,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'interpretation': interp
        })

    return pd.DataFrame(slopes_data)


def main():
    """Extract segment-paradigm slopes from fitted LMM."""
    logger.info("=" * 60)
    logger.info("Step 03: Extract Segment-Paradigm Slopes")
    logger.info("=" * 60)

    # Define paths
    model_file = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
    output_file = RQ_DIR / "data" / "step03_segment_paradigm_slopes.csv"

    # --- Load fitted model ---
    logger.info(f"Loading model from: {model_file}")
    with open(model_file, 'rb') as f:
        result = pickle.load(f)
    logger.info("Model loaded successfully")

    # --- Extract slopes ---
    logger.info("\nExtracting segment-paradigm slopes...")
    slopes_df = extract_slopes_from_piecewise_lmm(result)

    # --- Display results ---
    logger.info("\n" + "=" * 60)
    logger.info("SEGMENT-PARADIGM SLOPES")
    logger.info("=" * 60)
    logger.info(f"\n{slopes_df.to_string(index=False)}")

    # --- Validation ---
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check row count
    if len(slopes_df) != 6:
        logger.error(f"CRITICAL: Expected 6 slopes, got {len(slopes_df)}")
        sys.exit(1)
    logger.info("VALIDATION - PASS: 6 slopes extracted")

    # Check for NaN
    nan_count = slopes_df['slope'].isna().sum()
    if nan_count > 0:
        logger.error(f"CRITICAL: {nan_count} NaN slopes")
        sys.exit(1)
    logger.info("VALIDATION - PASS: No NaN in slopes")

    # Check SE > 0
    se_zero = (slopes_df['SE'] <= 0).sum()
    if se_zero > 0:
        logger.error(f"CRITICAL: {se_zero} slopes have SE <= 0")
        sys.exit(1)
    logger.info("VALIDATION - PASS: All SEs > 0")

    # Check CI ordering
    ci_issue = (slopes_df['CI_lower'] > slopes_df['CI_upper']).sum()
    if ci_issue > 0:
        logger.error(f"CRITICAL: {ci_issue} slopes have CI_lower > CI_upper")
        sys.exit(1)
    logger.info("VALIDATION - PASS: CI ordering correct")

    # --- Save output ---
    slopes_df.to_csv(output_file, index=False)
    logger.info(f"\nOutput saved: {output_file}")

    # --- Summary interpretation ---
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    for segment in ['Early', 'Late']:
        logger.info(f"\n{segment} segment slopes (per day):")
        seg_data = slopes_df[slopes_df['Segment'] == segment]
        for _, row in seg_data.iterrows():
            p_str = f"p={row['p_value']:.4f}" if row['p_value'] >= 0.001 else f"p<0.001"
            logger.info(f"  {row['paradigm']}: {row['slope']:.4f} ({p_str}) - {row['interpretation']}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 03 COMPLETE")
    logger.info("=" * 60)
    logger.info("Slope extraction complete: 6 segment-paradigm slopes computed")
    logger.info("Delta method SEs propagated successfully")

    return slopes_df


if __name__ == "__main__":
    main()
