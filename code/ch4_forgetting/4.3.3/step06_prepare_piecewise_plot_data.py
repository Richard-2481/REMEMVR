#!/usr/bin/env python3
"""
Step 06: Prepare Piecewise Plot Data
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Aggregate observed means and model predictions per paradigm x segment
x timepoint for two-panel piecewise trajectory visualization with dual-scale
plots (theta + probability scales per Decision D069).
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

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step06_prepare_piecewise_plot_data.log"
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


def theta_to_probability(theta, a=1.0, b=0.0):
    """
    Convert IRT theta to probability using 2PL model.

    P(theta) = 1 / (1 + exp(-a * (theta - b)))

    With a=1, b=0 (standard normal), this is the standard logistic.
    """
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def main():
    """Prepare plot data for piecewise trajectory visualization."""
    logger.info("=" * 60)
    logger.info("Step 06: Prepare Piecewise Plot Data")
    logger.info("=" * 60)

    # Define paths
    input_file = RQ_DIR / "data" / "step01_piecewise_lmm_input.csv"
    model_file = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
    slopes_file = RQ_DIR / "data" / "step03_segment_paradigm_slopes.csv"
    theta_output = RQ_DIR / "data" / "step06_piecewise_theta_data.csv"
    prob_output = RQ_DIR / "data" / "step06_piecewise_probability_data.csv"

    # --- Load data ---
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    logger.info(f"Loading model from: {model_file}")
    with open(model_file, 'rb') as f:
        result = pickle.load(f)

    logger.info(f"Loading slopes from: {slopes_file}")
    slopes_df = pd.read_csv(slopes_file)

    # Build slopes lookup
    slopes_lookup = {}
    for _, row in slopes_df.iterrows():
        key = (row['Segment'], row['paradigm'])
        slopes_lookup[key] = row['slope']

    # --- Aggregate observed data ---
    logger.info("\nAggregating observed data...")

    # Group by Segment, paradigm_code, test to get observed means
    observed = df.groupby(['Segment', 'paradigm_code', 'test', 'test_code']).agg({
        'theta': ['mean', 'std', 'count'],
        'Days_within': 'mean'
    }).reset_index()

    # Flatten column names
    observed.columns = ['Segment', 'paradigm', 'test', 'test_code',
                       'theta_mean', 'theta_std', 'n', 'Days_within']

    # Compute 95% CI
    observed['theta_se'] = observed['theta_std'] / np.sqrt(observed['n'])
    observed['CI_lower'] = observed['theta_mean'] - 1.96 * observed['theta_se']
    observed['CI_upper'] = observed['theta_mean'] + 1.96 * observed['theta_se']

    logger.info(f"Observed data: {len(observed)} segment-paradigm-test combinations")

    # --- Generate model predictions ---
    logger.info("\nGenerating model predictions...")

    # Get fixed effects for prediction
    fe_params = result.fe_params

    # Create prediction grid
    predictions = []

    for segment in ['Early', 'Late']:
        # Define Days_within range for this segment
        seg_data = df[df['Segment'] == segment]['Days_within']
        days_min = seg_data.min()
        days_max = seg_data.max()

        # Generate grid points
        n_points = 20 if segment == 'Early' else 60
        days_grid = np.linspace(days_min, days_max, n_points)

        for paradigm in ['IFR', 'ICR', 'IRE']:
            for days in days_grid:
                # Compute predicted theta using fixed effects
                # This is a manual prediction based on the model formula

                # Base: Intercept + Days_within coefficient * days
                pred = fe_params['Intercept']
                pred += fe_params['Days_within'] * days

                # Add Segment effects
                if segment == 'Late':
                    pred += fe_params['Segment[T.Late]']
                    pred += fe_params['Days_within:Segment[T.Late]'] * days

                # Add paradigm effects
                if paradigm == 'ICR':
                    pred += fe_params['paradigm_code[T.ICR]']
                    pred += fe_params['Days_within:paradigm_code[T.ICR]'] * days
                    if segment == 'Late':
                        pred += fe_params['Segment[T.Late]:paradigm_code[T.ICR]']
                        pred += fe_params['Days_within:Segment[T.Late]:paradigm_code[T.ICR]'] * days
                elif paradigm == 'IRE':
                    pred += fe_params['paradigm_code[T.IRE]']
                    pred += fe_params['Days_within:paradigm_code[T.IRE]'] * days
                    if segment == 'Late':
                        pred += fe_params['Segment[T.Late]:paradigm_code[T.IRE]']
                        pred += fe_params['Days_within:Segment[T.Late]:paradigm_code[T.IRE]'] * days

                predictions.append({
                    'Segment': segment,
                    'paradigm': paradigm,
                    'Days_within': days,
                    'theta_predicted': pred,
                    'data_type': 'prediction'
                })

    pred_df = pd.DataFrame(predictions)
    logger.info(f"Prediction grid: {len(pred_df)} points")

    # --- Merge observed and predicted data ---
    logger.info("\nMerging observed and predicted data...")

    # Add slopes to predictions
    pred_df['slope'] = pred_df.apply(
        lambda row: slopes_lookup.get((row['Segment'], row['paradigm']), np.nan),
        axis=1
    )

    # Add observed data points
    observed_rows = []
    for _, row in observed.iterrows():
        observed_rows.append({
            'Segment': row['Segment'],
            'paradigm': row['paradigm'],
            'Days_within': row['Days_within'],
            'test': row['test_code'],
            'theta_observed': row['theta_mean'],
            'theta_predicted': np.nan,  # Will fill from predictions
            'CI_lower': row['CI_lower'],
            'CI_upper': row['CI_upper'],
            'slope': slopes_lookup.get((row['Segment'], row['paradigm']), np.nan),
            'data_type': 'observed'
        })

    obs_df = pd.DataFrame(observed_rows)

    # Fill in predicted values at observed timepoints
    for idx, row in obs_df.iterrows():
        # Find closest prediction
        seg_preds = pred_df[
            (pred_df['Segment'] == row['Segment']) &
            (pred_df['paradigm'] == row['paradigm'])
        ]
        if len(seg_preds) > 0:
            closest_idx = (seg_preds['Days_within'] - row['Days_within']).abs().idxmin()
            obs_df.loc[idx, 'theta_predicted'] = pred_df.loc[closest_idx, 'theta_predicted']

    # Combine
    theta_data = pd.concat([
        pred_df[['Segment', 'paradigm', 'Days_within', 'theta_predicted', 'slope']].assign(
            test=np.nan, theta_observed=np.nan, CI_lower=np.nan, CI_upper=np.nan, data_type='prediction'
        ),
        obs_df
    ], ignore_index=True)

    # Reorder columns
    theta_cols = ['Segment', 'paradigm', 'Days_within', 'test', 'theta_observed',
                  'theta_predicted', 'CI_lower', 'CI_upper', 'slope', 'data_type']
    theta_data = theta_data[theta_cols]

    # --- Convert to probability scale (Decision D069) ---
    logger.info("\nConverting to probability scale (Decision D069)...")

    prob_data = theta_data.copy()
    prob_data['prob_observed'] = theta_to_probability(prob_data['theta_observed'])
    prob_data['prob_predicted'] = theta_to_probability(prob_data['theta_predicted'])
    prob_data['CI_lower'] = theta_to_probability(prob_data['CI_lower'])
    prob_data['CI_upper'] = theta_to_probability(prob_data['CI_upper'])

    # Keep theta columns for reference
    prob_data = prob_data.drop(columns=['theta_observed', 'theta_predicted'])

    prob_cols = ['Segment', 'paradigm', 'Days_within', 'test', 'prob_observed',
                 'prob_predicted', 'CI_lower', 'CI_upper', 'slope', 'data_type']
    prob_data = prob_data[prob_cols]

    # --- Validation ---
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check both files exist (will exist after saving)
    logger.info("Plot data preparation complete: theta scale + probability scale")

    # Check observed data count
    n_observed = len(theta_data[theta_data['data_type'] == 'observed'])
    logger.info(f"Observed data: {n_observed} segment-paradigm-test combinations")
    # But some tests span both segments, so may be different
    if n_observed < 6:
        logger.warning(f"WARNING: Expected at least 6 observed points, got {n_observed}")

    # Check prediction count
    n_predicted = len(theta_data[theta_data['data_type'] == 'prediction'])
    logger.info(f"Predicted data: {n_predicted} timepoints across 6 segment-paradigm combinations")

    # Check all segment-paradigm combinations present
    combos = theta_data.groupby(['Segment', 'paradigm']).size()
    if len(combos) < 6:
        logger.error(f"CRITICAL: Expected 6 segment-paradigm combinations, got {len(combos)}")
        sys.exit(1)
    logger.info("VALIDATION - PASS: All 6 segment-paradigm combinations present")

    # Check probability range
    prob_min = prob_data['prob_predicted'].min()
    prob_max = prob_data['prob_predicted'].max()
    if prob_min < 0 or prob_max > 1:
        logger.error(f"CRITICAL: Probability values outside [0, 1]: [{prob_min}, {prob_max}]")
        sys.exit(1)
    logger.info(f"VALIDATION - PASS: Probability values in [0, 1] range [{prob_min:.3f}, {prob_max:.3f}]")

    # Check NaN in predictions
    nan_preds = theta_data['theta_predicted'].isna().sum()
    if nan_preds > n_observed:  # Some NaN expected for observed rows without matching prediction
        logger.warning(f"WARNING: {nan_preds} NaN in theta_predicted (expected {n_observed} for observed-only rows)")

    logger.info("VALIDATION - PASS: Dual-scale plot data created (Decision D069)")

    # --- Save outputs ---
    theta_data.to_csv(theta_output, index=False)
    logger.info(f"\nTheta-scale data saved: {theta_output}")

    prob_data.to_csv(prob_output, index=False)
    logger.info(f"Probability-scale data saved: {prob_output}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 06 COMPLETE")
    logger.info("=" * 60)

    return theta_data, prob_data


if __name__ == "__main__":
    main()
