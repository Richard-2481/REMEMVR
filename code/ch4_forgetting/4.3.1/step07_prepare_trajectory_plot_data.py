#!/usr/bin/env python3
"""Prepare Trajectory Plot Data: Create plot source CSVs for dual-scale trajectory visualization (Decision D069)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_trajectory_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def theta_to_probability(theta):
    """Convert IRT theta to probability using 2PL formula (difficulty=0, discrimination=1).

    P = 1 / (1 + exp(-theta))

    This is the standard logistic transformation.
    """
    return 1.0 / (1.0 + np.exp(-theta))


def parse_fixed_effects(df_fe):
    """Parse fixed effects DataFrame into a structured dictionary.

    The fixed effects have structure:
    - Intercept: baseline for reference group (Free_Recall)
    - Factor effects: adjustments for Cued_Recall, Recognition
    - log_Days: time slope for reference group
    - Interaction terms: differential slopes for other factors

    Returns:
        dict with keys:
            - intercept: float
            - factor_cued: float (adjustment for Cued_Recall)
            - factor_recog: float (adjustment for Recognition)
            - log_days_slope: float (time slope for Free_Recall)
            - log_days_cued_interaction: float (differential slope for Cued_Recall)
            - log_days_recog_interaction: float (differential slope for Recognition)
    """
    fe_dict = {}

    for _, row in df_fe.iterrows():
        effect = row['effect']
        coef = row['coefficient']

        if effect == 'Intercept':
            fe_dict['intercept'] = coef
        elif 'Cued_Recall' in effect and 'log_Days' not in effect:
            fe_dict['factor_cued'] = coef
        elif 'Recognition' in effect and 'log_Days' not in effect:
            fe_dict['factor_recog'] = coef
        elif effect == 'log_Days':
            fe_dict['log_days_slope'] = coef
        elif 'log_Days' in effect and 'Cued_Recall' in effect:
            fe_dict['log_days_cued_interaction'] = coef
        elif 'log_Days' in effect and 'Recognition' in effect:
            fe_dict['log_days_recog_interaction'] = coef

    return fe_dict


def predict_theta(tsvr_hours_log, paradigm, fe_dict):
    """Predict theta given log(TSVR_hours + 1) and paradigm.

    Model: theta = intercept + factor_effect + (log_days_slope + interaction) * log_time

    Args:
        tsvr_hours_log: log(TSVR_hours + 1) value(s)
        paradigm: 'free_recall', 'cued_recall', or 'recognition'
        fe_dict: parsed fixed effects dictionary

    Returns:
        Predicted theta value(s)
    """
    # Base prediction for reference group (free_recall)
    theta = fe_dict['intercept'] + fe_dict['log_days_slope'] * tsvr_hours_log

    # Add factor adjustments for non-reference groups
    if paradigm == 'cued_recall':
        theta += fe_dict.get('factor_cued', 0)
        theta += fe_dict.get('log_days_cued_interaction', 0) * tsvr_hours_log
    elif paradigm == 'recognition':
        theta += fe_dict.get('factor_recog', 0)
        theta += fe_dict.get('log_days_recog_interaction', 0) * tsvr_hours_log

    return theta


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Prepare Trajectory Plot Data")
        # Load Input Data
        log("Loading LMM input data...")
        df_lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        log(f"step04_lmm_input.csv ({len(df_lmm_input)} rows, {len(df_lmm_input.columns)} cols)")

        log("Loading fixed effects...")
        df_fixed_effects = pd.read_csv(RQ_DIR / "data" / "step05_fixed_effects.csv")
        log(f"step05_fixed_effects.csv ({len(df_fixed_effects)} rows)")

        # Log fixed effects for debugging
        log("Fixed effects from best model:")
        for _, row in df_fixed_effects.iterrows():
            if pd.notna(row['coefficient']):
                log(f"       {row['effect']}: {row['coefficient']:.4f}")
        # Parse Fixed Effects
        log("Parsing fixed effects structure...")
        fe_dict = parse_fixed_effects(df_fixed_effects)
        log(f"Intercept: {fe_dict.get('intercept', 'N/A'):.4f}")
        log(f"log_Days slope (Free_Recall): {fe_dict.get('log_days_slope', 'N/A'):.4f}")
        log(f"Cued_Recall adjustment: {fe_dict.get('factor_cued', 0):.4f}")
        log(f"Recognition adjustment: {fe_dict.get('factor_recog', 0):.4f}")
        # Compute Observed Means per Paradigm x Test
        log("Computing observed means per paradigm x test...")

        # Group by paradigm and test, compute statistics
        observed_stats = df_lmm_input.groupby(['paradigm', 'test']).agg(
            theta_observed=('theta', 'mean'),
            theta_std=('theta', 'std'),
            n=('theta', 'count'),
            TSVR_hours_mean=('TSVR_hours', 'mean'),
            TSVR_hours_log_mean=('TSVR_hours_log', 'mean')
        ).reset_index()

        # Compute 95% CI: mean +/- 1.96 * (std / sqrt(n))
        observed_stats['CI_lower'] = observed_stats['theta_observed'] - 1.96 * (observed_stats['theta_std'] / np.sqrt(observed_stats['n']))
        observed_stats['CI_upper'] = observed_stats['theta_observed'] + 1.96 * (observed_stats['theta_std'] / np.sqrt(observed_stats['n']))

        log(f"Observed statistics computed for {len(observed_stats)} paradigm x test combinations")

        # Log summary per paradigm
        for paradigm in ['free_recall', 'cued_recall', 'recognition']:
            subset = observed_stats[observed_stats['paradigm'] == paradigm]
            log(f"{paradigm}: {len(subset)} test sessions, mean theta range [{subset['theta_observed'].min():.3f}, {subset['theta_observed'].max():.3f}]")
        # Generate Prediction Grid
        log("Generating prediction grid...")

        # Create TSVR_hours grid from 0 to 200 hours (50 points)
        tsvr_grid = np.linspace(0, 200, 50)

        # Convert to log scale (log(TSVR + 1) to handle 0)
        tsvr_log_grid = np.log(tsvr_grid + 1)

        log(f"Prediction grid: {len(tsvr_grid)} points from 0 to 200 hours")
        # Compute Model Predictions
        log("Computing model predictions for each paradigm...")

        prediction_rows = []
        paradigms = ['free_recall', 'cued_recall', 'recognition']

        for paradigm in paradigms:
            for i, (hours, log_hours) in enumerate(zip(tsvr_grid, tsvr_log_grid)):
                theta_pred = predict_theta(log_hours, paradigm, fe_dict)
                prediction_rows.append({
                    'TSVR_hours': hours,
                    'TSVR_hours_log': log_hours,
                    'theta_predicted': theta_pred,
                    'paradigm': paradigm,
                    'data_type': 'predicted'
                })

        df_predictions = pd.DataFrame(prediction_rows)
        log(f"Generated {len(df_predictions)} prediction points")
        # Create Theta-Scale Plot Data
        log("Creating theta-scale plot data...")

        # Observed data (from grouped stats)
        df_observed_theta = observed_stats[['TSVR_hours_mean', 'theta_observed', 'CI_lower', 'CI_upper', 'paradigm']].copy()
        df_observed_theta = df_observed_theta.rename(columns={'TSVR_hours_mean': 'TSVR_hours'})
        df_observed_theta['theta_predicted'] = np.nan  # No prediction for observed points
        df_observed_theta['data_type'] = 'observed'

        # Predicted data (with CI from model - using placeholder SE-based CI)
        # Note: Proper prediction CI would require the full model object
        # Using a simple approximation based on typical SE
        df_pred_theta = df_predictions[['TSVR_hours', 'theta_predicted', 'paradigm', 'data_type']].copy()
        df_pred_theta['theta_observed'] = np.nan

        # Approximate CI for predictions (using mean SE from fixed effects)
        mean_se = df_fixed_effects['std_error'].dropna().mean()
        df_pred_theta['CI_lower'] = df_pred_theta['theta_predicted'] - 1.96 * mean_se
        df_pred_theta['CI_upper'] = df_pred_theta['theta_predicted'] + 1.96 * mean_se

        # Combine observed and predicted
        df_theta_plot = pd.concat([df_observed_theta, df_pred_theta], ignore_index=True)

        # Reorder columns for clarity
        df_theta_plot = df_theta_plot[['TSVR_hours', 'theta_observed', 'theta_predicted', 'CI_lower', 'CI_upper', 'paradigm', 'data_type']]

        log(f"Theta plot data: {len(df_theta_plot)} rows (observed + predicted)")
        # Create Probability-Scale Plot Data
        log("Creating probability-scale plot data (IRT 2PL transformation)...")

        # Transform theta to probability
        df_prob_plot = df_theta_plot.copy()
        df_prob_plot['probability_observed'] = theta_to_probability(df_prob_plot['theta_observed'])
        df_prob_plot['probability_predicted'] = theta_to_probability(df_prob_plot['theta_predicted'])
        df_prob_plot['CI_lower_prob'] = theta_to_probability(df_prob_plot['CI_lower'])
        df_prob_plot['CI_upper_prob'] = theta_to_probability(df_prob_plot['CI_upper'])

        # Rename and select columns
        df_prob_plot = df_prob_plot.rename(columns={
            'CI_lower_prob': 'CI_lower',
            'CI_upper_prob': 'CI_upper'
        })
        df_prob_plot = df_prob_plot[['TSVR_hours', 'probability_observed', 'probability_predicted', 'CI_lower', 'CI_upper', 'paradigm', 'data_type']]

        log(f"Probability plot data: {len(df_prob_plot)} rows")
        # Save Outputs
        log("Saving theta-scale plot data...")
        theta_output_path = RQ_DIR / "plots" / "step07_trajectory_theta_data.csv"
        df_theta_plot.to_csv(theta_output_path, index=False, encoding='utf-8')
        log(f"{theta_output_path} ({len(df_theta_plot)} rows)")

        log("Saving probability-scale plot data...")
        prob_output_path = RQ_DIR / "plots" / "step07_trajectory_probability_data.csv"
        df_prob_plot.to_csv(prob_output_path, index=False, encoding='utf-8')
        log(f"{prob_output_path} ({len(df_prob_plot)} rows)")
        # Validation
        log("Running validation checks...")

        # Check 1: Theta plot file exists
        assert theta_output_path.exists(), "Theta plot file not created"
        log("Theta plot file exists")

        # Check 2: Probability plot file exists
        assert prob_output_path.exists(), "Probability plot file not created"
        log("Probability plot file exists")

        # Check 3: All paradigms represented
        paradigms_in_data = set(df_theta_plot['paradigm'].unique())
        expected_paradigms = {'free_recall', 'cued_recall', 'recognition'}
        assert paradigms_in_data == expected_paradigms, f"Missing paradigms: {expected_paradigms - paradigms_in_data}"
        log("All paradigms represented")

        # Check 4: Theta predictions in valid range [-3, 3]
        theta_pred = df_theta_plot['theta_predicted'].dropna()
        if len(theta_pred) > 0:
            theta_range_valid = theta_pred.between(-3, 3).all()
            if not theta_range_valid:
                log(f"Some theta predictions outside [-3, 3]: min={theta_pred.min():.3f}, max={theta_pred.max():.3f}")
            else:
                log("Theta predictions in valid range [-3, 3]")

        # Check 5: Probability predictions in [0, 1]
        prob_pred = df_prob_plot['probability_predicted'].dropna()
        assert prob_pred.between(0, 1).all(), "Probability predictions outside [0, 1]"
        log("Probability predictions in [0, 1]")

        # Check 6: CI bounds valid
        valid_ci = (df_theta_plot['CI_lower'] <= df_theta_plot['CI_upper']).all()
        assert valid_ci, "CI bounds invalid (lower > upper)"
        log("CI bounds valid")

        # Check 7: Minimum rows
        assert len(df_theta_plot) >= 12, f"Too few rows: {len(df_theta_plot)} < 12"
        log(f"Minimum rows met ({len(df_theta_plot)} >= 12)")

        log("Step 07 complete")

        # Summary
        log("")
        log("=" * 60)
        log("SUMMARY")
        log("=" * 60)
        log(f"Theta-scale data: {theta_output_path}")
        log(f"  - Rows: {len(df_theta_plot)}")
        log(f"  - Observed points: {len(df_theta_plot[df_theta_plot['data_type'] == 'observed'])}")
        log(f"  - Predicted points: {len(df_theta_plot[df_theta_plot['data_type'] == 'predicted'])}")
        log(f"")
        log(f"Probability-scale data: {prob_output_path}")
        log(f"  - Rows: {len(df_prob_plot)}")
        log(f"  - Probability range: [{prob_pred.min():.3f}, {prob_pred.max():.3f}]")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
