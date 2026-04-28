#!/usr/bin/env python3
"""Prepare Piecewise Trajectory Plot Data: Prepare plot data for piecewise trajectory visualization (theta + probability scales)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy import stats

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import plotting tool for probability conversion
from tools.plotting import convert_theta_to_probability

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.2
LOG_FILE = RQ_DIR / "logs" / "step05_prepare_piecewise_plot_data.log"

# RQ 5.2.1 dependency path (domain-specific item parameters)
RQ521_DIR = Path(__file__).resolve().parents[2] / "5.2.1"  # results/ch5/5.2.1


# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Prepare Piecewise Trajectory Plot Data")
        log("=" * 60)
        # Load Input Data
        #           Item parameters from RQ 5.1 for theta-to-probability conversion

        log("Loading input data...")

        # Load piecewise LMM input
        input_path = RQ_DIR / "data" / "step00_piecewise_lmm_input.csv"
        df_piecewise = pd.read_csv(input_path)
        log(f"step00_piecewise_lmm_input.csv ({len(df_piecewise)} rows, {len(df_piecewise.columns)} cols)")
        log(f"         Columns: {list(df_piecewise.columns)}")

        # Load fitted LMM model for predictions
        model_path = RQ_DIR / "data" / "step01_piecewise_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))
        log(f"step01_piecewise_lmm_model.pkl (MixedLMResults object)")

        # Load RQ 5.2.1 item parameters for theta-to-probability conversion
        items_path = RQ521_DIR / "data" / "step03_item_parameters.csv"
        df_items = pd.read_csv(items_path)
        log(f"RQ 5.2.1 step03_item_parameters.csv ({len(df_items)} items)")

        # Calculate mean discrimination for probability transformation
        mean_discrimination = df_items['a'].mean()
        log(f"         Mean item discrimination (a): {mean_discrimination:.4f}")
        # Aggregate Theta Data by Domain, Segment, Test
        # Uses scipy.stats for confidence interval calculation

        log("")
        log("Aggregating theta scores by domain, test, and Segment...")

        # Group by domain, test, and Segment
        grouped = df_piecewise.groupby(['domain', 'test', 'Segment'])

        # Compute statistics for each group
        agg_data = []
        for (domain, test, segment), group_df in grouped:
            theta_values = group_df['theta'].values
            n_obs = len(theta_values)
            mean_theta = np.mean(theta_values)

            # Compute 95% CI using t-distribution
            sem = stats.sem(theta_values)
            ci_margin = sem * stats.t.ppf(0.975, n_obs - 1)
            ci_lower = mean_theta - ci_margin
            ci_upper = mean_theta + ci_margin

            # Get representative TSVR_hours (median) for this test
            time = group_df['TSVR_hours'].median()

            agg_data.append({
                'time': time,
                'test': test,
                'domain': domain,
                'Segment': segment,
                'mean_theta': mean_theta,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'n_obs': n_obs
            })

        df_theta_plot = pd.DataFrame(agg_data)
        log(f"Aggregated to {len(df_theta_plot)} groups (domain x test combinations)")
        # Generate Model Predictions for Smooth Trajectories

        log("")
        log("Generating model predictions for trajectory lines...")

        # Get predictions from the fitted model
        # Using the group means as representative values
        predicted_theta = []

        for idx, row in df_theta_plot.iterrows():
            # Create prediction data matching the model formula
            # Formula: theta ~ Days_within * C(Segment, Treatment('Early')) * C(domain, Treatment('what'))
            pred_data = pd.DataFrame({
                'Days_within': [row['time'] / 24 if row['Segment'] == 'Early' else (row['time'] - df_piecewise[df_piecewise['Segment'] == 'Late']['TSVR_hours'].min()) / 24],
                'Segment': [row['Segment']],
                'domain': [row['domain']]
            })

            try:
                # Get prediction from model (using fixed effects only for trajectory)
                pred_val = lmm_model.predict(pred_data)
                predicted_theta.append(pred_val.values[0] if hasattr(pred_val, 'values') else pred_val[0])
            except Exception as e:
                # Fallback to mean theta if prediction fails
                log(f"         Warning: Prediction failed for {row['domain']}/{row['Segment']}/test{row['test']}: {e}")
                predicted_theta.append(row['mean_theta'])

        df_theta_plot['predicted_theta'] = predicted_theta
        log(f"Generated {len(predicted_theta)} predictions")
        # Save Theta-Scale Plot Data
        # Output: Theta-scale data for psychometrician-interpretable plots

        log("")
        log("Saving theta-scale plot data...")

        # Reorder columns for output
        theta_output_cols = ['time', 'test', 'domain', 'Segment', 'mean_theta', 'CI_lower', 'CI_upper', 'predicted_theta', 'n_obs']
        df_theta_plot = df_theta_plot[theta_output_cols]

        theta_output_path = RQ_DIR / "plots" / "step05_piecewise_theta_data.csv"
        df_theta_plot.to_csv(theta_output_path, index=False, encoding='utf-8')
        log(f"plots/step05_piecewise_theta_data.csv ({len(df_theta_plot)} rows)")

        # Log theta statistics
        log(f"         Theta range: [{df_theta_plot['mean_theta'].min():.3f}, {df_theta_plot['mean_theta'].max():.3f}]")
        # Transform Theta to Probability Scale (Decision D069)
        # Formula: P = 1 / (1 + exp(-(a * (theta - b))))
        # Using: a = mean discrimination from RQ 5.1, b = 0 (reference difficulty)

        log("")
        log("Transforming theta to probability scale (Decision D069)...")
        log(f"         Using IRT 2PL: P = 1 / (1 + exp(-(a * (theta - b))))")
        log(f"         a = {mean_discrimination:.4f} (mean discrimination)")
        log(f"         b = 0.0 (reference difficulty)")

        # Transform all theta values to probability
        df_prob_plot = df_theta_plot.copy()

        # Transform mean_theta
        df_prob_plot['mean_probability'] = convert_theta_to_probability(
            theta=df_theta_plot['mean_theta'].values,
            discrimination=mean_discrimination,
            difficulty=0.0
        )

        # Transform CI bounds
        df_prob_plot['CI_lower'] = convert_theta_to_probability(
            theta=df_theta_plot['CI_lower'].values,
            discrimination=mean_discrimination,
            difficulty=0.0
        )

        df_prob_plot['CI_upper'] = convert_theta_to_probability(
            theta=df_theta_plot['CI_upper'].values,
            discrimination=mean_discrimination,
            difficulty=0.0
        )

        # Transform predicted theta
        df_prob_plot['predicted_probability'] = convert_theta_to_probability(
            theta=df_theta_plot['predicted_theta'].values,
            discrimination=mean_discrimination,
            difficulty=0.0
        )

        # Drop theta-specific columns
        df_prob_plot = df_prob_plot.drop(columns=['mean_theta', 'predicted_theta'])

        # Rename for clarity
        df_prob_plot = df_prob_plot.rename(columns={
            'CI_lower': 'CI_lower',  # Keep same name, values are now probabilities
            'CI_upper': 'CI_upper'
        })

        log(f"Transformed {len(df_prob_plot)} rows to probability scale")
        log(f"         Probability range: [{df_prob_plot['mean_probability'].min():.3f}, {df_prob_plot['mean_probability'].max():.3f}]")
        # Save Probability-Scale Plot Data
        # Output: Probability-scale data for general audience interpretable plots

        log("")
        log("Saving probability-scale plot data...")

        # Reorder columns for output
        prob_output_cols = ['time', 'test', 'domain', 'Segment', 'mean_probability', 'CI_lower', 'CI_upper', 'predicted_probability', 'n_obs']
        df_prob_plot = df_prob_plot[prob_output_cols]

        prob_output_path = RQ_DIR / "plots" / "step05_piecewise_probability_data.csv"
        df_prob_plot.to_csv(prob_output_path, index=False, encoding='utf-8')
        log(f"plots/step05_piecewise_probability_data.csv ({len(df_prob_plot)} rows)")
        # Run Validation
        # Validate all criteria from 4_analysis.yaml

        log("")
        log("Running validation checks...")

        validation_errors = []

        # Check 1: Theta plot data row count (8 due to When exclusion)
        if len(df_theta_plot) != 8:
            validation_errors.append(f"Theta plot data has {len(df_theta_plot)} rows, expected 8 (2 domains x 4 tests - When excluded)")
        else:
            log("Theta plot data has exactly 8 rows (When excluded)")

        # Check 2: Probability plot data row count (8 due to When exclusion)
        if len(df_prob_plot) != 8:
            validation_errors.append(f"Probability plot data has {len(df_prob_plot)} rows, expected 8 (2 domains x 4 tests - When excluded)")
        else:
            log("Probability plot data has exactly 8 rows (When excluded)")

        # Check 3: All domains present (When excluded)
        expected_domains = {'what', 'where'}  # When excluded
        actual_domains = set(df_theta_plot['domain'].unique())
        if actual_domains != expected_domains:
            validation_errors.append(f"Domain mismatch: expected {expected_domains}, got {actual_domains}")
        else:
            log("All domains present: {what, where} (When excluded)")

        # Check 4: All tests present
        expected_tests = {1, 2, 3, 4}  # T1-T4 session numbers (not nominal days)
        actual_tests = set(df_theta_plot['test'].unique())
        if actual_tests != expected_tests:
            validation_errors.append(f"Test mismatch: expected {expected_tests}, got {actual_tests}")
        else:
            log("All tests present: {1, 2, 3, 4}")

        # Check 5: Segment assignment correct (T1,T2 = Early; T3,T4 = Late)
        early_tests = df_theta_plot[df_theta_plot['Segment'] == 'Early']['test'].unique()
        late_tests = df_theta_plot[df_theta_plot['Segment'] == 'Late']['test'].unique()
        if not (set(early_tests) == {1, 2} and set(late_tests) == {3, 4}):
            validation_errors.append(f"Segment assignment wrong: Early={early_tests}, Late={late_tests}")
        else:
            log("Segment assignment correct: T1,T2 -> 'Early'; T3,T4 -> 'Late'")

        # Check 6: Theta range valid
        theta_min = df_theta_plot['mean_theta'].min()
        theta_max = df_theta_plot['mean_theta'].max()
        if theta_min < -3 or theta_max > 3:
            validation_errors.append(f"Theta range [{theta_min:.3f}, {theta_max:.3f}] outside expected [-3, 3]")
        else:
            log(f"Theta range valid: [{theta_min:.3f}, {theta_max:.3f}]")

        # Check 7: Probability range valid
        prob_min = df_prob_plot['mean_probability'].min()
        prob_max = df_prob_plot['mean_probability'].max()
        if prob_min < 0 or prob_max > 1:
            validation_errors.append(f"Probability range [{prob_min:.3f}, {prob_max:.3f}] outside expected [0, 1]")
        else:
            log(f"Probability range valid: [{prob_min:.3f}, {prob_max:.3f}]")

        # Check 8: CI structure valid (theta)
        theta_ci_valid = all(df_theta_plot['CI_lower'] < df_theta_plot['mean_theta'])
        theta_ci_valid = theta_ci_valid and all(df_theta_plot['mean_theta'] < df_theta_plot['CI_upper'])
        if not theta_ci_valid:
            validation_errors.append("Theta CI structure invalid: CI_lower < mean < CI_upper not satisfied")
        else:
            log("Theta CI structure valid: CI_lower < mean < CI_upper")

        # Check 9: CI structure valid (probability)
        prob_ci_valid = all(df_prob_plot['CI_lower'] < df_prob_plot['mean_probability'])
        prob_ci_valid = prob_ci_valid and all(df_prob_plot['mean_probability'] < df_prob_plot['CI_upper'])
        if not prob_ci_valid:
            validation_errors.append("Probability CI structure invalid: CI_lower < mean < CI_upper not satisfied")
        else:
            log("Probability CI structure valid: CI_lower < mean < CI_upper")

        # Check 10: No NaN values
        theta_nans = df_theta_plot.isna().sum().sum()
        prob_nans = df_prob_plot.isna().sum().sum()
        if theta_nans > 0 or prob_nans > 0:
            validation_errors.append(f"NaN values found: theta_data={theta_nans}, prob_data={prob_nans}")
        else:
            log("No NaN values in output data")

        # Report validation results
        log("")
        if validation_errors:
            log("Validation failed with errors:")
            for err in validation_errors:
                log(f"       - {err}")
            raise ValueError(f"Validation failed: {validation_errors[0]}")
        else:
            log("All validation checks passed!")
        # Summary

        log("")
        log("=" * 60)
        log("Step 05 Complete")
        log("=" * 60)
        log(f"Outputs:")
        log(f"  - plots/step05_piecewise_theta_data.csv ({len(df_theta_plot)} rows)")
        log(f"  - plots/step05_piecewise_probability_data.csv ({len(df_prob_plot)} rows)")
        log("")
        log("Decision D069 Implementation:")
        log("  - Theta scale: For psychometricians and statistical rigor")
        log("  - Probability scale: For general audience interpretability")
        log(f"  - Transformation: IRT 2PL with a={mean_discrimination:.4f}, b=0")
        log("")
        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
