#!/usr/bin/env python3
"""Prepare Piecewise Trajectory Plot Data: Prepare two-panel plot data (Early segment 0-1 days | Late segment 0-6 days)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy import stats
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq6
LOG_FILE = RQ_DIR / "logs" / "step06_prepare_piecewise_plot_data.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions (INLINE IMPLEMENTATION - to extract to tools/ later)

def aggregate_observed_data(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Aggregate observed theta scores by Congruence and Days_within.

    Args:
        df: DataFrame with theta, Segment, Congruence, Days_within
        segment: 'Early' or 'Late'

    Returns:
        DataFrame with observed means and 95% CIs
    """
    df_seg = df[df['Segment'] == segment].copy()

    # Round Days_within to nearest 0.1 day for grouping
    df_seg['Days_within_rounded'] = (df_seg['Days_within'] * 10).round() / 10

    # Group by Congruence and Days_within_rounded
    grouped = df_seg.groupby(['Congruence', 'Days_within_rounded'])['theta'].agg([
        ('theta_observed', 'mean'),
        ('SE_mean', lambda x: x.std() / np.sqrt(len(x))),
        ('n_obs', 'count')
    ]).reset_index()

    # Compute 95% CI
    grouped['CI_lower_observed'] = grouped['theta_observed'] - 1.96 * grouped['SE_mean']
    grouped['CI_upper_observed'] = grouped['theta_observed'] + 1.96 * grouped['SE_mean']

    # Rename for output
    grouped = grouped.rename(columns={'Days_within_rounded': 'Days_within'})

    return grouped[['Days_within', 'Congruence', 'theta_observed',
                     'CI_lower_observed', 'CI_upper_observed', 'n_obs']]

def generate_predictions(lmm_model: MixedLMResults, segment: str,
                         days_grid: np.ndarray) -> pd.DataFrame:
    """
    Generate model predictions for plot grid.

    Args:
        lmm_model: Fitted LMM model
        segment: 'Early' or 'Late'
        days_grid: Days_within values for prediction

    Returns:
        DataFrame with predicted theta values
    """
    predictions = []

    for congruence in ['Common', 'Congruent', 'Incongruent']:
        for days in days_grid:
            # Create prediction row
            # Note: We need to construct a DataFrame with proper structure for prediction
            pred_row = pd.DataFrame({
                'Days_within': [days],
                'Segment': pd.Categorical([segment], categories=['Early', 'Late']),
                'Congruence': pd.Categorical([congruence],
                                              categories=['Common', 'Congruent', 'Incongruent'])
            })

            # Predict theta (use population-level prediction)
            try:
                theta_pred = lmm_model.predict(exog=pred_row)
                predictions.append({
                    'Days_within': days,
                    'Congruence': congruence,
                    'theta_predicted': float(theta_pred.iloc[0]) if hasattr(theta_pred, 'iloc') else float(theta_pred[0])
                })
            except:
                # If prediction fails, use NaN (will be filtered out later)
                predictions.append({
                    'Days_within': days,
                    'Congruence': congruence,
                    'theta_predicted': np.nan
                })

    return pd.DataFrame(predictions)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Prepare Piecewise Trajectory Plot Data")
        # Load Data

        log("Loading piecewise data and fitted model...")

        data_path = RQ_DIR / "data" / "step01_lmm_input_piecewise.csv"
        df_data = pd.read_csv(data_path, encoding='utf-8')

        # Restore categorical coding
        df_data['Congruence'] = pd.Categorical(
            df_data['Congruence'],
            categories=['Common', 'Congruent', 'Incongruent'],
            ordered=False
        )
        df_data['Segment'] = pd.Categorical(
            df_data['Segment'],
            categories=['Early', 'Late'],
            ordered=False
        )

        model_path = RQ_DIR / "data" / "step02_piecewise_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))

        log(f"Data ({len(df_data)} rows) and model")
        # Aggregate Observed Data

        log("Computing observed means and CIs by segment...")

        df_early_obs = aggregate_observed_data(df_data, 'Early')
        df_late_obs = aggregate_observed_data(df_data, 'Late')

        log(f"Early segment: {len(df_early_obs)} observed points")
        log(f"Late segment: {len(df_late_obs)} observed points")
        # Generate Model Predictions

        log("Generating model predictions on grid...")

        # Early segment grid: 20 points from 0 to 1 day
        early_grid = np.linspace(0, 1, 20)
        df_early_pred = generate_predictions(lmm_model, 'Early', early_grid)

        # Late segment grid: 60 points from 0 to 6 days (within Late segment)
        late_grid = np.linspace(0, 6, 60)
        df_late_pred = generate_predictions(lmm_model, 'Late', late_grid)

        log(f"Early predictions: {len(df_early_pred)} points")
        log(f"Late predictions: {len(df_late_pred)} points")
        # Merge Observed and Predicted

        log("Merging observed and predicted data...")

        # Merge Early
        df_early_plot = df_early_pred.merge(
            df_early_obs,
            on=['Days_within', 'Congruence'],
            how='left'  # Keep all predictions, fill observed where available
        )
        df_early_plot['Data_Type'] = 'Early'

        # Merge Late
        df_late_plot = df_late_pred.merge(
            df_late_obs,
            on=['Days_within', 'Congruence'],
            how='left'
        )
        df_late_plot['Data_Type'] = 'Late'

        log(f"Early plot data: {len(df_early_plot)} rows")
        log(f"Late plot data: {len(df_late_plot)} rows")
        # Validate and Save

        log("Validating plot data...")

        # Check Early row count (approximately 60: 3 congruence x 20 grid points)
        if not (50 <= len(df_early_plot) <= 70):
            log(f"Early plot row count unusual: {len(df_early_plot)} (expected ~60)")

        # Check Late row count (approximately 180: 3 congruence x 60 grid points)
        if not (170 <= len(df_late_plot) <= 190):
            log(f"Late plot row count unusual: {len(df_late_plot)} (expected ~180)")

        # Check all congruence types present
        for df_plot, name in [(df_early_plot, 'Early'), (df_late_plot, 'Late')]:
            congruence_types = set(df_plot['Congruence'].unique())
            expected_types = {'Common', 'Congruent', 'Incongruent'}
            if congruence_types != expected_types:
                raise ValueError(
                    f"{name} plot missing congruence types: "
                    f"expected {expected_types}, found {congruence_types}"
                )
        log(f"All congruence types present in both segments")

        # Check Days_within ranges
        if df_early_plot['Days_within'].min() < 0 or df_early_plot['Days_within'].max() > 1:
            log(f"Early Days_within out of range [0, 1]: "
                f"[{df_early_plot['Days_within'].min():.2f}, {df_early_plot['Days_within'].max():.2f}]")

        if df_late_plot['Days_within'].min() < 0 or df_late_plot['Days_within'].max() > 6:
            log(f"Late Days_within out of range [0, 6]: "
                f"[{df_late_plot['Days_within'].min():.2f}, {df_late_plot['Days_within'].max():.2f}]")

        log(f"Days_within ranges valid")

        # Check theta ranges
        theta_cols = ['theta_observed', 'theta_predicted']
        for df_plot, name in [(df_early_plot, 'Early'), (df_late_plot, 'Late')]:
            for col in theta_cols:
                if col in df_plot.columns:
                    valid_theta = df_plot[col].dropna()
                    if len(valid_theta) > 0:
                        if valid_theta.min() < -3 or valid_theta.max() > 3:
                            log(f"{name} {col} out of plausible range [-3, 3]: "
                                f"[{valid_theta.min():.2f}, {valid_theta.max():.2f}]")

        log(f"Theta ranges plausible")

        # Save Early plot data
        early_path = RQ_DIR / "plots" / "step06_piecewise_early_data.csv"
        df_early_plot.to_csv(early_path, index=False, encoding='utf-8')
        log(f"{early_path.name} ({len(df_early_plot)} rows)")

        # Save Late plot data
        late_path = RQ_DIR / "plots" / "step06_piecewise_late_data.csv"
        df_late_plot.to_csv(late_path, index=False, encoding='utf-8')
        log(f"{late_path.name} ({len(df_late_plot)} rows)")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
