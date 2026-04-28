"""
RQ 6.1.1 - Step 05c: Population-Level Model-Averaged Predictions

PURPOSE:
Generate population-level (fixed-effects only) model-averaged predictions
for confidence trajectories, analogous to Ch5 5.1.1 step05c. This enables
direct visual comparison of functional forms in cross-chapter analyses.

INPUT:
- data/step04_lmm_input.csv (training data with TSVR_hours)
- data/step05b_competitive_models.csv (competitive models, ΔAIC < 7)

OUTPUT:
- data/step05c_population_predictions.csv (population-level trajectory)

RATIONALE:
Ch6 step05b generates individual-level predictions (with random effects),
but cross-chapter comparisons (e.g., RQ 6.9.1) require population-level
predictions to compare functional forms. This script fills that gap.

Date: 2026-02-04
RQ: ch6/6.1.1
Step: 05c
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_averaging import compute_model_averaged_predictions

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05c_population_predictions.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 05c: Population-Level Model-Averaged Predictions")
        log("=" * 80)

        # Load data
        log("\nLoading input data...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")

        # Load full competitive models from step05b (ΔAIC < 7, 48 models)
        competitive_full = pd.read_csv(RQ_DIR / "data" / "step05b_competitive_models.csv")

        # Filter to ΔAIC < 2 to match Ch5 threshold
        competitive = competitive_full[competitive_full['delta_AIC'] < 2.0].copy()

        # Renormalize weights among ΔAIC < 2 models
        competitive['renorm_weight'] = competitive['akaike_weight'] / competitive['akaike_weight'].sum()

        log(f"  Training data: {len(lmm_input)} observations")
        log(f"  Competitive models: {len(competitive)}")

        best = competitive.iloc[0]
        log(f"  Best model: {best['model_name']} (renorm_weight={best['renorm_weight']:.4f})")
        log(f"  Cumulative weight: {competitive['renorm_weight'].sum():.1%}")

        # Create prediction grid (population-level)
        log("\nCreating population-level prediction grid...")
        tsvr_max = lmm_input['TSVR_hours'].max()
        tsvr_grid = np.linspace(1, tsvr_max, 100)
        pred_grid = pd.DataFrame({
            'TSVR_hours': tsvr_grid,
            'UID': 'POPULATION',  # Dummy UID for fixed-effects only
        })
        log(f"  Grid points: {len(pred_grid)}")
        log(f"  TSVR range: [1, {tsvr_max:.1f}] hours")

        # Compute model-averaged predictions
        log("\nComputing model-averaged predictions...")

        # Create comparison dataframe in the format expected by the function
        # Drop old akaike_weight column and use renorm_weight
        comparison_for_function = competitive.drop(columns=['akaike_weight']).rename(
            columns={'renorm_weight': 'akaike_weight'}
        )
        comparison_for_function['delta_AIC'] = comparison_for_function['AIC'] - comparison_for_function['AIC'].min()

        results = compute_model_averaged_predictions(
            data=lmm_input,
            comparison=comparison_for_function,
            outcome_var='theta_All',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            delta_aic_threshold=2.0,  # Match Ch5 threshold for visual consistency
            prediction_grid=pred_grid,
            reml=False,
        )

        # Extract results
        averaged_preds = results['averaged_predictions']
        models_used = results['models_used']
        weights = results['weights_normalized']
        pred_var = results['prediction_variance']
        effective_n = results['effective_n_models']

        log(f"\nModel averaging complete:")
        log(f"  Models used: {len(models_used)}")
        log(f"  Effective N models: {effective_n:.2f}")
        log(f"  Prediction variance: [{pred_var.min():.6f}, {pred_var.max():.6f}]")

        # Save population-level predictions
        log("\nSaving population-level predictions...")
        output = pd.DataFrame({
            'TSVR_hours': pred_grid['TSVR_hours'],
            'theta_averaged': averaged_preds,
            'prediction_variance': pred_var,
            'prediction_se': np.sqrt(pred_var),
        })

        output_path = RQ_DIR / "data" / "step05c_population_predictions.csv"
        output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"  ✓ {output_path.name} ({len(output)} rows)")

        # Summary stats
        log("\nPrediction characteristics:")
        log(f"  Mean theta: {averaged_preds.mean():.4f}")
        log(f"  SD theta: {averaged_preds.std():.4f}")
        log(f"  Range: [{averaged_preds.min():.4f}, {averaged_preds.max():.4f}]")
        log(f"  Total decline (t=1 to t=246): {averaged_preds.iloc[0] - averaged_preds.iloc[-1]:.4f} theta units")

        log("\n" + "=" * 80)
        log("Step 05c Complete")
        log(f"  Models averaged: {len(models_used)}")
        log(f"  Output: {output_path.name}")
        log("=" * 80)

    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
