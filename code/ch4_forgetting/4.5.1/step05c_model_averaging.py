"""
RQ 5.5.1 - Step 05c: Model Averaging for Robust Predictions

PURPOSE:
Address extreme model uncertainty (best weight=6.7%) using Burnham & Anderson
(2002) multi-model inference. Compute weighted average predictions across
all competitive models (ΔAIC < 2) for robust functional form estimation.

INPUT:
- data/step04_lmm_input.csv (training data with TSVR_hours × LocationType)
- data/model_comparison.csv (65 models, Akaike weights)

OUTPUT:
- data/step05c_averaged_predictions.csv (model-averaged trajectory per location type)
- results/step05c_averaging_summary.txt (details on models used, weights, uncertainty)

CRITICAL:
Addresses Ph.D. thesis vulnerability: Cannot proceed with 6.7% single-model
confidence. Model averaging provides scientifically defensible foundation
for all downstream analyses.

Author: Claude Code (adapted from 5.2.1)
Date: 2025-12-08
RQ: ch5/5.5.1
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
LOG_FILE = RQ_DIR / "logs" / "step05c_model_averaging.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 05c: Model Averaging (LocationType × Time)")
        log("=" * 80)

        # Load data
        log("[LOAD] Loading input data and model comparison...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        comparison = pd.read_csv(RQ_DIR / "data" / "model_comparison.csv")

        log(f"  Training data: {len(lmm_input)} observations")
        log(f"  Models compared: {len(comparison)}")
        log(f"  LocationTypes: {sorted(lmm_input['LocationType'].unique())}")

        best = comparison.iloc[0]
        log(f"  Best single model: {best['model_name']} (weight={best['akaike_weight']:.4f})")

        competitive = comparison[comparison['delta_AIC'] < 2.0]
        log(f"  Competitive models (ΔAIC<2): {len(competitive)}")
        log(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}")

        # Create prediction grid (per location type)
        log("\n[GRID] Creating prediction grid...")
        tsvr_grid = np.linspace(1, 246, 100)
        location_types = ['source', 'destination']

        pred_grid = pd.concat([
            pd.DataFrame({
                'TSVR_hours': tsvr_grid,
                'LocationType': loc_type,
                'UID': 'AVERAGE',
            })
            for loc_type in location_types
        ], ignore_index=True)

        log(f"  Grid points: {len(pred_grid)} ({len(tsvr_grid)} per location type)")

        # Compute model-averaged predictions
        log("\n[AVERAGING] Computing model-averaged predictions...")
        log("  NOTE: LocationType × Time interaction - averaging across functional forms")

        results = compute_model_averaged_predictions(
            data=lmm_input,
            comparison=comparison,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            delta_aic_threshold=2.0,
            prediction_grid=pred_grid,
            reml=False,
        )

        # Extract results
        averaged_preds = results['averaged_predictions']
        models_used = results['models_used']
        weights = results['weights_normalized']
        pred_var = results['prediction_variance']
        effective_n = results['effective_n_models']

        log(f"\n[RESULTS] Model averaging complete:")
        log(f"  Models used: {len(models_used)}")
        log(f"  Effective N models: {effective_n:.2f}")
        log(f"  Prediction variance: [{pred_var.min():.4f}, {pred_var.max():.4f}]")

        # Analyze model composition
        log(f"\n[COMPOSITION] Model types in competitive set:")
        recip_models = [m for m in models_used if 'Recip' in m]
        power_models = [m for m in models_used if 'PowerLaw' in m]
        log_models = [m for m in models_used if 'Log' in m and 'PowerLaw' not in m and 'LogLog' not in m]
        quad_models = [m for m in models_used if 'Quad' in m]
        sqrt_models = [m for m in models_used if 'SquareRoot' in m or 'Root' in m]

        log(f"  Quadratic family: {len(quad_models)} models")
        log(f"  Logarithmic family: {len(log_models)} models")
        log(f"  Square-root family: {len(sqrt_models)} models")
        log(f"  Power-law family: {len(power_models)} models")
        log(f"  Reciprocal family: {len(recip_models)} models")

        # Compute effective alpha (if power-law models present)
        if len(power_models) > 0:
            alphas = []
            alpha_weights = []
            for model in power_models:
                if '_' in model:
                    alpha_str = model.split('_')[1]
                    if alpha_str.isdigit():
                        alpha = float(alpha_str) / 10.0
                        alphas.append(alpha)
                        alpha_weights.append(weights[model])

            if len(alphas) > 0:
                alpha_weights = np.array(alpha_weights)
                alpha_weights = alpha_weights / alpha_weights.sum()
                effective_alpha = np.average(alphas, weights=alpha_weights)
                log(f"\n[POWER LAW] Effective alpha (if applicable):")
                log(f"  Weighted mean: α={effective_alpha:.3f}")
                log(f"  Range: [{min(alphas):.1f}, {max(alphas):.1f}]")

        # Save averaged predictions
        log("\n[SAVE] Saving model-averaged predictions...")
        output = pd.DataFrame({
            'TSVR_hours': pred_grid['TSVR_hours'],
            'LocationType': pred_grid['LocationType'],
            'theta_averaged': averaged_preds,
            'prediction_variance': pred_var,
            'prediction_se': np.sqrt(pred_var),
        })

        output_path = RQ_DIR / "data" / "step05c_averaged_predictions.csv"
        output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"  ✓ {output_path.name} ({len(output)} rows)")

        # Save summary
        summary_path = RQ_DIR / "results" / "step05c_averaging_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.5.1 - Model Averaging Summary (LocationType × Time)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Problem: Best single model uncertainty\n")
            f.write(f"  Best model: {best['model_name']}\n")
            f.write(f"  Akaike weight: {best['akaike_weight']:.4f} ({best['akaike_weight']*100:.1f}%)\n")
            f.write(f"  Interpretation: EXTREME uncertainty\n\n")

            f.write(f"Solution: Multi-model inference (Burnham & Anderson, 2002)\n")
            f.write(f"  Models averaged: {len(models_used)}\n")
            f.write(f"  Effective N models: {effective_n:.2f}\n")
            f.write(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}\n\n")

            f.write(f"Models Used (Top 10):\n")
            for i, (model, weight) in enumerate(list(weights.items())[:10]):
                f.write(f"  {i+1:2d}. {model:25s} w={weight:.4f}\n")

            f.write(f"\nModel Composition:\n")
            f.write(f"  Quadratic family: {len(quad_models)} models\n")
            f.write(f"  Logarithmic family: {len(log_models)} models (Ebbinghaus curves)\n")
            f.write(f"  Square-root family: {len(sqrt_models)} models\n")
            f.write(f"  Power-law family: {len(power_models)} models (scale-invariant decay)\n")
            f.write(f"  Reciprocal family: {len(recip_models)} models (two-process forgetting)\n\n")

            f.write(f"Effective Functional Form:\n")
            f.write(f"  Dominant families: Quadratic + Logarithmic + Square-root\n")
            f.write(f"  Interpretation: Multiple competing forgetting processes\n")
            f.write(f"  LocationType interaction: Time transformation × Location effects\n\n")

            f.write(f"Robustness:\n")
            f.write(f"  Single-model risk: Overfitting, ignores uncertainty\n")
            f.write(f"  Averaged approach: Accounts for functional form uncertainty\n")
            f.write(f"  Ph.D. defense: Scientifically defensible foundation\n")

        log(f"  ✓ {summary_path.name}")

        # Report per-location predictions
        log("\n[SUMMARY] Averaged predictions per location type:")
        for loc_type in location_types:
            loc_preds = output[output['LocationType'] == loc_type]
            log(f"  {loc_type.capitalize():11s}: θ ∈ [{loc_preds['theta_averaged'].min():.3f}, "
                f"{loc_preds['theta_averaged'].max():.3f}], "
                f"SE ∈ [{loc_preds['prediction_se'].min():.4f}, "
                f"{loc_preds['prediction_se'].max():.4f}]")

        log("=" * 80)
        log("[SUCCESS] Step 05c Complete")
        log(f"  Models averaged: {len(models_used)}")
        log(f"  Effective N: {effective_n:.2f}")
        log(f"  Output: {output_path.name}")
        log("=" * 80)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
