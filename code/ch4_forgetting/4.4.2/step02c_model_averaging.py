"""
RQ 5.4.2 - Step 02c: Model Averaging for Robust Predictions

PURPOSE:
Address extreme model uncertainty (best weight=6.04%) using Burnham & Anderson
(2002) multi-model inference. Compute weighted average predictions across
all competitive models (ΔAIC < 2) for robust functional form estimation.

INPUT:
- data/step01_lmm_input_piecewise.csv (training data with TSVR_hours × Congruence)
- data/step02b_model_comparison.csv (65 models, Akaike weights)

OUTPUT:
- data/step02c_averaged_predictions.csv (model-averaged trajectory per congruence level)
- results/step02c_averaging_summary.txt (details on models used, weights, uncertainty)

CRITICAL:
Addresses Ph.D. thesis vulnerability: Cannot proceed with 6.04% single-model
confidence. Model averaging provides scientifically defensible foundation
for all downstream analyses.

CONTEXT:
- Original piecewise (step02): AIC = 2581.55 (with random slopes)
- Extended kitchen sink (step02b): PowerLaw_01 AIC = 2593.41, 6.04% weight (random intercepts only)
- 15 competitive models (ΔAIC < 2): Extreme uncertainty
- This step: Model averaging MANDATORY for thesis defense

Date: 2025-12-09
RQ: ch5/5.4.2
Step: 02c
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_averaging import compute_model_averaged_predictions

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02c_model_averaging.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 02c: Model Averaging (Congruence × Time)")
        log("=" * 80)

        # Load data
        log("Loading input data and model comparison...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step01_lmm_input_piecewise.csv")
        comparison = pd.read_csv(RQ_DIR / "data" / "step02b_model_comparison.csv")

        log(f"  Training data: {len(lmm_input)} observations")
        log(f"  Models compared: {len(comparison)}")
        log(f"  Congruence levels: {sorted(lmm_input['Congruence'].unique())}")

        best = comparison.iloc[0]
        log(f"  Best single model: {best['model_name']} (weight={best['akaike_weight']:.4f})")

        competitive = comparison[comparison['delta_AIC'] < 2.0]
        log(f"  Competitive models (ΔAIC<2): {len(competitive)}")
        log(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}")

        # Create prediction grid (per congruence level)
        log("\nCreating prediction grid...")
        tsvr_grid = np.linspace(1, 246, 100)
        congruence_levels = ['Common', 'Congruent', 'Incongruent']

        pred_grid = pd.concat([
            pd.DataFrame({
                'TSVR_hours': tsvr_grid,
                'Congruence': level,
                'UID': 'AVERAGE',
            })
            for level in congruence_levels
        ], ignore_index=True)

        log(f"  Grid points: {len(pred_grid)} ({len(tsvr_grid)} per congruence level)")

        # Compute model-averaged predictions
        log("\nComputing model-averaged predictions...")
        log("  NOTE: Congruence × Time interaction - averaging across functional forms")

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

        log(f"\nModel averaging complete:")
        log(f"  Models used: {len(models_used)}")
        log(f"  Effective N models: {effective_n:.2f}")
        log(f"  Prediction variance: [{pred_var.min():.4f}, {pred_var.max():.4f}]")

        # Analyze model composition
        log(f"\nModel types in competitive set:")
        recip_models = [m for m in models_used if 'Recip' in m]
        power_models = [m for m in models_used if 'PowerLaw' in m]
        log_models = [m for m in models_used if 'Log' in m and 'PowerLaw' not in m and 'LogLog' not in m]

        log(f"  Reciprocal family: {len(recip_models)} models")
        log(f"  Power-law family: {len(power_models)} models")
        log(f"  Logarithmic family: {len(log_models)} models")

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
        log("\nSaving model-averaged predictions...")
        output = pd.DataFrame({
            'TSVR_hours': pred_grid['TSVR_hours'],
            'Congruence': pred_grid['Congruence'],
            'theta_averaged': averaged_preds,
            'prediction_variance': pred_var,
        })

        output_path = RQ_DIR / "data" / "step02c_averaged_predictions.csv"
        output.to_csv(output_path, index=False)
        log(f"  ✓ {output_path.name}")

        # Save summary
        summary_path = RQ_DIR / "results" / "step02c_averaging_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RQ 5.4.2 - Model Averaging Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Models used: {len(models_used)}\n")
            f.write(f"Effective N models: {effective_n:.2f}\n")
            f.write(f"ΔAIC threshold: 2.0\n\n")
            f.write("Model Weights (Normalized):\n")
            f.write("-" * 80 + "\n")
            for model, weight in sorted(weights.items(), key=lambda x: -x[1]):
                f.write(f"  {model:30s} {weight:.4f}\n")
            f.write("\n")
            f.write(f"Prediction Variance Range: [{pred_var.min():.4f}, {pred_var.max():.4f}]\n")

        log(f"  ✓ {summary_path.name}")

        log("\n" + "=" * 80)
        log("Step 02c complete")
        log("=" * 80)

    except Exception as e:
        log(f"{type(e).__name__}: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
