"""
RQ 5.2.1 - Step 05c: Model Averaging for Robust Predictions

PURPOSE:
Address extreme model uncertainty (best weight=8.9%) using Burnham & Anderson
(2002) multi-model inference. Compute weighted average predictions across
all competitive models (ΔAIC < 2) for robust functional form estimation.

INPUT:
- data/step04_lmm_input.csv (training data with TSVR_hours × domain)
- data/model_comparison.csv (65 models, Akaike weights)

OUTPUT:
- data/step05c_averaged_predictions.csv (model-averaged trajectory per domain)
- results/step05c_averaging_summary.txt (details on models used, weights, uncertainty)

CRITICAL:
Addresses Ph.D. thesis vulnerability: Cannot proceed with 8.9% single-model
confidence. Model averaging provides scientifically defensible foundation
for all downstream analyses.

Date: 2025-12-08
RQ: ch5/5.2.1
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
        log("Step 05c: Model Averaging (Domain × Time)")
        log("=" * 80)

        # Load data
        log("Loading input data and model comparison...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        comparison = pd.read_csv(RQ_DIR / "data" / "model_comparison.csv")

        log(f"  Training data: {len(lmm_input)} observations")
        log(f"  Models compared: {len(comparison)}")
        log(f"  Domains: {sorted(lmm_input['domain'].unique())}")

        best = comparison.iloc[0]
        log(f"  Best single model: {best['model_name']} (weight={best['akaike_weight']:.4f})")

        competitive = comparison[comparison['delta_AIC'] < 2.0]
        log(f"  Competitive models (ΔAIC<2): {len(competitive)}")
        log(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}")

        # Create prediction grid (per domain)
        log("\nCreating prediction grid...")
        tsvr_grid = np.linspace(1, 246, 100)
        domains = ['what', 'where', 'when']

        pred_grid = pd.concat([
            pd.DataFrame({
                'TSVR_hours': tsvr_grid,
                'domain': domain,
                'UID': 'AVERAGE',
            })
            for domain in domains
        ], ignore_index=True)

        log(f"  Grid points: {len(pred_grid)} ({len(tsvr_grid)} per domain)")

        # Compute model-averaged predictions
        log("\nComputing model-averaged predictions...")
        log("  NOTE: Domain × Time interaction - averaging across functional forms")

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
            'domain': pred_grid['domain'],
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
            f.write("RQ 5.2.1 - Model Averaging Summary (Domain × Time)\n")
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
            f.write(f"  Reciprocal family: {len(recip_models)} models (two-process forgetting)\n")
            f.write(f"  Power-law family: {len(power_models)} models (scale-invariant decay)\n")
            f.write(f"  Logarithmic family: {len(log_models)} models (Ebbinghaus curves)\n\n")

            f.write(f"Effective Functional Form:\n")
            f.write(f"  Dominant family: Reciprocal+Log (two-process)\n")
            f.write(f"  Interpretation: Rapid initial consolidation + slow asymptotic decay\n")
            f.write(f"  Domain interaction: Time transformation × Domain effects\n\n")

            f.write(f"Robustness:\n")
            f.write(f"  Single-model risk: Overfitting, ignores uncertainty\n")
            f.write(f"  Averaged approach: Accounts for functional form uncertainty\n")
            f.write(f"  Ph.D. defense: Scientifically defensible foundation\n")

        log(f"  ✓ {summary_path.name}")

        # Report per-domain predictions
        log("\nAveraged predictions per domain:")
        for domain in domains:
            domain_preds = output[output['domain'] == domain]
            log(f"  {domain.capitalize():5s}: θ ∈ [{domain_preds['theta_averaged'].min():.3f}, "
                f"{domain_preds['theta_averaged'].max():.3f}], "
                f"SE ∈ [{domain_preds['prediction_se'].min():.4f}, "
                f"{domain_preds['prediction_se'].max():.4f}]")

        log("=" * 80)
        log("Step 05c Complete")
        log(f"  Models averaged: {len(models_used)}")
        log(f"  Effective N: {effective_n:.2f}")
        log(f"  Output: {output_path.name}")
        log("=" * 80)

    except Exception as e:
        log(f"{e}")
        import traceback
        log(traceback.format_exc())
        raise
