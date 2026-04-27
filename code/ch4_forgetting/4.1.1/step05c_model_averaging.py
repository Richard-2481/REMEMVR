"""
RQ 5.1.1 - Step 05c: Model Averaging for Robust Predictions

PURPOSE:
Address extreme model uncertainty (best weight=5.6%) using Burnham & Anderson
(2002) multi-model inference. Compute weighted average predictions across
all competitive models (ΔAIC < 2) for robust functional form estimation.

INPUT:
- data/step04_lmm_input.csv (training data with TSVR_hours)
- data/step05_model_comparison.csv (66 models, Akaike weights)

OUTPUT:
- data/step05c_averaged_predictions.csv (model-averaged trajectory)
- results/step05c_averaging_summary.txt (details on models used, weights, uncertainty)

CRITICAL:
Addresses Ph.D. thesis vulnerability: Cannot proceed with 5.6% single-model
confidence. Model averaging provides scientifically defensible foundation
for all downstream analyses.

Author: g_code
Date: 2025-12-08
RQ: ch5/5.1.1
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
        log("[START] Step 05c: Model Averaging")
        log("=" * 80)

        # Load data
        log("[LOAD] Loading input data and model comparison...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        comparison = pd.read_csv(RQ_DIR / "data" / "step05_model_comparison.csv")
        
        log(f"  Training data: {len(lmm_input)} observations")
        log(f"  Models compared: {len(comparison)}")
        
        best = comparison.iloc[0]
        log(f"  Best single model: {best['model_name']} (weight={best['akaike_weight']:.4f})")
        
        competitive = comparison[comparison['delta_AIC'] < 2.0]
        log(f"  Competitive models (ΔAIC<2): {len(competitive)}")
        log(f"  Cumulative weight: {competitive['akaike_weight'].sum():.1%}")
        
        # Create prediction grid
        log("\n[GRID] Creating prediction grid...")
        tsvr_grid = np.linspace(1, 246, 100)
        pred_grid = pd.DataFrame({
            'TSVR_hours': tsvr_grid,
            'UID': 'AVERAGE',  # Dummy for population-level predictions
        })
        log(f"  Grid points: {len(pred_grid)}")
        
        # Compute model-averaged predictions
        log("\n[AVERAGING] Computing model-averaged predictions...")
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
        
        # Compute effective alpha (weighted average across power-law models)
        power_law_models = [m for m in models_used if 'PowerLaw' in m]
        if len(power_law_models) > 0:
            alphas = []
            alpha_weights = []
            for model in power_law_models:
                # Extract alpha from model name (e.g., 'PowerLaw_04' -> 0.4)
                if '_' in model:
                    alpha_str = model.split('_')[1]
                    alpha = float(alpha_str) / 10.0
                    alphas.append(alpha)
                    alpha_weights.append(weights[model])
            
            if len(alphas) > 0:
                alpha_weights = np.array(alpha_weights)
                alpha_weights = alpha_weights / alpha_weights.sum()  # Renormalize
                effective_alpha = np.average(alphas, weights=alpha_weights)
                log(f"\n[POWER LAW] Effective alpha:")
                log(f"  Weighted mean: α={effective_alpha:.3f}")
                log(f"  Range: [{min(alphas):.1f}, {max(alphas):.1f}]")
                log(f"  Power-law models: {len(power_law_models)}/{len(models_used)}")
        
        # Save averaged predictions
        log("\n[SAVE] Saving model-averaged predictions...")
        output = pd.DataFrame({
            'TSVR_hours': pred_grid['TSVR_hours'],
            'theta_averaged': averaged_preds,
            'prediction_variance': pred_var,
            'prediction_se': np.sqrt(pred_var),
        })
        
        output_path = RQ_DIR / "data" / "step05c_averaged_predictions.csv"
        output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"  ✓ {output_path.name} ({len(output)} rows)")
        
        # Save summary
        summary_path = RQ_DIR / "results" / "step05c_averaging_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RQ 5.1.1 - Model Averaging Summary\n")
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
            
            if len(power_law_models) > 0:
                f.write(f"\nEffective Functional Form:\n")
                f.write(f"  Family: Power law (Wixted & Ebbesen, 1991)\n")
                f.write(f"  Effective α: {effective_alpha:.3f}\n")
                f.write(f"  Formula: θ(t) = β₀ + β₁(t+1)^(-{effective_alpha:.3f})\n")
                f.write(f"  Interpretation: Proportional decay, scale-invariant\n\n")
            
            f.write(f"Robustness:\n")
            f.write(f"  Single-model risk: Overfitting, ignores uncertainty\n")
            f.write(f"  Averaged approach: Accounts for functional form uncertainty\n")
            f.write(f"  Ph.D. defense: Scientifically defensible foundation\n")
        
        log(f"  ✓ {summary_path.name}")
        
        log("=" * 80)
        log("[SUCCESS] Step 05c Complete")
        log(f"  Effective α: {effective_alpha:.3f}")
        log(f"  Models averaged: {len(models_used)}")
        log(f"  Output: {output_path.name}")
        log("=" * 80)
        
    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
