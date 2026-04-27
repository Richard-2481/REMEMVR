"""
RQ 5.1.1 - Step 07: Prepare Functional Form Plot Data

PURPOSE:
Generate plot data for best model (PowerLaw_04) forgetting trajectory.
Creates prediction grid for visualization.

INPUT:
- data/step04_lmm_input.csv (observed data)
- data/step05_model_comparison.csv (best model identification)

OUTPUT:
- plots/step07_functional_form_data.csv (observed + predicted trajectory)

Author: g_code
Date: 2025-12-08
RQ: ch5/5.1.1
Step: 07
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_plot_data.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 07: Prepare Plot Data")
        log("=" * 80)

        # Load data
        log("[LOAD] Loading input data...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        comparison = pd.read_csv(RQ_DIR / "data" / "step05_model_comparison.csv")
        
        best_model_name = comparison.iloc[0]['model_name']
        log(f"  Best model: {best_model_name}")
        
        # Create time transformations
        log("[TRANSFORM] Creating time transformations...")
        lmm_input['TSVR_days'] = lmm_input['TSVR_hours'] / 24.0
        lmm_input['log_TSVR'] = np.log(lmm_input['TSVR_days'] + 1)
        lmm_input['PowerLaw_04'] = (lmm_input['TSVR_days'] + 1) ** (-0.4)
        
        # Fit best model
        log(f"[FIT] Fitting {best_model_name} model...")
        formula = 'theta ~ PowerLaw_04'
        model = smf.mixedlm(formula, lmm_input, groups=lmm_input['UID']).fit(reml=False)
        log(f"  ✓ Model converged")
        log(f"  AIC: {model.aic:.2f}")
        
        # Create prediction grid
        log("[PREDICT] Creating prediction grid...")
        tsvr_grid = np.linspace(1, 246, 100)
        pred_data = pd.DataFrame({
            'TSVR_hours': tsvr_grid,
            'TSVR_days': tsvr_grid / 24.0
        })
        pred_data['PowerLaw_04'] = (pred_data['TSVR_days'] + 1) ** (-0.4)
        pred_data['predicted_theta'] = model.predict(pred_data)
        
        # Combine with observed data
        log("[COMBINE] Creating plot dataset...")
        observed = lmm_input[['TSVR_hours', 'theta', 'UID']].copy()
        observed['data_type'] = 'observed'
        
        predicted = pred_data[['TSVR_hours', 'predicted_theta']].copy()
        predicted['data_type'] = 'predicted'
        predicted['UID'] = 'ALL'
        predicted.rename(columns={'predicted_theta': 'theta'}, inplace=True)
        
        plot_data = pd.concat([observed, predicted], ignore_index=True)
        
        # Save
        log("[SAVE] Saving plot data...")
        plot_path = RQ_DIR / "plots" / "step07_functional_form_data.csv"
        plot_data.to_csv(plot_path, index=False, encoding='utf-8')
        log(f"  ✓ {plot_path.name} ({len(plot_data)} rows)")
        
        log("=" * 80)
        log("[SUCCESS] Step 07 Complete")
        log(f"  Model: {best_model_name}")
        log(f"  Observed: {len(observed)} points")
        log(f"  Predicted: {len(predicted)} points")
        log("=" * 80)
        
    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
