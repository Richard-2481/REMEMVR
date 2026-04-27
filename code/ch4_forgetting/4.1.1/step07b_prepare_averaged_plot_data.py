"""
RQ 5.1.1 - Step 07b: Prepare Plot Data with Model-Averaged Predictions

PURPOSE:
Create plot data using model-averaged predictions (α_eff=0.410) instead of
single best model. Provides robust trajectory accounting for functional form
uncertainty.

INPUT:
- data/step04_lmm_input.csv (observed data)
- data/step05c_averaged_predictions.csv (model-averaged trajectory)

OUTPUT:
- plots/step07b_averaged_trajectory_data.csv (observed + averaged predictions)

Author: g_code
Date: 2025-12-08
RQ: ch5/5.1.1
Step: 07b
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07b_averaged_plot_data.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 07b: Prepare Averaged Plot Data")
        log("=" * 80)

        # Load data
        log("[LOAD] Loading observed data and averaged predictions...")
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step04_lmm_input.csv")
        averaged = pd.read_csv(RQ_DIR / "data" / "step05c_averaged_predictions.csv")
        
        log(f"  Observed: {len(lmm_input)} points")
        log(f"  Predicted: {len(averaged)} points")
        
        # Prepare observed data
        observed = lmm_input[['TSVR_hours', 'theta', 'UID']].copy()
        observed['data_type'] = 'observed'
        observed['prediction_se'] = np.nan
        
        # Prepare predicted data
        predicted = averaged[['TSVR_hours', 'theta_averaged', 'prediction_se']].copy()
        predicted.rename(columns={'theta_averaged': 'theta'}, inplace=True)
        predicted['UID'] = 'AVERAGED'
        predicted['data_type'] = 'predicted'
        
        # Combine
        log("[COMBINE] Creating combined plot dataset...")
        plot_data = pd.concat([observed, predicted], ignore_index=True)
        
        # Add confidence intervals for predictions
        plot_data['theta_lower'] = np.where(
            plot_data['data_type'] == 'predicted',
            plot_data['theta'] - 1.96 * plot_data['prediction_se'],
            np.nan
        )
        plot_data['theta_upper'] = np.where(
            plot_data['data_type'] == 'predicted',
            plot_data['theta'] + 1.96 * plot_data['prediction_se'],
            np.nan
        )
        
        # Save
        log("[SAVE] Saving plot data...")
        plot_path = RQ_DIR / "plots" / "step07b_averaged_trajectory_data.csv"
        plot_data.to_csv(plot_path, index=False, encoding='utf-8')
        log(f"  ✓ {plot_path.name} ({len(plot_data)} rows)")
        
        log("\n[SUMMARY]")
        log(f"  Observed: {len(observed)} participant-level points")
        log(f"  Predicted: {len(predicted)} averaged trajectory points")
        log(f"  Confidence bands: ±1.96 SE (95% CI)")
        log(f"  Functional form: Power law (α_eff=0.410)")
        
        log("=" * 80)
        log("[SUCCESS] Step 07b Complete")
        log(f"  Output: {plot_path.name}")
        log("=" * 80)
        
    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
