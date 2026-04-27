#!/usr/bin/env python3
"""
Step 10: Compare to Accuracy Prediction
Compare calibration prediction to accuracy prediction from RQ 7.1.1 or 7.3.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import glob

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
PROJECT_ROOT = RQ_DIR.parents[2]
LOG_FILE = RQ_DIR / "logs" / "step10_accuracy_comparison.log"

def log(msg):
    """Write to log file and print"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    log("[START] Step 10: Accuracy Comparison")
    
    # Load our calibration results
    log("[LOAD] Loading calibration prediction results...")
    calib_reg = pd.read_csv(DATA_DIR / "step04_model_comparison.csv")
    calib_pred = pd.read_csv(DATA_DIR / "step09_individual_predictors.csv")
    calib_effect = pd.read_csv(DATA_DIR / "step08_effect_summary.csv")
    
    calib_r2 = calib_reg[calib_reg['model'] == 'Full_Model']['r2'].iloc[0]
    
    # Try to find accuracy prediction results (7.3.1 or 7.1.1)
    log("[SEARCH] Looking for accuracy prediction results...")
    accuracy_found = False
    accuracy_r2 = None
    accuracy_source = None
    
    # Check RQ 7.3.1 (confidence vs accuracy)
    rq731_path = PROJECT_ROOT / "results" / "ch7" / "7.3.1" / "data"
    if rq731_path.exists():
        # Try different possible filenames
        files = list(rq731_path.glob("*hierarchical*.csv")) + list(rq731_path.glob("*model*.csv"))
        if files:
            try:
                acc_df = pd.read_csv(files[0])
                # Try different column names
                if 'R_squared' in acc_df.columns:
                    # Look for cognitive or full model
                    if 'Cognitive' in acc_df['model'].values:
                        accuracy_r2 = acc_df[acc_df['model'] == 'Cognitive']['R_squared'].iloc[0]
                    else:
                        accuracy_r2 = acc_df[acc_df['model'].str.contains('Full|Cognitive', case=False, na=False)]['R_squared'].iloc[0]
                    accuracy_source = "RQ 7.3.1"
                    accuracy_found = True
                    log(f"[FOUND] Accuracy results from {accuracy_source}: R² = {accuracy_r2:.4f}")
                elif 'r2' in acc_df.columns:
                    accuracy_r2 = acc_df[acc_df['model'].str.contains('Full', case=False, na=False)]['r2'].iloc[0]
                    accuracy_source = "RQ 7.3.1"  
                    accuracy_found = True
                    log(f"[FOUND] Accuracy results from {accuracy_source}: R² = {accuracy_r2:.4f}")
            except:
                pass
    
    # Check RQ 7.1.1 if not found
    if not accuracy_found:
        rq711_path = PROJECT_ROOT / "results" / "ch7" / "7.1.1" / "data"
        if rq711_path.exists():
            files = list(rq711_path.glob("*regression*.csv"))
            if files:
                try:
                    acc_df = pd.read_csv(files[0])
                    if 'r2' in acc_df.columns:
                        accuracy_r2 = acc_df['r2'].iloc[0] if 'r2' in acc_df else None
                        accuracy_source = "RQ 7.1.1"
                        accuracy_found = True
                        log(f"[FOUND] Accuracy results from {accuracy_source}: R² = {accuracy_r2:.4f}")
                except:
                    pass
    
    # Create comparison
    log("[ANALYSIS] Creating calibration vs accuracy comparison...")
    
    comparison_data = {
        'measure': ['calibration_quality', 'accuracy'],
        'r2': [calib_r2, accuracy_r2 if accuracy_found else np.nan],
        'f2': [calib_effect['overall_f2'].iloc[0], 
                (accuracy_r2/(1-accuracy_r2) if accuracy_r2 and accuracy_r2 < 1 else np.nan) if accuracy_found else np.nan],
        'interpretation': [calib_effect['overall_interpretation'].iloc[0],
                          ('negligible' if accuracy_r2 and accuracy_r2 < 0.02 else 
                           ('small' if accuracy_r2 < 0.13 else 
                            ('medium' if accuracy_r2 < 0.26 else 'large'))) if accuracy_found else 'not_available'],
        'source': ['Current RQ (7.3.2)', accuracy_source if accuracy_found else 'not_found']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Theoretical interpretation
    if accuracy_found and accuracy_r2 is not None:
        calibration_harder = calib_r2 < accuracy_r2
        log(f"[RESULT] Calibration prediction R² = {calib_r2:.4f}")
        log(f"[RESULT] Accuracy prediction R² = {accuracy_r2:.4f}")
        log(f"[RESULT] Calibration harder to predict: {calibration_harder}")
        
        # Compute difference
        r2_diff = accuracy_r2 - calib_r2
        log(f"[RESULT] R² difference = {r2_diff:.4f}")
        
        interpretation = (
            "Supports metacognitive dissociation: Calibration quality (matching confidence to accuracy) "
            "is harder to predict than raw accuracy, suggesting it involves distinct cognitive processes "
            "beyond those measured by standard cognitive tests."
            if calibration_harder else
            "Does not support metacognitive dissociation hypothesis."
        )
    else:
        interpretation = "No accuracy comparison available - cannot test metacognitive dissociation hypothesis"
        log("[WARNING] No accuracy prediction results found for comparison")
    
    # Save results
    log("[SAVE] Saving comparison results...")
    comparison_df.to_csv(DATA_DIR / "step10_accuracy_comparison.csv", index=False)
    log(f"[SAVED] step10_accuracy_comparison.csv")
    
    # Save interpretation
    interp_df = pd.DataFrame([{
        'comparison_performed': accuracy_found,
        'calibration_r2': calib_r2,
        'accuracy_r2': accuracy_r2 if accuracy_found else np.nan,
        'r2_difference': (accuracy_r2 - calib_r2) if accuracy_found and accuracy_r2 is not None else np.nan,
        'calibration_harder': calibration_harder if accuracy_found and accuracy_r2 is not None else None,
        'interpretation': interpretation
    }])
    
    interp_df.to_csv(DATA_DIR / "step10_interpretation.csv", index=False)
    log(f"[SAVED] step10_interpretation.csv")
    
    log("[SUCCESS] Step 10 complete")