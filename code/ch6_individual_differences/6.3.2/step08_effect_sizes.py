#!/usr/bin/env python3
"""
Step 08: Effect Sizes
Extract and interpret effect sizes from regression results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs" / "step08_effect_sizes.log"

def log(msg):
    """Write to log file and print"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    log("Step 08: Effect Sizes")
    
    # Load regression results
    log("Loading regression results...")
    reg_df = pd.read_csv(DATA_DIR / "step04_regression_results.csv")
    model_df = pd.read_csv(DATA_DIR / "step04_model_comparison.csv")
    
    # Extract effect sizes
    log("Computing effect sizes...")
    full_model = model_df[model_df['model'] == 'Full_Model'].iloc[0]
    r2 = full_model['r2']
    adj_r2 = full_model['adj_r2']
    
    # Cohen's f^2 = R^2 / (1 - R^2)
    f2 = r2 / (1 - r2) if r2 < 1 else 0
    
    # Individual effect sizes from sr_squared
    effect_data = []
    for _, row in reg_df.iterrows():
        predictor = row['predictor']
        beta = row['beta']
        sr2 = row['sr_squared']
        
        # Individual f^2
        ind_f2 = sr2 / (1 - r2) if r2 < 1 else 0
        
        # Interpretation
        if abs(ind_f2) < 0.02:
            interpretation = "negligible"
        elif abs(ind_f2) < 0.15:
            interpretation = "small"
        elif abs(ind_f2) < 0.35:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        effect_data.append({
            'predictor': predictor,
            'beta': beta,
            'sr_squared': sr2,
            'f_squared': ind_f2,
            'interpretation': interpretation
        })
    
    effect_df = pd.DataFrame(effect_data)
    
    # Save results
    log("Saving effect size results...")
    effect_df.to_csv(DATA_DIR / "step08_effect_sizes.csv", index=False)
    log(f"step08_effect_sizes.csv ({len(effect_df)} predictors)")
    
    # Summary statistics
    summary_df = pd.DataFrame([{
        'overall_r2': r2,
        'overall_adj_r2': adj_r2,
        'overall_f2': f2,
        'overall_interpretation': 'negligible' if f2 < 0.02 else ('small' if f2 < 0.15 else 'medium'),
        'n_predictors': len(effect_df),
        'n_significant': len(effect_df[effect_df['f_squared'] >= 0.02])
    }])
    
    summary_df.to_csv(DATA_DIR / "step08_effect_summary.csv", index=False)
    log(f"step08_effect_summary.csv")
    
    log(f"Overall R² = {r2:.4f}, f² = {f2:.4f}")
    log(f"Predictors with at least small effect: {summary_df['n_significant'].iloc[0]}/{summary_df['n_predictors'].iloc[0]}")
    
    log("Step 08 complete")