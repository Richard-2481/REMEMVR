"""
Random Slopes Comparison for RQ 5.3.9
Purpose: Test intercepts-only vs intercepts+slopes random effects structure
Required for quality validation (Section 4.4)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from statsmodels.regression.mixed_linear_model import MixedLM

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = RQ_DIR / "data" / "step02_lmm_input.csv"
OUTPUT_FILE = RQ_DIR / "data" / "random_slopes_comparison.csv"
LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"

def log(msg):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def main():
    log("Random Slopes Comparison")
    
    # Load data
    log(f"Loading data from {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    log(f"Loaded {len(data)} observations")
    
    # Define formula (same fixed effects for both models)
    formula = "Response ~ Time * Difficulty_c * C(paradigm)"
    
    # MODEL 1: Intercepts-only
    log("\n[MODEL 1] Fitting intercepts-only model...")
    model_intercepts = MixedLM.from_formula(
        formula,
        data=data,
        groups=data['UID'],
        re_formula='1'  # Intercepts only
    )
    result_intercepts = model_intercepts.fit(reml=False)  # Use ML for AIC comparison
    log(f"[MODEL 1] Converged: {result_intercepts.converged}")
    log(f"[MODEL 1] AIC: {result_intercepts.aic:.2f}")
    log(f"[MODEL 1] Log-likelihood: {result_intercepts.llf:.2f}")
    
    # MODEL 2: Intercepts + slopes
    log("\n[MODEL 2] Fitting intercepts + slopes model...")
    model_slopes = MixedLM.from_formula(
        formula,
        data=data,
        groups=data['UID'],
        re_formula='~Time'  # Intercepts + slopes on Time
    )
    result_slopes = model_slopes.fit(reml=False)  # Use ML for AIC comparison
    log(f"[MODEL 2] Converged: {result_slopes.converged}")
    log(f"[MODEL 2] AIC: {result_slopes.aic:.2f}")
    log(f"[MODEL 2] Log-likelihood: {result_slopes.llf:.2f}")
    
    # Extract random slope variance (if converged)
    if result_slopes.converged:
        # Get covariance matrix of random effects
        cov_re = result_slopes.cov_re
        if hasattr(cov_re, 'iloc') and cov_re.shape[0] >= 2:
            slope_var = cov_re.iloc[1, 1]
            slope_sd = np.sqrt(slope_var)
            log(f"[MODEL 2] Random slope variance: {slope_var:.6f}")
            log(f"[MODEL 2] Random slope SD: {slope_sd:.6f}")
        else:
            slope_var = np.nan
            slope_sd = np.nan
            log(f"[MODEL 2] WARNING: Could not extract slope variance (boundary issue)")
    else:
        slope_var = np.nan
        slope_sd = np.nan
        log(f"[MODEL 2] WARNING: Model did not converge")
    
    # Compute ΔAIC (positive = slopes preferred, negative = intercepts preferred)
    delta_aic = result_intercepts.aic - result_slopes.aic
    log(f"\nΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    
    # Interpret outcome
    if delta_aic > 2:
        outcome = "Option A: Slopes improve fit (ΔAIC > 2)"
        recommendation = "Use slopes model - individual differences confirmed"
        log(f"{outcome}")
        log(f"{recommendation}")
    elif not result_slopes.converged or slope_var < 1e-10:
        outcome = "Option B: Slopes don't converge / overfit"
        recommendation = "Keep intercepts-only - insufficient data for stable slope estimation"
        log(f"{outcome}")
        log(f"{recommendation}")
    else:
        outcome = "Option C: Slopes converge but don't improve (|ΔAIC| < 2)"
        recommendation = "Keep intercepts-only (homogeneous effects CONFIRMED via test)"
        log(f"{outcome}")
        log(f"{recommendation}")
    
    # Save comparison results
    comparison = pd.DataFrame({
        'model': ['Intercepts_Only', 'Intercepts_Slopes'],
        'aic': [result_intercepts.aic, result_slopes.aic],
        'log_likelihood': [result_intercepts.llf, result_slopes.llf],
        'converged': [result_intercepts.converged, result_slopes.converged],
        'delta_aic': [0.0, delta_aic],
        'slope_variance': [0.0, slope_var],
        'slope_sd': [0.0, slope_sd],
        'outcome': [outcome, outcome],
        'recommendation': [recommendation, recommendation]
    })
    
    comparison.to_csv(OUTPUT_FILE, index=False)
    log(f"\nComparison results to {OUTPUT_FILE}")
    log("Random slopes comparison finished")
    
    return comparison

if __name__ == '__main__':
    main()
