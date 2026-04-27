#!/usr/bin/env python3
"""
RQ 6.4.3: Random Slopes Comparison
Tests intercepts-only vs intercepts+slopes to validate random effects structure.
MANDATORY per improvement_taxonomy.md Section 4.4.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

# File paths
RQ_DIR = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.3")
INPUT_FILE = RQ_DIR / "data" / "step00_lmm_input.csv"
OUTPUT_FILE = RQ_DIR / "data" / "random_slopes_comparison.csv"
LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"

def log(msg):
    """Log to file and print."""
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def main():
    """Compare intercepts-only vs intercepts+slopes random effects."""
    
    log("=" * 80)
    log("RANDOM SLOPES COMPARISON: RQ 6.4.3")
    log("=" * 80)
    
    # Load data
    log(f"\nLoading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    log(f"Loaded {len(df)} observations")
    
    # Prepare categorical variables
    df['Paradigm_cat'] = pd.Categorical(df['Paradigm'], 
                                        categories=['IFR', 'ICR', 'IRE'])
    
    log("\n" + "=" * 80)
    log("MODEL 1: INTERCEPTS-ONLY (Current implementation check)")
    log("=" * 80)
    
    # Fit intercepts-only model
    log("\nFitting intercepts-only model...")
    log("Formula: theta_confidence ~ log_TSVR * C(Paradigm_cat) * Age_c")
    log("Random: ~1 (intercepts only)")
    
    model_intercepts = smf.mixedlm(
        "theta_confidence ~ log_TSVR * C(Paradigm_cat) * Age_c",
        data=df,
        groups=df['UID'],
        re_formula="~1"  # Intercepts only
    )
    
    result_intercepts = model_intercepts.fit(reml=False, method='powell', maxiter=2000)
    
    log(f"\nConverged: {result_intercepts.converged}")
    log(f"AIC: {result_intercepts.aic:.2f}")
    log(f"BIC: {result_intercepts.bic:.2f}")
    log(f"Log-likelihood: {result_intercepts.llf:.2f}")
    log(f"Random intercept variance: {result_intercepts.cov_re.iloc[0,0]:.6f}")
    
    log("\n" + "=" * 80)
    log("MODEL 2: INTERCEPTS + SLOPES (Testing random slope necessity)")
    log("=" * 80)
    
    # Fit intercepts + slopes model
    log("\nFitting intercepts+slopes model...")
    log("Formula: theta_confidence ~ log_TSVR * C(Paradigm_cat) * Age_c")
    log("Random: ~log_TSVR (intercepts + slopes on time)")
    
    model_slopes = smf.mixedlm(
        "theta_confidence ~ log_TSVR * C(Paradigm_cat) * Age_c",
        data=df,
        groups=df['UID'],
        re_formula="~log_TSVR"  # Intercepts + slopes
    )
    
    result_slopes = model_slopes.fit(reml=False, method='powell', maxiter=2000)
    
    log(f"\nConverged: {result_slopes.converged}")
    log(f"AIC: {result_slopes.aic:.2f}")
    log(f"BIC: {result_slopes.bic:.2f}")
    log(f"Log-likelihood: {result_slopes.llf:.2f}")
    
    if result_slopes.converged and result_slopes.cov_re.shape[0] >= 2:
        intercept_var = result_slopes.cov_re.iloc[0,0]
        slope_var = result_slopes.cov_re.iloc[1,1]
        cov = result_slopes.cov_re.iloc[0,1] if result_slopes.cov_re.shape[0] > 1 else 0.0
        
        log(f"Random intercept variance: {intercept_var:.6f}")
        log(f"Random slope variance: {slope_var:.6f}")
        log(f"Random slope SD: {np.sqrt(slope_var):.6f}")
        log(f"Intercept-slope covariance: {cov:.6f}")
    else:
        log("WARNING: Slopes model did not converge or has singular covariance")
        slope_var = np.nan
    
    log("\n" + "=" * 80)
    log("MODEL COMPARISON")
    log("=" * 80)
    
    delta_aic = result_intercepts.aic - result_slopes.aic
    delta_bic = result_intercepts.bic - result_slopes.bic
    
    log(f"\nΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    log(f"ΔBIC (Intercepts - Slopes): {delta_bic:.2f}")
    
    # Interpretation
    log("\n" + "=" * 80)
    log("INTERPRETATION")
    log("=" * 80)
    
    if not result_slopes.converged:
        outcome = "Option B: Slopes don't converge"
        interpretation = "Insufficient data for stable slope estimation. Retain intercepts-only."
        action = "Keep intercepts-only model (homogeneous effects ASSUMED, not confirmed)"
    elif delta_aic > 2:
        outcome = "Option A: Slopes improve fit (ΔAIC > 2)"
        interpretation = f"Random slopes variance ({slope_var:.6f}) is substantial. AIC favors complexity."
        action = "Should use slopes model for downstream analyses (heterogeneous effects confirmed)"
    elif abs(delta_aic) <= 2:
        outcome = "Option C: Slopes converge but don't improve (|ΔAIC| ≤ 2)"
        interpretation = f"Random slopes variance negligible ({slope_var:.6f}). Simpler model preferred."
        action = "Keep intercepts-only model (homogeneous effects CONFIRMED via empirical test)"
    else:
        outcome = "Inconclusive"
        interpretation = "Unexpected result"
        action = "Manual review required"
    
    log(f"\n{outcome}")
    log(f"Interpretation: {interpretation}")
    log(f"Action: {action}")
    
    # Save comparison table
    comparison_df = pd.DataFrame({
        'model': ['Intercepts_Only', 'Intercepts_Slopes'],
        'converged': [result_intercepts.converged, result_slopes.converged],
        'aic': [result_intercepts.aic, result_slopes.aic],
        'bic': [result_intercepts.bic, result_slopes.bic],
        'loglik': [result_intercepts.llf, result_slopes.llf],
        'delta_aic': [0.0, delta_aic],
        'delta_bic': [0.0, delta_bic],
        'random_intercept_var': [
            result_intercepts.cov_re.iloc[0,0],
            intercept_var if result_slopes.converged else np.nan
        ],
        'random_slope_var': [
            0.0,
            slope_var if result_slopes.converged else np.nan
        ],
        'outcome': [outcome, outcome],
        'action': [action, action]
    })
    
    comparison_df.to_csv(OUTPUT_FILE, index=False)
    log(f"\nComparison table saved to {OUTPUT_FILE}")
    
    log("\n" + "=" * 80)
    log("COMPARISON COMPLETE")
    log("=" * 80)
    
    return comparison_df

if __name__ == "__main__":
    # Clear log file
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("")
    
    comparison = main()
    print("\nComparison results:")
    print(comparison[['model', 'aic', 'delta_aic', 'random_slope_var', 'outcome']])
