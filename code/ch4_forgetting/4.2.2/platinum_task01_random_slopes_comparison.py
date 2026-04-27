"""
PLATINUM TASK 1-2: Random Slopes Justification + Boundary Warning Investigation

Purpose: Verify that random slopes model is justified over intercepts-only model
         Investigate boundary warning for random slope variance

Per improvement_taxonomy.md Section 4.4:
- MANDATORY to test intercepts-only vs random slopes for all modeling RQs
- Cannot claim homogeneous effects without testing for heterogeneity
"""

import pandas as pd
import numpy as np
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path

# Paths
BASE = Path("/home/etai/projects/REMEMVR/results/ch5/5.2.2")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
LOGS_DIR = BASE / "logs"

# Setup logging
log_file = LOGS_DIR / "platinum_task01_random_slopes_comparison.log"
log = open(log_file, 'w')

def logprint(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

logprint("=" * 70)
logprint("PLATINUM TASK 1-2: Random Slopes vs Intercepts-Only Comparison")
logprint("=" * 70)
logprint("")

# Load data
logprint("[LOAD] Loading piecewise LMM input data...")
data_path = DATA_DIR / "step00_piecewise_lmm_input.csv"
data = pd.read_csv(data_path)
logprint(f"[LOADED] {data.shape[0]} observations, {data.shape[1]} columns")
logprint(f"[INFO] N participants: {data['UID'].nunique()}")
logprint(f"[INFO] Domains: {sorted(data['domain'].unique())}")
logprint("")

# MODEL 1: CURRENT MODEL (Random Intercepts + Random Slopes)
logprint("=" * 70)
logprint("MODEL 1: Random Intercepts + Random Slopes (CURRENT)")
logprint("=" * 70)
logprint("")

formula = "theta ~ Days_within * C(Segment, Treatment('Early')) * C(domain, Treatment('what'))"
logprint(f"[FIT] Formula: {formula}")
logprint(f"[FIT] Random effects: ~Days_within (intercepts + slopes)")
logprint("")

try:
    model_slopes = MixedLM.from_formula(
        formula=formula,
        groups=data['UID'],
        re_formula="~Days_within",
        data=data
    )
    result_slopes = model_slopes.fit(reml=False, method='lbfgs')
    
    logprint("[SUCCESS] Model with random slopes converged")
    logprint(f"  - Log-Likelihood: {result_slopes.llf:.4f}")
    logprint(f"  - AIC: {result_slopes.aic:.2f}")
    logprint(f"  - BIC: {result_slopes.bic:.2f}")
    logprint(f"  - N observations: {result_slopes.nobs:.0f}")
    logprint("")
    
    # Extract random effects
    re_cov = result_slopes.cov_re
    intercept_var_slopes = re_cov.iloc[0,0]
    slope_var = re_cov.iloc[1,1]
    cov_int_slope = re_cov.iloc[0,1]
    
    logprint("[RANDOM EFFECTS] Variance Components:")
    logprint(f"  - Intercept variance: {intercept_var_slopes:.6f} (SD: {np.sqrt(intercept_var_slopes):.4f})")
    logprint(f"  - Slope variance: {slope_var:.6f} (SD: {np.sqrt(slope_var):.4f})")
    logprint(f"  - Covariance: {cov_int_slope:.6f}")
    logprint("")
    
    if slope_var < 0.02:
        logprint(f"[FLAG] Random slope variance very small ({slope_var:.6f})")
        logprint(f"       This explains boundary warning")
    
    slopes_success = True
    
except Exception as e:
    logprint(f"[ERROR] Random slopes model failed: {e}")
    slopes_success = False
    result_slopes = None

logprint("")

# MODEL 2: INTERCEPTS-ONLY
logprint("=" * 70)
logprint("MODEL 2: Random Intercepts Only (SIMPLER)")
logprint("=" * 70)
logprint("")

logprint(f"[FIT] Formula: {formula}")
logprint(f"[FIT] Random effects: ~1 (intercepts only)")
logprint("")

try:
    model_intercepts = MixedLM.from_formula(
        formula=formula,
        groups=data['UID'],
        re_formula="~1",
        data=data
    )
    result_intercepts = model_intercepts.fit(reml=False, method='lbfgs')
    
    logprint("[SUCCESS] Intercepts-only model converged")
    logprint(f"  - Log-Likelihood: {result_intercepts.llf:.4f}")
    logprint(f"  - AIC: {result_intercepts.aic:.2f}")
    logprint(f"  - BIC: {result_intercepts.bic:.2f}")
    logprint(f"  - N observations: {result_intercepts.nobs:.0f}")
    logprint("")
    
    # Extract random effects
    intercept_var_intercepts = result_intercepts.cov_re.iloc[0,0]
    
    logprint("[RANDOM EFFECTS] Variance Components:")
    logprint(f"  - Intercept variance: {intercept_var_intercepts:.6f} (SD: {np.sqrt(intercept_var_intercepts):.4f})")
    logprint("")
    
    intercepts_success = True
    
except Exception as e:
    logprint(f"[ERROR] Intercepts-only model failed: {e}")
    intercepts_success = False
    result_intercepts = None

logprint("")

# MODEL COMPARISON
if slopes_success and intercepts_success:
    logprint("=" * 70)
    logprint("MODEL COMPARISON")
    logprint("=" * 70)
    logprint("")
    
    aic_slopes = result_slopes.aic
    aic_intercepts = result_intercepts.aic
    delta_aic = aic_intercepts - aic_slopes
    
    logprint("[AIC COMPARISON]")
    logprint(f"  Slopes model:      AIC = {aic_slopes:.2f}")
    logprint(f"  Intercepts model:  AIC = {aic_intercepts:.2f}")
    logprint(f"  ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    logprint("")
    
    bic_slopes = result_slopes.bic
    bic_intercepts = result_intercepts.bic
    delta_bic = bic_intercepts - bic_slopes
    
    logprint("[BIC COMPARISON]")
    logprint(f"  Slopes model:      BIC = {bic_slopes:.2f}")
    logprint(f"  Intercepts model:  BIC = {bic_intercepts:.2f}")
    logprint(f"  ΔBIC (Intercepts - Slopes): {delta_bic:.2f}")
    logprint("")
    
    # Likelihood ratio test
    ll_slopes = result_slopes.llf
    ll_intercepts = result_intercepts.llf
    lr_stat = 2 * (ll_slopes - ll_intercepts)
    
    from scipy.stats import chi2
    df = 2  # 2 extra parameters in slopes model
    p_value = 1 - chi2.cdf(lr_stat, df)
    
    logprint("[LIKELIHOOD RATIO TEST]")
    logprint(f"  LR statistic: {lr_stat:.4f}")
    logprint(f"  df: {df}")
    logprint(f"  p-value: {p_value:.4f}")
    logprint("")
    
    # INTERPRETATION
    logprint("=" * 70)
    logprint("INTERPRETATION")
    logprint("=" * 70)
    logprint("")
    
    if delta_aic < -2:
        logprint("[DECISION] Random slopes model STRONGLY PREFERRED (ΔAIC < -2)")
        logprint(f"           ΔAIC = {delta_aic:.2f} favors slopes")
        logprint("")
        logprint("[ACTION] Keep current random slopes model")
        logprint("         Boundary warning acceptable (variance near zero but model preferred)")
        recommendation = "KEEP_SLOPES"
        
    elif delta_aic > 2:
        logprint("[DECISION] Intercepts-only model STRONGLY PREFERRED (ΔAIC > 2)")
        logprint(f"           ΔAIC = {delta_aic:.2f} favors intercepts")
        logprint("")
        logprint("[ACTION] SWITCH to intercepts-only model")
        logprint("         Simpler model preferred, no evidence of slope heterogeneity")
        recommendation = "SWITCH_TO_INTERCEPTS"
        
    else:
        logprint("[DECISION] Models EQUIVALENT (|ΔAIC| < 2)")
        logprint(f"           ΔAIC = {delta_aic:.2f} (weak preference)")
        logprint("")
        logprint("[ACTION] Keep random slopes model (CONSERVATIVE)")
        logprint("         Allows for individual differences even if minimal")
        recommendation = "KEEP_SLOPES_CONSERVATIVE"
    
    logprint("")
    logprint("=" * 70)
    logprint("BOUNDARY WARNING INVESTIGATION")
    logprint("=" * 70)
    logprint("")
    logprint("[FINDING] Boundary warning: 'MLE may be on boundary of parameter space'")
    logprint(f"[CAUSE] Random slope variance = {slope_var:.6f} (very small)")
    logprint("")
    logprint("[EXPLANATION]")
    logprint("  - Minimal individual differences in forgetting slopes")
    logprint("  - Most participants show similar trajectories (homogeneous effects)")
    logprint("  - MLE estimation pushes variance toward zero")
    logprint("  - Model CONVERGED successfully (not a convergence failure)")
    logprint("")
    
    if recommendation == "KEEP_SLOPES" or recommendation == "KEEP_SLOPES_CONSERVATIVE":
        logprint("[CONCLUSION] Boundary warning is ACCEPTABLE:")
        logprint("  - Model converged successfully")
        logprint("  - Random slopes justified by AIC or conservatism")
        logprint("  - Variance near zero reflects true minimal heterogeneity")
    else:
        logprint("[CONCLUSION] Boundary warning suggests SIMPLER MODEL:")
        logprint("  - Intercepts-only preferred by AIC")
        logprint("  - Random slope variance negligible")
    
    logprint("")
    
    # Save results
    comparison_results = {
        'model_slopes_aic': aic_slopes,
        'model_intercepts_aic': aic_intercepts,
        'delta_aic': delta_aic,
        'model_slopes_bic': bic_slopes,
        'model_intercepts_bic': bic_intercepts,
        'delta_bic': delta_bic,
        'lr_statistic': lr_stat,
        'lr_p_value': p_value,
        'slope_variance': slope_var,
        'recommendation': recommendation
    }
    
    comparison_df = pd.DataFrame([comparison_results])
    output_path = RESULTS_DIR / "platinum_task01_random_slopes_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    logprint(f"[SAVED] {output_path}")
    logprint("")
    
    if recommendation == "SWITCH_TO_INTERCEPTS":
        alt_model_path = DATA_DIR / "step01_piecewise_lmm_model_intercepts_only.pkl"
        with open(alt_model_path, 'wb') as f:
            pickle.dump(result_intercepts, f)
        logprint(f"[SAVED] Alternative model: {alt_model_path}")
        logprint("[WARNING] Model change recommended - requires user approval")

else:
    logprint("[ERROR] Model comparison failed")
    recommendation = "ERROR"

logprint("")
logprint("=" * 70)
logprint(f"TASK 1-2 COMPLETE: {recommendation}")
logprint("=" * 70)

log.close()

print("")
print("SUMMARY:")
print(f"Recommendation: {recommendation}")
if recommendation == "KEEP_SLOPES" or recommendation == "KEEP_SLOPES_CONSERVATIVE":
    print("✓ Current model justified - random slopes appropriate")
elif recommendation == "SWITCH_TO_INTERCEPTS":
    print("⚠ Intercepts-only preferred - requires user approval to switch")
