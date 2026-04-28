#!/usr/bin/env python3
"""
Step 09: Power Analysis for RQ 7.3.1
Compute post-hoc power and sensitivity analysis for effect detection
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step09_power_analysis.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def compute_power_for_f2(f2, n, k, alpha=0.05):
    """
    Compute post-hoc power for given Cohen's f².
    
    Parameters:
    - f2: Cohen's f² effect size
    - n: Sample size
    - k: Number of predictors
    - alpha: Significance level
    """
    # Convert f² to R²
    r2 = f2 / (1 + f2)
    
    # Numerator and denominator degrees of freedom
    df_num = k
    df_denom = n - k - 1
    
    # Non-centrality parameter
    lambda_nc = f2 * n
    
    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, df_num, df_denom)
    
    # Power calculation using non-central F distribution
    power = 1 - stats.ncf.cdf(f_crit, df_num, df_denom, lambda_nc)
    
    return power

def compute_minimum_detectable_f2(n, k, alpha=0.05, power=0.80):
    """
    Compute minimum detectable Cohen's f² at given power.
    
    Uses iterative search to find the f² that yields desired power.
    """
    # Binary search for minimum detectable effect size
    f2_low, f2_high = 0.001, 2.0
    tolerance = 0.0001
    
    while (f2_high - f2_low) > tolerance:
        f2_mid = (f2_low + f2_high) / 2
        power_mid = compute_power_for_f2(f2_mid, n, k, alpha)
        
        if power_mid < power:
            f2_low = f2_mid
        else:
            f2_high = f2_mid
    
    return f2_mid

try:
    log("Step 09: Power Analysis")
    log("Purpose: Post-hoc power and sensitivity analysis")
    
    # Load effect sizes
    log("Loading effect size results...")
    effect_df = pd.read_csv(RQ_DIR / "data" / "step08_effect_sizes.csv")
    log(f"Effect sizes: {len(effect_df)} effects")
    
    # Load sample information
    log("Loading analysis dataset for sample size...")
    df = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")
    n = len(df)
    k = 8  # Number of predictors (age, sex, education, RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T)
    
    log(f"Sample size: N={n}")
    log(f"Number of predictors: k={k}")
    log(f"Degrees of freedom: ({k}, {n-k-1})")
    
    # Power analysis parameters
    alpha_uncorrected = 0.05
    alpha_bonferroni = 0.00179 / 5  # 0.000358 for cognitive tests
    power_target = 0.80
    
    log(f"Alpha (uncorrected): {alpha_uncorrected:.4f}")
    log(f"Alpha (Bonferroni): {alpha_bonferroni:.6f}")
    log(f"Target power: {power_target:.2f}")
    
    # Power analysis results storage
    power_results = []
    
    # Overall model power
    log("Computing post-hoc power for observed effects...")
    overall_model = effect_df[effect_df['predictor'] == 'Overall_Model'].iloc[0]
    f2_overall = overall_model['cohens_f2']
    
    # Power for overall model (uses uncorrected alpha)
    power_overall = compute_power_for_f2(f2_overall, n, k, alpha_uncorrected)
    min_detectable_overall = compute_minimum_detectable_f2(n, k, alpha_uncorrected, power_target)
    
    power_results.append({
        'test': 'Overall_Model',
        'observed_f2': f2_overall,
        'power_observed': power_overall,
        'min_detectable_f2': min_detectable_overall,
        'power_80': power_target,
        'alpha_used': alpha_uncorrected
    })
    
    log(f"Overall model: f²={f2_overall:.4f}, power={power_overall:.3f}")
    
    # Individual cognitive tests (use Bonferroni corrected alpha)
    cognitive_tests = ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
    
    for test in cognitive_tests:
        test_data = effect_df[effect_df['predictor'] == test].iloc[0]
        f2_test = test_data['cohens_f2']
        sr2_test = test_data['sr2']
        
        # Power for individual test (uses Bonferroni alpha)
        # For individual predictors, k=1 (single predictor test)
        power_test = compute_power_for_f2(f2_test, n, 1, alpha_bonferroni)
        min_detectable_test = compute_minimum_detectable_f2(n, 1, alpha_bonferroni, power_target)
        
        power_results.append({
            'test': test,
            'observed_f2': f2_test,
            'power_observed': power_test,
            'min_detectable_f2': min_detectable_test,
            'power_80': power_target,
            'alpha_used': alpha_bonferroni
        })
        
        log(f"{test}: f²={f2_test:.4f}, power={power_test:.3f} "
            f"(alpha={alpha_bonferroni:.6f})")
    
    # Create power analysis DataFrame
    power_df = pd.DataFrame(power_results)
    
    # Sensitivity analysis
    log("Computing minimum detectable effects at 80% power...")
    
    for _, row in power_df.iterrows():
        if row['test'] == 'Overall_Model':
            log(f"Overall model: min f²={row['min_detectable_f2']:.4f} "
                f"at 80% power (alpha={row['alpha_used']:.4f})")
        else:
            log(f"{row['test']}: min f²={row['min_detectable_f2']:.4f} "
                f"at 80% power (alpha={row['alpha_used']:.6f})")
    
    # Check power adequacy
    log("Checking power adequacy...")
    adequate_power = []
    
    for _, row in power_df.iterrows():
        if row['power_observed'] >= 0.80:
            adequate_power.append(row['test'])
            log(f"{row['test']}: power={row['power_observed']:.3f} ≥ 0.80")
        else:
            log(f"{row['test']}: power={row['power_observed']:.3f} < 0.80")
    
    if len(adequate_power) == 0:
        log("All tests are underpowered (power < 0.80)")
        log("This limits ability to detect true null effects")
    
    # Sample size for adequate power
    log("[SAMPLE SIZE] Computing required N for 80% power...")
    
    # For overall model
    n_required_overall = int(min_detectable_overall / f2_overall * n) if f2_overall > 0 else np.inf
    log(f"[SAMPLE SIZE] Overall model: N≈{n_required_overall} needed for 80% power")
    
    # For individual tests (using most promising predictor)
    bvmt_data = effect_df[effect_df['predictor'] == 'BVMT_T'].iloc[0]
    f2_bvmt = bvmt_data['cohens_f2']
    if f2_bvmt > 0:
        # Rough approximation: scale based on effect size ratio
        n_required_bvmt = int(n * (min_detectable_test / f2_bvmt))
        log(f"[SAMPLE SIZE] BVMT_T: N≈{n_required_bvmt} needed for 80% power at α={alpha_bonferroni:.6f}")
    
    # Save results
    output_path = RQ_DIR / "data" / "step09_power_analysis.csv"
    power_df.to_csv(output_path, index=False)
    log(f"Power analysis: {output_path}")
    
    # Validation
    log("Checking power analysis validity...")
    all_power_valid = all((0 <= p <= 1) for p in power_df['power_observed'])
    all_f2_positive = all(f >= 0 for f in power_df['observed_f2'])
    alpha_correct = power_df[power_df['test'] != 'Overall_Model']['alpha_used'].iloc[0] == alpha_bonferroni
    
    if all_power_valid and all_f2_positive and alpha_correct:
        log("Power analysis PASSED all checks")
    else:
        log("Some checks failed:")
        if not all_power_valid:
            log("  - Power values outside [0,1] range")
        if not all_f2_positive:
            log("  - Negative f² values detected")
        if not alpha_correct:
            log("  - Bonferroni correction not properly applied")
    
    log("Step 09 complete")
    
except Exception as e:
    log(f"Critical error in power analysis: {str(e)}")
    import traceback
    log(f"{traceback.format_exc()}")
    raise
