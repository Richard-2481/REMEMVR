#!/usr/bin/env python3
"""
Step 4: Steiger's Z-test for Dependent Correlations
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Test whether age-related decline differs significantly between RAVLT and REMEMVR
Uses Steiger's Z-test which accounts for the dependency (both correlations share Age variable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_steiger_test.log"

# Add project root to path for tools import
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def steiger_z_test(r12, r13, r23, n):
    """
    Steiger's Z-test for comparing two dependent correlations
    
    r12: correlation between Age and RAVLT
    r13: correlation between Age and REMEMVR 
    r23: correlation between RAVLT and REMEMVR
    n: sample size
    
    Tests H0: |r12| = |r13| vs H1: |r12| > |r13|
    """
    # Fisher's z-transformation
    z12 = 0.5 * math.log((1 + r12) / (1 - r12))
    z13 = 0.5 * math.log((1 + r13) / (1 - r13))
    
    # Calculate determinant and average correlation
    r_avg = (r12**2 + r13**2) / 2
    determinant = 1 + 2*r12*r13*r23 - r12**2 - r13**2 - r23**2
    
    # Steiger's Z statistic
    if determinant > 0:
        z = (z12 - z13) * math.sqrt((n-3) / (2 * (1 - r23**2)))
        # More precise formula accounting for dependency
        var_diff = (2 * (1 - r23**2)) / (n - 3)
        z_precise = (z12 - z13) / math.sqrt(var_diff)
    else:
        log(f"WARNING: Determinant = {determinant:.4f} (non-positive)")
        z_precise = np.nan
    
    return z_precise

def bootstrap_correlation_difference(df, n_bootstrap=1000, seed=42):
    """Bootstrap the difference in absolute correlations"""
    np.random.seed(seed)
    n = len(df)
    differences = []
    
    for _ in range(n_bootstrap):
        # Participant-level resampling
        indices = np.random.choice(n, n, replace=True)
        df_boot = df.iloc[indices]
        
        r_age_ravlt = df_boot['Age'].corr(df_boot['RAVLT_Total'])
        r_age_rememvr = df_boot['Age'].corr(df_boot['theta_all'])
        
        # Difference in absolute values
        diff = abs(r_age_ravlt) - abs(r_age_rememvr)
        differences.append(diff)
    
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return differences, ci_lower, ci_upper

def main():
    log("=" * 60)
    log("Step 4: Steiger's Z-test for Dependent Correlations")
    log("=" * 60)
    
    # Load correlation results and merged data
    corr_path = RQ_DIR / "data" / "step03_correlations.csv"
    extra_path = RQ_DIR / "data" / "step03_extra_correlations.csv"
    merged_path = RQ_DIR / "data" / "step03_merged_data.csv"
    
    df_corr = pd.read_csv(corr_path)
    df_extra = pd.read_csv(extra_path)
    df_merged = pd.read_csv(merged_path)
    
    # Extract correlations
    r_age_ravlt = df_corr[df_corr['variable_pair'] == 'Age_RAVLT']['r'].iloc[0]
    r_age_rememvr = df_corr[df_corr['variable_pair'] == 'Age_REMEMVR']['r'].iloc[0]
    r_ravlt_rememvr = df_extra['r_ravlt_rememvr'].iloc[0]
    n = len(df_merged)
    
    log(f"Input correlations:")
    log(f"  r(Age, RAVLT) = {r_age_ravlt:.3f}")
    log(f"  r(Age, REMEMVR) = {r_age_rememvr:.3f}")
    log(f"  r(RAVLT, REMEMVR) = {r_ravlt_rememvr:.3f}")
    log(f"  N = {n}")
    
    # Perform Steiger's Z-test
    log("\n" + "=" * 40)
    log("STEIGER'S Z-TEST")
    log("=" * 40)
    
    # Test for difference in ABSOLUTE correlations
    z_stat = steiger_z_test(r_age_ravlt, r_age_rememvr, r_ravlt_rememvr, n)
    
    # One-tailed test: H1: |r_age_ravlt| > |r_age_rememvr|
    if not np.isnan(z_stat):
        # Use absolute values for comparison
        if abs(r_age_ravlt) > abs(r_age_rememvr):
            # Z should be positive for one-tailed test
            z_for_test = abs(z_stat)
        else:
            z_for_test = -abs(z_stat)
        
        p_one_tailed = 1 - stats.norm.cdf(z_for_test)
        
        log(f"\nResults:")
        log(f"  Z statistic = {z_stat:.3f}")
        log(f"  p-value (one-tailed) = {p_one_tailed:.4f}")
        
        if p_one_tailed < 0.05:
            log(f"  → SIGNIFICANT: RAVLT shows significantly more age decline than REMEMVR")
            interpretation = "Significant difference - VR scaffolding supported"
        elif p_one_tailed < 0.10:
            log(f"  → MARGINAL: Trending toward significance (p < 0.10)")
            interpretation = "Marginal difference - weak VR scaffolding support"
        else:
            log(f"  → NON-SIGNIFICANT: No significant difference in age correlations")
            interpretation = "No significant difference"
    else:
        p_one_tailed = np.nan
        interpretation = "Test failed - correlation matrix issue"
    
    # Calculate effect sizes
    log("\n" + "=" * 40)
    log("EFFECT SIZE CALCULATIONS")
    log("=" * 40)
    
    # Raw difference in absolute correlations
    correlation_difference = abs(r_age_ravlt) - abs(r_age_rememvr)
    log(f"\nCorrelation difference (|r_RAVLT| - |r_REMEMVR|): {correlation_difference:.3f}")
    
    # Bootstrap CI for difference
    differences, ci_lower, ci_upper = bootstrap_correlation_difference(df_merged, n_bootstrap=1000, seed=42)
    log(f"Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Does CI exclude zero?
    if ci_lower > 0:
        log("  → CI excludes zero: Robust evidence for difference")
    elif ci_upper < 0:
        log("  → CI entirely negative: REMEMVR shows MORE decline (unexpected)")
    else:
        log("  → CI includes zero: Difference not robust")
    
    # Power analysis (post-hoc)
    if not np.isnan(z_stat):
        # Approximate power for observed effect
        effect_z = abs(z_stat) / math.sqrt(n)
        ncp = effect_z * math.sqrt(n)  # Non-centrality parameter
        z_crit = stats.norm.ppf(0.95)  # One-tailed critical value
        power_achieved = 1 - stats.norm.cdf(z_crit - ncp)
        log(f"\nPost-hoc power: {power_achieved:.3f}")
    else:
        power_achieved = np.nan
    
    # Minimum detectable effect at 80% power
    z_80 = stats.norm.ppf(0.80)
    z_crit = stats.norm.ppf(0.95)
    min_detectable_z = (z_crit + z_80) / math.sqrt(n)
    min_detectable_r = math.tanh(min_detectable_z)
    log(f"Minimum detectable correlation difference (80% power): {min_detectable_r:.3f}")
    
    # Effect size category
    if abs(correlation_difference) >= 0.30:
        effect_category = "Large"
    elif abs(correlation_difference) >= 0.15:
        effect_category = "Medium"
    elif abs(correlation_difference) >= 0.05:
        effect_category = "Small"
    else:
        effect_category = "Negligible"
    
    log(f"Effect size category: {effect_category}")
    
    # STEIGER'S Z-TEST FOR PERCENT RETENTION (if available)
    log("\n" + "=" * 40)
    log("STEIGER'S Z-TEST: RAVLT Pct Ret vs REMEMVR")
    log("=" * 40)

    if 'r_age_pctret' in df_extra.columns and not pd.isna(df_extra['r_age_pctret'].iloc[0]):
        r_age_pctret = df_extra['r_age_pctret'].iloc[0]
        r_pctret_rememvr = df_extra['r_pctret_rememvr'].iloc[0]
        n_pctret = int(df_extra['n_pctret'].iloc[0])

        log(f"  r(Age, RAVLT_Pct_Ret) = {r_age_pctret:.3f}")
        log(f"  r(Age, REMEMVR) = {r_age_rememvr:.3f}")
        log(f"  r(RAVLT_Pct_Ret, REMEMVR) = {r_pctret_rememvr:.3f}")
        log(f"  N = {n_pctret}")

        z_stat_pctret = steiger_z_test(r_age_pctret, r_age_rememvr, r_pctret_rememvr, n_pctret)

        if not np.isnan(z_stat_pctret):
            if abs(r_age_pctret) > abs(r_age_rememvr):
                z_for_test_pctret = abs(z_stat_pctret)
            else:
                z_for_test_pctret = -abs(z_stat_pctret)
            p_pctret_one_tailed = 1 - stats.norm.cdf(z_for_test_pctret)

            log(f"\n  Z statistic = {z_stat_pctret:.3f}")
            log(f"  p-value (one-tailed) = {p_pctret_one_tailed:.4f}")

            if p_pctret_one_tailed < 0.05:
                pctret_interp = "Significant - Pct Ret shows more age decline than REMEMVR"
            else:
                pctret_interp = "Non-significant difference"
            log(f"  Interpretation: {pctret_interp}")
        else:
            p_pctret_one_tailed = np.nan
            pctret_interp = "Test failed"
            log("  Steiger's test failed for Pct Ret comparison")

        # Append to steiger results
        steiger_pctret = pd.DataFrame([{
            'z_statistic': z_stat_pctret,
            'p_value_one_tailed': p_pctret_one_tailed,
            'r_pctret': r_age_pctret,
            'r_rememvr': r_age_rememvr,
            'r_correlation': r_pctret_rememvr,
            'n_participants': n_pctret,
            'interpretation': pctret_interp
        }])
    else:
        log("  RAVLT Pct Ret data not available in extra correlations")
        steiger_pctret = None

    # VR SCAFFOLDING HYPOTHESIS EVALUATION
    log("\n" + "=" * 40)
    log("VR SCAFFOLDING HYPOTHESIS EVALUATION")
    log("=" * 40)
    
    scaffolding_criteria = {
        'ravlt_decline': r_age_ravlt < -0.20,
        'rememvr_invariant': abs(r_age_rememvr) < 0.20,
        'difference_exists': abs(r_age_ravlt) > abs(r_age_rememvr),
        'significant_test': p_one_tailed < 0.05 if not np.isnan(p_one_tailed) else False,
        'ci_excludes_zero': ci_lower > 0
    }
    
    log("Criteria met:")
    for criterion, met in scaffolding_criteria.items():
        log(f"  {criterion}: {'✓' if met else '✗'}")
    
    criteria_met = sum(scaffolding_criteria.values())
    if criteria_met >= 4:
        scaffolding_support = "STRONG"
    elif criteria_met >= 3:
        scaffolding_support = "MODERATE"
    elif criteria_met >= 2:
        scaffolding_support = "WEAK"
    else:
        scaffolding_support = "MINIMAL"
    
    log(f"\nOverall VR Scaffolding Support: {scaffolding_support}")
    
    # Save results
    steiger_results = pd.DataFrame([{
        'z_statistic': z_stat,
        'p_value_one_tailed': p_one_tailed,
        'r_ravlt': r_age_ravlt,
        'r_rememvr': r_age_rememvr,
        'r_correlation': r_ravlt_rememvr,
        'n_participants': n,
        'interpretation': interpretation
    }])
    
    effect_results = pd.DataFrame([{
        'correlation_difference': correlation_difference,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'power_achieved': power_achieved,
        'minimum_detectable': min_detectable_r,
        'effect_size_category': effect_category
    }])
    
    steiger_path = RQ_DIR / "data" / "step04_steiger_test.csv"
    effect_path = RQ_DIR / "data" / "step04_effect_sizes.csv"
    
    # If pctret Steiger test was run, append to main results
    if steiger_pctret is not None:
        steiger_results = pd.concat([steiger_results, steiger_pctret], ignore_index=True)
        steiger_results.insert(0, 'comparison', ['Age_RAVLT_vs_REMEMVR', 'Age_PctRet_vs_REMEMVR'])
    else:
        steiger_results.insert(0, 'comparison', ['Age_RAVLT_vs_REMEMVR'])

    steiger_results.to_csv(steiger_path, index=False)
    effect_results.to_csv(effect_path, index=False)

    log(f"\nSaved Steiger test results to: {steiger_path}")
    log(f"Saved effect sizes to: {effect_path}")
    
    log("\nStep 4 completed successfully")

if __name__ == "__main__":
    main()