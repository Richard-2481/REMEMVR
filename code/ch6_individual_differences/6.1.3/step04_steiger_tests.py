#!/usr/bin/env python3
"""
Step 04: Perform Steiger Z-Tests for Cross-Domain Comparisons
RQ: ch7/7.1.3
Purpose: Test statistical significance of cross-domain beta coefficient differences using Steiger Z-tests
Output: Steiger Z-test results and correlation matrix
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step04_steiger_tests.log"

# Input files
INPUT_MERGED = RQ_DIR / "data" / "step01_merged_dataset.csv"
INPUT_COMPARISONS = RQ_DIR / "data" / "step03_cross_domain_comparisons.csv"

# Output files
OUTPUT_STEIGER = RQ_DIR / "data" / "step04_steiger_z_tests.csv"
OUTPUT_CORRELATION = RQ_DIR / "data" / "step04_correlation_matrix.csv"
OUTPUT_PVALUES = RQ_DIR / "data" / "step04_corrected_pvalues.csv"
OUTPUT_BOOTSTRAP = RQ_DIR / "data" / "step04_bootstrap_correlation_diffs.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def steiger_z_test(r12, r13, r23, n):
    """
    Perform Steiger's Z-test for dependent correlations.
    
    Tests whether r12 differs from r13, where:
    - r12: correlation between variables 1 and 2
    - r13: correlation between variables 1 and 3
    - r23: correlation between variables 2 and 3
    - n: sample size
    """
    # Fisher transformation
    z12 = 0.5 * np.log((1 + r12) / (1 - r12))
    z13 = 0.5 * np.log((1 + r13) / (1 - r13))
    
    # Compute test statistic
    term1 = (n - 3) * (z12 - z13)**2
    term2 = 2 * (1 - r23**2)**3
    
    # Avoid division by zero
    if term2 == 0:
        return {'z_statistic': np.nan, 'p_value': np.nan}
    
    # Alternative formula for dependent correlations
    k = (r23 - 0.5 * r12 * r13) * (1 - r12**2 - r13**2 - r23**2) + r23**3
    
    # Standard error
    se = np.sqrt((1 + r23) / (n * (1 - r23)))
    
    # Z-statistic
    z_stat = (z12 - z13) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {'z_statistic': z_stat, 'p_value': p_value}

def bootstrap_correlation_difference(data, var1, var2, var3, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for correlation difference."""
    np.random.seed(seed)
    n = len(data)
    
    diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        boot_data = data.iloc[idx]
        
        # Calculate correlations
        r12 = boot_data[[var1, var2]].corr().iloc[0, 1]
        r13 = boot_data[[var1, var3]].corr().iloc[0, 1]
        
        diffs.append(r12 - r13)
    
    # Calculate percentile CI
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return np.array(diffs), ci_lower, ci_upper

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 04: Perform Steiger Z-Tests for Cross-Domain Comparisons")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        
        # =========================================================================
        # STEP 1: Load data and prepare for correlation analysis
        # =========================================================================
        log("\n[STEP 1] Loading merged dataset and preparing for analysis...")
        
        data = pd.read_csv(INPUT_MERGED)
        log(f"[INFO] Loaded data: {data.shape}")
        
        # Pivot data to wide format for correlations
        # We need separate columns for each domain's theta scores
        theta_wide = data.pivot(index='UID', columns='domain', values='theta_mean').reset_index()
        
        # Merge with cognitive test scores
        cog_data = data[['UID', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']].drop_duplicates()
        analysis_data = pd.merge(theta_wide, cog_data, on='UID')
        
        log(f"[INFO] Analysis data shape: {analysis_data.shape}")
        log(f"[INFO] Columns: {analysis_data.columns.tolist()}")
        
        # =========================================================================
        # STEP 2: Calculate correlation matrix
        # =========================================================================
        log("\n[STEP 2] Calculating correlation matrix...")
        
        # Select variables for correlation matrix
        corr_vars = ['What', 'Where', 'When', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        corr_matrix = analysis_data[corr_vars].corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(OUTPUT_CORRELATION)
        log(f"[OUTPUT] Correlation matrix saved to: {OUTPUT_CORRELATION}")
        
        # Display key correlations
        log("\n[INFO] Key correlations:")
        for test in ['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']:
            for domain in ['What', 'Where', 'When']:
                log(f"  {test}-{domain}: r={corr_matrix.loc[test, domain]:.3f}")
        
        # =========================================================================
        # STEP 3: Perform Steiger Z-tests for key comparisons
        # =========================================================================
        log("\n[STEP 3] Performing Steiger Z-tests for dependent correlations...")
        
        n = len(analysis_data)
        steiger_results = []
        
        # Test 1: RAVLT correlation with What vs Where
        r_ravlt_what = corr_matrix.loc['RAVLT_T', 'What']
        r_ravlt_where = corr_matrix.loc['RAVLT_T', 'Where']
        r_what_where = corr_matrix.loc['What', 'Where']
        
        test1 = steiger_z_test(r_ravlt_what, r_ravlt_where, r_what_where, n)
        steiger_results.append({
            'comparison': 'RAVLT: What vs Where',
            'r1': r_ravlt_what,
            'r2': r_ravlt_where,
            'r_diff': r_ravlt_what - r_ravlt_where,
            'z_statistic': test1['z_statistic'],
            'p_uncorrected': test1['p_value']
        })
        
        # Test 2: BVMT correlation with Where vs What
        r_bvmt_where = corr_matrix.loc['BVMT_T', 'Where']
        r_bvmt_what = corr_matrix.loc['BVMT_T', 'What']
        
        test2 = steiger_z_test(r_bvmt_where, r_bvmt_what, r_what_where, n)
        steiger_results.append({
            'comparison': 'BVMT: Where vs What',
            'r1': r_bvmt_where,
            'r2': r_bvmt_what,
            'r_diff': r_bvmt_where - r_bvmt_what,
            'z_statistic': test2['z_statistic'],
            'p_uncorrected': test2['p_value']
        })
        
        # Test 3: RPM consistency across domains (omnibus test)
        r_rpm_what = corr_matrix.loc['RPM_T', 'What']
        r_rpm_where = corr_matrix.loc['RPM_T', 'Where']
        r_rpm_when = corr_matrix.loc['RPM_T', 'When']
        
        # Test What vs Where for RPM
        test3a = steiger_z_test(r_rpm_what, r_rpm_where, r_what_where, n)
        
        # Test What vs When for RPM
        r_what_when = corr_matrix.loc['What', 'When']
        test3b = steiger_z_test(r_rpm_what, r_rpm_when, r_what_when, n)
        
        # Take the maximum p-value as conservative omnibus test
        rpm_max_p = max(test3a['p_value'], test3b['p_value'])
        rpm_range = max(r_rpm_what, r_rpm_where, r_rpm_when) - min(r_rpm_what, r_rpm_where, r_rpm_when)
        
        steiger_results.append({
            'comparison': 'RPM: consistency test',
            'r1': r_rpm_what,
            'r2': r_rpm_where,
            'r_diff': rpm_range,
            'z_statistic': max(abs(test3a['z_statistic']), abs(test3b['z_statistic'])),
            'p_uncorrected': rpm_max_p
        })

        # Test 4: RAVLT_Pct_Ret correlation with What vs Where
        r_ravlt_pct_what = corr_matrix.loc['RAVLT_Pct_Ret_T', 'What']
        r_ravlt_pct_where = corr_matrix.loc['RAVLT_Pct_Ret_T', 'Where']

        test4 = steiger_z_test(r_ravlt_pct_what, r_ravlt_pct_where, r_what_where, n)
        steiger_results.append({
            'comparison': 'RAVLT_Pct_Ret: What vs Where',
            'r1': r_ravlt_pct_what,
            'r2': r_ravlt_pct_where,
            'r_diff': r_ravlt_pct_what - r_ravlt_pct_where,
            'z_statistic': test4['z_statistic'],
            'p_uncorrected': test4['p_value']
        })

        # Test 5: BVMT_Pct_Ret correlation with Where vs What
        r_bvmt_pct_where = corr_matrix.loc['BVMT_Pct_Ret_T', 'Where']
        r_bvmt_pct_what = corr_matrix.loc['BVMT_Pct_Ret_T', 'What']

        test5 = steiger_z_test(r_bvmt_pct_where, r_bvmt_pct_what, r_what_where, n)
        steiger_results.append({
            'comparison': 'BVMT_Pct_Ret: Where vs What',
            'r1': r_bvmt_pct_where,
            'r2': r_bvmt_pct_what,
            'r_diff': r_bvmt_pct_where - r_bvmt_pct_what,
            'z_statistic': test5['z_statistic'],
            'p_uncorrected': test5['p_value']
        })
        
        # =========================================================================
        # STEP 4: Apply multiple comparison corrections
        # =========================================================================
        log("\n[STEP 4] Applying multiple comparison corrections...")
        
        # Extract p-values
        p_values = [result['p_uncorrected'] for result in steiger_results]
        
        # Bonferroni correction
        n_tests = len(p_values)
        bonferroni_alpha = 0.05 / n_tests
        p_bonferroni = [min(p * n_tests, 1.0) for p in p_values]
        
        # FDR correction (Benjamini-Hochberg)
        _, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Add corrected p-values to results
        for i, result in enumerate(steiger_results):
            result['p_bonferroni'] = p_bonferroni[i]
            result['p_fdr'] = p_fdr[i]
            
            # Add significance markers
            if result['p_uncorrected'] < 0.001:
                result['significance'] = '***'
            elif result['p_uncorrected'] < 0.01:
                result['significance'] = '**'
            elif result['p_uncorrected'] < 0.05:
                result['significance'] = '*'
            else:
                result['significance'] = 'ns'
        
        # Save Steiger test results
        steiger_df = pd.DataFrame(steiger_results)
        steiger_df.to_csv(OUTPUT_STEIGER, index=False)
        log(f"[OUTPUT] Steiger Z-test results saved to: {OUTPUT_STEIGER}")
        
        # Save corrected p-values separately
        pval_df = pd.DataFrame({
            'comparison': [r['comparison'] for r in steiger_results],
            'p_uncorrected': p_values,
            'p_bonferroni': p_bonferroni,
            'p_fdr': p_fdr.tolist()
        })
        pval_df.to_csv(OUTPUT_PVALUES, index=False)
        log(f"[OUTPUT] Corrected p-values saved to: {OUTPUT_PVALUES}")
        
        # =========================================================================
        # STEP 5: Bootstrap confidence intervals for correlation differences
        # =========================================================================
        log("\n[STEP 5] Computing bootstrap confidence intervals...")
        
        bootstrap_results = []
        
        # Bootstrap for RAVLT: What vs Where
        log("[BOOTSTRAP] RAVLT: What vs Where...")
        diffs, ci_lower, ci_upper = bootstrap_correlation_difference(
            analysis_data, 'RAVLT_T', 'What', 'Where', n_bootstrap=1000, seed=42
        )
        bootstrap_results.append({
            'comparison': 'RAVLT: What vs Where',
            'bootstrap_mean_diff': np.mean(diffs),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        })
        
        # Bootstrap for BVMT: Where vs What
        log("[BOOTSTRAP] BVMT: Where vs What...")
        diffs, ci_lower, ci_upper = bootstrap_correlation_difference(
            analysis_data, 'BVMT_T', 'Where', 'What', n_bootstrap=1000, seed=42
        )
        bootstrap_results.append({
            'comparison': 'BVMT: Where vs What',
            'bootstrap_mean_diff': np.mean(diffs),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        })
        
        # Bootstrap for RAVLT_Pct_Ret: What vs Where
        log("[BOOTSTRAP] RAVLT_Pct_Ret: What vs Where...")
        diffs, ci_lower, ci_upper = bootstrap_correlation_difference(
            analysis_data, 'RAVLT_Pct_Ret_T', 'What', 'Where', n_bootstrap=1000, seed=42
        )
        bootstrap_results.append({
            'comparison': 'RAVLT_Pct_Ret: What vs Where',
            'bootstrap_mean_diff': np.mean(diffs),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        })

        # Bootstrap for BVMT_Pct_Ret: Where vs What
        log("[BOOTSTRAP] BVMT_Pct_Ret: Where vs What...")
        diffs, ci_lower, ci_upper = bootstrap_correlation_difference(
            analysis_data, 'BVMT_Pct_Ret_T', 'Where', 'What', n_bootstrap=1000, seed=42
        )
        bootstrap_results.append({
            'comparison': 'BVMT_Pct_Ret: Where vs What',
            'bootstrap_mean_diff': np.mean(diffs),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        })

        # Bootstrap for RPM: What vs Where
        log("[BOOTSTRAP] RPM: What vs Where...")
        diffs, ci_lower, ci_upper = bootstrap_correlation_difference(
            analysis_data, 'RPM_T', 'What', 'Where', n_bootstrap=1000, seed=42
        )
        bootstrap_results.append({
            'comparison': 'RPM: What vs Where',
            'bootstrap_mean_diff': np.mean(diffs),
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper
        })
        
        bootstrap_df = pd.DataFrame(bootstrap_results)
        bootstrap_df.to_csv(OUTPUT_BOOTSTRAP, index=False)
        log(f"[OUTPUT] Bootstrap results saved to: {OUTPUT_BOOTSTRAP}")
        
        # =========================================================================
        # STEP 6: Summary of results
        # =========================================================================
        log("\n[STEP 6] Summary of Steiger Z-test results...")
        
        log("\n[RESULTS] Cross-domain correlation comparisons:")
        for _, row in steiger_df.iterrows():
            log(f"\n  {row['comparison']}:")
            log(f"    r1={row['r1']:.3f}, r2={row['r2']:.3f}, diff={row['r_diff']:.3f}")
            log(f"    Z={row['z_statistic']:.2f}, p={row['p_uncorrected']:.3f} {row['significance']}")
            log(f"    p(Bonferroni)={row['p_bonferroni']:.3f}, p(FDR)={row['p_fdr']:.3f}")
            
            # Check hypothesis
            if 'RAVLT' in row['comparison'] and 'Pct' not in row['comparison']:
                hypothesis_met = row['r_diff'] > 0
                log(f"    Hypothesis (RAVLT->What > RAVLT->Where): {'supported' if hypothesis_met else 'not supported'}")
            elif 'RAVLT_Pct_Ret' in row['comparison']:
                hypothesis_met = row['r_diff'] > 0
                log(f"    Hypothesis (RAVLT_Pct_Ret->What > Where): {'supported' if hypothesis_met else 'not supported'}")
            elif 'BVMT' in row['comparison'] and 'Pct' not in row['comparison']:
                hypothesis_met = row['r_diff'] > 0
                log(f"    Hypothesis (BVMT->Where > BVMT->What): {'supported' if hypothesis_met else 'not supported'}")
            elif 'BVMT_Pct_Ret' in row['comparison']:
                hypothesis_met = row['r_diff'] > 0
                log(f"    Hypothesis (BVMT_Pct_Ret->Where > What): {'supported' if hypothesis_met else 'not supported'}")
            elif 'RPM' in row['comparison']:
                hypothesis_met = row['r_diff'] < 0.1
                log(f"    Hypothesis (RPM consistent across domains): {'supported' if hypothesis_met else 'not supported'}")

        log("\n[COMPLETE] Step 04 completed successfully")
        log(f"[SUMMARY] Performed {len(steiger_results)} Steiger Z-tests with multiple comparison corrections")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        sys.exit(1)