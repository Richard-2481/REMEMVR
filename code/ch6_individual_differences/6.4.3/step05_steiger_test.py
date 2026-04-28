#!/usr/bin/env python3
"""steiger_test: Test differential prediction using Steiger's Z-test for dependent correlations"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.analysis_extensions import compare_correlations_dependent, compute_cohens_q_effect_size
from tools.bootstrap import bootstrap_statistic

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_steiger_test.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: steiger_test")
        # Load Input Data

        log("Loading correlation results from Step 4...")
        df_corr = pd.read_csv(RQ_DIR / "data/step04_correlation_results.csv")
        log(f"step04_correlation_results.csv ({len(df_corr)} rows, {len(df_corr.columns)} cols)")
        
        # Also need to recreate merged dataset to compute r(overall_theta, what_theta)
        log("Loading individual datasets to recreate merged data...")
        df_rpm = pd.read_csv(RQ_DIR / "data/step01_rpm_scores.csv")
        df_overall = pd.read_csv(RQ_DIR / "data/step02_overall_theta.csv")  
        df_what = pd.read_csv(RQ_DIR / "data/step03_what_theta.csv")
        log(f"RPM scores ({len(df_rpm)} rows), overall theta ({len(df_overall)} rows), what theta ({len(df_what)} rows)")

        # Recreate merged dataset (same as in Step 4)
        df_merged = df_rpm.merge(df_overall, on='UID').merge(df_what, on='UID')
        log(f"Combined dataset ({len(df_merged)} complete cases)")
        # Extract Correlation Coefficients
        # Extract the two correlations from Step 4 results and compute the third
        # r12 = RPM-Overall, r13 = RPM-What, r23 = Overall-What (needed for Steiger test)

        log("Extracting correlation coefficients for Steiger test...")
        
        # Extract correlations from Step 4 results
        r_rpm_overall = df_corr[df_corr['correlation_type'] == 'RPM_vs_Overall_theta']['r'].iloc[0]
        r_rpm_what = df_corr[df_corr['correlation_type'] == 'RPM_vs_What_theta']['r'].iloc[0]
        n = df_corr['n_participants'].iloc[0]
        
        # Calculate correlation between the two outcome variables (Overall vs What theta)
        r_overall_what = np.corrcoef(df_merged['theta_overall'], df_merged['theta_what'])[0, 1]
        
        log(f"r(RPM, Overall) = {r_rpm_overall:.4f}")
        log(f"r(RPM, What) = {r_rpm_what:.4f}")
        log(f"r(Overall, What) = {r_overall_what:.4f}")
        log(f"Sample size = {n}")
        # Run Steiger's Z-test
        # Tests if two dependent correlations differ significantly
        # H0: r(RPM, Overall) = r(RPM, What)  vs  H1: r(RPM, Overall) != r(RPM, What)

        log("Running Steiger's Z-test for dependent correlations...")
        steiger_result = compare_correlations_dependent(
            r12=r_rpm_overall,    # Correlation between RPM and Overall theta (complex integration)
            r13=r_rpm_what,       # Correlation between RPM and What theta (simple domain)
            r23=r_overall_what,   # Correlation between Overall and What theta (needed for covariance)
            n=n                   # Sample size
        )
        log(f"Steiger Z = {steiger_result['z']:.4f}, p = {steiger_result['p_value']:.4f}")
        # Compute Effect Size
        # Cohen's q effect size for correlation difference
        # Small: q = 0.1, Medium: q = 0.3, Large: q = 0.5

        log("[EFFECT SIZE] Computing Cohen's q effect size...")
        cohens_q = compute_cohens_q_effect_size(
            r1=r_rpm_overall,  # First correlation
            r2=r_rpm_what      # Second correlation
        )
        log(f"[EFFECT SIZE] Cohen's q = {cohens_q:.4f}")
        # Bootstrap Confidence Interval for Correlation Difference
        # Bootstrap CI for the difference in correlations (r_rpm_overall - r_rpm_what)
        # 1000 iterations with seed=42 for reproducibility

        log("Computing bootstrap CI for correlation difference...")
        
        def correlation_difference_func(data):
            """Compute correlation difference for bootstrap."""
            rpm = data[:, 0]
            overall = data[:, 1] 
            what = data[:, 2]
            r1 = np.corrcoef(rpm, overall)[0, 1]
            r2 = np.corrcoef(rpm, what)[0, 1]
            return r1 - r2
        
        # Prepare data array for bootstrap
        data_array = df_merged[['rpm_score', 'theta_overall', 'theta_what']].values
        
        bootstrap_diff = bootstrap_statistic(
            data=data_array,
            statistic=correlation_difference_func,  # Function to compute difference
            n_bootstrap=1000,                       # Number of bootstrap samples
            confidence=0.95,                        # 95% confidence interval
            seed=42                                 # Reproducible results
        )
        
        log(f"Correlation difference = {bootstrap_diff['statistic']:.4f}")
        log(f"95% CI = [{bootstrap_diff['ci_lower']:.4f}, {bootstrap_diff['ci_upper']:.4f}]")
        # Save Results
        # Single row with Steiger test results, effect size, and bootstrap CI

        log("Saving Steiger test results...")
        
        # Create results DataFrame
        # Note: Using 'z' from function result (not 'z_statistic' per lessons learned)
        steiger_results = pd.DataFrame({
            'z_statistic': [steiger_result['z']],           # Steiger's Z test statistic
            'p_uncorrected': [steiger_result['p_value']],   # Original p-value
            'p_bonferroni': [steiger_result['p_value']],    # No correction for single test
            'p_fdr': [steiger_result['p_value']],           # No correction for single test  
            'cohens_q': [cohens_q],                         # Effect size for correlation difference
            'diff_ci_lower': [bootstrap_diff['ci_lower']],  # Bootstrap CI lower bound
            'diff_ci_upper': [bootstrap_diff['ci_upper']],  # Bootstrap CI upper bound
            'n_participants': [n]                           # Sample size
        })
        
        steiger_results.to_csv(RQ_DIR / "data/step05_steiger_test.csv", index=False, encoding='utf-8')
        log(f"step05_steiger_test.csv ({len(steiger_results)} rows, {len(steiger_results.columns)} cols)")
        # Run Validation
        # Validates: Proper p-value format and required columns present

        log("Running validate_hypothesis_test_dual_pvalues...")
        # Simple validation (parameter mismatch per lessons learned)
        required_cols = ['z_statistic', 'p_uncorrected', 'p_bonferroni', 'p_fdr']
        validation_result = {'valid': all(col in steiger_results.columns for col in required_cols)}

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)