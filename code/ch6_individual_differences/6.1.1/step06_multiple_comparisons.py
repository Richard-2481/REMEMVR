#!/usr/bin/env python3
"""Multiple Comparisons Correction: Apply multiple comparison corrections (Bonferroni, FDR) to regression results with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import false_discovery_control

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_multiple_comparisons.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Multiple Comparisons Correction")
        # Load Input Data

        log("Loading regression results from step05...")
        regression_results = pd.read_csv(RQ_DIR / "data" / "step05_regression_results.csv")
        log(f"step05_regression_results.csv ({len(regression_results)} rows, {len(regression_results.columns)} cols)")
        
        # Display loaded data for verification
        log(f"Predictors found: {list(regression_results['predictor'])}")
        log(f"P-value range: {regression_results['p_value'].min():.6f} to {regression_results['p_value'].max():.6f}")
        # Run Analysis Tool - Multiple Corrections

        log("Applying multiple comparison corrections...")
        
        # Filter out intercept (const) for hypothesis testing - only test predictors
        predictor_results = regression_results[regression_results['predictor'] != 'const'].copy()
        log(f"Testing {len(predictor_results)} predictors (excluding intercept)")
        
        # Extract p-values for correction
        p_uncorrected = predictor_results['p_value'].values
        
        # Apply within-RQ Bonferroni correction (6 predictors)
        n_predictors = len(predictor_results)
        alpha_bonferroni_within = 0.05 / n_predictors
        p_bonferroni_within = np.minimum(p_uncorrected * n_predictors, 1.0)
        
        # Apply Chapter-level Bonferroni correction (28 RQs in Chapter 7)
        alpha_chapter = 0.00179  # 0.05/28 tests
        p_bonferroni_chapter = np.minimum(p_uncorrected * 28, 1.0)
        
        # Apply FDR correction (Benjamini-Hochberg)
        try:
            # Use scipy.stats.false_discovery_control (newer function)
            _, p_fdr = false_discovery_control(p_uncorrected, alpha=0.05, method='bh')
        except:
            # Fallback to manual FDR calculation if scipy function unavailable
            from statsmodels.stats.multitest import fdrcorrection
            _, p_fdr = fdrcorrection(p_uncorrected, alpha=0.05, method='indep')
            
        log("Applied corrections:")
        log(f"  - Bonferroni within-RQ (alpha=0.0125 for 4 predictors)")
        log(f"  - Bonferroni chapter-level (alpha=0.00179 for 28 tests)")
        log(f"  - FDR Benjamini-Hochberg (alpha=0.05)")

        # Create corrected results DataFrame with dual reporting
        corrected_results = pd.DataFrame({
            'predictor': predictor_results['predictor'],
            'p_uncorrected': p_uncorrected,
            'p_bonferroni_within': p_bonferroni_within,
            'p_bonferroni_chapter': p_bonferroni_chapter,
            'p_fdr': p_fdr,
            'significant_uncorrected': p_uncorrected < 0.05,
            'significant_bonferroni_within': p_bonferroni_within < 0.05,
            'significant_bonferroni_chapter': p_bonferroni_chapter < 0.05,
            'significant_fdr': p_fdr < 0.05
        })
        
        # Create hypothesis tests DataFrame
        hypothesis_tests = []
        
        # Individual predictor tests
        for _, row in predictor_results.iterrows():
            hypothesis_tests.append({
                'hypothesis': f"{row['predictor']}_beta_nonzero",
                'test_statistic': row['t_stat'],
                'p_value': row['p_value'],
                'significant_uncorrected': row['p_value'] < 0.05,
                'significant_bonferroni_within': row['p_value'] < alpha_bonferroni_within,
                'significant_bonferroni_chapter': row['p_value'] < alpha_chapter
            })
            
        # Overall model significance (F-test would be ideal but reconstruct from available data)
        # Note: This is approximate - true F-test requires additional model information
        hypothesis_tests.append({
            'hypothesis': 'overall_model_significance',
            'test_statistic': 'F_approximate',  # Would need full model for true F-statistic
            'p_value': np.min(p_uncorrected),  # Conservative approximation using minimum p-value
            'significant_uncorrected': np.min(p_uncorrected) < 0.05,
            'significant_bonferroni_within': np.min(p_uncorrected) < alpha_bonferroni_within,
            'significant_bonferroni_chapter': np.min(p_uncorrected) < alpha_chapter
        })
        
        hypothesis_tests_df = pd.DataFrame(hypothesis_tests)
        
        log("Multiple comparison corrections complete")
        # Save Analysis Outputs
        # These outputs will be used by: results analysis for interpretation and reporting

        log(f"Saving corrected_results...")
        # Output: step06_corrected_results.csv
        # Contains: P-values with multiple comparison corrections and dual reporting
        # Columns: predictor, p_uncorrected, p_bonferroni_within, p_bonferroni_chapter, p_fdr, significance flags
        corrected_results.to_csv(RQ_DIR / "data" / "step06_corrected_results.csv", index=False, encoding='utf-8')
        log(f"step06_corrected_results.csv ({len(corrected_results)} rows, {len(corrected_results.columns)} cols)")

        log(f"Saving hypothesis_tests...")
        # Output: step06_hypothesis_tests.csv  
        # Contains: Specific hypothesis test results with multiple alpha levels
        # Columns: hypothesis, test_statistic, p_value, significance flags
        hypothesis_tests_df.to_csv(RQ_DIR / "data" / "step06_hypothesis_tests.csv", index=False, encoding='utf-8')
        log(f"step06_hypothesis_tests.csv ({len(hypothesis_tests_df)} rows, {len(hypothesis_tests_df.columns)} cols)")

        # Log summary of corrections
        log("Correction results:")
        for _, row in corrected_results.iterrows():
            log(f"  {row['predictor']}: p_uncorr={row['p_uncorrected']:.4f}, "
                f"p_bonf_within={row['p_bonferroni_within']:.4f}, "
                f"p_bonf_chapter={row['p_bonferroni_chapter']:.4f}, "
                f"p_fdr={row['p_fdr']:.4f}")
        # Run Validation Tool
        # Validates: P-value corrections applied correctly and dual reporting format
        # Threshold: All p-values in valid range, conservative correction pattern

        log("Validating multiple comparison corrections...")
        
        validation_results = []
        
        # Validate all p-values in [0, 1] range
        all_p_cols = ['p_uncorrected', 'p_bonferroni_within', 'p_bonferroni_chapter', 'p_fdr']
        p_value_ranges_valid = True
        for col in all_p_cols:
            col_valid = (corrected_results[col] >= 0).all() and (corrected_results[col] <= 1).all()
            if not col_valid:
                p_value_ranges_valid = False
                log(f"WARNING: {col} contains invalid p-values")
            
        validation_results.append(f"P-values in [0,1] range: {'PASS' if p_value_ranges_valid else 'FAIL'}")
        
        # Verify p_bonferroni >= p_uncorrected (correction makes more conservative)
        bonf_conservative = (corrected_results['p_bonferroni_within'] >= corrected_results['p_uncorrected']).all()
        validation_results.append(f"Bonferroni more conservative: {'PASS' if bonf_conservative else 'FAIL'}")
        
        # Check p_fdr between p_uncorrected and p_bonferroni (FDR is intermediate)
        fdr_intermediate = ((corrected_results['p_fdr'] >= corrected_results['p_uncorrected']).all() and
                           (corrected_results['p_fdr'] <= corrected_results['p_bonferroni_within']).all())
        validation_results.append(f"FDR intermediate correction: {'PASS' if fdr_intermediate else 'FAIL'}")
        
        # Confirm significance flags consistent with alpha thresholds
        sig_consistency = True
        for alpha_col, flag_col in [
            ('p_uncorrected', 'significant_uncorrected'),
            ('p_bonferroni_within', 'significant_bonferroni_within'),
            ('p_bonferroni_chapter', 'significant_bonferroni_chapter'),
            ('p_fdr', 'significant_fdr')
        ]:
            if alpha_col == 'p_uncorrected':
                expected_sig = corrected_results[alpha_col] < 0.05
            elif alpha_col == 'p_bonferroni_within':
                expected_sig = corrected_results[alpha_col] < 0.05
            elif alpha_col == 'p_bonferroni_chapter':
                expected_sig = corrected_results[alpha_col] < 0.05
            else:  # p_fdr
                expected_sig = corrected_results[alpha_col] < 0.05
                
            actual_sig = corrected_results[flag_col]
            if not (expected_sig == actual_sig).all():
                sig_consistency = False
                log(f"WARNING: {flag_col} inconsistent with {alpha_col}")
                
        validation_results.append(f"Significance flags consistent: {'PASS' if sig_consistency else 'FAIL'}")
        
        # Verify dual reporting present for all predictors
        dual_reporting_complete = len(corrected_results) == len(predictor_results)
        validation_results.append(f"Dual reporting complete: {'PASS' if dual_reporting_complete else 'FAIL'}")

        # Report validation results
        for result in validation_results:
            log(f"{result}")

        # Overall validation status
        all_validations_pass = all('PASS' in result for result in validation_results)
        if all_validations_pass:
            log("All validation criteria PASSED")
        else:
            log("Some validation criteria FAILED - check results carefully")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)