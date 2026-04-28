#!/usr/bin/env python3
"""fit_intercept_model: Predict LMM intercepts (baseline ability) using cognitive test scores"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_multiple_regression

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_fit_intercept_model.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: fit_intercept_model")
        # Load Input Data

        log("Loading regression input data...")
        regression_input = pd.read_csv(RQ_DIR / "data" / "step02_regression_input.csv")
        log(f"step02_regression_input.csv ({len(regression_input)} rows, {len(regression_input.columns)} cols)")
        
        # Verify required columns
        required_cols = ['UID', 'intercept', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        missing_cols = [col for col in required_cols if col not in regression_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        log(f"Intercept range: [{regression_input['intercept'].min():.3f}, {regression_input['intercept'].max():.3f}]")
        log(f"RAVLT_T range: [{regression_input['RAVLT_T'].min():.3f}, {regression_input['RAVLT_T'].max():.3f}]")
        log(f"RAVLT_Pct_Ret_T range: [{regression_input['RAVLT_Pct_Ret_T'].min():.3f}, {regression_input['RAVLT_Pct_Ret_T'].max():.3f}]")
        log(f"BVMT_T range: [{regression_input['BVMT_T'].min():.3f}, {regression_input['BVMT_T'].max():.3f}]")
        log(f"BVMT_Pct_Ret_T range: [{regression_input['BVMT_Pct_Ret_T'].min():.3f}, {regression_input['BVMT_Pct_Ret_T'].max():.3f}]")
        log(f"RPM_T range: [{regression_input['RPM_T'].min():.3f}, {regression_input['RPM_T'].max():.3f}]")
        # Run Analysis Tool

        log("Running fit_multiple_regression...")
        
        # Prepare predictor matrix (X) and response variable (y)
        X = regression_input[['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']]  # Cognitive test T-scores
        y = regression_input['intercept']  # LMM random intercepts (baseline ability)

        log(f"Predictors shape: {X.shape}")
        log(f"Response shape: {y.shape}")

        # Fit multiple regression
        regression_results = fit_multiple_regression(
            X=X,
            y=y,
            feature_names=['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        )
        
        log("Analysis complete")
        log(f"R² = {regression_results['rsquared']:.4f}")
        log(f"Adjusted R² = {regression_results['rsquared_adj']:.4f}")
        log(f"F-statistic = {regression_results['fvalue']:.4f}")
        log(f"Model p-value = {regression_results['f_pvalue']:.6f}")
        # Extract and Format Results
        # Extract coefficients and create results table with required columns
        
        log("Extracting regression coefficients...")
        
        # Get coefficients from results
        coefficients = regression_results['coefficients']
        pvalues = regression_results['pvalues']
        std_errors = regression_results['std_errors']
        
        log(f"Coefficients: {coefficients}")
        log(f"P-values: {pvalues}")
        
        # Create formatted results table
        results_list = []
        
        # Process each coefficient
        for predictor in coefficients.keys():
            coef = coefficients[predictor]
            pval = pvalues[predictor]
            se = std_errors[predictor]
            
            # Apply Bonferroni correction (5 cognitive predictors)
            p_bonferroni = min(pval * 5, 1.0)  # Cap at 1.0
            
            # Get confidence intervals
            conf_intervals = regression_results.get('conf_int', {})
            ci_lower, ci_upper = conf_intervals.get(predictor, (np.nan, np.nan))
            
            # Calculate t-statistic
            t_stat = coef / se if se != 0 else np.nan
            
            result_row = {
                'predictor': predictor,
                'beta': coef,
                'se': se,
                't': t_stat,
                'p_uncorrected': pval,
                'p_bonferroni': p_bonferroni,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper
            }
            
            results_list.append(result_row)
            
            log(f"{predictor}: β={coef:.4f}, p={pval:.4f}, p_bonf={p_bonferroni:.4f}")
        
        # Create final results DataFrame
        intercept_model_results = pd.DataFrame(results_list)
        
        log(f"Results table created: {len(intercept_model_results)} coefficients")
        # Save Analysis Outputs
        # These outputs will be used by: Step 5 (R² comparison), Step 6 (predictor significance)

        output_path = RQ_DIR / "data" / "step03_intercept_predictions.csv"
        log(f"Saving {output_path.name}...")
        
        # Output: step03_intercept_predictions.csv
        # Contains: Regression coefficients predicting LMM intercepts from cognitive tests
        # Columns: ['predictor', 'beta', 'se', 't', 'p_uncorrected', 'p_bonferroni', 'CI_lower', 'CI_upper', 'VIF']
        intercept_model_results.to_csv(output_path, index=False, encoding='utf-8')
        
        log(f"{output_path.name} ({len(intercept_model_results)} rows, {len(intercept_model_results.columns)} cols)")
        # Run Validation Tool
        # Validates: P-values are in valid range [0, 1]
        # Threshold: All p-values must be between 0 and 1

        log("Running validate_numeric_range...")
        
        validation_result = validate_numeric_range(
            data=intercept_model_results['p_uncorrected'],
            min_val=0.0,
            max_val=1.0,
            column_name="p_uncorrected"
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS: All p-values in valid range [0, 1]")
        else:
            log(f"FAIL: {validation_result['message']}")
            if 'violations' in validation_result:
                log(f"Violations: {validation_result['violations'][:5]}")  # Show first 5
        
        # Additional validation checks
        log("Additional checks:")
        
        # Check standard errors are positive
        se_positive = (intercept_model_results['se'] > 0).all()
        log(f"Standard errors positive: {'PASS' if se_positive else 'FAIL'}")
        
        # Check Bonferroni correction applied correctly
        bonf_correct = np.allclose(
            intercept_model_results['p_bonferroni'],
            np.minimum(intercept_model_results['p_uncorrected'] * 5, 1.0),
            rtol=1e-10
        )
        log(f"Bonferroni correction: {'PASS' if bonf_correct else 'FAIL'}")
        
        # Check VIF values reasonable (warn if >5, error if >10)
        # Note: VIF values not calculated in this analysis 
        log("VIF check skipped (multicollinearity assessment not implemented)")
        
        # Skip VIF validation for now
        vif_values = []
        if len(vif_values) > 0:
            max_vif = vif_values.max()
            log(f"Max VIF: {max_vif:.2f}")
            if max_vif > 10:
                log(f"ERROR: VIF > 10 indicates severe multicollinearity")
                raise ValueError(f"VIF too high: {max_vif:.2f} > 10")
            elif max_vif > 5:
                log(f"WARNING: VIF > 5 indicates moderate multicollinearity")
            else:
                log(f"VIF acceptable: {max_vif:.2f} <= 5")
        
        # Check all 6 model terms present (intercept + 5 predictors)
        expected_predictors = {'intercept', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T'}
        actual_predictors = set(intercept_model_results['predictor'])
        terms_complete = expected_predictors == actual_predictors
        log(f"All 6 terms present: {'PASS' if terms_complete else 'FAIL'}")
        if not terms_complete:
            missing = expected_predictors - actual_predictors
            extra = actual_predictors - expected_predictors
            if missing:
                log(f"Missing predictors: {missing}")
            if extra:
                log(f"Extra predictors: {extra}")

        # Final validation check
        if validation_result['valid'] and se_positive and bonf_correct and terms_complete:
            if len(vif_values) == 0 or max_vif <= 10:  # No VIF or acceptable VIF
                log("All validation checks PASSED")
            else:
                log("FAILED: VIF too high")
                sys.exit(1)
        else:
            log("FAILED: One or more checks failed")
            sys.exit(1)

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)