#!/usr/bin/env python3
"""fit_slope_model: Predict LMM slopes (forgetting rate) using cognitive test scores"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Import statsmodels for regression (custom implementation due to signature mismatch)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_fit_slope_model.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Regression Function (Due to Signature Mismatch)

def fit_multiple_regression_custom(X, y, add_constant=True, return_diagnostics=True):
    """
    Custom implementation of multiple regression with diagnostics.
    
    Based on gcode_lessons.md: When function signature mismatch found,
    write custom code instead of using mismatched function.
    
    Original tools.analysis_regression.fit_multiple_regression has signature:
    (X, y, feature_names) but 4_analysis.yaml expects (X, y, add_constant, return_diagnostics)
    """
    # Add constant term if requested
    if add_constant:
        X_reg = sm.add_constant(X)
        predictor_names = ['intercept'] + list(X.columns if hasattr(X, 'columns') else [f'X{i}' for i in range(X.shape[1])])
    else:
        X_reg = X
        predictor_names = list(X.columns if hasattr(X, 'columns') else [f'X{i}' for i in range(X.shape[1])])
    
    # Fit OLS model
    model = sm.OLS(y, X_reg).fit()
    
    # Extract coefficients
    coefficients = []
    conf_int = model.conf_int()  # Returns (n_params, 2) numpy array
    
    for i, predictor in enumerate(predictor_names):
        coefficients.append({
            'predictor': predictor,
            'beta': model.params.iloc[i],
            'se': model.bse.iloc[i],
            't': model.tvalues.iloc[i],
            'p_uncorrected': model.pvalues.iloc[i],
            'p_bonferroni': model.pvalues.iloc[i] * len(predictor_names),  # Bonferroni correction
            'CI_lower': conf_int.iloc[i, 0],  # conf_int is DataFrame
            'CI_upper': conf_int.iloc[i, 1]   # conf_int is DataFrame
        })
    
    # Compute VIF if diagnostics requested
    if return_diagnostics and add_constant:
        # VIF computed on predictors only (exclude constant)
        try:
            for i, coef_data in enumerate(coefficients[1:], 1):  # Skip intercept
                vif_val = variance_inflation_factor(X_reg.values, i)
                coef_data['VIF'] = vif_val
            # Set VIF for intercept to NaN (not applicable)
            coefficients[0]['VIF'] = np.nan
        except Exception as e:
            log(f"VIF calculation failed: {e}")
            for coef_data in coefficients:
                coef_data['VIF'] = np.nan
    else:
        # No VIF calculation
        for coef_data in coefficients:
            coef_data['VIF'] = np.nan
    
    results = {
        'model': model,
        'coefficients': pd.DataFrame(coefficients),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'p_value': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic
    }
    
    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: fit_slope_model")
        # Load Input Data

        log("Loading regression input data...")
        regression_input = pd.read_csv(RQ_DIR / "data/step02_regression_input.csv")
        log(f"step02_regression_input.csv ({len(regression_input)} rows, {len(regression_input.columns)} cols)")
        
        # Verify required columns exist
        required_columns = ['slope', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        missing_cols = [col for col in required_columns if col not in regression_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_columns}")
        # Run Analysis Tool

        log("Running custom fit_multiple_regression for slope prediction...")
        
        # Prepare predictors and outcome
        X = regression_input[['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']]  # Cognitive test predictors
        y = regression_input['slope']  # LMM random slope (forgetting rate)

        log(f"Predictors: {list(X.columns)}")
        log(f"Outcome: slope (forgetting rate)")
        log(f"Sample size: N={len(X)}")
        
        # Fit regression model
        slope_results = fit_multiple_regression_custom(
            X=X,
            y=y,
            add_constant=True,  # Add intercept term
            return_diagnostics=True  # Compute VIF and other diagnostics
        )
        
        log("Slope prediction analysis complete")
        log(f"R² = {slope_results['r2']:.4f}, Adjusted R² = {slope_results['adj_r2']:.4f}")
        log(f"F({len(X.columns)}, {len(X) - len(X.columns) - 1}) = {slope_results['f_statistic']:.3f}, p = {slope_results['p_value']:.6f}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 5 (R² comparison), Step 6 (predictor significance)

        log("Saving step04_slope_predictions.csv...")
        # Output: step04_slope_predictions.csv
        # Contains: Regression coefficients predicting LMM slopes
        # Columns: predictor, beta, se, t, p_uncorrected, p_bonferroni, CI_lower, CI_upper, VIF
        slope_model_results = slope_results['coefficients']
        slope_model_results.to_csv(RQ_DIR / "data/step04_slope_predictions.csv", index=False, encoding='utf-8')
        log(f"step04_slope_predictions.csv ({len(slope_model_results)} rows, {len(slope_model_results.columns)} cols)")
        
        # Log key results
        log("Slope prediction model coefficients:")
        for idx, row in slope_model_results.iterrows():
            if row['predictor'] == 'intercept':
                log(f"  {row['predictor']}: β = {row['beta']:.4f}, SE = {row['se']:.4f}, p = {row['p_uncorrected']:.6f}")
            else:
                log(f"  {row['predictor']}: β = {row['beta']:.4f}, SE = {row['se']:.4f}, p = {row['p_uncorrected']:.6f}, VIF = {row['VIF']:.3f}")
        # Run Validation Tool
        # Validates: p-values are in valid range [0, 1]
        # Threshold: min=0.0, max=1.0

        log("Running validate_numeric_range...")
        validation_result = validate_numeric_range(
            data=slope_model_results['p_uncorrected'],
            min_val=0.0,
            max_val=1.0,
            column_name="p_uncorrected"
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation checks
        # Check standard errors are positive
        negative_se = slope_model_results[slope_model_results['se'] <= 0]
        if len(negative_se) > 0:
            log(f"WARNING: {len(negative_se)} negative/zero standard errors found")
        else:
            log("PASS: All standard errors positive")

        # Check Bonferroni correction applied correctly
        expected_bonf = slope_model_results['p_uncorrected'] * len(slope_model_results)  # All terms
        bonf_correct = np.allclose(slope_model_results['p_bonferroni'], expected_bonf, rtol=1e-10)
        if bonf_correct:
            log("PASS: Bonferroni correction applied correctly")
        else:
            log("WARNING: Bonferroni correction may be incorrect")

        # Check VIF values (warn if >5, error if >10)
        high_vif = slope_model_results[slope_model_results['VIF'] > 5]['predictor'].tolist()
        extreme_vif = slope_model_results[slope_model_results['VIF'] > 10]['predictor'].tolist()
        
        if extreme_vif:
            log(f"ERROR: Extreme multicollinearity detected (VIF > 10): {extreme_vif}")
        elif high_vif:
            log(f"WARNING: High multicollinearity detected (VIF > 5): {high_vif}")
        else:
            log("PASS: VIF values reasonable (all <= 5)")

        # Check all 4 model terms present
        expected_terms = ['intercept', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        actual_terms = slope_model_results['predictor'].tolist()
        if set(actual_terms) == set(expected_terms):
            log("PASS: All 6 model terms present (intercept + 5 predictors)")
        else:
            missing_terms = set(expected_terms) - set(actual_terms)
            extra_terms = set(actual_terms) - set(expected_terms)
            if missing_terms:
                log(f"ERROR: Missing model terms: {missing_terms}")
            if extra_terms:
                log(f"WARNING: Extra model terms: {extra_terms}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)