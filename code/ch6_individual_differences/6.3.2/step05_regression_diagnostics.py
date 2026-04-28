#!/usr/bin/env python3
"""Regression Diagnostics: Comprehensive regression diagnostics with remedial actions to validate statistical"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import durbin_watson

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2
LOG_FILE = RQ_DIR / "logs" / "step05_regression_diagnostics.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom Diagnostic Functions (Due to Signature Mismatches)

def compute_regression_diagnostics_custom(df, predictor_vars, dependent_var, vif_threshold=5.0, 
                                        cooks_d_threshold=0.04, leverage_threshold=0.12):
    """
    Custom regression diagnostics computation.
    
    Computes VIF, Cook's D, leverage, studentized residuals, and assumption tests.
    """
    
    # Prepare data
    X = df[predictor_vars].copy()
    y = df[dependent_var].copy()
    
    # Handle categorical variables (sex)
    if 'sex' in X.columns and X['sex'].dtype == 'object':
        X = pd.get_dummies(X, columns=['sex'], drop_first=True)
    
    # Remove missing data
    analysis_data = pd.concat([X, y], axis=1).dropna()
    X_clean = analysis_data.iloc[:, :-1]
    y_clean = analysis_data.iloc[:, -1]
    
    # Add constant and fit model
    X_const = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_const).fit()
    
    # Get residuals and fitted values
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Compute diagnostics
    diagnostics = {}
    
    # 1. VIF (Variance Inflation Factors)
    vif_data = []
    for i, col in enumerate(X_const.columns[1:]):  # Skip constant
        vif_val = variance_inflation_factor(X_const.values, i + 1)
        vif_data.append({
            'predictor': col,
            'vif': vif_val,
            'vif_flag': vif_val > vif_threshold
        })
    
    diagnostics['vif'] = pd.DataFrame(vif_data)
    
    # 2. Outlier detection using OLS influence
    influence = OLSInfluence(model)
    
    # Cook's distance
    cooks_d = influence.cooks_distance[0]
    
    # Leverage (hat values)
    leverage = influence.hat_matrix_diag
    
    # Studentized residuals
    studentized_resid = influence.resid_studentized_external
    
    # Create outlier dataframe
    outlier_data = []
    for i in range(len(y_clean)):
        outlier_flag = (cooks_d[i] > cooks_d_threshold) or \
                      (leverage[i] > leverage_threshold) or \
                      (abs(studentized_resid[i]) > 3)  # 3-sigma rule
        
        outlier_data.append({
            'observation': i,
            'cooks_d': cooks_d[i],
            'leverage': leverage[i],
            'studentized_residual': studentized_resid[i],
            'outlier_flag': outlier_flag
        })
    
    diagnostics['outliers'] = pd.DataFrame(outlier_data)
    
    # 3. Model statistics
    diagnostics['model_stats'] = {
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'condition_number': np.linalg.cond(X_const),
        'n_observations': len(y_clean),
        'n_predictors': X_const.shape[1] - 1  # Exclude constant
    }
    
    return model, diagnostics

def validate_regression_assumptions_custom(model, X, significance_level=0.05,
                                         vif_threshold=5.0, durbin_watson_range=[1.5, 2.5]):
    """
    Custom regression assumption validation.
    
    Tests:
    1. Normality of residuals (Shapiro-Wilk)
    2. Homoscedasticity (Breusch-Pagan)
    3. Independence (Durbin-Watson)
    4. Multicollinearity (VIF)
    """
    
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    assumption_results = []
    
    # 1. Normality test (Shapiro-Wilk)
    try:
        if len(residuals) <= 5000:  # Shapiro-Wilk limit
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_result = "PASS" if shapiro_p > significance_level else "FAIL"
            remedial_action = "None needed" if shapiro_p > significance_level else "Consider robust regression or transformation"
            
            assumption_results.append({
                'test': 'Normality (Shapiro-Wilk)',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'threshold': significance_level,
                'result': normality_result,
                'remedial_action': remedial_action
            })
        else:
            # Use Kolmogorov-Smirnov for larger samples
            ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            normality_result = "PASS" if ks_p > significance_level else "FAIL"
            remedial_action = "None needed" if ks_p > significance_level else "Consider robust regression or transformation"
            
            assumption_results.append({
                'test': 'Normality (Kolmogorov-Smirnov)',
                'statistic': ks_stat,
                'p_value': ks_p,
                'threshold': significance_level,
                'result': normality_result,
                'remedial_action': remedial_action
            })
    except Exception as e:
        log(f"Normality test failed: {e}")
    
    # 2. Homoscedasticity test (Breusch-Pagan)
    try:
        bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X)
        homoscedasticity_result = "PASS" if bp_p > significance_level else "FAIL"
        remedial_action = "None needed" if bp_p > significance_level else "Consider weighted least squares or robust standard errors"
        
        assumption_results.append({
            'test': 'Homoscedasticity (Breusch-Pagan)',
            'statistic': bp_lm,
            'p_value': bp_p,
            'threshold': significance_level,
            'result': homoscedasticity_result,
            'remedial_action': remedial_action
        })
    except Exception as e:
        log(f"Breusch-Pagan test failed: {e}")
    
    # 3. Independence test (Durbin-Watson)
    try:
        dw_stat = durbin_watson(residuals)
        dw_result = "PASS" if durbin_watson_range[0] <= dw_stat <= durbin_watson_range[1] else "FAIL"
        remedial_action = "None needed" if dw_result == "PASS" else "Consider time-series methods or additional predictors"
        
        assumption_results.append({
            'test': 'Independence (Durbin-Watson)',
            'statistic': dw_stat,
            'p_value': np.nan,  # Durbin-Watson doesn't have exact p-value
            'threshold': f"Range {durbin_watson_range}",
            'result': dw_result,
            'remedial_action': remedial_action
        })
    except Exception as e:
        log(f"Durbin-Watson test failed: {e}")
    
    # 4. Multicollinearity check (VIF)
    try:
        X_const = sm.add_constant(X) if 'const' not in X.columns else X
        max_vif = 0
        high_vif_predictors = []
        
        for i, col in enumerate(X_const.columns[1:]):  # Skip constant
            vif_val = variance_inflation_factor(X_const.values, i + 1)
            max_vif = max(max_vif, vif_val)
            if vif_val > vif_threshold:
                high_vif_predictors.append(f"{col}={vif_val:.2f}")
        
        multicollinearity_result = "PASS" if max_vif <= vif_threshold else "FAIL"
        remedial_action = "None needed" if multicollinearity_result == "PASS" else f"Consider removing predictors: {', '.join(high_vif_predictors)}"
        
        assumption_results.append({
            'test': 'Multicollinearity (VIF)',
            'statistic': max_vif,
            'p_value': np.nan,  # VIF doesn't have p-value
            'threshold': vif_threshold,
            'result': multicollinearity_result,
            'remedial_action': remedial_action
        })
    except Exception as e:
        log(f"VIF calculation failed: {e}")
    
    return pd.DataFrame(assumption_results)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Regression Diagnostics")
        # Load Input Data

        log("Loading regression results and analysis dataset...")
        
        # Load regression results to get predictors
        regression_results = pd.read_csv(RQ_DIR / "data" / "step04_regression_results.csv")
        log(f"step04_regression_results.csv ({len(regression_results)} rows)")
        
        # Load analysis dataset
        df = pd.read_csv(RQ_DIR / "data" / "step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Extract predictor variables (excluding VIF-related columns)
        predictor_vars = ['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        dependent_var = 'calibration_quality'
        
        log(f"Predictor variables: {predictor_vars}")
        log(f"Dependent variable: {dependent_var}")
        # Compute Regression Diagnostics

        log("Computing regression diagnostics...")
        
        model, diagnostics = compute_regression_diagnostics_custom(
            df, 
            predictor_vars, 
            dependent_var,
            vif_threshold=5.0,
            cooks_d_threshold=0.04,  # 4/n for ~100 participants
            leverage_threshold=0.12   # 2(k+1)/n for 6 predictors
        )
        
        log("Regression diagnostics computed")
        log(f"Model R² = {diagnostics['model_stats']['r_squared']:.4f}")
        log(f"Condition number = {diagnostics['model_stats']['condition_number']:.2f}")
        
        # Report VIF values
        high_vif_count = diagnostics['vif']['vif_flag'].sum()
        log(f"High VIF predictors: {high_vif_count}/{len(diagnostics['vif'])}")
        
        # Report outliers
        outlier_count = diagnostics['outliers']['outlier_flag'].sum()
        log(f"Outlier observations: {outlier_count}/{len(diagnostics['outliers'])}")
        # Validate Regression Assumptions
        # Validates: Normality, homoscedasticity, independence, multicollinearity
        # Criteria: Significance level 0.05, VIF threshold 5.0, DW range [1.5, 2.5]

        log("Validating regression assumptions...")
        
        # Prepare data for assumption tests
        X_analysis = df[predictor_vars].copy()
        if 'sex' in X_analysis.columns and X_analysis['sex'].dtype == 'object':
            X_analysis = pd.get_dummies(X_analysis, columns=['sex'], drop_first=True)
        
        # Remove missing data to match model
        analysis_data = pd.concat([X_analysis, df[dependent_var]], axis=1).dropna()
        X_clean = analysis_data.iloc[:, :-1]
        
        assumption_results = validate_regression_assumptions_custom(
            model, 
            X_clean,
            significance_level=0.05,
            vif_threshold=5.0,
            durbin_watson_range=[1.5, 2.5]
        )
        
        log("Assumption validation complete")
        
        # Report assumption test results
        for _, test_result in assumption_results.iterrows():
            test_name = test_result['test']
            result = test_result['result']
            p_val = test_result['p_value']
            if not pd.isna(p_val):
                log(f"{test_name}: {result} (p = {p_val:.4f})")
            else:
                log(f"{test_name}: {result} (statistic = {test_result['statistic']:.3f})")
        # Save Diagnostic Outputs
        # These outputs will be used by: Step 06 (cross-validation awareness), Step 07 (power analysis context)

        log("Saving diagnostic results...")
        
        # Output: step05_diagnostics.csv
        # Contains: Comprehensive regression diagnostics with remedial actions
        assumption_results.to_csv(RQ_DIR / "data" / "step05_diagnostics.csv", index=False, encoding='utf-8')
        log(f"step05_diagnostics.csv ({len(assumption_results)} rows, {len(assumption_results.columns)} cols)")
        
        # Output: step05_outliers.csv
        # Contains: Outlier and influential point analysis
        diagnostics['outliers'].to_csv(RQ_DIR / "data" / "step05_outliers.csv", index=False, encoding='utf-8')
        log(f"step05_outliers.csv ({len(diagnostics['outliers'])} rows, {len(diagnostics['outliers'].columns)} cols)")
        # Summary Validation

        log("Overall regression model assessment...")
        
        # Count assumption violations
        failed_assumptions = assumption_results['result'].eq('FAIL').sum()
        total_assumptions = len(assumption_results)
        
        # Count severe outliers
        severe_outliers = diagnostics['outliers']['outlier_flag'].sum()
        total_observations = len(diagnostics['outliers'])
        
        log(f"Failed assumptions: {failed_assumptions}/{total_assumptions}")
        log(f"Outlier observations: {severe_outliers}/{total_observations} ({100*severe_outliers/total_observations:.1f}%)")
        
        # Overall model quality assessment
        model_quality = "GOOD"
        if failed_assumptions > total_assumptions / 2:
            model_quality = "POOR"
        elif failed_assumptions > 0 or severe_outliers > total_observations * 0.05:
            model_quality = "MODERATE"
        
        log(f"Overall model quality: {model_quality}")
        
        # Recommendations
        if model_quality == "POOR":
            log("Consider major model revisions: robust regression, transformations, or different modeling approach")
        elif model_quality == "MODERATE":
            log("Consider minor adjustments: robust standard errors, outlier investigation")
        else:
            log("Model diagnostics acceptable, proceed with confidence")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)