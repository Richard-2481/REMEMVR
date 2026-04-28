#!/usr/bin/env python3
"""model_diagnostics: Comprehensive regression diagnostics and assumption checking"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import warnings
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.3
LOG_FILE = RQ_DIR / "logs" / "step05_model_diagnostics.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Diagnostic Functions

def test_normality(residuals):
    """
    Test residual normality using Shapiro-Wilk test.
    """
    if len(residuals) > 5000:
        # Use Kolmogorov-Smirnov for large samples
        statistic, p_value = stats.kstest(residuals, 'norm')
        test_name = 'Kolmogorov-Smirnov'
    else:
        # Use Shapiro-Wilk for smaller samples
        statistic, p_value = stats.shapiro(residuals)
        test_name = 'Shapiro-Wilk'
    
    interpretation = 'Normal' if p_value > 0.05 else 'Non-normal'
    
    return {
        'test': f'{test_name} Normality',
        'statistic': statistic,
        'p_value': p_value,
        'interpretation': interpretation
    }

def test_homoscedasticity(model):
    """
    Test homoscedasticity using Breusch-Pagan test.
    """
    try:
        lm_statistic, lm_p_value, f_statistic, f_p_value = het_breuschpagan(
            model.resid, model.model.exog
        )
        
        interpretation = 'Homoscedastic' if lm_p_value > 0.05 else 'Heteroscedastic'
        
        return {
            'test': 'Breusch-Pagan Homoscedasticity',
            'statistic': lm_statistic,
            'p_value': lm_p_value,
            'interpretation': interpretation
        }
    except Exception as e:
        log(f"Breusch-Pagan test failed: {str(e)}")
        return {
            'test': 'Breusch-Pagan Homoscedasticity',
            'statistic': np.nan,
            'p_value': np.nan,
            'interpretation': 'Test failed'
        }

def test_linearity(y, fitted_values):
    """
    Test linearity by examining correlation between outcome and fitted values.
    High correlation indicates good linear relationship.
    """
    try:
        correlation, p_value = stats.pearsonr(y, fitted_values)
        
        # High correlation (>0.5) indicates good linearity
        interpretation = 'Linear' if correlation > 0.5 else 'Non-linear'
        
        return {
            'test': 'Linearity (Y vs Fitted)',
            'statistic': correlation,
            'p_value': p_value,
            'interpretation': interpretation
        }
    except Exception as e:
        log(f"Linearity test failed: {str(e)}")
        return {
            'test': 'Linearity (Y vs Fitted)',
            'statistic': np.nan,
            'p_value': np.nan,
            'interpretation': 'Test failed'
        }

def test_independence(residuals):
    """
    Test independence using Durbin-Watson test.
    """
    try:
        dw_statistic = durbin_watson(residuals)
        
        # Durbin-Watson interpretation: 
        # Values around 2.0 indicate no autocorrelation
        # Values < 1.5 or > 2.5 suggest autocorrelation
        if 1.5 <= dw_statistic <= 2.5:
            interpretation = 'Independent'
        else:
            interpretation = 'Autocorrelated'
        
        return {
            'test': 'Durbin-Watson Independence',
            'statistic': dw_statistic,
            'p_value': np.nan,  # DW test doesn't provide p-value directly
            'interpretation': interpretation
        }
    except Exception as e:
        log(f"Durbin-Watson test failed: {str(e)}")
        return {
            'test': 'Durbin-Watson Independence',
            'statistic': np.nan,
            'p_value': np.nan,
            'interpretation': 'Test failed'
        }

def identify_outliers(model, uids, criteria_dict):
    """
    Identify outliers using multiple criteria.
    """
    try:
        # Get influence measures
        influence = OLSInfluence(model)
        
        # Standardized residuals
        std_resid = influence.resid_studentized_external
        
        # Cook's distance
        cooks_d = influence.cooks_distance[0]  # First element is the distances
        
        # Leverage (hat values)
        leverage = influence.hat_matrix_diag
        
        n_obs = len(model.resid)
        n_params = len(model.params)
        
        # Apply criteria
        outlier_flags = []
        for i in range(n_obs):
            flags = []
            
            # Standardized residuals > 3 SD
            if abs(std_resid[i]) > 3:
                flags.append('high_residual')
            
            # Cook's distance > 4/n
            cook_threshold = 4 / n_obs
            if cooks_d[i] > cook_threshold:
                flags.append('high_cooks_d')
            
            # Leverage > 2p/n
            leverage_threshold = 2 * n_params / n_obs
            if leverage[i] > leverage_threshold:
                flags.append('high_leverage')
            
            # Overall outlier flag
            outlier_flag = len(flags) > 0
            outlier_flags.append('|'.join(flags) if flags else 'normal')
        
        # Create outlier analysis dataframe
        outlier_df = pd.DataFrame({
            'UID': uids,
            'standardized_residual': std_resid,
            'cooks_distance': cooks_d,
            'leverage': leverage,
            'outlier_flag': outlier_flags
        })
        
        return outlier_df
        
    except Exception as e:
        log(f"Outlier analysis failed: {str(e)}")
        return pd.DataFrame({
            'UID': uids,
            'standardized_residual': [np.nan] * len(uids),
            'cooks_distance': [np.nan] * len(uids),
            'leverage': [np.nan] * len(uids),
            'outlier_flag': ['unknown'] * len(uids)
        })

def calculate_vif(X_df):
    """
    Calculate Variance Inflation Factor for each predictor.
    """
    vif_data = []
    
    for i, column in enumerate(X_df.columns):
        try:
            # For each predictor, regress it on all other predictors
            y_vif = X_df.iloc[:, i].values
            X_vif = X_df.drop(columns=[column]).values
            
            # Add intercept to X_vif
            X_vif_with_intercept = sm.add_constant(X_vif)
            
            # Fit regression
            vif_model = sm.OLS(y_vif, X_vif_with_intercept).fit()
            
            # Calculate VIF = 1 / (1 - R²)
            r_squared = vif_model.rsquared
            vif = 1 / (1 - r_squared) if r_squared < 0.99 else np.inf
            
            # Interpret VIF
            if vif < 5:
                concern = 'No concern'
            elif vif < 10:
                concern = 'Moderate concern'
            else:
                concern = 'High concern'
            
            vif_data.append({
                'predictor': column,
                'vif': vif,
                'multicollinearity_concern': concern
            })
            
        except Exception as e:
            log(f"VIF calculation failed for {column}: {str(e)}")
            vif_data.append({
                'predictor': column,
                'vif': np.nan,
                'multicollinearity_concern': 'Calculation failed'
            })
    
    return pd.DataFrame(vif_data)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Model diagnostics")
        # Load Data and Refit Final Model

        log("Loading analysis dataset...")
        input_path = RQ_DIR / "data" / "step02_analysis_dataset.csv"
        df = pd.read_csv(input_path)
        log(f"Analysis dataset ({len(df)} rows, {len(df.columns)} cols)")
        
        # Check regression results exist
        regression_path = RQ_DIR / "data" / "step04_hierarchical_regression.csv"
        if not regression_path.exists():
            raise FileNotFoundError("step04 hierarchical regression results not found")
        
        # Prepare variables for final model
        predictor_vars = ['age', 'education_numeric', 'rehearsal_frequency', 'mnemonic_use']
        outcome_var = 'theta_all'
        required_vars = predictor_vars + [outcome_var, 'UID']
        
        # Check required columns
        missing_cols = [col for col in required_vars if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Complete case analysis
        df_complete = df[required_vars].dropna()
        n_complete = len(df_complete)
        log(f"[COMPLETE CASES] n={n_complete} participants")
        
        if n_complete < 20:
            raise ValueError(f"Insufficient sample size for diagnostics (n={n_complete})")
        # Refit Final Hierarchical Model
        # Refit: Full model with all predictors for diagnostic analysis

        log("Refitting final hierarchical regression model...")
        
        # Extract variables
        y = df_complete[outcome_var].values
        X = df_complete[predictor_vars].values
        uids = df_complete['UID'].values
        
        # Add intercept and fit model
        X_with_intercept = sm.add_constant(X)
        final_model = sm.OLS(y, X_with_intercept).fit()
        
        log(f"Model summary - R² = {final_model.rsquared:.4f}, F = {final_model.fvalue:.2f}")
        log(f"Observations: {final_model.nobs}, Parameters: {len(final_model.params)}")
        # Test Regression Assumptions
        # Tests: Normality, homoscedasticity, linearity, independence

        log("Testing regression assumptions...")
        
        assumption_tests = []
        
        # 1. Normality of residuals
        log("Testing normality of residuals...")
        normality_result = test_normality(final_model.resid)
        assumption_tests.append(normality_result)
        log(f"{normality_result['test']}: {normality_result['interpretation']} (p = {normality_result['p_value']:.6f})")
        
        # 2. Homoscedasticity
        log("Testing homoscedasticity...")
        homoscedasticity_result = test_homoscedasticity(final_model)
        assumption_tests.append(homoscedasticity_result)
        log(f"{homoscedasticity_result['interpretation']} (p = {homoscedasticity_result['p_value']:.6f})")
        
        # 3. Linearity
        log("Testing linearity...")
        linearity_result = test_linearity(y, final_model.fittedvalues)
        assumption_tests.append(linearity_result)
        log(f"{linearity_result['interpretation']} (r = {linearity_result['statistic']:.3f})")
        
        # 4. Independence
        log("Testing independence...")
        independence_result = test_independence(final_model.resid)
        assumption_tests.append(independence_result)
        log(f"{independence_result['interpretation']} (DW = {independence_result['statistic']:.3f})")
        
        # Save assumption test results
        assumption_df = pd.DataFrame(assumption_tests)
        assumption_path = RQ_DIR / "data" / "step05_assumption_tests.csv"
        assumption_df.to_csv(assumption_path, index=False, encoding='utf-8')
        log(f"Assumption tests: {assumption_path}")
        # Identify Outliers and Influential Cases
        # Criteria: Standardized residuals > 3SD, Cook's D > 4/n, leverage > 2p/n

        log("Identifying outliers and influential cases...")
        
        criteria = {
            'standardized_residuals_3sd': 3,
            'cooks_distance_4n': 4 / n_complete,
            'leverage_2pn': 2 * len(final_model.params) / n_complete
        }
        
        log(f"Cook's D threshold: {criteria['cooks_distance_4n']:.4f}")
        log(f"Leverage threshold: {criteria['leverage_2pn']:.4f}")
        
        outlier_df = identify_outliers(final_model, uids, criteria)
        
        # Count outliers by type
        outlier_counts = outlier_df['outlier_flag'].value_counts()
        log("[OUTLIER COUNTS] Outlier summary:")
        for flag, count in outlier_counts.items():
            log(f"  {flag}: {count} cases")
        
        # Save outlier analysis
        outlier_path = RQ_DIR / "data" / "step05_outlier_analysis.csv"
        outlier_df.to_csv(outlier_path, index=False, encoding='utf-8')
        log(f"Outlier analysis: {outlier_path}")
        # Check Multicollinearity (VIF Analysis)
        # VIF: Variance Inflation Factor for each predictor
        # Threshold: VIF > 5 indicates multicollinearity concern

        log("Calculating variance inflation factors...")
        
        # Create predictor dataframe for VIF calculation
        X_df = pd.DataFrame(X, columns=predictor_vars)
        vif_df = calculate_vif(X_df)
        
        log("Multicollinearity analysis:")
        for _, row in vif_df.iterrows():
            log(f"  {row['predictor']}: VIF = {row['vif']:.2f} ({row['multicollinearity_concern']})")
        
        # Save VIF analysis
        vif_path = RQ_DIR / "data" / "step05_vif_analysis.csv"
        vif_df.to_csv(vif_path, index=False, encoding='utf-8')
        log(f"VIF analysis: {vif_path}")
        # Run Validation
        # Validation: Check VIF values are in reasonable range (1.0 to 20.0)
        # Custom validation due to function signature mismatch

        log("Validating VIF analysis...")
        
        try:
            # Use tools validation function if compatible
            vif_validation = validate_numeric_range(
                data=vif_df['vif'].values,
                min_val=1.0,
                max_val=20.0,
                column_name='vif'
            )
            log("VIF validation completed via tools function")
            
        except Exception as e:
            log(f"Custom validation due to function limitations: {str(e)}")
            
            # Manual VIF validation
            vif_values = vif_df['vif'].values
            vif_valid = (~np.isnan(vif_values)) & (vif_values >= 1.0) & (vif_values <= 20.0)
            
            validation_checks = {
                'vif_range_valid': vif_valid.all(),
                'no_extreme_vif': (vif_values < 10.0).all() if len(vif_values) > 0 else True,
                'no_missing_vif': not np.isnan(vif_values).any(),
                'valid': True
            }
            
            validation_checks['valid'] = all([
                validation_checks['vif_range_valid'],
                validation_checks['no_missing_vif']
            ])
            
            for check, result in validation_checks.items():
                status = "" if result else ""
                log(f"{status} {check}: {result}")
        # Summary of Diagnostic Results
        
        log("Model diagnostic results:")
        
        # Assumption summary
        log("Assumption test summary:")
        passed_assumptions = sum(1 for test in assumption_tests 
                               if test['interpretation'] in ['Normal', 'Homoscedastic', 'Linear', 'Independent'])
        log(f"  Assumptions passed: {passed_assumptions} / {len(assumption_tests)}")
        
        # Outlier summary
        total_outliers = sum(1 for flag in outlier_df['outlier_flag'] if flag != 'normal')
        outlier_percent = total_outliers / len(outlier_df) * 100
        log(f"  Total outliers: {total_outliers} / {len(outlier_df)} ({outlier_percent:.1f}%)")
        
        # VIF summary
        high_vif_count = sum(1 for vif in vif_df['vif'] if vif > 5.0)
        log(f"  High VIF predictors: {high_vif_count} / {len(vif_df)}")
        
        # Overall model quality
        if passed_assumptions >= 3 and outlier_percent < 10 and high_vif_count == 0:
            log("Model diagnostics: GOOD (assumptions met, few outliers, low multicollinearity)")
        elif passed_assumptions >= 2 and outlier_percent < 15:
            log("Model diagnostics: ACCEPTABLE (some concerns but usable)")
        else:
            log("Model diagnostics: CONCERNING (multiple assumption violations)")

        log("Step 05 complete - comprehensive model diagnostics")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)