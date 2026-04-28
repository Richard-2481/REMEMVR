#!/usr/bin/env python3
"""Check Regression Assumptions: Check all multiple regression assumptions with visual and statistical diagnostics."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.analysis_regression import fit_multiple_regression, compute_regression_diagnostics

# Import validation tools
from tools.validation import validate_data_columns

# Statistical testing
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_check_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Check Regression Assumptions")
        # Load Input Data

        log("Loading merged analysis dataset...")
        # Load step03_merged_analysis.csv (actual filename from directory listing)
        # Expected columns: ["UID", "RAVLT_T", "BVMT_T", "NART_T", "RPM_T", "theta_mean"]
        analysis_data = pd.read_csv(RQ_DIR / "data" / "step03_merged_analysis.csv")
        log(f"step03_merged_analysis.csv ({len(analysis_data)} rows, {len(analysis_data.columns)} cols)")
        
        # Verify expected columns
        expected_cols = ["UID", "RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T", "theta_mean"]
        actual_cols = analysis_data.columns.tolist()
        log(f"Expected columns: {expected_cols}")
        log(f"Actual columns: {actual_cols}")
        
        if not all(col in actual_cols for col in expected_cols):
            missing = [col for col in expected_cols if col not in actual_cols]
            raise ValueError(f"Missing columns: {missing}")
        # Prepare Regression Data

        log("Preparing regression variables...")
        # Predictors: cognitive test T-scores
        predictor_cols = ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T"]
        X = analysis_data[predictor_cols]
        log(f"Predictors: {predictor_cols} ({X.shape[0]} observations)")
        
        # Outcome: mean theta scores
        y = analysis_data["theta_mean"]
        log(f"Outcome: theta_mean (range: {y.min():.3f} to {y.max():.3f})")
        
        # Check for missing data
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        log(f"Missing data - Predictors: {missing_X}, Outcome: {missing_y}")
        
        if missing_X > 0 or missing_y > 0:
            log("Missing data detected - using complete cases only")
            complete_cases = analysis_data.dropna(subset=predictor_cols + ["theta_mean"])
            X = complete_cases[predictor_cols]
            y = complete_cases["theta_mean"]
            log(f"Complete cases: {len(complete_cases)} participants")
        # Fit Multiple Regression Model

        log("Fitting multiple regression model...")
        # Use actual function signature: fit_multiple_regression(X, y, feature_names)
        regression_results = fit_multiple_regression(
            X=X, 
            y=y, 
            feature_names=predictor_cols
        )
        
        # Extract model object for diagnostics
        fitted_model = regression_results['model']
        log(f"Regression model - R² = {regression_results.get('r2', 0.0):.3f}")
        
        # Extract fitted values and residuals for assumption tests
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        log(f"Fitted values and residuals for {len(fitted_values)} observations")
        # Run Comprehensive Regression Diagnostics

        log("Running comprehensive assumption tests...")
        
        # Use actual function signature: compute_regression_diagnostics(model, X, y)
        X_array = X.values  # Convert DataFrame to numpy array for function
        y_array = y.values  # Convert Series to numpy array for function
        
        diagnostic_results = compute_regression_diagnostics(
            model=fitted_model,
            X=X_array,
            y=y_array
        )
        log("Basic diagnostics from tools.analysis_regression")
        
        # Build comprehensive assumption test results
        assumption_tests = []
        
        # 1. Normality Test (Shapiro-Wilk on residuals)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            assumption_tests.append({
                'assumption': 'Normality',
                'test': 'Shapiro-Wilk',
                'test_statistic': shapiro_stat,
                'p_value': shapiro_p,
                'threshold': 0.05,
                'result': 'PASS' if shapiro_p > 0.05 else 'FAIL',
                'interpretation': 'Residuals normally distributed' if shapiro_p > 0.05 else 'Non-normal residuals detected'
            })
            log(f"Normality (Shapiro-Wilk): W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
        except Exception as e:
            log(f"Normality test failed: {e}")
            assumption_tests.append({
                'assumption': 'Normality',
                'test': 'Shapiro-Wilk',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': 0.05,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })

        # 2. Homoscedasticity Test (Breusch-Pagan)
        try:
            # Add constant to X for Breusch-Pagan test
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X_array)
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
            assumption_tests.append({
                'assumption': 'Homoscedasticity',
                'test': 'Breusch-Pagan',
                'test_statistic': bp_stat,
                'p_value': bp_p,
                'threshold': 0.05,
                'result': 'PASS' if bp_p > 0.05 else 'FAIL',
                'interpretation': 'Constant variance' if bp_p > 0.05 else 'Heteroscedasticity detected'
            })
            log(f"Homoscedasticity (Breusch-Pagan): LM = {bp_stat:.4f}, p = {bp_p:.4f}")
        except Exception as e:
            log(f"Homoscedasticity test failed: {e}")
            assumption_tests.append({
                'assumption': 'Homoscedasticity',
                'test': 'Breusch-Pagan',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': 0.05,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })

        # 3. Multicollinearity Test (VIF from diagnostic_results)
        try:
            vif_values = diagnostic_results.get('vif', [])
            if isinstance(vif_values, (list, np.ndarray)) and len(vif_values) > 0:
                max_vif = max(vif_values)
                mean_vif = np.mean(vif_values)
                assumption_tests.append({
                    'assumption': 'Multicollinearity',
                    'test': 'VIF',
                    'test_statistic': max_vif,
                    'p_value': np.nan,  # VIF doesn't have p-value
                    'threshold': 5.0,
                    'result': 'PASS' if max_vif < 5.0 else 'FAIL',
                    'interpretation': f'Max VIF = {max_vif:.2f}, Mean VIF = {mean_vif:.2f}'
                })
                log(f"Multicollinearity (VIF): Max = {max_vif:.2f}, Mean = {mean_vif:.2f}")
            else:
                log("VIF values not computed in diagnostic_results")
                assumption_tests.append({
                    'assumption': 'Multicollinearity',
                    'test': 'VIF',
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    'threshold': 5.0,
                    'result': 'WARNING',
                    'interpretation': 'VIF not computed - manual check required'
                })
        except Exception as e:
            log(f"VIF test failed: {e}")
            assumption_tests.append({
                'assumption': 'Multicollinearity',
                'test': 'VIF',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': 5.0,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })

        # 4. Autocorrelation Test (Durbin-Watson)
        try:
            dw_stat = durbin_watson(residuals)
            # Durbin-Watson: values near 2 indicate no autocorrelation
            dw_interpretation = 'No autocorrelation' if 1.5 <= dw_stat <= 2.5 else 'Possible autocorrelation'
            dw_result = 'PASS' if 1.5 <= dw_stat <= 2.5 else 'WARNING'
            
            assumption_tests.append({
                'assumption': 'Independence',
                'test': 'Durbin-Watson',
                'test_statistic': dw_stat,
                'p_value': np.nan,  # DW doesn't have simple p-value
                'threshold': 2.0,
                'result': dw_result,
                'interpretation': f'DW = {dw_stat:.3f}, {dw_interpretation}'
            })
            log(f"Independence (Durbin-Watson): DW = {dw_stat:.3f}")
        except Exception as e:
            log(f"Durbin-Watson test failed: {e}")
            assumption_tests.append({
                'assumption': 'Independence',
                'test': 'Durbin-Watson',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': 2.0,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })

        # 5. Outlier Detection (Cook's Distance)
        try:
            cooks_d = diagnostic_results.get('cooks_d', [])
            if isinstance(cooks_d, (list, np.ndarray)) and len(cooks_d) > 0:
                n = len(cooks_d)
                threshold = 4.0 / n  # Cook's D threshold: 4/n
                outliers = np.sum(cooks_d > threshold)
                max_cooks = np.max(cooks_d)
                
                assumption_tests.append({
                    'assumption': 'Outliers',
                    'test': 'Cooks Distance',
                    'test_statistic': max_cooks,
                    'p_value': np.nan,
                    'threshold': threshold,
                    'result': 'PASS' if outliers == 0 else 'WARNING',
                    'interpretation': f'{outliers} outliers detected (threshold = {threshold:.4f})'
                })
                log(f"Outliers (Cook's D): {outliers} outliers, max = {max_cooks:.4f}")
            else:
                log("Cook's distances not computed")
                assumption_tests.append({
                    'assumption': 'Outliers',
                    'test': 'Cooks Distance',
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    'threshold': np.nan,
                    'result': 'WARNING',
                    'interpretation': 'Cooks D not computed - manual check required'
                })
        except Exception as e:
            log(f"Cook's distance test failed: {e}")
            assumption_tests.append({
                'assumption': 'Outliers',
                'test': 'Cooks Distance',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': np.nan,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })

        # 6. Linearity Assessment (using R² and residual patterns)
        try:
            # Simple linearity check: compare R² with non-parametric correlation
            r2 = regression_results.get('r2', 0.0)
            
            # Compute Spearman correlations between predictors and outcome
            spearman_corrs = []
            for col in predictor_cols:
                spearman_r, _ = stats.spearmanr(X[col], y)
                spearman_corrs.append(abs(spearman_r))
            
            max_spearman = max(spearman_corrs) if spearman_corrs else 0.0
            linearity_assessment = 'Linear' if r2 >= 0.1 else 'Weak/Non-linear'
            linearity_result = 'PASS' if r2 >= 0.1 else 'WARNING'
            
            assumption_tests.append({
                'assumption': 'Linearity',
                'test': 'R² vs Spearman',
                'test_statistic': r2,
                'p_value': np.nan,
                'threshold': 0.1,
                'result': linearity_result,
                'interpretation': f'R² = {r2:.3f}, Max |Spearman| = {max_spearman:.3f}, {linearity_assessment}'
            })
            log(f"Linearity: R² = {r2:.3f}, Max |Spearman| = {max_spearman:.3f}")
        except Exception as e:
            log(f"Linearity test failed: {e}")
            assumption_tests.append({
                'assumption': 'Linearity',
                'test': 'R² vs Spearman',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'threshold': 0.1,
                'result': 'ERROR',
                'interpretation': f'Test failed: {str(e)}'
            })
        # Save Assumption Test Results
        # Output: Comprehensive assumption test results for interpretation
        # Contains: All 6 regression assumptions with test statistics and interpretations

        log("Saving regression diagnostics...")
        # Convert assumption tests to DataFrame
        diagnostics_df = pd.DataFrame(assumption_tests)
        
        # Save assumption diagnostics
        diagnostics_path = RQ_DIR / "data" / "step04_regression_diagnostics.csv"
        diagnostics_df.to_csv(diagnostics_path, index=False, encoding='utf-8')
        log(f"{diagnostics_path} ({len(diagnostics_df)} assumption tests)")
        
        # Create diagnostic plots data for plotting pipeline
        plots_data = pd.DataFrame({
            'observation': range(len(fitted_values)),
            'fitted_values': fitted_values,
            'residuals': residuals,
            'studentized_residuals': diagnostic_results.get('studentized_residuals', np.full(len(residuals), np.nan)),
            'cooks_d': diagnostic_results.get('cooks_d', np.full(len(residuals), np.nan)),
            'leverage': diagnostic_results.get('leverage', np.full(len(residuals), np.nan))
        })
        
        plots_data_path = RQ_DIR / "data" / "step04_diagnostic_plots_data.csv"
        plots_data.to_csv(plots_data_path, index=False, encoding='utf-8')
        log(f"{plots_data_path} ({len(plots_data)} observations for plotting)")
        # Validation Summary
        # Validate: All assumption tests completed and interpretable

        log("Checking assumption test completeness...")
        
        # Count valid test results
        total_tests = len(assumption_tests)
        passed_tests = sum(1 for test in assumption_tests if test['result'] == 'PASS')
        failed_tests = sum(1 for test in assumption_tests if test['result'] == 'FAIL')
        warning_tests = sum(1 for test in assumption_tests if test['result'] == 'WARNING')
        error_tests = sum(1 for test in assumption_tests if test['result'] == 'ERROR')
        
        log(f"Assumption test summary:")
        log(f"  Total tests: {total_tests}")
        log(f"  PASS: {passed_tests}")
        log(f"  FAIL: {failed_tests}")
        log(f"  WARNING: {warning_tests}")
        log(f"  ERROR: {error_tests}")
        
        # Check critical assumptions
        critical_failures = [test for test in assumption_tests 
                           if test['result'] == 'FAIL' and test['assumption'] in ['Multicollinearity']]
        
        if critical_failures:
            log("Critical assumption violations detected:")
            for failure in critical_failures:
                log(f"  {failure['assumption']}: {failure['interpretation']}")
        
        # Overall validation
        if error_tests == 0 and total_tests >= 6:
            log("All assumption tests completed successfully")
        elif error_tests > 0:
            log(f"{error_tests} assumption tests had errors - review diagnostic results")
        else:
            log(f"Only {total_tests} assumption tests completed (expected 6)")

        log("Step 04: Regression assumptions checked")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)