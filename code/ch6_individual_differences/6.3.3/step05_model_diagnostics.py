#!/usr/bin/env python3
"""Model diagnostics: Comprehensive model diagnostics and assumption checking for hierarchical regression."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statistical packages
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

from tools.analysis_extensions import validate_regression_assumptions

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_model_diagnostics.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Model diagnostics")
        # Load Input Data

        log("Loading regression results...")
        regression_results = pd.read_csv(RQ_DIR / "data/step04_regression_results.csv")
        log(f"step04_regression_results.csv ({len(regression_results)} rows, {len(regression_results.columns)} cols)")

        log("Loading analysis dataset...")
        analysis_data = pd.read_csv(RQ_DIR / "data/step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_data)} rows, {len(analysis_data.columns)} cols)")
        # Refit Full Model for Diagnostics

        log("Refitting full model for diagnostic analysis...")
        
        # Prepare full model data (from step04 specification)
        X_vars = ['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        X = analysis_data[X_vars].copy()
        y = analysis_data['hce_rate'].copy()
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X_with_const).fit()
        log("Model refitted successfully")
        
        # Extract diagnostic quantities
        fitted_values = model.fittedvalues
        residuals = model.resid
        standardized_residuals = model.resid_pearson  # Pearson residuals
        
        # Influence measures
        influence = OLSInfluence(model)
        cooks_distance = influence.cooks_distance[0]  # First element is the distances
        leverage = influence.hat_matrix_diag
        
        log(f"Diagnostic quantities: {len(fitted_values)} observations")
        # Multicollinearity Check (VIF)
        # Test: Variance Inflation Factor for each predictor
        # Threshold: VIF > 5.0 indicates problematic multicollinearity

        log("Computing VIF for multicollinearity check...")
        
        vif_data = []
        for i, var in enumerate(X.columns):
            vif_value = 1 / (1 - sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1)).fit().rsquared)
            vif_data.append({
                'variable': var,
                'vif': vif_value
            })
        
        vif_df = pd.DataFrame(vif_data)
        max_vif = vif_df['vif'].max()
        vif_threshold = 5.0
        
        multicollinearity_violation = max_vif > vif_threshold
        log(f"Maximum VIF: {max_vif:.3f} (threshold: {vif_threshold})")
        log(f"Multicollinearity concern: {multicollinearity_violation}")
        # Normality Check (Shapiro-Wilk)
        # Test: Shapiro-Wilk test of normality on residuals
        # Threshold: p < 0.05 indicates non-normal residuals

        log("Testing normality of residuals...")
        
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality_violation = shapiro_p < 0.05
        log(f"Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        log(f"Violation detected: {normality_violation}")
        # Homoscedasticity Check (Breusch-Pagan)
        # Test: Breusch-Pagan test for heteroscedasticity
        # Threshold: p < 0.05 indicates heteroscedasticity (violation of homoscedasticity)

        log("Testing homoscedasticity...")
        
        bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_with_const)
        heteroscedasticity_violation = bp_p < 0.05
        log(f"Breusch-Pagan: LM={bp_lm:.4f}, p={bp_p:.6f}")
        log(f"Violation detected: {heteroscedasticity_violation}")
        # Outlier Detection (Cook's Distance and Leverage)
        # Test: Cook's distance > 4/n and leverage > 3*k/n identify problematic observations
        # Threshold: Cook's > 0.04, leverage > 0.18 (for n=100, k=6)

        log("Identifying outliers and high leverage points...")
        
        n_obs = len(analysis_data)
        n_predictors = len(X_vars)
        
        cooks_threshold = 4 / n_obs  # 0.04 for n=100
        leverage_threshold = 3 * (n_predictors + 1) / n_obs  # 3*(k+1)/n, +1 for intercept
        
        outliers_cooks = np.sum(cooks_distance > cooks_threshold)
        outliers_leverage = np.sum(leverage > leverage_threshold)
        
        log(f"Cook's distance outliers: {outliers_cooks} (threshold: {cooks_threshold:.4f})")
        log(f"High leverage points: {outliers_leverage} (threshold: {leverage_threshold:.4f})")
        # Durbin-Watson Test (Autocorrelation)
        # Test: Durbin-Watson test for autocorrelation in residuals
        # Threshold: Values close to 2.0 indicate no autocorrelation

        log("Testing for autocorrelation...")
        
        dw_stat = durbin_watson(residuals)
        # DW interpretation: ~2 = no autocorrelation, <1.5 or >2.5 concerning
        autocorrelation_concern = dw_stat < 1.5 or dw_stat > 2.5
        log(f"Durbin-Watson: {dw_stat:.4f}")
        log(f"Concern detected: {autocorrelation_concern}")
        # Save Diagnostic Results
        # These outputs document assumption violations and guide remedial actions

        log("Saving diagnostic test results...")
        
        # Create diagnostic summary
        diagnostic_results = []
        
        # Multicollinearity
        diagnostic_results.append({
            'assumption': 'multicollinearity',
            'test_statistic': max_vif,
            'p_value': np.nan,  # VIF doesn't have p-value
            'interpretation': 'VIOLATION' if multicollinearity_violation else 'OK',
            'remedial_action': 'document_and_proceed' if multicollinearity_violation else 'none'
        })
        
        # Normality
        diagnostic_results.append({
            'assumption': 'normality',
            'test_statistic': shapiro_stat,
            'p_value': shapiro_p,
            'interpretation': 'VIOLATION' if normality_violation else 'OK',
            'remedial_action': 'use_bootstrap_cis' if normality_violation else 'none'
        })
        
        # Homoscedasticity
        diagnostic_results.append({
            'assumption': 'homoscedasticity',
            'test_statistic': bp_lm,
            'p_value': bp_p,
            'interpretation': 'VIOLATION' if heteroscedasticity_violation else 'OK',
            'remedial_action': 'use_robust_se' if heteroscedasticity_violation else 'none'
        })
        
        # Outliers (using Cook's distance)
        outliers_detected = outliers_cooks > 0
        diagnostic_results.append({
            'assumption': 'outliers',
            'test_statistic': np.max(cooks_distance),
            'p_value': np.nan,  # Cook's distance doesn't have p-value
            'interpretation': 'DETECTED' if outliers_detected else 'OK',
            'remedial_action': 'sensitivity_analysis' if outliers_detected else 'none'
        })
        
        # Autocorrelation
        diagnostic_results.append({
            'assumption': 'autocorrelation',
            'test_statistic': dw_stat,
            'p_value': np.nan,  # DW doesn't provide exact p-value easily
            'interpretation': 'CONCERN' if autocorrelation_concern else 'OK',
            'remedial_action': 'investigate_structure' if autocorrelation_concern else 'none'
        })
        
        diagnostics_df = pd.DataFrame(diagnostic_results)
        diagnostics_df.to_csv(RQ_DIR / "data/step05_model_diagnostics.csv", index=False, encoding='utf-8')
        log(f"step05_model_diagnostics.csv ({len(diagnostics_df)} rows, {len(diagnostics_df.columns)} cols)")

        # Create diagnostic plot data
        log("Saving diagnostic plot data...")
        
        plot_data = pd.DataFrame({
            'observation': range(len(fitted_values)),
            'fitted': fitted_values,
            'residuals': residuals,
            'standardized_residuals': standardized_residuals,
            'cooks_distance': cooks_distance,
            'leverage': leverage
        })
        
        plot_data.to_csv(RQ_DIR / "data/step05_diagnostic_plot_data.csv", index=False, encoding='utf-8')
        log(f"step05_diagnostic_plot_data.csv ({len(plot_data)} rows, {len(plot_data.columns)} cols)")
        # Run Validation Tool
        # Validates: Diagnostic results meet expected criteria
        # Threshold: significance_level = 0.05

        log("Running validate_regression_assumptions...")
        validation_result = validate_regression_assumptions(
            residuals=residuals.values,
            X=X.values,
            significance_level=0.05
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Summary of violations
        violations = [result for result in diagnostic_results if result['interpretation'] in ['VIOLATION', 'DETECTED', 'CONCERN']]
        log(f"Assumption violations detected: {len(violations)}")
        for violation in violations:
            log(f"- {violation['assumption']}: {violation['interpretation']} -> {violation['remedial_action']}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)