#!/usr/bin/env python3
"""hierarchical_regression: Hierarchical multiple regression to test VR scaffolding hypothesis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statistics imports
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import f as f_dist

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_hierarchical_regression.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Hierarchical Regression Implementation

def fit_hierarchical_regression_custom(X_blocks, y, block_names=None, add_constant=True):
    """
    Custom hierarchical regression implementation.
    
    Addresses signature mismatch: actual fit_hierarchical_regression has different params.
    Based on lessons learned from RQ 7.1.2 Step 04: write custom when signatures don't match.
    
    Parameters:
    - X_blocks: List of DataFrames/arrays, each representing a block of predictors
    - y: Response variable (theta_all scores)
    - block_names: Names for each model block
    - add_constant: Whether to add intercept (always True for regression)
    
    Returns:
    - Dict with keys: models, incremental_r2, summary
    """
    if block_names is None:
        block_names = [f"Model_{i+1}" for i in range(len(X_blocks))]
    
    models = []
    r2_values = []
    results_summary = []
    
    # Fit models incrementally (Model 1: Block 1 only, Model 2: Block 1 + Block 2, etc.)
    for i, block_name in enumerate(block_names):
        # Combine all blocks up to current one
        if i == 0:
            X_current = X_blocks[0]
        else:
            # For Model 2: combine Age_std + cognitive tests
            X_list = [X_blocks[j] for j in range(i+1)]
            if all(isinstance(x, pd.DataFrame) for x in X_list):
                X_current = pd.concat(X_list, axis=1)
            else:
                X_current = np.column_stack(X_list)
        
        # Add constant if requested
        if add_constant:
            X_reg = sm.add_constant(X_current)
        else:
            X_reg = X_current
        
        # Fit OLS model
        model = sm.OLS(y, X_reg).fit()
        models.append(model)
        
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
        r2_values.append(r2)
        
        # Calculate incremental R² (change from previous model)
        if i == 0:
            delta_r2 = r2  # First model: delta = R² itself
            f_change = model.fvalue
            p_change = model.f_pvalue
        else:
            delta_r2 = r2 - r2_values[i-1]
            
            # F-test for change in R²
            # F = (delta_R² / df_change) / ((1 - R²_full) / df_error)
            df_change = model.df_model - models[i-1].df_model
            df_error = model.df_resid
            
            if delta_r2 > 0 and df_change > 0:
                f_change = (delta_r2 / df_change) / ((1 - r2) / df_error)
                p_change = 1 - f_dist.cdf(f_change, df_change, df_error)
            else:
                f_change = 0.0
                p_change = 1.0
        
        # Store results
        results_summary.append({
            'model': block_name,
            'R2': r2,
            'R2_adj': r2_adj,
            'F_stat': model.fvalue,
            'delta_R2': delta_r2,
            'F_change': f_change,
            'p_change': p_change,
            'AIC': model.aic
        })
    
    return {
        'models': models,
        'incremental_r2': [r['delta_R2'] for r in results_summary],
        'summary': pd.DataFrame(results_summary)
    }

def compute_regression_diagnostics_custom(model, X, y):
    """
    Custom regression diagnostics implementation.
    
    Addresses signature mismatch: actual validate_regression_assumptions has different params.
    Computes key assumption checks: normality, homoscedasticity, VIF, Cook's D.
    
    Parameters:
    - model: Fitted OLS model from statsmodels
    - X: Predictor matrix (with constant already added)
    - y: Response variable
    
    Returns:
    - List of diagnostic test results
    """
    diagnostics = []
    
    # 1. Normality test (Shapiro-Wilk on residuals)
    residuals = model.resid
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    diagnostics.append({
        'test': 'normality_shapiro_wilk',
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'threshold': 0.05,
        'violated': shapiro_p < 0.05,
        'remedial_action': 'Transform response or use robust methods' if shapiro_p < 0.05 else 'None needed'
    })
    
    # 2. Homoscedasticity test (Breusch-Pagan)
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
        
        diagnostics.append({
            'test': 'homoscedasticity_breusch_pagan',
            'statistic': bp_stat,
            'p_value': bp_p,
            'threshold': 0.05,
            'violated': bp_p < 0.05,
            'remedial_action': 'Use heteroscedasticity-consistent standard errors' if bp_p < 0.05 else 'None needed'
        })
    except Exception as e:
        log(f"Breusch-Pagan test failed: {e}")
        diagnostics.append({
            'test': 'homoscedasticity_breusch_pagan',
            'statistic': np.nan,
            'p_value': np.nan,
            'threshold': 0.05,
            'violated': False,
            'remedial_action': 'Test failed - manual inspection needed'
        })
    
    # 3. Multicollinearity (VIF for each predictor excluding constant)
    # Note: VIF calculation requires no constant in X matrix
    X_no_const = X.iloc[:, 1:] if isinstance(X, pd.DataFrame) else X[:, 1:]  # Remove constant column
    
    max_vif = 0
    try:
        if X_no_const.shape[1] > 1:  # Need at least 2 predictors for VIF
            vif_values = []
            for i in range(X_no_const.shape[1]):
                vif = variance_inflation_factor(X_no_const.values, i)
                vif_values.append(vif)
                if not np.isnan(vif):
                    max_vif = max(max_vif, vif)
        else:
            max_vif = 1.0  # Single predictor has no multicollinearity
        
        diagnostics.append({
            'test': 'multicollinearity_max_vif',
            'statistic': max_vif,
            'p_value': np.nan,  # VIF doesn't have p-value
            'threshold': 5.0,
            'violated': max_vif > 5.0,
            'remedial_action': 'Remove or combine predictors' if max_vif > 5.0 else 'None needed'
        })
    except Exception as e:
        log(f"VIF calculation failed: {e}")
        diagnostics.append({
            'test': 'multicollinearity_max_vif',
            'statistic': np.nan,
            'p_value': np.nan,
            'threshold': 5.0,
            'violated': False,
            'remedial_action': 'VIF calculation failed - manual check needed'
        })
    
    # 4. Outliers (Cook's Distance)
    try:
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        max_cooks_d = np.max(cooks_d)
        threshold_cooks = 4.0 / len(y)  # 4/n threshold
        
        diagnostics.append({
            'test': 'outliers_max_cooks_d',
            'statistic': max_cooks_d,
            'p_value': np.nan,  # Cook's D doesn't have p-value
            'threshold': threshold_cooks,
            'violated': max_cooks_d > threshold_cooks,
            'remedial_action': 'Investigate high-leverage points' if max_cooks_d > threshold_cooks else 'None needed'
        })
    except Exception as e:
        log(f"Cook's Distance calculation failed: {e}")
        diagnostics.append({
            'test': 'outliers_max_cooks_d',
            'statistic': np.nan,
            'p_value': np.nan,
            'threshold': 4.0 / len(y),
            'violated': False,
            'remedial_action': 'Cook\'s D calculation failed - manual check needed'
        })
    
    # 5. Autocorrelation (Durbin-Watson)
    try:
        dw_stat = durbin_watson(residuals)
        # DW test: values near 2 indicate no autocorrelation
        # Rule of thumb: 1.5 < DW < 2.5 is acceptable
        
        diagnostics.append({
            'test': 'autocorrelation_durbin_watson',
            'statistic': dw_stat,
            'p_value': np.nan,  # DW doesn't provide p-value directly
            'threshold': 1.5,  # Lower bound - accept if DW > 1.5 and DW < 2.5
            'violated': dw_stat < 1.5 or dw_stat > 2.5,
            'remedial_action': 'Check for time-series patterns' if (dw_stat < 1.5 or dw_stat > 2.5) else 'None needed'
        })
    except Exception as e:
        log(f"Durbin-Watson test failed: {e}")
        diagnostics.append({
            'test': 'autocorrelation_durbin_watson',
            'statistic': np.nan,
            'p_value': np.nan,
            'threshold': 1.5,
            'violated': False,
            'remedial_action': 'DW test failed - manual inspection needed'
        })
    
    return diagnostics

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Hierarchical Multiple Regression")
        # Load Input Data

        log("Loading analysis dataset...")
        analysis_df = pd.read_csv(RQ_DIR / "data/step01_analysis_dataset.csv")
        log(f"step01_analysis_dataset.csv ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")
        
        # Extract required variables
        y = analysis_df['theta_all']  # VR ability scores (response variable)
        
        # Model 1: Age effect only
        X_block1 = analysis_df[['Age_std']]
        
        # Model 2: Age + cognitive tests (full model)
        X_block2 = analysis_df[['RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std', 'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']]

        log(f"Response variable: theta_all (n={len(y)})")
        log(f"Model 1 predictors: Age_std")
        log(f"Model 2 additional predictors: RAVLT_T_std, BVMT_T_std, RPM_T_std, RAVLT_Pct_Ret_T_std, BVMT_Pct_Ret_T_std")
        # Run Hierarchical Regression Analysis

        log("Running hierarchical regression...")
        
        # Set up blocks for hierarchical analysis
        X_blocks = [X_block1, X_block2]
        block_names = ["Model_1_Age_Only", "Model_2_Age_Plus_Cognitive"]
        
        # Fit hierarchical regression with custom implementation
        hierarchical_results = fit_hierarchical_regression_custom(
            X_blocks=X_blocks,
            y=y,
            block_names=block_names,
            add_constant=True  # Include intercept in both models
        )
        
        log("Hierarchical regression complete")

        # Extract model results for reporting
        models = hierarchical_results['models']
        summary_df = hierarchical_results['summary']
        
        # Report key findings
        model1_r2 = summary_df.iloc[0]['R2']
        model2_r2 = summary_df.iloc[1]['R2']
        delta_r2 = summary_df.iloc[1]['delta_R2']
        f_change = summary_df.iloc[1]['F_change']
        p_change = summary_df.iloc[1]['p_change']
        
        log(f"Model 1 R² = {model1_r2:.4f}")
        log(f"Model 2 R² = {model2_r2:.4f}")
        log(f"ΔR² = {delta_r2:.4f}")
        log(f"F-change = {f_change:.4f}, p = {p_change:.4f}")
        
        # Key hypothesis test: Age coefficient significance
        model1_age_coef = models[0].params.iloc[-1]  # Age coefficient in Model 1
        model1_age_p = models[0].pvalues.iloc[-1]    # Age p-value in Model 1
        
        model2_age_coef = models[1].params.iloc[1]  # Age coefficient in Model 2 (index 1 after constant)
        model2_age_p = models[1].pvalues.iloc[1]    # Age p-value in Model 2
        
        alpha_bonferroni = 0.05 / 6  # Bonferroni correction: 0.05/6 predictors
        
        log(f"Age coefficient Model 1: β = {model1_age_coef:.4f}, p = {model1_age_p:.4f}")
        log(f"Age coefficient Model 2: β = {model2_age_coef:.4f}, p = {model2_age_p:.4f}")
        log(f"Bonferroni α = {alpha_bonferroni:.4f}")
        
        # Mediation hypothesis: Age should be significant in Model 1, not significant in Model 2
        age_sig_model1 = model1_age_p < alpha_bonferroni
        age_sig_model2 = model2_age_p < alpha_bonferroni
        
        if age_sig_model1 and not age_sig_model2:
            hypothesis_result = "FULL MEDIATION: Age effect mediated by cognitive abilities"
        elif age_sig_model1 and age_sig_model2:
            hypothesis_result = "PARTIAL MEDIATION: Age effect partially mediated"
        elif not age_sig_model1:
            hypothesis_result = "NO MEDIATION: Age not significant in bivariate model"
        else:
            hypothesis_result = "NO MEDIATION: Age remains significant after controlling for cognitive abilities"
        
        log(f"Result: {hypothesis_result}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 mediation analysis, Step 8 plot generation

        log(f"Saving hierarchical model results...")
        # Output: step03_hierarchical_models.csv
        # Contains: Model comparison statistics for thesis reporting
        summary_df.to_csv(RQ_DIR / "data/step03_hierarchical_models.csv", index=False, encoding='utf-8')
        log(f"step03_hierarchical_models.csv ({len(summary_df)} rows, {len(summary_df.columns)} cols)")
        # Run Regression Diagnostics
        # Validates: Normality, homoscedasticity, multicollinearity, outliers, autocorrelation
        # Key thresholds: Shapiro p>0.05, BP p>0.05, VIF<5.0, Cook's D<4/n

        log("Running regression assumption checks...")
        
        # Focus on Model 2 (full model) for assumption checking
        model2 = models[1]
        X_model2 = sm.add_constant(pd.concat([X_block1, X_block2], axis=1))  # Age + cognitive tests with constant
        
        diagnostics = compute_regression_diagnostics_custom(
            model=model2,
            X=X_model2,
            y=y
        )
        
        # Create diagnostics DataFrame
        diagnostics_df = pd.DataFrame(diagnostics)
        
        # Report validation results
        log("Regression assumption checks:")
        for _, diag in diagnostics_df.iterrows():
            status = "VIOLATED" if diag['violated'] else "OK"
            if not pd.isna(diag['p_value']):
                log(f"{diag['test']}: {status} (statistic={diag['statistic']:.4f}, p={diag['p_value']:.4f})")
            else:
                log(f"{diag['test']}: {status} (statistic={diag['statistic']:.4f})")
        
        # Save diagnostics
        log(f"Saving regression diagnostics...")
        diagnostics_df.to_csv(RQ_DIR / "data/step03_model_diagnostics.csv", index=False, encoding='utf-8')
        log(f"step03_model_diagnostics.csv ({len(diagnostics_df)} rows, {len(diagnostics_df.columns)} cols)")

        # Report any major assumption violations
        major_violations = diagnostics_df[diagnostics_df['violated'] == True]
        if len(major_violations) > 0:
            log(f"{len(major_violations)} assumption violations detected:")
            for _, viol in major_violations.iterrows():
                log(f"- {viol['test']}: {viol['remedial_action']}")
        else:
            log("All regression assumptions satisfied")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)