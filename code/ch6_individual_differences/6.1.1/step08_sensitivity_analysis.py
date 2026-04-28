#!/usr/bin/env python3
"""sensitivity_analysis: Conduct sensitivity analyses and compute comprehensive effect sizes for multiple regression model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.power as smp
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_sensitivity_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Analysis Parameters

# From 4_analysis.yaml parameters
SENSITIVITY_PREDICTORS = ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "RPM_T"]  # Exclude NART
OUTCOME = "theta_mean"
EFFECT_SIZE_THRESHOLDS = {
    "small": 0.02,
    "medium": 0.15, 
    "large": 0.35
}
ALPHA_UNCORRECTED = 0.05
ALPHA_BONFERRONI = 0.05 / 6  # 0.05/6 predictors
ALPHA_CHAPTER = 0.00179   # 0.05/28 tests in Chapter 7
RANDOM_STATE = 42

# Helper Functions

def fit_regression_model(X, y, add_constant=True):
    """Fit OLS regression and return results with diagnostics."""
    if add_constant:
        X_with_const = sm.add_constant(X)
    else:
        X_with_const = X
    
    model = sm.OLS(y, X_with_const)
    result = model.fit()
    
    return result

def calculate_cohens_f_squared(r_squared):
    """Calculate Cohen's f² effect size: f² = R²/(1-R²)."""
    if r_squared >= 1.0:
        return np.inf
    return r_squared / (1 - r_squared)

def interpret_cohens_f_squared(f_squared):
    """Interpret Cohen's f² effect size using conventional thresholds."""
    if f_squared < 0.02:
        return "negligible"
    elif f_squared < 0.15:
        return "small"
    elif f_squared < 0.35:
        return "medium"
    else:
        return "large"

def calculate_semi_partial_correlations(X, y):
    """Calculate semi-partial correlations (sr²) for each predictor."""
    # Fit full model
    X_full = sm.add_constant(X)
    full_model = sm.OLS(y, X_full).fit()
    full_r2 = full_model.rsquared
    
    sr_squared = {}
    
    # For each predictor, compute unique variance contribution
    for i, col in enumerate(X.columns):
        # Fit reduced model without this predictor
        X_reduced = X.drop(columns=[col])
        if len(X_reduced.columns) > 0:
            X_reduced_const = sm.add_constant(X_reduced)
            reduced_model = sm.OLS(y, X_reduced_const).fit()
            reduced_r2 = reduced_model.rsquared
        else:
            reduced_r2 = 0.0
        
        # Semi-partial correlation squared is the unique contribution
        sr_squared[col] = full_r2 - reduced_r2
    
    return sr_squared, full_r2

def compute_power_analysis(n, k, r2, alpha_levels):
    """Compute post-hoc power analysis for multiple alpha levels."""
    power_results = []
    
    effect_size = calculate_cohens_f_squared(r2)
    
    for alpha in alpha_levels:
        try:
            # Using F-test power for multiple regression
            power = smp.ftest_power(effect_size, dfnum=k-1, dfden=n-k, alpha=alpha)
            
            if power >= 0.8:
                interpretation = "adequate"
            elif power >= 0.7:
                interpretation = "moderate" 
            elif power >= 0.5:
                interpretation = "low"
            else:
                interpretation = "insufficient"
            
            power_results.append({
                'analysis_type': f'Multiple_Regression_Alpha_{alpha}',
                'power': power,
                'n': n,
                'k': k,
                'r2': r2,
                'alpha': alpha,
                'interpretation': interpretation
            })
        except Exception as e:
            log(f"Power calculation failed for alpha={alpha}: {e}")
            power_results.append({
                'analysis_type': f'Multiple_Regression_Alpha_{alpha}',
                'power': np.nan,
                'n': n,
                'k': k, 
                'r2': r2,
                'alpha': alpha,
                'interpretation': 'calculation_failed'
            })
    
    return power_results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: Sensitivity Analysis")
        # Load Input Data

        log("Loading input data...")
        
        # Load merged analysis dataset (using actual file path)
        # NOTE: 4_analysis.yaml specifies step03_analysis_dataset.csv but actual file is step03_merged_analysis.csv
        analysis_data = pd.read_csv(RQ_DIR / "data" / "step03_merged_analysis.csv")
        log(f"step03_merged_analysis.csv ({len(analysis_data)} rows, {len(analysis_data.columns)} cols)")
        
        # Load regression results from step05 (for comparison)
        regression_results = pd.read_csv(RQ_DIR / "data" / "step05_regression_results.csv") 
        log(f"step05_regression_results.csv ({len(regression_results)} rows, {len(regression_results.columns)} cols)")

        # Verify expected columns
        required_cols = ["UID", "RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T", "theta_mean"]
        missing_cols = [col for col in required_cols if col not in analysis_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        # Fit Multiple Regression Models for Sensitivity Analysis
        # Models: (1) Full with all 4 predictors, (2) Without NART, (3) Episodic only

        log("Fitting multiple regression models...")
        
        # Prepare outcome variable
        y = analysis_data# Model 1: Full model with all predictors (RAVLT, BVMT, NART, RPM)
        X_full = analysis_data[["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T"]]
        model_full = fit_regression_model(X_full, y, add_constant=True)
        log(f"Full model: R² = {model_full.rsquared:.4f}, Adj R² = {model_full.rsquared_adj:.4f}")
        
        # Model 2: Without NART (sensitivity to language validity)
        X_no_nart = analysis_data# ["RAVLT_T", "BVMT_T", "RPM_T"]
        model_no_nart = fit_regression_model(X_no_nart, y, add_constant=True)
        log(f"Without NART: R² = {model_no_nart.rsquared:.4f}, Adj R² = {model_no_nart.rsquared_adj:.4f}")
        
        # Model 3: Episodic memory only (RAVLT + BVMT)
        X_episodic = analysis_data[["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T"]]
        model_episodic = fit_regression_model(X_episodic, y, add_constant=True)
        log(f"Episodic only: R² = {model_episodic.rsquared:.4f}, Adj R² = {model_episodic.rsquared_adj:.4f}")
        # Compare Model Performance and Compute Effect Sizes
        # Compare R² values and compute comprehensive effect size metrics

        log("Computing model comparisons and effect sizes...")
        
        # Model comparison results
        sensitivity_results = []
        
        # Add full model
        sensitivity_results.append({
            'model': 'Full_Model',
            'r2': model_full.rsquared,
            'adj_r2': model_full.rsquared_adj,
            'f_statistic': model_full.fvalue,
            'p_value': model_full.f_pvalue,
            'delta_r2': 0.0  # Reference model
        })
        
        # Add without NART model
        delta_r2_no_nart = model_full.rsquared - model_no_nart.rsquared
        sensitivity_results.append({
            'model': 'Without_NART',
            'r2': model_no_nart.rsquared,
            'adj_r2': model_no_nart.rsquared_adj,
            'f_statistic': model_no_nart.fvalue,
            'p_value': model_no_nart.f_pvalue,
            'delta_r2': delta_r2_no_nart
        })
        
        # Add episodic only model
        delta_r2_episodic = model_full.rsquared - model_episodic.rsquared
        sensitivity_results.append({
            'model': 'Episodic_Only',
            'r2': model_episodic.rsquared,
            'adj_r2': model_episodic.rsquared_adj,
            'f_statistic': model_episodic.fvalue,
            'p_value': model_episodic.f_pvalue,
            'delta_r2': delta_r2_episodic
        })
        
        log(f"Model comparison: ΔNART = {delta_r2_no_nart:.4f}, ΔEpisodic = {delta_r2_episodic:.4f}")
        
        # Effect size calculations
        effect_sizes = []
        
        # Cohen's f² for each model
        for result in sensitivity_results:
            model_name = result['model']
            r2 = result['r2']
            f_squared = calculate_cohens_f_squared(r2)
            interpretation = interpret_cohens_f_squared(f_squared)
            
            effect_sizes.append({
                'effect_type': f'Cohens_f2_{model_name}',
                'value': f_squared,
                'interpretation': interpretation,
                'reference': 'Cohen (1988)'
            })
        
        # Semi-partial correlations for full model
        try:
            sr_squared, total_r2 = calculate_semi_partial_correlations(X_full, y)
            log(f"Semi-partial correlations: {sr_squared}")
            log(f"Sum of sr² = {sum(sr_squared.values()):.4f}, Total R² = {total_r2:.4f}")
            
            for predictor, sr2_value in sr_squared.items():
                effect_sizes.append({
                    'effect_type': f'Semi_partial_r2_{predictor}',
                    'value': sr2_value,
                    'interpretation': interpret_cohens_f_squared(sr2_value),  # Use same thresholds
                    'reference': 'Unique variance contribution'
                })
        except Exception as e:
            log(f"Semi-partial correlation calculation failed: {e}")
        # Compute Post-Hoc Power Analysis
        # Calculate statistical power for different alpha levels

        log("Computing post-hoc power analysis...")
        
        # Power analysis for multiple alpha levels
        alpha_levels = [ALPHA_UNCORRECTED, ALPHA_BONFERRONI, ALPHA_CHAPTER]
        n = len(analysis_data)
        k_full = len(X_full.columns) + 1  # +1 for intercept
        k_no_nart = len(X_no_nart.columns) + 1
        k_episodic = len(X_episodic.columns) + 1
        
        power_results = []
        
        # Power for full model
        power_full = compute_power_analysis(n, k_full, model_full.rsquared, alpha_levels)
        power_results.extend(power_full)
        
        # Power for without NART model  
        power_no_nart = compute_power_analysis(n, k_no_nart, model_no_nart.rsquared, alpha_levels)
        for result in power_no_nart:
            result['analysis_type'] = result['analysis_type'].replace('Multiple_Regression', 'Without_NART')
        power_results.extend(power_no_nart)
        
        log(f"Power analysis completed for {len(power_results)} conditions")
        # Save Analysis Outputs
        # Save sensitivity analysis results, effect sizes, and power analysis
        # These outputs will be used by: results analysis for final interpretation

        log("Saving analysis outputs...")
        
        # Convert to DataFrames and save
        sensitivity_df = pd.DataFrame(sensitivity_results)
        effect_sizes_df = pd.DataFrame(effect_sizes)
        power_analysis_df = pd.DataFrame(power_results)
        
        # Save sensitivity analysis
        sensitivity_path = RQ_DIR / "data" / "step08_sensitivity_analysis.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False, encoding='utf-8')
        log(f"{sensitivity_path.name} ({len(sensitivity_df)} rows, {len(sensitivity_df.columns)} cols)")
        
        # Save effect sizes
        effect_sizes_path = RQ_DIR / "data" / "step08_effect_sizes.csv"
        effect_sizes_df.to_csv(effect_sizes_path, index=False, encoding='utf-8')
        log(f"{effect_sizes_path.name} ({len(effect_sizes_df)} rows, {len(effect_sizes_df.columns)} cols)")
        
        # Save power analysis
        power_path = RQ_DIR / "data" / "step08_power_analysis.csv"
        power_analysis_df.to_csv(power_path, index=False, encoding='utf-8')
        log(f"{power_path.name} ({len(power_analysis_df)} rows, {len(power_analysis_df.columns)} cols)")
        # Validate Analysis Results
        # Validate R² values, effect sizes, and power calculations
        # Criteria: R² in [0,1], f² ≥ 0, power in [0,1], interpretations provided

        log("Validating analysis results...")
        
        validation_passed = True
        validation_messages = []
        
        # Validate R² values in [0, 1] range
        r2_values = sensitivity_df['r2'].tolist()
        if all(0 <= r2 <= 1 for r2 in r2_values):
            validation_messages.append("All R² values in valid range [0, 1]")
        else:
            validation_passed = False
            validation_messages.append("Some R² values outside valid range [0, 1]")
        
        # Check Cohen's f² is positive
        f_squared_values = [es['value'] for es in effect_sizes if 'Cohens_f2' in es['effect_type']]
        if all(f2 >= 0 for f2 in f_squared_values):
            validation_messages.append("All Cohen's f² values ≥ 0")
        else:
            validation_passed = False
            validation_messages.append("Some Cohen's f² values < 0")
        
        # Verify semi-partial correlations sum approximately to R²
        sr2_values = [es['value'] for es in effect_sizes if 'Semi_partial_r2' in es['effect_type']]
        if sr2_values:
            sr2_sum = sum(sr2_values)
            full_r2 = model_full.rsquared
            if abs(sr2_sum - full_r2) < 0.05:  # Allow 5% tolerance
                validation_messages.append(f"Semi-partial correlations sum ≈ R² ({sr2_sum:.4f} ≈ {full_r2:.4f})")
            else:
                validation_passed = False
                validation_messages.append(f"Semi-partial correlations sum ≠ R² ({sr2_sum:.4f} ≠ {full_r2:.4f})")
        
        # Confirm power values in [0, 1] range
        power_values = [p for p in power_analysis_df['power'].tolist() if not pd.isna(p)]
        if all(0 <= p <= 1 for p in power_values):
            validation_messages.append("All power values in valid range [0, 1]")
        else:
            validation_passed = False
            validation_messages.append("Some power values outside valid range [0, 1]")
        
        # Check effect size interpretations provided
        interpretations = effect_sizes_df['interpretation'].tolist()
        if all(interp and interp != '' for interp in interpretations):
            validation_messages.append("Interpretations provided for all effect sizes")
        else:
            validation_passed = False
            validation_messages.append("Missing interpretations for some effect sizes")
        
        # Power calculated for multiple alpha levels
        unique_alphas = power_analysis_df['alpha'].nunique()
        if unique_alphas >= 3:
            validation_messages.append(f"Power calculated for {unique_alphas} alpha levels")
        else:
            validation_passed = False
            validation_messages.append(f"Power calculated for only {unique_alphas} alpha levels")
        
        # Report validation results
        for msg in validation_messages:
            log(msg)
        
        if validation_passed:
            log("All criteria passed")
        else:
            log("Some criteria failed - document limitations in interpretation")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)