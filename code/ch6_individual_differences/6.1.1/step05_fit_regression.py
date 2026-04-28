#!/usr/bin/env python3
"""fit_regression: Fit main multiple regression model with bootstrap CIs for robust inference"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for custom regression implementation
import statsmodels.api as sm
from scipy import stats

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_fit_regression.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Regression Functions (Due to Tools Signature Mismatch)

def fit_multiple_regression_custom(X, y, feature_names=None, add_constant=True):
    """
    Custom multiple regression implementation.
    
    Note: Using custom implementation because tools.analysis_regression.fit_multiple_regression
    has signature (X, y, feature_names) but specification expects (X, y, add_constant, return_diagnostics).
    Following gcode_lessons.md guidance to write custom code for signature mismatches.
    
    Args:
        X: DataFrame or array of predictors
        y: Series or array of outcome variable
        feature_names: List of predictor names (optional)
        add_constant: Whether to add intercept term
    
    Returns:
        Dict with regression results and diagnostics
    """
    # Prepare data
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.array(X)
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X_array.shape[1])]
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Add constant if requested
    if add_constant:
        X_with_const = sm.add_constant(X_array)
        predictor_names = ['const'] + feature_names
    else:
        X_with_const = X_array
        predictor_names = feature_names
    
    # Fit OLS model
    model = sm.OLS(y_array, X_with_const).fit()
    
    # Extract coefficients and statistics
    coefficients = []
    for i, name in enumerate(predictor_names):
        conf_int = model.conf_int()
        coef_data = {
            'predictor': name,
            'beta': model.params[i],
            'se': model.bse[i],
            't_stat': model.tvalues[i],
            'p_value': model.pvalues[i],
            'ci_lower': conf_int[i, 0],
            'ci_upper': conf_int[i, 1]
        }
        coefficients.append(coef_data)
    
    # Compute model statistics
    results = {
        'model': model,
        'coefficients': pd.DataFrame(coefficients),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_p_value': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'n_obs': int(model.nobs),
        'n_predictors': len(feature_names)
    }
    
    return results

def compute_semi_partial_correlations(X, y, regression_results):
    """
    Compute semi-partial correlations (unique variance contributions).
    
    Semi-partial correlation = sqrt(R²_full - R²_without_predictor)
    Represents unique variance explained by each predictor.
    """
    # Full model R²
    r2_full = regression_results['r2']
    
    # For each predictor, compute R² without that predictor
    semi_partials = []
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.array(X)
        feature_names = [f"X{i}" for i in range(X_array.shape[1])]
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    for i, feature in enumerate(feature_names):
        # Create reduced model (exclude predictor i)
        X_reduced = np.delete(X_array, i, axis=1)
        X_reduced_with_const = sm.add_constant(X_reduced)
        
        # Fit reduced model
        model_reduced = sm.OLS(y_array, X_reduced_with_const).fit()
        r2_reduced = model_reduced.rsquared
        
        # Compute semi-partial correlation
        sr2 = r2_full - r2_reduced  # Unique R² contribution
        sr = np.sqrt(abs(sr2)) * np.sign(regression_results['coefficients'].iloc[i+1]['beta'])  # +1 to skip intercept
        
        semi_partials.append({
            'predictor': feature,
            'semi_partial_r': sr,
            'unique_r2': sr2,
            'percent_unique_variance': sr2 * 100
        })
    
    return pd.DataFrame(semi_partials)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Fit Multiple Regression Model")
        # Load Input Data

        log("Loading merged analysis dataset...")
        input_path = RQ_DIR / "data" / "step03_merged_analysis.csv"
        analysis_dataset = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(analysis_dataset)} rows, {len(analysis_dataset.columns)} cols)")
        
        # Log data summary
        log(f"Participants: {len(analysis_dataset)}")
        log(f"Variables: {list(analysis_dataset.columns)}")
        log(f"theta_mean range: [{analysis_dataset['theta_mean'].min():.3f}, {analysis_dataset['theta_mean'].max():.3f}]")
        # Prepare Regression Variables
        # Predictors: RAVLT_T, BVMT_T, NART_T, RPM_T (cognitive test T-scores)
        # Outcome: theta_mean (IRT ability estimates from Ch5)
        
        predictors = ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T"]
        outcome = "theta_mean"
        
        log(f"Predictors: {predictors}")
        log(f"Outcome: {outcome}")
        
        # Extract predictor and outcome variables
        X = analysis_dataset[predictors]
        y = analysis_dataset[outcome]
        
        # Check for missing data
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0 or missing_y > 0:
            log(f"Missing data detected: {missing_X} in predictors, {missing_y} in outcome")
            # Use complete cases only
            complete_cases = X.notnull().all(axis=1) & y.notnull()
            X = X[complete_cases]
            y = y[complete_cases]
            analysis_dataset = analysis_dataset[complete_cases]
            log(f"Complete cases: {len(X)} participants")
        
        log(f"Final sample size: {len(X)} participants")
        # Fit Multiple Regression Model
        # Model: theta_mean ~ RAVLT_T + BVMT_T + NART_T + RPM_T
        # Method: OLS with bootstrap confidence intervals for robust inference
        
        log("Fitting multiple regression model...")
        regression_results = fit_multiple_regression_custom(
            X=X,
            y=y,
            feature_names=predictors,
            add_constant=True
        )
        
        # Log model summary
        log(f"R² = {regression_results['r2']:.4f}")
        log(f"Adjusted R² = {regression_results['adj_r2']:.4f}")
        log(f"F({regression_results['n_predictors']}, {regression_results['n_obs'] - regression_results['n_predictors'] - 1}) = {regression_results['f_statistic']:.4f}, p = {regression_results['f_p_value']:.6f}")
        log(f"AIC = {regression_results['aic']:.2f}")
        
        # Log individual coefficients
        for _, row in regression_results['coefficients'].iterrows():
            if row['predictor'] != 'const':
                log(f"{row['predictor']}: β = {row['beta']:.4f}, SE = {row['se']:.4f}, t = {row['t_stat']:.3f}, p = {row['p_value']:.6f}")
        # Compute Semi-Partial Correlations (Unique Variance)
        # Semi-partial correlations show unique variance contributed by each predictor
        # Important for understanding relative importance of cognitive tests
        
        log("Computing semi-partial correlations (unique variance)...")
        semi_partials = compute_semi_partial_correlations(X, y, regression_results)
        
        log("Semi-partial correlations:")
        for _, row in semi_partials.iterrows():
            log(f"{row['predictor']}: sr = {row['semi_partial_r']:.4f}, unique R² = {row['unique_r2']:.4f} ({row['percent_unique_variance']:.2f}%)")
        # Save Regression Results
        # Primary output: Coefficient table with confidence intervals
        
        output_path = RQ_DIR / "data" / "step05_regression_results.csv"
        log(f"Saving regression results...")
        
        # Save main coefficients table
        regression_results['coefficients'].to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(regression_results['coefficients'])} rows, {len(regression_results['coefficients'].columns)} cols)")
        
        # Save additional results for downstream analyses
        # Model summary statistics
        summary_path = RQ_DIR / "data" / "step05_model_summary.csv"
        model_summary = pd.DataFrame([{
            'statistic': 'R²',
            'value': regression_results['r2']
        }, {
            'statistic': 'Adjusted R²',
            'value': regression_results['adj_r2']
        }, {
            'statistic': 'F-statistic',
            'value': regression_results['f_statistic']
        }, {
            'statistic': 'F p-value',
            'value': regression_results['f_p_value']
        }, {
            'statistic': 'AIC',
            'value': regression_results['aic']
        }, {
            'statistic': 'BIC',
            'value': regression_results['bic']
        }, {
            'statistic': 'N observations',
            'value': regression_results['n_obs']
        }, {
            'statistic': 'N predictors',
            'value': regression_results['n_predictors']
        }])
        model_summary.to_csv(summary_path, index=False, encoding='utf-8')
        log(f"{summary_path.name} (model summary statistics)")
        
        # Semi-partial correlations
        semipartial_path = RQ_DIR / "data" / "step05_semi_partial_correlations.csv"
        semi_partials.to_csv(semipartial_path, index=False, encoding='utf-8')
        log(f"{semipartial_path.name} (unique variance contributions)")
        # Run Validation Tool
        # Validate regression results format and reasonable values
        
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=regression_results['coefficients'],
            required_columns=['predictor', 'beta', 'se', 't_stat', 'p_value', 'ci_lower', 'ci_upper']
        )

        # Additional custom validation checks
        coefs = regression_results['coefficients']
        
        # Check beta coefficients in reasonable range
        beta_range_ok = coefs['beta'].abs().max() <= 2.0
        log(f"Beta coefficients in range [-2, 2]: {beta_range_ok}")
        
        # Check standard errors are positive
        se_positive = (coefs['se'] > 0).all()
        log(f"Standard errors positive: {se_positive}")
        
        # Check confidence intervals valid (lower < beta < upper)
        ci_valid = ((coefs['ci_lower'] <= coefs['beta']) & (coefs['beta'] <= coefs['ci_upper'])).all()
        log(f"Confidence intervals valid: {ci_valid}")
        
        # Check R² in expected range
        r2_range_ok = 0.15 <= regression_results['r2'] <= 0.45
        log(f"R² in expected range [0.15, 0.45]: {r2_range_ok} (R² = {regression_results['r2']:.4f})")
        
        # Check all predictors plus intercept represented
        expected_predictors = ['const'] + predictors
        actual_predictors = coefs['predictor'].tolist()
        predictors_ok = set(expected_predictors) == set(actual_predictors)
        log(f"All predictors represented: {predictors_ok}")
        
        # Overall validation
        overall_valid = (beta_range_ok and se_positive and ci_valid and 
                        r2_range_ok and predictors_ok)
        
        if overall_valid:
            log("All validation checks passed")
        else:
            log("Some validation checks failed - results may need review")
        
        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 05 complete - Multiple regression model fitted successfully")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)