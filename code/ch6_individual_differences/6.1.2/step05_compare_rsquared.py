#!/usr/bin/env python3
"""Compare R² Values with Bootstrap: Bootstrap comparison of R² values: intercept vs slope prediction"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.api as sm
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_compare_rsquared.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Bootstrap R² Function (Due to Signature Mismatch)

def bootstrap_r_squared_comparison(X, y_intercept, y_slope, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """
    Custom bootstrap function for R² comparison.
    
    Created due to signature mismatch between 4_analysis.yaml specification and
    tools.analysis_regression.bootstrap_regression_ci actual implementation.
    
    Parameters:
    - X: Predictors (RAVLT_T, BVMT_T, RPM_T)
    - y_intercept: LMM random intercepts 
    - y_slope: LMM random slopes
    - n_bootstrap: Number of bootstrap iterations
    - confidence_level: Confidence level for CIs
    - random_state: Random seed for reproducibility
    
    Returns:
    - Dictionary with R² values, CIs, and comparison statistics
    """
    np.random.seed(random_state)
    
    # Add constant for regression
    X_with_const = sm.add_constant(X)
    
    # Original model fits
    model_intercept = sm.OLS(y_intercept, X_with_const).fit()
    model_slope = sm.OLS(y_slope, X_with_const).fit()
    
    # Original R² values
    r2_intercept_orig = model_intercept.rsquared
    r2_slope_orig = model_slope.rsquared
    r2_adj_intercept_orig = model_intercept.rsquared_adj
    r2_adj_slope_orig = model_slope.rsquared_adj
    
    # Bootstrap arrays
    r2_intercept_boot = []
    r2_slope_boot = []
    r2_diff_boot = []
    
    n_samples = len(X)
    
    log(f"Starting {n_bootstrap} bootstrap iterations...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            log(f"Completed {i + 1}/{n_bootstrap} iterations")
            
        # Bootstrap sample
        indices = resample(range(n_samples), n_samples=n_samples, random_state=random_state+i)
        X_boot = X.iloc[indices]
        y_intercept_boot = y_intercept.iloc[indices]
        y_slope_boot = y_slope.iloc[indices]
        
        # Add constant for bootstrap sample
        X_boot_const = sm.add_constant(X_boot)
        
        try:
            # Fit models on bootstrap sample
            model_int_boot = sm.OLS(y_intercept_boot, X_boot_const).fit()
            model_slope_boot = sm.OLS(y_slope_boot, X_boot_const).fit()
            
            # Store R² values
            r2_int = model_int_boot.rsquared
            r2_slope = model_slope_boot.rsquared
            
            r2_intercept_boot.append(r2_int)
            r2_slope_boot.append(r2_slope)
            r2_diff_boot.append(r2_int - r2_slope)
            
        except Exception as e:
            log(f"Bootstrap iteration {i} failed: {e}")
            continue
    
    log(f"Completed {len(r2_intercept_boot)} successful iterations")
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower_pct = (alpha/2) * 100
    ci_upper_pct = (1 - alpha/2) * 100
    
    r2_int_ci_lower = np.percentile(r2_intercept_boot, ci_lower_pct)
    r2_int_ci_upper = np.percentile(r2_intercept_boot, ci_upper_pct)
    
    r2_slope_ci_lower = np.percentile(r2_slope_boot, ci_lower_pct)
    r2_slope_ci_upper = np.percentile(r2_slope_boot, ci_upper_pct)
    
    r2_diff_ci_lower = np.percentile(r2_diff_boot, ci_lower_pct)
    r2_diff_ci_upper = np.percentile(r2_diff_boot, ci_upper_pct)
    
    # One-tailed test: is intercept model R² significantly higher than slope model R²?
    p_value = np.mean(np.array(r2_diff_boot) <= 0)  # Proportion of bootstrap differences <= 0
    
    return {
        'r2_intercept_orig': r2_intercept_orig,
        'r2_slope_orig': r2_slope_orig,
        'r2_adj_intercept_orig': r2_adj_intercept_orig,
        'r2_adj_slope_orig': r2_adj_slope_orig,
        'r2_intercept_ci_lower': r2_int_ci_lower,
        'r2_intercept_ci_upper': r2_int_ci_upper,
        'r2_slope_ci_lower': r2_slope_ci_lower,
        'r2_slope_ci_upper': r2_slope_ci_upper,
        'r2_difference': r2_intercept_orig - r2_slope_orig,
        'r2_diff_ci_lower': r2_diff_ci_lower,
        'r2_diff_ci_upper': r2_diff_ci_upper,
        'p_value': p_value,
        'n_bootstrap_success': len(r2_intercept_boot)
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Compare R² Values with Bootstrap")
        # Load Input Data
        
        log("Loading input data...")
        
        # Load regression input from step02
        # Expected columns: UID, intercept, slope, se_intercept, se_slope, RAVLT_T, BVMT_T, RPM_T
        # Expected rows: ~100
        regression_input = pd.read_csv(RQ_DIR / "data" / "step02_regression_input.csv")
        log(f"step02_regression_input.csv ({len(regression_input)} rows, {len(regression_input.columns)} cols)")
        
        # Load intercept model results from step03
        # Expected columns: predictor, beta, se, t, p_uncorrected, p_bonferroni, CI_lower, CI_upper
        # Expected rows: ~4
        intercept_results = pd.read_csv(RQ_DIR / "data" / "step03_intercept_predictions.csv")
        log(f"step03_intercept_predictions.csv ({len(intercept_results)} rows, {len(intercept_results.columns)} cols)")
        
        # Load slope model results from step04
        # Expected columns: predictor, beta, se, t, p_uncorrected, p_bonferroni, CI_lower, CI_upper, VIF
        # Expected rows: ~4
        slope_results = pd.read_csv(RQ_DIR / "data" / "step04_slope_predictions.csv")
        log(f"step04_slope_predictions.csv ({len(slope_results)} rows, {len(slope_results.columns)} cols)")
        # Run Bootstrap R² Comparison
        
        log("Running custom bootstrap R² comparison...")
        
        # Extract predictor variables and outcomes
        X = regression_input[['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']]  # Cognitive test T-scores
        y_intercept = regression_input['intercept']  # LMM random intercepts (baseline ability)
        y_slope = regression_input['slope']  # LMM random slopes (forgetting rate)
        
        # Run bootstrap comparison with specification parameters
        bootstrap_results = bootstrap_r_squared_comparison(
            X=X,
            y_intercept=y_intercept,
            y_slope=y_slope,
            n_bootstrap=1000,  # Number of bootstrap iterations
            confidence_level=0.95,  # 95% confidence intervals
            random_state=42  # Reproducible results
        )
        log("Bootstrap analysis complete")
        # Create Results DataFrame
        # These outputs will be used by: Step 6 for predictor significance testing
        
        log("Creating results DataFrame...")
        
        # Create output DataFrame matching 4_analysis.yaml specification
        # Columns: model, r_squared, adj_r_squared, bootstrap_ci_lower, bootstrap_ci_upper, 
        #         difference, difference_ci_lower, difference_ci_upper, p_value
        # Row count: 3 (intercept model, slope model, difference test)
        
        r_squared_comparison = pd.DataFrame([
            {
                'model': 'intercept',
                'r_squared': bootstrap_results['r2_intercept_orig'],
                'adj_r_squared': bootstrap_results['r2_adj_intercept_orig'],
                'bootstrap_ci_lower': bootstrap_results['r2_intercept_ci_lower'],
                'bootstrap_ci_upper': bootstrap_results['r2_intercept_ci_upper'],
                'difference': np.nan,  # Not applicable for individual models
                'difference_ci_lower': np.nan,
                'difference_ci_upper': np.nan,
                'p_value': np.nan
            },
            {
                'model': 'slope',
                'r_squared': bootstrap_results['r2_slope_orig'],
                'adj_r_squared': bootstrap_results['r2_adj_slope_orig'],
                'bootstrap_ci_lower': bootstrap_results['r2_slope_ci_lower'],
                'bootstrap_ci_upper': bootstrap_results['r2_slope_ci_upper'],
                'difference': np.nan,  # Not applicable for individual models
                'difference_ci_lower': np.nan,
                'difference_ci_upper': np.nan,
                'p_value': np.nan
            },
            {
                'model': 'difference',
                'r_squared': bootstrap_results['r2_difference'],
                'adj_r_squared': np.nan,  # Not applicable for difference
                'bootstrap_ci_lower': np.nan,  # Not applicable for difference
                'bootstrap_ci_upper': np.nan,
                'difference': bootstrap_results['r2_difference'],
                'difference_ci_lower': bootstrap_results['r2_diff_ci_lower'],
                'difference_ci_upper': bootstrap_results['r2_diff_ci_upper'],
                'p_value': bootstrap_results['p_value']
            }
        ])
        
        # Save results to CSV
        # Output: step05_r_squared_comparison.csv
        # Contains: Bootstrap comparison of R² values between intercept and slope models
        # Columns: model, r_squared, adj_r_squared, bootstrap_ci_lower, bootstrap_ci_upper, difference, difference_ci_lower, difference_ci_upper, p_value
        output_path = RQ_DIR / "data" / "step05_r_squared_comparison.csv"
        r_squared_comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step05_r_squared_comparison.csv ({len(r_squared_comparison)} rows, {len(r_squared_comparison.columns)} cols)")
        
        # Report results
        log(f"Intercept model R² = {bootstrap_results['r2_intercept_orig']:.4f} (95% CI: {bootstrap_results['r2_intercept_ci_lower']:.4f}-{bootstrap_results['r2_intercept_ci_upper']:.4f})")
        log(f"Slope model R² = {bootstrap_results['r2_slope_orig']:.4f} (95% CI: {bootstrap_results['r2_slope_ci_lower']:.4f}-{bootstrap_results['r2_slope_ci_upper']:.4f})")
        log(f"R² difference = {bootstrap_results['r2_difference']:.4f} (95% CI: {bootstrap_results['r2_diff_ci_lower']:.4f}-{bootstrap_results['r2_diff_ci_upper']:.4f})")
        log(f"One-tailed p-value = {bootstrap_results['p_value']:.4f} (H1: intercept R² > slope R²)")
        log(f"Bootstrap iterations successful: {bootstrap_results['n_bootstrap_success']}/1000")
        # Run Validation Tool
        # Validates: R² values are in valid range [0, 1]
        # Threshold: min_val=0.0, max_val=1.0
        
        log("Running validate_numeric_range...")
        validation_result = validate_numeric_range(
            data=r_squared_comparison['r_squared'].dropna(),  # Remove NaN values for difference row
            min_val=0.0,  # R² minimum value
            max_val=1.0,  # R² maximum value
            column_name="r_squared"  # Column being validated
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
            
        # Additional validation checks
        log("Checking bootstrap CI validity...")
        
        # Check that bootstrap CIs are well-formed (lower < upper)
        for _, row in r_squared_comparison.iterrows():
            if not pd.isna(row['bootstrap_ci_lower']) and not pd.isna(row['bootstrap_ci_upper']):
                if row['bootstrap_ci_lower'] >= row['bootstrap_ci_upper']:
                    raise ValueError(f"Bootstrap CI malformed for {row['model']}: lower ({row['bootstrap_ci_lower']}) >= upper ({row['bootstrap_ci_upper']})")
                    
        # Check that p-value is in valid range
        p_val = bootstrap_results['p_value']
        if not (0.0 <= p_val <= 1.0):
            raise ValueError(f"P-value out of range: {p_val} (expected [0, 1])")
            
        # Check that all 3 rows are present
        expected_models = {'intercept', 'slope', 'difference'}
        actual_models = set(r_squared_comparison['model'])
        if actual_models != expected_models:
            raise ValueError(f"Missing models: expected {expected_models}, got {actual_models}")
            
        log("All checks passed!")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)