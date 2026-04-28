#!/usr/bin/env python3
"""hierarchical_regression: Test incremental prediction of cognitive tests beyond demographics in confidence"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_hierarchical_regression.log"


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
    
    Signature mismatch found: tools.analysis_regression.fit_hierarchical_regression
    - Expected: (X_blocks, y, block_names, add_constant) 
    - Actual: (X_blocks, y, block_names)
    
    Creating custom implementation .
    """
    if block_names is None:
        block_names = [f"Block_{i+1}" for i in range(len(X_blocks))]
    
    models = []
    results = []
    
    for i, (X_block, block_name) in enumerate(zip(X_blocks, block_names)):
        # Add constant if requested
        if add_constant:
            X_reg = sm.add_constant(X_block)
        else:
            X_reg = X_block
        
        # Fit OLS model
        model = sm.OLS(y, X_reg).fit()
        models.append(model)
        
        # Extract key statistics
        result = {
            'model': block_name,
            'R_squared': model.rsquared,
            'adj_R_squared': model.rsquared_adj,
            'F_stat': model.fvalue,
            'p_value': model.f_pvalue,
            'n_params': len(model.params),
            'n_obs': model.nobs,
            'aic': model.aic,
            'bic': model.bic
        }
        results.append(result)
    
    # Compute hierarchical F-test for model comparison
    if len(models) == 2:
        # Compare nested models
        model_1, model_2 = models[0], models[1]
        
        # Hierarchical F-test
        r2_change = model_2.rsquared - model_1.rsquared
        df_change = model_2.df_model - model_1.df_model
        df_error = model_2.df_resid
        
        if df_change > 0:
            f_change = (r2_change / df_change) / ((1 - model_2.rsquared) / df_error)
            p_change = 1 - stats.f.cdf(f_change, df_change, df_error)
        else:
            f_change = np.nan
            p_change = np.nan
        
        hierarchical_test = {
            'r2_change': r2_change,
            'f_change': f_change,
            'p_change': p_change,
            'df_change': df_change
        }
    else:
        hierarchical_test = None
    
    return {
        'models': models,
        'results': results,
        'hierarchical_test': hierarchical_test
    }

def bootstrap_r_squared_ci(X, y, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Bootstrap confidence intervals for R-squared."""
    np.random.seed(random_state)
    
    n_obs = len(y)
    r_squared_values = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_indices = resample(range(n_obs), n_samples=n_obs, random_state=None)
        X_boot = X.iloc[boot_indices] if hasattr(X, 'iloc') else X[boot_indices]
        y_boot = y.iloc[boot_indices] if hasattr(y, 'iloc') else y[boot_indices]
        
        # Fit model and extract R-squared
        try:
            X_reg = sm.add_constant(X_boot)
            model = sm.OLS(y_boot, X_reg).fit()
            r_squared_values.append(model.rsquared)
        except:
            # Skip if model fitting fails
            continue
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(r_squared_values, lower_percentile)
    ci_upper = np.percentile(r_squared_values, upper_percentile)
    
    return ci_lower, ci_upper

def compute_cohens_f2_custom(r_squared):
    """Compute Cohen's f² effect size: f² = R² / (1 - R²)"""
    if r_squared >= 1.0:
        return np.inf
    elif r_squared <= 0.0:
        return 0.0
    else:
        return r_squared / (1 - r_squared)

def validate_model_convergence_custom(models, r_squared_values):
    """
    Custom validation function.
    
    Signature mismatch: tools.validation.validate_model_convergence expects 'lmm_result'
    but we have regression results. Creating custom validation.
    """
    validation_results = {
        'models_converged': True,
        'r_squared_valid': True,
        'nested_models_valid': True,
        'issues': []
    }
    
    # Check model convergence
    for i, model in enumerate(models):
        if not hasattr(model, 'converged') or model.converged is False:
            validation_results['models_converged'] = False
            validation_results['issues'].append(f"Model {i+1} did not converge")
    
    # Check R² values are in valid range [0, 1]
    for i, r2 in enumerate(r_squared_values):
        if not (0 <= r2 <= 1):
            validation_results['r_squared_valid'] = False
            validation_results['issues'].append(f"Model {i+1} R² = {r2:.4f} outside valid range [0, 1]")
    
    # Check nested models (Model 2 R² >= Model 1 R²)
    if len(r_squared_values) == 2:
        if r_squared_values[1] < r_squared_values[0]:
            validation_results['nested_models_valid'] = False
            validation_results['issues'].append(f"Model 2 R² ({r_squared_values[1]:.4f}) < Model 1 R² ({r_squared_values[0]:.4f})")
    
    validation_results['all_passed'] = all([
        validation_results['models_converged'],
        validation_results['r_squared_valid'], 
        validation_results['nested_models_valid']
    ])
    
    return validation_results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Hierarchical Regression")
        # Load Input Data

        log("Loading analysis dataset...")
        input_path = RQ_DIR / "data" / "step04_analysis_dataset.csv"
        df = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['confidence_theta', 'age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        # Prepare Predictor Blocks
        # Block 1: Demographics only (age, sex, education)
        # Block 2: Demographics + Cognitive tests (age, sex, education, RAVLT_T, BVMT_T, RPM_T)
        
        log("Setting up hierarchical predictor blocks...")
        
        # Handle categorical sex variable
        df_analysis = df.copy()
        if df_analysis['sex'].dtype == 'object':
            # Convert to dummy variable (assuming 'M'/'F' or similar)
            df_analysis['sex_dummy'] = (df_analysis['sex'] == 'M').astype(int)
            log("Sex variable converted to dummy coding (M=1, F=0)")
        else:
            df_analysis['sex_dummy'] = df_analysis['sex']
            log("Sex variable already numeric")
        
        # Define predictor blocks
        block_1_predictors = ['age', 'sex_dummy', 'education']
        block_2_predictors = ['age', 'sex_dummy', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        
        X_block1 = df_analysis[block_1_predictors]
        X_block2 = df_analysis[block_2_predictors]
        y = df_analysis['confidence_theta']
        
        log(f"Block 1 (Demographics): {block_1_predictors}")
        log(f"Block 2 (Demographics + Cognitive): {block_2_predictors}")
        log(f"Outcome variable: confidence_theta (N={len(y)})")
        # Fit Hierarchical Regression
        # Fit both models and compute hierarchical comparison
        
        log("Running hierarchical regression...")
        
        # Use custom implementation due to signature mismatch
        hierarchical_results = fit_hierarchical_regression_custom(
            X_blocks=[X_block1, X_block2],
            y=y,
            block_names=['Demographics', 'Cognitive'],
            add_constant=True
        )
        
        models = hierarchical_results['models']
        results = hierarchical_results['results']
        hierarchical_test = hierarchical_results['hierarchical_test']
        
        log("Hierarchical regression models fitted")
        log(f"Model 1 R² = {results[0]['R_squared']:.4f}")
        log(f"Model 2 R² = {results[1]['R_squared']:.4f}")
        
        if hierarchical_test:
            log(f"ΔR² = {hierarchical_test['r2_change']:.4f}")
            log(f"F-change = {hierarchical_test['f_change']:.3f}")
            log(f"p-change = {hierarchical_test['p_change']:.6f}")
        # Bootstrap Confidence Intervals
        # Compute bootstrap 95% CIs for R² (1000 iterations, seed=42)
        
        log("Computing bootstrap confidence intervals (n=1000, seed=42)...")
        
        # Bootstrap CIs for both models
        ci_results = []
        for i, (X_block, block_name) in enumerate([(X_block1, 'Demographics'), (X_block2, 'Cognitive')]):
            ci_lower, ci_upper = bootstrap_r_squared_ci(
                X_block, y, 
                n_bootstrap=1000, 
                confidence_level=0.95, 
                random_state=42
            )
            ci_results.append((ci_lower, ci_upper))
            log(f"{block_name} model 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        # Compute Effect Sizes
        # Calculate Cohen's f² = R²/(1-R²) for both models
        
        log("Computing Cohen's f² effect sizes...")
        
        cohens_f2_values = []
        for i, result in enumerate(results):
            f2 = compute_cohens_f2_custom(result['R_squared'])
            cohens_f2_values.append(f2)
            log(f"{result['model']} Cohen's f² = {f2:.4f}")
        # Create Output DataFrame
        # Format results for output CSV
        
        output_data = []
        for i, result in enumerate(results):
            ci_lower, ci_upper = ci_results[i]
            
            output_row = {
                'model': result['model'],
                'R_squared': result['R_squared'],
                'adj_R_squared': result['adj_R_squared'],
                'F_stat': result['F_stat'],
                'p_value': result['p_value'],
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cohens_f2': cohens_f2_values[i]
            }
            output_data.append(output_row)
        
        output_df = pd.DataFrame(output_data)
        # Save Results
        # Save hierarchical regression comparison results
        
        output_path = RQ_DIR / "data" / "step05_hierarchical_models.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(output_df)} rows, {len(output_df.columns)} cols)")
        
        # Display summary
        log("Hierarchical regression results:")
        for _, row in output_df.iterrows():
            log(f"  {row['model']}: R²={row['R_squared']:.4f}, F={row['F_stat']:.2f}, p={row['p_value']:.6f}")
        # Run Validation
        # Validate model convergence and R² values
        
        log("Running model validation...")
        
        r_squared_values = [result['R_squared'] for result in results]
        validation_result = validate_model_convergence_custom(models, r_squared_values)
        
        # Report validation results
        if validation_result['all_passed']:
            log("All checks PASSED")
            log(f"Models converged: {validation_result['models_converged']}")
            log(f"R² values valid: {validation_result['r_squared_valid']}")
            log(f"Nested models valid: {validation_result['nested_models_valid']}")
        else:
            log("Some checks FAILED:")
            for issue in validation_result['issues']:
                log(f"Issue: {issue}")

        log("Step 05: Hierarchical regression complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)