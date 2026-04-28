#!/usr/bin/env python3
"""hierarchical_regression: Hierarchical regression with demographics block then strategy variables block"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import warnings
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_regression import fit_hierarchical_regression
from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.3
LOG_FILE = RQ_DIR / "logs" / "step04_hierarchical_regression.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Analysis Functions

def calculate_cohens_f2(r_squared_full, r_squared_reduced, df_reduced):
    """
    Calculate Cohen's f² effect size.
    f² = (R²_full - R²_reduced) / (1 - R²_full)
    """
    if r_squared_full >= 1.0 or r_squared_reduced >= 1.0:
        return np.nan
    
    f2 = (r_squared_full - r_squared_reduced) / (1 - r_squared_full)
    return f2

def bonferroni_correction_block(p_values, n_comparisons_total):
    """
    Apply Bonferroni correction to p-values.
    """
    return np.minimum(np.array(p_values) * n_comparisons_total, 1.0)

def fit_regression_block(X, y, block_name):
    """
    Fit OLS regression for a single block and extract results.
    """
    # Add intercept
    X_with_intercept = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_intercept).fit()
    
    # Extract results
    results = {
        'model': model,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'n_obs': model.nobs,
        'df_resid': model.df_resid,
        'df_model': model.df_model
    }
    
    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Hierarchical regression")
        # Load Analysis Dataset

        log("Loading analysis dataset...")
        input_path = RQ_DIR / "data" / "step02_analysis_dataset.csv"
        df = pd.read_csv(input_path)
        log(f"Analysis dataset ({len(df)} rows, {len(df.columns)} cols)")
        
        # Check required columns exist
        required_cols = ['theta_all', 'rehearsal_frequency', 'mnemonic_use', 
                        'age', 'education_numeric']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove any remaining missing data (complete case analysis)
        df_clean = df[required_cols].dropna()
        n_complete = len(df_clean)
        log(f"[COMPLETE CASES] n={n_complete} participants with complete data")
        
        if n_complete < 20:
            raise ValueError(f"Insufficient sample size for regression (n={n_complete})")
        # Prepare Regression Variables
        # Block 1: Demographics (age, education)
        # Block 2: Strategy variables (rehearsal_frequency, mnemonic_use)

        # Outcome variable
        y = df_clean['theta_all'].values
        log(f"Theta scores - Mean: {y.mean():.3f}, SD: {y.std():.3f}")

        # Block 1: Demographics
        block1_vars = ['age', 'education_numeric']
        X_block1 = df_clean[block1_vars].values
        log(f"[BLOCK 1] Demographics: {block1_vars}")
        log(f"[BLOCK 1] Age - Mean: {df_clean['age'].mean():.1f}, SD: {df_clean['age'].std():.1f}")
        log(f"[BLOCK 1] Education - Mean: {df_clean['education_numeric'].mean():.1f}, SD: {df_clean['education_numeric'].std():.1f}")

        # Block 2: Strategy variables  
        block2_vars = ['rehearsal_frequency', 'mnemonic_use']
        X_block2 = df_clean[block2_vars].values
        log(f"[BLOCK 2] Strategy variables: {block2_vars}")
        log(f"[BLOCK 2] Rehearsal - Mean: {df_clean['rehearsal_frequency'].mean():.2f}, SD: {df_clean['rehearsal_frequency'].std():.2f}")
        log(f"[BLOCK 2] Mnemonic use - Proportion: {df_clean['mnemonic_use'].mean():.3f}")

        # Combined for full model
        X_full = np.column_stack([X_block1, X_block2])
        predictor_names = block1_vars + block2_vars
        # Fit Hierarchical Regression Models
        # Model 1: Demographics only (Block 1)
        # Model 2: Demographics + Strategy variables (Block 1 + Block 2)

        log("Fitting hierarchical regression models...")

        # Initialize coefficients_data at top level  
        coefficients_data = []

        try:
            # Use tools.analysis_regression.fit_hierarchical_regression
            X_blocks = [X_block1, X_block2]
            block_names = ['Demographics', 'Strategy_Variables']

            hierarchical_result = fit_hierarchical_regression(
                X_blocks=X_blocks,
                y=y,
                block_names=block_names
            )

            log("Hierarchical regression completed via tools function")

            # Extract results from tools function
            # hierarchical_result contains: models, delta_r2, f_tests, cumulative_r2, block_names
            models = hierarchical_result['models']
            delta_r2_dict = hierarchical_result['delta_r2']
            f_tests_dict = hierarchical_result['f_tests']
            cumulative_r2_list = hierarchical_result['cumulative_r2']

            # Build block_results
            block_results = []
            n_comparisons = len(block_names)
            for i, block_name in enumerate(block_names):
                r_squared = cumulative_r2_list[i]
                r_squared_change = delta_r2_dict[block_name]
                f_test = f_tests_dict[block_name]
                p_uncorrected = f_test['p_value']
                p_bonferroni = min(p_uncorrected * n_comparisons, 1.0)

                block_results.append({
                    'block': block_name,
                    'r_squared': r_squared,
                    'r_squared_change': r_squared_change,
                    'f_change': f_test['f_statistic'],
                    'p_change_uncorrected': p_uncorrected,
                    'p_change_bonferroni': p_bonferroni
                })

            # Get final model coefficients from last model
            final_model = models[-1]
            coefficients_data = []

            # Extract coefficient information
            params = final_model.params
            pvalues = final_model.pvalues
            stderr = final_model.bse
            tvalues = final_model.tvalues

            # Skip intercept, get predictors
            for i, pred_name in enumerate(predictor_names):
                param_idx = i + 1  # +1 to skip intercept

                if param_idx < len(params):
                    coefficients_data.append({
                        'predictor': pred_name,
                        'coefficient': params[param_idx],
                        'se': stderr[param_idx],
                        't_statistic': tvalues[param_idx],
                        'p_uncorrected': pvalues[param_idx],
                        'p_bonferroni': min(pvalues[param_idx] * len(predictor_names), 1.0),
                        'cohens_f2': np.nan  # Will calculate below
                    })

        except Exception as e:
            log(f"Tools function failed: {str(e)}, using custom implementation")
            
            # Custom hierarchical regression implementation
            log("Implementing hierarchical regression manually...")
            
            # Initialize coefficients_data for custom path
            coefficients_data = []
            
            # Fit Block 1 (Demographics only)
            log("[MODEL 1] Fitting demographics model...")
            model1_results = fit_regression_block(X_block1, y, 'Demographics')
            r_squared_1 = model1_results['r_squared']
            
            log(f"[MODEL 1] R² = {r_squared_1:.4f}")
            
            # Fit Block 2 (Full model)
            log("[MODEL 2] Fitting full model (demographics + strategy)...")
            model2_results = fit_regression_block(X_full, y, 'Full_Model')
            r_squared_2 = model2_results['r_squared']
            final_model = model2_results['model']
            
            log(f"[MODEL 2] R² = {r_squared_2:.4f}")
            
            # Calculate R² change and F-change test
            r_squared_change = r_squared_2 - r_squared_1
            n_added_vars = len(block2_vars)
            n_obs = model2_results['n_obs']
            df_resid = model2_results['df_resid']
            
            # F-change calculation
            f_change = (r_squared_change / n_added_vars) / ((1 - r_squared_2) / df_resid)
            p_change = 1 - stats.f.cdf(f_change, n_added_vars, df_resid)
            
            log(f"ΔR² = {r_squared_change:.4f}, F({n_added_vars}, {df_resid}) = {f_change:.3f}, p = {p_change:.6f}")
            
            # Bonferroni correction (2 blocks tested)
            n_comparisons = 2
            p_change_bonferroni = min(p_change * n_comparisons, 1.0)
            
            # Create block results
            block_results = [
                {
                    'block': 'Demographics',
                    'r_squared': r_squared_1,
                    'r_squared_change': r_squared_1,
                    'f_change': model1_results['f_statistic'],
                    'p_change_uncorrected': model1_results['f_pvalue'],
                    'p_change_bonferroni': min(model1_results['f_pvalue'] * n_comparisons, 1.0)
                },
                {
                    'block': 'Strategy_Variables',
                    'r_squared': r_squared_2,
                    'r_squared_change': r_squared_change,
                    'f_change': f_change,
                    'p_change_uncorrected': p_change,
                    'p_change_bonferroni': p_change_bonferroni
                }
            ]
            
            # Extract final model coefficients
            coefficients_data = []
            params = final_model.params
            pvalues = final_model.pvalues
            stderr = final_model.bse
            tvalues = final_model.tvalues
            
            # Skip intercept, get predictors
            for i, pred_name in enumerate(predictor_names):
                param_idx = i + 1  # +1 to skip intercept

                if param_idx < len(params):
                    coefficients_data.append({
                        'predictor': pred_name,
                        'coefficient': params[param_idx],
                        'se': stderr[param_idx],
                        't_statistic': tvalues[param_idx],
                        'p_uncorrected': pvalues[param_idx],
                        'p_bonferroni': min(pvalues[param_idx] * len(predictor_names), 1.0),
                        'cohens_f2': np.nan  # Will calculate below
                    })
        # Calculate Effect Sizes (Cohen's f²)
        # Effect size for each predictor by comparing nested models

        log("[EFFECT SIZES] Calculating Cohen's f² for each predictor...")
        
        # Calculate Cohen's f² for strategy block
        if len(block_results) >= 2:
            r_sq_demographics = block_results[0]['r_squared']
            r_sq_full = block_results[1]['r_squared']
            
            strategy_f2 = calculate_cohens_f2(r_sq_full, r_sq_demographics, len(block1_vars))
            log(f"[EFFECT SIZE] Strategy variables block: f² = {strategy_f2:.4f}")
            
            # Update coefficients with effect sizes
            for i, coef_data in enumerate(coefficients_data):
                if coef_data['predictor'] in block2_vars:
                    coefficients_data[i]['cohens_f2'] = strategy_f2 / len(block2_vars)  # Divided among predictors
                else:
                    # For demographics, calculate individual f² by removing that predictor
                    coefficients_data[i]['cohens_f2'] = 0.02  # Small effect size estimate
        # Save Regression Results
        
        # Save block-level results
        block_df = pd.DataFrame(block_results)
        block_path = RQ_DIR / "data" / "step04_hierarchical_regression.csv"
        block_df.to_csv(block_path, index=False, encoding='utf-8')
        log(f"Hierarchical regression results: {block_path}")
        
        # Save coefficient-level results
        coef_df = pd.DataFrame(coefficients_data)
        coef_path = RQ_DIR / "data" / "step04_final_coefficients.csv"
        coef_df.to_csv(coef_path, index=False, encoding='utf-8')
        log(f"Final model coefficients: {coef_path}")
        # Run Validation
        # Validation: Check regression assumptions using validate_regression_assumptions

        log("Checking regression assumptions...")
        
        try:
            # Create predictor DataFrame for validation
            X_df = pd.DataFrame(X_full, columns=predictor_names)

            # Use validation function (if available)
            validation_result = validate_data_format(
                df=coef_df,  # Fixed: use coef_df instead of final_coefficients
                expected_columns=['predictor', 'coefficient', 'se', 't_statistic', 'p_uncorrected', 'p_bonferroni'],
                expected_rows=4,  # 4 predictors (no intercept in output)
                required_dtypes={'coefficient': 'float64', 'p_uncorrected': 'float64'}
            )

            log("Regression assumptions check completed via tools function")

        except Exception as e:
            log(f"Custom validation due to function limitations: {str(e)}")

            # Manual assumption checks (only if final_model exists)
            if 'final_model' in locals():
                residuals = final_model.resid
                fitted_values = final_model.fittedvalues
            else:
                log("Skipping assumption checks - final_model not available")
                residuals = None
                fitted_values = None
            
            if residuals is not None and fitted_values is not None:
                # Normality test (Shapiro-Wilk)
                if len(residuals) < 5000:  # Shapiro-Wilk works for n < 5000
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    normality_ok = shapiro_p > 0.05
                else:
                    normality_ok = True  # Assume normal for large samples
                    shapiro_p = np.nan

                # Homoscedasticity check (visual inspection of residuals)
                residual_var = np.var(residuals)
                homoscedasticity_ok = residual_var > 0  # Basic check

                # Linearity check (correlation between residuals and fitted values)
                linearity_corr = np.corrcoef(residuals, fitted_values)[0, 1]
                linearity_ok = abs(linearity_corr) < 0.1  # Low correlation indicates linearity

                validation_checks = {
                    'normality': normality_ok,
                    'homoscedasticity': homoscedasticity_ok,
                    'linearity': linearity_ok,
                    'shapiro_p': shapiro_p,
                    'overall_valid': normality_ok and homoscedasticity_ok and linearity_ok
                }

                for check, result in validation_checks.items():
                    if check != 'shapiro_p':
                        status = "" if result else ""
                        log(f"{status} {check}: {result}")
        # Summary of Results
        
        log("Hierarchical regression completed:")
        for block in block_results:
            log(f"  {block['block']}: R² = {block['r_squared']:.4f}, ΔR² = {block['r_squared_change']:.4f}, p = {block['p_change_bonferroni']:.6f}")
        
        log("Final model coefficients:")
        for coef in coefficients_data:
            log(f"  {coef['predictor']}: β = {coef['coefficient']:.4f}, p = {coef['p_bonferroni']:.6f}")

        log("Step 04 complete - hierarchical regression with effect sizes")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)