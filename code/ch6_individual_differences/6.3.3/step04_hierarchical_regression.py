#!/usr/bin/env python3
"""hierarchical_regression: Fit hierarchical regression testing incremental validity of cognitive predictors"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_hierarchical_regression

from tools.analysis_extensions import validate_regression_assumptions

# Import bootstrap for confidence intervals
from tools.analysis_regression import bootstrap_regression_ci

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_hierarchical_regression.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Hierarchical Regression")
        # Load Analysis Dataset

        log("Loading analysis dataset...")
        analysis_df = pd.read_csv(RQ_DIR / "data" / "step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")
        
        # Check for required columns
        required_cols = ['hce_rate', 'age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        missing_cols = [col for col in required_cols if col not in analysis_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        log(f"HCE rate range: {analysis_df['hce_rate'].min():.3f} to {analysis_df['hce_rate'].max():.3f}")
        # Prepare Hierarchical Regression Data
        # Model 1: Demographics only (age_c, sex, education)
        # Model 2: Demographics + Cognitive (add ravlt_c, bvmt_c, rpm_c)

        log("Setting up hierarchical regression blocks...")
        
        # Outcome variable
        y = analysis_df['hce_rate'].values
        
        # Block 1: Demographics
        demographics_predictors = ['age_c', 'sex', 'education']
        X_demographics = analysis_df[demographics_predictors].values
        
        # Block 2: Full model (Demographics + Cognitive)
        full_predictors = ['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        X_full = analysis_df[full_predictors].values
        
        # Prepare blocks for hierarchical regression
        X_blocks = [X_demographics, X_full]
        block_names = ['Demographics', 'Full_Model']
        
        log(f"Demographics block: {X_demographics.shape[1]} predictors")
        log(f"Full model block: {X_full.shape[1]} predictors")
        log(f"Sample size: N = {len(y)}")
        # Run Hierarchical Regression

        log("Running fit_hierarchical_regression...")
        hierarchical_results = fit_hierarchical_regression(
            X_blocks=X_blocks,
            y=y,
            block_names=block_names
        )
        log("Hierarchical regression complete")
        
        # Extract key results
        models = hierarchical_results['models']
        delta_r2 = hierarchical_results['delta_r2'] 
        f_tests = hierarchical_results['f_tests']
        cumulative_r2 = hierarchical_results['cumulative_r2']
        
        log(f"Demographics model R² = {cumulative_r2[0]:.4f}")
        log(f"Full model R² = {cumulative_r2[1]:.4f}")
        log(f"Incremental R² (cognitive) = {delta_r2['Full_Model']:.4f}")
        log(f"F-change = {f_tests['Full_Model']['f_statistic']:.3f}, p = {f_tests['Full_Model']['p_value']:.6f}")
        # Extract Regression Coefficients with Dual P-Values
        # Extract coefficients from both models
        # Apply multiple comparison corrections (Bonferroni, FDR)
        # Generate bootstrap confidence intervals

        log("Extracting coefficients with dual p-value reporting...")
        
        results_list = []
        
        # Model 1: Demographics only
        model_demo = models[0]
        demo_predictors_with_intercept = ['intercept'] + demographics_predictors
        
        demo_ci = model_demo.conf_int()  # Get confidence intervals as array
        for i, predictor in enumerate(demo_predictors_with_intercept):
            results_list.append({
                'model': 'Demographics',
                'predictor': predictor,
                'beta': model_demo.params[i],
                'se': model_demo.bse[i],
                't_stat': model_demo.tvalues[i],
                'p_uncorrected': model_demo.pvalues[i],
                'ci_lower': demo_ci[i, 0],  # Access array directly
                'ci_upper': demo_ci[i, 1]   # Access array directly
            })
        
        # Model 2: Full model  
        model_full = models[1]
        full_predictors_with_intercept = ['intercept'] + full_predictors
        full_ci = model_full.conf_int()  # Get confidence intervals as array
        
        for i, predictor in enumerate(full_predictors_with_intercept):
            results_list.append({
                'model': 'Full_Model',
                'predictor': predictor,
                'beta': model_full.params[i],
                'se': model_full.bse[i],
                't_stat': model_full.tvalues[i],
                'p_uncorrected': model_full.pvalues[i],
                'ci_lower': full_ci[i, 0],  # Access array directly
                'ci_upper': full_ci[i, 1]   # Access array directly
            })
        
        regression_results = pd.DataFrame(results_list)
        
        # Apply multiple comparison corrections
        # Bonferroni: α = 0.000448 (within-RQ correction from 4_analysis.yaml)
        alpha_bonferroni = 0.000448
        alpha_family = 0.05
        
        log("Applying multiple comparison corrections...")
        
        # Bonferroni correction (conservative)
        n_tests = len(regression_results)
        regression_results['p_bonferroni'] = np.minimum(
            regression_results['p_uncorrected'] * n_tests, 1.0
        )
        
        # FDR correction (Benjamini-Hochberg)
        _, p_fdr, _, _ = multipletests(
            regression_results['p_uncorrected'], 
            alpha=alpha_family, 
            method='fdr_bh'
        )
        regression_results['p_fdr'] = p_fdr
        
        log(f"Bonferroni correction (α = {alpha_bonferroni})")
        log(f"FDR correction (α = {alpha_family})")
        # Generate Bootstrap Confidence Intervals
        # Bootstrap CIs for robustness (handles non-normality)
        # 1000 iterations with seed=42 for reproducibility

        log("Computing bootstrap confidence intervals...")
        
        try:
            # Bootstrap for full model (most important)
            bootstrap_ci_results = bootstrap_regression_ci(
                X=X_full,
                y=y,
                n_bootstrap=1000,
                confidence_level=0.95,
                random_state=42
            )
            
            # Update Full_Model rows with bootstrap CIs
            full_model_mask = regression_results['model'] == 'Full_Model'
            bootstrap_predictors = ['intercept'] + full_predictors
            
            for i, predictor in enumerate(bootstrap_predictors):
                mask = full_model_mask & (regression_results['predictor'] == predictor)
                if mask.any():
                    idx = regression_results[mask].index[0]
                    regression_results.loc[idx, 'ci_lower'] = bootstrap_ci_results.loc[i, 'ci_lower']
                    regression_results.loc[idx, 'ci_upper'] = bootstrap_ci_results.loc[i, 'ci_upper']
            
            log("Updated Full_Model confidence intervals with bootstrap estimates")
        
        except Exception as e:
            log(f"Bootstrap CI computation failed: {e}")
            log("Using OLS confidence intervals for all models")
        # Create Model Comparison Table
        # Compare models by R², AIC, F-change tests
        # Essential for interpreting incremental validity

        log("Creating model comparison table...")
        
        comparison_list = []
        
        # Demographics model
        comparison_list.append({
            'model': 'Demographics',
            'R2': cumulative_r2[0],
            'R2_adj': models[0].rsquared_adj,
            'F_stat': models[0].fvalue,
            'p_value': models[0].f_pvalue,
            'AIC': models[0].aic,
            'delta_R2': delta_r2['Demographics'],
            'F_change': f_tests['Demographics']['f_statistic']
        })
        
        # Full model
        comparison_list.append({
            'model': 'Full_Model', 
            'R2': cumulative_r2[1],
            'R2_adj': models[1].rsquared_adj,
            'F_stat': models[1].fvalue,
            'p_value': models[1].f_pvalue,
            'AIC': models[1].aic,
            'delta_R2': delta_r2['Full_Model'],
            'F_change': f_tests['Full_Model']['f_statistic']
        })
        
        model_comparison = pd.DataFrame(comparison_list)
        log("Model comparison table")
        # Save Results
        # Save regression coefficients and model comparison
        # These outputs will be used by downstream diagnostic and effect size steps

        log("Saving regression results...")
        
        # Save regression results
        regression_output_path = RQ_DIR / "data" / "step04_regression_results.csv"
        regression_results.to_csv(regression_output_path, index=False, encoding='utf-8')
        log(f"step04_regression_results.csv ({len(regression_results)} rows, {len(regression_results.columns)} cols)")
        
        # Save model comparison
        comparison_output_path = RQ_DIR / "data" / "step04_model_comparison.csv" 
        model_comparison.to_csv(comparison_output_path, index=False, encoding='utf-8')
        log(f"step04_model_comparison.csv ({len(model_comparison)} rows, {len(model_comparison.columns)} cols)")
        
        # Log key findings
        log("[KEY FINDINGS]")
        log(f"  Demographics R² = {cumulative_r2[0]:.4f}")
        log(f"  Full model R² = {cumulative_r2[1]:.4f}")
        log(f"  Incremental R² (cognitive) = {delta_r2['Full_Model']:.4f}")
        log(f"  F-change = {f_tests['Full_Model']['f_statistic']:.3f}")
        log(f"  p-value (uncorrected) = {f_tests['Full_Model']['p_value']:.6f}")
        log(f"  p-value significant at α = {alpha_bonferroni}? {f_tests['Full_Model']['p_value'] < alpha_bonferroni}")
        # Run Validation
        # Validates: Residual normality, homoscedasticity, etc.
        # Threshold: α = 0.05 for assumption tests

        log("Running validate_regression_assumptions...")
        
        # Use full model for assumption testing
        residuals = models[1].resid
        X_for_validation = X_full
        
        validation_result = validate_regression_assumptions(
            residuals=residuals,
            X=X_for_validation,
            significance_level=0.05
        )
        
        # Report validation results
        if isinstance(validation_result, dict):
            for assumption, results in validation_result.items():
                if isinstance(results, dict):
                    status = "PASS" if results.get('assumption_met', True) else "FAIL"
                    test_stat = results.get('test_statistic', 'N/A')
                    p_val = results.get('p_value', 'N/A')
                    log(f"{assumption}: {status} (stat={test_stat}, p={p_val})")
                else:
                    log(f"{assumption}: {results}")
        else:
            log(f"{validation_result}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)