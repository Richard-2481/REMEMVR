#!/usr/bin/env python3
"""hierarchical_regression: Fit hierarchical regression models with bootstrap CIs and dual p-values. Block 1 contains"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_hierarchical_regression, bootstrap_regression_ci

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
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
        # Load Input Data

        log("Loading analysis dataset...")
        input_path = RQ_DIR / "data/step03_analysis_dataset.csv"
        analysis_df = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")

        # Verify expected columns
        expected_cols = ['UID', 'theta_all', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z', 'Age_z']
        missing_cols = [col for col in expected_cols if col not in analysis_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"All expected columns present: {expected_cols}")
        # Prepare Data for Hierarchical Regression
        # Block 1: Age_z only (control for demographic effects)
        # Block 2: Age_z + Education_z + VR_Experience_z + Typical_Sleep_z (full model)

        log("Preparing predictor blocks...")
        
        # Define predictor blocks
        block1_predictors = ['Age_z']
        block2_predictors = ['Age_z', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z']
        
        # Extract predictor matrices
        X_block1 = analysis_df[block1_predictors].values
        X_block2 = analysis_df[block2_predictors].values
        X_blocks = [X_block1, X_block2]
        
        # Extract outcome variable
        y = analysis_df['theta_all'].values
        
        # Block names for output
        block_names = ['Control', 'Full']
        
        log(f"Block 1 (Control): {block1_predictors}")
        log(f"Block 2 (Full): {block2_predictors}")
        log(f"Outcome variable: theta_all (N={len(y)})")
        # Run Hierarchical Regression Analysis

        log("Running fit_hierarchical_regression...")
        hierarchy_results = fit_hierarchical_regression(
            X_blocks=X_blocks,
            y=y,
            block_names=block_names
        )
        log("Hierarchical regression complete")
        # Compute Bootstrap Confidence Intervals
        # Parameters: 1000 iterations with seed=42 for reproducibility

        log("Computing bootstrap confidence intervals...")
        bootstrap_results = bootstrap_regression_ci(
            X=X_block2,  # Use full model predictors
            y=y,
            n_bootstrap=1000,
            seed=42,
            alpha=0.05  # 95% confidence intervals
        )
        log("Bootstrap analysis complete (1000 iterations)")
        # Apply Multiple Comparison Corrections
        # Decision D068: Dual p-values (uncorrected + corrected)
        # Corrections: Bonferroni and FDR for 3 main predictors (excluding Age_z control)

        log("Applying multiple comparison corrections...")
        
        # Import statsmodels for multiple comparison corrections
        from statsmodels.stats.multitest import multipletests
        
        # Get full model results
        full_model = hierarchy_results['models'][1]  # Second model is the full model
        
        # Define main predictors (excluding Age_z which is the control)
        main_predictors = ['Education_z', 'VR_Experience_z', 'Typical_Sleep_z']
        all_predictors = ['Age_z', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z']
        
        # Extract p-values for main predictors only
        main_p_values = []
        for pred in main_predictors:
            idx = all_predictors.index(pred) + 1  # +1 for intercept in statsmodels
            main_p_values.append(full_model.pvalues[idx])  # pvalues is numpy array, not pandas
        
        # Bonferroni correction
        p_bonferroni = [min(p * len(main_predictors), 1.0) for p in main_p_values]
        
        # FDR correction
        _, p_fdr, _, _ = multipletests(main_p_values, method='fdr_bh')
        
        log(f"Bonferroni correction applied (k={len(main_predictors)})")
        log(f"FDR correction applied")
        # Save Model Comparison Results
        # Output: Model comparison between control and full models
        # Contains: R2, adjusted R2, F-statistic, AIC, BIC for each model

        log("Saving model comparison results...")
        
        model_results = []
        for i, (name, model) in enumerate(zip(block_names, hierarchy_results['models'])):
            model_results.append({
                'model': name,
                'R2': model.rsquared,
                'adj_R2': model.rsquared_adj,
                'F_stat': model.fvalue,
                'F_p': model.f_pvalue,
                'AIC': model.aic,
                'BIC': model.bic,
                'N': int(model.nobs)
            })
        
        model_df = pd.DataFrame(model_results)
        model_output_path = RQ_DIR / "data/step04_regression_models.csv"
        model_df.to_csv(model_output_path, index=False, encoding='utf-8')
        log(f"{model_output_path.name} ({len(model_df)} models)")
        # Save Coefficient Results with Bootstrap CIs and Dual P-values
        # Output: Regression coefficients with bootstrap CIs and corrected p-values
        # Contains: beta, SE, CI bounds, uncorrected p, Bonferroni p, FDR p

        log("Saving coefficients with bootstrap CIs and dual p-values...")
        
        coef_results = []
        for i, pred in enumerate(all_predictors):
            # Get model coefficient and SE
            coef_idx = i + 1  # +1 for intercept in statsmodels
            beta = full_model.params[coef_idx]  # params is numpy array
            se = full_model.bse[coef_idx]      # bse is numpy array
            p_uncorrected = full_model.pvalues[coef_idx]  # pvalues is numpy array
            
            # Get bootstrap CI bounds
            # bootstrap_results is a Dict with 'ci_bounds' key containing per-predictor CIs
            if 'ci_bounds' in bootstrap_results and len(bootstrap_results['ci_bounds']) > i:
                ci_lower, ci_upper = bootstrap_results['ci_bounds'][i]
            else:
                # Fallback: use model confidence interval if bootstrap fails
                conf_int = full_model.conf_int()
                ci_lower = conf_int[coef_idx, 0]  # conf_int is numpy array
                ci_upper = conf_int[coef_idx, 1]  # conf_int is numpy array
            
            # Get corrected p-values for main predictors
            if pred in main_predictors:
                main_idx = main_predictors.index(pred)
                p_bonf = p_bonferroni[main_idx]
                p_fdr_val = p_fdr[main_idx]
            else:
                # Age_z (control): no correction applied
                p_bonf = p_uncorrected
                p_fdr_val = p_uncorrected
            
            coef_results.append({
                'predictor': pred,
                'beta': beta,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonf,
                'p_fdr': p_fdr_val
            })
        
        coef_df = pd.DataFrame(coef_results)
        coef_output_path = RQ_DIR / "data/step04_coefficients_ci.csv"
        coef_df.to_csv(coef_output_path, index=False, encoding='utf-8')
        log(f"{coef_output_path.name} ({len(coef_df)} coefficients)")
        # Run Validation Tool
        # Validates: p-values are in [0, 1] range with no NaN/infinite values
        # Checks: uncorrected, Bonferroni, and FDR p-values

        log("Running validate_probability_range...")
        validation_result = validate_probability_range(
            probability_df=coef_df,
            prob_columns=['p_uncorrected', 'p_bonferroni', 'p_fdr']
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
        # Summary Statistics

        control_r2 = model_df[model_df['model'] == 'Control']['R2'].iloc[0]
        full_r2 = model_df[model_df['model'] == 'Full']['R2'].iloc[0]
        r2_change = full_r2 - control_r2
        
        log(f"Control model R2: {control_r2:.4f}")
        log(f"Full model R2: {full_r2:.4f}")
        log(f"R2 change: {r2_change:.4f}")
        log(f"Bootstrap CIs computed: 95% confidence intervals")
        log(f"Multiple comparisons: Bonferroni + FDR for {len(main_predictors)} predictors")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)