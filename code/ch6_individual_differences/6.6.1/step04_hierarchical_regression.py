#!/usr/bin/env python3
"""Hierarchical Regression Analysis: Hierarchical multiple regression testing whether cognitive tests (RAVLT, BVMT, RPM)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step04_hierarchical_regression.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Hierarchical Regression Analysis")
        # Load Input Data

        log("Loading step03 merged and standardized data...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'slope', 'age_std', 'sex', 'education_std',
                        'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                        'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"All required columns present")
        log(f"Sample size: {len(df)} participants")
        # Prepare Data for Regression
        # Response variable: slope (forgetting rate)
        # Predictor blocks:
        #   Block 1 (Demographics): age_std, sex, education_std
        #   Block 2 (Cognitive): RAVLT_T_std, BVMT_T_std, RPM_T_std

        log("Preparing regression data...")

        # Response variable
        y = df['slope'].values
        log(f"Response variable (slope): mean={y.mean():.6f}, std={y.std():.6f}")

        # Model 1: Demographics only
        X1_cols = ['age_std', 'sex', 'education_std']
        X1 = df[X1_cols].values
        X1_with_const = sm.add_constant(X1)
        log(f"Model 1 (Demographics): {X1_cols}")

        # Model 2: Full model (Demographics + Cognitive)
        X2_cols = ['age_std', 'sex', 'education_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                   'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        X2 = df[X2_cols].values
        X2_with_const = sm.add_constant(X2)
        log(f"Model 2 (Full): {X2_cols}")
        # Fit Model 1 (Demographics Only)

        log("Fitting Model 1 (Demographics)...")
        model1 = sm.OLS(y, X1_with_const).fit()

        log(f"[MODEL1] R² = {model1.rsquared:.4f}")
        log(f"[MODEL1] Adjusted R² = {model1.rsquared_adj:.4f}")
        log(f"[MODEL1] F-statistic = {model1.fvalue:.4f}, p = {model1.f_pvalue:.6f}")
        log(f"[MODEL1] df_model = {model1.df_model}, df_resid = {model1.df_resid}")
        # Fit Model 2 (Full Model)

        log("Fitting Model 2 (Full Model)...")
        model2 = sm.OLS(y, X2_with_const).fit()

        log(f"[MODEL2] R² = {model2.rsquared:.4f}")
        log(f"[MODEL2] Adjusted R² = {model2.rsquared_adj:.4f}")
        log(f"[MODEL2] F-statistic = {model2.fvalue:.4f}, p = {model2.f_pvalue:.6f}")
        log(f"[MODEL2] df_model = {model2.df_model}, df_resid = {model2.df_resid}")
        # Compute Incremental R² and F-test
        # Test whether cognitive block significantly improves prediction

        log("Computing incremental R² (Model 2 vs Model 1)...")
        delta_r2 = model2.rsquared - model1.rsquared
        delta_df = model2.df_model - model1.df_model

        # F-test for incremental R²
        # F = (delta_R² / delta_df) / ((1 - R²_full) / df_resid_full)
        f_incremental = (delta_r2 / delta_df) / ((1 - model2.rsquared) / model2.df_resid)
        p_incremental = 1 - stats.f.cdf(f_incremental, delta_df, model2.df_resid)

        log(f"Delta R² = {delta_r2:.4f}")
        log(f"F({delta_df}, {model2.df_resid}) = {f_incremental:.4f}, p = {p_incremental:.6f}")
        # Extract Coefficients and Compute Semi-Partial R²
        # For Model 2 (Full Model), extract all statistics for each predictor
        # Decision D068: Report BOTH uncorrected AND corrected p-values

        log("Extracting Model 2 coefficients...")

        # Get confidence intervals (returns numpy array)
        conf_int = model2.conf_int()

        # Prepare results list
        results_list = []

        # Parameter names (with intercept)
        param_names = ['const'] + X2_cols

        # Decision D068: Multiple comparison corrections
        cognitive_predictors = ['RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                                'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        n_cognitive = len(cognitive_predictors)
        alpha_bonferroni = 0.05 / n_cognitive  # 0.01 (for 5 cognitive predictors)
        alpha_chapter = 0.00179  # Ch7 global correction

        for i, param_name in enumerate(param_names):
            beta = model2.params[i]
            se = model2.bse[i]
            ci_lower = conf_int[i, 0]
            ci_upper = conf_int[i, 1]
            t_stat = model2.tvalues[i]
            p_uncorrected = model2.pvalues[i]

            # Semi-partial r² (unique contribution)
            # sr² = (t² / (t² + df_resid)) * (1 - R²_full)
            sr_squared = (t_stat**2 / (t_stat**2 + model2.df_resid)) * (1 - model2.rsquared)

            # Apply corrections ONLY to cognitive predictors (Decision D068)
            if param_name in cognitive_predictors:
                p_bonferroni = p_uncorrected * n_cognitive  # Bonferroni correction
                p_bonferroni = min(p_bonferroni, 1.0)  # Cap at 1.0
                p_chapter = p_uncorrected  # Report uncorrected for chapter comparison
            else:
                p_bonferroni = np.nan  # Not applicable to demographics/intercept
                p_chapter = np.nan

            results_list.append({
                'predictor': param_name,
                'beta': beta,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                't_stat': t_stat,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
                'p_chapter': p_chapter,
                'sr_squared': sr_squared,
                'model': 'Full'
            })

            log(f"{param_name}: beta={beta:.6f}, SE={se:.6f}, t={t_stat:.3f}, p={p_uncorrected:.6f}, sr²={sr_squared:.6f}")
        # Save Regression Results
        # Output: step04_regression_results.csv with all coefficients

        log("Saving regression results...")
        output_path = RQ_DIR / "data" / "step04_regression_results.csv"
        results_df = pd.DataFrame(results_list)

        # Add model comparison statistics as metadata rows
        model_comparison = [
            {
                'predictor': 'MODEL1_R2',
                'beta': model1.rsquared,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                't_stat': np.nan,
                'p_uncorrected': model1.f_pvalue,
                'p_bonferroni': np.nan,
                'p_chapter': np.nan,
                'sr_squared': np.nan,
                'model': 'Demographics'
            },
            {
                'predictor': 'MODEL2_R2',
                'beta': model2.rsquared,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                't_stat': np.nan,
                'p_uncorrected': model2.f_pvalue,
                'p_bonferroni': np.nan,
                'p_chapter': np.nan,
                'sr_squared': np.nan,
                'model': 'Full'
            },
            {
                'predictor': 'DELTA_R2',
                'beta': delta_r2,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                't_stat': f_incremental,
                'p_uncorrected': p_incremental,
                'p_bonferroni': np.nan,
                'p_chapter': np.nan,
                'sr_squared': np.nan,
                'model': 'Incremental'
            }
        ]

        model_comp_df = pd.DataFrame(model_comparison)
        final_df = pd.concat([results_df, model_comp_df], ignore_index=True)

        final_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(final_df)} rows, {len(final_df.columns)} cols)")
        # Basic Assumption Validation
        # Manual checks: normality of residuals, homoscedasticity

        log("Checking regression assumptions...")

        # Normality of residuals (Shapiro-Wilk test)
        residuals = model2.resid
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        log(f"Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")

        if shapiro_p > 0.05:
            log(f"Residuals are approximately normal (p > 0.05)")
        else:
            log(f"Residuals may deviate from normality (p < 0.05)")

        # Homoscedasticity (Breusch-Pagan test)
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X2_with_const)
        log(f"Breusch-Pagan test: LM={bp_stat:.4f}, p={bp_p:.6f}")

        if bp_p > 0.05:
            log(f"Homoscedasticity assumption met (p > 0.05)")
        else:
            log(f"Heteroscedasticity detected (p < 0.05)")

        # Multicollinearity (VIF)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = []
        for i, col in enumerate(X2_cols):
            vif = variance_inflation_factor(X2, i)
            vif_data.append({'predictor': col, 'VIF': vif})
            log(f"VIF for {col}: {vif:.2f}")

        max_vif = max([v['VIF'] for v in vif_data])
        if max_vif < 10:
            log(f"No multicollinearity issues (max VIF={max_vif:.2f} < 10)")
        else:
            log(f"Multicollinearity detected (max VIF={max_vif:.2f} >= 10)")

        log("Step 04 complete - Hierarchical regression analysis finished")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
