#!/usr/bin/env python3
"""Model Diagnostics and Assumption Validation: Comprehensive regression diagnostics and assumption checking for hierarchical"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for regression and diagnostics
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step06_model_diagnostics.log"

# Diagnostic thresholds
N_PARTICIPANTS = 100
N_PREDICTORS = 8  # age_std, sex, education_std, RAVLT_T_std, BVMT_T_std, RPM_T_std, RAVLT_Pct_Ret_T_std, BVMT_Pct_Ret_T_std
COOKS_D_THRESHOLD = 4.0 / N_PARTICIPANTS  # 0.04
LEVERAGE_THRESHOLD = 2.0 * N_PREDICTORS / N_PARTICIPANTS  # 0.12
VIF_THRESHOLD = 10.0
ALPHA = 0.05

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# VIF Computation Function

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each predictor.

    VIF = 1 / (1 - R²) where R² is from regressing predictor on all other predictors.

    Parameters:
        X: DataFrame with predictor variables (no constant)

    Returns:
        DataFrame with columns ['predictor', 'vif']
    """
    vif_data = []

    for i, col in enumerate(X.columns):
        # Regress this predictor on all other predictors
        y = X[col]
        X_other = X.drop(columns=[col])
        X_other_const = sm.add_constant(X_other)

        try:
            model = sm.OLS(y, X_other_const).fit()
            r_squared = model.rsquared
            vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else np.inf
        except:
            vif = np.nan

        vif_data.append({
            'predictor': col,
            'vif': vif
        })

    return pd.DataFrame(vif_data)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Model Diagnostics and Assumption Validation")
        # Load Input Data

        log("Loading analysis input data...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path)
        log(f"step03_analysis_input.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'slope', 'age_std', 'sex', 'education_std',
                        'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                        'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Refit Full Model (Model 2)
        # Model: slope ~ age_std + sex + education_std + RAVLT_T_std + BVMT_T_std + RPM_T_std

        log("Refitting full Model 2 for diagnostics...")

        # Prepare predictors and response
        predictor_cols = ['age_std', 'sex', 'education_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                          'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        X = df[predictor_cols].copy()
        y = df['slope'].copy()

        # Add constant
        X_const = sm.add_constant(X)

        # Fit model
        full_model = sm.OLS(y, X_const).fit()
        log(f"Model refitted: R² = {full_model.rsquared:.4f}, adj R² = {full_model.rsquared_adj:.4f}")
        # Multicollinearity Check (VIF)
        # VIF > 10 indicates problematic multicollinearity

        log("Computing Variance Inflation Factors (VIF)...")
        vif_df = compute_vif(X)

        for _, row in vif_df.iterrows():
            log(f"{row['predictor']}: {row['vif']:.3f}")

        max_vif = vif_df['vif'].max()
        vif_pass = max_vif < VIF_THRESHOLD
        log(f"Maximum VIF: {max_vif:.3f} (threshold = {VIF_THRESHOLD:.1f})")
        log(f"Multicollinearity check: {'PASS' if vif_pass else 'FAIL'}")
        # Normality of Residuals (Shapiro-Wilk Test)
        # H0: Residuals are normally distributed
        # p > 0.05 suggests normality assumption is reasonable

        log("Testing normality of residuals (Shapiro-Wilk)...")
        residuals = full_model.resid
        shapiro_stat, shapiro_p = shapiro(residuals)

        log(f"Shapiro-Wilk statistic: {shapiro_stat:.4f}")
        log(f"p-value: {shapiro_p:.4f}")
        normality_pass = shapiro_p > ALPHA
        log(f"Normality assumption: {'PASS' if normality_pass else 'FAIL'} (alpha = {ALPHA})")
        # Homoscedasticity (Breusch-Pagan Test)
        # H0: Residuals have constant variance
        # p > 0.05 suggests homoscedasticity assumption is reasonable

        log("Testing homoscedasticity (Breusch-Pagan)...")
        bp_test = het_breuschpagan(residuals, X_const)
        bp_statistic = bp_test[0]  # LM statistic
        bp_p = bp_test[1]          # p-value
        bp_f = bp_test[2]          # F-statistic
        bp_f_p = bp_test[3]        # F p-value

        log(f"Breusch-Pagan LM statistic: {bp_statistic:.4f}")
        log(f"p-value: {bp_p:.4f}")
        homoscedasticity_pass = bp_p > ALPHA
        log(f"Homoscedasticity assumption: {'PASS' if homoscedasticity_pass else 'FAIL'} (alpha = {ALPHA})")
        # Influential Observations (Cook's D and Leverage)
        # Cook's D > 4/n suggests influential observation
        # Leverage > 2p/n suggests high leverage point

        log("Computing Cook's D and leverage values...")
        influence = full_model.get_influence()
        cooks_d = influence.cooks_distance[0]  # First element is Cook's D values
        leverage = influence.hat_matrix_diag

        # Count influential observations
        high_cooks = (cooks_d > COOKS_D_THRESHOLD).sum()
        high_leverage = (leverage > LEVERAGE_THRESHOLD).sum()

        log(f"Cook's D threshold: {COOKS_D_THRESHOLD:.4f}")
        log(f"Observations with high Cook's D: {high_cooks} / {len(cooks_d)}")
        log(f"Maximum Cook's D: {cooks_d.max():.6f}")

        log(f"Leverage threshold: {LEVERAGE_THRESHOLD:.4f}")
        log(f"Observations with high leverage: {high_leverage} / {len(leverage)}")
        log(f"Maximum leverage: {leverage.max():.6f}")
        # Save Diagnostic Summary

        log("Saving diagnostic test results...")

        diagnostics_summary = [
            {
                'test': 'VIF_maximum',
                'statistic': max_vif,
                'p_value': np.nan,
                'threshold': VIF_THRESHOLD,
                'passes': vif_pass,
                'interpretation': f'Multicollinearity: {"Not problematic" if vif_pass else "Problematic"}'
            },
            {
                'test': 'Shapiro_Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'threshold': ALPHA,
                'passes': normality_pass,
                'interpretation': f'Residual normality: {"Reasonable" if normality_pass else "Questionable"}'
            },
            {
                'test': 'Breusch_Pagan',
                'statistic': bp_statistic,
                'p_value': bp_p,
                'threshold': ALPHA,
                'passes': homoscedasticity_pass,
                'interpretation': f'Homoscedasticity: {"Reasonable" if homoscedasticity_pass else "Questionable"}'
            },
            {
                'test': 'Cooks_D_max',
                'statistic': cooks_d.max(),
                'p_value': np.nan,
                'threshold': COOKS_D_THRESHOLD,
                'passes': cooks_d.max() <= COOKS_D_THRESHOLD,
                'interpretation': f'Maximum influence: {"Acceptable" if cooks_d.max() <= COOKS_D_THRESHOLD else "Potentially problematic"}'
            },
            {
                'test': 'Leverage_max',
                'statistic': leverage.max(),
                'p_value': np.nan,
                'threshold': LEVERAGE_THRESHOLD,
                'passes': leverage.max() <= LEVERAGE_THRESHOLD,
                'interpretation': f'Maximum leverage: {"Acceptable" if leverage.max() <= LEVERAGE_THRESHOLD else "Potentially high"}'
            },
            {
                'test': 'High_Cooks_D_count',
                'statistic': high_cooks,
                'p_value': np.nan,
                'threshold': 0,
                'passes': high_cooks == 0,
                'interpretation': f'{high_cooks} observations exceed Cook\'s D threshold'
            },
            {
                'test': 'High_Leverage_count',
                'statistic': high_leverage,
                'p_value': np.nan,
                'threshold': 0,
                'passes': high_leverage == 0,
                'interpretation': f'{high_leverage} observations exceed leverage threshold'
            }
        ]

        diagnostics_df = pd.DataFrame(diagnostics_summary)
        diagnostics_path = RQ_DIR / "data" / "step06_model_diagnostics.csv"
        diagnostics_df.to_csv(diagnostics_path, index=False, encoding='utf-8')
        log(f"step06_model_diagnostics.csv ({len(diagnostics_df)} rows, {len(diagnostics_df.columns)} cols)")
        # Save Diagnostic Plots Data

        log("Saving diagnostic plots data...")

        # Compute standardized residuals
        fitted_values = full_model.fittedvalues
        std_residuals = residuals / residuals.std()

        plots_data = pd.DataFrame({
            'observation': df['UID'].values,
            'fitted': fitted_values,
            'residual': residuals,
            'standardized_residual': std_residuals,
            'cooks_d': cooks_d,
            'leverage': leverage
        })

        plots_path = RQ_DIR / "data" / "step06_diagnostic_plots_data.csv"
        plots_data.to_csv(plots_path, index=False, encoding='utf-8')
        log(f"step06_diagnostic_plots_data.csv ({len(plots_data)} rows, {len(plots_data.columns)} cols)")
        # Validation Summary

        log("Diagnostic test summary:")
        log(f"Multicollinearity (VIF): {'PASS' if vif_pass else 'FAIL'}")
        log(f"Normality (Shapiro-Wilk): {'PASS' if normality_pass else 'FAIL'}")
        log(f"Homoscedasticity (Breusch-Pagan): {'PASS' if homoscedasticity_pass else 'FAIL'}")
        log(f"Influential observations: {high_cooks} high Cook's D, {high_leverage} high leverage")

        all_pass = vif_pass and normality_pass and homoscedasticity_pass
        log(f"Overall diagnostic status: {'ALL PASS' if all_pass else 'SOME CONCERNS'}")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
