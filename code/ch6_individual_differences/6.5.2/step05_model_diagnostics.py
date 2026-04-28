#!/usr/bin/env python3
"""model_diagnostics: Check regression assumptions and identify remedial actions for the full regression model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_model_diagnostics.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Model Diagnostics")
        # Load Input Data

        log("Loading analysis dataset...")
        input_df = pd.read_csv(RQ_DIR / "data/step01_analysis_dataset.csv")
        log(f"step01_analysis_dataset.csv ({len(input_df)} rows, {len(input_df.columns)} cols)")

        # Verify required columns
        required_cols = ["UID", "theta_all", "dass_dep", "dass_anx", "dass_str", "age", "nart_score"]
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        # Prepare Regression Model Data
        # Prepare data for full regression model: theta_all ~ age + nart_score + dass_dep + dass_anx + dass_str
        # This replicates the model from Steps 3-4 for diagnostic computation

        log("Setting up regression model data...")
        
        # Define predictors and outcome
        predictor_cols = ["age", "nart_score", "dass_dep", "dass_anx", "dass_str"]
        outcome_col = "theta_all"
        
        # Extract data for regression
        X = input_df[predictor_cols].copy()
        y = input_df[outcome_col].copy()
        
        log(f"Predictors: {predictor_cols}")
        log(f"Outcome: {outcome_col}")
        log(f"Sample size: {len(X)} observations")

        # Check for missing data
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0 or missing_y > 0:
            raise ValueError(f"Missing data detected: X has {missing_X} missing, y has {missing_y} missing")
        
        log("No missing data in regression variables")
        # Fit Full Regression Model

        log("Fitting full regression model...")
        
        # Add constant term for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit OLS model
        full_model = sm.OLS(y, X_with_const).fit()
        
        log("Regression model fitted successfully")
        log(f"R-squared: {full_model.rsquared:.4f}")
        log(f"Adj. R-squared: {full_model.rsquared_adj:.4f}")
        log(f"F-statistic: {full_model.fvalue:.4f} (p={full_model.f_pvalue:.6f})")
        # Compute Model-Level Diagnostics
        # Compute VIF, condition number, heteroscedasticity, autocorrelation tests
        # Focus on identifying multicollinearity and assumption violations

        log("Computing model-level diagnostics...")
        
        # 1. Variance Inflation Factors (VIF)
        # VIF > 5 indicates problematic multicollinearity, VIF > 10 is severe
        vif_data = []
        for i in range(X_with_const.shape[1]):
            if i == 0:  # Skip constant term
                continue
            vif = variance_inflation_factor(X_with_const.values, i)
            predictor_name = X_with_const.columns[i]
            vif_data.append({
                'predictor': predictor_name,
                'VIF': vif
            })
        
        log(f"Computed VIF for {len(vif_data)} predictors")
        
        # Check VIF thresholds
        high_vif = [item for item in vif_data if item['VIF'] > 5.0]
        severe_vif = [item for item in vif_data if item['VIF'] > 10.0]
        
        if high_vif:
            log(f"[VIF WARNING] {len(high_vif)} predictors with VIF > 5.0: {[item['predictor'] for item in high_vif]}")
        if severe_vif:
            log(f"[VIF SEVERE] {len(severe_vif)} predictors with VIF > 10.0: {[item['predictor'] for item in severe_vif]}")
        
        # 2. Condition Number
        condition_number = np.linalg.cond(X_with_const.values)
        log(f"Condition number: {condition_number:.2f}")
        
        # 3. Breusch-Pagan Test for Homoscedasticity
        # H0: Homoscedastic residuals, H1: Heteroscedastic residuals
        bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(full_model.resid, X_with_const)
        log(f"[BREUSCH-PAGAN] LM statistic: {bp_lm:.4f}, p-value: {bp_lm_pvalue:.6f}")
        
        # 4. Manual Durbin-Watson Test for Autocorrelation
        # Values around 2 indicate no autocorrelation
        # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        residuals = full_model.resid.values
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        log(f"[DURBIN-WATSON] Test statistic: {dw_stat:.4f} (Manual calculation)")
        
        # Create model diagnostics dataframe
        model_diagnostics = []
        for vif_item in vif_data:
            model_diagnostics.append({
                'predictor': vif_item['predictor'],
                'VIF': vif_item['VIF'],
                'condition_number': condition_number,
                'breusch_pagan_p': bp_lm_pvalue,
                'durbin_watson': dw_stat
            })
        
        model_diagnostics_df = pd.DataFrame(model_diagnostics)
        log(f"Model-level diagnostics computed for {len(model_diagnostics_df)} predictors")
        # Compute Observation-Level Diagnostics
        # Cook's D, leverage, standardized residuals for outlier detection
        
        log("Computing observation-level diagnostics...")
        
        # Get fitted values and residuals
        fitted_values = full_model.fittedvalues
        residuals = full_model.resid
        
        # Standardized residuals
        residual_std = residuals / np.sqrt(full_model.mse_resid)
        
        # Leverage (hat matrix diagonal)
        leverage = full_model.get_influence().hat_matrix_diag
        
        # Cook's Distance
        cooks_d = full_model.get_influence().cooks_distance[0]
        
        # Create observation-level diagnostics dataframe
        residual_analysis = pd.DataFrame({
            'observation': range(len(fitted_values)),
            'fitted': fitted_values,
            'residual': residuals,
            'standardized_residual': residual_std,
            'leverage': leverage,
            'cooks_d': cooks_d
        })
        
        log(f"Observation-level diagnostics computed for {len(residual_analysis)} observations")
        
        # Check Cook's D threshold (4/N)
        n_obs = len(residual_analysis)
        cooks_d_threshold = 4.0 / n_obs
        influential_obs = residual_analysis[residual_analysis['cooks_d'] > cooks_d_threshold]
        
        if len(influential_obs) > 0:
            log(f"[COOKS-D WARNING] {len(influential_obs)} observations exceed Cook's D threshold {cooks_d_threshold:.4f}")
            log(f"[COOKS-D] Influential observations: {influential_obs['observation'].tolist()}")
        # Assumption Tests
        # Shapiro-Wilk normality test on residuals
        # Additional diagnostic summary
        
        log("Running assumption tests...")
        
        # Shapiro-Wilk test for residual normality
        sw_stat, sw_pvalue = stats.shapiro(residuals)
        log(f"[SHAPIRO-WILK] Normality test on residuals: W={sw_stat:.4f}, p={sw_pvalue:.6f}")
        
        # Summary of assumption violations
        assumption_violations = []
        
        # VIF violations  
        if high_vif:
            assumption_violations.append(f"Multicollinearity: {len(high_vif)} predictors with VIF > 5.0")
        if severe_vif:
            assumption_violations.append(f"Severe multicollinearity: {len(severe_vif)} predictors with VIF > 10.0")
            
        # Condition number
        if condition_number > 30:
            assumption_violations.append(f"High condition number: {condition_number:.2f} (threshold 30)")
            
        # Residual normality
        if sw_pvalue < 0.05:
            assumption_violations.append(f"Non-normal residuals: Shapiro-Wilk p={sw_pvalue:.6f}")
            
        # Homoscedasticity
        if bp_lm_pvalue < 0.05:
            assumption_violations.append(f"Heteroscedasticity: Breusch-Pagan p={bp_lm_pvalue:.6f}")
            
        # Autocorrelation (Durbin-Watson should be around 2)
        if dw_stat < 1.5 or dw_stat > 2.5:
            assumption_violations.append(f"Autocorrelation concern: Durbin-Watson = {dw_stat:.4f}")
            
        # Influential observations
        if len(influential_obs) > 0:
            assumption_violations.append(f"Influential observations: {len(influential_obs)} exceed Cook's D threshold")
        
        log(f"{len(assumption_violations)} potential violations identified")
        # Save Analysis Outputs
        # Save model diagnostics, residual analysis, and assumption check summary

        log("Saving model diagnostics...")
        model_diagnostics_df.to_csv(RQ_DIR / "data/step05_model_diagnostics.csv", index=False, encoding='utf-8')
        log(f"step05_model_diagnostics.csv ({len(model_diagnostics_df)} rows, {len(model_diagnostics_df.columns)} cols)")

        log("Saving residual analysis...")
        residual_analysis.to_csv(RQ_DIR / "data/step05_residual_analysis.csv", index=False, encoding='utf-8')
        log(f"step05_residual_analysis.csv ({len(residual_analysis)} rows, {len(residual_analysis.columns)} cols)")

        # Create assumption checks summary text
        assumption_summary = []
        assumption_summary.append("REGRESSION MODEL DIAGNOSTIC SUMMARY")
        assumption_summary.append("=" * 50)
        assumption_summary.append(f"Model: theta_all ~ age + nart_score + dass_dep + dass_anx + dass_str")
        assumption_summary.append(f"Sample size: {n_obs}")
        assumption_summary.append(f"R-squared: {full_model.rsquared:.4f}")
        assumption_summary.append("")
        
        assumption_summary.append("DIAGNOSTIC TEST RESULTS:")
        assumption_summary.append(f"- Shapiro-Wilk normality test: W={sw_stat:.4f}, p={sw_pvalue:.6f}")
        assumption_summary.append(f"- Breusch-Pagan homoscedasticity test: LM={bp_lm:.4f}, p={bp_lm_pvalue:.6f}")
        assumption_summary.append(f"- Durbin-Watson autocorrelation test: DW={dw_stat:.4f}")
        assumption_summary.append(f"- Condition number: {condition_number:.2f}")
        assumption_summary.append("")
        
        assumption_summary.append("VIF VALUES (Multicollinearity Check):")
        for item in vif_data:
            vif_status = "OK" if item['VIF'] <= 5.0 else "HIGH" if item['VIF'] <= 10.0 else "SEVERE"
            assumption_summary.append(f"- {item['predictor']}: VIF = {item['VIF']:.3f} ({vif_status})")
        assumption_summary.append("")
        
        assumption_summary.append("INFLUENTIAL OBSERVATIONS:")
        assumption_summary.append(f"- Cook's D threshold (4/N): {cooks_d_threshold:.4f}")
        assumption_summary.append(f"- Observations exceeding threshold: {len(influential_obs)}")
        if len(influential_obs) > 0:
            assumption_summary.append(f"- Influential observation IDs: {influential_obs['observation'].tolist()}")
        assumption_summary.append("")
        
        assumption_summary.append("ASSUMPTION VIOLATIONS:")
        if assumption_violations:
            for violation in assumption_violations:
                assumption_summary.append(f"- {violation}")
        else:
            assumption_summary.append("- No major violations detected")
        assumption_summary.append("")
        
        assumption_summary.append("REMEDIAL ACTIONS:")
        if high_vif:
            assumption_summary.append("- Consider ridge regression or principal components for multicollinearity")
        if sw_pvalue < 0.05:
            assumption_summary.append("- Consider bootstrap confidence intervals for non-normal residuals")
        if bp_lm_pvalue < 0.05:
            assumption_summary.append("- Consider robust standard errors for heteroscedasticity")
        if len(influential_obs) > 0:
            assumption_summary.append("- Consider robust regression or examine influential cases")
        if len(assumption_violations) == 0:
            assumption_summary.append("- Standard OLS inference appears appropriate")
        assumption_summary.append("")
        
        # Known issue from Step 4
        assumption_summary.append("NOTE FROM STEP 4:")
        assumption_summary.append("- High multicollinearity already detected in prior analysis")
        assumption_summary.append("- DASS subscales likely correlated (anxiety, depression, stress overlap)")
        assumption_summary.append("- Consider reporting bootstrap CIs as in Steps 3-4 for robustness")
        
        assumption_text = "\n".join(assumption_summary)
        
        log("Saving assumption checks summary...")
        with open(RQ_DIR / "data/step05_assumption_checks.txt", 'w', encoding='utf-8') as f:
            f.write(assumption_text)
        log(f"step05_assumption_checks.txt")
        # Run Validation Tool
        # Note: This is designed for LMM but we're using OLS - may have compatibility issues
        # Will attempt validation with available parameters

        log("Attempting validation...")
        
        try:
            # Note: validation function expects MixedLMResults but we have OLS results
            # This may fail due to signature mismatch, but we'll attempt it
            validation_result = validate_lmm_assumptions_comprehensive(
                lmm_result=full_model,  # OLS results (may not be compatible)
                data=input_df,
                output_dir=RQ_DIR / "plots",  # plots folder for validation outputs
                acf_lag1_threshold=0.1,
                alpha=0.05
            )
            
            # Report validation results
            if isinstance(validation_result, dict):
                for key, value in validation_result.items():
                    log(f"{key}: {value}")
            else:
                log(f"{validation_result}")
                
        except Exception as e:
            log(f"[VALIDATION WARNING] Validation function failed (expected with OLS model): {str(e)}")
            log("Proceeding with manual diagnostic assessment")
            
            # Manual validation summary
            validation_status = "PASS" if len(assumption_violations) == 0 else "FAIL"
            log(f"[VALIDATION MANUAL] Overall status: {validation_status}")
            log(f"[VALIDATION MANUAL] {len(assumption_violations)} violations detected")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)