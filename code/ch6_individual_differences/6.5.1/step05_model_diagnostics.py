#!/usr/bin/env python3
"""model_diagnostics: Comprehensive regression diagnostics and assumption testing"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_regression_diagnostics

from tools.validation import validate_variance_positivity

# Additional imports for diagnostics
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_model_diagnostics.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: model_diagnostics")
        # Load Input Data

        log("Loading input data...")
        
        # Load analysis dataset for model fitting
        analysis_df = pd.read_csv(RQ_DIR / "data/step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")
        
        # Load model results for reference
        model_results = pd.read_csv(RQ_DIR / "data/step04_regression_models.csv")
        log(f"step04_regression_models.csv ({len(model_results)} rows, {len(model_results.columns)} cols)")
        # Re-fit Full Model for Diagnostics

        log("Re-fitting full regression model...")
        
        # Prepare data matrices (same as step04 full model)
        X = analysis_df[['Age_z', 'Education_z', 'VR_Experience_z', 'Typical_Sleep_z']].values
        X_with_const = sm.add_constant(X)  # Add intercept
        y = analysis_df['theta_all'].values
        
        # Fit full model
        model = sm.OLS(y, X_with_const).fit()
        log("Model re-fitted successfully")
        log(f"R² = {model.rsquared:.4f}, N = {model.nobs}")
        # Compute Regression Diagnostics

        log("Running compute_regression_diagnostics...")
        diagnostics = compute_regression_diagnostics(model, X, y)
        log("Comprehensive diagnostics computed")
        # Extract Core Diagnostic Values
        # Extract key values using LESSONS LEARNED about numpy indexing
        # CRITICAL: Statsmodels attributes are numpy arrays, not pandas (lesson #17)
        
        log("Extracting diagnostic components...")
        
        # Extract residuals and fitted values (numpy arrays)
        residuals = model.resid  # numpy array
        fitted = model.fittedvalues  # numpy array
        leverage = model.get_influence().hat_matrix_diag  # numpy array
        cooks_d = model.get_influence().cooks_distance[0]  # numpy array
        
        log(f"Residuals: {len(residuals)} observations")
        log(f"Cook's D range: [{cooks_d.min():.6f}, {cooks_d.max():.6f}]")
        # Assumption Tests
        # Tests: Shapiro-Wilk (normality), Breusch-Pagan (homoscedasticity), VIF (multicollinearity)
        # Thresholds: p>0.05 for normality/homoscedasticity, VIF<5 for multicollinearity

        log("Running assumption tests...")
        diagnostic_tests = []
        
        # Normality test (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        diagnostic_tests.append({
            'test': 'Shapiro-Wilk',
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'interpretation': 'Normal' if shapiro_p > 0.05 else 'Non-normal'
        })
        log(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.6f}")
        
        # Homoscedasticity test (Breusch-Pagan)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_with_const)
        diagnostic_tests.append({
            'test': 'Breusch-Pagan',
            'statistic': bp_lm,
            'p_value': bp_lm_p,
            'interpretation': 'Homoscedastic' if bp_lm_p > 0.05 else 'Heteroscedastic'
        })
        log(f"Breusch-Pagan: LM = {bp_lm:.4f}, p = {bp_lm_p:.6f}")
        
        # VIF calculations (multicollinearity)
        vif_values = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X, i)
            vif_values.append(vif)
        
        max_vif = max(vif_values)
        diagnostic_tests.append({
            'test': 'VIF_max',
            'statistic': max_vif,
            'p_value': np.nan,
            'interpretation': 'Acceptable' if max_vif < 5 else 'Multicollinear'
        })
        log(f"VIF: max = {max_vif:.4f}")
        
        # Cook's Distance outliers (threshold = 4/N)
        n_outliers = sum(cooks_d > 4/len(analysis_df))
        threshold = 4/len(analysis_df)
        diagnostic_tests.append({
            'test': 'Cooks_D_outliers',
            'statistic': n_outliers,
            'p_value': np.nan,
            'interpretation': f'{n_outliers} outliers (threshold={threshold:.4f})'
        })
        log(f"Cook's D: {n_outliers} outliers > {threshold:.4f}")
        # Save Analysis Outputs
        # These outputs will be used by: step06+ for effect size analysis and reporting

        log("Saving diagnostic outputs...")
        
        # Save diagnostics summary
        diagnostics_summary = pd.DataFrame(diagnostic_tests)
        diagnostics_summary.to_csv(RQ_DIR / "data/step05_diagnostics_summary.csv", index=False, encoding='utf-8')
        log(f"step05_diagnostics_summary.csv ({len(diagnostics_summary)} rows, {len(diagnostics_summary.columns)} cols)")
        
        # Save residual data
        residual_data = pd.DataFrame({
            'UID': analysis_df['UID'],
            'residual': residuals,
            'fitted': fitted,
            'leverage': leverage,
            'cooks_d': cooks_d
        })
        residual_data.to_csv(RQ_DIR / "data/step05_residuals_data.csv", index=False, encoding='utf-8')
        log(f"step05_residuals_data.csv ({len(residual_data)} rows, {len(residual_data.columns)} cols)")
        
        # Save outlier analysis
        outliers = residual_data[residual_data['cooks_d'] > threshold]
        if len(outliers) > 0:
            outliers_df = outliers[['UID', 'cooks_d']].copy()
            outliers_df['outlier_flag'] = True
            outliers_df.to_csv(RQ_DIR / "data/step05_outlier_analysis.csv", index=False, encoding='utf-8')
            log(f"step05_outlier_analysis.csv ({len(outliers_df)} outliers identified)")
        else:
            # Create empty file with correct columns
            empty_outliers = pd.DataFrame(columns=['UID', 'cooks_d', 'outlier_flag'])
            empty_outliers.to_csv(RQ_DIR / "data/step05_outlier_analysis.csv", index=False, encoding='utf-8')
            log("step05_outlier_analysis.csv (0 outliers identified)")
        # Run Validation Tool
        # Validates: All diagnostic statistics are valid numbers (not NaN/infinite)
        # Threshold: Variance components should be positive/finite

        log("Running validate_variance_positivity...")
        
        # Prepare validation data (use diagnostics summary with test names and statistics)
        validation_result = validate_variance_positivity(
            variance_df=diagnostics_summary,
            component_col='test',
            value_col='statistic'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            
            # Check validation passed
            if validation_result.get('valid', False):
                log("Diagnostic statistics validation PASSED")
            else:
                log("Diagnostic statistics validation FAILED")
        else:
            log(f"{validation_result}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)