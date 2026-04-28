#!/usr/bin/env python3
"""Diagnostic Validation: Validate regression assumptions and check model diagnostics for optimal model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# from tools.validation import validate_regression_assumptions  # Function does not exist

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_diagnostic_validation.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Outlier thresholds
N = 100
COOKS_D_THRESHOLD = 4.0 / N  # 0.04
LEVERAGE_THRESHOLD = 0.20
STUDENTIZED_THRESHOLD = 3.0
ALPHA = 0.05

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Diagnostic Validation")
        # Load Data and Refit Optimal Model
        log("Loading data and refitting optimal model...")

        df_analysis = pd.read_csv(RQ_DIR / 'data' / 'step02_analysis_input.csv')
        df_models = pd.read_csv(RQ_DIR / 'data' / 'step03_nested_models.csv')
        df_comparison = pd.read_csv(RQ_DIR / 'data' / 'step05_model_comparison.csv')

        # Get optimal model
        optimal_model_name = df_comparison[df_comparison['selected_as_optimal']].iloc[0]['model']
        optimal_predictors_str = df_models[df_models['model_name'] == optimal_model_name]['predictor_list'].iloc[0]
        optimal_predictors = [p.strip() for p in optimal_predictors_str.split(',')]

        log(f"Optimal model: {optimal_model_name} ({len(optimal_predictors)} predictors)")

        # Fit model
        X = df_analysis[optimal_predictors]
        y = df_analysis['Theta_All']
        X_const = sm.add_constant(X, has_constant='add')
        model = sm.OLS(y, X_const).fit()

        log(f"R²={model.rsquared:.4f}, N={len(df_analysis)}")
        # Assumption Test 1 - Normality of Residuals (Shapiro-Wilk)
        log("\nShapiro-Wilk test for normality of residuals...")

        residuals = model.resid
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        if shapiro_p > ALPHA:
            log(f"Residuals approximately normal (W={shapiro_stat:.4f}, p={shapiro_p:.4f})")
        else:
            log(f"Residuals deviate from normality (W={shapiro_stat:.4f}, p={shapiro_p:.4f})")
        # Assumption Test 2 - Homoscedasticity (Breusch-Pagan)
        log("Breusch-Pagan test for homoscedasticity...")

        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)

        if bp_p > ALPHA:
            log(f"Homoscedasticity assumption met (LM={bp_stat:.4f}, p={bp_p:.4f})")
        else:
            log(f"Heteroscedasticity detected (LM={bp_stat:.4f}, p={bp_p:.4f})")
        # Multicollinearity Test - Variance Inflation Factor (VIF)
        log("Computing variance inflation factors...")

        vif_results = []

        for i, predictor in enumerate(X.columns):
            # VIF computation (exclude intercept)
            vif_value = variance_inflation_factor(X_const.values, i + 1)  # +1 to skip intercept

            if vif_value < 5.0:
                severity = "LOW"
            elif vif_value < 10.0:
                severity = "MODERATE"
            else:
                severity = "HIGH"

            log(f"{predictor}: {vif_value:.4f} ({severity})")

            vif_results.append({
                'predictor': predictor,
                'vif': vif_value
            })

        df_vif = pd.DataFrame(vif_results)

        # Check for severe multicollinearity
        max_vif = df_vif['vif'].max()
        if max_vif < 10.0:
            log(f"No severe multicollinearity (max VIF={max_vif:.4f})")
        else:
            log(f"Severe multicollinearity detected (max VIF={max_vif:.4f})")
        # Outlier Analysis
        log("\nIdentifying influential cases...")

        # Cook's distance
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # Leverage
        leverage = influence.hat_matrix_diag

        # Studentized residuals
        studentized_resid = influence.resid_studentized_internal

        # Identify outliers
        outliers_cooks = cooks_d > COOKS_D_THRESHOLD
        outliers_leverage = leverage > LEVERAGE_THRESHOLD
        outliers_studentized = np.abs(studentized_resid) > STUDENTIZED_THRESHOLD

        n_outliers_cooks = outliers_cooks.sum()
        n_outliers_leverage = outliers_leverage.sum()
        n_outliers_studentized = outliers_studentized.sum()

        log(f"Cook's D > {COOKS_D_THRESHOLD:.4f}: {n_outliers_cooks} cases")
        log(f"Leverage > {LEVERAGE_THRESHOLD:.4f}: {n_outliers_leverage} cases")
        log(f"|Studentized| > {STUDENTIZED_THRESHOLD:.1f}: {n_outliers_studentized} cases")

        # Create outlier dataframe
        df_outliers = pd.DataFrame({
            'UID': df_analysis['UID'].values,
            'cooks_d': cooks_d,
            'leverage': leverage,
            'studentized_residual': studentized_resid
        })

        # Identify influential cases (Cook's D exceeds threshold)
        influential = df_outliers[df_outliers['cooks_d'] > COOKS_D_THRESHOLD]
        if len(influential) > 0:
            log(f"{len(influential)} influential cases:")
            for _, row in influential.iterrows():
                log(f"  {row['UID']}: Cook's D={row['cooks_d']:.4f}")
        else:
            log(f"No influential outliers detected")
        # Save Outputs
        log("\nSaving diagnostic results...")

        # Assumption tests
        assumption_tests = [
            {'test_name': 'Shapiro-Wilk', 'statistic': shapiro_stat, 'p_value': shapiro_p},
            {'test_name': 'Breusch-Pagan', 'statistic': bp_stat, 'p_value': bp_p},
            {'test_name': 'VIF_max', 'statistic': max_vif, 'p_value': np.nan}
        ]
        df_assumptions = pd.DataFrame(assumption_tests)
        assumptions_file = OUTPUT_DIR / "step07_assumption_tests.csv"
        df_assumptions.to_csv(assumptions_file, index=False, encoding='utf-8')
        log(f"{assumptions_file}")

        # VIF results
        vif_file = OUTPUT_DIR / "step07_vif_results.csv"
        df_vif.to_csv(vif_file, index=False, encoding='utf-8')
        log(f"{vif_file}")

        # Outlier analysis
        outliers_file = OUTPUT_DIR / "step07_outlier_analysis.csv"
        df_outliers.to_csv(outliers_file, index=False, encoding='utf-8')
        log(f"{outliers_file}")

        # Diagnostic data (residuals, fitted)
        df_diagnostic = pd.DataFrame({
            'UID': df_analysis['UID'].values,
            'residual': residuals.values,
            'fitted': model.fittedvalues.values
        })
        diagnostic_file = OUTPUT_DIR / "step07_diagnostic_data.csv"
        df_diagnostic.to_csv(diagnostic_file, index=False, encoding='utf-8')
        log(f"{diagnostic_file}")
        # Validation Check
        log("\nRunning validation tool...")

        log("Manual diagnostics complete - comprehensive validation tool not available")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
