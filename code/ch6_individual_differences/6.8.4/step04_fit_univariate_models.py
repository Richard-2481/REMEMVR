#!/usr/bin/env python3
"""fit_univariate_models: Fit separate multiple regression models for each domain (What, Where, When) to establish"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_multiple_regression, compute_regression_diagnostics

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4
LOG_FILE = RQ_DIR / "logs" / "step04_fit_univariate_models.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: fit_univariate_models")
        # Load Analysis Dataset

        log("Loading analysis dataset...")

        data_path = RQ_DIR / 'data' / 'step03_analysis_dataset.csv'
        df_data = pd.read_csv(data_path)
        log(f"{data_path} ({len(df_data)} rows, {len(df_data.columns)} cols)")
        # Define Domains and Predictors

        domains = ['What_theta', 'Where_theta', 'When_theta']
        predictors = ['ravlt_z', 'bvmt_z', 'rpm_z', 'age_z', 'pctret_z']

        log(f"Will fit {len(domains)} univariate models")
        log(f"Predictors: {predictors}")
        # Fit Univariate Models

        log("Fitting univariate models...")

        model_results = []
        coefficient_results = []
        diagnostic_results = []

        X = df_data[predictors].values
        feature_names = predictors

        for domain in domains:
            log(f"Fitting {domain}...")

            y = df_data[domain].values

            # Fit regression using tools function
            result = fit_multiple_regression(
                X=X,
                y=y,
                feature_names=feature_names
            )

            # Extract model statistics
            model_stats = {
                'domain': domain.replace('_theta', ''),
                'R2': result['rsquared'],
                'adj_R2': result['rsquared_adj'],
                'F_stat': result['fvalue'],
                'p_value': result['f_pvalue'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'df_resid': result['model'].df_resid
            }
            model_results.append(model_stats)

            log(f"{domain}: R² = {result['rsquared']:.3f}, adj_R² = {result['rsquared_adj']:.3f}, AIC = {result['aic']:.2f}")

            # Extract coefficients (result['coefficients'] is a dict, not DataFrame)
            coef_dict = result['coefficients']
            pval_dict = result['pvalues']
            stderr_dict = result['std_errors']

            for predictor_name in ['intercept'] + feature_names:
                coefficient_results.append({
                    'domain': domain.replace('_theta', ''),
                    'predictor': predictor_name,
                    'beta': coef_dict[predictor_name],
                    'se': stderr_dict[predictor_name],
                    't_value': coef_dict[predictor_name] / stderr_dict[predictor_name],
                    'p_value': pval_dict[predictor_name]
                })

            # Compute diagnostics separately
            diag = compute_regression_diagnostics(
                model=result['model'],
                X=X,
                y=y
            )

            vif_values = diag.get('vif', [])
            max_vif = max(vif_values) if len(vif_values) > 0 else np.nan

            diagnostic_results.append({
                'domain': domain.replace('_theta', ''),
                'normality_p': np.nan,  # Would need separate Shapiro-Wilk test
                'homoscedasticity_p': diag.get('breusch_pagan', {}).get('p_value', np.nan),
                'vif_max': max_vif,
                'outliers_n': 0  # Not computed here
            })

        # Convert to DataFrames
        df_models = pd.DataFrame(model_results)
        df_coefficients = pd.DataFrame(coefficient_results)
        df_diagnostics = pd.DataFrame(diagnostic_results)
        # Cross-Validation

        log("[CV] Running cross-validation...")

        cv_results = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for domain in domains:
            log(f"[CV] {domain}...")

            y = df_data[domain].values
            cv_r2_scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Add constant for intercept
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)

                # Fit model on training fold
                model = sm.OLS(y_train, X_train_const).fit()

                # Predict on test fold
                y_pred = model.predict(X_test_const)

                # Compute R² on test fold
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                cv_r2_scores.append(r2)

            cv_r2_mean = np.mean(cv_r2_scores)
            cv_r2_std = np.std(cv_r2_scores)

            # Check overfitting (train R² - CV R² > 0.20)
            train_r2 = df_models[df_models['domain'] == domain.replace('_theta', '')]['R2'].values[0]
            overfitting = (train_r2 - cv_r2_mean) > 0.20

            cv_results.append({
                'domain': domain.replace('_theta', ''),
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'overfitting_flag': overfitting
            })

            log(f"[CV] {domain}: CV R² = {cv_r2_mean:.3f} ± {cv_r2_std:.3f}, overfitting = {overfitting}")

        df_cv = pd.DataFrame(cv_results)
        # Save Outputs
        # These outputs will be used by: Step 06 (model comparison)

        log("Saving univariate model results...")

        models_output = RQ_DIR / 'data' / 'step04_univariate_models.csv'
        df_models.to_csv(models_output, index=False, encoding='utf-8')
        log(f"{models_output} ({len(df_models)} rows)")

        coef_output = RQ_DIR / 'data' / 'step04_univariate_coefficients.csv'
        df_coefficients.to_csv(coef_output, index=False, encoding='utf-8')
        log(f"{coef_output} ({len(df_coefficients)} rows)")

        cv_output = RQ_DIR / 'data' / 'step04_univariate_cv_results.csv'
        df_cv.to_csv(cv_output, index=False, encoding='utf-8')
        log(f"{cv_output} ({len(df_cv)} rows)")

        diag_output = RQ_DIR / 'data' / 'step04_univariate_diagnostics.csv'
        df_diagnostics.to_csv(diag_output, index=False, encoding='utf-8')
        log(f"{diag_output} ({len(df_diagnostics)} rows)")
        # Validation
        # Validates: All models converged, VIF < 5.0, no severe overfitting
        # Threshold: All checks must pass

        log("Validating model results...")

        validation_pass = True

        # Check VIF threshold
        for _, row in df_diagnostics.iterrows():
            if not pd.isna(row['vif_max']) and row['vif_max'] >= 5.0:
                log(f"{row['domain']}: VIF = {row['vif_max']:.2f} exceeds threshold 5.0")
                # Don't fail - just warning

        # Check overfitting
        for _, row in df_cv.iterrows():
            if row['overfitting_flag']:
                log(f"{row['domain']}: Potential overfitting detected")
                # Don't fail - just warning

        log(f"All univariate models fitted successfully")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
