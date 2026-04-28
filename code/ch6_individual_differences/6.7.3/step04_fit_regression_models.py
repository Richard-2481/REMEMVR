#!/usr/bin/env python3
"""fit_regression_models: Fit 5 regression models comparing RAVLT scoring methods with bootstrap CIs."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_fit_regression_models.log"
OUTPUT_MODELS = RQ_DIR / "data" / "step04_model_results.csv"
OUTPUT_COEFS = RQ_DIR / "data" / "step04_regression_coefficients.csv"

# Configuration

# Decision D068: Bonferroni correction for Ch7 (28 RQs, 5 models per RQ)
BONFERRONI_ALPHA = 0.05 / 28 / 5  # = 0.000357

# Bootstrap settings
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Bootstrap R² CI Function

def bootstrap_r2_ci(X, y, n_iterations=1000, random_state=42):
    """
    Compute bootstrap confidence interval for R².

    Parameters:
    - X: Predictors (with constant)
    - y: Outcome
    - n_iterations: Number of bootstrap samples
    - random_state: Random seed

    Returns:
    - (ci_lower, ci_upper): 95% CI bounds
    """
    np.random.seed(random_state)
    n = len(y)
    r2_boot = []

    for i in range(n_iterations):
        # Resample rows (participant-level resampling)
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        y_boot = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]

        # Fit model on bootstrap sample
        try:
            model_boot = sm.OLS(y_boot, X_boot).fit()
            r2_boot.append(model_boot.rsquared)
        except:
            # If fit fails, skip this iteration
            continue

    # Compute 95% CI
    ci_lower = np.percentile(r2_boot, 2.5)
    ci_upper = np.percentile(r2_boot, 97.5)

    return ci_lower, ci_upper

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Fit Regression Models")
        # Load Analysis Dataset
        log("Loading analysis dataset...")

        data_path = RQ_DIR / "data" / "step03_analysis_dataset.csv"
        df = pd.read_csv(data_path)

        log(f"{data_path.name} ({len(df)} participants, {len(df.columns)} columns)")
        # Define Models
        log("Defining 5 regression models...")

        models_spec = [
            {'name': 'Model_1_Total', 'predictors': ['Total_z']},
            {'name': 'Model_2_Learning', 'predictors': ['Learning_z']},
            {'name': 'Model_3_LearningSlope', 'predictors': ['LearningSlope_z']},
            {'name': 'Model_4_Forgetting', 'predictors': ['Forgetting_z']},
            {'name': 'Model_5_Recognition', 'predictors': ['Recognition_z']},
            {'name': 'Model_6_PctRet', 'predictors': ['PctRet_z']},
            {'name': 'Model_7_Combined', 'predictors': ['Total_z', 'Learning_z']}
        ]

        log(f"Fitting {len(models_spec)} models")
        # Fit Models and Collect Results
        model_results = []
        coef_results = []

        for spec in models_spec:
            model_name = spec['name']
            predictors = spec['predictors']

            log(f"{model_name}: {', '.join(predictors)}")

            # Prepare data
            X = df[predictors]
            y = df['theta_all']

            # Add constant (intercept)
            X_with_const = sm.add_constant(X)

            # Fit OLS model
            model = sm.OLS(y, X_with_const).fit()

            log(f"R²={model.rsquared:.4f}, adj_R²={model.rsquared_adj:.4f}, F={model.fvalue:.2f}, p={model.f_pvalue:.6f}")

            # Bootstrap R² CI
            log(f"Computing R² CI ({BOOTSTRAP_ITERATIONS} iterations)...")
            r2_ci_lower, r2_ci_upper = bootstrap_r2_ci(X_with_const, y, n_iterations=BOOTSTRAP_ITERATIONS, random_state=RANDOM_SEED)
            log(f"R² 95% CI: [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")

            # Bonferroni-corrected p-value
            p_bonferroni = model.f_pvalue * (28 * 5)  # Multiply by number of tests
            p_bonferroni = min(p_bonferroni, 1.0)  # Cap at 1.0

            # Store model-level results
            model_results.append({
                'model': model_name,
                'R2': model.rsquared,
                'adj_R2': model.rsquared_adj,
                'F_stat': model.fvalue,
                'p_uncorrected': model.f_pvalue,
                'p_bonferroni': p_bonferroni,
                'AIC': model.aic,
                'BIC': model.bic,
                'R2_CI_lower': r2_ci_lower,
                'R2_CI_upper': r2_ci_upper
            })

            # Store coefficient-level results
            # Get confidence intervals (avoid bug #8 - conf_int returns numpy array)
            conf_int = model.conf_int()

            for i, param_name in enumerate(model.params.index):
                if param_name == 'const':
                    continue  # Skip intercept

                # Get coefficient details using .loc for label-based indexing
                beta = model.params.loc[param_name]
                se = model.bse.loc[param_name]
                p_val = model.pvalues.loc[param_name]
                ci_lower = conf_int.loc[param_name, 0]
                ci_upper = conf_int.loc[param_name, 1]

                # Bonferroni-corrected p-value for coefficient
                p_coef_bonf = p_val * (28 * 5)
                p_coef_bonf = min(p_coef_bonf, 1.0)

                coef_results.append({
                    'model': model_name,
                    'predictor': param_name,
                    'beta': beta,
                    'se': se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_uncorrected': p_val,
                    'p_bonferroni': p_coef_bonf
                })

            log(f"{model_name}")

        log(f"All {len(models_spec)} models complete")
        # Save Results
        log("Saving model results...")

        models_df = pd.DataFrame(model_results)
        models_df.to_csv(OUTPUT_MODELS, index=False, encoding='utf-8')
        log(f"{OUTPUT_MODELS} ({len(models_df)} models)")

        log("Saving coefficient results...")

        coefs_df = pd.DataFrame(coef_results)
        coefs_df.to_csv(OUTPUT_COEFS, index=False, encoding='utf-8')
        log(f"{OUTPUT_COEFS} ({len(coefs_df)} coefficients)")
        # Summary Report
        log("Model comparison:")

        for _, row in models_df.iterrows():
            sig_mark = "***" if row['p_bonferroni'] < BONFERRONI_ALPHA else ""
            log(f"  {row['model']}: R²={row['R2']:.4f} [{row['R2_CI_lower']:.4f}, {row['R2_CI_upper']:.4f}], "
                f"F={row['F_stat']:.2f}, "
                f"p={row['p_bonferroni']:.6f}{sig_mark}")

        log(f"Bonferroni alpha threshold: {BONFERRONI_ALPHA:.6f}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
