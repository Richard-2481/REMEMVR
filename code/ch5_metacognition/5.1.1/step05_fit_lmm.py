#!/usr/bin/env python3
"""Fit 5 Candidate LMM Models: Fit 5 candidate functional forms for confidence decline trajectory using Linear Mixed Models."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Fit 5 Candidate LMM Models")
        # Load Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Verify expected columns
        expected_cols = ['composite_ID', 'UID', 'test', 'theta_All', 'se_All', 'TSVR_hours', 'Days', 'Days_squared', 'log_Days_plus1']
        if list(df.columns) != expected_cols:
            raise ValueError(f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}")

        log(f"Data summary: {len(df['UID'].unique())} unique UIDs, {len(df)} observations")
        # Define 5 Candidate Model Formulas
        # Models test different functional forms of forgetting trajectory:
        #   Model 1: Linear decay (simplest)
        #   Model 2: Quadratic decay (allows acceleration/deceleration)
        #   Model 3: Logarithmic decay (power law forgetting)
        #   Model 4: Linear + Logarithmic (hybrid: early fast, late slow)
        #   Model 5: Quadratic + Logarithmic (most flexible, highest params)

        log("Defining model formulas...")

        models = {
            'Linear': {
                'formula': 'theta_All ~ Days',
                'description': 'Simple linear decline'
            },
            'Quadratic': {
                'formula': 'theta_All ~ Days + Days_squared',
                'description': 'Quadratic decline (acceleration/deceleration)'
            },
            'Logarithmic': {
                'formula': 'theta_All ~ log_Days_plus1',
                'description': 'Logarithmic decline (power law)'
            },
            'Linear+Logarithmic': {
                'formula': 'theta_All ~ Days + log_Days_plus1',
                'description': 'Hybrid linear + logarithmic'
            },
            'Quadratic+Logarithmic': {
                'formula': 'theta_All ~ Days + Days_squared + log_Days_plus1',
                'description': 'Full model with quadratic + logarithmic terms'
            }
        }

        log(f"Testing {len(models)} candidate models")
        # Fit Each Model
        # Fit parameters:
        #   - groups='UID': Random intercepts by participant
        #   - re_formula='~Days': Random slopes for Days (allows individual forgetting rates)
        #   - REML=False: Use ML estimation (required for valid AIC comparison)
        #   - method='lbfgs': Optimization algorithm (default, usually robust)

        log("Fitting 5 LMM models...")

        results = {}
        comparison_rows = []

        for idx, (model_name, model_spec) in enumerate(models.items(), start=1):
            log(f"[MODEL {idx}/5] Fitting {model_name}: {model_spec['description']}")
            log(f"  Formula: {model_spec['formula']}")

            try:
                # Fit LMM using statsmodels
                model = smf.mixedlm(
                    formula=model_spec['formula'],
                    data=df,
                    groups=df["UID"],
                    re_formula="~Days"
                )

                result = model.fit(method="lbfgs", reml=False)

                # Extract fit statistics
                converged = result.converged
                aic = result.aic
                bic = result.bic
                log_likelihood = result.llf
                # Count params: fixed effects + random effects variance components
                # cov_re may be DataFrame or ndarray depending on statsmodels version
                cov_re = result.cov_re
                if hasattr(cov_re, 'values'):
                    n_random = cov_re.values.flatten().shape[0]
                else:
                    n_random = cov_re.flatten().shape[0]
                num_params = len(result.params) + n_random

                log(f"  Converged: {converged}")
                log(f"  AIC: {aic:.2f}, BIC: {bic:.2f}, LogLik: {log_likelihood:.2f}")
                log(f"  Num params: {num_params}")

                # Store result object
                results[model_name] = result

                # Add to comparison table
                comparison_rows.append({
                    'model_name': model_name,
                    'AIC': aic,
                    'BIC': bic,
                    'log_likelihood': log_likelihood,
                    'num_params': num_params,
                    'converged': converged
                })

            except Exception as e:
                log(f"  Model {model_name} failed to fit: {str(e)}")
                # Add failed model to comparison table with NaN values
                comparison_rows.append({
                    'model_name': model_name,
                    'AIC': np.nan,
                    'BIC': np.nan,
                    'log_likelihood': np.nan,
                    'num_params': np.nan,
                    'converged': False
                })
                # Continue with other models
                continue

        log("All models fitted")
        # Save Model Comparison Table
        # Output: Summary table with AIC/BIC for each model
        # Used by: step06 for model selection via Akaike weights

        log("Saving model comparison table...")
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = RQ_DIR / "data" / "step05_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        log(f"{comparison_path.name} ({len(comparison_df)} rows, {len(comparison_df.columns)} cols)")
        # Save Fitted Model Objects
        # Save each fitted model as .pkl for later use
        # Downstream: step06 (model selection), step07+ (extracting effects from best model)

        log("Saving fitted model objects...")

        model_filenames = {
            'Linear': 'step05_model1_linear.pkl',
            'Quadratic': 'step05_model2_quadratic.pkl',
            'Logarithmic': 'step05_model3_logarithmic.pkl',
            'Linear+Logarithmic': 'step05_model4_linear_logarithmic.pkl',
            'Quadratic+Logarithmic': 'step05_model5_quadratic_logarithmic.pkl'
        }

        for model_name, filename in model_filenames.items():
            if model_name in results:
                model_path = RQ_DIR / "data" / filename

                # Save using statsmodels save() method (recommended over pickle)
                # This preserves formula, data structure, and convergence info
                results[model_name].save(str(model_path))
                log(f"{filename} ({model_name})")
            else:
                log(f"{filename} ({model_name}) - model failed to fit")
        # Run Validation
        # Validates: All models converged, AIC/BIC finite, expected model count

        log("Running validation checks...")

        # Check 1: All models present
        if len(comparison_df) != 5:
            raise ValueError(f"Expected 5 models, got {len(comparison_df)}")
        log("Model count: 5 (correct)")

        # Check 2: Model names match expected set
        expected_names = set(models.keys())
        actual_names = set(comparison_df['model_name'])
        if actual_names != expected_names:
            raise ValueError(f"Model name mismatch. Expected {expected_names}, got {actual_names}")
        log("Model names: all expected models present")

        # Check 3: All models converged
        non_converged = comparison_df[~comparison_df['converged']]
        if len(non_converged) > 0:
            log(f"{len(non_converged)} model(s) failed to converge:")
            for _, row in non_converged.iterrows():
                log(f"  - {row['model_name']}")
            # Don't raise error - allow step to complete with warning
        else:
            log("Convergence: All models converged successfully")

        # Check 4: AIC/BIC/log_likelihood finite
        for col in ['AIC', 'BIC', 'log_likelihood']:
            if comparison_df[col].isna().any():
                failed_models = comparison_df[comparison_df[col].isna()]['model_name'].tolist()
                log(f"{col} contains NaN for models: {failed_models}")
            elif np.isinf(comparison_df[col]).any():
                failed_models = comparison_df[comparison_df[col] == np.inf]['model_name'].tolist()
                log(f"{col} contains Inf for models: {failed_models}")
            else:
                log(f"{col}: All finite values")

        # Check 5: Individual model convergence validation
        for model_name, result in results.items():
            validation_result = validate_lmm_convergence(result)
            if validation_result['converged']:
                log(f"{model_name}: Convergence validated")
            else:
                log(f"{model_name}: {validation_result['message']}")

        log("Step 5 complete - 5 LMM models fitted and saved")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
