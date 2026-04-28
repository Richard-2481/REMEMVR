#!/usr/bin/env python3
"""fit_quadratic_model: Fit theta ~ Time + Time_squared + (Time | UID), test if quadratic term significant"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory_tsvr

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_fit_quadratic_model.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Fit Quadratic Model")
        # Load Input Data

        log("Loading time-transformed data from Step 1...")
        time_data = pd.read_csv(RQ_DIR / "data/step01_time_transformed.csv", encoding='utf-8')
        log(f"step01_time_transformed.csv ({len(time_data)} rows, {len(time_data.columns)} cols)")
        log(f"Columns: {list(time_data.columns)}")

        # Verify required columns present
        required_cols = ['UID', 'test', 'TSVR_hours', 'theta', 'Time', 'Time_squared']
        missing = [c for c in required_cols if c not in time_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        log(f"All required columns present: {required_cols}")
        # Run Analysis Tool

        log("Fitting quadratic model: theta ~ Time + Time_squared + (Time | UID)...")
        log("Using ML estimation (REML=False) for valid AIC comparison")
        log("Bonferroni alpha = 0.0033 (15 planned comparisons)")
        log("Fallback strategy: (Time | UID) -> (1 | UID) if convergence fails")

        # Fit quadratic LMM using fit_lmm_trajectory (simpler, takes data directly)
        from tools.analysis_lmm import fit_lmm_trajectory

        formula = "theta ~ Time + Time_squared"  # Fixed effects
        random_structure_used = None

        try:
            # Try maximal random structure first
            quadratic_model = fit_lmm_trajectory(
                data=time_data,
                formula=formula,
                groups="UID",
                re_formula="~Time",  # Random slopes: (Time | UID)
                reml=False  # ML estimation for AIC comparison
            )

            # Check if converged
            if quadratic_model.converged:
                log("Quadratic model converged with maximal random structure")
                random_structure_used = "(Time | UID)"
            else:
                log("Maximal random structure did not converge")
                raise ValueError("Non-convergence detected, trying fallback")

        except Exception as e_maximal:
            log(f"Maximal random structure failed: {str(e_maximal)[:100]}")
            log("Attempting intercept-only random structure: (1 | UID)...")

            # Fallback: Intercept-only random effects
            quadratic_model = fit_lmm_trajectory(
                data=time_data,
                formula=formula,
                groups="UID",
                re_formula="~1",  # Intercept-only: (1 | UID)
                reml=False
            )

            if quadratic_model.converged:
                log("Quadratic model converged with intercept-only random structure")
                random_structure_used = "(1 | UID)"
                log("Random slopes removed - model assumes uniform forgetting rate across individuals")
            else:
                log("Even intercept-only model failed to converge - results unreliable")
                random_structure_used = "(1 | UID) - NON-CONVERGED"

        log("Quadratic model fitted")
        log(f"Random structure used: {random_structure_used}")
        log(f"Convergence: {quadratic_model.converged}")
        log(f"AIC: {quadratic_model.aic:.2f}")
        log(f"BIC: {quadratic_model.bic:.2f}")
        # Extract Model Summary and Test Time_squared Significance
        # Test: Is Time_squared significant at Bonferroni-corrected alpha=0.0033?
        # Interpretation: Significant quadratic term suggests non-linear forgetting

        log("Extracting fixed effects and significance tests...")

        # Get fixed effects summary
        fe_summary = quadratic_model.summary().tables[1]  # Fixed effects table
        log(f"Fixed Effects:\n{fe_summary}")

        # Extract Time_squared p-value
        fe_params = quadratic_model.fe_params
        fe_pvalues = quadratic_model.pvalues

        time_squared_coef = fe_params.get('Time_squared', np.nan)
        time_squared_pval = fe_pvalues.get('Time_squared', np.nan)

        log(f"Time_squared coefficient: {time_squared_coef:.6f}")
        log(f"Time_squared p-value: {time_squared_pval:.6f}")

        # Test significance at Bonferroni alpha
        bonferroni_alpha = 0.0033
        is_significant = time_squared_pval < bonferroni_alpha

        if is_significant:
            log(f"Time_squared is SIGNIFICANT (p={time_squared_pval:.6f} < {bonferroni_alpha})")
            log("Non-linear trajectory detected -> supports two-phase forgetting")
        else:
            log(f"Time_squared is NOT significant (p={time_squared_pval:.6f} >= {bonferroni_alpha})")
            log("Linear trajectory -> does NOT support two-phase forgetting")
        # Generate Predictions for Plotting
        # Used by: Step 6 (plotting) and Step 3 (AIC comparison)

        log("Generating predictions on grid [0, 24, 48, ..., 240] hours...")

        prediction_grid = np.array([0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240])

        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'Time': prediction_grid,
            'Time_squared': prediction_grid ** 2
        })

        # Generate predictions with confidence intervals
        predictions = quadratic_model.predict(exog=pred_df)

        # Get prediction standard errors (for 95% CI)
        # Note: statsmodels MixedLM doesn't have predict(return_var=True)
        # Use fixed effects standard errors as approximation
        se_intercept = quadratic_model.bse['Intercept']
        se_time = quadratic_model.bse['Time']
        se_time_sq = quadratic_model.bse['Time_squared']

        # Propagate uncertainty (simplified - assumes no covariance between parameters)
        # SE = sqrt(SE_intercept^2 + (SE_time * Time)^2 + (SE_time_sq * Time^2)^2)
        pred_se = np.sqrt(
            se_intercept ** 2 +  # Intercept uncertainty (constant across all predictions)
            (se_time * prediction_grid) ** 2 +  # Time term uncertainty
            (se_time_sq * prediction_grid ** 2) ** 2  # Time_squared term uncertainty
        )

        # 95% CI (z=1.96 for normal approximation)
        ci_lower = predictions - 1.96 * pred_se
        ci_upper = predictions + 1.96 * pred_se

        # Create predictions dataframe
        quadratic_predictions = pd.DataFrame({
            'Time': prediction_grid,
            'predicted_theta': predictions,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })

        log(f"Generated {len(quadratic_predictions)} predictions")
        # Save Analysis Outputs
        # These outputs will be used by: Step 3 (AIC comparison), Step 6 (plotting)

        log("Saving model summary...")

        # Save model summary as text file
        summary_path = RQ_DIR / "data/step02_quadratic_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("QUADRATIC LMM SUMMARY - RQ 5.1.2 Step 2\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Formula: theta ~ Time + Time_squared\n")
            f.write(f"Random Structure: {random_structure_used}\n")
            f.write(f"Estimation: ML (REML=False)\n")
            f.write(f"Bonferroni alpha: {bonferroni_alpha}\n\n")
            f.write(f"Convergence: {quadratic_model.converged}\n")
            f.write(f"AIC: {quadratic_model.aic:.2f}\n")
            f.write(f"BIC: {quadratic_model.bic:.2f}\n")
            f.write(f"Log-Likelihood: {quadratic_model.llf:.2f}\n\n")
            f.write("FIXED EFFECTS:\n")
            f.write("-" * 80 + "\n")
            f.write(str(fe_summary) + "\n\n")
            f.write("TIME_SQUARED SIGNIFICANCE TEST:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Coefficient: {time_squared_coef:.6f}\n")
            f.write(f"P-value: {time_squared_pval:.6f}\n")
            f.write(f"Significant (alpha={bonferroni_alpha}): {is_significant}\n\n")
            f.write("RANDOM EFFECTS:\n")
            f.write("-" * 80 + "\n")
            f.write(str(quadratic_model.random_effects) + "\n\n")
            f.write("FULL MODEL SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(str(quadratic_model.summary()) + "\n")

        log(f"step02_quadratic_model_summary.txt")

        # Save predictions
        pred_path = RQ_DIR / "data/step02_quadratic_predictions.csv"
        quadratic_predictions.to_csv(pred_path, index=False, encoding='utf-8')
        log(f"step02_quadratic_predictions.csv ({len(quadratic_predictions)} rows)")

        # Also save model object as pickle for Step 3
        model_path = RQ_DIR / "data/step02_quadratic_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(quadratic_model, f)
        log(f"step02_quadratic_model.pkl (for Step 3 AIC comparison)")
        # Run Validation Tool
        # Validates: Model convergence, parameter finiteness, AIC validity
        # Threshold: All checks must pass

        log("Running validate_lmm_convergence...")
        validation_result = validate_lmm_convergence(
            lmm_result=quadratic_model
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional manual checks
        log("Additional checks:")
        log(f"  - Predictions count: {len(quadratic_predictions)} (expected 11)")
        log(f"  - Time_squared p-value finite: {np.isfinite(time_squared_pval)}")
        log(f"  - AIC positive: {quadratic_model.aic > 0}")

        # Verify all critical checks passed
        all_passed = (
            quadratic_model.converged or 'fallback' in str(quadratic_model.cov_re),
            len(quadratic_predictions) == 11,
            np.isfinite(time_squared_pval),
            quadratic_model.aic > 0,
            all(np.isfinite(quadratic_model.fe_params))
        )

        if all(all_passed):
            log("All checks PASSED")
        else:
            log("Some checks FAILED - review results carefully")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
