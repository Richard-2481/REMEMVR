#!/usr/bin/env python3
"""fit_lmm: Fit Linear Mixed Model (LMM) with full 3-way Age_c x LocationType x Time"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import mixedlm

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lmm.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit LMM with 3-Way Age x LocationType x Time Interactions")
        # Load Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step01_lmm_input.csv"
        lmm_input = pd.read_csv(input_path)
        log(f"step01_lmm_input.csv ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Validate expected structure
        expected_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours', 'log_TSVR',
                         'Age', 'Age_c', 'LocationType', 'theta', 'se']
        if list(lmm_input.columns) != expected_cols:
            raise ValueError(f"Column mismatch. Expected: {expected_cols}, Got: {list(lmm_input.columns)}")
        if len(lmm_input) != 800:
            raise ValueError(f"Expected 800 rows, got {len(lmm_input)}")
        log(f"Input structure validated: 800 rows, 10 columns")
        # Fit LMM with Formula-Based Specification
        # Fixed effects: Main effects (TSVR_hours, log_TSVR, Age_c, LocationType),
        #                2-way interactions (TSVR:Age_c, log_TSVR:Age_c, TSVR:LocationType,
        #                log_TSVR:LocationType, Age_c:LocationType),
        #                3-way interactions (TSVR:Age_c:LocationType, log_TSVR:Age_c:LocationType)
        # Random effects: Random intercept + TSVR_hours slope per participant (UID)

        log("Fitting LMM with 3-way Age_c x LocationType x Time interactions...")

        # Define formula (from 4_analysis.yaml specification)
        formula = (
            "theta ~ TSVR_hours + log_TSVR + Age_c + LocationType + "
            "TSVR_hours:Age_c + log_TSVR:Age_c + "
            "TSVR_hours:LocationType + log_TSVR:LocationType + "
            "Age_c:LocationType + "
            "TSVR_hours:Age_c:LocationType + log_TSVR:Age_c:LocationType"
        )

        # Random effects formula: random intercept + TSVR_hours slope per UID
        re_formula = "~TSVR_hours"

        log(f"Fixed effects: {formula}")
        log(f"Random effects: {re_formula}")
        log(f"Grouping variable: UID")

        # Fit model with ML estimation (REML=False for AIC/BIC comparison)
        # method='lbfgs' is default optimizer, maxiter=1000 for convergence
        lmm_model = mixedlm(
            formula=formula,
            data=lmm_input,
            groups=lmm_input['UID'],
            re_formula=re_formula
        ).fit(reml=False, method='lbfgs', maxiter=1000)

        log(f"Model fitting complete")
        log(f"Converged: {lmm_model.converged}")
        # Extract Model Components
        # Extract convergence status, AIC/BIC, fixed effects, random variances

        log("Extracting model components...")

        # Fixed effects table
        # Extract fixed effect names from fe_params index
        fe_names = lmm_model.fe_params.index.tolist()
        n_fe = len(fe_names)

        # Get confidence intervals
        conf_int = lmm_model.conf_int()

        # Extract only the fixed effects portion (first n_fe rows)
        fixed_effects = pd.DataFrame({
            'term': fe_names,
            'coef': lmm_model.fe_params.values,
            'se': lmm_model.bse_fe.values,
            'z': lmm_model.tvalues.iloc[:n_fe].values,
            'p': lmm_model.pvalues.iloc[:n_fe].values,
            'ci_lower': conf_int.iloc[:n_fe, 0].values,
            'ci_upper': conf_int.iloc[:n_fe, 1].values
        })
        log(f"Fixed effects: {len(fixed_effects)} terms")

        # Model summary statistics
        aic = lmm_model.aic
        bic = lmm_model.bic
        log_likelihood = lmm_model.llf
        n_obs = lmm_model.nobs
        n_groups = len(lmm_model.model.group_labels)  # Correct way to get number of groups

        log(f"AIC: {aic:.2f}, BIC: {bic:.2f}, LogLik: {log_likelihood:.2f}")
        log(f"N observations: {n_obs}, N groups (UIDs): {n_groups}")

        # Random effects variance components
        # statsmodels stores these in cov_re (covariance matrix of random effects)
        random_effects_cov = lmm_model.cov_re
        log(f"Variance components extracted")
        # Save Outputs
        # Save model object (.pkl), summary text (.txt), fixed effects table (.csv)

        log("Saving model outputs...")

        # Save model object using statsmodels save method (NOT pickle.load)
        # CRITICAL: Use MixedLMResults.save() method to avoid patsy/eval errors
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"
        lmm_model.save(str(model_path))
        log(f"Model object: step02_lmm_model.pkl")

        # Save model summary as text
        summary_path = RQ_DIR / "data" / "step02_lmm_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LMM SUMMARY - RQ 5.5.3 Step 02\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model Formula:\n{formula}\n\n")
            f.write(f"Random Effects Formula:\n{re_formula}\n\n")
            f.write(f"Grouping Variable: UID\n\n")
            f.write(f"Convergence Status: {lmm_model.converged}\n")
            f.write(f"N Observations: {n_obs}\n")
            f.write(f"N Groups (UIDs): {n_groups}\n\n")
            f.write(f"Model Fit Statistics:\n")
            f.write(f"  AIC: {aic:.4f}\n")
            f.write(f"  BIC: {bic:.4f}\n")
            f.write(f"  Log-Likelihood: {log_likelihood:.4f}\n\n")
            f.write("Fixed Effects:\n")
            f.write(fixed_effects.to_string(index=False))
            f.write("\n\n")
            f.write("Random Effects Variance-Covariance Matrix:\n")
            f.write(str(random_effects_cov))
            f.write("\n\n")
            f.write("=" * 80 + "\n")
        log(f"Model summary: step02_lmm_summary.txt")

        # Save fixed effects table
        fixed_effects_path = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fixed_effects.to_csv(fixed_effects_path, index=False, encoding='utf-8')
        log(f"Fixed effects table: step02_fixed_effects.csv ({len(fixed_effects)} rows)")
        # Validate Model Convergence
        # Validates: Model converged successfully with positive variance components

        log("Validating model convergence...")
        validation_result = validate_lmm_convergence(lmm_model)

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check if validation passed
        if validation_result.get('converged', False):
            log("Model converged successfully")
        else:
            log("Model did not converge - results may be unreliable")

        # Additional validation checks
        log("Checking fixed effects count...")
        if len(fixed_effects) != 12:
            log(f"Expected 12 fixed effects, got {len(fixed_effects)}")
        else:
            log(f"12 fixed effects present as expected")

        log("Checking standard errors...")
        if (fixed_effects['se'] <= 0).any():
            log(f"Some standard errors are non-positive")
        else:
            log(f"All standard errors positive")

        log("Checking AIC/BIC...")
        if np.isnan(aic) or np.isnan(bic) or np.isinf(aic) or np.isinf(bic):
            log(f"AIC or BIC is NaN/inf")
        else:
            log(f"AIC and BIC are finite")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
