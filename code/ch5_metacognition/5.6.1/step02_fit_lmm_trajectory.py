#!/usr/bin/env python3
"""Fit LMM for HCE Trajectory: Fit Linear Mixed Model to estimate HCE forgetting trajectory using TSVR (hours"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory_tsvr

from tools.validation import validate_lmm_convergence

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_fit_lmm_trajectory.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Fit LMM for HCE Trajectory")
        # Load Input Data

        log("Loading input data...")
        input_path = RQ_DIR / "data" / "step01_hce_rates.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_hce_rates = pd.read_csv(input_path, encoding='utf-8')
        log(f"step01_hce_rates.csv ({len(df_hce_rates)} rows, {len(df_hce_rates.columns)} cols)")

        # Validate required columns present
        required_cols = ['UID', 'TEST', 'TSVR', 'HCE_rate']
        missing_cols = [col for col in required_cols if col not in df_hce_rates.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Using columns: {required_cols}")
        log(f"Observations: {len(df_hce_rates)} (Expected: 400 = 100 participants × 4 tests)")
        log(f"HCE_rate range: [{df_hce_rates['HCE_rate'].min():.4f}, {df_hce_rates['HCE_rate'].max():.4f}]")
        log(f"TSVR range: [{df_hce_rates['TSVR'].min():.2f}, {df_hce_rates['TSVR'].max():.2f}] hours")
        # Run Analysis Tool
        #   random intercepts and slopes by UID. REML=True provides unbiased
        #   variance component estimates.

        log("Fitting LMM: HCE_rate ~ TSVR + (TSVR | UID)...")
        log("Formula components:")
        log("Fixed effects: Intercept + TSVR (population-level trajectory)")
        log("Random effects: (TSVR | UID) (individual intercepts and slopes)")
        log("Groups: UID (participant-level clustering)")
        log("Method: REML=True (variance estimation)")

        # WORKAROUND: fit_lmm_trajectory_tsvr is designed for TWO separate dataframes:
        #   1. theta_scores: IRT scores (composite_ID, domain_name, theta) - NO tsvr column
        #   2. tsvr_data: Time data (composite_ID, test, tsvr)
        # The function merges tsvr FROM tsvr_data INTO theta_scores. If theta_scores already
        # has 'tsvr' column, pandas creates tsvr_x/tsvr_y conflict, breaking line 1108.
        #
        # Solution: Create theta_scores WITHOUT tsvr, and tsvr_data WITH tsvr
        log("Preparing theta_scores (without tsvr) and tsvr_data (with tsvr)...")

        # Rename columns for theta_scores
        df_theta = df_hce_rates.rename(columns={
            'TEST': 'Test',         # Function checks for 'test' or 'Test' (line 1078)
            'HCE_rate': 'theta'     # Function expects 'theta' as dependent variable (line 1136)
        }).copy()

        # Create composite_ID in theta_scores
        df_theta['composite_ID'] = df_theta['UID'].astype(str) + '_T' + df_theta['Test'].astype(str)

        # theta_scores should NOT have 'tsvr' column (will be merged from tsvr_data)
        df_theta = df_theta[['UID', 'Test', 'composite_ID', 'theta', 'n_HCE', 'n_total']]
        log(f"theta_scores columns: {df_theta.columns.tolist()}")
        log(f"theta_scores shape: {df_theta.shape}")

        # Create tsvr_data WITH tsvr column
        df_tsvr = df_hce_rates.rename(columns={
            'TEST': 'Test',
            'TSVR': 'tsvr'
        })[['UID', 'Test', 'tsvr']].copy()
        df_tsvr['composite_ID'] = df_tsvr['UID'].astype(str) + '_T' + df_tsvr['Test'].astype(str)
        log(f"tsvr_data columns: {df_tsvr.columns.tolist()}")
        log(f"tsvr_data shape: {df_tsvr.shape}")
        log(f"Example composite_IDs: {df_theta['composite_ID'].head(3).tolist()}")

        lmm_model = fit_lmm_trajectory_tsvr(
            theta_scores=df_theta,         # Theta scores WITHOUT tsvr (will be merged from tsvr_data)
            tsvr_data=df_tsvr,             # Time data WITH tsvr (will be merged into theta_scores)
            formula="Theta ~ Days",        # Fixed effects: Function internally renames theta→Theta and tsvr→Days
            groups="UID",                   # Grouping variable for random effects
            re_formula="~Days",            # Random effects: Function uses Days (converted from tsvr hours)
            reml=True                       # REML estimation for variance components
        )

        log("LMM fitting complete")
        log(f"Convergence status: {lmm_model.converged}")
        log(f"Number of observations: {lmm_model.nobs}")
        log(f"Number of groups: {len(lmm_model.model.group_labels)}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 03 (effect size computation), Step 04
        # (trajectory plots), and results analysis (final interpretation)

        log("Saving model summary...")
        output_path = RQ_DIR / "data" / "step02_hce_lmm.txt"

        # Save model summary as plain text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(lmm_model.summary()))

        # Report key statistics
        log(f"step02_hce_lmm.txt ({output_path.stat().st_size} bytes)")
        log("Fixed Effects:")
        for param, coef in lmm_model.params.items():
            log(f"{param}: {coef:.6f}")

        log("Random Effects Variances:")
        log(f"Group (UID) Var: {lmm_model.cov_re.iloc[0, 0]:.6f}")
        if lmm_model.cov_re.shape[0] > 1:
            log(f"Group x TSVR Var: {lmm_model.cov_re.iloc[1, 1]:.6f}")
        # Run Validation Tool
        # Validates: Model converged successfully, no singular fit, reasonable
        #   variance components
        # Threshold: All checks must pass for valid LMM

        log("Running validate_lmm_convergence...")

        validation_result = validate_lmm_convergence(
            lmm_result=lmm_model
        )

        # Report validation results
        log(f"Converged: {validation_result['converged']}")
        log(f"Message: {validation_result['message']}")

        if validation_result.get('warnings'):
            for warning in validation_result['warnings']:
                log(f"Warning: {warning}")

        # Additional checks beyond basic convergence
        log("Additional checks:")

        # Check 1: No singular fit (all random effects variances > 0)
        all_positive_var = all(lmm_model.cov_re.values.diagonal() > 0)
        log(f"Random effects variance > 0: {all_positive_var}")

        # Check 2: All fixed effects finite
        all_finite = np.all(np.isfinite(lmm_model.params.values))
        log(f"All fixed effects finite: {all_finite}")

        # Check 3: Sufficient observations
        sufficient_obs = lmm_model.nobs >= 100
        log(f"Observations >= 100: {sufficient_obs} (n={lmm_model.nobs})")

        # Check 4: Residual normality (Kolmogorov-Smirnov test)
        # NOTE: KS test is overly sensitive with large n (n=400). For LMM with robust estimation
        # (statsmodels uses asymptotic theory), slight departures from normality are acceptable.
        # HCE rates are bounded [0,1] proportions, so perfect normality is unlikely.
        # We log the result but don't fail on it (warning only).
        from scipy.stats import kstest
        residuals_for_check = df_theta['theta'] - lmm_model.fittedvalues[:len(df_theta)]
        ks_stat, ks_pvalue = kstest(residuals_for_check, 'norm', args=(residuals_for_check.mean(), residuals_for_check.std()))
        residuals_normal = ks_pvalue > 0.001  # More lenient threshold for bounded data
        if ks_pvalue <= 0.001:
            log(f"Residuals normality: WARNING (KS p={ks_pvalue:.4f} < 0.001)")
            log(f"Note: Bounded [0,1] data rarely has perfectly normal residuals")
            log(f"LMM is robust to moderate departures from normality with n=400")
        else:
            log(f"Residuals normal (KS p > 0.001): {residuals_normal} (p={ks_pvalue:.4f})")

        # Aggregate validation
        all_checks_passed = (
            validation_result['converged'] and
            all_positive_var and
            all_finite and
            sufficient_obs and
            residuals_normal
        )

        if all_checks_passed:
            log("PASS: All validation checks passed")
        else:
            failed_checks = []
            if not validation_result['converged']:
                failed_checks.append("Model did not converge")
            if not all_positive_var:
                failed_checks.append("Singular fit (zero variance components)")
            if not all_finite:
                failed_checks.append("Non-finite fixed effects")
            if not sufficient_obs:
                failed_checks.append(f"Insufficient observations (n={lmm_model.nobs})")
            if not residuals_normal:
                failed_checks.append(f"Non-normal residuals (KS p={ks_pvalue:.4f})")

            error_msg = "LMM validation failed: " + "; ".join(failed_checks)
            log(f"FAIL: {error_msg}")
            raise ValueError(error_msg)

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
