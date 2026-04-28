#!/usr/bin/env python3
"""Test Time Effect on HCE Rate (Dual P-Values per D068): Extract TSVR fixed effect from LMM and compute dual p-values (Wald + LRT) per"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import fit_lmm_trajectory, extract_fixed_effects_from_lmm

from tools.validation import validate_hypothesis_test_dual_pvalues

# Import statsmodels for LMM
import statsmodels.formula.api as smf
from scipy import stats

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_test_time_effect.log"


# Logging Function

def log(msg, flush=True):
    """Write to both log file and console with UTF-8 encoding."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=flush)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Test Time Effect on HCE Rate")
        # Load Input Data

        log("Loading HCE rates from Step 01...")
        input_path = RQ_DIR / "data" / "step01_hce_rates.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_hce = pd.read_csv(input_path)
        log(f"step01_hce_rates.csv ({len(df_hce)} rows, {len(df_hce.columns)} cols)")

        # Validate expected columns
        required_cols = ['UID', 'TEST', 'TSVR', 'HCE_rate']
        missing_cols = [col for col in required_cols if col not in df_hce.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"{df_hce['UID'].nunique()} participants, {df_hce['TEST'].nunique()} test sessions")
        # Fit Full LMM (with TSVR fixed effect)

        log("Fitting Full LMM: HCE_rate ~ TSVR + (TSVR | UID)...")

        # Fit full model using statsmodels directly (for LRT comparison)
        # Formula: HCE_rate ~ TSVR + (TSVR | UID)
        # REML=False for LRT (must use ML estimation for nested model comparison)
        full_model = smf.mixedlm(
            formula='HCE_rate ~ TSVR',
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~TSVR'
        )

        full_result = full_model.fit(method=['lbfgs'], reml=False)

        if not full_result.converged:
            log("Full model did not converge (continuing anyway)")

        log(f"[FULL MODEL] Converged: {full_result.converged}")
        log(f"[FULL MODEL] Log-likelihood: {full_result.llf:.2f}")
        log(f"[FULL MODEL] AIC: {full_result.aic:.2f}")
        # Extract TSVR Fixed Effect (Wald p-value)

        log("Extracting TSVR fixed effect from full model...")

        fe_table = extract_fixed_effects_from_lmm(full_result)

        # Find TSVR row in fixed effects table
        tsvr_row = fe_table[fe_table['Term'] == 'TSVR']

        if len(tsvr_row) == 0:
            raise ValueError("TSVR coefficient not found in fixed effects table")

        tsvr_coef = tsvr_row['Coef'].values[0]
        tsvr_se = tsvr_row['Std_Err'].values[0]
        tsvr_p_wald = tsvr_row['P_value'].values[0]

        log(f"[TSVR EFFECT] Coefficient: {tsvr_coef:.6f}")
        log(f"[TSVR EFFECT] SE: {tsvr_se:.6f}")
        log(f"[TSVR EFFECT] p_wald: {tsvr_p_wald:.6f}")
        # Fit Reduced LMM (no TSVR fixed effect)
        # Reduced model: HCE_rate ~ 1 + (TSVR | UID) (intercept only, no TSVR slope)

        log("Fitting Reduced LMM: HCE_rate ~ 1 + (TSVR | UID)...")

        reduced_model = smf.mixedlm(
            formula='HCE_rate ~ 1',  # Intercept only (no TSVR fixed effect)
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~TSVR'  # Keep same random structure for valid comparison
        )

        reduced_result = reduced_model.fit(method=['lbfgs'], reml=False)

        if not reduced_result.converged:
            log("Reduced model did not converge (continuing anyway)")

        log(f"[REDUCED MODEL] Converged: {reduced_result.converged}")
        log(f"[REDUCED MODEL] Log-likelihood: {reduced_result.llf:.2f}")
        log(f"[REDUCED MODEL] AIC: {reduced_result.aic:.2f}")
        # Compute Likelihood Ratio Test (LRT) p-value
        # LRT statistic: -2 * (log-likelihood_reduced - log-likelihood_full)
        # Degrees of freedom: Number of fixed effects added (1 for TSVR)
        # Distribution: Chi-square with df degrees of freedom

        log("Computing Likelihood Ratio Test for TSVR effect...")

        # LRT chi-square statistic
        lrt_statistic = -2 * (reduced_result.llf - full_result.llf)

        # Degrees of freedom (1 parameter added: TSVR slope)
        df_lrt = 1

        # p-value from chi-square distribution
        p_lrt = 1 - stats.chi2.cdf(lrt_statistic, df_lrt)

        log(f"Chi-square statistic: {lrt_statistic:.4f}")
        log(f"Degrees of freedom: {df_lrt}")
        log(f"p_lrt: {p_lrt:.6f}")
        # Format Output with Dual P-Values (Decision D068)
        # Output: Single row with TSVR effect, coefficient, SE, p_wald, p_lrt, significant
        # significant: True if EITHER p_wald < 0.05 OR p_lrt < 0.05

        log("Creating output DataFrame with dual p-values...")

        # Determine significance (either p-value < 0.05)
        significant = (tsvr_p_wald < 0.05) or (p_lrt < 0.05)

        df_time_effect = pd.DataFrame({
            'effect': ['TSVR'],
            'coefficient': [tsvr_coef],
            'SE': [tsvr_se],
            'p_wald': [tsvr_p_wald],
            'p_lrt': [p_lrt],
            'significant': [significant]
        })

        log(f"Time effect results:")
        log(f"  - effect: TSVR")
        log(f"  - coefficient: {tsvr_coef:.6f}")
        log(f"  - SE: {tsvr_se:.6f}")
        log(f"  - p_wald: {tsvr_p_wald:.6f}")
        log(f"  - p_lrt: {p_lrt:.6f}")
        log(f"  - significant: {significant}")
        # Save Output
        # File: data/step03_time_effect.csv
        # Contains: TSVR effect with dual p-values per Decision D068

        output_path = RQ_DIR / "data" / "step03_time_effect.csv"
        df_time_effect.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_time_effect.csv ({len(df_time_effect)} rows, {len(df_time_effect.columns)} cols)")
        # Validate Output (Decision D068 Dual P-Values)
        # Validates: Required term present, both p-values present, valid ranges

        log("Running validate_hypothesis_test_dual_pvalues...")

        # Prepare validation input (rename columns to match validator expectations)
        df_validation = df_time_effect.copy()
        df_validation['term'] = df_validation['effect']  # Validator expects 'term' column
        df_validation['p_uncorrected'] = df_validation['p_wald']  # Rename for validator
        df_validation['p_bonferroni'] = df_validation['p_lrt']  # Use LRT as "correction" (conceptually different but dual reporting)

        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=df_validation,
            required_terms=['TSVR'],
            alpha_bonferroni=0.05
        )

        # Report validation results
        if validation_result['valid']:
            log(f"[VALIDATION - PASS] {validation_result['message']}")
        else:
            log(f"[VALIDATION - FAIL] {validation_result['message']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional validation checks (per 4_analysis.yaml criteria)
        log("Checking additional criteria...")

        # Check p-values in [0, 1] range
        if not (0 <= tsvr_p_wald <= 1):
            raise ValueError(f"p_wald out of range [0, 1]: {tsvr_p_wald}")
        if not (0 <= p_lrt <= 1):
            raise ValueError(f"p_lrt out of range [0, 1]: {p_lrt}")
        log("[VALIDATION - PASS] Both p-values in [0, 1] range")

        # Check SE > 0
        if tsvr_se <= 0:
            raise ValueError(f"SE must be positive, got: {tsvr_se}")
        log("[VALIDATION - PASS] SE > 0 (positive standard error)")

        # Check coefficient in scientifically reasonable range [-0.01, 0.01]
        if not (-0.01 <= tsvr_coef <= 0.01):
            log(f"[VALIDATION - WARNING] Coefficient {tsvr_coef:.6f} outside typical range [-0.01, 0.01] (not fatal)")
        else:
            log(f"[VALIDATION - PASS] Coefficient in reasonable range [-0.01, 0.01]")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
