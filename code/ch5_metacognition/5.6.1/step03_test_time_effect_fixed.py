#!/usr/bin/env python3
"""Test Time Effect on HCE Rate (Dual P-Values per D068): Extract Days fixed effect from LMM and compute dual p-values (Wald + LRT) per"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM
import statsmodels.formula.api as smf
from scipy import stats

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1
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
        # Clear old log
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w') as f:
            f.write("")

        log("Step 03 (FIXED): Test Time Effect on HCE Rate")
        log("CRITICAL FIX: Using Days (TSVR/24) consistently with step02")
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
        # Convert TSVR to Days (CRITICAL - must match step02)
        log("Converting TSVR hours to Days...")
        df_hce['Days'] = df_hce['TSVR'] / 24.0
        log(f"Days range: [{df_hce['Days'].min():.2f}, {df_hce['Days'].max():.2f}]")
        log(f"TSVR range: [{df_hce['TSVR'].min():.2f}, {df_hce['TSVR'].max():.2f}] hours")
        # Fit Full LMM (with Days fixed effect) using REML
        log("Fitting Full LMM: HCE_rate ~ Days + (Days | UID) using REML...")

        full_model = smf.mixedlm(
            formula='HCE_rate ~ Days',
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~Days'
        )

        # Use REML=True for stable variance estimation (like step02)
        full_result_reml = full_model.fit(method='powell', reml=True)

        log(f"[FULL MODEL REML] Converged: {full_result_reml.converged}")
        log(f"[FULL MODEL REML] Log-likelihood (REML): {full_result_reml.llf:.2f}")

        # Extract Days coefficient from REML model (authoritative)
        days_coef = full_result_reml.params['Days']
        days_se = full_result_reml.bse['Days']
        days_z = full_result_reml.tvalues['Days']
        days_p_wald_reml = full_result_reml.pvalues['Days']

        log(f"[REML EFFECT] Days coefficient: {days_coef:.6f}")
        log(f"[REML EFFECT] SE: {days_se:.6f}")
        log(f"[REML EFFECT] z: {days_z:.4f}")
        log(f"[REML EFFECT] p_wald (REML): {days_p_wald_reml:.6f}")
        # Fit Full LMM (with Days fixed effect) using ML for LRT
        log("Fitting Full LMM: HCE_rate ~ Days + (Days | UID) using ML for LRT...")

        full_result_ml = full_model.fit(method='powell', reml=False)

        log(f"[FULL MODEL ML] Converged: {full_result_ml.converged}")
        log(f"[FULL MODEL ML] Log-likelihood: {full_result_ml.llf:.2f}")
        log(f"[FULL MODEL ML] AIC: {full_result_ml.aic:.2f}")
        # Fit Reduced LMM (no Days fixed effect) using ML
        log("Fitting Reduced LMM: HCE_rate ~ 1 + (Days | UID) using ML...")

        reduced_model = smf.mixedlm(
            formula='HCE_rate ~ 1',  # Intercept only
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~Days'
        )

        reduced_result_ml = reduced_model.fit(method='powell', reml=False)

        log(f"[REDUCED MODEL ML] Converged: {reduced_result_ml.converged}")
        log(f"[REDUCED MODEL ML] Log-likelihood: {reduced_result_ml.llf:.2f}")
        log(f"[REDUCED MODEL ML] AIC: {reduced_result_ml.aic:.2f}")
        # Compute Likelihood Ratio Test (LRT) p-value
        log("Computing Likelihood Ratio Test for Days effect...")

        # LRT chi-square statistic (must be positive if full model is better)
        lrt_statistic = -2 * (reduced_result_ml.llf - full_result_ml.llf)

        # Degrees of freedom (1 parameter added: Days slope)
        df_lrt = 1

        # Ensure LRT statistic is non-negative (handle numerical issues)
        if lrt_statistic < 0:
            log(f"LRT statistic negative ({lrt_statistic:.4f}), setting to 0 (boundary case)")
            lrt_statistic = 0.0

        # p-value from chi-square distribution
        p_lrt = 1 - stats.chi2.cdf(lrt_statistic, df_lrt)

        log(f"Chi-square statistic: {lrt_statistic:.4f}")
        log(f"Degrees of freedom: {df_lrt}")
        log(f"p_lrt: {p_lrt:.6f}")
        # Determine authoritative p-values
        # PRIMARY: REML p-value (more stable for small variance components)
        # SECONDARY: LRT p-value (if ML converged properly)

        log("Selecting authoritative p-values...")

        # Check ML convergence quality
        ml_valid = (
            full_result_ml.converged and
            reduced_result_ml.converged and
            lrt_statistic >= 0
        )

        log(f"Full ML converged: {full_result_ml.converged}")
        log(f"Reduced ML converged: {reduced_result_ml.converged}")
        log(f"LRT statistic valid (>=0): {lrt_statistic >= 0}")
        log(f"ML-based LRT valid: {ml_valid}")

        # Use REML p-value as primary (always valid if step02 converged)
        p_wald_final = days_p_wald_reml

        if ml_valid:
            p_lrt_final = p_lrt
            log(f"Using ML-based LRT p-value: {p_lrt_final:.6f}")
        else:
            # Fall back to REML p-value (same as Wald, indicating D068 partial compliance)
            p_lrt_final = days_p_wald_reml
            log(f"ML LRT invalid, falling back to REML p-value for both")
            log(f"D068 dual p-value reporting degraded (both p-values from REML)")
        # Format Output with Dual P-Values (Decision D068)
        log("Creating output DataFrame with dual p-values...")

        # Determine significance (REML p-value is authoritative)
        significant = p_wald_final < 0.05

        df_time_effect = pd.DataFrame({
            'effect': ['Days'],
            'coefficient': [days_coef],
            'SE': [days_se],
            'p_wald': [p_wald_final],
            'p_lrt': [p_lrt_final],
            'significant': [significant],
            'ml_valid': [ml_valid],
            'note': ['REML primary, ML LRT valid' if ml_valid else 'REML primary, ML LRT invalid']
        })

        log(f"Time effect results:")
        log(f"  - effect: Days")
        log(f"  - coefficient: {days_coef:.6f}")
        log(f"  - SE: {days_se:.6f}")
        log(f"  - p_wald (REML): {p_wald_final:.6f}")
        log(f"  - p_lrt: {p_lrt_final:.6f}")
        log(f"  - significant: {significant}")
        log(f"  - ml_valid: {ml_valid}")
        # Save Output
        output_path = RQ_DIR / "data" / "step03_time_effect.csv"
        df_time_effect.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step03_time_effect.csv ({len(df_time_effect)} rows, {len(df_time_effect.columns)} cols)")
        # Validate Output
        log("Running validation checks...")

        # Check p-values in [0, 1] range
        if not (0 <= p_wald_final <= 1):
            raise ValueError(f"p_wald out of range [0, 1]: {p_wald_final}")
        if not (0 <= p_lrt_final <= 1):
            raise ValueError(f"p_lrt out of range [0, 1]: {p_lrt_final}")
        log("[VALIDATION - PASS] Both p-values in [0, 1] range")

        # Check SE > 0
        if days_se <= 0:
            raise ValueError(f"SE must be positive, got: {days_se}")
        log("[VALIDATION - PASS] SE > 0 (positive standard error)")

        # Check coefficient in scientifically reasonable range
        if not (-0.1 <= days_coef <= 0.1):
            log(f"[VALIDATION - WARNING] Coefficient {days_coef:.6f} outside typical range [-0.1, 0.1]")
        else:
            log(f"[VALIDATION - PASS] Coefficient in reasonable range")

        # Check consistency with step02
        log("Checking consistency with step02...")
        log(f"Step02 result: β=-0.003, p<.001")
        log(f"Step03 result: β={days_coef:.6f}, p={p_wald_final:.6f}")

        # Both should show significant negative effect
        if days_coef < 0 and p_wald_final < 0.05:
            log("[VALIDATION - PASS] Consistent with step02 (significant negative Days effect)")
        else:
            log("[VALIDATION - WARNING] Potential inconsistency with step02")

        log("Step 03 (FIXED) complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
