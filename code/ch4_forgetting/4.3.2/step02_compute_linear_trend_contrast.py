#!/usr/bin/env python3
"""Compute Linear Trend Contrast: Test linear trend contrast within RQ 5.3 LMM using polynomial weights [-1, 0, +1]."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statsmodels for loading model
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.2
RQ3_DIR = RQ_DIR.parent / "5.3.1"  # results/ch5/5.3.1 (dependency)
LOG_FILE = RQ_DIR / "logs" / "step02_compute_linear_trend_contrast.log"

# Contrast weights (linear trend)
CONTRAST_WEIGHTS = {
    'free_recall': -1,
    'cued_recall': 0,
    'recognition': +1
}

# Significance thresholds
ALPHA_UNCORRECTED = 0.05
BONFERRONI_N_TESTS = 15  # Thesis-wide correction
ALPHA_BONFERRONI = 0.05 / BONFERRONI_N_TESTS  # = 0.0033
Z_CRITICAL = 1.96

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file for fresh run
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 02: Compute Linear Trend Contrast")
        # Verify Step 1 Complete and Load Data
        log("Verifying Step 1 completed...")

        marginal_means_path = RQ_DIR / "data" / "step01_marginal_means.csv"
        if not marginal_means_path.exists():
            raise FileNotFoundError(f"Step 1 not complete - missing {marginal_means_path}")

        marginal_means = pd.read_csv(marginal_means_path)
        log(f"marginal_means: {len(marginal_means)} rows")
        # Load Model and Extract Coefficients
        log("Loading fitted LMM model...")

        model_path = RQ3_DIR / "data" / "step05_lmm_fitted_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))
        log(f"Model with {len(lmm_model.params)} parameters")

        # Get fixed effect coefficients and covariance matrix
        params = lmm_model.params
        cov_params = lmm_model.cov_params()
        coef_names = list(params.index)

        log(f"Coefficient names: {coef_names}")
        # Identify Slope Coefficients - EXACT MATCHING
        # The model has:
        # - log_Days: slope for Free_Recall (reference level)
        # - log_Days:C(Factor)[T.Cued_Recall]: DIFFERENCE in slope for Cued vs Free
        # - log_Days:C(Factor)[T.Recognition]: DIFFERENCE in slope for Recognition vs Free

        log("Identifying slope coefficient names (exact matching)...")

        # Known coefficient names from the fitted model (EXACT)
        log_days_idx = 'log_Days'
        cued_interaction_idx = "log_Days:C(Factor, Treatment('Free_Recall'))[T.Cued_Recall]"
        recog_interaction_idx = "log_Days:C(Factor, Treatment('Free_Recall'))[T.Recognition]"

        # Verify coefficients exist
        for coef in [log_days_idx, cued_interaction_idx, recog_interaction_idx]:
            if coef not in coef_names:
                raise ValueError(f"Expected coefficient not found: {coef}")

        log(f"Base slope (Free_Recall): {log_days_idx}")
        log(f"Cued_Recall slope diff: {cued_interaction_idx}")
        log(f"Recognition slope diff: {recog_interaction_idx}")
        # Compute Paradigm-Specific Slopes
        # Slope_Free = b_log_days
        # Slope_Cued = b_log_days + b_cued_interaction
        # Slope_Recog = b_log_days + b_recog_interaction

        log("Computing paradigm-specific forgetting slopes...")

        b_log_days = params[log_days_idx]
        b_cued_int = params[cued_interaction_idx]
        b_recog_int = params[recog_interaction_idx]

        slope_free = b_log_days
        slope_cued = b_log_days + b_cued_int
        slope_recog = b_log_days + b_recog_int

        log(f"Free_Recall slope = {slope_free:.4f}")
        log(f"Cued_Recall slope = {slope_cued:.4f}")
        log(f"Recognition slope = {slope_recog:.4f}")
        # Compute Linear Contrast Estimate
        # Contrast = (+1)*slope_recog + (0)*slope_cued + (-1)*slope_free
        # = slope_recog - slope_free
        # = (b_log_days + b_recog_int) - b_log_days
        # = b_recog_int

        log("Computing linear trend contrast estimate...")

        contrast_estimate = (
            CONTRAST_WEIGHTS['recognition'] * slope_recog +
            CONTRAST_WEIGHTS['cued_recall'] * slope_cued +
            CONTRAST_WEIGHTS['free_recall'] * slope_free
        )

        log(f"Contrast weights: Free={CONTRAST_WEIGHTS['free_recall']}, "
            f"Cued={CONTRAST_WEIGHTS['cued_recall']}, Recog={CONTRAST_WEIGHTS['recognition']}")
        log(f"Linear trend contrast estimate = {contrast_estimate:.4f}")
        # Compute Contrast Standard Error
        # The contrast in terms of coefficients is:
        # Contrast = (+1)*slope_recog + (-1)*slope_free
        #          = (+1)*(b_log_days + b_recog_int) + (-1)*b_log_days
        #          = b_recog_int
        # So the contrast is simply the Recognition interaction term!
        # SE_contrast = SE(b_recog_int)

        log("Computing contrast standard error...")

        # The linear contrast = Recognition interaction coefficient
        # SE = SE(recog_int) from model
        recog_idx_pos = coef_names.index(recog_interaction_idx)
        contrast_variance = cov_params.iloc[recog_idx_pos, recog_idx_pos]
        contrast_se = np.sqrt(contrast_variance)

        log(f"Contrast SE = {contrast_se:.4f}")
        # Compute z-statistic and p-values
        log("Computing z-statistic and p-values...")

        z_value = contrast_estimate / contrast_se
        p_value_uncorrected = 2 * stats.norm.sf(abs(z_value))  # Two-tailed
        p_value_bonferroni = min(1.0, p_value_uncorrected * BONFERRONI_N_TESTS)

        log(f"z-value = {z_value:.4f}")
        log(f"p-value (uncorrected) = {p_value_uncorrected:.6f}")
        log(f"p-value (Bonferroni) = {p_value_bonferroni:.6f}")

        # Determine significance
        significant_uncorrected = p_value_uncorrected < ALPHA_UNCORRECTED
        significant_bonferroni = p_value_bonferroni < ALPHA_UNCORRECTED  # Compare corrected p to 0.05

        log(f"Significant (uncorrected, p < 0.05): {significant_uncorrected}")
        log(f"Significant (Bonferroni, p < 0.05): {significant_bonferroni}")
        # Compute 95% Confidence Interval
        log("Computing 95% confidence interval...")

        ci_lower = contrast_estimate - Z_CRITICAL * contrast_se
        ci_upper = contrast_estimate + Z_CRITICAL * contrast_se

        log(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        # Create Output DataFrame
        log("Building contrast results DataFrame...")

        contrast_results = pd.DataFrame({
            'contrast_name': ['Linear_Trend'],
            'estimate': [contrast_estimate],
            'SE': [contrast_se],
            'z_value': [z_value],
            'p_value_uncorrected': [p_value_uncorrected],
            'p_value_bonferroni': [p_value_bonferroni],
            'significant_uncorrected': [significant_uncorrected],
            'significant_bonferroni': [significant_bonferroni],
            'CI_lower': [ci_lower],
            'CI_upper': [ci_upper]
        })

        log(f"Contrast results:\n{contrast_results.to_string()}")
        # Generate Interpretation
        log("Generating contrast interpretation...")

        # Determine direction and effect
        # Note: All slopes are NEGATIVE (forgetting = theta decreases over time)
        # More negative slope = faster forgetting
        # Less negative slope = slower forgetting
        if contrast_estimate < 0:
            direction = "negative"
            effect_description = "forgetting rate INCREASES from Free Recall to Recognition"
            practical_meaning = "More retrieval support is associated with FASTER forgetting"
        else:
            direction = "positive"
            effect_description = "forgetting rate DECREASES from Free Recall to Recognition"
            practical_meaning = "More retrieval support is associated with SLOWER forgetting"

        # Format p-value for display
        if p_value_uncorrected < 0.001:
            p_display = "< .001"
        else:
            p_display = f"= {p_value_uncorrected:.3f}"

        interpretation_text = f"""RQ 5.3.2 - Linear Trend Contrast Analysis
=============================================

RESEARCH QUESTION:
Is there a systematic linear trend in forgetting rates across paradigms
(Free Recall -> Cued Recall -> Recognition)?

CONTRAST SPECIFICATION:
- Weights: Free_Recall = -1, Cued_Recall = 0, Recognition = +1
- Tests: Linear trend in SLOPES (forgetting rates), not intercepts

RESULTS:
- Contrast estimate: {contrast_estimate:.4f}
- Standard error: {contrast_se:.4f}
- z-value: {z_value:.3f}
- p-value (uncorrected): {p_display}
- p-value (Bonferroni-corrected, n=15): {p_value_bonferroni:.4f}
- 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]

SIGNIFICANCE:
- Uncorrected (alpha = 0.05): {"SIGNIFICANT" if significant_uncorrected else "NOT SIGNIFICANT"}
- Bonferroni-corrected: {"SIGNIFICANT" if significant_bonferroni else "NOT SIGNIFICANT"}

INTERPRETATION:
The linear trend contrast estimate is {direction}.
This means {effect_description}.
In practical terms: {practical_meaning}.

PARADIGM-SPECIFIC SLOPES (for reference):
- Free Recall: {slope_free:.4f} (slowest forgetting - least negative)
- Cued Recall: {slope_cued:.4f}
- Recognition: {slope_recog:.4f} (fastest forgetting - most negative)

Note: All slopes are negative because theta decreases over time (forgetting).
A more negative slope = faster forgetting.
A less negative slope = slower forgetting.

STATISTICAL NOTE (D068):
Per thesis decision D068, we report both uncorrected and Bonferroni-corrected
p-values. The Bonferroni correction accounts for 15 planned comparisons across
the thesis to control the family-wise error rate.
"""
        # Validate Output
        log("Checking output validity...")

        # Check for NaN
        if contrast_results.isnull().any().any():
            raise ValueError("Output contains NaN values")

        # Check SE > 0
        if contrast_results['SE'].iloc[0] <= 0:
            raise ValueError("SE must be positive")

        # Check p-value range
        if not 0 <= p_value_uncorrected <= 1:
            raise ValueError(f"p_value_uncorrected out of range: {p_value_uncorrected}")
        if not 0 <= p_value_bonferroni <= 1:
            raise ValueError(f"p_value_bonferroni out of range: {p_value_bonferroni}")

        # Check Bonferroni >= uncorrected
        if p_value_bonferroni < p_value_uncorrected:
            raise ValueError("Bonferroni p-value cannot be less than uncorrected")

        # Check CI ordering
        if not (ci_lower < contrast_estimate < ci_upper):
            raise ValueError("CI ordering violated")

        log("All validation checks passed")
        # Save Outputs
        # Save contrast results CSV
        output_csv_path = RQ_DIR / "data" / "step02_linear_trend_contrast.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        contrast_results.to_csv(output_csv_path, index=False, encoding='utf-8')
        log(f"{output_csv_path}")

        # Save interpretation text
        output_txt_path = RQ_DIR / "data" / "step02_contrast_interpretation.txt"
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(interpretation_text)
        log(f"{output_txt_path}")

        log("Step 02 complete - Linear trend contrast computed")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
