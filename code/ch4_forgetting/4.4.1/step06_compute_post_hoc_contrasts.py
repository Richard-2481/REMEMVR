#!/usr/bin/env python3
"""Post-Hoc Contrasts and Effect Sizes (Decision D068): Extract pairwise slope contrasts with dual p-value reporting (D068) and compute"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_compute_post_hoc_contrasts.log"

# Input files
INPUT_MODEL = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"

# Output files
OUTPUT_CONTRASTS = RQ_DIR / "results" / "step06_post_hoc_contrasts.csv"
OUTPUT_EFFECTS = RQ_DIR / "results" / "step06_effect_sizes.csv"

# Comparisons to make
COMPARISONS = ["congruent-common", "incongruent-common", "congruent-incongruent"]
FAMILY_ALPHA = 0.05

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Contrast Computation Function

def compute_pairwise_contrasts(lmm_result, comparisons, family_alpha=0.05):
    """
    Compute post-hoc pairwise contrasts with dual reporting (D068).

    For congruence analysis with Log model, we're comparing INTERCEPT differences
    (immediate memory) and SLOPE differences (forgetting rate) between conditions.
    """
    k = len(comparisons)
    alpha_corrected = family_alpha / k

    log(f"  Family-wise alpha: {family_alpha}")
    log(f"  Number of comparisons: {k}")
    log(f"  Bonferroni-corrected alpha: {alpha_corrected:.4f}")

    # Get parameter names
    params = lmm_result.params
    param_names = params.index.tolist()

    log(f"\n  Available parameters:")
    for p in param_names:
        log(f"    {p}: {params[p]:.4f}")

    results = []

    for comparison in comparisons:
        parts = comparison.split("-")
        level1, level2 = parts[0].strip(), parts[1].strip()

        log(f"\n  Computing contrast: {level1} vs {level2}")

        # For slope comparisons, we look at interaction terms with TSVR_log
        # Reference is common, so:
        # - congruent-common: TSVR_log:C(...)[T.congruent]
        # - incongruent-common: TSVR_log:C(...)[T.incongruent]
        # - congruent-incongruent: difference of the two above

        # Find relevant coefficient names
        def find_slope_coef(level):
            """Find the slope interaction coefficient for a level."""
            for name in param_names:
                if "TSVR_log" in name and f"T.{level}" in name and ":" in name:
                    return name
            return None

        if level2 == "common":
            # Direct contrast with reference
            coef_name = find_slope_coef(level1)
            if coef_name:
                beta = params[coef_name]
                se = lmm_result.bse[coef_name]
                p_uncorrected = lmm_result.pvalues[coef_name]
            else:
                log(f"    Could not find slope coefficient for {level1}")
                beta, se, p_uncorrected = np.nan, np.nan, np.nan

        elif level1 == "common":
            # Reverse contrast (common vs non-reference)
            coef_name = find_slope_coef(level2)
            if coef_name:
                beta = -params[coef_name]  # Negative of coefficient
                se = lmm_result.bse[coef_name]
                p_uncorrected = lmm_result.pvalues[coef_name]
            else:
                log(f"    Could not find slope coefficient for {level2}")
                beta, se, p_uncorrected = np.nan, np.nan, np.nan

        else:
            # Neither is reference - need to compute difference
            coef_name1 = find_slope_coef(level1)
            coef_name2 = find_slope_coef(level2)

            if coef_name1 and coef_name2:
                beta1 = params[coef_name1]
                beta2 = params[coef_name2]
                beta = beta1 - beta2

                # Delta method SE
                try:
                    cov_matrix = lmm_result.cov_params()
                    var1 = cov_matrix.loc[coef_name1, coef_name1]
                    var2 = cov_matrix.loc[coef_name2, coef_name2]
                    cov12 = cov_matrix.loc[coef_name1, coef_name2]
                    se = np.sqrt(var1 + var2 - 2 * cov12)
                except Exception:
                    se1 = lmm_result.bse[coef_name1]
                    se2 = lmm_result.bse[coef_name2]
                    se = np.sqrt(se1**2 + se2**2)
                    log(f"    Using approximate SE (covariance extraction failed)")

                # Two-tailed p-value
                z = beta / se
                p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                log(f"    Could not find slope coefficients for contrast")
                beta, se, p_uncorrected = np.nan, np.nan, np.nan

        # Calculate z-statistic and corrected p-value
        if np.isfinite(beta) and np.isfinite(se) and se > 0:
            z = beta / se
            p_corrected = min(p_uncorrected * k, 1.0)
        else:
            z = np.nan
            p_corrected = np.nan

        # Significance flags
        sig_uncorrected = p_uncorrected < 0.05 if np.isfinite(p_uncorrected) else False
        sig_corrected = p_uncorrected < alpha_corrected if np.isfinite(p_uncorrected) else False

        results.append({
            "comparison": comparison,
            "beta": beta,
            "se": se,
            "z": z,
            "p_uncorrected": p_uncorrected,
            "alpha_corrected": alpha_corrected,
            "p_corrected": p_corrected,
            "sig_uncorrected": sig_uncorrected,
            "sig_corrected": sig_corrected
        })

        log(f"    beta={beta:.4f}, se={se:.4f}, z={z:.4f}, p_unc={p_uncorrected:.4f}")

    return pd.DataFrame(results)

# Effect Size Computation Function

def compute_effect_sizes(lmm_result, include_interactions=True):
    """Compute Cohen's f-squared effect sizes for fixed effects."""
    results = []

    for param_name in lmm_result.params.index:
        # Skip intercept and variance components
        if param_name == "Intercept":
            continue
        if "Group" in param_name or "Var" in param_name or "Cov" in param_name:
            continue

        # Skip interactions if not requested
        if not include_interactions and ":" in param_name:
            continue

        beta = lmm_result.params[param_name]
        se = lmm_result.bse[param_name]

        # Simplified f-squared approximation
        n = lmm_result.nobs
        f_squared = (beta / se) ** 2 / n

        # Interpret using Cohen 1988 thresholds
        if f_squared < 0.02:
            interpretation = "negligible"
        elif f_squared < 0.15:
            interpretation = "small"
        elif f_squared < 0.35:
            interpretation = "medium"
        else:
            interpretation = "large"

        results.append({
            "effect": param_name,
            "f_squared": f_squared,
            "interpretation": interpretation
        })

    return pd.DataFrame(results)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Post-Hoc Contrasts and Effect Sizes")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Fitted Model
        log("\nLoading fitted LMM model...")

        best_model = MixedLMResults.load(str(INPUT_MODEL))
        log(f"{INPUT_MODEL.name}")
        log(f"  AIC: {best_model.aic:.2f}")
        log(f"  N observations: {best_model.nobs}")
        # Compute Pairwise Contrasts
        log("\nComputing pairwise slope contrasts...")

        df_contrasts = compute_pairwise_contrasts(
            lmm_result=best_model,
            comparisons=COMPARISONS,
            family_alpha=FAMILY_ALPHA
        )

        log(f"\nComputed {len(df_contrasts)} contrasts")
        # Compute Effect Sizes
        log("\nComputing effect sizes...")

        df_effects = compute_effect_sizes(
            lmm_result=best_model,
            include_interactions=True
        )

        log(f"Computed effect sizes for {len(df_effects)} effects")

        # Summary
        log("\n  Effect Size Summary:")
        for interp in ["large", "medium", "small", "negligible"]:
            n = len(df_effects[df_effects["interpretation"] == interp])
            if n > 0:
                log(f"    {interp}: {n}")
        # Save Outputs
        log("\nSaving output files...")

        # Ensure results directory exists
        (RQ_DIR / "results").mkdir(parents=True, exist_ok=True)

        # Save contrasts
        df_contrasts.to_csv(OUTPUT_CONTRASTS, index=False, encoding='utf-8')
        log(f"{OUTPUT_CONTRASTS.name} ({len(df_contrasts)} rows)")

        # Save effect sizes
        df_effects.to_csv(OUTPUT_EFFECTS, index=False, encoding='utf-8')
        log(f"{OUTPUT_EFFECTS.name} ({len(df_effects)} rows)")
        # Validation
        log("\nValidating results...")

        # Check all contrasts computed
        expected_contrasts = 3
        if len(df_contrasts) != expected_contrasts:
            log(f"Expected {expected_contrasts} contrasts, got {len(df_contrasts)}")
        else:
            log(f"All {expected_contrasts} contrasts computed")

        # Check dual p-values present
        has_dual = ("p_uncorrected" in df_contrasts.columns and
                   "p_corrected" in df_contrasts.columns)
        if has_dual:
            log("Dual p-values present")
        else:
            log("Missing p-value columns")

        # Check Bonferroni alpha
        expected_alpha = FAMILY_ALPHA / len(COMPARISONS)
        actual_alpha = df_contrasts["alpha_corrected"].iloc[0]
        if abs(actual_alpha - expected_alpha) < 0.001:
            log(f"Bonferroni alpha correct: {actual_alpha:.4f}")
        else:
            log(f"Bonferroni alpha {actual_alpha:.4f} != expected {expected_alpha:.4f}")

        # Check p-values valid
        p_values = df_contrasts["p_uncorrected"].dropna()
        if all((p_values >= 0) & (p_values <= 1)):
            log("All p-values in [0, 1]")
        else:
            log("Invalid p-values detected")

        # Check effect sizes computed
        if len(df_effects) > 0:
            log(f"Effect sizes computed: {len(df_effects)} effects")
        else:
            log("No effect sizes computed")

        # Summary of contrast results
        log("\n  Contrast Results Summary:")
        n_sig_unc = df_contrasts["sig_uncorrected"].sum()
        n_sig_cor = df_contrasts["sig_corrected"].sum()
        log(f"    Significant (uncorrected alpha=0.05): {n_sig_unc}/{len(df_contrasts)}")
        log(f"    Significant (corrected alpha={expected_alpha:.4f}): {n_sig_cor}/{len(df_contrasts)}")

        log("\nStep 06 complete (Post-Hoc Contrasts)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
