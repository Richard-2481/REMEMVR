#!/usr/bin/env python3
"""Compute Post-hoc Pairwise Contrasts: Compute pairwise paradigm contrasts with dual p-values (Decision D068) and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from tools.analysis_lmm import compute_effect_sizes_cohens

from tools.validation import validate_lmm_convergence

# Import for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_compute_post_hoc_contrasts.log"


# Logging Function

def log(msg):
    # Ensure logs directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)


# Custom Contrast Computation (for RQ 5.3 paradigm factors)

def compute_paradigm_contrasts(
    lmm_result,
    comparisons: List[str],
    family_alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute post-hoc pairwise contrasts for paradigm factor with dual reporting (Decision D068).

    This is a RQ 5.3-specific implementation because the generic compute_contrasts_pairwise
    function uses hardcoded factor level names (What, Where, When) from RQ 5.1 domain analysis.

    Implements dual reporting of p-values:
    - Uncorrected (alpha = 0.05)
    - Bonferroni-corrected (alpha_corrected = family_alpha / k)

    where k = number of comparisons in this RQ.

    Args:
        lmm_result: Fitted MixedLM result object
        comparisons: List of comparison strings, e.g., ["Cued_Recall-Free_Recall", "Recognition-Free_Recall"]
        family_alpha: Family-wise alpha level (default: 0.05)

    Returns:
        DataFrame with columns:
        - comparison: Comparison label
        - beta: Estimated effect size
        - se: Standard error
        - z: z-statistic
        - p_uncorrected: Uncorrected p-value
        - alpha_corrected: Bonferroni-corrected alpha threshold
        - p_corrected: Corrected p-value (p * k)
        - sig_uncorrected: Significant at alpha=0.05 (bool)
        - sig_corrected: Significant at alpha_corrected (bool)
    """
    log("\n" + "=" * 60)
    log("POST-HOC PAIRWISE CONTRASTS (Decision D068)")
    log("=" * 60)

    k = len(comparisons)
    alpha_corrected = family_alpha / k

    log(f"Family-wise alpha: {family_alpha}")
    log(f"Number of comparisons: {k}")
    log(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")

    # Get coefficient names from model
    param_names = list(lmm_result.params.index)
    log(f"\nAvailable coefficient names:")
    for name in param_names:
        log(f"  {name}")

    # Detect factor name and reference level from coefficient names
    # Looking for patterns like: C(Factor, Treatment('Free_Recall'))[T.Cued_Recall]
    factor_name = None
    reference_level = None

    for name in param_names:
        if "Treatment('" in name and "[T." in name:
            # Extract factor name and reference level
            # Example: C(Factor, Treatment('Free_Recall'))[T.Cued_Recall]
            import re
            match = re.search(r"C\((\w+),\s*Treatment\('([^']+)'\)\)\[T\.(\w+)\]", name)
            if match:
                factor_name = match.group(1)
                reference_level = match.group(2)
                break

    if factor_name is None:
        raise ValueError("Could not detect factor name from model coefficients")

    log(f"\nDetected factor name: {factor_name}")
    log(f"Detected reference level: {reference_level}")

    # Helper function to find coefficient name for a given level
    def find_coef_name(level: str) -> str:
        """Find the coefficient name for a factor level in the model."""
        # Full treatment coding: C(Factor, Treatment('Free_Recall'))[T.Cued_Recall]
        pattern = f"C({factor_name}, Treatment('{reference_level}'))[T.{level}]"
        if pattern in param_names:
            return pattern
        return None

    results = []

    for comparison in comparisons:
        # Parse comparison string (e.g., "Cued_Recall-Free_Recall" -> Cued_Recall - Free_Recall)
        parts = comparison.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid comparison format: {comparison}. Expected 'A-B'")

        level1, level2 = parts[0].strip(), parts[1].strip()

        log(f"\n  Processing: {level1} vs {level2}")

        # Case 1: level2 is the reference level (e.g., Cued_Recall-Free_Recall)
        # -> Use level1 coefficient directly
        if level2 == reference_level:
            coef_name = find_coef_name(level1)
            if coef_name is None:
                log(f"    Warning: Coefficient for '{level1}' not found. Skipping.")
                continue

            beta = lmm_result.params[coef_name]
            se = lmm_result.bse[coef_name]
            z = beta / se
            p_uncorrected = lmm_result.pvalues[coef_name]
            log(f"    Found direct coefficient: {coef_name}")

        # Case 2: level1 is the reference level (e.g., Free_Recall-Cued_Recall)
        # -> Use negative of level2 coefficient
        elif level1 == reference_level:
            coef_name = find_coef_name(level2)
            if coef_name is None:
                log(f"    Warning: Coefficient for '{level2}' not found. Skipping.")
                continue

            beta = -lmm_result.params[coef_name]
            se = lmm_result.bse[coef_name]  # SE is symmetric
            z = beta / se
            p_uncorrected = lmm_result.pvalues[coef_name]  # Same p-value (two-tailed)
            log(f"    Using negative of: {coef_name}")

        # Case 3: Neither is the reference level (e.g., Recognition-Cued_Recall)
        # -> Compute difference: beta1 - beta2 with delta method SE
        else:
            coef_name1 = find_coef_name(level1)
            coef_name2 = find_coef_name(level2)

            if coef_name1 is None:
                log(f"    Warning: Coefficient for '{level1}' not found. Skipping.")
                continue
            if coef_name2 is None:
                log(f"    Warning: Coefficient for '{level2}' not found. Skipping.")
                continue

            beta1 = lmm_result.params[coef_name1]
            beta2 = lmm_result.params[coef_name2]
            beta = beta1 - beta2

            # Delta method SE: sqrt(Var(b1) + Var(b2) - 2*Cov(b1,b2))
            try:
                cov_matrix = lmm_result.cov_params()
                var1 = cov_matrix.loc[coef_name1, coef_name1]
                var2 = cov_matrix.loc[coef_name2, coef_name2]
                cov12 = cov_matrix.loc[coef_name1, coef_name2]
                se = np.sqrt(var1 + var2 - 2 * cov12)
                log(f"    Using delta method: {coef_name1} - {coef_name2}")
            except Exception as e:
                # Fallback: approximate SE as sqrt(se1^2 + se2^2) assuming independence
                se1 = lmm_result.bse[coef_name1]
                se2 = lmm_result.bse[coef_name2]
                se = np.sqrt(se1**2 + se2**2)
                log(f"    Using approximate SE (cov extraction failed): {str(e)[:50]}")

            z = beta / se
            # Two-tailed p-value from z
            p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))

        p_corrected = min(p_uncorrected * k, 1.0)  # Cap at 1.0

        # Significance flags
        sig_uncorrected = p_uncorrected < 0.05
        sig_corrected = p_uncorrected < alpha_corrected

        results.append({
            'comparison': comparison,
            'beta': beta,
            'se': se,
            'z': z,
            'p_uncorrected': p_uncorrected,
            'alpha_corrected': alpha_corrected,
            'p_corrected': p_corrected,
            'sig_uncorrected': sig_uncorrected,
            'sig_corrected': sig_corrected
        })

    df_contrasts = pd.DataFrame(results)

    # Summary
    if len(df_contrasts) > 0:
        log(f"\nResults:")
        log(f"  Significant (uncorrected alpha=0.05): {df_contrasts['sig_uncorrected'].sum()}/{k}")
        log(f"  Significant (corrected alpha={alpha_corrected:.4f}): {df_contrasts['sig_corrected'].sum()}/{k}")
    else:
        log(f"\nWARNING: No contrasts computed!")

    log("=" * 60 + "\n")

    return df_contrasts


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Compute Post-hoc Pairwise Contrasts")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Best LMM Model from Step 5

        log("\nLoading best LMM model from Step 5...")

        # Load fitted model using MixedLMResults.load() (NOT pickle.load())
        # CRITICAL: pickle.load() causes patsy/eval errors with statsmodels
        model_path = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        best_result = MixedLMResults.load(str(model_path))
        log(f"Best LMM model from {model_path}")
        log(f"  AIC: {best_result.aic:.2f}")
        log(f"  N observations: {best_result.nobs}")
        log(f"  N groups: {len(best_result.model.group_labels)}")

        # Also load fixed effects CSV for reference
        fixed_effects_path = RQ_DIR / "data" / "step05_fixed_effects.csv"
        df_fixed_effects = pd.read_csv(fixed_effects_path, encoding='utf-8')
        log(f"Fixed effects reference from {fixed_effects_path}")
        log(f"  Fixed effects: {len(df_fixed_effects)} terms")
        # Run Pairwise Contrast Analysis
        #               dual p-value reporting (uncorrected + Bonferroni-corrected)

        log("\nComputing pairwise contrasts (Decision D068)...")

        # Define pairwise comparisons based on paradigm levels
        # Reference level is Free_Recall (Treatment coding in LMM)
        # NOTE: Must use exact factor level names from model (capitalized with underscores)
        comparisons = [
            "Cued_Recall-Free_Recall",
            "Recognition-Free_Recall",
            "Recognition-Cued_Recall"
        ]

        # Family-wise alpha for Bonferroni correction
        family_alpha = 0.05

        log(f"  Comparisons: {comparisons}")
        log(f"  Family-wise alpha: {family_alpha}")
        log(f"  Bonferroni-corrected alpha: {family_alpha / len(comparisons):.4f}")

        contrasts = compute_paradigm_contrasts(
            lmm_result=best_result,
            comparisons=comparisons,
            family_alpha=family_alpha
        )

        if len(contrasts) > 0:
            log("Pairwise contrasts computed")
            log(f"  Significant (uncorrected): {contrasts['sig_uncorrected'].sum()}/{len(contrasts)}")
            log(f"  Significant (Bonferroni): {contrasts['sig_corrected'].sum()}/{len(contrasts)}")
        else:
            raise ValueError("No contrasts could be computed - check factor level names")
        # Compute Effect Sizes (Cohen's f-squared)

        log("\nComputing effect sizes (Cohen's f-squared)...")

        effect_sizes = compute_effect_sizes_cohens(
            lmm_result=best_result,
            include_interactions=True  # Include interaction terms for paradigm comparisons
        )

        log("Effect sizes computed")
        log(f"  Total effects: {len(effect_sizes)}")
        # Save Analysis Outputs
        # These outputs will be used by: results pipeline for final report generation

        # Ensure results directory exists
        results_dir = RQ_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save contrasts to results/ (final report)
        contrasts_path = results_dir / "step06_post_hoc_contrasts.csv"
        contrasts.to_csv(contrasts_path, index=False, encoding='utf-8')
        log(f"\n{contrasts_path}")
        log(f"  Rows: {len(contrasts)}, Columns: {len(contrasts.columns)}")

        # Save effect sizes to results/ (final report)
        effect_sizes_path = results_dir / "step06_effect_sizes.csv"
        effect_sizes.to_csv(effect_sizes_path, index=False, encoding='utf-8')
        log(f"{effect_sizes_path}")
        log(f"  Rows: {len(effect_sizes)}, Columns: {len(effect_sizes.columns)}")
        # Run Validation Tool
        # Validates: Model convergence status
        # Threshold: converged = True

        log("\nRunning validation checks...")

        validation_result = validate_lmm_convergence(best_result)

        # Report validation results
        log(f"Convergence: {'PASS' if validation_result['converged'] else 'FAIL'}")
        log(f"Message: {validation_result['message']}")

        # Additional validation: Check contrasts output
        validation_issues = []

        # Check all 3 pairwise comparisons computed
        if len(contrasts) < 3:
            validation_issues.append(f"Expected 3+ contrasts, got {len(contrasts)}")

        # Check p_uncorrected in [0, 1]
        if not contrasts['p_uncorrected'].between(0, 1).all():
            validation_issues.append("p_uncorrected values outside [0, 1] range")

        # Check p_corrected >= p_uncorrected (Bonferroni inflates p-values)
        if not (contrasts['p_corrected'] >= contrasts['p_uncorrected']).all():
            validation_issues.append("p_corrected should be >= p_uncorrected")

        # Check Bonferroni alpha applied correctly (0.05 / 3 = 0.0167)
        expected_alpha = family_alpha / len(comparisons)
        if not np.allclose(contrasts['alpha_corrected'].iloc[0], expected_alpha, rtol=0.01):
            validation_issues.append(f"Expected alpha_corrected={expected_alpha:.4f}")

        if validation_issues:
            for issue in validation_issues:
                log(f"[VALIDATION WARNING] {issue}")
        else:
            log("All contrast validation checks passed")
        # Summary
        log("\n" + "=" * 60)
        log("STEP 06 SUMMARY: Post-hoc Pairwise Contrasts")
        log("=" * 60)

        log("\nPairwise Contrasts (Decision D068 - Dual p-value reporting):")
        for _, row in contrasts.iterrows():
            sig_mark_unc = "*" if row['sig_uncorrected'] else ""
            sig_mark_cor = "**" if row['sig_corrected'] else ""
            log(f"  {row['comparison']}: beta={row['beta']:.4f}, z={row['z']:.2f}, "
                f"p_unc={row['p_uncorrected']:.4f}{sig_mark_unc}, "
                f"p_cor={row['p_corrected']:.4f}{sig_mark_cor}")

        log("\nEffect Sizes (Cohen's f-squared):")
        for _, row in effect_sizes.iterrows():
            log(f"  {row['effect']}: f^2={row['f_squared']:.4f} ({row['interpretation']})")

        log("\n" + "=" * 60)
        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
