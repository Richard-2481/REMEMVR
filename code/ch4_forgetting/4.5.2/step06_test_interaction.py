#!/usr/bin/env python3
"""
==============================================================================
Step 06: Test LocationType x Phase Interaction
==============================================================================
RQ: 5.5.2 - Source-Destination Consolidation (Two-Phase)

Purpose:
    Extract the 3-way interaction term (Days_within:Segment:LocationType) from
    the fitted LMM and test whether consolidation benefit differs between
    Source and Destination memory types.

Decision D068 Compliance:
    - Report BOTH p_uncorrected and p_bonferroni (2 main tests)
    - Significance threshold: p_bonferroni < 0.025 (family-wise alpha = 0.05)

Effect Size:
    - Cohen's f^2 for the 3-way interaction
    - Computed by comparing full model vs reduced model (without 3-way interaction)
    - f^2 = (R2_full - R2_reduced) / (1 - R2_full)

Input:
    - data/step03_lmm_coefficients.csv (8 rows of fixed effects)
    - data/step02_lmm_input_long.csv (800 rows of LMM input data)

Output:
    - data/step06_interaction_tests.csv (1 row: 3-way interaction test)
==============================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# PATH SETUP

RQ_DIR = Path(__file__).resolve().parent.parent  # results/ch5/5.5.2

# Folder conventions 
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"

# Input
INPUT_COEFFS = DATA_DIR / "step03_lmm_coefficients.csv"
INPUT_DATA = DATA_DIR / "step02_lmm_input_long.csv"

# Output
OUTPUT_FILE = DATA_DIR / "step06_interaction_tests.csv"
LOG_FILE = LOGS_DIR / "step06_test_interaction.log"

# Create directories if needed
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING

def log(msg: str) -> None:
    """Log message to console and file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file
LOG_FILE.write_text("", encoding="utf-8")

# MAIN EXECUTION

if __name__ == "__main__":
    try:
        log("Step 06: Test LocationType x Phase Interaction")
        # Load Coefficients from Step 3

        log("Loading coefficients from Step 3...")
        df_coeffs = pd.read_csv(INPUT_COEFFS)
        log(f"{INPUT_COEFFS.name} ({len(df_coeffs)} rows)")

        # Find the 3-way interaction term
        interaction_term = "Days_within:Segment[T.Late]:LocationType[T.Destination]"

        row = df_coeffs[df_coeffs['term'] == interaction_term]
        if len(row) == 0:
            log(f"3-way interaction term not found: {interaction_term}")
            log(f"Available terms: {df_coeffs['term'].tolist()}")
            raise ValueError(f"Interaction term '{interaction_term}' not in coefficients")

        row = row.iloc[0]
        estimate = row['estimate']
        se = row['SE']
        z_score = row['z_score']
        p_uncorrected = row['p_value']

        log(f"3-way interaction term: {interaction_term}")
        log(f"  Estimate: {estimate:.4f}")
        log(f"  SE: {se:.4f}")
        log(f"  z-score: {z_score:.2f}")
        log(f"  p_uncorrected: {p_uncorrected:.4f}")
        # Apply Bonferroni Correction (Decision D068)

        log("Applying Bonferroni correction (2 main tests)...")
        n_tests = 2  # Source consolidation + Destination consolidation
        p_bonferroni = min(p_uncorrected * n_tests, 1.0)
        significant_bonferroni = p_bonferroni < 0.025  # alpha/2 = 0.025

        log(f"  p_bonferroni: {p_bonferroni:.4f}")
        log(f"  Significant at alpha=0.025: {significant_bonferroni}")
        # Compute Cohen's f² Effect Size
        # Need to fit full and reduced models to compare R²

        log("[EFFECT SIZE] Computing Cohen's f² via model comparison...")
        log("Loading LMM input data...")
        df = pd.read_csv(INPUT_DATA)
        log(f"{INPUT_DATA.name} ({len(df)} rows)")

        # Set up treatment coding
        df['LocationType'] = pd.Categorical(df['LocationType'],
                                            categories=['Source', 'Destination'])
        df['Segment'] = pd.Categorical(df['Segment'],
                                       categories=['Early', 'Late'])

        # Fit full model
        log("Fitting full model (with 3-way interaction)...")
        formula_full = "theta ~ Days_within * Segment * LocationType"
        try:
            model_full = smf.mixedlm(formula_full, data=df, groups=df['UID'],
                                     re_formula="~Days_within")
            result_full = model_full.fit(reml=False, method='lbfgs', maxiter=200)
            random_structure = "~Days_within"
        except:
            log("Random slopes failed, using intercept-only")
            model_full = smf.mixedlm(formula_full, data=df, groups=df['UID'],
                                     re_formula="~1")
            result_full = model_full.fit(reml=False, method='lbfgs', maxiter=200)
            random_structure = "~1"

        log(f"Full model converged: {result_full.converged}")

        # Fit reduced model (without 3-way interaction)
        log("Fitting reduced model (without 3-way interaction)...")
        # Remove 3-way interaction but keep 2-way interactions
        formula_reduced = "theta ~ Days_within + Segment + LocationType + Days_within:Segment + Days_within:LocationType + Segment:LocationType"

        try:
            if random_structure == "~Days_within":
                model_reduced = smf.mixedlm(formula_reduced, data=df, groups=df['UID'],
                                            re_formula="~Days_within")
                result_reduced = model_reduced.fit(reml=False, method='lbfgs', maxiter=200)
            else:
                model_reduced = smf.mixedlm(formula_reduced, data=df, groups=df['UID'],
                                            re_formula="~1")
                result_reduced = model_reduced.fit(reml=False, method='lbfgs', maxiter=200)
        except:
            log("Reduced model with random slopes failed, using intercept-only")
            model_reduced = smf.mixedlm(formula_reduced, data=df, groups=df['UID'],
                                        re_formula="~1")
            result_reduced = model_reduced.fit(reml=False, method='lbfgs', maxiter=200)

        log(f"Reduced model converged: {result_reduced.converged}")

        # Compute pseudo-R² for LMM (using variance reduction)
        # R² = 1 - (residual variance / total variance)
        y = df['theta']
        var_total = y.var()

        # Full model residual variance
        y_pred_full = result_full.fittedvalues
        residuals_full = y - y_pred_full
        var_resid_full = residuals_full.var()
        r2_full = 1 - (var_resid_full / var_total)

        # Reduced model residual variance
        y_pred_reduced = result_reduced.fittedvalues
        residuals_reduced = y - y_pred_reduced
        var_resid_reduced = residuals_reduced.var()
        r2_reduced = 1 - (var_resid_reduced / var_total)

        log(f"[R²] Full model R²: {r2_full:.4f}")
        log(f"[R²] Reduced model R²: {r2_reduced:.4f}")
        log(f"[R²] R² difference: {r2_full - r2_reduced:.4f}")

        # Cohen's f²
        if r2_full >= 1.0:
            cohens_f2 = 0.0  # Perfect fit edge case
        else:
            cohens_f2 = (r2_full - r2_reduced) / (1 - r2_full)

        # Handle negative f² (can happen with negligible effects)
        if cohens_f2 < 0:
            cohens_f2 = 0.0

        log(f"[EFFECT SIZE] Cohen's f²: {cohens_f2:.4f}")

        # Interpret effect size
        if cohens_f2 < 0.02:
            effect_interpretation = "negligible"
        elif cohens_f2 < 0.15:
            effect_interpretation = "small"
        elif cohens_f2 < 0.35:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        log(f"Effect size: {effect_interpretation}")
        # Save Results

        log(f"Saving interaction test results to {OUTPUT_FILE.name}...")

        result_df = pd.DataFrame({
            'Term': [interaction_term],
            'Estimate': [estimate],
            'SE': [se],
            'z_score': [z_score],
            'p_uncorrected': [p_uncorrected],
            'p_bonferroni': [p_bonferroni],
            'Significant_bonferroni': [significant_bonferroni],
            'Cohens_f2': [cohens_f2],
            'Effect_interpretation': [effect_interpretation]
        })

        result_df.to_csv(OUTPUT_FILE, index=False)
        log(f"{OUTPUT_FILE.name} ({len(result_df)} rows, {len(result_df.columns)} cols)")
        # Validation

        log("Checking output structure...")

        # Check row count
        if len(result_df) != 1:
            raise ValueError(f"Expected 1 row, got {len(result_df)}")
        log("Row count: 1")

        # Check column count
        expected_cols = ['Term', 'Estimate', 'SE', 'z_score', 'p_uncorrected',
                        'p_bonferroni', 'Significant_bonferroni', 'Cohens_f2',
                        'Effect_interpretation']
        if list(result_df.columns) != expected_cols:
            raise ValueError(f"Column mismatch: {list(result_df.columns)}")
        log("Column count: 9")

        # Check p-value relationship (Bonferroni >= uncorrected)
        if p_bonferroni < p_uncorrected:
            raise ValueError("Bonferroni correction decreased p-value - logic error")
        log("Bonferroni p >= uncorrected p")

        # Check Decision D068 compliance
        log("Decision D068: Both p_uncorrected and p_bonferroni columns present")

        # Check effect size non-negative
        if cohens_f2 < 0:
            raise ValueError(f"Negative Cohen's f²: {cohens_f2}")
        log(f"Cohen's f² non-negative: {cohens_f2:.4f}")
        # SUCCESS

        log("Step 06 complete")
        log(f"[KEY RESULT] 3-way interaction: beta={estimate:.4f}, p_uncorrected={p_uncorrected:.4f}, p_bonferroni={p_bonferroni:.4f}")
        log(f"[KEY RESULT] Significant at Bonferroni alpha=0.025: {significant_bonferroni}")
        log(f"[KEY RESULT] Effect size: Cohen's f²={cohens_f2:.4f} ({effect_interpretation})")

        if significant_bonferroni:
            log("LocationType x Phase interaction IS significant")
            log("Source and Destination show DIFFERENT consolidation patterns")
        else:
            log("LocationType x Phase interaction NOT significant")
            log("Source and Destination show SIMILAR consolidation patterns")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
