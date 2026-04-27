#!/usr/bin/env python3
# =============================================================================
# Random Slopes Testing: Intercepts-Only vs Intercepts+Slopes Comparison
# =============================================================================
"""
Step ID: step05d
Step Name: Random Slopes Comparison (MANDATORY - Section 4.4 BLOCKER)
RQ: ch5/5.2.1
Generated: 2025-12-27 (rq_platinum agent)

PURPOSE:
Test whether random slopes improve model fit vs intercepts-only for top 10
competitive models from extended kitchen sink comparison. This is MANDATORY
per improvement_taxonomy.md Section 4.4: "Cannot claim homogeneous effects
without testing for heterogeneity."

CRITICAL REQUIREMENT:
All modeling RQs MUST compare:
- Option A: Random intercepts-only (re_formula='1')
- Option B: Random intercepts + slopes (re_formula='~time_var')

This prevents claiming "individual differences in forgetting rates" without
evidence that slopes improve fit vs simpler intercepts-only model.

EXPECTED INPUTS:
- data/step04_lmm_input.csv (from IRT -> TSVR merge)
  Columns: [composite_ID, UID, test, TSVR_hours, domain, theta]
  Format: CSV, long format (1 row per composite_ID x domain)
  Expected rows: ~1200

EXPECTED OUTPUTS:
- results/step05d_random_slopes_comparison.csv
  Columns: [model_name, aic_intercepts, aic_slopes, delta_aic,
            slopes_improve_fit, slope_variance, converged_int, converged_slopes]
  Format: CSV (comparison for 10 competitive models)
  Expected rows: 10

- results/step05d_slopes_summary.txt
  Format: Text report summarizing findings

VALIDATION CRITERIA:
- At least 8/10 models converge for both structures
- If slopes improve fit (ΔAIC > 2): Report slope variance
- If slopes don't improve (ΔAIC < 2): Recommend intercepts-only
- Document convergence failures (boundary warnings)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Dict, Tuple
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05d_random_slopes_comparison.log"

# Top 10 competitive models from extended kitchen sink (ΔAIC < 2)
TOP_MODELS = {
    'Recip+Log': {
        'formula': 'Ability ~ recip_Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'  # Primary time variable for this model
    },
    'PowerLaw_Log': {
        'formula': 'Ability ~ Days_pow_neg05 + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'
    },
    'CubeRoot+Log': {
        'formula': 'Ability ~ cbrt_Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'
    },
    'Tanh+Log': {
        'formula': 'Ability ~ tanh_Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'
    },
    'SquareRoot+Lin': {
        'formula': 'Ability ~ sqrt_Days + Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~Days'
    },
    'Lin+Log': {
        'formula': 'Ability ~ Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~Days'
    },
    'Exp+Log': {
        'formula': 'Ability ~ neg_Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'
    },
    'Recip+Lin': {
        'formula': 'Ability ~ recip_Days + Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~Days'
    },
    'PowerLaw+Recip+Log': {
        'formula': 'Ability ~ Days_pow_neg05 + recip_Days + log_Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~log_Days'
    },
    'PowerLaw_Lin': {
        'formula': 'Ability ~ Days_pow_neg05 + Days + C(Factor, Treatment("What"))',
        're_intercepts': '1',
        're_slopes': '~Days'
    }
}

# =============================================================================
# Logging Function
# =============================================================================

def log(msg: str) -> None:
    """Write to both log file and console."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all time transformations needed for 10 competitive models.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with TSVR_hours column

    Returns
    -------
    pd.DataFrame
        Data with all time transformations added
    """
    df_prep = df.copy()

    # Convert TSVR_hours to Days
    df_prep['Days'] = df_prep['TSVR_hours'] / 24.0

    # Create all time transformations
    df_prep['Days_sq'] = df_prep['Days'] ** 2
    df_prep['log_Days'] = np.log(df_prep['Days'] + 1)
    df_prep['sqrt_Days'] = np.sqrt(df_prep['Days'])
    df_prep['cbrt_Days'] = np.cbrt(df_prep['Days'])
    df_prep['recip_Days'] = 1.0 / (df_prep['Days'] + 1)
    df_prep['Days_pow_neg05'] = (df_prep['Days'] + 1) ** (-0.5)
    df_prep['neg_Days'] = -df_prep['Days']
    df_prep['tanh_Days'] = np.tanh(df_prep['Days'])

    # Capitalize domain for treatment coding
    df_prep['Factor'] = df_prep['domain'].str.capitalize()

    # Rename theta -> Ability
    df_prep['Ability'] = df_prep['theta']

    return df_prep

# =============================================================================
# Model Fitting Functions
# =============================================================================

def fit_model_pair(
    model_name: str,
    model_spec: Dict,
    data: pd.DataFrame
) -> Tuple[object, object, Dict]:
    """
    Fit both intercepts-only and intercepts+slopes for a given model.

    Parameters
    ----------
    model_name : str
        Name of model
    model_spec : Dict
        Model specification with formula and re_formula
    data : pd.DataFrame
        Prepared LMM data

    Returns
    -------
    Tuple[object, object, Dict]
        (result_intercepts, result_slopes, summary_dict)
    """
    formula = model_spec['formula']
    re_intercepts = model_spec['re_intercepts']
    re_slopes = model_spec['re_slopes']

    log(f"\n{'='*60}")
    log(f"Model: {model_name}")
    log(f"{'='*60}")
    log(f"Formula: {formula}")

    # Fit intercepts-only
    log(f"\n[FIT] Intercepts-only (re_formula='{re_intercepts}')...")
    try:
        model_int = smf.mixedlm(
            formula,
            data=data,
            groups=data['UID'],
            re_formula=re_intercepts
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_int = model_int.fit(reml=False, method='powell')

            # Check for convergence warnings
            conv_warnings_int = [str(warning.message) for warning in w
                                if 'convergence' in str(warning.message).lower()]

        aic_int = result_int.aic
        converged_int = result_int.converged and len(conv_warnings_int) == 0

        log(f"  AIC: {aic_int:.2f}")
        log(f"  Converged: {converged_int}")
        if conv_warnings_int:
            log(f"  Warnings: {conv_warnings_int}")

    except Exception as e:
        log(f"  [ERROR] Intercepts-only failed: {str(e)}")
        result_int = None
        aic_int = np.inf
        converged_int = False

    # Fit intercepts+slopes
    log(f"\n[FIT] Intercepts+slopes (re_formula='{re_slopes}')...")
    try:
        model_slopes = smf.mixedlm(
            formula,
            data=data,
            groups=data['UID'],
            re_formula=re_slopes
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_slopes = model_slopes.fit(reml=False, method='powell')

            # Check for convergence warnings
            conv_warnings_slopes = [str(warning.message) for warning in w
                                   if 'convergence' in str(warning.message).lower()]

        aic_slopes = result_slopes.aic
        converged_slopes = result_slopes.converged and len(conv_warnings_slopes) == 0

        # Extract slope variance (if converged and slopes present)
        if converged_slopes and result_slopes.cov_re.shape[0] > 1:
            slope_var = result_slopes.cov_re.iloc[1, 1]
        else:
            slope_var = np.nan

        log(f"  AIC: {aic_slopes:.2f}")
        log(f"  Converged: {converged_slopes}")
        log(f"  Slope variance: {slope_var:.4f}" if not np.isnan(slope_var) else "  Slope variance: NA")
        if conv_warnings_slopes:
            log(f"  Warnings: {conv_warnings_slopes}")

    except Exception as e:
        log(f"  [ERROR] Intercepts+slopes failed: {str(e)}")
        result_slopes = None
        aic_slopes = np.inf
        converged_slopes = False
        slope_var = np.nan

    # Compare AICs
    delta_aic = aic_int - aic_slopes  # Positive = slopes improve fit
    slopes_improve = delta_aic > 2.0

    log(f"\n[COMPARISON]")
    log(f"  ΔAIC (intercepts - slopes): {delta_aic:.2f}")
    log(f"  Slopes improve fit (ΔAIC > 2): {slopes_improve}")

    # Interpretation
    if not converged_int and not converged_slopes:
        log(f"  [INTERPRETATION] Both models failed to converge")
        interpretation = "BOTH_FAILED"
    elif not converged_slopes:
        log(f"  [INTERPRETATION] Slopes failed, use intercepts-only")
        interpretation = "SLOPES_FAILED"
    elif not converged_int:
        log(f"  [INTERPRETATION] Intercepts failed but slopes converged (unusual)")
        interpretation = "INTERCEPTS_FAILED"
    elif slopes_improve:
        log(f"  [INTERPRETATION] Slopes improve fit substantially (ΔAIC={delta_aic:.2f})")
        log(f"  → Use slopes model, report slope variance = {slope_var:.4f}")
        interpretation = "SLOPES_WIN"
    else:
        log(f"  [INTERPRETATION] Slopes don't improve fit (ΔAIC={delta_aic:.2f} < 2)")
        log(f"  → Use intercepts-only (simpler model), homogeneous effects")
        interpretation = "INTERCEPTS_WIN"

    # Summary dict
    summary = {
        'model_name': model_name,
        'aic_intercepts': aic_int,
        'aic_slopes': aic_slopes,
        'delta_aic': delta_aic,
        'slopes_improve_fit': slopes_improve,
        'slope_variance': slope_var,
        'converged_intercepts': converged_int,
        'converged_slopes': converged_slopes,
        'interpretation': interpretation
    }

    return result_int, result_slopes, summary

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 05d: Random Slopes Comparison")
        log("="*60)
        log("\nPURPOSE: Test intercepts-only vs intercepts+slopes for top 10 models")
        log("REQUIREMENT: improvement_taxonomy.md Section 4.4 MANDATORY")
        log("BLOCKER: Cannot claim heterogeneous effects without testing slopes")

        # Load data
        log("\n[LOAD] Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_lmm = pd.read_csv(input_path, encoding='utf-8')
        log(f"  Loaded: {len(df_lmm)} rows, {len(df_lmm.columns)} cols")

        # Prepare data
        log("\n[PREP] Creating time transformations...")
        df_prep = prepare_data(df_lmm)
        log(f"  Created: Days, log_Days, sqrt_Days, cbrt_Days, recip_Days, etc.")

        # Fit all model pairs
        results = []

        for model_name, model_spec in TOP_MODELS.items():
            _, _, summary = fit_model_pair(model_name, model_spec, df_prep)
            results.append(summary)

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Save results
        output_path = RQ_DIR / "results" / "step05d_random_slopes_comparison.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"\n[SAVE] Saved results: {output_path}")

        # Generate summary report
        log("\n" + "="*60)
        log("SUMMARY REPORT")
        log("="*60)

        n_converged_both = sum((df_results['converged_intercepts']) &
                               (df_results['converged_slopes']))
        n_slopes_improve = sum(df_results['slopes_improve_fit'] &
                              df_results['converged_slopes'])
        n_slopes_failed = sum(~df_results['converged_slopes'])
        n_intercepts_win = sum((~df_results['slopes_improve_fit']) &
                              (df_results['converged_slopes']))

        log(f"\nConvergence:")
        log(f"  Both structures converged: {n_converged_both}/10")
        log(f"  Slopes failed to converge: {n_slopes_failed}/10")

        log(f"\nModel Selection:")
        log(f"  Slopes improve fit (ΔAIC > 2): {n_slopes_improve}/10")
        log(f"  Intercepts-only preferred (ΔAIC < 2): {n_intercepts_win}/10")

        log(f"\nSlope Variance (for models where slopes converged):")
        slopes_converged = df_results[df_results['converged_slopes']]
        if len(slopes_converged) > 0:
            mean_slope_var = slopes_converged['slope_variance'].mean()
            log(f"  Mean slope variance: {mean_slope_var:.4f}")
            log(f"  Range: [{slopes_converged['slope_variance'].min():.4f}, "
                f"{slopes_converged['slope_variance'].max():.4f}]")

        # Final recommendation
        log(f"\n" + "="*60)
        log("RECOMMENDATION")
        log("="*60)

        if n_slopes_improve >= 7:
            log("\n✅ SLOPES WIN (≥7/10 models)")
            log("   → Use random slopes models")
            log("   → Report: 'Individual differences in forgetting rates confirmed'")
            log("   → Document: Mean slope variance = {:.4f}".format(mean_slope_var))
        elif n_slopes_failed >= 5:
            log("\n⚠️ SLOPES FAILED (≥5/10 models)")
            log("   → Use intercepts-only models")
            log("   → Document: 'Random slopes attempted, convergence failed'")
            log("   → Explain: Insufficient data (N=100, 4 timepoints)")
        elif n_intercepts_win >= 7:
            log("\n✅ INTERCEPTS WIN (≥7/10 models)")
            log("   → Use intercepts-only models")
            log("   → Report: 'Homogeneous forgetting rates confirmed'")
            log("   → Document: 'Slopes tested, variance negligible (ΔAIC < 2)'")
        else:
            log("\n⚠️ MIXED RESULTS")
            log("   → Model-specific decision required")
            log("   → Use slopes for models where ΔAIC > 2")
            log("   → Use intercepts for models where ΔAIC < 2")

        # Save summary report
        summary_path = RQ_DIR / "results" / "step05d_slopes_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("RANDOM SLOPES TESTING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Convergence:\n")
            f.write(f"  Both structures converged: {n_converged_both}/10\n")
            f.write(f"  Slopes failed to converge: {n_slopes_failed}/10\n\n")
            f.write(f"Model Selection:\n")
            f.write(f"  Slopes improve fit (ΔAIC > 2): {n_slopes_improve}/10\n")
            f.write(f"  Intercepts-only preferred: {n_intercepts_win}/10\n\n")
            if len(slopes_converged) > 0:
                f.write(f"Slope Variance:\n")
                f.write(f"  Mean: {mean_slope_var:.4f}\n")
                f.write(f"  Range: [{slopes_converged['slope_variance'].min():.4f}, "
                       f"{slopes_converged['slope_variance'].max():.4f}]\n\n")
            f.write("="*60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(df_results.to_string(index=False))

        log(f"\n[SAVE] Saved summary: {summary_path}")

        log("\n" + "="*60)
        log("[SUCCESS] Step 05d complete")
        log("="*60)

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
