#!/usr/bin/env python3
"""
RQ 6.3.3: Random Slopes Comparison
==================================
Tests whether random slopes on TSVR_hours improve model fit compared to
intercepts-only random effects structure.

MANDATORY for quality validation (per validation process Step 12).

Expected outcome: Random slope variance near zero (σ²=0.000006 from summary.md),
suggesting homogeneous decline rates (age-invariant trajectories).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.3
LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"

# Input file (from step01)
INPUT_FILE = RQ_DIR / "data" / "step01_lmm_input.csv"


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def compare_random_slopes():
    """
    Compare intercepts-only vs intercepts+slopes random effects structures.

    Tests via Likelihood Ratio Test (LRT) and AIC comparison.
    """
    log("=" * 70)
    log("RANDOM SLOPES COMPARISON FOR RQ 6.3.3")
    log("=" * 70)
    log("\nPurpose: Test if random slopes on TSVR_hours improve model fit")
    log("Context: MANDATORY for quality validation (validation Step 12)")
    log("")

    # Load data
    log(f"Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    log(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    log(f"Participants: {df['UID'].nunique()}, Observations per UID: {len(df) // df['UID'].nunique()}")

    # Formula (same fixed effects for both models)
    formula = "theta_confidence ~ TSVR_hours * Age_c * C(Domain)"
    log(f"\nFixed effects formula: {formula}")
    # Model 1: Intercepts-only (random intercepts by UID)
    log("\n" + "-" * 70)
    log("MODEL 1: INTERCEPTS-ONLY (baseline)")
    log("-" * 70)
    log("Random effects: (1 | UID) - random intercept only")

    model_intercepts = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df['UID'],
        re_formula="~1"  # Intercepts only
    )

    log("\nFitting intercepts-only model...")
    result_intercepts = model_intercepts.fit(method='powell', maxiter=1000, reml=False)
    log(f"Converged: {result_intercepts.converged}")
    log(f"Log-likelihood: {result_intercepts.llf:.4f}")
    log(f"AIC: {result_intercepts.aic:.2f}")
    log(f"BIC: {result_intercepts.bic:.2f}")

    # Random effects variance
    log(f"\nRandom effects variance (intercepts): {result_intercepts.cov_re.iloc[0, 0]:.6f}")
    # Model 2: Intercepts + Slopes (random intercept + random slope on TSVR_hours)
    log("\n" + "-" * 70)
    log("MODEL 2: INTERCEPTS + SLOPES")
    log("-" * 70)
    log("Random effects: (1 + TSVR_hours | UID) - random intercept + slope")

    model_slopes = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df['UID'],
        re_formula="~TSVR_hours"  # Intercepts + slopes
    )

    log("\nFitting intercepts+slopes model...")
    result_slopes = model_slopes.fit(method='powell', maxiter=1000, reml=False)
    log(f"Converged: {result_slopes.converged}")
    log(f"Log-likelihood: {result_slopes.llf:.4f}")
    log(f"AIC: {result_slopes.aic:.2f}")
    log(f"BIC: {result_slopes.bic:.2f}")

    # Random effects variance
    log(f"\nRandom effects covariance matrix:")
    log(f"{result_slopes.cov_re}")

    if result_slopes.converged and result_slopes.cov_re.shape[0] >= 2:
        slope_var = result_slopes.cov_re.iloc[1, 1]
        slope_sd = np.sqrt(slope_var)
        log(f"\nRandom slope variance: {slope_var:.8f}")
        log(f"Random slope SD: {slope_sd:.8f}")
    else:
        slope_var = np.nan
        slope_sd = np.nan
        log(f"\nWARNING: Could not extract random slope variance (convergence issue?)")
    # Model Comparison
    log("\n" + "=" * 70)
    log("MODEL COMPARISON")
    log("=" * 70)

    delta_aic = result_intercepts.aic - result_slopes.aic
    delta_bic = result_intercepts.bic - result_slopes.bic
    delta_llf = result_slopes.llf - result_intercepts.llf

    log(f"\nAIC comparison:")
    log(f"  Intercepts-only AIC: {result_intercepts.aic:.2f}")
    log(f"  Intercepts+slopes AIC: {result_slopes.aic:.2f}")
    log(f"  ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    log(f"  Interpretation: {'Slopes improve fit' if delta_aic > 2 else 'No improvement' if delta_aic < -2 else 'Equivalent fit'}")

    log(f"\nBIC comparison:")
    log(f"  Intercepts-only BIC: {result_intercepts.bic:.2f}")
    log(f"  Intercepts+slopes BIC: {result_slopes.bic:.2f}")
    log(f"  ΔBIC (Intercepts - Slopes): {delta_bic:.2f}")
    log(f"  Interpretation: {'Slopes improve fit' if delta_bic > 2 else 'No improvement' if delta_bic < -2 else 'Equivalent fit'}")

    log(f"\nLog-likelihood comparison:")
    log(f"  ΔLog-likelihood (Slopes - Intercepts): {delta_llf:.4f}")

    # Likelihood Ratio Test (LRT)
    # H0: Slopes model does not improve fit (slope variance = 0)
    # Test statistic: -2 * (LLF_intercepts - LLF_slopes) ~ χ²(df=2)
    # df=2 because slopes model adds 2 parameters: slope variance + intercept-slope covariance
    from scipy import stats as sp_stats
    lrt_stat = -2 * (result_intercepts.llf - result_slopes.llf)
    lrt_p = sp_stats.chi2.sf(lrt_stat, df=2)

    log(f"\nLikelihood Ratio Test (LRT):")
    log(f"  LRT statistic: {lrt_stat:.4f}")
    log(f"  df: 2 (slope variance + covariance)")
    log(f"  p-value: {lrt_p:.4f}")
    log(f"  Significant (p < 0.05): {'YES - Slopes improve fit' if lrt_p < 0.05 else 'NO - Slopes do not improve fit'}")
    # Interpretation (3 outcomes from validation Step 12C)
    log("\n" + "=" * 70)
    log("INTERPRETATION")
    log("=" * 70)

    # Determine outcome
    slopes_converged = result_slopes.converged
    boundary_warning = slope_var < 1e-6 if not np.isnan(slope_var) else True
    slopes_improve = delta_aic > 2

    if not slopes_converged or boundary_warning:
        outcome = "OPTION B: Slopes Don't Converge / Overfit"
        recommendation = "Keep intercepts-only model"
        interpretation = "Insufficient data for stable slope estimation"
        impact = "Cannot definitively test homogeneity hypothesis"
        log(f"\n🟡 {outcome}")
        log(f"   - Random slope variance: {slope_var:.8f} (boundary estimate)")
        log(f"   - Convergence: {slopes_converged}")
        log(f"   - INTERPRETATION: {interpretation}")
        log(f"   - RECOMMENDATION: {recommendation}")
        log(f"   - LIMITATION: {impact}")

    elif slopes_improve:
        outcome = "OPTION A: Slopes Improve Fit (ΔAIC > 2)"
        recommendation = "Use slopes model for downstream analyses"
        interpretation = "Individual differences in decline rates CONFIRMED"
        impact = "May reduce model uncertainty (individual trajectories captured)"
        log(f"\n🔴 {outcome}")
        log(f"   - ΔAIC: {delta_aic:.2f} > 2")
        log(f"   - Random slope SD: {slope_sd:.6f}")
        log(f"   - INTERPRETATION: {interpretation}")
        log(f"   - RECOMMENDATION: {recommendation}")
        log(f"   - IMPACT: {impact}")

    else:
        outcome = "OPTION C: Slopes Converge But Don't Improve (ΔAIC < 2)"
        recommendation = "Keep intercepts-only model (validated choice, not assumption)"
        interpretation = "Homogeneous effects CONFIRMED (tested and validated)"
        impact = "Can now claim homogeneity with evidence (not assumption)"
        log(f"\n🟢 {outcome}")
        log(f"   - ΔAIC: {delta_aic:.2f}, |ΔAIC| < 2")
        log(f"   - Random slope variance: {slope_var:.8f} (negligible)")
        log(f"   - INTERPRETATION: {interpretation}")
        log(f"   - RECOMMENDATION: {recommendation}")
        log(f"   - STRENGTH: {impact}")
    # Save Results
    log("\n" + "=" * 70)
    log("SAVING RESULTS")
    log("=" * 70)

    comparison = pd.DataFrame({
        'model': ['Intercepts_Only', 'Intercepts_Slopes'],
        'converged': [result_intercepts.converged, result_slopes.converged],
        'log_likelihood': [result_intercepts.llf, result_slopes.llf],
        'aic': [result_intercepts.aic, result_slopes.aic],
        'bic': [result_intercepts.bic, result_slopes.bic],
        'delta_aic': [0.0, delta_aic],
        'delta_bic': [0.0, delta_bic],
        'random_slope_var': [0.0, slope_var if not np.isnan(slope_var) else 0.0],
        'random_slope_sd': [0.0, slope_sd if not np.isnan(slope_sd) else 0.0]
    })

    output_path = RQ_DIR / "data" / "random_slopes_comparison.csv"
    comparison.to_csv(output_path, index=False)
    log(f"\nComparison table saved: {output_path}")

    # Save detailed summary
    summary_path = RQ_DIR / "data" / "random_slopes_comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM SLOPES COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Outcome: {outcome}\n\n")
        f.write(f"Model Comparison:\n")
        f.write(f"  Intercepts-only AIC: {result_intercepts.aic:.2f}\n")
        f.write(f"  Intercepts+slopes AIC: {result_slopes.aic:.2f}\n")
        f.write(f"  ΔAIC: {delta_aic:.2f}\n\n")
        f.write(f"  Intercepts-only BIC: {result_intercepts.bic:.2f}\n")
        f.write(f"  Intercepts+slopes BIC: {result_slopes.bic:.2f}\n")
        f.write(f"  ΔBIC: {delta_bic:.2f}\n\n")
        f.write(f"Likelihood Ratio Test:\n")
        f.write(f"  LRT statistic: {lrt_stat:.4f}\n")
        f.write(f"  p-value: {lrt_p:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'YES' if lrt_p < 0.05 else 'NO'}\n\n")
        f.write(f"Random Slope Variance: {slope_var:.8f}\n")
        f.write(f"Random Slope SD: {slope_sd:.8f}\n\n")
        f.write(f"Recommendation: {recommendation}\n")
        f.write(f"Interpretation: {interpretation}\n")
        f.write(f"Impact: {impact}\n")

    log(f"Summary saved: {summary_path}")

    log("\n" + "=" * 70)
    log("RANDOM SLOPES COMPARISON COMPLETE")
    log("=" * 70)
    log(f"\nFINAL RECOMMENDATION: {recommendation}")
    log(f"INTERPRETATION: {interpretation}")

    return comparison, outcome, recommendation


def main():
    """Execute random slopes comparison."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.3.3: Random Slopes Comparison\n")
        f.write("=" * 70 + "\n\n")

    try:
        comparison, outcome, recommendation = compare_random_slopes()

        log("\n✅ Random slopes comparison completed successfully")
        log(f"   Output: {RQ_DIR / 'data' / 'random_slopes_comparison.csv'}")
        log(f"   Summary: {RQ_DIR / 'data' / 'random_slopes_comparison_summary.txt'}")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
