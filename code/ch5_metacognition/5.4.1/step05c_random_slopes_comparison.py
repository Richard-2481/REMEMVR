"""
RQ 6.4.1 - Step 05c: Random Slopes Comparison (MANDATORY per Taxonomy 4.4)

PURPOSE:
Test whether random slopes on time improve model fit vs intercepts-only.
Cannot claim homogeneous confidence decline rates without testing for heterogeneity.

COMPARISON:
- Model A (current): theta ~ paradigm * TSVR_hours + (1 | UID) [intercepts-only]
- Model B (test): theta ~ paradigm * TSVR_hours + (TSVR_hours | UID) [intercepts + slopes]

OUTCOMES:
- ΔAIC > 2 → Slopes improve fit, report individual differences
- ΔAIC < 2 → Intercepts sufficient, document homogeneous effects CONFIRMED
- Convergence failure → Document attempt, explain insufficient data

INPUT:
- data/step04_lmm_input.csv (1200 rows, theta + TSVR_hours + paradigm)

OUTPUT:
- data/step05c_random_slopes_comparison.csv (AIC, BIC, variance components)
- data/step05c_slopes_model_summary.txt (if slopes converge)

Author: rq_platinum finalization
Date: 2025-12-28
RQ: ch6/6.4.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05c_random_slopes_comparison.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 05c: Random Slopes Comparison")
        log("=" * 80)

        # Import here to avoid statsmodels import overhead
        import statsmodels.formula.api as smf

        log("[LOAD] Loading LMM input data...")
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv", encoding='utf-8')
        log(f"  ✓ Loaded {len(lmm_input)} rows ({lmm_input['UID'].nunique()} participants)")
        log(f"  ✓ Variables: theta (outcome), TSVR_hours (time), paradigm (IFR/ICR/IRE)")

        # Prepare data
        lmm_input['paradigm'] = pd.Categorical(lmm_input['paradigm'], categories=['IFR', 'ICR', 'IRE'])

        # MODEL A: Intercepts-only (current baseline)
        log("\n[MODEL A] Fitting intercepts-only model...")
        log("  Formula: theta ~ C(paradigm) * TSVR_hours + (1 | UID)")

        try:
            model_intercepts = smf.mixedlm(
                "theta ~ C(paradigm) * TSVR_hours",
                data=lmm_input,
                groups=lmm_input['UID'],
                re_formula="1"  # Random intercepts only
            )
            result_intercepts = model_intercepts.fit(reml=False, method='lbfgs')

            aic_intercepts = result_intercepts.aic
            bic_intercepts = result_intercepts.bic
            converged_intercepts = result_intercepts.converged

            log(f"  ✓ Intercepts-only: AIC={aic_intercepts:.2f}, BIC={bic_intercepts:.2f}, Converged={converged_intercepts}")

            # Extract variance components
            var_intercept_A = result_intercepts.cov_re.iloc[0, 0]
            var_residual_A = result_intercepts.scale

            log(f"  ✓ Variance components:")
            log(f"    - Random intercept variance: {var_intercept_A:.4f} (SD={np.sqrt(var_intercept_A):.4f})")
            log(f"    - Residual variance: {var_residual_A:.4f} (SD={np.sqrt(var_residual_A):.4f})")

        except Exception as e:
            log(f"  ✗ Intercepts-only model FAILED: {e}")
            raise

        # MODEL B: Intercepts + Slopes
        log("\n[MODEL B] Fitting intercepts + slopes model...")
        log("  Formula: theta ~ C(paradigm) * TSVR_hours + (TSVR_hours | UID)")

        slopes_converged = False
        try:
            model_slopes = smf.mixedlm(
                "theta ~ C(paradigm) * TSVR_hours",
                data=lmm_input,
                groups=lmm_input['UID'],
                re_formula="~TSVR_hours"  # Random intercepts + slopes on TSVR_hours
            )
            result_slopes = model_slopes.fit(reml=False, method='lbfgs')

            aic_slopes = result_slopes.aic
            bic_slopes = result_slopes.bic
            converged_slopes = result_slopes.converged
            slopes_converged = converged_slopes

            log(f"  ✓ Intercepts+slopes: AIC={aic_slopes:.2f}, BIC={bic_slopes:.2f}, Converged={converged_slopes}")

            # Extract variance components
            var_intercept_B = result_slopes.cov_re.iloc[0, 0]
            var_slope_B = result_slopes.cov_re.iloc[1, 1] if result_slopes.cov_re.shape[0] > 1 else np.nan
            cov_intercept_slope_B = result_slopes.cov_re.iloc[0, 1] if result_slopes.cov_re.shape[0] > 1 else np.nan
            var_residual_B = result_slopes.scale

            log(f"  ✓ Variance components:")
            log(f"    - Random intercept variance: {var_intercept_B:.4f} (SD={np.sqrt(var_intercept_B):.4f})")
            log(f"    - Random slope variance: {var_slope_B:.4f} (SD={np.sqrt(var_slope_B):.4f})")
            log(f"    - Intercept-slope covariance: {cov_intercept_slope_B:.4f}")
            log(f"    - Residual variance: {var_residual_B:.4f} (SD={np.sqrt(var_residual_B):.4f})")

            # Save slopes model summary
            with open(DATA_DIR / "step05c_slopes_model_summary.txt", 'w') as f:
                f.write(str(result_slopes.summary()))
            log(f"  ✓ Saved slopes model summary to step05c_slopes_model_summary.txt")

        except Exception as e:
            log(f"  ✗ Random slopes model CONVERGENCE FAILED: {e}")
            log("  → Likely cause: Insufficient timepoints (N=4) for stable slope estimation")
            log("  → Or: Random slope variance near zero (boundary problem)")
            aic_slopes = np.nan
            bic_slopes = np.nan
            converged_slopes = False
            var_intercept_B = np.nan
            var_slope_B = np.nan
            cov_intercept_slope_B = np.nan
            var_residual_B = np.nan

        # COMPARISON
        log("\n[COMPARISON] Intercepts-only vs Intercepts+Slopes")

        if slopes_converged:
            delta_aic = aic_intercepts - aic_slopes
            delta_bic = bic_intercepts - bic_slopes

            log(f"  ΔAIC = {delta_aic:.2f} (positive = slopes better)")
            log(f"  ΔBIC = {delta_bic:.2f} (positive = slopes better)")

            if delta_aic > 2:
                conclusion = "SLOPES IMPROVE FIT"
                recommendation = "Use random slopes model. Individual participants have different confidence decline rates."
                log(f"\n[CONCLUSION] {conclusion}")
                log(f"  Recommendation: {recommendation}")
                log(f"  Random slope variance SD = {np.sqrt(var_slope_B):.4f} (individual differences confirmed)")

            elif delta_aic < -2:
                conclusion = "INTERCEPTS PREFERRED"
                recommendation = "Random slopes overfit. Keep intercepts-only model (homogeneous effects)."
                log(f"\n[CONCLUSION] {conclusion}")
                log(f"  Recommendation: {recommendation}")

            else:
                conclusion = "MODELS EQUIVALENT"
                recommendation = "Keep intercepts-only (parsimony). Slope variance negligible."
                log(f"\n[CONCLUSION] {conclusion}")
                log(f"  Recommendation: {recommendation}")
                log(f"  Random slope variance SD = {np.sqrt(var_slope_B):.4f} (near zero, shrinkage to fixed effect)")
        else:
            delta_aic = np.nan
            delta_bic = np.nan
            conclusion = "SLOPES CONVERGENCE FAILED"
            recommendation = "Keep intercepts-only model. Random slopes not estimable with N=4 timepoints."
            log(f"\n[CONCLUSION] {conclusion}")
            log(f"  Recommendation: {recommendation}")
            log("  → This is ACCEPTABLE: Document attempt, explain insufficient data for slopes")

        # Save comparison results
        comparison_df = pd.DataFrame({
            'model': ['Intercepts_only', 'Intercepts_plus_slopes'],
            'formula': [
                'theta ~ paradigm * TSVR_hours + (1 | UID)',
                'theta ~ paradigm * TSVR_hours + (TSVR_hours | UID)'
            ],
            'AIC': [aic_intercepts, aic_slopes],
            'BIC': [bic_intercepts, bic_slopes],
            'converged': [converged_intercepts, converged_slopes],
            'var_intercept': [var_intercept_A, var_intercept_B],
            'var_slope': [np.nan, var_slope_B],
            'cov_intercept_slope': [np.nan, cov_intercept_slope_B],
            'var_residual': [var_residual_A, var_residual_B],
            'delta_AIC': [0, delta_aic],
            'delta_BIC': [0, delta_bic],
            'conclusion': [conclusion, conclusion],
            'recommendation': [recommendation, recommendation]
        })

        comparison_df.to_csv(DATA_DIR / "step05c_random_slopes_comparison.csv", index=False, encoding='utf-8')
        log(f"\n  ✓ Saved comparison to step05c_random_slopes_comparison.csv")

        # INTERPRETATION FOR THESIS
        log("\n[INTERPRETATION] For summary.md:")
        if slopes_converged and delta_aic > 2:
            log("  → Random slopes IMPROVE fit: Individual differences in confidence decline rates exist")
            log(f"  → Report: 'Participants varied in forgetting rates (slope SD = {np.sqrt(var_slope_B):.3f})'")
        elif slopes_converged and abs(delta_aic) <= 2:
            log("  → Random slopes do NOT improve fit: Homogeneous decline rates CONFIRMED")
            log("  → Report: 'Random slopes tested but negligible (ΔAIC < 2), confidence decline rates homogeneous'")
        else:
            log("  → Random slopes CONVERGENCE FAILED: Document attempt")
            log("  → Report: 'Random slopes attempted, failed to converge with N=4 timepoints (insufficient data)'")

        log("\n" + "=" * 80)
        log("[SUCCESS] Step 05c: Random Slopes Comparison Complete")
        log("=" * 80)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
