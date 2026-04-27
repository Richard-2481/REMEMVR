#!/usr/bin/env python3
"""
Random Slopes Comparison - MANDATORY per improvement_taxonomy.md Section 4.4

PURPOSE:
Test whether random slopes improve model fit compared to intercepts-only.
This is REQUIRED to claim homogeneous effects - we cannot assume no individual
differences without testing for them.

APPROACH:
1. Fit intercepts-only model (random intercepts by UID)
2. Fit intercepts+slopes model (random intercepts + slopes on TSVR_log by UID)
3. Compare AIC
4. Report random slope variance
5. Document convergence

EXPECTED OUTCOMES:
- Option A: Slopes improve fit (ΔAIC > 2) → Use slopes, report individual differences
- Option B: Convergence failure → Document attempt, explain why
- Option C: Slopes converge but don't improve (ΔAIC < 2) → Keep intercepts, confirm homogeneous effects
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"
INPUT_LMM = RQ_DIR / "data" / "step04_lmm_input.csv"
OUTPUT_REPORT = RQ_DIR / "results" / "random_slopes_comparison.txt"

def log(msg):
    """Write to log file and console."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("="*70)
        log("RANDOM SLOPES COMPARISON - MANDATORY CHECK")
        log("="*70)
        log(f"Date: 2025-12-27")
        log(f"Purpose: Test intercepts-only vs intercepts+slopes")
        log("")

        # Load data
        log("[LOAD] Loading LMM input data...")
        df = pd.read_csv(INPUT_LMM)
        log(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        log(f"  N participants: {df['UID'].nunique()}")

        # Ensure categorical coding
        df["congruence"] = pd.Categorical(
            df["congruence"],
            categories=["common", "congruent", "incongruent"],
            ordered=True
        )

        # Use Log model (best from step05)
        formula = "theta ~ TSVR_log * C(congruence, Treatment('common'))"

        # =====================================================================
        # MODEL 1: Intercepts-only (current assumption in step05)
        # =====================================================================
        log("\n[FIT] Model 1: Intercepts-only")
        log(f"  Formula: {formula}")
        log(f"  Random effects: ~1 (intercepts only)")

        try:
            model_intercepts = smf.mixedlm(
                formula=formula,
                data=df,
                groups=df['UID'],
                re_formula="~1"  # INTERCEPTS ONLY
            )
            result_intercepts = model_intercepts.fit(method=['lbfgs'], reml=False)

            if result_intercepts.converged:
                log(f"  [CONVERGED] Yes")
            else:
                log(f"  [CONVERGED] No (WARNING)")

            log(f"  AIC: {result_intercepts.aic:.2f}")
            log(f"  BIC: {result_intercepts.bic:.2f}")
            log(f"  Random intercept variance: {result_intercepts.cov_re.iloc[0,0]:.4f}")

        except Exception as e:
            log(f"  [ERROR] Intercepts-only model failed: {str(e)}")
            raise

        # =====================================================================
        # MODEL 2: Intercepts + Slopes
        # =====================================================================
        log("\n[FIT] Model 2: Intercepts + Slopes")
        log(f"  Formula: {formula}")
        log(f"  Random effects: ~TSVR_log (intercepts + slopes)")

        try:
            model_slopes = smf.mixedlm(
                formula=formula,
                data=df,
                groups=df['UID'],
                re_formula="~TSVR_log"  # INTERCEPTS + SLOPES
            )
            result_slopes = model_slopes.fit(method=['lbfgs'], reml=False)

            if result_slopes.converged:
                log(f"  [CONVERGED] Yes")
                convergence_status = "SUCCESS"
            else:
                log(f"  [CONVERGED] No (WARNING)")
                convergence_status = "FAILED"

            log(f"  AIC: {result_slopes.aic:.2f}")
            log(f"  BIC: {result_slopes.bic:.2f}")

            # Extract random effects variances
            if hasattr(result_slopes.cov_re, 'iloc'):
                intercept_var = result_slopes.cov_re.iloc[0, 0]
                slope_var = result_slopes.cov_re.iloc[1, 1]
                cov = result_slopes.cov_re.iloc[0, 1]
            else:
                intercept_var = result_slopes.cov_re[0, 0]
                slope_var = result_slopes.cov_re[1, 1]
                cov = result_slopes.cov_re[0, 1]

            log(f"  Random intercept variance: {intercept_var:.4f}")
            log(f"  Random slope variance: {slope_var:.4f}")
            log(f"  Intercept-slope covariance: {cov:.4f}")

            # Check for boundary warnings (variance near zero)
            if slope_var < 0.001:
                log(f"  [WARNING] Random slope variance very small ({slope_var:.6f})")
                log(f"           This suggests slopes may not be needed.")

        except Exception as e:
            log(f"  [ERROR] Intercepts+slopes model failed: {str(e)}")
            convergence_status = "ERROR"
            result_slopes = None

        # =====================================================================
        # COMPARISON
        # =====================================================================
        log("\n" + "="*70)
        log("MODEL COMPARISON")
        log("="*70)

        if result_slopes is not None and result_slopes.converged:
            delta_aic = result_intercepts.aic - result_slopes.aic
            delta_bic = result_intercepts.bic - result_slopes.bic

            log(f"Intercepts-only AIC: {result_intercepts.aic:.2f}")
            log(f"Intercepts+slopes AIC: {result_slopes.aic:.2f}")
            log(f"ΔAIC (intercepts - slopes): {delta_aic:.2f}")
            log(f"ΔBIC (intercepts - slopes): {delta_bic:.2f}")
            log("")

            # Decision criteria
            if delta_aic > 2:
                decision = "SLOPES IMPROVE FIT"
                recommendation = "Use random slopes model. Individual differences in forgetting rates exist."
                log(f"[DECISION] {decision}")
                log(f"  Slopes model AIC is {delta_aic:.2f} points LOWER (better).")
                log(f"  ΔAIC > 2 indicates strong evidence for random slopes.")
                log(f"  Random slope variance = {slope_var:.4f} (non-negligible)")
                log("")
                log(f"[RECOMMENDATION] {recommendation}")
                log(f"  - Update step05 to use random slopes explicitly")
                log(f"  - Report in summary.md: Individual differences in forgetting rates")
                log(f"  - Document slope variance in results")

            elif delta_aic < -2:
                decision = "INTERCEPTS-ONLY PREFERRED"
                recommendation = "Random slopes overfit. Use intercepts-only model (homogeneous effects confirmed)."
                log(f"[DECISION] {decision}")
                log(f"  Intercepts-only AIC is {abs(delta_aic):.2f} points LOWER (better).")
                log(f"  Slopes model is MORE complex but WORSE fit.")
                log(f"  Random slope variance = {slope_var:.4f}")
                log("")
                log(f"[RECOMMENDATION] {recommendation}")
                log(f"  - Keep current intercepts-only specification")
                log(f"  - Document that random slopes were tested and rejected")
                log(f"  - Homogeneous forgetting rates across participants confirmed")

            else:  # -2 <= delta_aic <= 2
                decision = "AMBIGUOUS (ΔAIC < 2)"
                recommendation = "Use simpler model (intercepts-only) per parsimony principle."
                log(f"[DECISION] {decision}")
                log(f"  ΔAIC = {delta_aic:.2f} (within [-2, 2] range)")
                log(f"  Models essentially equivalent in fit.")
                log(f"  Random slope variance = {slope_var:.4f}")
                log("")
                log(f"[RECOMMENDATION] {recommendation}")
                log(f"  - Use intercepts-only per parsimony (simpler model)")
                log(f"  - Document that slopes were tested but provided negligible improvement")
                log(f"  - Individual differences in forgetting rates are minimal")

        elif convergence_status == "FAILED":
            decision = "SLOPES CONVERGENCE FAILURE"
            recommendation = "Use intercepts-only. Document convergence failure."
            log(f"[DECISION] {decision}")
            log(f"  Random slopes model did not converge.")
            log(f"  Likely reason: Only 4 timepoints insufficient for stable slope estimation.")
            log("")
            log(f"[RECOMMENDATION] {recommendation}")
            log(f"  - Use intercepts-only model")
            log(f"  - Document in summary.md: Random slopes attempted but convergence failed")
            log(f"  - Note: 4 timepoints (T1-T4) may be insufficient for random slopes")

        else:  # ERROR
            decision = "SLOPES MODEL ERROR"
            recommendation = "Use intercepts-only. Document error."
            log(f"[DECISION] {decision}")
            log(f"  Random slopes model threw error during fitting.")
            log("")
            log(f"[RECOMMENDATION] {recommendation}")
            log(f"  - Use intercepts-only model")
            log(f"  - Document error in validation.md")

        # =====================================================================
        # SAVE REPORT
        # =====================================================================
        log("\n[SAVE] Writing comparison report...")

        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RANDOM SLOPES COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: 2025-12-27\n")
            f.write(f"Purpose: Test intercepts-only vs intercepts+slopes (MANDATORY per Section 4.4)\n\n")

            f.write("MODEL 1: Intercepts-only\n")
            f.write("-" * 40 + "\n")
            f.write(f"AIC: {result_intercepts.aic:.2f}\n")
            f.write(f"BIC: {result_intercepts.bic:.2f}\n")
            f.write(f"Random intercept variance: {result_intercepts.cov_re.iloc[0,0]:.4f}\n\n")

            if result_slopes is not None and convergence_status == "SUCCESS":
                f.write("MODEL 2: Intercepts + Slopes\n")
                f.write("-" * 40 + "\n")
                f.write(f"AIC: {result_slopes.aic:.2f}\n")
                f.write(f"BIC: {result_slopes.bic:.2f}\n")
                f.write(f"Random intercept variance: {intercept_var:.4f}\n")
                f.write(f"Random slope variance: {slope_var:.4f}\n")
                f.write(f"Intercept-slope covariance: {cov:.4f}\n\n")

                f.write("COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(f"ΔAIC (intercepts - slopes): {delta_aic:.2f}\n")
                f.write(f"ΔBIC (intercepts - slopes): {delta_bic:.2f}\n\n")
            else:
                f.write(f"MODEL 2: {convergence_status}\n\n")

            f.write("DECISION\n")
            f.write("-" * 40 + "\n")
            f.write(f"{decision}\n\n")

            f.write("RECOMMENDATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"{recommendation}\n\n")

            f.write("INTERPRETATION\n")
            f.write("-" * 40 + "\n")
            if result_slopes is not None and convergence_status == "SUCCESS":
                if delta_aic > 2:
                    f.write("Random slopes improve model fit, indicating individual differences in\n")
                    f.write("forgetting rates exist. Some participants forget faster/slower than others.\n")
                elif delta_aic < -2:
                    f.write("Random slopes overfit the data. Forgetting rates are homogeneous across\n")
                    f.write("participants (no meaningful individual differences).\n")
                else:
                    f.write("Random slopes provide negligible improvement. Individual differences in\n")
                    f.write("forgetting rates are minimal. Simpler model (intercepts-only) preferred.\n")
            else:
                f.write("Random slopes model failed to converge, likely due to limited timepoints (N=4).\n")
                f.write("With only 4 observations per participant, slope estimation is unstable.\n")

        log(f"[SAVED] {OUTPUT_REPORT.name}")
        log("\n[SUCCESS] Random slopes comparison complete")
        log(f"[ACTION REQUIRED] Update validation.md with findings")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
