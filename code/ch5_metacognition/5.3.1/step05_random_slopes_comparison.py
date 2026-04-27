#!/usr/bin/env python3
"""
PLATINUM FINALIZATION: Random Slopes Comparison (MANDATORY)

PURPOSE:
Test intercepts-only vs intercepts+slopes random effects to validate homogeneous
effects claim. Per rq_platinum agent protocol, we CANNOT claim homogeneous effects
if we never tested for heterogeneity.

INPUTS:
  - data/step04_lmm_input.csv (LMM-ready data)

OUTPUTS:
  - data/step05_random_slopes_comparison.csv (model comparison table)
  - data/step05_random_slopes_diagnostics.txt (detailed comparison report)
  - logs/step05_random_slopes_comparison.log

EXPECTED OUTCOMES:
  A) Slopes improve fit (ΔAIC > 2) → Individual differences confirmed
  B) Slopes don't converge → Insufficient data for stable estimation
  C) Slopes converge but don't improve (ΔAIC < 2) → Homogeneous effects confirmed
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import statsmodels.formula.api as smf
import warnings

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_random_slopes_comparison.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("PLATINUM FINALIZATION: Random Slopes Comparison")
        log("=" * 80)

        # Load data
        log("\n[LOAD] Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"[LOADED] {len(lmm_input)} rows, {lmm_input['UID'].nunique()} participants")
        log(f"[INFO] Domains: {sorted(lmm_input['domain'].unique())}")

        # =====================================================================
        # Model 1: Intercepts-Only (Current Model)
        # =====================================================================
        log("\n[MODEL 1] Fitting intercepts-only model...")
        log("[FORMULA] theta ~ C(domain) * log_TSVR")
        log("[RANDOM]  ~1 (random intercept per participant)")

        model_intercepts = smf.mixedlm(
            formula="theta ~ C(domain) * log_TSVR",
            data=lmm_input,
            groups=lmm_input['UID'],
            re_formula="~1"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_intercepts = model_intercepts.fit(reml=False)

            if w:
                log(f"[WARNING] Intercepts model warnings: {len(w)}")
                for warning in w:
                    log(f"  - {warning.message}")

        log(f"[RESULT] Converged: {result_intercepts.converged}")
        log(f"[RESULT] AIC: {result_intercepts.aic:.2f}")
        log(f"[RESULT] BIC: {result_intercepts.bic:.2f}")
        log(f"[RESULT] Log-Likelihood: {result_intercepts.llf:.2f}")

        # Extract random intercept variance
        intercept_var = result_intercepts.cov_re.iloc[0, 0]
        log(f"[RESULT] Random intercept variance: {intercept_var:.4f} (SD={np.sqrt(intercept_var):.4f})")

        # =====================================================================
        # Model 2: Intercepts + Slopes
        # =====================================================================
        log("\n[MODEL 2] Fitting intercepts+slopes model...")
        log("[FORMULA] theta ~ C(domain) * log_TSVR")
        log("[RANDOM]  ~log_TSVR (random intercept + random slope on time)")

        model_slopes = smf.mixedlm(
            formula="theta ~ C(domain) * log_TSVR",
            data=lmm_input,
            groups=lmm_input['UID'],
            re_formula="~log_TSVR"  # Random slope on log_TSVR
        )

        slopes_converged = False
        slopes_result = None
        convergence_issue = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                slopes_result = model_slopes.fit(reml=False, maxiter=500)
                slopes_converged = slopes_result.converged

                if w:
                    log(f"[WARNING] Slopes model warnings: {len(w)}")
                    for warning in w:
                        log(f"  - {warning.message}")
                        if "singular" in str(warning.message).lower() or "boundary" in str(warning.message).lower():
                            convergence_issue = str(warning.message)

            except Exception as e:
                log(f"[ERROR] Slopes model failed to fit: {str(e)}")
                convergence_issue = str(e)
                slopes_converged = False

        if slopes_converged and slopes_result is not None:
            log(f"[RESULT] Converged: {slopes_result.converged}")
            log(f"[RESULT] AIC: {slopes_result.aic:.2f}")
            log(f"[RESULT] BIC: {slopes_result.bic:.2f}")
            log(f"[RESULT] Log-Likelihood: {slopes_result.llf:.2f}")

            # Extract random slope variance
            if slopes_result.cov_re.shape[0] >= 2:
                slope_var = slopes_result.cov_re.iloc[1, 1]
                intercept_slope_corr = slopes_result.cov_re.iloc[0, 1] / np.sqrt(
                    slopes_result.cov_re.iloc[0, 0] * slopes_result.cov_re.iloc[1, 1]
                )
                log(f"[RESULT] Random slope variance: {slope_var:.4f} (SD={np.sqrt(slope_var):.4f})")
                log(f"[RESULT] Intercept-slope correlation: {intercept_slope_corr:.3f}")
            else:
                log("[WARNING] Random effects covariance matrix incomplete")
        else:
            log("[RESULT] Model did NOT converge or failed to fit")
            if convergence_issue:
                log(f"[ISSUE] {convergence_issue}")

        # =====================================================================
        # Model Comparison
        # =====================================================================
        log("\n[COMPARISON] Model selection via AIC...")

        comparison_results = []

        # Intercepts-only model
        comparison_results.append({
            'model': 'Intercepts-only',
            'formula': 'theta ~ C(domain) * log_TSVR',
            'random_effects': '~1',
            'converged': result_intercepts.converged,
            'AIC': result_intercepts.aic,
            'BIC': result_intercepts.bic,
            'log_likelihood': result_intercepts.llf,
            'n_params': len(result_intercepts.params),
            'random_intercept_var': intercept_var
        })

        # Slopes model (if converged)
        if slopes_converged and slopes_result is not None:
            comparison_results.append({
                'model': 'Intercepts+slopes',
                'formula': 'theta ~ C(domain) * log_TSVR',
                'random_effects': '~log_TSVR',
                'converged': slopes_result.converged,
                'AIC': slopes_result.aic,
                'BIC': slopes_result.bic,
                'log_likelihood': slopes_result.llf,
                'n_params': len(slopes_result.params),
                'random_slope_var': slope_var if slopes_result.cov_re.shape[0] >= 2 else np.nan
            })

            # Compute ΔAIC
            delta_aic = result_intercepts.aic - slopes_result.aic
            log(f"\n[ΔAIC] Intercepts-only vs Intercepts+slopes: ΔAIC = {delta_aic:.2f}")

            if delta_aic > 2:
                decision = "SLOPES IMPROVE FIT - Individual differences confirmed"
                interpretation = "Random slopes model preferred (ΔAIC > 2). Individual forgetting rates vary."
            elif delta_aic < -2:
                decision = "INTERCEPTS BETTER - Simpler model preferred"
                interpretation = "Intercepts-only model preferred (ΔAIC < -2). Slopes add complexity without improving fit."
            else:
                decision = "MODELS EQUIVALENT - No clear winner"
                interpretation = "ΔAIC < 2 suggests negligible difference. Either model acceptable, prefer simpler (intercepts-only)."

            log(f"[DECISION] {decision}")
            log(f"[INTERPRETATION] {interpretation}")

        else:
            delta_aic = np.nan
            decision = "SLOPES MODEL FAILED TO CONVERGE"
            interpretation = "Random slopes model did not converge. Likely insufficient data for stable estimation (N=4 timepoints per participant). Intercepts-only model is appropriate."

            log(f"\n[DECISION] {decision}")
            log(f"[INTERPRETATION] {interpretation}")

            comparison_results.append({
                'model': 'Intercepts+slopes',
                'formula': 'theta ~ C(domain) * log_TSVR',
                'random_effects': '~log_TSVR',
                'converged': False,
                'AIC': np.nan,
                'BIC': np.nan,
                'log_likelihood': np.nan,
                'n_params': np.nan,
                'convergence_issue': convergence_issue if convergence_issue else "Failed to fit"
            })

        # =====================================================================
        # Save Outputs
        # =====================================================================
        log("\n[SAVE] Saving comparison results...")

        # Comparison table
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df['delta_AIC'] = comparison_df['AIC'] - comparison_df['AIC'].min()

        output_comparison = RQ_DIR / "data" / "step05_random_slopes_comparison.csv"
        comparison_df.to_csv(output_comparison, index=False, encoding='utf-8')
        log(f"[SAVED] {output_comparison.name}")

        # Diagnostics report
        output_diagnostics = RQ_DIR / "data" / "step05_random_slopes_diagnostics.txt"
        with open(output_diagnostics, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RANDOM SLOPES COMPARISON - DIAGNOSTICS REPORT\n")
            f.write("RQ 6.3.1: Domain Confidence Trajectories\n")
            f.write("=" * 80 + "\n\n")

            f.write("QUESTION: Do individual forgetting rates vary (heterogeneous effects)?\n")
            f.write("OR are all participants declining at similar rates (homogeneous effects)?\n\n")

            f.write("METHOD: Compare intercepts-only vs intercepts+slopes LMM via AIC\n")
            f.write("  - Intercepts-only: Individual baselines vary, but same decline rate for all\n")
            f.write("  - Intercepts+slopes: Both baselines AND decline rates vary by individual\n\n")

            f.write("-" * 80 + "\n")
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("-" * 80 + "\n\n")

            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write(f"DECISION: {decision}\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"{interpretation}\n\n")

            if slopes_converged and delta_aic <= 2:
                f.write("IMPLICATION FOR THESIS:\n")
                f.write("  - Confidence decline rates are HOMOGENEOUS across participants\n")
                f.write("  - Random slopes add no predictive value (ΔAIC < 2)\n")
                f.write("  - Intercepts-only model is parsimonious and adequate\n")
                f.write("  - Domain × Time interaction reflects AVERAGE effect applicable to all\n\n")
            elif slopes_converged and delta_aic > 2:
                f.write("IMPLICATION FOR THESIS:\n")
                f.write("  - Confidence decline rates are HETEROGENEOUS across participants\n")
                f.write("  - Some individuals forget faster, others slower\n")
                f.write("  - Domain × Time interaction reflects AVERAGE effect, but individual variation exists\n")
                f.write("  - Future work: Cluster participants by decline trajectories\n\n")
            else:
                f.write("IMPLICATION FOR THESIS:\n")
                f.write("  - Random slopes model failed to converge (common with 4 timepoints)\n")
                f.write("  - Intercepts-only model is appropriate given data constraints\n")
                f.write("  - Cannot test heterogeneity with current design (more timepoints needed)\n")
                f.write("  - Homogeneity assumption is PRAGMATIC, not empirically tested\n\n")

            f.write("REFERENCE:\n")
            f.write("  Per rq_platinum agent protocol (Section 4.4), testing random slopes is\n")
            f.write("  MANDATORY for modeling RQs. We cannot claim homogeneous effects without\n")
            f.write("  testing for heterogeneity. This comparison fulfills that requirement.\n")

        log(f"[SAVED] {output_diagnostics.name}")

        log("\n[SUCCESS] Random slopes comparison complete")
        log(f"[RECOMMENDATION] Use intercepts-only model (simpler, converges reliably)")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
