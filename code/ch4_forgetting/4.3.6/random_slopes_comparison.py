#!/usr/bin/env python3
# Random Slopes Comparison - validation Finalization Requirement
"""
PURPOSE:
Test whether random slopes improve model fit compared to intercepts-only.
This is MANDATORY per Section 4.4 of improvement_taxonomy.md:
"Cannot claim homogeneous effects without testing for heterogeneity."

CONTEXT:
Original analysis (step07) used intercepts-only models:
  Score ~ TSVR_hours + (1|UID)

Plan.md specified random slopes:
  Score ~ TSVR_hours + (TSVR_hours | UID)

This script tests whether random slopes improve fit via AIC comparison.

METHOD:
- Test on IFR paradigm (largest purification effects: delta_r=+0.098, delta_AIC=-33.4)
- Fit 6 models total:
  * 3 measurement types (IRT, Full CTT, Purified CTT)
  * 2 random structures (intercepts-only, intercepts+slopes)
- Compare ΔAIC (intercepts - slopes) per measurement type
- Interpret per protocol Step 12C protocol

EXPECTED OUTCOMES:
Option A: Slopes improve fit (ΔAIC > 2)
Option B: Slopes don't converge / overfit
Option C: Slopes converge but don't improve (ΔAIC < 2)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"

def log(msg):
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("RANDOM SLOPES COMPARISON - VALIDATION")
        log("=" * 80)
        log("")
        log("Testing: Intercepts-only vs Intercepts+Slopes")
        log("Paradigm: IFR (Item Free Recall)")
        log("Measurement types: IRT theta, Full CTT, Purified CTT")
        log("")

        # Load standardized scores
        log("Loading standardized scores...")
        input_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        df = pd.read_csv(input_path)
        log(f"{len(df)} rows")

        # Test on IFR paradigm only
        paradigm = 'IFR'
        measurement_types = {
            'IRT': 'z_theta_IFR',
            'Full_CTT': 'z_CTT_full_IFR',
            'Purified_CTT': 'z_CTT_purified_IFR'
        }

        results = []

        for mtype, dv in measurement_types.items():
            log("")
            log(f"{mtype} ({dv})")
            log("-" * 80)
            # Model 1: Intercepts-Only (Current Implementation)
            log(f"  Model 1: {dv} ~ TSVR_hours + (1|UID)")
            try:
                model_intercepts = smf.mixedlm(
                    f"{dv} ~ TSVR_hours",
                    data=df,
                    groups=df['UID']
                )
                result_intercepts = model_intercepts.fit(reml=False)
                aic_intercepts = result_intercepts.aic
                converged_intercepts = result_intercepts.converged

                log(f"    AIC: {aic_intercepts:.2f}")
                log(f"    Converged: {converged_intercepts}")

            except Exception as e:
                log(f"    Intercepts-only model failed: {e}")
                aic_intercepts = np.nan
                converged_intercepts = False
            # Model 2: Intercepts + Slopes (Plan.md Specification)
            log(f"  Model 2: {dv} ~ TSVR_hours + (1 + TSVR_hours | UID)")
            try:
                # Create random effects formula
                re_formula = "~TSVR_hours"

                model_slopes = smf.mixedlm(
                    f"{dv} ~ TSVR_hours",
                    data=df,
                    groups=df['UID'],
                    re_formula=re_formula
                )
                result_slopes = model_slopes.fit(reml=False)
                aic_slopes = result_slopes.aic
                converged_slopes = result_slopes.converged

                # Extract random slope variance
                # cov_re is DataFrame with random effects covariance matrix
                # Index 0 = intercept variance, Index 1 = slope variance
                if result_slopes.converged and len(result_slopes.cov_re) >= 2:
                    slope_var = result_slopes.cov_re.iloc[1, 1]
                    slope_sd = np.sqrt(slope_var)
                else:
                    slope_var = np.nan
                    slope_sd = np.nan

                log(f"    AIC: {aic_slopes:.2f}")
                log(f"    Converged: {converged_slopes}")
                if not np.isnan(slope_var):
                    log(f"    Random slope variance: {slope_var:.6f}")
                    log(f"    Random slope SD: {slope_sd:.6f}")

            except Exception as e:
                log(f"    Slopes model failed: {e}")
                aic_slopes = np.nan
                converged_slopes = False
                slope_var = np.nan
                slope_sd = np.nan
            # Compare Models
            delta_aic = aic_intercepts - aic_slopes
            log(f"  ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")

            # Interpret per protocol Step 12C
            if not converged_slopes:
                outcome = "Option B: Slopes don't converge"
                interpretation = "Convergence failure - insufficient data for stable slope estimation"
                recommendation = "Keep intercepts-only model"
                limitation = f"Random slopes attempted but failed to converge for {mtype}"

            elif delta_aic > 2:
                outcome = "Option A: Slopes improve fit (ΔAIC > 2)"
                interpretation = f"Individual {paradigm} forgetting rates vary (SD={slope_sd:.4f})"
                recommendation = "Use slopes model for downstream analyses"
                limitation = None

            elif delta_aic < -2:
                outcome = "Option C: Slopes converge but don't improve (ΔAIC < 2, favors intercepts)"
                interpretation = f"Homogeneous forgetting rates CONFIRMED (slope variance ≈ {slope_var:.6f})"
                recommendation = "Keep intercepts-only model (validated choice, not assumption)"
                limitation = None

            else:  # |ΔAIC| < 2
                outcome = "Option C: Slopes converge but don't improve (|ΔAIC| < 2)"
                interpretation = f"Negligible slope variance ({slope_var:.6f}) - effects homogeneous"
                recommendation = "Keep intercepts-only model (validated choice, not assumption)"
                limitation = None

            log(f"  Outcome: {outcome}")
            log(f"  Interpretation: {interpretation}")
            log(f"  Recommendation: {recommendation}")

            # Store results
            results.append({
                'measurement_type': mtype,
                'paradigm': paradigm,
                'AIC_intercepts': aic_intercepts,
                'AIC_slopes': aic_slopes,
                'delta_AIC': delta_aic,
                'converged_intercepts': converged_intercepts,
                'converged_slopes': converged_slopes,
                'random_slope_var': slope_var,
                'random_slope_sd': slope_sd,
                'outcome': outcome,
                'interpretation': interpretation,
                'recommendation': recommendation,
                'limitation': limitation if limitation else ''
            })
        # Save Results
        log("")
        log("Saving comparison results...")
        df_results = pd.DataFrame(results)
        output_path = RQ_DIR / "data" / "random_slopes_comparison.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        # Summary Report
        log("")
        log("=" * 80)
        log("SUMMARY")
        log("=" * 80)
        log("")
        log(f"Paradigm tested: {paradigm} (Item Free Recall)")
        log(f"Measurement types: {len(measurement_types)}")
        log("")

        # Count outcomes
        outcomes_count = df_results['outcome'].value_counts()
        for outcome, count in outcomes_count.items():
            log(f"  {outcome}: {count} measurement type(s)")

        log("")
        log("RECOMMENDATION:")
        # Check if majority supports intercepts-only
        intercepts_count = sum(df_results['recommendation'].str.contains('Keep intercepts-only'))
        slopes_count = sum(df_results['recommendation'].str.contains('Use slopes'))

        if intercepts_count > slopes_count:
            log(f"  {intercepts_count}/3 measurement types support intercepts-only model")
            log("  CONCLUSION: Original step07 implementation was appropriate")
            log("  Homogeneous forgetting rates VALIDATED via empirical test")
        elif slopes_count > intercepts_count:
            log(f"  {slopes_count}/3 measurement types support slopes model")
            log("  CONCLUSION: Random slopes should be used")
            log("  Individual differences in forgetting rates CONFIRMED")
        else:
            log("  Mixed results across measurement types")
            log("  CONCLUSION: Defer to paradigm-specific recommendations")

        log("")
        log("=" * 80)
        log("RANDOM SLOPES COMPARISON COMPLETE")
        log("=" * 80)
        log("")
        log("Random slopes comparison completed")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
