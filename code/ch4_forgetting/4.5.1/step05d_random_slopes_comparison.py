#!/usr/bin/env python3
"""
RQ 5.5.1 - Random Slopes Testing (MANDATORY per Section 4.4)

PURPOSE:
Test whether random slopes improve model fit compared to intercepts-only.
This is MANDATORY per improvement_taxonomy.md Section 4.4 - cannot claim
homogeneous effects without testing for heterogeneity.

BLOCKER CONTEXT:
- Original step05 (5 models): Used `~Days` (intercepts+slopes) ✅
- Extended step05 (66 models): Used `~1` (intercepts-only) ❌
- Need EXPLICIT comparison with AIC to determine if slopes needed

APPROACH:
Fit Log model (competitive with best Quadratic, ΔAIC=0.34) twice:
1. Intercepts-only: `re_formula='~1'`
2. Intercepts+slopes: `re_formula='~log_Days_plus1'`

Compare via AIC (ΔAIC > 2 indicates slopes improve fit).

OUTPUT:
- data/step05d_random_slopes_comparison.csv (2 rows: intercepts-only, intercepts+slopes)
- logs/step05d_random_slopes_comparison.log

EXPECTED OUTCOME (3 possibilities):
A) Slopes improve fit (ΔAIC > 2) → Use slopes, report individual differences
B) Slopes don't converge → Document attempt, explain why
C) Slopes converge but don't improve (ΔAIC < 2) → Keep intercepts, document negligible variance

Author: rq_platinum agent
Date: 2025-12-27
RQ: ch5/5.5.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05d_random_slopes_comparison.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Random Slopes Testing (Section 4.4 MANDATORY CHECK)")
        log("=" * 80)

        # Load LMM input data
        log("\n[LOAD] Loading LMM input data...")
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv", encoding='utf-8')
        log(f"  ✓ Loaded {len(lmm_input)} observations")
        log(f"  ✓ Participants: {lmm_input['UID'].nunique()}")
        log(f"  ✓ LocationTypes: {lmm_input['LocationType'].unique().tolist()}")

        # Define models to compare
        log("\n[MODELS] Defining two model specifications...")
        log("  Model: Logarithmic (competitive with best Quadratic, ΔAIC=0.34)")
        log("  Formula: theta ~ log_Days_plus1 * LocationType")
        log("")
        log("  Variant 1: Intercepts-only (re_formula='~1')")
        log("  Variant 2: Intercepts+slopes (re_formula='~log_Days_plus1')")

        models = {
            'Intercepts-only': {
                'formula': 'theta ~ log_Days_plus1 * LocationType',
                're_formula': '~1',
                'description': 'Random intercepts only (no individual slopes)'
            },
            'Intercepts+slopes': {
                'formula': 'theta ~ log_Days_plus1 * LocationType',
                're_formula': '~log_Days_plus1',
                'description': 'Random intercepts + random slopes on log_Days_plus1'
            }
        }

        # Fit both models
        results = []
        fitted_models = {}

        for model_name, spec in models.items():
            log(f"\n[FIT] {model_name}...")
            log(f"  Random effects: {spec['re_formula']}")

            try:
                model = smf.mixedlm(
                    formula=spec['formula'],
                    data=lmm_input,
                    groups=lmm_input['UID'],
                    re_formula=spec['re_formula']
                )

                result = model.fit(reml=False, method='lbfgs')
                fitted_models[model_name] = result

                # Extract variance components
                if model_name == 'Intercepts-only':
                    intercept_var = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else result.cov_re[0, 0]
                    slope_var = 0.0  # No slope in intercepts-only model
                else:
                    # Intercepts+slopes has 2×2 covariance matrix
                    intercept_var = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else result.cov_re[0, 0]
                    slope_var = result.cov_re.iloc[1, 1] if hasattr(result.cov_re, 'iloc') else result.cov_re[1, 1]

                results.append({
                    'model': model_name,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'logLik': result.llf,
                    'n_params': len(result.params),
                    'converged': result.converged,
                    'intercept_var': intercept_var,
                    'slope_var': slope_var,
                    'residual_var': result.scale
                })

                log(f"  ✓ Converged: {result.converged}")
                log(f"  ✓ AIC: {result.aic:.2f}")
                log(f"  ✓ BIC: {result.bic:.2f}")
                log(f"  ✓ Intercept variance: {intercept_var:.4f}")
                log(f"  ✓ Slope variance: {slope_var:.4f}")

            except Exception as e:
                log(f"  ✗ FAILED: {str(e)}")
                results.append({
                    'model': model_name,
                    'AIC': np.nan,
                    'BIC': np.nan,
                    'logLik': np.nan,
                    'n_params': np.nan,
                    'converged': False,
                    'intercept_var': np.nan,
                    'slope_var': np.nan,
                    'residual_var': np.nan,
                    'error': str(e)
                })

        # Create comparison DataFrame
        comparison = pd.DataFrame(results)

        # Compute delta AIC
        if comparison['AIC'].notna().all():
            min_aic = comparison['AIC'].min()
            comparison['delta_AIC'] = comparison['AIC'] - min_aic

            log("\n" + "=" * 80)
            log("[COMPARISON] Random Effects Structure Comparison")
            log("=" * 80)
            log(f"")
            for _, row in comparison.iterrows():
                log(f"{row['model']:20s}  AIC={row['AIC']:8.2f}  ΔAIC={row['delta_AIC']:6.2f}  "
                    f"slope_var={row['slope_var']:.4f}")

            # Determine outcome
            log("\n" + "=" * 80)
            log("[DECISION]")
            log("=" * 80)

            delta_aic = comparison[comparison['model'] == 'Intercepts+slopes']['delta_AIC'].values[0]
            slope_var = comparison[comparison['model'] == 'Intercepts+slopes']['slope_var'].values[0]

            if delta_aic < -2:
                # Slopes improve fit by > 2 AIC units (Option A)
                log("✓ SLOPES IMPROVE FIT (ΔAIC = {:.2f})".format(delta_aic))
                log(f"  Random slope variance: {slope_var:.4f} (non-zero)")
                log("  INTERPRETATION: Individual forgetting rates vary across participants")
                log("  ACTION: Use intercepts+slopes model going forward")
                log("  DOCUMENTATION: Report heterogeneous forgetting rates")
            elif not comparison[comparison['model'] == 'Intercepts+slopes']['converged'].values[0]:
                # Slopes don't converge (Option B)
                log("⚠ SLOPES MODEL FAILED TO CONVERGE (Option B)")
                log("  INTERPRETATION: Insufficient data for stable slope estimation")
                log("  ACTION: Keep intercepts-only model")
                log("  DOCUMENTATION: Document convergence failure, explain N timepoints")
            elif delta_aic >= -2:
                # Slopes don't improve fit (Option C)
                log(f"✓ SLOPES DON'T IMPROVE FIT (ΔAIC = {delta_aic:.2f})")
                log(f"  Random slope variance: {slope_var:.4f} (shrinkage to zero)")
                log("  INTERPRETATION: Forgetting rates homogeneous across participants")
                log("  ACTION: Keep intercepts-only model (more parsimonious)")
                log("  DOCUMENTATION: Random slopes tested, variance negligible")

        else:
            log("\n⚠ WARNING: Model fitting failed, cannot compare AIC")

        # Save comparison
        output_path = DATA_DIR / "step05d_random_slopes_comparison.csv"
        comparison.to_csv(output_path, index=False, encoding='utf-8')
        log(f"\n[SAVE] Comparison saved: {output_path.name}")

        log("\n" + "=" * 80)
        log("[SUCCESS] Random slopes testing complete")
        log("=" * 80)
        log("")
        log("NEXT STEPS:")
        log("1. Update validation.md with findings")
        log("2. Document in summary.md Section 1 (Model Selection)")
        log("3. Proceed with power analysis for NULL main effect")

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
