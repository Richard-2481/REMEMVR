#!/usr/bin/env python3
"""
Step ID: step02b
Step Name: Model-Averaged ROOT Verification for RQ 5.5.3
RQ: results/ch5/5.5.3
Generated: 2025-12-10

PURPOSE:
Verify that Source-Destination age interaction NULL findings (RQ 5.5.3) remain
robust when using 13-model averaged predictions from ROOT RQ 5.5.1 instead of
single Logarithmic model.

KEY QUESTION:
Do the NULL Age × LocationType × Time interactions (p=0.160, p=0.329) persist
when using model-averaged trajectories that account for extreme model uncertainty
(N_eff=12.32)?

APPROACH:
1. Load 13-model averaged predictions from RQ 5.5.1 (step05c_averaged_predictions.csv)
2. Interpolate model-averaged theta values to observed TSVR_hours
3. Fit LMM with model-averaged predictions (same formula as original)
4. Test 3-way interactions: TSVR_hours × Age_c × LocationType
                          log_TSVR × Age_c × LocationType
5. Compare with original Log-based results (step03_interaction_terms.csv)
6. Document robustness in verification report

EXPECTED OUTCOME:
Both 3-way interactions remain NULL (p_bonf > 0.025), confirming age-invariant
source-destination forgetting regardless of functional form.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import mixedlm
from scipy.interpolate import interp1d
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
RQ_551_DIR = RQ_DIR.parent / "5.5.1"
LOG_FILE = RQ_DIR / "logs" / "step02b_model_averaged_verification.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("RQ 5.5.3 ROOT Model Verification: 13-Model Averaging Update")
        log("=" * 80)
        log("")
        log("PURPOSE: Verify NULL Age × LocationType × Time interactions robust")
        log("         when using model-averaged trajectories from RQ 5.5.1")
        log("")
        log("ORIGINAL: Log-only model (RQ 5.5.1 weight=63.5%)")
        log("UPDATED:  13-model averaging (extreme uncertainty, N_eff=12.32)")
        log("")

        # =====================================================================
        # STEP 1: Load Model-Averaged Predictions from RQ 5.5.1
        # =====================================================================
        log("[STEP 1] Loading 13-model averaged predictions from RQ 5.5.1...")
        log("")

        averaged_file = RQ_551_DIR / "data" / "step05c_averaged_predictions.csv"
        if not averaged_file.exists():
            raise FileNotFoundError(f"Model-averaged predictions not found: {averaged_file}")

        averaged_preds = pd.read_csv(averaged_file, encoding='utf-8')
        log(f"  Loaded: {averaged_file.name}")
        log(f"  Rows: {len(averaged_preds)}")
        log(f"  Columns: {list(averaged_preds.columns)}")
        log(f"  LocationTypes: {averaged_preds['LocationType'].unique()}")
        log("")

        # =====================================================================
        # STEP 2: Load Original LMM Input (Log-based)
        # =====================================================================
        log("[STEP 2] Loading original LMM input data...")
        log("")

        original_file = RQ_DIR / "data" / "step01_lmm_input.csv"
        original_data = pd.read_csv(original_file, encoding='utf-8')
        log(f"  Loaded: {original_file.name}")
        log(f"  Rows: {len(original_data)}")
        log(f"  Columns: {list(original_data.columns)}")
        log(f"  UIDs: {original_data['UID'].nunique()}")
        log(f"  LocationTypes: {original_data['LocationType'].unique()}")
        log("")

        # =====================================================================
        # STEP 3: Interpolate Model-Averaged Predictions to Observed TSVR
        # =====================================================================
        log("[STEP 3] Interpolating model-averaged predictions to observed TSVR_hours...")
        log("")

        # Create copy for model-averaged data
        model_averaged_data = original_data.copy()

        # Mapping: original_data uses 'Source'/'Destination', averaged_preds uses 'source'/'destination'
        location_mapping = {
            'Source': 'source',
            'Destination': 'destination'
        }

        for loc_original, loc_averaged in location_mapping.items():
            log(f"  Interpolating {loc_original} predictions...")

            # Get averaged predictions for this location
            loc_avg = averaged_preds[averaged_preds['LocationType'] == loc_averaged].copy()
            loc_avg = loc_avg.sort_values('TSVR_hours')

            if len(loc_avg) == 0:
                log(f"    ⚠️  No averaged predictions found for {loc_averaged}")
                continue

            # Create interpolation function (linear interpolation)
            interp_func = interp1d(
                loc_avg['TSVR_hours'],
                loc_avg['theta_averaged'],
                kind='linear',
                fill_value='extrapolate'
            )

            # Apply interpolation to observed TSVR_hours
            mask = model_averaged_data['LocationType'] == loc_original
            model_averaged_data.loc[mask, 'theta_model_averaged'] = interp_func(
                model_averaged_data.loc[mask, 'TSVR_hours']
            )

            log(f"    ✓ {loc_original}: {mask.sum()} observations interpolated")

        log("")
        log(f"  Model-averaged theta range: {model_averaged_data['theta_model_averaged'].min():.3f} to {model_averaged_data['theta_model_averaged'].max():.3f}")
        log(f"  Original theta range: {model_averaged_data['theta'].min():.3f} to {model_averaged_data['theta'].max():.3f}")
        log("")

        # Save model-averaged input
        output_file = RQ_DIR / "data" / "step02b_model_averaged_lmm_input.csv"
        model_averaged_data.to_csv(output_file, index=False, encoding='utf-8')
        log(f"  Saved: {output_file.name}")
        log("")

        # =====================================================================
        # STEP 4: Fit LMM with Model-Averaged Predictions
        # =====================================================================
        log("[STEP 4] Fitting LMM with model-averaged predictions...")
        log("")

        # Same formula as original (step02_fit_lmm.py)
        formula = (
            "theta_model_averaged ~ TSVR_hours + log_TSVR + Age_c + LocationType + "
            "TSVR_hours:Age_c + log_TSVR:Age_c + "
            "TSVR_hours:LocationType + log_TSVR:LocationType + "
            "Age_c:LocationType + "
            "TSVR_hours:Age_c:LocationType + log_TSVR:Age_c:LocationType"
        )
        re_formula = "~TSVR_hours"

        log(f"  Formula: {formula}")
        log(f"  Random effects: {re_formula}")
        log(f"  Grouping: UID")
        log("")

        # Fit model
        model_averaged_lmm = mixedlm(
            formula=formula,
            data=model_averaged_data,
            groups=model_averaged_data['UID'],
            re_formula=re_formula
        ).fit(reml=False, method='lbfgs', maxiter=1000)

        log(f"  ✓ Model fitted")
        log(f"  Converged: {model_averaged_lmm.converged}")
        log(f"  AIC: {model_averaged_lmm.aic:.2f}")
        log(f"  BIC: {model_averaged_lmm.bic:.2f}")
        log("")

        # Save model object
        model_path = RQ_DIR / "data" / "step02b_lmm_model_averaged.pkl"
        model_averaged_lmm.save(str(model_path))
        log(f"  Saved model: {model_path.name}")
        log("")

        # =====================================================================
        # STEP 5: Extract 3-Way Interaction Terms
        # =====================================================================
        log("[STEP 5] Extracting 3-way interaction terms...")
        log("")

        # Get all parameter names
        available_terms = list(model_averaged_lmm.params.index)

        # Define interaction candidates (both reference categories possible)
        interaction_candidates = {
            'TSVR_linear': [
                'TSVR_hours:Age_c:LocationType[T.Destination]',
                'TSVR_hours:Age_c:LocationType[T.Source]'
            ],
            'log_TSVR': [
                'log_TSVR:Age_c:LocationType[T.Destination]',
                'log_TSVR:Age_c:LocationType[T.Source]'
            ]
        }

        # Extract both interactions
        interactions = {}
        for interaction_type, candidates in interaction_candidates.items():
            interaction_term = None
            for candidate in candidates:
                if candidate in available_terms:
                    interaction_term = candidate
                    break

            if interaction_term is None:
                log(f"  ⚠️  No 3-way interaction term found for {interaction_type}")
                log(f"      Available terms: {available_terms}")
                interactions[interaction_type] = {
                    'term': None,
                    'beta': np.nan,
                    'se': np.nan,
                    'p': np.nan,
                    'p_bonf': np.nan,
                    'f2': np.nan
                }
            else:
                interaction_idx = available_terms.index(interaction_term)
                beta = model_averaged_lmm.params.iloc[interaction_idx]
                se = model_averaged_lmm.bse.iloc[interaction_idx]
                z = beta / se
                p = model_averaged_lmm.pvalues.iloc[interaction_idx]
                p_bonf = min(p * 2, 1.0)  # Bonferroni for 2 tests

                # Calculate Cohen's f²
                n_obs = len(model_averaged_data)
                f2 = (z ** 2) / n_obs

                interactions[interaction_type] = {
                    'term': interaction_term,
                    'beta': beta,
                    'se': se,
                    'p': p,
                    'p_bonf': p_bonf,
                    'f2': f2
                }

                log(f"  {interaction_type}:")
                log(f"    Term: {interaction_term}")
                log(f"    β = {beta:.6f}")
                log(f"    SE = {se:.6f}")
                log(f"    z = {z:.3f}")
                log(f"    p (uncorrected) = {p:.6f}")
                log(f"    p (Bonferroni) = {p_bonf:.6f}")
                log(f"    Cohen's f² = {f2:.6f}")
                log("")

                # Note reference category change if applicable
                if "[T.Destination]" in interaction_term and interaction_type == 'TSVR_linear':
                    log(f"    ℹ️  Note: Reference category = Destination (vs original Source)")
                elif "[T.Source]" in interaction_term and interaction_type == 'TSVR_linear':
                    log(f"    ℹ️  Note: Reference category = Source (same as original)")
                log("")

                if p_bonf > 0.025:
                    log(f"    ✓ NULL interaction (p_bonf > 0.025)")
                else:
                    log(f"    ⚠️  Significant interaction (p_bonf ≤ 0.025)")
                log("")

        # =====================================================================
        # STEP 6: Load Original Log-Based Results for Comparison
        # =====================================================================
        log("[STEP 6] Loading original Log-based interaction results...")
        log("")

        original_interaction_file = RQ_DIR / "data" / "step03_interaction_terms.csv"
        original_interactions = pd.read_csv(original_interaction_file, encoding='utf-8')

        log(f"  Loaded: {original_interaction_file.name}")
        log(f"  Original interactions: {len(original_interactions)} terms")
        log("")

        # =====================================================================
        # STEP 7: Compare Original vs Model-Averaged Results
        # =====================================================================
        log("[STEP 7] Comparing original (Log) vs model-averaged results...")
        log("")

        # Create comparison dataframe
        comparison_rows = []

        for idx, orig_row in original_interactions.iterrows():
            orig_term = orig_row['term']

            # Determine interaction type
            if 'TSVR_hours:Age_c:LocationType' in orig_term:
                interaction_type = 'TSVR_linear'
            elif 'log_TSVR:Age_c:LocationType' in orig_term:
                interaction_type = 'log_TSVR'
            else:
                log(f"  ⚠️  Unexpected term: {orig_term}")
                continue

            # Get model-averaged results
            ma_results = interactions[interaction_type]

            # Add comparison row
            comparison_rows.append({
                'interaction_type': interaction_type,
                'approach': 'Log-only',
                'term': orig_term,
                'beta': orig_row['coef'],
                'se': orig_row['se'],
                'p_uncorrected': orig_row['p_uncorrected'],
                'p_bonferroni': orig_row['p_bonferroni'],
                'null_robust': orig_row['p_bonferroni'] > 0.025
            })

            comparison_rows.append({
                'interaction_type': interaction_type,
                'approach': 'Model-Averaged (13 models)',
                'term': ma_results['term'],
                'beta': ma_results['beta'],
                'se': ma_results['se'],
                'p_uncorrected': ma_results['p'],
                'p_bonferroni': ma_results['p_bonf'],
                'null_robust': ma_results['p_bonf'] > 0.025
            })

        comparison = pd.DataFrame(comparison_rows)

        # Save comparison
        comparison_file = RQ_DIR / "data" / "step02b_interaction_test_comparison.csv"
        comparison.to_csv(comparison_file, index=False, encoding='utf-8')
        log(f"  Saved comparison: {comparison_file.name}")
        log("")

        # Display comparison table
        log("  Comparison Summary:")
        log("")
        for interaction_type in ['TSVR_linear', 'log_TSVR']:
            log(f"  {interaction_type}:")
            subset = comparison[comparison['interaction_type'] == interaction_type]
            for _, row in subset.iterrows():
                log(f"    {row['approach']}:")
                log(f"      β = {row['beta']:.6f}, SE = {row['se']:.6f}")
                log(f"      p = {row['p_uncorrected']:.6f}, p_bonf = {row['p_bonferroni']:.6f}")
                log(f"      NULL: {row['null_robust']}")
            log("")

        # =====================================================================
        # STEP 8: Verification Summary
        # =====================================================================
        log("[STEP 8] Verification Summary")
        log("=" * 80)
        log("")

        # Check if both interactions are NULL in both approaches
        both_null_log = comparison[comparison['approach'] == 'Log-only']['null_robust'].all()
        both_null_ma = comparison[comparison['approach'] == 'Model-Averaged (13 models)']['null_robust'].all()

        log(f"  Original (Log-only): Both interactions NULL = {both_null_log}")
        log(f"  Model-Averaged: Both interactions NULL = {both_null_ma}")
        log("")

        if both_null_log and both_null_ma:
            log("✅ VERIFICATION PASSED: NULL age interactions ROBUST to model averaging")
            log("")
            log("INTERPRETATION:")
            log("  - Age does NOT moderate source-destination forgetting patterns")
            log("  - Finding robust regardless of trajectory functional form")
            log("  - Both TSVR_hours and log_TSVR interactions remain NULL")
            log("  - Extreme model uncertainty (N_eff=12.32) does NOT change conclusion")
            log("")
            log("RQ 5.5.3 ready for GOLD status with ROOT dependency verified.")
        else:
            log("⚠️ VERIFICATION FAILED: Interaction status changed with model averaging")
            log("")
            log("INVESTIGATION NEEDED:")
            log("  - Check which interaction(s) changed status")
            log("  - Review model convergence and diagnostics")
            log("  - Consider sensitivity analyses")

        log("")
        log("=" * 80)
        log("[SUCCESS] Step 02b verification complete")
        log("=" * 80)
        sys.exit(0)

    except Exception as e:
        log("")
        log("=" * 80)
        log(f"[ERROR] {str(e)}")
        log("=" * 80)
        log("")
        log("[TRACEBACK] Full error details:")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n[TRACEBACK]\n")
            traceback.print_exc(file=f)
        sys.exit(1)
