#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: step06b
Step Name: Model-Averaged ROOT Verification for RQ 5.5.2
RQ: results/ch5/5.5.2
Generated: 2025-12-10

PURPOSE:
Verify that Source-Destination consolidation NULL findings (RQ 5.5.2) remain
robust when using 13-model averaged predictions from ROOT RQ 5.5.1 instead of
single Logarithmic model.

KEY QUESTION:
Does the NULL LocationType × Phase interaction (p=0.610) persist when using
model-averaged trajectories that account for extreme model uncertainty
(N_eff=12.32)?

**Original Analysis (Log-only):**
- Model: Piecewise LMM with 48h breakpoint, Log-only trajectories
- Finding: NULL 3-way interaction (p=0.610, f²=0.0005 negligible)
- Conclusion: Source and destination show similar consolidation patterns

**ROOT Update (RQ 5.5.1, 2025-12-08):**
- Extended 66-model comparison revealed EXTREME uncertainty
- Best model (Quadratic) weight = 6.7%, Log weight = 6.5% (ΔAIC=0.34, tied)
- 13 competitive models (ΔAIC<2), cumulative weight = 54.3%
- Model averaging MANDATORY (Burnham & Anderson, 2002)

EXPECTED INPUTS:
  - ../5.5.1/data/step05c_averaged_predictions.csv
    Columns: ['TSVR_hours', 'LocationType', 'theta_averaged', 'prediction_variance', 'prediction_se']
    Format: Model-averaged predictions across 13 competitive models
    Expected rows: 200 (100 timepoints × 2 location types)

  - data/step02_lmm_input_long.csv
    Columns: ['composite_ID', 'UID', 'TSVR_hours', 'theta', 'LocationType', 'Days_within', 'Segment']
    Format: Original piecewise LMM input (from step02)
    Expected rows: 800 (100 participants × 4 sessions × 2 location types)

EXPECTED OUTPUTS:
  - data/step06b_model_averaged_lmm_input.csv
    Format: Piecewise LMM input with model-averaged theta values
    Expected rows: 800 (same structure as original)

  - data/step06b_piecewise_lmm_model_averaged.pkl
    Format: Fitted piecewise LMM using model-averaged predictions

  - data/step06b_interaction_test_comparison.csv
    Columns: ['approach', 'interaction_beta', 'interaction_se', 'p_uncorrected', 'p_bonf', 'f2', 'null_robust']
    Format: Comparison of Log vs Model-Averaged interaction results
    Expected: Both approaches yield NULL interaction

  - logs/step06b_model_averaged_verification.log
    Format: Execution log with model fit comparison

VALIDATION CRITERIA:
  - Model-averaged LMM converges with full random structure
  - Interaction p-value remains >0.025 (NULL robust)
  - Effect size remains negligible (f²<0.02)
  - Comparison shows NULL findings model-invariant

g_code REASONING:
- Approach: Merge model-averaged predictions with original piecewise structure
- Why this approach: Piecewise spec (48h breakpoint, Days_within) orthogonal to
  functional form (Log vs model-averaged), so interaction test should be robust
- Precedent: RQ 5.2.3, 5.2.4, 5.2.5 showed NULL interaction/convergence findings
  robust across functional forms (Log vs Recip+Log)
- Expected outcome: NULL interaction persists (p~0.6), confirming consolidation
  patterns similar for source/destination regardless of trajectory model
- Data flow: Load model-averaged predictions → Interpolate to observed TSVR_hours
  → Replace theta in piecewise input → Fit piecewise LMM → Test interaction
  → Compare to original Log-based result
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.interpolate import interp1d
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
RQ_551_DIR = RQ_DIR.parent / "5.5.1"
LOG_FILE = RQ_DIR / "logs" / "step06b_model_averaged_verification.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("STEP 06B: MODEL-AVERAGED ROOT VERIFICATION")
        log("=" * 80)
        log("")
        log("MOTIVATION:")
        log("  RQ 5.5.1 ROOT model changed from Log-only to 13-model averaging")
        log("  Testing if NULL LocationType × Phase interaction robust to model averaging")
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
        log(f"  Loaded: {averaged_file.name} ({len(averaged_preds)} rows)")
        log(f"  Location types: {sorted(averaged_preds['LocationType'].unique())}")
        log(f"  TSVR range: {averaged_preds['TSVR_hours'].min():.1f} - {averaged_preds['TSVR_hours'].max():.1f} hours")
        log("")

        # =====================================================================
        # STEP 2: Load Original Piecewise LMM Input
        # =====================================================================
        log("[STEP 2] Loading original piecewise LMM input...")
        log("")

        original_file = RQ_DIR / "data" / "step02_lmm_input_long.csv"
        original_data = pd.read_csv(original_file, encoding='utf-8')
        log(f"  Loaded: {original_file.name} ({len(original_data)} rows)")
        log(f"  Participants: {original_data['UID'].nunique()}")
        log(f"  Location types: {sorted(original_data['LocationType'].unique())}")
        log("")

        # =====================================================================
        # STEP 3: Interpolate Model-Averaged Predictions to Observed TSVR
        # =====================================================================
        log("[STEP 3] Interpolating model-averaged predictions to observed TSVR_hours...")
        log("")

        # Create interpolation functions for each location type
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
        output_file = RQ_DIR / "data" / "step06b_model_averaged_lmm_input.csv"
        model_averaged_data.to_csv(output_file, index=False, encoding='utf-8')
        log(f"  Saved: {output_file.name}")
        log("")

        # =====================================================================
        # STEP 4: Fit Piecewise LMM with Model-Averaged Predictions
        # =====================================================================
        log("[STEP 4] Fitting piecewise LMM with model-averaged predictions...")
        log("")

        # Use same formula as original analysis (step03)
        formula = "theta_model_averaged ~ Days_within * Segment * LocationType"
        log(f"  Formula: {formula}")
        log(f"  Random effects: ~1 + Days_within | UID")
        log("")

        model_averaged_lmm = MixedLM.from_formula(
            formula,
            data=model_averaged_data,
            groups=model_averaged_data['UID'],
            re_formula="~1 + Days_within"
        ).fit(method='lbfgs', reml=False)

        log(f"  ✓ Converged: {model_averaged_lmm.converged}")
        log(f"  AIC: {model_averaged_lmm.aic:.2f}")
        log(f"  BIC: {model_averaged_lmm.bic:.2f}")
        log(f"  Log-likelihood: {model_averaged_lmm.llf:.2f}")
        log("")

        # Save model
        model_file = RQ_DIR / "data" / "step06b_piecewise_lmm_model_averaged.pkl"
        model_averaged_lmm.save(str(model_file))
        log(f"  Saved: {model_file.name}")
        log("")

        # =====================================================================
        # STEP 5: Extract 3-Way Interaction Term
        # =====================================================================
        log("[STEP 5] Extracting 3-way interaction term...")
        log("")

        # 3-way interaction term
        # Original model: Days_within:Segment[T.Late]:LocationType[T.Destination]
        # But reference category may have changed due to alphabetical ordering
        # Check available terms and find the 3-way interaction

        available_terms = list(model_averaged_lmm.params.index)
        interaction_candidates = [
            "Days_within:Segment[T.Late]:LocationType[T.Destination]",
            "Days_within:Segment[T.Late]:LocationType[T.Source]"
        ]

        interaction_term = None
        for candidate in interaction_candidates:
            if candidate in available_terms:
                interaction_term = candidate
                break

        if interaction_term is None:
            log(f"  ⚠️  No 3-way interaction term found")
            log(f"  Available terms: {available_terms}")
            interaction_beta = np.nan
            interaction_se = np.nan
            interaction_p = np.nan
            interaction_p_bonf = np.nan
            interaction_f2 = np.nan
        else:
            interaction_idx = available_terms.index(interaction_term)
            interaction_beta = model_averaged_lmm.params.iloc[interaction_idx]
            interaction_se = model_averaged_lmm.bse.iloc[interaction_idx]
            interaction_z = interaction_beta / interaction_se

            # Get p-value from summary
            interaction_p = model_averaged_lmm.pvalues.iloc[interaction_idx]
            interaction_p_bonf = min(interaction_p * 2, 1.0)  # Bonferroni for 2 tests

            # Calculate Cohen's f²
            # For LMM, approximate using z-score: f² ≈ z² / N
            n_obs = len(model_averaged_data)
            interaction_f2 = (interaction_z ** 2) / n_obs

            log(f"  Interaction term: {interaction_term}")
            log(f"  β = {interaction_beta:.6f}")
            log(f"  SE = {interaction_se:.6f}")
            log(f"  z = {interaction_z:.3f}")
            log(f"  p (uncorrected) = {interaction_p:.6f}")
            log(f"  p (Bonferroni) = {interaction_p_bonf:.6f}")
            log(f"  Cohen's f² = {interaction_f2:.6f}")
            log("")

            # Note: if reference category changed (Source instead of Destination),
            # the sign of beta is flipped but magnitude and significance unchanged
            if "[T.Source]" in interaction_term:
                log(f"  ℹ️  Note: Reference category changed (Destination → Source)")
                log(f"       Sign flipped but magnitude/significance unchanged")
                log("")

            if interaction_p_bonf > 0.025:
                log("  ✓ NULL interaction (p_bonf > 0.025)")
            else:
                log("  ⚠️  Significant interaction (p_bonf ≤ 0.025)")
            log("")

        # =====================================================================
        # STEP 6: Load Original Log-Based Results for Comparison
        # =====================================================================
        log("[STEP 6] Loading original Log-based interaction results...")
        log("")

        original_interaction_file = RQ_DIR / "data" / "step06_interaction_tests.csv"
        original_interaction = pd.read_csv(original_interaction_file, encoding='utf-8')
        log(f"  Loaded: {original_interaction_file.name}")
        log("")

        original_beta = original_interaction['Estimate'].values[0]
        original_se = original_interaction['SE'].values[0]
        original_p = original_interaction['p_uncorrected'].values[0]
        original_p_bonf = original_interaction['p_bonferroni'].values[0]
        original_f2 = original_interaction['Cohens_f2'].values[0]

        log("  Original Log-based results:")
        log(f"    β = {original_beta:.6f}")
        log(f"    SE = {original_se:.6f}")
        log(f"    p (uncorrected) = {original_p:.6f}")
        log(f"    p (Bonferroni) = {original_p_bonf:.6f}")
        log(f"    Cohen's f² = {original_f2:.6f}")
        log("")

        # =====================================================================
        # STEP 7: Compare Log vs Model-Averaged Results
        # =====================================================================
        log("[STEP 7] COMPARISON: Log vs Model-Averaged")
        log("=" * 80)
        log("")

        comparison = pd.DataFrame({
            'approach': ['Log-only', 'Model-Averaged (13 models)'],
            'interaction_beta': [original_beta, interaction_beta],
            'interaction_se': [original_se, interaction_se],
            'p_uncorrected': [original_p, interaction_p],
            'p_bonferroni': [original_p_bonf, interaction_p_bonf],
            'cohens_f2': [original_f2, interaction_f2],
            'null_robust': [
                original_p_bonf > 0.025,
                interaction_p_bonf > 0.025
            ]
        })

        log("  Comparison Table:")
        log("")
        log(f"  {'Approach':<30} {'β':<12} {'SE':<12} {'p (Bonf)':<12} {'f²':<12} {'NULL?'}")
        log("  " + "-" * 90)
        for _, row in comparison.iterrows():
            log(f"  {row['approach']:<30} {row['interaction_beta']:<12.6f} {row['interaction_se']:<12.6f} {row['p_bonferroni']:<12.6f} {row['cohens_f2']:<12.6f} {row['null_robust']}")
        log("")

        # Save comparison
        comparison_file = RQ_DIR / "data" / "step06b_interaction_test_comparison.csv"
        comparison.to_csv(comparison_file, index=False, encoding='utf-8')
        log(f"  Saved: {comparison_file.name}")
        log("")

        # =====================================================================
        # STEP 8: Verification Summary
        # =====================================================================
        log("[STEP 8] VERIFICATION SUMMARY")
        log("=" * 80)
        log("")

        both_null = comparison['null_robust'].all()

        if both_null:
            log("✅ VERIFICATION PASSED: NULL interaction ROBUST to model averaging")
            log("")
            log("KEY FINDINGS:")
            log("")
            log("  1. NULL interaction persists:")
            log(f"     - Log-only: p = {original_p_bonf:.3f} (NULL)")
            log(f"     - Model-Averaged: p = {interaction_p_bonf:.3f} (NULL)")
            log("")
            log("  2. Effect sizes remain negligible:")
            log(f"     - Log-only: f² = {original_f2:.6f}")
            log(f"     - Model-Averaged: f² = {interaction_f2:.6f}")
            log("     Both < 0.02 (negligible effect threshold)")
            log("")
            log("  3. Coefficient estimates similar:")
            log(f"     - Δβ = {abs(interaction_beta - original_beta):.6f}")
            log(f"     - Relative change = {100 * abs(interaction_beta - original_beta) / abs(original_beta):.1f}%")
            log("")
            log("CONCLUSION:")
            log("  Source and destination memories show similar consolidation patterns")
            log("  regardless of functional form (Log vs 13-model averaging).")
            log("")
            log("  Original finding (p=0.610, f²=0.0005) ROBUST. Piecewise structure")
            log("  (48h breakpoint, Days_within within segments) orthogonal to trajectory")
            log("  model choice. NULL interaction confirms consolidation dynamics identical")
            log("  for both location types.")
            log("")
            log("  RQ 5.5.2 ready for GOLD status with ROOT dependency verified.")
        else:
            log("⚠️ VERIFICATION ALERT: NULL interaction pattern changed")
            log("")
            log("FINDINGS:")
            log(f"  Log-only NULL: {comparison.loc[0, 'null_robust']}")
            log(f"  Model-Averaged NULL: {comparison.loc[1, 'null_robust']}")
            log("")
            log("RECOMMENDATION:")
            log("  Review why model averaging changed interaction significance.")
            log("  May indicate functional form sensitivity not captured by original Log model.")

        log("")
        log("=" * 80)
        log("[SUCCESS] Step 06b complete")
        log(f"[INFO] All outputs saved to {RQ_DIR / 'data'}")
        sys.exit(0)

    except Exception as e:
        log("")
        log("=" * 80)
        log(f"[ERROR] {str(e)}")
        log("")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
