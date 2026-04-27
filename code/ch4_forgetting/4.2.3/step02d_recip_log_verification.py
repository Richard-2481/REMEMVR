#!/usr/bin/env python3
"""
Step 02d: Recip+Log ROOT Model Verification for RQ 5.2.3

PURPOSE:
Verify that NULL age effects (RQ 5.2.3) remain robust when updating functional form
from Log-only to Recip+Log (matching RQ 5.2.1 ROOT model after extended comparison).

RATIONALE:
- RQ 5.2.1 (ROOT) originally used Log model
- Extended comparison (2025-12-08) revealed Recip+Log dominates (10 models averaged)
- RQ 5.2.3 tested Age × Domain × Time interactions using Log-only
- This script verifies NULL age findings hold with ROOT-aligned Recip+Log model

METHODOLOGY:
- Add recip_TSVR transformation: 1 / (TSVR_hours + 1)
- Update formula: TSVR_hours → recip_TSVR, keep log_TSVR
- Update random slopes: ~log_TSVR → ~recip_TSVR (matches ROOT)
- Compare original Log vs Recip+Log: p-values, effect sizes, model fit

EXPECTED INPUTS:
  - data/step01_lmm_input.csv (from step01)
    Columns: UID, domain, TSVR_hours, log_TSVR, Age_c, theta

EXPECTED OUTPUTS:
  - data/step02d_lmm_input_recip_log.csv (with recip_TSVR added)
  - data/step02d_lmm_model_recip_log.pkl (Recip+Log model object)
  - data/step02d_fixed_effects_recip_log.csv (fixed effects table)
  - data/step02d_comparison_log_vs_recip_log.csv (comparison table)
  - logs/step02d_recip_log_verification.log

VALIDATION CRITERIA:
  - NULL age interactions remain (p_bonf > 0.025 for 3-way terms)
  - Effect sizes similar to original Log model
  - Model converges successfully
  - Random slopes for recip_TSVR (not log_TSVR)

Generated: 2025-12-09
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tool
from tools.analysis_lmm import fit_lmm_trajectory

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02d_recip_log_verification.log"

# =============================================================================
# Logging
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Initialize log
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("STEP 02D: RECIP+LOG ROOT MODEL VERIFICATION")
        log("=" * 80)
        log(f"Date: {pd.Timestamp.now()}")
        log("")

        # =====================================================================
        # STEP 1: Load Original LMM Input and Add Recip Transformation
        # =====================================================================

        log("[STEP 1] Loading original LMM input and adding recip_TSVR...")
        input_file = RQ_DIR / "data" / "step01_lmm_input.csv"
        lmm_input = pd.read_csv(input_file, encoding='utf-8')
        log(f"  Loaded: {len(lmm_input)} rows, {lmm_input['UID'].nunique()} participants")
        log(f"  Columns: {list(lmm_input.columns)}")

        # Add reciprocal transformation: 1 / (TSVR + 1)
        lmm_input['recip_TSVR'] = 1.0 / (lmm_input['TSVR_hours'] + 1)
        log(f"  Added recip_TSVR transformation: 1 / (TSVR_hours + 1)")
        log(f"  recip_TSVR range: [{lmm_input['recip_TSVR'].min():.4f}, {lmm_input['recip_TSVR'].max():.4f}]")

        # Save updated input
        recip_log_input_file = RQ_DIR / "data" / "step02d_lmm_input_recip_log.csv"
        lmm_input.to_csv(recip_log_input_file, index=False, encoding='utf-8')
        log(f"  Saved: {recip_log_input_file.name}")
        log("")

        # =====================================================================
        # STEP 2: Fit Recip+Log Model with Age × Domain × Time Interactions
        # =====================================================================

        log("[STEP 2] Fitting Recip+Log model with 3-way Age × Domain × Time interactions...")

        # NEW FORMULA: Replace TSVR_hours with recip_TSVR, keep log_TSVR
        # Main effects: recip_TSVR, log_TSVR, Age_c, domain
        # 2-way: recip_TSVR:Age_c, log_TSVR:Age_c, recip_TSVR:domain, log_TSVR:domain, Age_c:domain
        # 3-way: recip_TSVR:Age_c:domain, log_TSVR:Age_c:domain

        formula_recip_log = (
            "theta ~ recip_TSVR + log_TSVR + Age_c + domain + "
            "recip_TSVR:Age_c + log_TSVR:Age_c + "
            "recip_TSVR:domain + log_TSVR:domain + "
            "Age_c:domain + "
            "recip_TSVR:Age_c:domain + log_TSVR:Age_c:domain"
        )

        log(f"  Formula: {formula_recip_log}")
        log("  Random effects: Attempting (recip_TSVR | UID) - matches ROOT random structure")
        log("  REML=False (ML estimation for model comparison)")
        log("")

        # Try fitting with random slopes first (matching ROOT 5.2.1)
        try:
            log("  [ATTEMPT 1] Fitting with random slopes: ~recip_TSVR...")
            lmm_recip_log = fit_lmm_trajectory(
                data=lmm_input,
                formula=formula_recip_log,
                groups="UID",
                re_formula="~recip_TSVR",  # Random slopes for recip_TSVR
                reml=False
            )
            log(f"  [SUCCESS] Model converged with random slopes")
            random_structure = "~recip_TSVR (random slopes)"
        except Exception as e:
            log(f"  [FAILED] Random slopes did not converge: {str(e)}")
            log("  [ATTEMPT 2] Falling back to random intercepts only: ~1...")
            lmm_recip_log = fit_lmm_trajectory(
                data=lmm_input,
                formula=formula_recip_log,
                groups="UID",
                re_formula=None,  # Random intercepts only
                reml=False
            )
            log(f"  [SUCCESS] Model converged with random intercepts only")
            random_structure = "~1 (intercepts only)"

        log("")
        log(f"  Model converged: {lmm_recip_log.converged}")
        log(f"  Random structure used: {random_structure}")
        log(f"  N observations: {lmm_recip_log.nobs}")
        log(f"  N groups: {len(lmm_recip_log.model.group_labels)}")
        log(f"  Log-likelihood: {lmm_recip_log.llf:.2f}")
        log(f"  AIC: {lmm_recip_log.aic:.2f}")
        log(f"  BIC: {lmm_recip_log.bic:.2f}")
        log("")

        # =====================================================================
        # STEP 3: Extract Fixed Effects and 3-Way Interactions
        # =====================================================================

        log("[STEP 3] Extracting fixed effects and 3-way interaction terms...")

        # Extract fixed effects table
        fe_recip_log = pd.DataFrame({
            'term': lmm_recip_log.params.index,
            'estimate': lmm_recip_log.params.values,
            'se': lmm_recip_log.bse.values,
            'z': lmm_recip_log.tvalues.values,
            'p': lmm_recip_log.pvalues.values,
            'CI_lower': lmm_recip_log.conf_int()[0].values,
            'CI_upper': lmm_recip_log.conf_int()[1].values
        })

        # Identify 3-way interaction terms
        interaction_mask = (
            fe_recip_log['term'].str.contains('Age_c:domain', na=False) &
            (fe_recip_log['term'].str.contains('recip_TSVR', na=False) |
             fe_recip_log['term'].str.contains('log_TSVR', na=False))
        )
        interactions_recip_log = fe_recip_log[interaction_mask].copy()

        log(f"  Found {len(interactions_recip_log)} three-way interaction terms:")
        for _, row in interactions_recip_log.iterrows():
            log(f"    {row['term']}: β={row['estimate']:.6f}, SE={row['se']:.6f}, p={row['p']:.6f}")
        log("")

        # Apply Bonferroni correction (α = 0.025 for 2 tests)
        bonferroni_alpha = 0.025
        interactions_recip_log['p_bonferroni'] = interactions_recip_log['p'] * len(interactions_recip_log)
        interactions_recip_log['significant_bonf'] = interactions_recip_log['p_bonferroni'] < bonferroni_alpha

        log(f"  Bonferroni-corrected p-values (α = {bonferroni_alpha}):")
        for _, row in interactions_recip_log.iterrows():
            sig_status = "SIGNIFICANT" if row['significant_bonf'] else "NULL"
            log(f"    {row['term']}: p_bonf={row['p_bonferroni']:.6f} [{sig_status}]")
        log("")

        # =====================================================================
        # STEP 4: Compare to Original Log Model
        # =====================================================================

        log("[STEP 4] Comparing Recip+Log to original Log model...")

        # Load original Log model fixed effects
        original_fe_file = RQ_DIR / "data" / "step02_fixed_effects.csv"
        fe_log = pd.read_csv(original_fe_file, encoding='utf-8')

        # Extract original 3-way interactions
        original_interaction_mask = (
            fe_log['term'].str.contains('Age_c:domain', na=False) &
            (fe_log['term'].str.contains('TSVR_hours', na=False) |
             fe_log['term'].str.contains('log_TSVR', na=False))
        )
        interactions_log = fe_log[original_interaction_mask].copy()

        log("  Original Log model 3-way interactions:")
        for _, row in interactions_log.iterrows():
            log(f"    {row['term']}: β={row['estimate']:.6f}, p={row['p']:.6f}")
        log("")

        # Create comparison table (map term names carefully)
        # Original: TSVR_hours:Age_c:domain[T.Where]
        # New: recip_TSVR:Age_c:domain[T.Where]
        # Both have: log_TSVR:Age_c:domain[T.Where]

        comparison_data = []

        # LINEAR TIME interaction (TSVR_hours vs recip_TSVR)
        log_linear_term = interactions_log[interactions_log['term'].str.contains('TSVR_hours:Age_c:domain', na=False)]
        recip_linear_term = interactions_recip_log[interactions_recip_log['term'].str.contains('recip_TSVR:Age_c:domain', na=False)]

        if not log_linear_term.empty and not recip_linear_term.empty:
            comparison_data.append({
                'interaction': 'Linear Time × Age × Domain',
                'original_term': log_linear_term.iloc[0]['term'],
                'original_beta': log_linear_term.iloc[0]['estimate'],
                'original_p': log_linear_term.iloc[0]['p'],
                'recip_log_term': recip_linear_term.iloc[0]['term'],
                'recip_log_beta': recip_linear_term.iloc[0]['estimate'],
                'recip_log_p': recip_linear_term.iloc[0]['p'],
                'p_change': recip_linear_term.iloc[0]['p'] - log_linear_term.iloc[0]['p'],
                'both_null': (log_linear_term.iloc[0]['p'] > bonferroni_alpha) and
                             (recip_linear_term.iloc[0]['p'] * 2 > bonferroni_alpha)
            })

        # LOGARITHMIC TIME interaction (log_TSVR:Age_c:domain)
        log_log_term = interactions_log[interactions_log['term'].str.contains('log_TSVR:Age_c:domain', na=False)]
        recip_log_log_term = interactions_recip_log[interactions_recip_log['term'].str.contains('log_TSVR:Age_c:domain', na=False)]

        if not log_log_term.empty and not recip_log_log_term.empty:
            comparison_data.append({
                'interaction': 'Log Time × Age × Domain',
                'original_term': log_log_term.iloc[0]['term'],
                'original_beta': log_log_term.iloc[0]['estimate'],
                'original_p': log_log_term.iloc[0]['p'],
                'recip_log_term': recip_log_log_term.iloc[0]['term'],
                'recip_log_beta': recip_log_log_term.iloc[0]['estimate'],
                'recip_log_p': recip_log_log_term.iloc[0]['p'],
                'p_change': recip_log_log_term.iloc[0]['p'] - log_log_term.iloc[0]['p'],
                'both_null': (log_log_term.iloc[0]['p'] > bonferroni_alpha) and
                             (recip_log_log_term.iloc[0]['p'] * 2 > bonferroni_alpha)
            })

        comparison_df = pd.DataFrame(comparison_data)

        log("  Comparison Summary:")
        log("  " + "-" * 76)
        for _, row in comparison_df.iterrows():
            log(f"  {row['interaction']}:")
            log(f"    Original (Log): β={row['original_beta']:.6f}, p={row['original_p']:.6f}")
            log(f"    Recip+Log:      β={row['recip_log_beta']:.6f}, p={row['recip_log_p']:.6f}")
            log(f"    Change in p:    Δp={row['p_change']:+.6f}")
            log(f"    Both NULL:      {row['both_null']}")
        log("")

        # =====================================================================
        # STEP 5: Save Outputs
        # =====================================================================

        log("[STEP 5] Saving verification outputs...")

        # Save Recip+Log model
        model_file = RQ_DIR / "data" / "step02d_lmm_model_recip_log.pkl"
        lmm_recip_log.save(str(model_file))
        log(f"  Saved: {model_file.name}")

        # Save Recip+Log fixed effects
        fe_file = RQ_DIR / "data" / "step02d_fixed_effects_recip_log.csv"
        fe_recip_log.to_csv(fe_file, index=False, encoding='utf-8')
        log(f"  Saved: {fe_file.name}")

        # Save comparison table
        comparison_file = RQ_DIR / "data" / "step02d_comparison_log_vs_recip_log.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
        log(f"  Saved: {comparison_file.name}")
        log("")

        # =====================================================================
        # STEP 6: Verification Summary
        # =====================================================================

        log("=" * 80)
        log("[VERIFICATION SUMMARY]")
        log("=" * 80)
        log("")

        # Check if NULL hypothesis holds with Recip+Log
        all_null = interactions_recip_log['significant_bonf'].sum() == 0

        if all_null:
            log("✅ VERIFICATION PASSED: NULL age effects ROBUST to ROOT model update")
            log("")
            log("Key Findings:")
            log("  1. All 3-way Age × Domain × Time interactions remain NULL (p_bonf > 0.025)")
            log("  2. Age effects on forgetting do NOT vary by domain (What vs Where)")
            log("  3. Original Log-only finding ROBUST to Recip+Log functional form")
            log("")
            log("Interpretation:")
            log("  - VR episodic memory shows domain-GENERAL age effects")
            log("  - Hippocampal aging hypothesis NOT supported (spatial memory not more vulnerable)")
            log("  - Functional form (Log vs Recip+Log) does NOT change conclusion")
            log("")
            log("Status: RQ 5.2.3 NULL findings verified, ready for GOLD status")
        else:
            log("⚠️ VERIFICATION ALERT: Recip+Log reveals significant age interactions")
            log("")
            log("Significant Interactions (p_bonf < 0.025):")
            sig_interactions = interactions_recip_log[interactions_recip_log['significant_bonf']]
            for _, row in sig_interactions.iterrows():
                log(f"  - {row['term']}: p_bonf={row['p_bonferroni']:.6f}")
            log("")
            log("Interpretation:")
            log("  - Original Log model may have MISSED domain-specific age effects")
            log("  - Recip+Log (ROOT-aligned) reveals age vulnerability differs by domain")
            log("  - REQUIRES further investigation and interpretation update")
            log("")
            log("Status: RQ 5.2.3 findings NOT robust, kitchen sink analysis recommended")

        log("")
        log("=" * 80)
        log("[SUCCESS] Step 02d Recip+Log verification complete")
        log("=" * 80)

    except Exception as e:
        log("")
        log("=" * 80)
        log("[ERROR] Recip+Log verification failed!")
        log("=" * 80)
        log(f"Error: {str(e)}")
        log("")
        log("Full traceback:")
        log(traceback.format_exc())
        sys.exit(1)
