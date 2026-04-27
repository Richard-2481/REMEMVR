"""
RQ 5.4.2 - Step 02b: Fit Extended LMM Model Suite (Kitchen Sink)

PURPOSE:
Replace piecewise analysis with comprehensive continuous-time trajectory models.
Sensitivity analysis (summary.md) showed Lin+Log fits MUCH better than piecewise
(ΔAIC = -91). This step tests 70+ functional forms with Congruence × Time
interaction to find best-fitting forgetting trajectory.

RATIONALE:
Original piecewise approach assumed discrete consolidation (Days 0-1) vs decay
(Days 1-6) phases. Data rejected this assumption (continuous models far superior).
Kitchen sink approach lets data determine best functional form without theoretical
assumptions about discrete phase transitions.

INPUT:
- data/step01_lmm_input_piecewise.csv (1200 rows: UID, theta, TSVR_hours, Congruence)
  NOTE: Using this file for continuity, but IGNORING Segment/Days_within columns

OUTPUT:
- data/step02b_model_comparison.csv (70+ rows, AIC-sorted model comparison)
- data/step02b_best_model_summary.txt (best model details)
- logs/step02b_kitchen_sink.log (detailed execution log)

CRITICAL:
This replaces step02 (piecewise LMM) as primary analysis. Tests whether schema
congruence moderates OVERALL forgetting trajectory, not segment-specific slopes.

CONTEXT:
- Original step02 (piecewise): AIC = 2581.55
- Sensitivity test (Lin+Log): AIC = 2490.91 (ΔAIC = -91, overwhelming)
- RQ 5.4.1 (ROOT): Extended comparison found Recip+Log competitive (15 models ΔAIC<2)
- This step: Test if Recip+Log also best for consolidation question

Author: Claude Code
Date: 2025-12-09
RQ: ch5/5.4.2
Step: 02b
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import kitchen_sink model selection tool
from tools.model_selection import compare_lmm_models_kitchen_sink

# =============================================================================
# CONFIGURATION
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.4.2
LOG_FILE = RQ_DIR / "logs" / "step02b_kitchen_sink.log"
DATA_DIR = RQ_DIR / "data"

# =============================================================================
# LOGGING
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 02b: Kitchen Sink LMM Model Comparison")
        log("  Replacing piecewise analysis with continuous-time trajectory models")
        log("=" * 80)

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================

        log("[LOAD] Loading LMM input data...")
        input_path = DATA_DIR / "step01_lmm_input_piecewise.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"LMM input missing: {input_path}\n"
                "Run step01_prepare_piecewise_input.py first"
            )

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"  ✓ Loaded {input_path.name}")
        log(f"    Rows: {len(lmm_input)}")
        log(f"    Columns: {lmm_input.columns.tolist()}")

        # Verify required columns
        required_cols = ['UID', 'theta', 'TSVR_hours', 'Congruence']
        missing = [col for col in required_cols if col not in lmm_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        log(f"  ✓ TSVR unique values: {lmm_input['TSVR_hours'].nunique()}")
        log(f"  ✓ TSVR range: [{lmm_input['TSVR_hours'].min():.2f}, "
            f"{lmm_input['TSVR_hours'].max():.2f}] hours")
        log(f"  ✓ Theta range: [{lmm_input['theta'].min():.3f}, "
            f"{lmm_input['theta'].max():.3f}]")
        log(f"  ✓ Participants: {lmm_input['UID'].nunique()}")
        log(f"  ✓ Congruence levels: {sorted(lmm_input['Congruence'].unique())}")
        log(f"  ✓ Congruence counts: {lmm_input['Congruence'].value_counts().to_dict()}")

        # NOTE: Ignoring Segment and Days_within columns (piecewise variables)
        log("")
        log("  NOTE: Using continuous TSVR_hours, ignoring piecewise Segment/Days_within")
        log("        Rationale: Continuous models fit MUCH better (ΔAIC = -91)")

        # =====================================================================
        # STEP 2: Run Kitchen Sink Model Comparison
        # =====================================================================

        log("")
        log("[ANALYSIS] Running kitchen sink model comparison...")
        log("  Model suite: 70+ time transformations")
        log("  Outcome: theta (IRT ability estimates)")
        log("  Time variable: TSVR_hours (continuous)")
        log("  Factor 1: Congruence (Common, Congruent, Incongruent)")
        log("  Interaction: Congruence × time_transform")
        log("  Groups: UID (participants)")
        log("  Random effects: ~1 (intercepts only)")
        log("  Estimation: ML (reml=False, for AIC comparison)")
        log("")

        results = compare_lmm_models_kitchen_sink(
            data=lmm_input,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',

            # Congruence × Time interaction
            factor1_var='Congruence',
            factor1_reference='Common',  # Treatment coding: Common as reference
            factor2_var=None,

            # Random intercepts only (for model comparison stability)
            re_formula='~1',

            # ML for AIC comparison
            reml=False,

            # Save outputs
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
        )

        log("[DONE] Kitchen sink comparison complete")

        # =====================================================================
        # STEP 3: Extract and Report Results
        # =====================================================================

        log("")
        log("[RESULTS] Model comparison summary:")
        log("=" * 80)

        comparison = results['comparison']
        best_model_info = results['best_model']
        best_model_name = best_model_info['name']

        # Add rank column (1-indexed)
        comparison['rank'] = range(1, len(comparison) + 1)

        # Top 10 models
        log(f"TOP 10 MODELS (out of {len(comparison)}):")
        log("-" * 80)
        for i, row in comparison.head(10).iterrows():
            log(f"  {row['rank']:2d}. {row['model_name']:25s} "
                f"AIC={row['AIC']:7.2f}  ΔAIC={row['delta_AIC']:6.2f}  "
                f"weight={row['akaike_weight']*100:5.2f}%")

        log("=" * 80)
        log(f"BEST MODEL: {best_model_name}")
        log(f"  AIC: {best_model_info['AIC']:.2f}")
        log(f"  Weight: {best_model_info['weight']*100:.2f}%")
        # Note: best_model_info may not have all keys
        if 'loglik' in best_model_info:
            log(f"  Log-Likelihood: {best_model_info['loglik']:.2f}")
        if 'npar' in best_model_info:
            log(f"  Parameters: {best_model_info['npar']}")
        log("=" * 80)

        # Comparison with piecewise baseline
        log("")
        log("COMPARISON WITH PIECEWISE BASELINE:")
        log(f"  Piecewise (step02): AIC = 2581.55")
        log(f"  Best continuous:    AIC = {best_model_info['AIC']:.2f}")
        log(f"  ΔAIC = {best_model_info['AIC'] - 2581.55:.2f}")
        if best_model_info['AIC'] < 2581.55:
            log(f"  ✓ Continuous model SUPERIOR (negative ΔAIC)")
        else:
            log(f"  ✗ Piecewise still better (positive ΔAIC)")
        log("")

        # Check if model averaging needed (weight < 30%)
        best_weight = comparison.iloc[0]['akaike_weight']
        if best_weight < 0.30:
            log("")
            log("⚠️  WARNING: Best model weight < 30%")
            log(f"   Model uncertainty present (weight={best_weight*100:.1f}%)")
            log("   RECOMMENDATION: Run step02c_model_averaging.py")
            log("   See: Burnham & Anderson (2002) Model Selection section")
            log("")
        elif best_weight < 0.70:
            log("")
            log("ℹ️  NOTE: Best model weight 30-70%")
            log(f"   Substantial evidence for best model (weight={best_weight*100:.1f}%)")
            log("   Model averaging optional but recommended for robustness")
            log("")

        # =====================================================================
        # STEP 4: Save Model Comparison and Summary
        # =====================================================================

        # Rename output to step02b prefix
        comparison.to_csv(DATA_DIR / "step02b_model_comparison.csv", index=False)
        log(f"[SAVE] Model comparison saved: step02b_model_comparison.csv")

        summary_path = DATA_DIR / "step02b_best_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RQ 5.4.2 - Kitchen Sink Model Selection Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write("CONTEXT:\n")
            f.write("  Original piecewise analysis: AIC = 2581.55\n")
            f.write("  Sensitivity test (Lin+Log):  AIC = 2490.91 (ΔAIC = -91)\n")
            f.write("  This step: Comprehensive kitchen sink comparison\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"AIC: {best_model_info['AIC']:.2f}\n")
            f.write(f"Weight: {best_model_info['weight']*100:.2f}%\n")
            if 'loglik' in best_model_info:
                f.write(f"Log-Likelihood: {best_model_info['loglik']:.2f}\n")
            if 'npar' in best_model_info:
                f.write(f"Parameters: {best_model_info['npar']}\n")
            f.write("\n")

            f.write("Top 10 Models:\n")
            f.write("-" * 80 + "\n")
            for i, row in comparison.head(10).iterrows():
                f.write(f"{row['rank']:2d}. {row['model_name']:25s} "
                       f"AIC={row['AIC']:7.2f}  ΔAIC={row['delta_AIC']:6.2f}  "
                       f"weight={row['akaike_weight']*100:5.2f}%\n")

            if best_weight < 0.30:
                f.write("\n")
                f.write("⚠️  Model Uncertainty Detected\n")
                f.write(f"   Best weight: {best_weight*100:.1f}% < 30% threshold\n")
                f.write("   Action: Run model averaging (step02c_model_averaging.py)\n")
            elif best_weight < 0.70:
                f.write("\n")
                f.write("ℹ️  Moderate Model Uncertainty\n")
                f.write(f"   Best weight: {best_weight*100:.1f}% (30-70% range)\n")
                f.write("   Action: Model averaging optional\n")

        log(f"[SAVE] Summary written to: {summary_path.name}")

        log("=" * 80)
        log("[SUCCESS] Step 02b complete")
        log("=" * 80)

    except Exception as e:
        log("[ERROR] Step 02b failed")
        log(f"  {type(e).__name__}: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
