"""
RQ 5.2.1 - Step 05: Fit Extended LMM Model Suite (Kitchen Sink)

PURPOSE:
Compare 70+ LMM trajectory models with Domain × Time interaction using
comprehensive time transformations to identify best-fitting functional form.

INPUT:
- data/step04_lmm_input.csv (400 rows × 7 columns: UID, test, TSVR_hours, domain, theta, composite_ID)

OUTPUT:
- data/step05_model_comparison.csv (70+ rows, AIC-sorted model comparison)
- data/step05_best_model_summary.txt (best model details)
- logs/step05_kitchen_sink.log (detailed execution log)

CRITICAL:
Uses tools.model_selection.compare_lmm_models_kitchen_sink() with:
- Continuous TSVR_hours (NOT nominal Days or sessions)
- 70+ models (polynomial, logarithmic, power-law, root, reciprocal, exponential, trig, hyperbolic)
- Domain × Time interaction (factor1_var='domain')
- Random intercepts only (re_formula='~1') for model comparison stability
- ML estimation (reml=False) for AIC comparison

DESIGN PHILOSOPHY:
Zero assumptions about functional form - test EVERY mathematically plausible
time transformation. Let data determine best model via AIC.

Author: Claude Code (propagation from 5.1.1)
Date: 2025-12-08
RQ: ch5/5.2.1
Step: 05
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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.1
LOG_FILE = RQ_DIR / "logs" / "step05_kitchen_sink.log"
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
        log("[START] Step 05: Kitchen Sink LMM Model Comparison (Domain × Time)")
        log("=" * 80)

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================

        log("[LOAD] Loading LMM input data...")
        input_path = DATA_DIR / "step04_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"LMM input missing: {input_path}\n"
                "Run step04_merge_theta_tsvr.py first"
            )

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"  ✓ Loaded {input_path.name}")
        log(f"    Rows: {len(lmm_input)}")
        log(f"    Columns: {lmm_input.columns.tolist()}")

        # Verify required columns
        required_cols = ['UID', 'theta', 'TSVR_hours', 'domain']
        missing = [col for col in required_cols if col not in lmm_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        log(f"  ✓ TSVR unique values: {lmm_input['TSVR_hours'].nunique()}")
        log(f"  ✓ TSVR range: [{lmm_input['TSVR_hours'].min():.2f}, "
            f"{lmm_input['TSVR_hours'].max():.2f}] hours")
        log(f"  ✓ Theta range: [{lmm_input['theta'].min():.3f}, "
            f"{lmm_input['theta'].max():.3f}]")
        log(f"  ✓ Participants: {lmm_input['UID'].nunique()}")
        log(f"  ✓ Domain levels: {sorted(lmm_input['domain'].unique())}")
        log(f"  ✓ Domain counts: {lmm_input['domain'].value_counts().to_dict()}")

        # =====================================================================
        # STEP 2: Run Kitchen Sink Model Comparison
        # =====================================================================

        log("[ANALYSIS] Running kitchen sink model comparison...")
        log("  Model suite: 70+ time transformations")
        log("  Outcome: theta (IRT ability estimates)")
        log("  Time variable: TSVR_hours (continuous)")
        log("  Factor 1: domain (What, Where, When)")
        log("  Interaction: domain × time_transform")
        log("  Groups: UID (participants)")
        log("  Random effects: ~1 (intercepts only)")
        log("  Estimation: ML (reml=False, for AIC comparison)")

        results = compare_lmm_models_kitchen_sink(
            data=lmm_input,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',

            # Domain × Time interaction
            factor1_var='domain',
            factor1_reference='what',  # Treatment coding: What as reference
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

        log("[RESULTS] Model comparison summary:")
        log("=" * 80)

        comparison = results['comparison']
        best_model_info = results['best_model']
        best_model_name = best_model_info['name']

        # Top 10 models
        log(f"TOP 10 MODELS (out of {len(comparison)}):")
        log("-" * 80)
        for i, (_, row) in enumerate(comparison.head(10).iterrows(), 1):
            log(f"  {i:2d}. {row['model_name']:25s} "
                f"AIC={row['AIC']:7.2f}  ΔAIC={row['delta_AIC']:6.2f}  "
                f"weight={row['akaike_weight']*100:5.2f}%")

        log("=" * 80)
        log(f"BEST MODEL: {best_model_name}")
        log(f"  AIC: {best_model_info['AIC']:.2f}")
        log(f"  Weight: {best_model_info['weight']*100:.2f}%")
        log("=" * 80)

        # Check if model averaging needed (weight < 30%)
        best_weight = comparison.iloc[0]['akaike_weight']
        if best_weight < 0.30:
            log("")
            log("⚠️  WARNING: Best model weight < 30%")
            log(f"   Model uncertainty present (weight={best_weight*100:.1f}%)")
            log("   RECOMMENDATION: Run step05c_model_averaging.py")
            log("   See: Burnham & Anderson (2002) Model Selection section")
            log("")

        # =====================================================================
        # STEP 4: Save Summary
        # =====================================================================

        summary_path = DATA_DIR / "step05_best_model_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RQ 5.2.1 - Kitchen Sink Model Selection Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"AIC: {best_model_info['AIC']:.2f}\n")
            f.write(f"Weight: {best_model_info['weight']*100:.2f}%\n\n")

            f.write("Top 10 Models:\n")
            f.write("-" * 80 + "\n")
            for i, (_, row) in enumerate(comparison.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['model_name']:25s} "
                       f"AIC={row['AIC']:7.2f}  ΔAIC={row['delta_AIC']:6.2f}  "
                       f"weight={row['akaike_weight']*100:5.2f}%\n")

            if best_weight < 0.30:
                f.write("\n")
                f.write("⚠️  Model Uncertainty Detected\n")
                f.write(f"   Best weight: {best_weight*100:.1f}% < 30% threshold\n")
                f.write("   Action: Run model averaging (step05c_model_averaging.py)\n")

        log(f"[SAVE] Summary written to: {summary_path.name}")

        log("=" * 80)
        log("[SUCCESS] Step 05 complete")
        log("=" * 80)

    except Exception as e:
        log("[ERROR] Step 05 failed")
        log(f"  {type(e).__name__}: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
