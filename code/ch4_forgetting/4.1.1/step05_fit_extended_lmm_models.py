"""
RQ 5.1.1 - Step 05: Fit Extended LMM Model Suite (Kitchen Sink)

PURPOSE:
Compare 70+ LMM trajectory models using comprehensive time transformations
to identify best-fitting functional form for forgetting curves.

INPUT:
- data/step04_lmm_input.csv (400 rows × 6 columns with continuous TSVR_hours)

OUTPUT:
- data/step05_model_comparison.csv (70+ rows, AIC-sorted model comparison)
- data/step05_best_model_summary.txt (best model details)
- logs/step05_kitchen_sink.log (detailed execution log)

CRITICAL:
Uses tools.model_selection.compare_lmm_models_kitchen_sink() with:
- Continuous TSVR_hours (NOT nominal Days or sessions)
- 70+ models (polynomial, logarithmic, power-law, root, reciprocal, exponential, trig, hyperbolic)
- Random intercepts only (re_formula='~1') for model comparison stability
- ML estimation (reml=False) for AIC comparison

DESIGN PHILOSOPHY:
Zero assumptions about functional form - test EVERY mathematically plausible
time transformation. Let data determine best model via AIC.

Author: g_code
Date: 2025-12-08
RQ: ch5/5.1.1
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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
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
        log("[START] Step 05: Kitchen Sink LMM Model Comparison")
        log("=" * 80)

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================

        log("[LOAD] Loading LMM input data...")
        input_path = DATA_DIR / "step04_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"LMM input missing: {input_path}\n"
                "Run step04_prepare_lmm_input.py first"
            )

        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"  ✓ Loaded {input_path.name}")
        log(f"    Rows: {len(lmm_input)}")
        log(f"    Columns: {lmm_input.columns.tolist()}")

        # Verify required columns
        required_cols = ['UID', 'theta', 'TSVR_hours']
        missing = [col for col in required_cols if col not in lmm_input.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        log(f"  ✓ TSVR unique values: {lmm_input['TSVR_hours'].nunique()}")
        log(f"  ✓ TSVR range: [{lmm_input['TSVR_hours'].min():.2f}, "
            f"{lmm_input['TSVR_hours'].max():.2f}] hours")
        log(f"  ✓ Theta range: [{lmm_input['theta'].min():.3f}, "
            f"{lmm_input['theta'].max():.3f}]")
        log(f"  ✓ Participants: {lmm_input['UID'].nunique()}")

        # =====================================================================
        # STEP 2: Run Kitchen Sink Model Comparison
        # =====================================================================

        log("[ANALYSIS] Running kitchen sink model comparison...")
        log("  Model suite: 70+ time transformations")
        log("  Outcome: theta (IRT ability estimates)")
        log("  Time variable: TSVR_hours (continuous)")
        log("  Groups: UID (participants)")
        log("  Random effects: ~1 (intercepts only)")
        log("  Estimation: ML (reml=False, for AIC comparison)")
        log("  Interaction: None (simple trajectory)")

        results = compare_lmm_models_kitchen_sink(
            data=lmm_input,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',

            # No interaction factors (simple trajectory)
            factor1_var=None,
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
        best_model = results['best_model']
        log_model = results['log_model_info']
        top_10 = results['top_10']
        stats = results['summary_stats']

        log(f"  Models tested: {stats['n_models_tested']}")
        log(f"  Models converged: {stats['n_models_converged']}")
        log(f"  Models failed: {stats['n_models_failed']}")
        log(f"  Competitive models (ΔAIC < 2): {stats['n_competitive_models']}")
        log("")

        log(f"  BEST MODEL: {best_model['name']}")
        log(f"    AIC: {best_model['AIC']:.2f}")
        log(f"    Akaike weight: {best_model['weight']:.4f} ({best_model['weight_pct']:.1f}%)")
        log(f"    Uncertainty: {best_model['uncertainty']}")
        log(f"    Interpretation: {best_model['interpretation']}")
        log("")

        if 'rank' in log_model:
            log(f"  LOG MODEL (benchmark):")
            log(f"    Rank: #{log_model['rank']}")
            log(f"    AIC: {log_model['AIC']:.2f}")
            log(f"    ΔAIC: {log_model['delta_AIC']:.2f}")
            log(f"    Akaike weight: {log_model['weight']:.4f} ({log_model['weight_pct']:.1f}%)")

            if log_model['rank'] == 1:
                log("    → Logarithmic is BEST model")
            elif log_model['rank'] <= 3:
                log("    → Logarithmic in TOP 3")
            elif log_model['rank'] <= 10:
                log(f"    → Logarithmic DEMOTED to rank #{log_model['rank']}")
            else:
                log(f"    → Logarithmic WEAK support (rank #{log_model['rank']})")
        else:
            log(f"  LOG MODEL: {log_model['error']}")

        log("")
        log("  TOP 10 MODELS:")
        for idx, row in top_10.iterrows():
            log(f"    {idx+1:2d}. {row['model_name']:20s}  AIC={row['AIC']:7.2f}  "
                f"Δ={row['delta_AIC']:5.2f}  w={row['akaike_weight']:.4f}  "
                f"cum={row['cumulative_weight']:.4f}")

        log("=" * 80)

        # =====================================================================
        # STEP 4: Validate Outputs Saved
        # =====================================================================

        log("[VALIDATE] Checking output files...")

        comparison_path = DATA_DIR / "model_comparison.csv"
        summary_path = DATA_DIR / "best_model_summary.txt"

        if not comparison_path.exists():
            raise FileNotFoundError(f"Model comparison CSV not saved: {comparison_path}")
        log(f"  ✓ {comparison_path.name} ({len(comparison)} models)")

        if not summary_path.exists():
            raise FileNotFoundError(f"Best model summary not saved: {summary_path}")
        log(f"  ✓ {summary_path.name}")

        # Rename to step-prefixed names for RQ workflow
        comparison_final = DATA_DIR / "step05_model_comparison.csv"
        summary_final = DATA_DIR / "step05_best_model_summary.txt"

        comparison_path.rename(comparison_final)
        summary_path.rename(summary_final)

        log(f"  ✓ Renamed to {comparison_final.name}")
        log(f"  ✓ Renamed to {summary_final.name}")

        # =====================================================================
        # SUMMARY
        # =====================================================================

        log("=" * 80)
        log("[SUCCESS] Step 05 Complete")
        log(f"  Best model: {best_model['name']} (AIC={best_model['AIC']:.2f}, w={best_model['weight']:.4f})")
        log(f"  Log model: Rank #{log_model.get('rank', 'N/A')}")
        log(f"  Models converged: {stats['n_models_converged']}/{stats['n_models_tested']}")
        log(f"  Output: {comparison_final.name} ({len(comparison)} models)")
        log(f"  Ready for: Step 06 (model selection interpretation)")
        log("=" * 80)

    except Exception as e:
        log(f"\n[ERROR] Step 05 Failed: {e}")
        import traceback
        log(traceback.format_exc())
        raise
