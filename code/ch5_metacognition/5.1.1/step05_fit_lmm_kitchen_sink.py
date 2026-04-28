"""
RQ 6.1.1 - Step 05: Fit Extended LMM Model Suite (Kitchen Sink)

PURPOSE:
Compare 70+ LMM trajectory models using comprehensive time transformations
to identify best-fitting functional form for confidence decline curves.

INPUT:
- data/step04_lmm_input.csv (400 rows × 9 columns with continuous TSVR_hours)

OUTPUT:
- data/step05_model_comparison.csv (70+ rows, AIC-sorted model comparison)
- data/step05_best_model_summary.txt (best model details)
- logs/step05_kitchen_sink.log (detailed execution log)

CRITICAL:
Uses tools.model_selection.compare_lmm_models_kitchen_sink() with:
- Continuous TSVR_hours (NOT nominal Days or sessions)
- 70+ models (polynomial, logarithmic, power-law, root, reciprocal, exponential, trig, hyperbolic)
- Random intercepts only (re_formula='~1') - FIXED: was '~TSVR_hours' causing convergence failures
- ML estimation (reml=False) for AIC comparison
- NO interaction factors (simple trajectory)

Author: Updated for kitchen sink approach
Date: 2025-12-10
RQ: ch6/6.1.1
Step: 05
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.model_selection import compare_lmm_models_kitchen_sink

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_kitchen_sink.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Step 05: Kitchen Sink LMM Model Comparison")
        log("=" * 80)

        log("Loading LMM input data...")
        input_path = DATA_DIR / "step04_lmm_input.csv"
        lmm_input = pd.read_csv(input_path, encoding='utf-8')
        log(f"  ✓ Loaded {len(lmm_input)} rows, {len(lmm_input.columns)} columns")
        log(f"  ✓ Outcome: theta_All, TSVR range: [{lmm_input['TSVR_hours'].min():.2f}, {lmm_input['TSVR_hours'].max():.2f}]")

        log("Running kitchen sink model comparison...")
        log("  CORRECTED: Using re_formula='~1' (random intercepts only) to match Ch5")
        results = compare_lmm_models_kitchen_sink(
            data=lmm_input,
            outcome_var='theta_All',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            factor1_var=None,
            factor2_var=None,
            re_formula='~1',  # FIXED: was '~TSVR_hours' causing convergence failures
            reml=False,
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
        )

        comparison = results['comparison']
        best_model = results['best_model']
        log(f"Best model: {best_model['name']} (AIC={best_model['AIC']:.2f}, w={best_model['weight']:.4f})")
        log(f"  Converged: {results['summary_stats']['n_models_converged']}/{results['summary_stats']['n_models_tested']}")

        # Rename outputs
        (DATA_DIR / "model_comparison.csv").rename(DATA_DIR / "step05_model_comparison.csv")
        (DATA_DIR / "best_model_summary.txt").rename(DATA_DIR / "step05_best_model_summary.txt")
        log("=" * 80)

    except Exception as e:
        log(f"{e}")
        import traceback
        log(traceback.format_exc())
        raise
