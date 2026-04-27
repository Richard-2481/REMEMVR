"""
RQ 6.3.1 - Step 05: Fit Extended LMM Model Suite (Kitchen Sink) with Domain Interaction

PURPOSE:
Compare 70+ LMM trajectory models to identify best functional form for
domain-specific (What/Where/When) confidence decline trajectories.

INPUT:
- data/step04_lmm_input.csv (1200 rows = 400 observations × 3 domains)

OUTPUT:
- data/step05_model_comparison.csv
- data/step05_best_model_summary.txt  
- logs/step05_kitchen_sink.log

INTERACTION:
- factor1: domain (categorical: What, Where, When)
- Tests time×domain interaction effects

Author: Updated for kitchen sink approach
Date: 2025-12-10
RQ: ch6/6.3.1
"""

import sys
from pathlib import Path
import pandas as pd

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
        log("[START] Step 05: Kitchen Sink LMM - Domain Interaction")
        log("=" * 80)

        log("[LOAD] Loading LMM input data...")
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv", encoding='utf-8')
        log(f"  ✓ Loaded {len(lmm_input)} rows ({lmm_input['UID'].nunique()} participants × {lmm_input['domain'].nunique()} domains)")
        log(f"  ✓ Domains: {sorted(lmm_input['domain'].unique())}")

        log("[ANALYSIS] Running kitchen sink with domain interaction...")
        results = compare_lmm_models_kitchen_sink(
            data=lmm_input,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            factor1_var='domain',
            factor1_type='categorical',
            factor1_reference='What',  # Reference level
            factor2_var=None,
            re_formula='~TSVR_hours',
            reml=False,
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
        )

        best_model = results['best_model']
        log(f"[SUCCESS] Best model: {best_model['name']} (AIC={best_model['AIC']:.2f})")
        (DATA_DIR / "model_comparison.csv").rename(DATA_DIR / "step05_model_comparison.csv")
        (DATA_DIR / "best_model_summary.txt").rename(DATA_DIR / "step05_best_model_summary.txt")
        log("=" * 80)

    except Exception as e:
        log(f"[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
