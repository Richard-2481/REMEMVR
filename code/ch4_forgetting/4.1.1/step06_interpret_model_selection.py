"""
RQ 5.1.1 - Step 06: Interpret Model Selection Results

PURPOSE:
Interpret extended model comparison results, identifying best model and 
evaluating logarithmic benchmark performance.

INPUT:
- data/step05_model_comparison.csv (65+ models, AIC-sorted)

OUTPUT:
- results/step06_model_selection_summary.txt (interpretation)
- results/step06_competitive_models.csv (ΔAIC < 2)

Author: g_code
Date: 2025-12-08
RQ: ch5/5.1.1
Step: 06
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_interpret_selection.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 06: Interpret Model Selection")
        log("=" * 80)

        comparison_path = RQ_DIR / "data" / "step05_model_comparison.csv"
        comparison = pd.read_csv(comparison_path, encoding='utf-8')
        log(f"  ✓ Loaded {len(comparison)} models")
        
        best = comparison.iloc[0]
        log(f"\n[BEST MODEL] {best['model_name']}")
        log(f"  AIC: {best['AIC']:.2f}")
        log(f"  Akaike weight: {best['akaike_weight']:.4f}")
        
        competitive = comparison[comparison['delta_AIC'] < 2.0]
        log(f"\n[COMPETITIVE] {len(competitive)} models with ΔAIC < 2")
        
        log_row = comparison[comparison['model_name'] == 'Log']
        if len(log_row) > 0:
            log_idx = log_row.index[0]
            log(f"\n[LOG MODEL] Rank #{log_idx + 1}")
            log(f"  ΔAIC: {log_row.iloc[0]['delta_AIC']:.2f}")
        
        competitive_path = RQ_DIR / "results" / "step06_competitive_models.csv"
        competitive.to_csv(competitive_path, index=False, encoding='utf-8')
        
        summary_path = RQ_DIR / "results" / "step06_model_selection_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Best Model: {best['model_name']}\n")
            f.write(f"AIC: {best['AIC']:.2f}\n")
            f.write(f"Competitive models: {len(competitive)}\n")
        
        log("=" * 80)
        log("[SUCCESS] Step 06 Complete")
        log("=" * 80)
        
    except Exception as e:
        log(f"[ERROR] {e}")
        raise
