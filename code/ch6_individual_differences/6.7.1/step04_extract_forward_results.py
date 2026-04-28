#!/usr/bin/env python3
"""Step 04: Extract forward results from RQ 7.1.1 (optional dependency)"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_bidirectional_comparison.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 04: Extract Forward Results")

        # Load reverse results
        reverse_path = RQ_DIR / 'data' / 'step03_reverse_models.csv'
        df_reverse = pd.read_csv(reverse_path)
        log(f"Reverse results: {len(df_reverse)} models")

        # Try to load forward results from RQ 7.1.1
        forward_path = PROJECT_ROOT / 'results' / 'ch7' / '7.1.1' / 'data' / 'step03_forward_models.csv'

        if not forward_path.exists():
            log("RQ 7.1.1 forward results not available")
            log("Creating empty comparison file (reverse-only mode)")

            # Create empty DataFrame with warning
            rows = []
            for test_name, outcome_col in [('RAVLT', 'RAVLT_T'), ('BVMT', 'BVMT_T'),
                                           ('RAVLT_PctRet', 'RAVLT_PctRet_T'), ('BVMT_PctRet', 'BVMT_PctRet_T')]:
                r2_val = df_reverse[df_reverse['outcome']==outcome_col]['R2'].values
                rows.append({
                    'test': test_name,
                    'forward_R2': np.nan,
                    'reverse_R2': r2_val[0] if len(r2_val) > 0 else np.nan,
                    'asymmetry_ratio': np.nan,
                    'williams_t': np.nan,
                    'williams_p': np.nan,
                    'note': 'Forward results unavailable - RQ 7.1.1 not complete'
                })
            df_output = pd.DataFrame(rows)
        else:
            log(f"Forward results from RQ 7.1.1")
            df_forward = pd.read_csv(forward_path)

            # Extract R² values and compute asymmetry ratios
            # Placeholder for bidirectional comparison logic
            df_output = pd.DataFrame({
                'test': ['RAVLT', 'BVMT'],
                'note': ['Implementation requires RQ 7.1.1 data structure confirmation',
                        'Implementation requires RQ 7.1.1 data structure confirmation']
            })

        output_path = RQ_DIR / 'data' / 'step04_bidirectional_comparison.csv'
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
