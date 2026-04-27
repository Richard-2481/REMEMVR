#!/usr/bin/env python3
"""Step 05: Compute Cohen's f² effect sizes with bootstrap CIs"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import compute_cohens_f2
from tools.bootstrap import bootstrap_statistic

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_effect_sizes.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("[START] Step 05: Compute Effect Sizes")

        # Load regression results
        results_path = RQ_DIR / 'data' / 'step03_reverse_models.csv'
        df_results = pd.read_csv(results_path)

        # Load data for bootstrap
        data_path = RQ_DIR / 'data' / 'step02_standardized_data.csv'
        df_data = pd.read_csv(data_path)

        effect_sizes = []

        for _, row in df_results.iterrows():
            # Compute Cohen's f²: R²/(1-R²)
            r2 = row['R2']
            f2 = compute_cohens_f2(r2_full=r2, r2_reduced=0.0)

            # Interpret effect size
            if f2 < 0.02:
                interpretation = 'negligible'
            elif f2 < 0.15:
                interpretation = 'small'
            elif f2 < 0.35:
                interpretation = 'medium'
            else:
                interpretation = 'large'

            log(f"[{row['model']}] f²={f2:.4f} ({interpretation})")

            # Bootstrap CIs for f²
            def compute_f2_from_data(data_indices):
                X = df_data.iloc[data_indices][['REMEMVR_T']].values
                y = df_data.iloc[data_indices][row['outcome']].values
                from tools.analysis_regression import fit_multiple_regression
                model = fit_multiple_regression(X, y, feature_names=['REMEMVR_T'])
                return compute_cohens_f2(model['rsquared'], 0.0)

            indices = np.arange(len(df_data))
            boot_result = bootstrap_statistic(
                data=indices,
                statistic=compute_f2_from_data,
                n_bootstrap=1000,
                confidence=0.95,
                seed=42
            )

            effect_sizes.append({
                'model': row['model'],
                'cohens_f2': f2,
                'f2_ci_lower': boot_result['ci_lower'],
                'f2_ci_upper': boot_result['ci_upper'],
                'effect_interpretation': interpretation
            })

        df_effect_sizes = pd.DataFrame(effect_sizes)
        output_path = RQ_DIR / 'data' / 'step05_effect_sizes.csv'
        df_effect_sizes.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {output_path}")

        log("[SUCCESS] Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
