#!/usr/bin/env python3
"""fit_reverse_models: Fit reverse regression models (RAVLT/BVMT ~ REMEMVR) with bootstrap CIs and dual p-values"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_multiple_regression, bootstrap_regression_ci
from tools.bootstrap import bootstrap_statistic

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_reverse_models.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Fit Reverse Regression Models")

        # Load standardized data
        log("Loading T-scored variables...")
        input_path = RQ_DIR / 'data' / 'step02_standardized_data.csv'
        df = pd.read_csv(input_path)
        log(f"{len(df)} participants")

        # Fit reverse regression models
        # Models: REMEMVR_T → each outcome (total scores + percent retention)
        models_to_fit = [
            ('RAVLT_reverse', 'RAVLT_T'),
            ('BVMT_reverse', 'BVMT_T'),
            ('RAVLT_PctRet_reverse', 'RAVLT_PctRet_T'),
            ('BVMT_PctRet_reverse', 'BVMT_PctRet_T'),
        ]

        n_tests = len(models_to_fit)
        log(f"Fitting {n_tests} reverse models (Bonferroni alpha=0.05/{n_tests}={0.05/n_tests:.4f})")

        all_results = []

        for model_idx, (model_name, outcome_col) in enumerate(models_to_fit, 1):
            log(f"[MODEL {model_idx}] Fitting {outcome_col} ~ REMEMVR_T...")

            X = df[['REMEMVR_T']].values
            y = df[outcome_col].values

            model = fit_multiple_regression(X=X, y=y, feature_names=['REMEMVR_T'])

            log(f"R²={model['rsquared']:.4f}, adj_R²={model['rsquared_adj']:.4f}")
            log(f"Beta={model['coefficients']['REMEMVR_T']:.4f}, p={model['pvalues']['REMEMVR_T']:.4f}")

            # Bootstrap CIs
            log(f"Computing 95% CIs for {model_name}...")
            boot = bootstrap_regression_ci(X=X, y=y, n_bootstrap=1000, seed=42)
            ci_lower = boot['ci_lower'][1]
            ci_upper = boot['ci_upper'][1]
            log(f"Beta 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

            # Bonferroni correction
            p_uncorrected = model['pvalues']['REMEMVR_T']
            p_bonferroni = min(p_uncorrected * n_tests, 1.0)
            log(f"p_uncorrected={p_uncorrected:.4f}, p_bonferroni={p_bonferroni:.4f}")

            all_results.append({
                'model': model_name,
                'outcome': outcome_col,
                'predictor': 'REMEMVR_T',
                'R2': model['rsquared'],
                'adj_R2': model['rsquared_adj'],
                'beta': model['coefficients']['REMEMVR_T'],
                'se': model['std_errors']['REMEMVR_T'],
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_uncorrected': p_uncorrected,
                'p_bonferroni': p_bonferroni,
            })

        # Compile results
        log("Compiling model results...")
        results = pd.DataFrame(all_results)

        output_path = RQ_DIR / 'data' / 'step03_reverse_models.csv'
        results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        # Summary
        log("Reverse prediction results:")
        for r in all_results:
            log(f"  {r['model']}: R²={r['R2']:.3f}, p_bonf={r['p_bonferroni']:.4f}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
