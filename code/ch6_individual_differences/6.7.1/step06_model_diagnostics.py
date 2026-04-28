#!/usr/bin/env python3
"""Step 06: Regression diagnostics (normality, homoscedasticity, outliers)"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import fit_multiple_regression, compute_regression_diagnostics

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_diagnostics.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 06: Model Diagnostics")

        # Load data
        data_path = RQ_DIR / 'data' / 'step02_standardized_data.csv'
        df = pd.read_csv(data_path)

        diagnostics = []

        for outcome in ['RAVLT_T', 'BVMT_T', 'RAVLT_PctRet_T', 'BVMT_PctRet_T']:
            log(f"Testing {outcome} model...")

            X = df[['REMEMVR_T']].values
            y = df[outcome].values

            model_result = fit_multiple_regression(X, y, feature_names=['REMEMVR_T'])
            model = model_result['model']

            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(model.resid)
            normality_pass = shapiro_p > 0.05

            # Breusch-Pagan test for homoscedasticity
            diag = compute_regression_diagnostics(model, X, y)
            bp_p = diag['breusch_pagan']['p_value']
            homoscedasticity_pass = bp_p > 0.05

            # Cook's distance for outliers
            cooks_d = diag['cooks_d']
            threshold = 4 / len(df)
            outlier_count = np.sum(cooks_d > threshold)
            max_cooks = np.max(cooks_d)

            log(f"  Normality (Shapiro-Wilk): p={shapiro_p:.4f} {'' if normality_pass else ''}")
            log(f"  Homoscedasticity (BP): p={bp_p:.4f} {'' if homoscedasticity_pass else ''}")
            log(f"  Outliers (Cook's D): {outlier_count} exceed threshold {threshold:.4f}, max={max_cooks:.4f}")

            # Remedial actions
            if not normality_pass:
                remedial = "Use bootstrap CIs (already computed in step03)"
            elif not homoscedasticity_pass:
                remedial = "Compute HC3 robust SEs (heteroscedasticity-consistent)"
            elif outlier_count > 0:
                remedial = f"Sensitivity analysis: rerun without {outlier_count} outliers"
            else:
                remedial = "None needed"

            diagnostics.append({
                'model': f'{outcome}_reverse',
                'normality_p': shapiro_p,
                'bp_test_p': bp_p,
                'max_cooks_d': max_cooks,
                'outlier_count': outlier_count,
                'assumptions_met': normality_pass and homoscedasticity_pass and outlier_count == 0,
                'remedial_action': remedial
            })

        df_diagnostics = pd.DataFrame(diagnostics)
        output_path = RQ_DIR / 'data' / 'step06_model_diagnostics.csv'
        df_diagnostics.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
