#!/usr/bin/env python3
"""Step 07: 5-fold cross-validation to assess generalizability"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_regression import cross_validate_regression, fit_multiple_regression

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 07: Cross-Validation")

        # Load data
        data_path = RQ_DIR / 'data' / 'step02_standardized_data.csv'
        df = pd.read_csv(data_path)

        cv_results = []

        for outcome in ['RAVLT_T', 'BVMT_T', 'RAVLT_PctRet_T', 'BVMT_PctRet_T']:
            log(f"[CV] 5-fold cross-validation for {outcome} model...")

            X = df[['REMEMVR_T']].values
            y = df[outcome].values

            # Full model R² (train)
            full_model = fit_multiple_regression(X, y, feature_names=['REMEMVR_T'])
            train_r2 = full_model['rsquared']

            # Cross-validation
            cv_result = cross_validate_regression(X=X, y=y, n_folds=5, seed=42)

            # Calculate RMSE and MAE from fold predictions
            all_rmse = []
            all_mae = []
            for fold_name, fold_data in cv_result['fold_predictions'].items():
                y_true = fold_data['y_true']
                y_pred = fold_data['y_pred']
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mae = np.mean(np.abs(y_true - y_pred))
                all_rmse.append(rmse)
                all_mae.append(mae)

            mean_test_r2 = cv_result['mean_r2']
            generalization_gap = train_r2 - mean_test_r2
            overfitting_flag = generalization_gap > 0.10

            log(f"  Train R²: {train_r2:.4f}")
            log(f"  Test R² (mean): {mean_test_r2:.4f} (SD={cv_result['std_r2']:.4f})")
            log(f"  Generalization gap: {generalization_gap:.4f}")
            log(f"  RMSE (mean): {np.mean(all_rmse):.4f}")
            log(f"  MAE (mean): {np.mean(all_mae):.4f}")
            log(f"  Overfitting: {'YES' if overfitting_flag else 'NO'}")

            cv_results.append({
                'model': f'{outcome}_reverse',
                'mean_test_R2': mean_test_r2,
                'sd_test_R2': cv_result['std_r2'],
                'train_R2': train_r2,
                'generalization_gap': generalization_gap,
                'rmse_mean': np.mean(all_rmse),
                'mae_mean': np.mean(all_mae),
                'overfitting_flag': overfitting_flag
            })

        df_cv = pd.DataFrame(cv_results)
        output_path = RQ_DIR / 'data' / 'step07_cross_validation.csv'
        df_cv.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
