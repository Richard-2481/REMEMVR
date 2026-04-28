#!/usr/bin/env python3
"""cross_validation: 5-fold cross-validation to assess model generalizability and overfitting."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"
OUTPUT_CV = RQ_DIR / "data" / "step07_cv_results.csv"
OUTPUT_SUMMARY = RQ_DIR / "data" / "step07_cv_summary.csv"

# Configuration

N_FOLDS = 5
RANDOM_SEED = 42
OVERFITTING_THRESHOLD = 0.10

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Cross-Validation")
        # Load Data
        log("Loading analysis dataset...")

        data_path = RQ_DIR / "data" / "step03_analysis_dataset.csv"
        df = pd.read_csv(data_path)

        log(f"{data_path.name} ({len(df)} participants)")
        # Define Models
        models_spec = [
            {'name': 'Model_1_Total', 'predictors': ['Total_z']},
            {'name': 'Model_2_Learning', 'predictors': ['Learning_z']},
            {'name': 'Model_3_LearningSlope', 'predictors': ['LearningSlope_z']},
            {'name': 'Model_4_Forgetting', 'predictors': ['Forgetting_z']},
            {'name': 'Model_5_Recognition', 'predictors': ['Recognition_z']},
            {'name': 'Model_6_PctRet', 'predictors': ['PctRet_z']},
            {'name': 'Model_7_Combined', 'predictors': ['Total_z', 'Learning_z']}
        ]
        # K-Fold Cross-Validation
        log(f"[CV] Running {N_FOLDS}-fold cross-validation...")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        cv_results = []

        for spec in models_spec:
            model_name = spec['name']
            predictors = spec['predictors']

            log(f"[CV] {model_name}: {', '.join(predictors)}")

            X = df[predictors].values
            y = df['theta_all'].values

            fold_num = 1

            for train_idx, test_idx in kf.split(X):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Add constant
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)

                # Fit model on training data
                model = sm.OLS(y_train, X_train_const).fit()

                # Evaluate on training data
                y_train_pred = model.predict(X_train_const)
                train_r2 = 1 - (np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2))

                # Evaluate on test data
                y_test_pred = model.predict(X_test_const)
                test_r2 = 1 - (np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

                # Compute error metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae = mean_absolute_error(y_test, y_test_pred)

                cv_results.append({
                    'fold': fold_num,
                    'model': model_name,
                    'train_R2': train_r2,
                    'test_R2': test_r2,
                    'RMSE': rmse,
                    'MAE': mae
                })

                log(f"  Fold {fold_num}: train_R2={train_r2:.4f}, test_R2={test_r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

                fold_num += 1

            log(f"{model_name}")

        log(f"[CV] All models complete ({len(cv_results)} results)")
        # Aggregate Results
        log("Computing CV summary statistics...")

        cv_df = pd.DataFrame(cv_results)

        summary_results = []

        for spec in models_spec:
            model_name = spec['name']
            model_cv = cv_df[cv_df['model'] == model_name]

            mean_train_r2 = model_cv['train_R2'].mean()
            mean_test_r2 = model_cv['test_R2'].mean()
            sd_test_r2 = model_cv['test_R2'].std()
            mean_rmse = model_cv['RMSE'].mean()
            mean_mae = model_cv['MAE'].mean()

            # Compute train-test gap (overfitting indicator)
            train_test_gap = mean_train_r2 - mean_test_r2

            # Compute shrinkage (percentage)
            shrinkage = (train_test_gap / mean_train_r2) * 100 if mean_train_r2 > 0 else 0

            # Flag overfitting
            overfitting_flag = train_test_gap > OVERFITTING_THRESHOLD

            summary_results.append({
                'model': model_name,
                'mean_train_R2': mean_train_r2,
                'mean_test_R2': mean_test_r2,
                'sd_test_R2': sd_test_r2,
                'mean_RMSE': mean_rmse,
                'mean_MAE': mean_mae,
                'train_test_gap': train_test_gap,
                'shrinkage': shrinkage,
                'overfitting_flag': overfitting_flag
            })

            log(f"{model_name}:")
            log(f"  Mean test R²: {mean_test_r2:.4f} (SD={sd_test_r2:.4f})")
            log(f"  Train-test gap: {train_test_gap:.4f} (shrinkage={shrinkage:.1f}%)")
            if overfitting_flag:
                log(f"  Overfitting detected (gap > {OVERFITTING_THRESHOLD})")
            else:
                log(f"  No overfitting detected")
        # Save Results
        log("Saving CV results...")

        cv_df.to_csv(OUTPUT_CV, index=False, encoding='utf-8')
        log(f"{OUTPUT_CV} ({len(cv_df)} fold results)")

        log("Saving CV summary...")

        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding='utf-8')
        log(f"{OUTPUT_SUMMARY} ({len(summary_df)} models)")
        # Final Summary
        log("Cross-validation complete:")
        log(f"  Total CV runs: {len(cv_df)} ({N_FOLDS} folds × {len(models_spec)} models)")
        log(f"  Models with overfitting: {summary_df['overfitting_flag'].sum()}/{len(models_spec)}")

        # Best model by test R²
        best_model = summary_df.loc[summary_df['mean_test_R2'].idxmax()]
        log(f"  Best model (by test R²): {best_model['model']} (R²={best_model['mean_test_R2']:.4f})")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
