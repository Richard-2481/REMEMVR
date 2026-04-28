#!/usr/bin/env python3
"""Cross-Validation Analysis: 5-fold cross-validation for generalizability assessment of hierarchical"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import required libraries
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"

# Cross-validation parameters
N_FOLDS = 5
RANDOM_STATE = 42
SHUFFLE = True
OVERFITTING_THRESHOLD = 0.10  # Flag if train-test R² gap > 0.10

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Cross-Validation Analysis")
        # Load Input Data

        log("Loading analysis input data...")
        input_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        df = pd.read_csv(input_path)
        log(f"step03_analysis_input.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Verify required columns
        required_cols = ['UID', 'slope', 'age_std', 'sex', 'education_std',
                        'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                        'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Prepare Data for Cross-Validation

        log("Preparing data for cross-validation...")

        # Predictor columns (full Model 2)
        predictor_cols = ['age_std', 'sex', 'education_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                          'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        X = df[predictor_cols].values
        y = df['slope'].values

        log(f"Predictors: {predictor_cols}")
        log(f"Response: slope")
        log(f"Sample size: {len(y)}")
        # Initialize K-Fold Cross-Validation

        log(f"[CV] Initializing {N_FOLDS}-fold cross-validation...")
        log(f"[CV] Random state: {RANDOM_STATE}")
        log(f"[CV] Shuffle: {SHUFFLE}")

        kfold = KFold(n_splits=N_FOLDS, shuffle=SHUFFLE, random_state=RANDOM_STATE)
        # Run Cross-Validation

        log("[CV] Running cross-validation folds...")

        cv_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
            log(f"[CV] --- Fold {fold_idx}/{N_FOLDS} ---")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            log(f"[CV] Train size: {len(y_train)}, Test size: {len(y_test)}")

            # Add constant for statsmodels OLS
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)

            # Fit model on training data
            model = sm.OLS(y_train, X_train_const).fit()

            # Predict on training data
            y_train_pred = model.predict(X_train_const)
            train_r2 = r2_score(y_train, y_train_pred)

            # Predict on test data
            y_test_pred = model.predict(X_test_const)
            test_r2 = r2_score(y_test, y_test_pred)

            # Compute error metrics on test set
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)

            # Check for overfitting
            r2_gap = train_r2 - test_r2
            overfitting = r2_gap > OVERFITTING_THRESHOLD

            log(f"[CV] Train R²: {train_r2:.4f}")
            log(f"[CV] Test R²: {test_r2:.4f}")
            log(f"[CV] R² gap: {r2_gap:.4f} (threshold = {OVERFITTING_THRESHOLD:.2f})")
            log(f"[CV] RMSE: {rmse:.6f}")
            log(f"[CV] MAE: {mae:.6f}")
            log(f"[CV] Overfitting: {'YES' if overfitting else 'NO'}")

            # Store results
            cv_results.append({
                'fold': fold_idx,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': rmse,
                'mae': mae,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'overfitting_flag': overfitting
            })
        # Aggregate Cross-Validation Results

        log("[CV] Aggregating cross-validation results...")

        cv_df = pd.DataFrame(cv_results)

        # Compute summary statistics
        mean_train_r2 = cv_df['train_r2'].mean()
        mean_test_r2 = cv_df['test_r2'].mean()
        std_test_r2 = cv_df['test_r2'].std()
        mean_rmse = cv_df['rmse'].mean()
        mean_mae = cv_df['mae'].mean()
        n_overfitting_folds = cv_df['overfitting_flag'].sum()

        log(f"[CV] Mean train R²: {mean_train_r2:.4f}")
        log(f"[CV] Mean test R²: {mean_test_r2:.4f} (SD = {std_test_r2:.4f})")
        log(f"[CV] Mean RMSE: {mean_rmse:.6f}")
        log(f"[CV] Mean MAE: {mean_mae:.6f}")
        log(f"[CV] Folds with overfitting: {n_overfitting_folds} / {N_FOLDS}")
        # Save Cross-Validation Results

        log("Saving cross-validation results...")

        output_path = RQ_DIR / "data" / "step07_cross_validation.csv"
        cv_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step07_cross_validation.csv ({len(cv_df)} rows, {len(cv_df.columns)} cols)")
        # Validation

        log("Cross-validation validation checks:")

        # Check R² range
        r2_in_range = ((cv_df['train_r2'] >= 0) & (cv_df['train_r2'] <= 1)).all() and \
                      ((cv_df['test_r2'] >= 0) & (cv_df['test_r2'] <= 1)).all()
        log(f"R² values in [0, 1]: {'PASS' if r2_in_range else 'FAIL'}")

        # Check all folds completed
        all_folds_complete = len(cv_df) == N_FOLDS
        log(f"All {N_FOLDS} folds completed: {'PASS' if all_folds_complete else 'FAIL'}")

        # Check train R² >= test R² (expected pattern, but not required)
        train_geq_test = (cv_df['train_r2'] >= cv_df['test_r2']).sum()
        log(f"Folds with train R² >= test R²: {train_geq_test} / {N_FOLDS}")

        # Overfitting assessment
        if n_overfitting_folds > 0:
            log(f"WARNING: {n_overfitting_folds} folds show overfitting (R² gap > {OVERFITTING_THRESHOLD:.2f})")
        else:
            log(f"No overfitting detected across folds")

        validation_pass = r2_in_range and all_folds_complete
        log(f"Overall validation: {'PASS' if validation_pass else 'FAIL'}")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
