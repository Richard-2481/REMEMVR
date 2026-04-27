#!/usr/bin/env python3
"""
Step 07: Cross-validation for RQ 7.3.3.

This step runs 5-fold cross-validation to assess model generalizability
and detect overfitting.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
OUTPUT_CV_FILE = DATA_DIR / "step07_cross_validation.csv"
OUTPUT_SUMMARY_FILE = DATA_DIR / "step07_cv_summary.csv"
LOG_FILE = LOG_DIR / "step07_cross_validation.log"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Clear log
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")

# Main
if __name__ == "__main__":
    import traceback
    
    try:
        log("[START] Step 07: Cross-validation")
        
        # Load data
        log("[LOAD] Loading analysis dataset...")
        input_file = DATA_DIR / "step03_analysis_dataset.csv"
        df = pd.read_csv(input_file)
        log(f"[LOADED] {input_file.name} ({len(df)} rows, {len(df.columns)} cols)")
        
        # Extract features and target
        predictors = ['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        X = df[predictors].values
        y = df['hce_rate'].values
        
        log("[CV_SETUP] Setting up 5-fold cross-validation...")
        log(f"[CV_SETUP] Final dataset: {len(df)} observations, {len(predictors)} predictors")
        
        # Run CV manually using sklearn
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        log("[CV_RUN] Running cross-validation...")
        
        cv_results = []
        fold_num = 0
        
        for train_idx, test_idx in kf.split(X):
            fold_num += 1
            log(f"[CV_FOLD] Processing fold {fold_num}/5...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            log(f"[CV_FOLD] Fold {fold_num}: train={len(X_train)}, test={len(X_test)}")
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)
            gap = train_r2 - test_r2
            
            cv_results.append({
                'fold': fold_num,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': rmse,
                'mae': mae,
                'generalization_gap': gap,
                'overfitting': gap > 0.10
            })
            
            log(f"[CV_FOLD] Fold {fold_num} results:")
            log(f"[CV_FOLD]   Train R²: {train_r2:.4f}")
            log(f"[CV_FOLD]   Test R²: {test_r2:.4f}")
            log(f"[CV_FOLD]   RMSE: {rmse:.4f}")
            log(f"[CV_FOLD]   MAE: {mae:.4f}")
            log(f"[CV_FOLD]   Gap: {gap:.4f}")
        
        cv_df = pd.DataFrame(cv_results)
        
        log("[CV_RUN] Cross-validation complete: 5/5 folds successful")
        
        # Aggregate statistics
        log("[CV_AGGREGATE] Computing aggregated statistics...")
        
        summary_data = []
        for metric in ['train_r2', 'test_r2', 'rmse', 'mae', 'generalization_gap']:
            summary_data.append({
                'metric': metric,
                'mean': cv_df[metric].mean(),
                'std': cv_df[metric].std(),
                'min': cv_df[metric].min(),
                'max': cv_df[metric].max(),
                'overfitting_flag': cv_df['overfitting'].any() if metric == 'generalization_gap' else None
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        mean_gap = cv_df['generalization_gap'].mean()
        if mean_gap > 0.10:
            log("[CV_WARNING] OVERFITTING DETECTED: Gap > 0.10")
        
        # Save results
        log("[SAVE] Saving cross-validation results...")
        cv_df.to_csv(OUTPUT_CV_FILE, index=False)
        log(f"[SAVED] {OUTPUT_CV_FILE.name} ({len(cv_df)} rows, {len(cv_df.columns)} cols)")
        
        summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)
        log(f"[SAVED] {OUTPUT_SUMMARY_FILE.name} ({len(summary_df)} rows, {len(summary_df.columns)} cols)")
        
        # Summary
        log("[SUMMARY] Cross-validation results:")
        log(f"[SUMMARY] - Mean train R²: {cv_df['train_r2'].mean():.4f}")
        log(f"[SUMMARY] - Mean test R²: {cv_df['test_r2'].mean():.4f}")
        log(f"[SUMMARY] - Mean generalization gap: {mean_gap:.4f}")
        
        if cv_df['test_r2'].mean() < 0:
            log("[SUMMARY] - WARNING: Negative test R² indicates model performs worse than mean baseline")
        
        log("[SUCCESS] Step 07 complete")
        
    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        traceback.print_exc()
        sys.exit(1)