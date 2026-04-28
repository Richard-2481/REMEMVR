#!/usr/bin/env python3
"""
Step 07: Cross-Validation Analysis for RQ 7.3.1
Assess model generalizability using k-fold cross-validation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared from predictions."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

try:
    log("Step 07: Cross-Validation Analysis")
    log("Purpose: Assess model generalizability using 5-fold CV")
    
    # Load data
    log("Loading analysis dataset...")
    df = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")
    log(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Prepare data
    predictors = ['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
    X = df[predictors].values
    y = df['confidence_theta'].values
    
    log(f"Sample size: N={len(df)}")
    log(f"Predictors: {predictors}")
    log(f"Outcome: confidence_theta (M={y.mean():.3f}, SD={y.std():.3f})")
    
    # Initialize cross-validation
    n_folds = 5
    random_state = 42
    log(f"[CV] Initializing {n_folds}-fold cross-validation (seed={random_state})")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    cv_results = []
    
    # Perform cross-validation
    log("[CV] Starting cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        log(f"[FOLD {fold_idx}] Processing fold {fold_idx}/{n_folds}...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Add constant for intercept
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        # Fit model on training data
        try:
            model = sm.OLS(y_train, X_train_const)
            results = model.fit()
            
            # Predictions on training set
            y_train_pred = results.predict(X_train_const)
            train_r2 = results.rsquared
            
            # Predictions on test set
            y_test_pred = results.predict(X_test_const)
            test_r2 = calculate_r_squared(y_test, y_test_pred)
            
            # Error metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)
            
            # Store results
            cv_results.append({
                'fold': fold_idx,
                'train_R2': train_r2,
                'test_R2': test_r2,
                'rmse': rmse,
                'mae': mae
            })
            
            log(f"[FOLD {fold_idx}] Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
            log(f"[FOLD {fold_idx}] RMSE={rmse:.4f}, MAE={mae:.4f}")
            
        except Exception as e:
            log(f"Fold {fold_idx} failed: {str(e)}")
            cv_results.append({
                'fold': fold_idx,
                'train_R2': np.nan,
                'test_R2': np.nan,
                'rmse': np.nan,
                'mae': np.nan
            })
    
    # Create results DataFrame
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate summary statistics
    log("Cross-validation results:")
    mean_train_r2 = cv_df['train_R2'].mean()
    std_train_r2 = cv_df['train_R2'].std()
    mean_test_r2 = cv_df['test_R2'].mean()
    std_test_r2 = cv_df['test_R2'].std()
    mean_rmse = cv_df['rmse'].mean()
    mean_mae = cv_df['mae'].mean()
    
    log(f"Train R²: {mean_train_r2:.4f} ± {std_train_r2:.4f}")
    log(f"Test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    log(f"Mean RMSE: {mean_rmse:.4f}")
    log(f"Mean MAE: {mean_mae:.4f}")
    
    # Check for overfitting
    r2_gap = mean_train_r2 - mean_test_r2
    log(f"Train-Test R² gap: {r2_gap:.4f}")
    
    if r2_gap > 0.10:
        log("Potential overfitting detected (gap > 0.10)")
    elif r2_gap > 0.15:
        log("Substantial overfitting detected (gap > 0.15)")
    else:
        log("Model generalizes well (gap < 0.10)")
    
    # Check for outlier folds
    log("Checking for outlier folds...")
    test_r2_mean = cv_df['test_R2'].mean()
    test_r2_std = cv_df['test_R2'].std()
    
    for idx, row in cv_df.iterrows():
        z_score = abs((row['test_R2'] - test_r2_mean) / test_r2_std) if test_r2_std > 0 else 0
        if z_score > 2:
            log(f"Fold {row['fold']} is an outlier (Z={z_score:.2f})")
    
    # Save results
    output_path = RQ_DIR / "data" / "step07_cross_validation.csv"
    cv_df.to_csv(output_path, index=False)
    log(f"Cross-validation results: {output_path}")
    
    # Validation check
    log("Checking cross-validation quality...")
    all_folds_complete = not cv_df['test_R2'].isna().any()
    cv_r2_reasonable = 0 <= mean_test_r2 <= 1
    within_tolerance = r2_gap <= 0.15  # Within 15% as specified
    
    if all_folds_complete and cv_r2_reasonable and within_tolerance:
        log("Cross-validation PASSED all criteria")
    else:
        log("Some criteria not met:")
        if not all_folds_complete:
            log("  - Some folds failed to complete")
        if not cv_r2_reasonable:
            log("  - Test R² outside valid range")
        if not within_tolerance:
            log("  - Train-test gap exceeds 15% tolerance")
    
    log("Step 07 complete")
    
except Exception as e:
    log(f"Critical error in cross-validation: {str(e)}")
    import traceback
    log(f"{traceback.format_exc()}")
    raise
