#!/usr/bin/env python3
"""cross_validate_regression: Assess model generalizability using 5-fold cross-validation for all 3 DASS regression models"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import traceback

# Sklearn imports for cross-validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_cross_validate_regression.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# Custom Cross-Validation Implementation

def cross_validate_regression_custom(X: Union[np.ndarray, pd.DataFrame], 
                                    y: Union[np.ndarray, pd.Series], 
                                    n_folds: int = 5, 
                                    random_state: int = 42, 
                                    scoring_metrics: List[str] = ['mse', 'r2']) -> Dict[str, Any]:
    """
    Custom 5-fold cross-validation implementation with expected signature.
    
    Created due to signature mismatch in tools.analysis_regression.cross_validate_regression:
    - Expected: X, y, n_folds, random_state, scoring_metrics
    - Actual: X, y, n_folds, seed
    
    Returns metrics for train-test gap and overfitting assessment.
    """
    
    # Convert to numpy arrays for consistent handling
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
        
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Initialize KFold with fixed seed for reproducibility
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Track metrics across folds
    test_r2_scores = []
    train_r2_scores = []
    test_mse_scores = []
    test_mae_scores = []
    
    fold_num = 1
    for train_idx, test_idx in kf.split(X_array):
        log(f"  [CV] Processing fold {fold_num}/{n_folds}")
        
        # Split data
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # Add constant for intercept (using statsmodels convention)
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        # Fit model on training fold
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Predict on both training and test sets
        train_pred = model.predict(X_train_const)
        test_pred = model.predict(X_test_const)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Store metrics
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        test_mse_scores.append(test_mse)
        test_mae_scores.append(test_mae)
        
        log(f"    Fold {fold_num}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, RMSE={np.sqrt(test_mse):.3f}, MAE={test_mae:.3f}")
        fold_num += 1
    
    # Calculate summary statistics
    test_r2_mean = np.mean(test_r2_scores)
    test_r2_sd = np.std(test_r2_scores)
    train_r2_mean = np.mean(train_r2_scores)
    rmse_mean = np.sqrt(np.mean(test_mse_scores))
    mae_mean = np.mean(test_mae_scores)
    
    # Calculate train-test gap for overfitting assessment
    train_test_gap = train_r2_mean - test_r2_mean
    
    # Calculate shrinkage (proportion of predictive ability lost in CV)
    shrinkage = train_test_gap / train_r2_mean if train_r2_mean > 0 else 0
    
    return {
        'test_r2_scores': test_r2_scores,
        'train_r2_scores': train_r2_scores,
        'test_r2_mean': test_r2_mean,
        'test_r2_sd': test_r2_sd,
        'train_r2_mean': train_r2_mean,
        'rmse_mean': rmse_mean,
        'mae_mean': mae_mean,
        'train_test_gap': train_test_gap,
        'shrinkage': shrinkage,
        'n_folds': n_folds
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Cross-Validate Regression Models")
        # Load Input Data

        log("Loading analysis dataset...")
        analysis_dataset = pd.read_csv(RQ_DIR / "data/step02_analysis_dataset.csv")
        log(f"step02_analysis_dataset.csv ({len(analysis_dataset)} rows, {len(analysis_dataset.columns)} cols)")
        log(f"Available columns: {list(analysis_dataset.columns)}")
        
        # Extract predictors (all 3 DASS measures)
        predictor_cols = ['z_Dep', 'z_Anx', 'z_Str']
        X = analysis_dataset[predictor_cols]
        log(f"Predictors extracted: {predictor_cols}")
        
        # Extract outcome variables
        outcome_cols = ['theta_accuracy', 'confidence', 'calibration']
        log(f"Outcome variables: {outcome_cols}")
        # Run Cross-Validation for All 3 Models

        cv_results = []
        overfitting_threshold = 0.10  # Flag overfitting if train-test gap > 10%
        
        for outcome_name in outcome_cols:
            log(f"[CV] Running cross-validation for {outcome_name} model...")
            
            # Extract outcome variable
            y = analysis_dataset[outcome_name]
            
            # Remove any rows with missing values
            complete_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_complete = X[complete_mask]
            y_complete = y[complete_mask]
            
            log(f"[CV] Complete cases for {outcome_name}: {len(X_complete)} participants")
            
            # Run custom cross-validation
            cv_result = cross_validate_regression_custom(
                X=X_complete,
                y=y_complete,
                n_folds=5,
                random_state=42,
                scoring_metrics=['mse', 'r2', 'mae']
            )
            
            # Check for overfitting
            overfitting_flag = cv_result['train_test_gap'] > overfitting_threshold
            overfitting_status = "YES" if overfitting_flag else "NO"
            
            log(f"[CV] {outcome_name} results:")
            log(f"    Test R² Mean: {cv_result['test_r2_mean']:.3f} (SD: {cv_result['test_r2_sd']:.3f})")
            log(f"    Train R² Mean: {cv_result['train_r2_mean']:.3f}")
            log(f"    RMSE Mean: {cv_result['rmse_mean']:.3f}")
            log(f"    MAE Mean: {cv_result['mae_mean']:.3f}")
            log(f"    Train-Test Gap: {cv_result['train_test_gap']:.3f}")
            log(f"    Shrinkage: {cv_result['shrinkage']:.3f}")
            log(f"    Overfitting (gap > {overfitting_threshold}): {overfitting_status}")
            
            # Store results
            cv_results.append({
                'model': outcome_name,
                'test_r2_mean': cv_result['test_r2_mean'],
                'test_r2_sd': cv_result['test_r2_sd'],
                'rmse_mean': cv_result['rmse_mean'],
                'mae_mean': cv_result['mae_mean'],
                'train_test_gap': cv_result['train_test_gap'],
                'shrinkage': cv_result['shrinkage'],
                'overfitting_flag': overfitting_status
            })
        # Save Cross-Validation Results
        # These outputs will be used by: Step 6 (effect sizes) and Step 7 (synthesis)

        log("Saving cross-validation results...")
        cv_df = pd.DataFrame(cv_results)
        output_path = RQ_DIR / "data/step05_cross_validation.csv"
        cv_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step05_cross_validation.csv ({len(cv_df)} rows, {len(cv_df.columns)} cols)")
        
        # Log summary statistics
        log(f"Cross-validation completed for {len(cv_results)} models:")
        for result in cv_results:
            log(f"  {result['model']}: Test R² = {result['test_r2_mean']:.3f}, Gap = {result['train_test_gap']:.3f}, Overfitting = {result['overfitting_flag']}")
        # Run Validation Tool
        # Validates: Output dataframe structure and reasonable metric ranges

        log("Running validate_dataframe_structure...")
        validation_result = validate_dataframe_structure(
            df=cv_df,
            expected_rows=(3, 3),  # Exactly 3 models
            expected_columns=['model', 'test_r2_mean', 'test_r2_sd', 'rmse_mean', 'mae_mean', 'train_test_gap', 'shrinkage', 'overfitting_flag'],
            column_types={
                'model': [str, object],  # Must be list of types
                'test_r2_mean': [float, np.float64],
                'test_r2_sd': [float, np.float64],
                'rmse_mean': [float, np.float64],
                'mae_mean': [float, np.float64],
                'train_test_gap': [float, np.float64],
                'shrinkage': [float, np.float64],
                'overfitting_flag': [str, object]
            }
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional domain-specific validation
        log("Checking CV metrics reasonableness...")
        
        # Check R² values are in reasonable range
        r2_values = cv_df['test_r2_mean'].values
        if all(0.0 <= r2 <= 1.0 for r2 in r2_values):
            log("All R² values in valid range [0.0, 1.0]")
        else:
            log("Some R² values outside valid range")
        
        # Check for any extreme overfitting
        overfitting_count = sum(1 for flag in cv_df['overfitting_flag'] if flag == "YES")
        log(f"Models flagged for overfitting: {overfitting_count}/3")
        
        # Check shrinkage values are reasonable
        shrinkage_values = cv_df['shrinkage'].values
        if all(-0.5 <= s <= 1.0 for s in shrinkage_values):
            log("All shrinkage values in reasonable range [-0.5, 1.0]")
        else:
            log("Some shrinkage values outside reasonable range")

        log("Step 05 complete - Cross-validation analysis finished")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)