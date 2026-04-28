#!/usr/bin/env python3
"""Cross-Validation Assessment: Implement 5-fold cross-validation to assess model generalizability for both"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Standard library imports
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_cross_validation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom Cross-Validation Function

def cross_validate_regression_custom(X, y, n_folds=5, random_state=42, scoring_metrics=['mse', 'r2', 'mae']):
    """
    Custom cross-validation implementation to match expected signature.
    
    Per Ch7 lessons learned: When function signatures don't match 4_analysis.yaml,
    write custom implementation instead of using mismatched tool function.
    
    Args:
        X: Predictor variables (DataFrame or array)
        y: Response variable (Series or array) 
        n_folds: Number of CV folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        scoring_metrics: List of metrics to compute (default: ['mse', 'r2', 'mae'])
    
    Returns:
        Dict with keys: scores (per-fold metrics), mean_scores, std_scores, predictions
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Initialize metric storage
    fold_scores = {metric: [] for metric in scoring_metrics}
    all_predictions = []
    all_actuals = []
    train_scores = []
    test_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        # Split data for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model with constant term (intercept)
        X_train_const = sm.add_constant(X_train, has_constant='add')
        X_test_const = sm.add_constant(X_test, has_constant='add')
        
        # Use statsmodels for consistency with other steps
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Predictions
        y_pred = model.predict(X_test_const)
        
        # Store predictions for overall evaluation
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        
        # Compute fold metrics
        if 'mse' in scoring_metrics:
            mse = mean_squared_error(y_test, y_pred)
            fold_scores['mse'].append(mse)
            
        if 'r2' in scoring_metrics:
            r2 = r2_score(y_test, y_pred)
            fold_scores['r2'].append(r2)
            test_scores.append(r2)
            
        if 'mae' in scoring_metrics:
            mae = mean_absolute_error(y_test, y_pred)
            fold_scores['mae'].append(mae)
        
        # Training R² for overfitting detection
        y_train_pred = model.predict(X_train_const)
        train_r2 = r2_score(y_train, y_train_pred)
        train_scores.append(train_r2)
        
        log(f"[CV] Fold {fold_idx+1}: R² = {r2:.4f}, RMSE = {np.sqrt(mse):.4f}, MAE = {mae:.4f}")
    
    # Aggregate results
    mean_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}
    std_scores = {metric: np.std(scores) for metric, scores in fold_scores.items()}
    
    # Overall performance
    overall_r2 = r2_score(all_actuals, all_predictions)
    
    return {
        'scores': fold_scores,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'predictions': all_predictions,
        'actuals': all_actuals,
        'overall_r2': overall_r2,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'mean_train_r2': np.mean(train_scores),
        'mean_test_r2': np.mean(test_scores)
    }

def bootstrap_cv_confidence_intervals(X, y, n_folds=5, n_bootstrap=1000, random_state=42, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for cross-validation metrics.
    
    Args:
        X: Predictor variables
        y: Response variable
        n_folds: Number of CV folds
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed
        confidence_level: Confidence level for intervals
        
    Returns:
        Dict with confidence intervals for R², RMSE, MAE
    """
    np.random.seed(random_state)
    
    bootstrap_r2 = []
    bootstrap_rmse = []
    bootstrap_mae = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[boot_indices].reset_index(drop=True)
        y_boot = y.iloc[boot_indices].reset_index(drop=True)
        
        # Cross-validate on bootstrap sample
        cv_results = cross_validate_regression_custom(
            X_boot, y_boot, n_folds=n_folds, 
            random_state=random_state + i  # Different seed for each bootstrap
        )
        
        bootstrap_r2.append(cv_results['mean_scores']['r2'])
        bootstrap_rmse.append(np.sqrt(cv_results['mean_scores']['mse']))
        bootstrap_mae.append(cv_results['mean_scores']['mae'])
    
    # Confidence intervals using percentile method
    alpha = (1 - confidence_level) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100
    
    return {
        'r2_ci': (np.percentile(bootstrap_r2, lower_pct), np.percentile(bootstrap_r2, upper_pct)),
        'rmse_ci': (np.percentile(bootstrap_rmse, lower_pct), np.percentile(bootstrap_rmse, upper_pct)),
        'mae_ci': (np.percentile(bootstrap_mae, lower_pct), np.percentile(bootstrap_mae, upper_pct))
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Cross-Validation Assessment")
        # Load Input Data

        log("Loading analysis dataset...")
        input_path = RQ_DIR / "data" / "step01_analysis_dataset.csv"
        df = pd.read_csv(input_path)
        log(f"step01_analysis_dataset.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['theta_all', 'Age_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std', 'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract variables for analysis
        y = df['theta_all']  # Outcome variable: VR performance (theta scores)
        
        # Model 1: Age only
        X_model1 = df[['Age_std']]
        
        # Model 2: Age + cognitive tests  
        X_model2 = df[['Age_std', 'RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std', 'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']]
        
        log(f"y: {len(y)} observations")
        log(f"Model 1 predictors: {list(X_model1.columns)}")
        log(f"Model 2 predictors: {list(X_model2.columns)}")
        # Run Cross-Validation for Both Models

        log("Running 5-fold cross-validation...")
        
        # Model 1: Age only
        log("[CV] Model 1: Age only")
        cv_results_model1 = cross_validate_regression_custom(
            X=X_model1,
            y=y,
            n_folds=5,
            random_state=42,
            scoring_metrics=['mse', 'r2', 'mae']
        )
        
        # Model 2: Age + cognitive tests
        log("[CV] Model 2: Age + cognitive tests")
        cv_results_model2 = cross_validate_regression_custom(
            X=X_model2,
            y=y,
            n_folds=5,
            random_state=42,
            scoring_metrics=['mse', 'r2', 'mae']
        )
        
        log("Cross-validation complete for both models")
        # Overfitting Detection
        # Criteria: Train-test R² gap should be < 0.10
        # Flag overfitting if gap > 0.10 for either model
        
        log("Overfitting detection...")
        
        overfitting_threshold = 0.10
        
        # Model 1 overfitting check
        model1_gap = cv_results_model1['mean_train_r2'] - cv_results_model1['mean_test_r2']
        model1_overfitting = model1_gap > overfitting_threshold
        
        # Model 2 overfitting check  
        model2_gap = cv_results_model2['mean_train_r2'] - cv_results_model2['mean_test_r2']
        model2_overfitting = model2_gap > overfitting_threshold
        
        log(f"Model 1 gap: {model1_gap:.4f}, Flag: {model1_overfitting}")
        log(f"Model 2 gap: {model2_gap:.4f}, Flag: {model2_overfitting}")
        # Bootstrap Confidence Intervals
        
        log("Computing confidence intervals for CV metrics...")
        
        # Bootstrap CIs for Model 1
        log("Model 1...")
        ci_model1 = bootstrap_cv_confidence_intervals(
            X=X_model1, y=y, n_folds=5, n_bootstrap=1000, random_state=42
        )
        
        # Bootstrap CIs for Model 2
        log("Model 2...")
        ci_model2 = bootstrap_cv_confidence_intervals(
            X=X_model2, y=y, n_folds=5, n_bootstrap=1000, random_state=42
        )
        
        log("Bootstrap confidence intervals computed")
        # Save Cross-Validation Results
        # These outputs will be used by: Step 6 (effect sizes), Step 9 (summary), plotting pipeline (visualization)
        
        log("Saving cross-validation results...")
        
        # Prepare results dataframe
        results_data = []
        
        # Model 1 results
        results_data.append({
            'model': 'Model_1_Age_Only',
            'cv_R2_mean': cv_results_model1['mean_scores']['r2'],
            'cv_R2_sd': cv_results_model1['std_scores']['r2'], 
            'cv_RMSE_mean': np.sqrt(cv_results_model1['mean_scores']['mse']),
            'cv_RMSE_sd': np.sqrt(cv_results_model1['std_scores']['mse']),
            'cv_MAE_mean': cv_results_model1['mean_scores']['mae'],
            'cv_MAE_sd': cv_results_model1['std_scores']['mae'],
            'train_R2': cv_results_model1['mean_train_r2'],
            'test_R2': cv_results_model1['mean_test_r2'],
            'overfitting_gap': model1_gap,
            'overfitting_flag': model1_overfitting,
            'ci_R2_lower': ci_model1['r2_ci'][0],
            'ci_R2_upper': ci_model1['r2_ci'][1]
        })
        
        # Model 2 results
        results_data.append({
            'model': 'Model_2_Age_Plus_Cognitive',
            'cv_R2_mean': cv_results_model2['mean_scores']['r2'],
            'cv_R2_sd': cv_results_model2['std_scores']['r2'],
            'cv_RMSE_mean': np.sqrt(cv_results_model2['mean_scores']['mse']),
            'cv_RMSE_sd': np.sqrt(cv_results_model2['std_scores']['mse']),
            'cv_MAE_mean': cv_results_model2['mean_scores']['mae'],
            'cv_MAE_sd': cv_results_model2['std_scores']['mae'],
            'train_R2': cv_results_model2['mean_train_r2'],
            'test_R2': cv_results_model2['mean_test_r2'],
            'overfitting_gap': model2_gap,
            'overfitting_flag': model2_overfitting,
            'ci_R2_lower': ci_model2['r2_ci'][0],
            'ci_R2_upper': ci_model2['r2_ci'][1]
        })
        
        # Create DataFrame and save
        cv_results_df = pd.DataFrame(results_data)
        output_path = RQ_DIR / "data" / "step05_cross_validation.csv"
        cv_results_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step05_cross_validation.csv ({len(cv_results_df)} rows, {len(cv_results_df.columns)} cols)")
        # Run Validation Tool
        # Validates: R² values are in valid range [0.0, 1.0]
        # Threshold: Standard bounds for coefficient of determination
        
        log("Running validate_numeric_range...")
        
        # Validate R² values are in valid range
        r2_values = np.array([
            cv_results_model1['mean_scores']['r2'],
            cv_results_model2['mean_scores']['r2'],
            cv_results_model1['overall_r2'],
            cv_results_model2['overall_r2']
        ])
        
        validation_result = validate_numeric_range(
            data=r2_values,
            min_val=0.0,
            max_val=1.0,
            column_name='R2_values'
        )
        
        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
            
        # Additional validation: Check for negative R² (indicates very poor fit)
        negative_r2 = any(r2 < 0 for r2 in r2_values)
        if negative_r2:
            log(f"Negative R² detected - indicates model worse than baseline")
        
        # Log summary statistics
        log(f"Model 1 CV R²: {cv_results_model1['mean_scores']['r2']:.4f} ± {cv_results_model1['std_scores']['r2']:.4f}")
        log(f"Model 2 CV R²: {cv_results_model2['mean_scores']['r2']:.4f} ± {cv_results_model2['std_scores']['r2']:.4f}")
        log(f"Model 1 Overfitting: {model1_overfitting} (gap: {model1_gap:.4f})")
        log(f"Model 2 Overfitting: {model2_overfitting} (gap: {model2_gap:.4f})")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)