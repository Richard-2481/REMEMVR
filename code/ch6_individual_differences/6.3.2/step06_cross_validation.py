#!/usr/bin/env python3
"""Cross-validation: 5-fold cross-validation to assess generalizability and overfitting of the"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Any
import traceback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2
LOG_FILE = RQ_DIR / "logs" / "step06_cross_validation.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Custom Cross-validation Functions (Due to Signature Mismatch)

def cross_validate_regression_custom(X, y, n_folds=5, random_state=42, 
                                    scoring_metrics=['mse', 'r2', 'mae'], 
                                    overfitting_threshold=0.10):
    """
    Custom k-fold cross-validation for regression.
    
    Parameters match expected signature but use seed instead of random_state
    internally for sklearn compatibility.
    """
    
    # Initialize KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_results = []
    fold = 1
    
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Add constant for regression
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        try:
            # Fit model on training data
            model = sm.OLS(y_train, X_train_const).fit()
            
            # Predictions
            y_train_pred = model.predict(X_train_const)
            y_test_pred = model.predict(X_test_const)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # MAE (Mean Absolute Error)
            mae = mean_absolute_error(y_test, y_test_pred)
            
            cv_results.append({
                'fold': fold,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': rmse,
                'mae': mae,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })
            
        except Exception as e:
            log(f"Fold {fold} failed: {e}")
            cv_results.append({
                'fold': fold,
                'train_r2': np.nan,
                'test_r2': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })
        
        fold += 1
    
    cv_df = pd.DataFrame(cv_results)
    
    # Overfitting assessment
    train_r2_mean = cv_df['train_r2'].mean()
    test_r2_mean = cv_df['test_r2'].mean()
    r2_gap = train_r2_mean - test_r2_mean
    
    train_rmse_mean = cv_df['rmse'].mean()  # Note: Only test RMSE available
    test_rmse_mean = cv_df['rmse'].mean()
    
    train_mae_mean = cv_df['mae'].mean()  # Note: Only test MAE available
    test_mae_mean = cv_df['mae'].mean()
    
    overfitting_assessment = pd.DataFrame([
        {
            'metric': 'R²',
            'train_mean': train_r2_mean,
            'test_mean': test_r2_mean,
            'gap': r2_gap,
            'overfitting_flag': r2_gap > overfitting_threshold
        },
        {
            'metric': 'RMSE',
            'train_mean': np.nan,  # Not calculated in current implementation
            'test_mean': test_rmse_mean,
            'gap': np.nan,
            'overfitting_flag': False  # Cannot assess without train RMSE
        },
        {
            'metric': 'MAE',
            'train_mean': np.nan,  # Not calculated in current implementation  
            'test_mean': test_mae_mean,
            'gap': np.nan,
            'overfitting_flag': False  # Cannot assess without train MAE
        }
    ])
    
    return {
        'cv_results': cv_df,
        'overfitting_assessment': overfitting_assessment,
        'summary_stats': {
            'mean_test_r2': test_r2_mean,
            'std_test_r2': cv_df['test_r2'].std(),
            'mean_rmse': test_rmse_mean,
            'mean_mae': test_mae_mean,
            'overfitting_detected': r2_gap > overfitting_threshold
        }
    }

def validate_numeric_range_custom(df, range_checks):
    """
    Custom numeric range validation.
    
    Validates that specified columns fall within expected ranges.
    """
    validation_results = {}
    
    for column, (min_val, max_val) in range_checks.items():
        if column not in df.columns:
            validation_results[column] = {
                'valid': False,
                'message': f"Column '{column}' not found in data"
            }
            continue
        
        values = df[column].dropna()
        if len(values) == 0:
            validation_results[column] = {
                'valid': False,
                'message': f"No valid values in column '{column}'"
            }
            continue
        
        # Check range
        values_in_range = (values >= min_val) & (values <= max_val)
        n_valid = values_in_range.sum()
        n_total = len(values)
        
        proportion_valid = n_valid / n_total
        is_valid = proportion_valid >= 0.95  # 95% of values must be in range
        
        validation_results[column] = {
            'valid': is_valid,
            'message': f"{n_valid}/{n_total} values in range [{min_val}, {max_val}]",
            'min_observed': values.min(),
            'max_observed': values.max(),
            'proportion_valid': proportion_valid
        }
    
    return validation_results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Cross-validation")
        # Load Input Data

        log("Loading analysis dataset...")
        df = pd.read_csv(RQ_DIR / "data" / "step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns
        required_cols = ['calibration_quality', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age', 'sex', 'education']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"Dataset shape: {df.shape}")
        # Prepare Data for Cross-validation

        log("Preparing data for cross-validation...")
        
        predictor_vars = ['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        dependent_var = 'calibration_quality'
        
        # Extract relevant columns
        X = df[predictor_vars].copy()
        y = df[dependent_var].copy()
        
        # Handle categorical sex variable
        if 'sex' in X.columns and X['sex'].dtype == 'object':
            X = pd.get_dummies(X, columns=['sex'], drop_first=True)
            log("Converted sex to dummy variable")
        
        # Remove missing data
        analysis_data = pd.concat([X, y], axis=1).dropna()
        X_clean = analysis_data.iloc[:, :-1]
        y_clean = analysis_data.iloc[:, -1]
        
        log(f"Clean sample size: {len(y_clean)} participants")
        log(f"Predictors: {list(X_clean.columns)}")
        # Run Cross-validation

        log("Running 5-fold cross-validation...")
        
        cv_analysis = cross_validate_regression_custom(
            X_clean, 
            y_clean,
            n_folds=5,
            random_state=42,
            scoring_metrics=['mse', 'r2', 'mae'],
            overfitting_threshold=0.10  # 10% R² gap threshold
        )
        
        log("Cross-validation complete")
        
        # Report summary statistics
        summary = cv_analysis['summary_stats']
        log(f"Mean test R² = {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
        log(f"Mean RMSE = {summary['mean_rmse']:.4f}")
        log(f"Mean MAE = {summary['mean_mae']:.4f}")
        log(f"Overfitting detected: {summary['overfitting_detected']}")
        # Save Cross-validation Outputs
        # These outputs will be used by: Step 07 (power analysis using CV R²)

        log("Saving cross-validation results...")
        
        # Output: step06_cross_validation.csv
        # Contains: Cross-validation results by fold
        cv_analysis['cv_results'].to_csv(RQ_DIR / "data" / "step06_cross_validation.csv", index=False, encoding='utf-8')
        log(f"step06_cross_validation.csv ({len(cv_analysis['cv_results'])} rows, {len(cv_analysis['cv_results'].columns)} cols)")
        
        # Output: step06_overfitting_assessment.csv
        # Contains: Overfitting assessment and model stability
        cv_analysis['overfitting_assessment'].to_csv(RQ_DIR / "data" / "step06_overfitting_assessment.csv", index=False, encoding='utf-8')
        log(f"step06_overfitting_assessment.csv ({len(cv_analysis['overfitting_assessment'])} rows, {len(cv_analysis['overfitting_assessment'].columns)} cols)")
        # Run Validation Tool
        # Validates: Performance metrics fall within expected ranges
        # Criteria: train_r2 [0,1], test_r2 [0,1], rmse [0,10], mae [0,10]

        log("Running numeric range validation...")
        
        range_checks = {
            'train_r2': [0, 1],
            'test_r2': [0, 1],
            'rmse': [0, 10],
            'mae': [0, 10]
        }
        
        validation_result = validate_numeric_range_custom(cv_analysis['cv_results'], range_checks)

        # Report validation results
        all_valid = all(result['valid'] for result in validation_result.values())
        log(f"All metrics in expected ranges: {all_valid}")
        
        for metric, result in validation_result.items():
            if result['valid']:
                log(f"{metric}: PASS - {result['message']}")
            else:
                log(f"{metric}: FAIL - {result['message']}")
        # Model Stability Assessment

        log("Evaluating model stability and generalizability...")
        
        cv_results = cv_analysis['cv_results']
        
        # R² stability across folds
        r2_cv = cv_results['test_r2'].std() / cv_results['test_r2'].mean() if cv_results['test_r2'].mean() > 0 else np.inf
        r2_stability = "HIGH" if r2_cv < 0.2 else "MODERATE" if r2_cv < 0.5 else "LOW"
        
        # Consistent performance across folds
        min_r2 = cv_results['test_r2'].min()
        max_r2 = cv_results['test_r2'].max()
        r2_range = max_r2 - min_r2
        
        log(f"R² coefficient of variation: {r2_cv:.3f} ({r2_stability} stability)")
        log(f"R² range across folds: {min_r2:.3f} to {max_r2:.3f} (range = {r2_range:.3f})")
        
        # Overall assessment
        if summary['overfitting_detected']:
            log("Model shows signs of overfitting - interpret results with caution")
        elif r2_stability == "HIGH" and cv_results['test_r2'].mean() > 0.1:
            log("Model shows good generalizability and stability")
        elif r2_stability == "MODERATE":
            log("Model shows moderate stability - results are interpretable")
        else:
            log("Model shows poor stability - consider model revision")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)