#!/usr/bin/env python3
"""Cross-validation: 5-fold cross-validation to assess model generalizability and detect overfitting."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Statistical packages
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Cross-Validation Functions

def fit_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Fit model on training data and evaluate on test data."""
    try:
        # Add constant for intercept
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        
        # Fit model
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Predictions
        y_train_pred = model.predict(X_train_const)
        y_test_pred = model.predict(X_test_const)
        
        # Training metrics
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test metrics
        test_r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        
        # Generalization gap
        generalization_gap = train_r2 - test_r2
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'generalization_gap': generalization_gap,
            'success': True
        }
        
    except Exception as e:
        log(f"Model fitting failed: {str(e)}")
        return {
            'train_r2': np.nan,
            'test_r2': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'generalization_gap': np.nan,
            'success': False
        }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Cross-validation")
        # Load Input Data

        log("Loading analysis dataset...")
        analysis_data = pd.read_csv(RQ_DIR / "data/step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_data)} rows, {len(analysis_data.columns)} cols)")
        # Set Up Cross-Validation

        log("Setting up 5-fold cross-validation...")
        
        # Prepare predictors and outcome
        # Model formula from step04: hce_rate ~ age_c + sex + education + ravlt_c + bvmt_c + rpm_c
        X_vars = ['age_c', 'sex', 'education', 'ravlt_c', 'bvmt_c', 'rpm_c', 'ravlt_pct_ret_c', 'bvmt_pct_ret_c']
        
        # Check for missing values
        missing_data = analysis_data[X_vars + ['hce_rate']].isnull().any()
        if missing_data.any():
            log("Missing data detected:")
            for var in missing_data[missing_data].index:
                n_missing = analysis_data[var].isnull().sum()
                log(f"{var}: {n_missing} missing values")
            # Drop rows with missing data
            analysis_clean = analysis_data[X_vars + ['hce_rate']].dropna()
            log(f"Dropped {len(analysis_data) - len(analysis_clean)} rows with missing data")
        else:
            analysis_clean = analysis_data[X_vars + ['hce_rate']].copy()
            log("No missing data detected")
        
        X = analysis_clean[X_vars]
        y = analysis_clean['hce_rate']
        
        log(f"Final dataset: {len(X)} observations, {len(X_vars)} predictors")
        
        # Set up cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        log("Cross-validation configured: 5 folds, shuffle=True, random_state=42")
        # Perform Cross-Validation

        log("Running cross-validation...")
        
        cv_results = []
        fold_num = 1
        
        for train_idx, test_idx in cv.split(X):
            log(f"Processing fold {fold_num}/5...")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            log(f"Fold {fold_num}: train={len(X_train)}, test={len(X_test)}")
            
            # Fit and evaluate
            fold_result = fit_and_evaluate_model(X_train, y_train, X_test, y_test)
            
            if fold_result['success']:
                fold_result['fold'] = fold_num
                cv_results.append(fold_result)
                
                log(f"Fold {fold_num} results:")
                log(f"Train R²: {fold_result['train_r2']:.4f}")
                log(f"Test R²: {fold_result['test_r2']:.4f}")
                log(f"RMSE: {fold_result['rmse']:.4f}")
                log(f"MAE: {fold_result['mae']:.4f}")
                log(f"Gap: {fold_result['generalization_gap']:.4f}")
            else:
                log(f"Fold {fold_num} FAILED - skipping")
            
            fold_num += 1
        
        if len(cv_results) == 0:
            raise ValueError("All CV folds failed - cannot proceed")
        
        log(f"Cross-validation complete: {len(cv_results)}/5 folds successful")
        # Aggregate Cross-Validation Results

        log("Computing aggregated statistics...")
        
        cv_df = pd.DataFrame(cv_results)
        
        # Calculate summary statistics
        metrics = ['train_r2', 'test_r2', 'rmse', 'mae', 'generalization_gap']
        summary_stats = []
        
        for metric in metrics:
            values = cv_df[metric].values
            
            # Basic statistics
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Overfitting detection
            if metric == 'generalization_gap':
                # Gap > 0.10 indicates overfitting concern
                overfitting_flag = mean_val > 0.10
                warning_flag = mean_val > 0.05
            else:
                overfitting_flag = False
                warning_flag = False
            
            summary_stats.append({
                'metric': metric,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'overfitting_flag': overfitting_flag
            })
            
            log(f"{metric}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
            if metric == 'generalization_gap':
                if overfitting_flag:
                    log(f"OVERFITTING DETECTED: Gap > 0.10")
                elif warning_flag:
                    log(f"Generalization concern: Gap > 0.05")
        # Overfitting Analysis

        log("Analyzing overfitting indicators...")
        
        # Overall overfitting assessment
        mean_gap = np.mean(cv_df['generalization_gap'])
        mean_test_r2 = np.mean(cv_df['test_r2'])
        
        # Overfitting criteria
        severe_overfitting = mean_gap > 0.10
        moderate_overfitting = 0.05 < mean_gap <= 0.10
        good_generalization = mean_gap <= 0.05
        
        # Performance criteria
        poor_performance = mean_test_r2 < 0.02  # Very weak predictive power
        weak_performance = 0.02 <= mean_test_r2 < 0.05
        moderate_performance = 0.05 <= mean_test_r2 < 0.10
        good_performance = mean_test_r2 >= 0.10
        
        # Combined assessment
        if severe_overfitting:
            overall_assessment = "OVERFITTING_SEVERE"
        elif moderate_overfitting:
            overall_assessment = "OVERFITTING_MODERATE"
        elif poor_performance:
            overall_assessment = "POOR_PREDICTIVE_POWER"
        elif weak_performance:
            overall_assessment = "WEAK_PREDICTIVE_POWER"
        else:
            overall_assessment = "ADEQUATE_GENERALIZATION"
        
        log(f"Overall assessment: {overall_assessment}")
        log(f"Mean test R²: {mean_test_r2:.4f}")
        log(f"Mean generalization gap: {mean_gap:.4f}")
        # Save Cross-Validation Results
        # These outputs document model generalizability and overfitting assessment

        log("Saving cross-validation results...")
        
        # Save fold-level results
        cv_df.to_csv(RQ_DIR / "data/step07_cross_validation.csv", index=False, encoding='utf-8')
        log(f"step07_cross_validation.csv ({len(cv_df)} rows, {len(cv_df.columns)} cols)")

        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats)
        
        # Add overall assessment
        assessment_row = {
            'metric': 'overall_assessment',
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'overfitting_flag': severe_overfitting or moderate_overfitting
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([assessment_row])], ignore_index=True)
        
        summary_df.to_csv(RQ_DIR / "data/step07_cv_summary.csv", index=False, encoding='utf-8')
        log(f"step07_cv_summary.csv ({len(summary_df)} rows, {len(summary_df.columns)} cols)")
        # Run Validation Tool
        # Validates: R² values are in valid range [0,1]
        # Threshold: All R² values should be between 0 and 1

        log("Running validate_numeric_range...")
        
        # Validate R² values
        r2_values = list(cv_df['train_r2']) + list(cv_df['test_r2'])

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        elif moderate_overfitting:
            log("- Recommendation: Monitor for overfitting in final model")
        elif poor_performance:
            log("- Recommendation: Cognitive predictors show minimal HCE prediction")
        else:
            log("- Recommendation: Model shows adequate generalization")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)Step 07 complete
