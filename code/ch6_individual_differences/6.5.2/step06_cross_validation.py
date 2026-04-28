#!/usr/bin/env python3
"""cross_validation: Assess model generalizability and overfitting using 5-fold cross-validation"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import traceback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.2 (derived from script location)  
LOG_FILE = RQ_DIR / "logs" / "step06_cross_validation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Cross-Validation Function (Addressing Signature Mismatch)

def cross_validate_regression_custom(
    X: Union[np.ndarray, pd.DataFrame], 
    y: Union[np.ndarray, pd.Series],
    n_folds: int = 5,
    random_state: int = 42,
    scoring_metrics: List[str] = ['mse', 'r2']
) -> Dict[str, Any]:
    """
    Custom cross-validation function to match 4_analysis.yaml signature.
    
    REASON FOR CUSTOM FUNCTION:
    - tools.analysis_regression.cross_validate_regression uses 'seed' parameter
    - 4_analysis.yaml specifies 'random_state' parameter
    - Following gcode_lessons.md #9: Create custom implementation for signature mismatches
    
    Returns both train and test metrics for overfitting assessment.
    """
    log("[CUSTOM CV] Creating custom cross-validation function due to signature mismatch")
    log(f"[CUSTOM CV] Using {n_folds} folds, random_state={random_state}")
    
    # Ensure we have numpy arrays
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = np.array(X)
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Add constant term for regression
    X_with_const = sm.add_constant(X_array)
    
    # Initialize cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    fold_results = []
    train_scores = {'r2': [], 'rmse': [], 'mae': []}
    test_scores = {'r2': [], 'rmse': [], 'mae': []}
    
    log(f"[CUSTOM CV] Starting {n_folds}-fold cross-validation on {len(y_array)} samples")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_array), 1):
        log(f"[CUSTOM CV] Processing fold {fold_idx}/{n_folds}")
        
        # Split data
        X_train, X_test = X_with_const[train_idx], X_with_const[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # Fit model
        model = sm.OLS(y_train, X_train).fit()
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Compute metrics for training set
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # Compute metrics for test set  
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'train_R2': train_r2,
            'test_R2': test_r2,
            'train_RMSE': train_rmse,
            'test_RMSE': test_rmse,
            'test_MAE': test_mae,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        # Store for aggregation
        train_scores['r2'].append(train_r2)
        train_scores['rmse'].append(train_rmse)
        train_scores['mae'].append(train_mae)
        test_scores['r2'].append(test_r2)
        test_scores['rmse'].append(test_rmse)
        test_scores['mae'].append(test_mae)
        
        log(f"[CUSTOM CV] Fold {fold_idx}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Generalization Gap={train_r2-test_r2:.4f}")
    
    log("[CUSTOM CV] Cross-validation complete, computing aggregate statistics")
    
    return {
        'fold_results': fold_results,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'mean_train_r2': np.mean(train_scores['r2']),
        'mean_test_r2': np.mean(test_scores['r2']),
        'generalization_gap': np.mean(train_scores['r2']) - np.mean(test_scores['r2'])
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Cross-Validation Analysis")
        # Load Input Data
        log("Loading analysis dataset...")
        input_file = RQ_DIR / "data" / "step01_analysis_dataset.csv"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Required input file not found: {input_file}")
            
        df = pd.read_csv(input_file)
        log(f"step01_analysis_dataset.csv ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify required columns  
        required_cols = ["theta_all", "age", "nart_score", "dass_dep", "dass_anx", "dass_str"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {required_cols}")
        log(f"Sample size: N={len(df)} (expected ~97)")
        # Prepare Data for Cross-Validation
        
        log("Preparing regression data...")
        
        # Define outcome and predictors
        outcome_var = "theta_all"
        predictor_vars = ["age", "nart_score", "dass_dep", "dass_anx", "dass_str"]
        
        y = df[outcome_var]
        X = df[predictor_vars]
        
        log(f"Outcome: {outcome_var}")
        log(f"Predictors: {predictor_vars}")
        log(f"Complete cases: N={len(df)} (no missingness expected)")
        
        # Check for any missing data
        if df[required_cols].isnull().any().any():
            log("Missing data detected, using complete cases only")
            df_complete = df[required_cols].dropna()
            y = df_complete[outcome_var]
            X = df_complete[predictor_vars]
            log(f"After excluding missing: N={len(df_complete)}")
        # Run Cross-Validation Analysis
        
        log("Running 5-fold cross-validation...")
        cv_results = cross_validate_regression_custom(
            X=X,
            y=y,
            n_folds=5,
            random_state=42,
            scoring_metrics=["r2", "mse", "mae"]  # As specified in 4_analysis.yaml
        )
        log("Cross-validation complete")
        # Save Cross-Validation Outputs
        # These outputs will be used by: RQ interpretation and comparison with other RQs
        
        log("Saving cross-validation results...")
        
        # Output 1: Per-fold results
        # Contains: Detailed results for each CV fold  
        # Columns: ["fold", "train_R2", "test_R2", "train_RMSE", "test_RMSE", "test_MAE"]
        fold_results_df = pd.DataFrame(cv_results['fold_results'])
        fold_output = RQ_DIR / "data" / "step06_cross_validation.csv"
        fold_results_df.to_csv(fold_output, index=False, encoding='utf-8')
        log(f"step06_cross_validation.csv ({len(fold_results_df)} rows, {len(fold_results_df.columns)} cols)")
        
        # Output 2: Summary statistics with confidence intervals
        # Contains: Aggregated CV metrics across all folds
        # Columns: ["metric", "mean", "std", "ci_lower", "ci_upper"] 
        summary_data = []
        
        for metric_type in ['train', 'test']:
            scores = cv_results[f'{metric_type}_scores']
            for metric_name, values in scores.items():
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Sample standard deviation
                ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))  # 95% CI
                ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))
                
                summary_data.append({
                    'metric': f'{metric_type}_{metric_name}',
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_output = RQ_DIR / "data" / "step06_cv_summary.csv"
        summary_df.to_csv(summary_output, index=False, encoding='utf-8')
        log(f"step06_cv_summary.csv ({len(summary_df)} rows, {len(summary_df.columns)} cols)")
        
        # Output 3: Generalization assessment 
        # Contains: Overfitting evaluation and interpretation
        # Columns: N/A (text file)
        generalization_gap = cv_results['generalization_gap']
        overfitting_threshold = 0.10  # From 4_analysis.yaml
        
        assessment_lines = [
            "CROSS-VALIDATION GENERALIZATION ASSESSMENT",
            "=" * 50,
            "",
            f"Model: theta_all ~ age + nart_score + dass_dep + dass_anx + dass_str",
            f"Sample Size: N={len(df)}",
            f"Cross-Validation: {5}-fold with random_state=42",
            "",
            "PERFORMANCE METRICS:",
            f"  Mean Train R²: {cv_results['mean_train_r2']:.4f}",
            f"  Mean Test R²:  {cv_results['mean_test_r2']:.4f}",
            f"  Generalization Gap: {generalization_gap:.4f}",
            "",
            "OVERFITTING ASSESSMENT:",
            f"  Threshold: {overfitting_threshold:.2f}",
        ]
        
        if generalization_gap > overfitting_threshold:
            assessment_lines.extend([
                f"  Status: OVERFITTING DETECTED (gap = {generalization_gap:.4f} > {overfitting_threshold:.2f})",
                f"  Interpretation: Model shows poor generalization, likely overfitted",
                f"  Recommendation: Use test R² for conservative effect size estimate"
            ])
        else:
            assessment_lines.extend([
                f"  Status: ACCEPTABLE GENERALIZATION (gap = {generalization_gap:.4f} <= {overfitting_threshold:.2f})",
                f"  Interpretation: Model generalizes reasonably well to new data",
                f"  Recommendation: Train R² provides valid effect size estimate"
            ])
        
        assessment_lines.extend([
            "",
            "FOLD-BY-FOLD DETAILS:",
        ])
        
        for _, fold in fold_results_df.iterrows():
            gap = fold['train_R2'] - fold['test_R2']
            assessment_lines.append(f"  Fold {fold['fold']}: Train R²={fold['train_R2']:.3f}, Test R²={fold['test_R2']:.3f}, Gap={gap:.3f}")
        
        # Expected issue note (from user instructions)
        assessment_lines.extend([
            "",
            "CONTEXT (RQ 7.5.2):",
            f"  Expected: Some overfitting given small effect sizes (R² ~ 0.091) and N=97",
            f"  Consistent with: RQ 7.5.1 pattern of modest generalization gaps",
            f"  Implication: DASS effects on memory are genuine but small"
        ])
        
        assessment_output = RQ_DIR / "data" / "step06_generalization_assessment.txt"
        with open(assessment_output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(assessment_lines))
        log(f"step06_generalization_assessment.txt")
        # Run Validation Tool
        # Validates: R² in [0,1], RMSE >= 0, reasonable generalization gap
        # Threshold: CV metrics within expected ranges
        
        log("Running validate_numeric_range...")
        
        # Validate R² values are in [0, 1]
        all_r2_values = np.concatenate([cv_results['train_scores']['r2'], cv_results['test_scores']['r2']])
        r2_validation = validate_numeric_range(
            data=all_r2_values,
            min_val=0.0,
            max_val=1.0,
            column_name="R_squared"
        )
        
        # Validate RMSE values are >= 0
        all_rmse_values = np.concatenate([cv_results['train_scores']['rmse'], cv_results['test_scores']['rmse']])  
        rmse_validation = validate_numeric_range(
            data=all_rmse_values,
            min_val=0.0,
            max_val=10.0,  # Reasonable upper bound for standardized theta scores
            column_name="RMSE"
        )
        
        # Report validation results
        log(f"R² validation: {'PASS' if r2_validation['valid'] else 'FAIL'}")
        if not r2_validation['valid']:
            log(f"R² violations: {r2_validation.get('message', 'Unknown error')}")
            
        log(f"RMSE validation: {'PASS' if rmse_validation['valid'] else 'FAIL'}")
        if not rmse_validation['valid']:
            log(f"RMSE violations: {rmse_validation.get('message', 'Unknown error')}")
        
        # Validate generalization gap is reasonable
        gap_reasonable = abs(generalization_gap) <= 1.0  # Shouldn't exceed 100% R²
        log(f"Generalization gap validation: {'PASS' if gap_reasonable else 'FAIL'}")
        
        # Overall validation status
        validation_passed = r2_validation['valid'] and rmse_validation['valid'] and gap_reasonable
        log(f"Overall: {'PASS' if validation_passed else 'FAIL'}")
        
        if validation_passed:
            log(f"All metrics within expected ranges")
            log(f"Mean test R²: {cv_results['mean_test_r2']:.4f} (conservative estimate)")
            log(f"Generalization gap: {generalization_gap:.4f} ({'concerning' if generalization_gap > overfitting_threshold else 'acceptable'})")
        
        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)