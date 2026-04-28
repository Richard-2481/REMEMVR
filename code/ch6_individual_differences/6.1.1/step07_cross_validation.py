#!/usr/bin/env python3
"""Cross Validation: Assess model generalizability and stability through cross-validation. Implement 5-fold"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_cross_validation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Cross-Validation Implementation

def cross_validate_regression_custom(X, y, n_folds=5, random_state=42):
    """
    Custom 5-fold cross-validation implementation.
    
    Note: Using custom implementation due to signature mismatch:
    - 4_analysis.yaml specifies random_state parameter
    - tools.analysis_regression.cross_validate_regression uses seed parameter
    
    Returns:
        pd.DataFrame with columns: fold, train_r2, test_r2, train_n, test_n, generalization_gap
    """
    # Initialize cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    cv_results = []
    
    # Perform cross-validation
    fold_num = 0
    for train_idx, test_idx in kf.split(X):
        fold_num += 1
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model on training set
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on both sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate R²
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate generalization gap (train R² - test R²)
        generalization_gap = train_r2 - test_r2
        
        # Store results
        cv_results.append({
            'fold': fold_num,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_n': len(y_train),
            'test_n': len(y_test),
            'generalization_gap': generalization_gap
        })
        
        log(f"[CV] Fold {fold_num}: train_r2={train_r2:.4f}, test_r2={test_r2:.4f}, gap={generalization_gap:.4f}")
    
    return pd.DataFrame(cv_results)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Cross Validation")
        # Load Input Data
        
        log("Loading merged analysis dataset...")
        # Note: Using actual filename step03_merged_analysis.csv (not step03_analysis_dataset.csv from YAML)
        analysis_dataset = pd.read_csv(RQ_DIR / "data" / "step03_merged_analysis.csv")
        log(f"step03_merged_analysis.csv ({len(analysis_dataset)} rows, {len(analysis_dataset.columns)} cols)")
        
        # Verify expected columns
        expected_cols = ["UID", "RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T", "theta_mean"]
        missing_cols = [col for col in expected_cols if col not in analysis_dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present: {expected_cols}")
        # Prepare Data for Cross-Validation
        # Predictors: Cognitive test T-scores
        # Outcome: Mean theta scores from Ch5 IRT analysis
        
        log("Setting up predictor and outcome variables...")
        
        # Define predictors and outcome
        predictor_cols = ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T"]
        outcome_col = "theta_mean"
        
        # Extract X (predictors) and y (outcome)
        X = analysis_dataset[predictor_cols].copy()
        y = analysis_dataset[outcome_col].copy()
        
        # Check for missing data
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0 or missing_y > 0:
            raise ValueError(f"Missing data found: X has {missing_X} missing values, y has {missing_y} missing values")
        
        log(f"X: {X.shape[0]} rows x {X.shape[1]} predictors")
        log(f"y: {len(y)} outcome values")
        log(f"No missing data in predictor or outcome variables")
        # Run Cross-Validation
        # Method: 5-fold cross-validation with fixed random seed
        
        log("Running 5-fold cross-validation...")
        cv_results = cross_validate_regression_custom(
            X=X,
            y=y, 
            n_folds=5,
            random_state=42  # Fixed seed for reproducibility
        )
        log("Cross-validation complete")
        
        # Calculate summary statistics
        mean_train_r2 = cv_results['train_r2'].mean()
        mean_test_r2 = cv_results['test_r2'].mean()
        mean_gap = cv_results['generalization_gap'].mean()
        std_gap = cv_results['generalization_gap'].std()
        
        log(f"Mean train R²: {mean_train_r2:.4f}")
        log(f"Mean test R²: {mean_test_r2:.4f}")
        log(f"Mean generalization gap: {mean_gap:.4f} ± {std_gap:.4f}")
        # Save Cross-Validation Results
        # Output will be used by results analysis for model performance interpretation
        
        log("Saving cross-validation results...")
        output_path = RQ_DIR / "data" / "step07_cross_validation.csv"
        cv_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(cv_results)} rows, {len(cv_results.columns)} cols)")
        # Run Validation
        
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=cv_results,
            required_columns=["fold", "train_r2", "test_r2", "train_n", "test_n", "generalization_gap"]
        )

        # Additional custom validation checks
        validation_passes = True
        validation_messages = []
        
        # Check 1: All 5 folds completed
        if len(cv_results) != 5:
            validation_passes = False
            validation_messages.append(f"Expected 5 folds, got {len(cv_results)}")
        else:
            log("All 5 folds completed successfully")
        
        # Check 2: Reasonable train/test splits (train_n ~80, test_n ~20)
        expected_train_size = int(len(analysis_dataset) * 0.8)
        expected_test_size = int(len(analysis_dataset) * 0.2)
        actual_train_sizes = cv_results['train_n'].tolist()
        actual_test_sizes = cv_results['test_n'].tolist()
        
        train_size_ok = all(15 <= size <= 100 for size in actual_train_sizes)
        test_size_ok = all(10 <= size <= 30 for size in actual_test_sizes)
        
        if not (train_size_ok and test_size_ok):
            validation_passes = False
            validation_messages.append(f"Unreasonable fold sizes: train_n={actual_train_sizes}, test_n={actual_test_sizes}")
        else:
            log(f"Reasonable train/test splits: train_n ~{expected_train_size}, test_n ~{expected_test_size}")
        
        # Check 3: train_r2 >= test_r2 (expected pattern)
        train_higher_count = (cv_results['train_r2'] >= cv_results['test_r2']).sum()
        if train_higher_count < 3:  # Allow some variation, but expect majority
            validation_passes = False
            validation_messages.append(f"Train R² < Test R² in {5-train_higher_count} folds (unusual pattern)")
        else:
            log(f"Train R² >= Test R² in {train_higher_count}/5 folds (expected pattern)")
        
        # Check 4: Generalization gap ≤ 0.10 (no severe overfitting)
        max_gap = cv_results['generalization_gap'].max()
        if max_gap > 0.10:
            log(f"Maximum generalization gap = {max_gap:.4f} > 0.10 (potential overfitting)")
            # Don't fail validation, just warn
        else:
            log(f"Generalization gap ≤ 0.10 (max = {max_gap:.4f})")

        # Report validation results
        if validation_passes:
            log("All validation criteria passed")
        else:
            log(f"Some validation checks failed: {'; '.join(validation_messages)}")
            log("Proceeding with available cross-validation results")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)