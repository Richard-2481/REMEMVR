#!/usr/bin/env python3
"""bootstrap_sensitivity_analysis: Bootstrap confidence intervals for correlation difference as sensitivity analysis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_bootstrap_sensitivity_analysis.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Bootstrap Analysis Functions

def bootstrap_correlation_difference(data: pd.DataFrame, predictor_col: str = 'ravlt_total',
                                   n_bootstrap: int = 1000,
                                   seed: int = 42, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for correlation difference.

    For each bootstrap sample:
    - Compute r1_boot = correlation(predictor, FreeRecall)
    - Compute r2_boot = correlation(predictor, Recognition)
    - Compute diff_boot = r1_boot - r2_boot

    Returns 95% CI for correlation difference and tests if it excludes zero.
    """
    np.random.seed(seed)
    log(f"Starting {n_bootstrap} iterations with seed={seed}, predictor={predictor_col}")

    n_obs = len(data)
    bootstrap_differences = []

    # Extract variables
    ravlt = data[predictor_col].values
    free_recall = data['theta_free_recall'].values
    recognition = data['theta_recognition'].values
    
    # Compute original correlation difference
    r1_orig, _ = pearsonr(ravlt, free_recall)
    r2_orig, _ = pearsonr(ravlt, recognition)
    diff_orig = r1_orig - r2_orig
    
    log(f"r(RAVLT, FreeRecall) = {r1_orig:.4f}")
    log(f"r(RAVLT, Recognition) = {r2_orig:.4f}")
    log(f"Difference = {diff_orig:.4f}")
    
    # Bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_obs, size=n_obs, replace=True)
        
        ravlt_boot = ravlt[bootstrap_indices]
        free_recall_boot = free_recall[bootstrap_indices]
        recognition_boot = recognition[bootstrap_indices]
        
        # Compute correlations for this bootstrap sample
        try:
            r1_boot, _ = pearsonr(ravlt_boot, free_recall_boot)
            r2_boot, _ = pearsonr(ravlt_boot, recognition_boot)
            diff_boot = r1_boot - r2_boot
            bootstrap_differences.append(diff_boot)
        except Exception as e:
            log(f"Bootstrap iteration {i} failed: {e}")
            continue
            
        if (i + 1) % 100 == 0:
            log(f"Completed {i + 1}/{n_bootstrap} bootstrap iterations")
    
    bootstrap_differences = np.array(bootstrap_differences)
    n_successful = len(bootstrap_differences)
    
    if n_successful < n_bootstrap:
        log(f"Only {n_successful}/{n_bootstrap} bootstrap iterations successful")
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_differences, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_differences, 100 * (1 - alpha / 2))
    
    # Check if CI excludes zero
    excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    
    # Compute bootstrap mean (should be close to original difference)
    bootstrap_mean = np.mean(bootstrap_differences)
    
    log(f"Mean difference = {bootstrap_mean:.4f}")
    log(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    log(f"Excludes zero: {excludes_zero}")
    
    return {
        'original_difference': diff_orig,
        'bootstrap_mean': bootstrap_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'excludes_zero': excludes_zero,
        'n_bootstrap': n_successful,
        'seed': seed,
        'confidence_level': confidence_level,
        'bootstrap_differences': bootstrap_differences
    }

def validate_bootstrap_results_custom(bootstrap_results: Dict[str, Any], 
                                    n_bootstrap: int, confidence_level: float) -> Dict[str, Any]:
    """
    Custom validation for bootstrap results (since tools.validation.validate_bootstrap_stability 
    is designed for clustering Jaccard coefficients, not correlation bootstrap).
    """
    valid = True
    messages = []
    
    # Check bootstrap parameters
    if bootstrap_results['n_bootstrap'] != n_bootstrap:
        valid = False
        messages.append(f"Expected {n_bootstrap} bootstrap iterations, got {bootstrap_results['n_bootstrap']}")
    
    if bootstrap_results['confidence_level'] != confidence_level:
        valid = False
        messages.append(f"Expected confidence level {confidence_level}, got {bootstrap_results['confidence_level']}")
    
    # Check CI validity (ci_lower < ci_upper)
    if bootstrap_results['ci_lower'] >= bootstrap_results['ci_upper']:
        valid = False
        messages.append("Invalid confidence interval: ci_lower >= ci_upper")
    
    # Check bootstrap mean is within CI (sanity check)
    bootstrap_mean = bootstrap_results['bootstrap_mean']
    ci_lower = bootstrap_results['ci_lower']
    ci_upper = bootstrap_results['ci_upper']
    
    if not (ci_lower <= bootstrap_mean <= ci_upper):
        messages.append("Warning: bootstrap mean outside CI (unusual but possible)")
    
    # Check bootstrap differences distribution
    bootstrap_diffs = bootstrap_results['bootstrap_differences']
    if len(bootstrap_diffs) < 0.9 * n_bootstrap:  # Allow up to 10% failure rate
        valid = False
        messages.append(f"Too few successful bootstrap iterations: {len(bootstrap_diffs)}/{n_bootstrap}")
    
    if valid and not messages:
        messages.append("Bootstrap results validation passed")
    
    return {
        'valid': valid,
        'message': '; '.join(messages),
        'n_successful_bootstrap': len(bootstrap_diffs),
        'ci_width': ci_upper - ci_lower
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: bootstrap_sensitivity_analysis")
        # Load Input Data

        log("Loading correlation input data...")
        input_path = RQ_DIR / "data" / "step02_correlation_input.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        correlation_data = pd.read_csv(input_path)
        log(f"{input_path.name} ({len(correlation_data)} rows, {len(correlation_data.columns)} cols)")
        
        # Validate required columns
        required_cols = ['ravlt_total', 'ravlt_pct_ret', 'theta_free_recall', 'theta_recognition']
        missing_cols = [col for col in required_cols if col not in correlation_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"All required columns present: {required_cols}")
        # Run Bootstrap Analysis for Each Predictor

        bootstrap_params = {
            'n_bootstrap': 1000,
            'seed': 42,
            'confidence_level': 0.95
        }

        predictor_cols = ['ravlt_total', 'ravlt_pct_ret']
        all_output_rows = []

        for pred_col in predictor_cols:
            log(f"\nRunning bootstrap correlation difference for {pred_col}...")

            bootstrap_results = bootstrap_correlation_difference(
                data=correlation_data,
                predictor_col=pred_col,
                **bootstrap_params
            )
            log(f"Bootstrap analysis complete for {pred_col}")

            # Log key results
            log(f"{pred_col} correlation difference: {bootstrap_results['bootstrap_mean']:.4f}")
            log(f"{pred_col} 95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
            log(f"{pred_col} Excludes zero: {bootstrap_results['excludes_zero']}")

            all_output_rows.append({
                'predictor': pred_col,
                'statistic': 'correlation_difference',
                'value': bootstrap_results['bootstrap_mean'],
                'ci_lower': bootstrap_results['ci_lower'],
                'ci_upper': bootstrap_results['ci_upper'],
                'excludes_zero': bootstrap_results['excludes_zero'],
                'n_bootstrap': bootstrap_results['n_bootstrap'],
                'seed': bootstrap_results['seed']
            })

            # Run validation for this predictor
            log(f"Running custom bootstrap validation for {pred_col}...")
            validation_result = validate_bootstrap_results_custom(
                bootstrap_results=bootstrap_results,
                n_bootstrap=bootstrap_params['n_bootstrap'],
                confidence_level=bootstrap_params['confidence_level']
            )

            if isinstance(validation_result, dict):
                for key, value in validation_result.items():
                    log(f"{pred_col} {key}: {value}")
            else:
                log(f"{pred_col}: {validation_result}")

            # Final consistency check
            if bootstrap_results['excludes_zero']:
                log(f"{pred_col}: Bootstrap CI excludes zero - supports process-specificity hypothesis")
            else:
                log(f"{pred_col}: Bootstrap CI includes zero - does not support process-specificity hypothesis")
        # Save Analysis Outputs
        log("\nSaving bootstrap sensitivity results...")

        output_df = pd.DataFrame(all_output_rows)
        output_path = RQ_DIR / "data" / "step05_bootstrap_sensitivity.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(output_df)} rows, {len(output_df.columns)} cols)")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)