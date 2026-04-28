#!/usr/bin/env python3
"""step04_compute_correlations: Compute r(BVMT, Where) and r(BVMT, What) with bootstrap 95% confidence intervals."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.bootstrap import bootstrap_correlation_ci

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_compute_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute correlations with bootstrap CIs")
        # Load Input Data

        log("Loading analysis dataset...")
        analysis_df = pd.read_csv(RQ_DIR / "data" / "step03_analysis_dataset.csv")
        log(f"step03_analysis_dataset.csv ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")
        
        # Verify we have the expected columns
        expected_cols = ["UID", "bvmt_total", "bvmt_pct_ret", "Where_mean", "What_mean"]
        actual_cols = analysis_df.columns.tolist()
        log(f"Expected columns: {expected_cols}")
        log(f"Actual columns: {actual_cols}")
        
        # Check all required columns are present
        missing_cols = [col for col in expected_cols if col not in actual_cols]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        log(f"All required columns present")
        log(f"Dataset summary: {len(analysis_df)} participants")
        # Run Analysis Tool - Bootstrap Correlations

        correlations_to_compute = [
            {
                'name': 'BVMT_Where',
                'x_var': 'bvmt_total',
                'y_var': 'Where_mean',
                'description': 'Correlation between visuospatial memory (total) and Where domain'
            },
            {
                'name': 'BVMT_What',
                'x_var': 'bvmt_total',
                'y_var': 'What_mean',
                'description': 'Correlation between visuospatial memory (total) and What domain'
            },
            {
                'name': 'BVMTret_Where',
                'x_var': 'bvmt_pct_ret',
                'y_var': 'Where_mean',
                'description': 'Correlation between visuospatial retention and Where domain'
            },
            {
                'name': 'BVMTret_What',
                'x_var': 'bvmt_pct_ret',
                'y_var': 'What_mean',
                'description': 'Correlation between visuospatial retention and What domain'
            }
        ]
        
        correlation_results = []
        
        for corr_info in correlations_to_compute:
            log(f"Computing {corr_info['name']} correlation...")
            log(f"Variables: {corr_info['x_var']} vs {corr_info['y_var']}")
            
            # Extract variables as numpy arrays
            x_data = analysis_df[corr_info['x_var']].values
            y_data = analysis_df[corr_info['y_var']].values
            
            # Remove any NaN values (paired deletion)
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            log(f"Clean data: {len(x_clean)} valid pairs ({len(x_data) - len(x_clean)} removed)")
            
            if len(x_clean) < 10:
                raise ValueError(f"Insufficient data for {corr_info['name']}: only {len(x_clean)} valid pairs")
            
            # Compute bootstrap correlation with confidence intervals
            bootstrap_result = bootstrap_correlation_ci(
                x=x_clean,
                y=y_clean,
                n_bootstrap=1000,  # Number of bootstrap iterations
                confidence=0.95,   # 95% confidence intervals
                method="pearson",  # Pearson correlation method
                seed=42           # For reproducibility
            )
            
            # Extract results - use 'r' key based on gcode_lessons.md
            correlation_coef = bootstrap_result['r']  # CRITICAL: Key is 'r', not 'correlation'
            ci_lower = bootstrap_result['ci_lower']
            ci_upper = bootstrap_result['ci_upper']
            standard_error = bootstrap_result['se']
            
            log(f"{corr_info['name']}: r = {correlation_coef:.4f}, CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Compute p-value using correlation test (uncorrected)
            # Note: This is for descriptive purposes; formal testing in Step 05
            t_stat = correlation_coef * np.sqrt((len(x_clean) - 2) / (1 - correlation_coef**2))
            p_uncorrected = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_clean) - 2))
            
            # Effect size classification (Cohen's conventions for correlations)
            abs_r = abs(correlation_coef)
            if abs_r < 0.1:
                effect_size = "negligible"
            elif abs_r < 0.3:
                effect_size = "small"
            elif abs_r < 0.5:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            # Basic assumption check - linearity via visual inspection proxy
            # Check if correlation is reasonable given the data
            assumption_check = "visual_inspection_needed"
            if abs(correlation_coef) > 0.95:
                assumption_check = "potential_outliers"
            elif abs(correlation_coef) < 0.001:
                assumption_check = "potential_nonlinearity"
            else:
                assumption_check = "reasonable"
            
            # Store results
            correlation_results.append({
                'correlation': corr_info['name'],
                'r': correlation_coef,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': len(x_clean),
                'p_uncorrected': p_uncorrected,
                'assumption_check': assumption_check,
                'effect_size': effect_size
            })
            
            log(f"{corr_info['name']} effect size: {effect_size} (|r| = {abs_r:.3f})")
        # Save Analysis Outputs
        # These outputs will be used by: Step 05 (Steiger's Z-test for difference)

        log(f"Saving step04_correlations.csv...")
        # Output: Bootstrap correlation results with confidence intervals
        # Contains: Correlation coefficients, CIs, sample sizes, descriptive p-values
        # Columns: correlation, r, ci_lower, ci_upper, n, p_uncorrected, assumption_check, effect_size
        correlation_df = pd.DataFrame(correlation_results)
        correlation_df.to_csv(RQ_DIR / "data" / "step04_correlations.csv", index=False, encoding='utf-8')
        log(f"step04_correlations.csv ({len(correlation_df)} rows, {len(correlation_df.columns)} cols)")
        
        # Log summary of results
        log("Correlation results:")
        for _, row in correlation_df.iterrows():
            log(f"  {row['correlation']}: r = {row['r']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], p = {row['p_uncorrected']:.4f}, n = {row['n']}")
        # Run Validation Tool
        # Validates: Correlation coefficients are in valid range [-1, 1]
        # Threshold: All correlations must be finite and within bounds

        log("Running validate_numeric_range...")
        
        # Validate correlation coefficients
        r_values = correlation_df['r'].values
        validation_result = validate_numeric_range(
            data=r_values,
            min_val=-1.0,  # Minimum possible correlation
            max_val=1.0,   # Maximum possible correlation
            column_name="correlation_coefficients"
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional validation checks
        validation_issues = []
        
        # Check that CIs contain point estimates
        for _, row in correlation_df.iterrows():
            if not (row['ci_lower'] <= row['r'] <= row['ci_upper']):
                validation_issues.append(f"{row['correlation']}: CI does not contain point estimate")
        
        # Check CI ordering
        for _, row in correlation_df.iterrows():
            if row['ci_lower'] >= row['ci_upper']:
                validation_issues.append(f"{row['correlation']}: Invalid CI bounds (lower >= upper)")
        
        # Check for reasonable sample sizes
        for _, row in correlation_df.iterrows():
            if row['n'] < 50:
                log(f"{row['correlation']}: Small sample size (n = {row['n']})")
        
        if validation_issues:
            for issue in validation_issues:
                log(f"[VALIDATION ERROR] {issue}")
            raise ValueError(f"Validation failed: {validation_issues}")
        else:
            log("All correlation results passed validation checks")

        log("Step 04 complete")
        log(f"Ready for Step 05: Steiger's Z-test to compare r(BVMT,Where) vs r(BVMT,What)")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)