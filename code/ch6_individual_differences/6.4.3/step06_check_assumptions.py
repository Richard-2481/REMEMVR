#!/usr/bin/env python3
"""step06_check_assumptions: Validate assumptions for correlation analysis and identify potential outliers"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis tools
from scipy.stats import shapiro, zscore, chi2
from scipy.spatial.distance import mahalanobis

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_check_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Check Assumptions")
        # Load Input Data

        log("Loading input data...")
        
        # Load RPM scores
        df_rpm = pd.read_csv(RQ_DIR / "data/step01_rpm_scores.csv")
        log(f"step01_rpm_scores.csv ({len(df_rpm)} rows, {len(df_rpm.columns)} cols)")
        
        # Load overall theta scores
        df_overall = pd.read_csv(RQ_DIR / "data/step02_overall_theta.csv")
        log(f"step02_overall_theta.csv ({len(df_overall)} rows, {len(df_overall.columns)} cols)")
        
        # Load What theta scores
        df_what = pd.read_csv(RQ_DIR / "data/step03_what_theta.csv")
        log(f"step03_what_theta.csv ({len(df_what)} rows, {len(df_what.columns)} cols)")
        
        # Merge data (inner join to keep complete cases only)
        df_merged = df_rpm.merge(df_overall, on='UID').merge(df_what, on='UID')
        log(f"Combined dataset ({len(df_merged)} rows with complete data)")
        # Run Analysis Tool

        variables = ['rpm_score', 'theta_overall', 'theta_what']
        alpha = 0.05
        outlier_threshold = 3.29  # Conservative z-score cutoff (p < 0.001)
        
        log("Running assumption checks...")
        
        results = []
        
        # Test each variable for normality and outliers
        for var in variables:
            log(f"Checking assumptions for {var}...")
            
            # Shapiro-Wilk normality test
            # scipy.stats.shapiro(x) -> Tuple[float, float]
            stat, p_value = shapiro(df_merged[var])
            log(f"{var}: statistic={stat:.4f}, p={p_value:.6f}")
            
            # Univariate outlier detection (|z| > 3.29)
            z_scores = np.abs(zscore(df_merged[var]))
            outliers = np.sum(z_scores > outlier_threshold)
            log(f"{var}: {outliers} outliers found (|z| > {outlier_threshold})")
            
            # Determine if assumption met and remedial action
            assumption_met = p_value > alpha and outliers <= 5  # Conservative threshold
            
            if not assumption_met:
                if p_value <= alpha:
                    remedial_action = "Use bootstrap CIs (already computed)"
                else:
                    remedial_action = "Consider outlier removal for sensitivity analysis"
            else:
                remedial_action = "No action needed"
            
            results.append({
                'variable': var,
                'normality_p': p_value,
                'outlier_count': outliers,
                'assumption_met': assumption_met,
                'remedial_action': remedial_action
            })
            
            log(f"{var}: assumption_met={assumption_met}, action={remedial_action}")
        
        # Multivariate outlier detection (Mahalanobis distance)
        log("Checking multivariate outliers...")
        
        data_matrix = df_merged[variables].values
        mean_vec = np.mean(data_matrix, axis=0)
        cov_matrix = np.cov(data_matrix.T)
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            mahal_distances = [mahalanobis(row, mean_vec, inv_cov) for row in data_matrix]
            
            # Chi-square critical value for 3 variables at p < 0.001
            critical_value = chi2.ppf(0.999, df=len(variables))
            multivariate_outliers = np.sum(np.array(mahal_distances) > critical_value)
            
            log(f"{multivariate_outliers} multivariate outliers found (critical={critical_value:.3f})")
            
            results.append({
                'variable': 'multivariate',
                'normality_p': np.nan,
                'outlier_count': multivariate_outliers,
                'assumption_met': multivariate_outliers <= 3,  # Conservative threshold
                'remedial_action': "Report results with/without outliers" if multivariate_outliers > 3 else "No action needed"
            })
            
        except np.linalg.LinAlgError:
            log("Could not compute Mahalanobis distance - singular covariance matrix")
            results.append({
                'variable': 'multivariate',
                'normality_p': np.nan,
                'outlier_count': np.nan,
                'assumption_met': False,
                'remedial_action': "Could not compute - singular covariance matrix"
            })

        log("Analysis complete")
        # Save Analysis Outputs
        # These outputs will be used by: Downstream interpretation and sensitivity analysis

        log("Saving assumption_checks.csv...")
        
        # Output: step06_assumption_checks.csv
        # Contains: Assumption test results and remedial action recommendations
        # Columns: variable, normality_p, outlier_count, assumption_met, remedial_action
        assumption_results = pd.DataFrame(results)
        assumption_results.to_csv(RQ_DIR / "data/step06_assumption_checks.csv", index=False, encoding='utf-8')
        log(f"step06_assumption_checks.csv ({len(assumption_results)} rows, {len(assumption_results.columns)} cols)")
        # Run Validation Tool
        # Validates: Output format has required columns
        # Threshold: All expected columns must be present

        log("Running validate_data_format...")
        
        # Expected columns for assumption checks output
        expected_columns = ['variable', 'normality_p', 'outlier_count', 'assumption_met', 'remedial_action']
        
        validation_result = validate_data_format(
            df=assumption_results,
            required_cols=expected_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)