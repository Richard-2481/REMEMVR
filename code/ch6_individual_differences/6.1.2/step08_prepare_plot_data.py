#!/usr/bin/env python3
"""prepare_plot_data: Prepare plot-ready data files by aggregating results from regression analysis steps 3-7."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: prepare_plot_data")
        # Load Input Data from Previous Steps

        log("Loading regression analysis results...")
        
        # Load step 3: Intercept model predictions
        intercept_results = pd.read_csv(RQ_DIR / "data" / "step03_intercept_predictions.csv")
        log(f"step03_intercept_predictions.csv ({len(intercept_results)} rows, {len(intercept_results.columns)} cols)")
        
        # Load step 4: Slope model predictions  
        slope_results = pd.read_csv(RQ_DIR / "data" / "step04_slope_predictions.csv")
        log(f"step04_slope_predictions.csv ({len(slope_results)} rows, {len(slope_results.columns)} cols)")
        
        # Load step 5: R² comparison with bootstrap CI
        r_squared_results = pd.read_csv(RQ_DIR / "data" / "step05_r_squared_comparison.csv")
        log(f"step05_r_squared_comparison.csv ({len(r_squared_results)} rows, {len(r_squared_results.columns)} cols)")
        
        # Load step 6: Predictor significance testing
        significance_results = pd.read_csv(RQ_DIR / "data" / "step06_predictor_significance.csv")
        log(f"step06_predictor_significance.csv ({len(significance_results)} rows, {len(significance_results.columns)} cols)")
        
        # Load step 7: Model diagnostics
        diagnostics_results = pd.read_csv(RQ_DIR / "data" / "step07_model_diagnostics.csv")
        log(f"step07_model_diagnostics.csv ({len(diagnostics_results)} rows, {len(diagnostics_results.columns)} cols)")
        # Create Intercept vs Slope Comparison Data

        log("Creating intercept vs slope comparison dataset...")
        
        # Extract R² data for both models (intercept and slope)
        intercept_r2_row = r_squared_results[r_squared_results['model'] == 'intercept'].iloc[0]
        slope_r2_row = r_squared_results[r_squared_results['model'] == 'slope'].iloc[0]
        difference_row = r_squared_results[r_squared_results['model'] == 'difference'].iloc[0]
        
        # Get significant predictors for each model from step 6 results
        # Filter to significant predictors (excluding intercept term)
        intercept_sig = significance_results[
            (significance_results['outcome'] == 'intercept') & 
            (significance_results['predictor'] != 'intercept') &
            (significance_results['sig_bonferroni'] == 1)
        ]
        slope_sig = significance_results[
            (significance_results['outcome'] == 'slope') & 
            (significance_results['predictor'] != 'intercept') &
            (significance_results['sig_bonferroni'] == 1)
        ]
        
        # Create significant predictors summary
        intercept_sig_list = intercept_sig['predictor'].tolist() if not intercept_sig.empty else []
        slope_sig_list = slope_sig['predictor'].tolist() if not slope_sig.empty else []
        
        # Get key coefficients (excluding intercept term for interpretation)
        intercept_coefs = intercept_results[intercept_results['predictor'] != 'intercept'][['predictor', 'beta']].to_dict('records')
        slope_coefs = slope_results[slope_results['predictor'] != 'intercept'][['predictor', 'beta']].to_dict('records')
        
        # Build comparison dataset
        comparison_data = []
        
        # Intercept model row
        comparison_data.append({
            'model': 'intercept',
            'r_squared': intercept_r2_row['r_squared'],
            'r_squared_ci_lower': intercept_r2_row['bootstrap_ci_lower'],
            'r_squared_ci_upper': intercept_r2_row['bootstrap_ci_upper'],
            'adj_r_squared': intercept_r2_row['adj_r_squared'],
            'difference': np.nan,  # Not applicable for individual model
            'difference_ci_lower': np.nan,
            'difference_ci_upper': np.nan,
            'difference_p_value': np.nan,
            'significant_predictors': ';'.join(intercept_sig_list) if intercept_sig_list else 'none',
            'key_coefficients': ';'.join([f"{c['predictor']}:{c['beta']:.4f}" for c in intercept_coefs])
        })
        
        # Slope model row
        comparison_data.append({
            'model': 'slope',
            'r_squared': slope_r2_row['r_squared'],
            'r_squared_ci_lower': slope_r2_row['bootstrap_ci_lower'],
            'r_squared_ci_upper': slope_r2_row['bootstrap_ci_upper'],
            'adj_r_squared': slope_r2_row['adj_r_squared'],
            'difference': difference_row['difference'],
            'difference_ci_lower': difference_row['difference_ci_lower'],
            'difference_ci_upper': difference_row['difference_ci_upper'], 
            'difference_p_value': difference_row['p_value'],
            'significant_predictors': ';'.join(slope_sig_list) if slope_sig_list else 'none',
            'key_coefficients': ';'.join([f"{c['predictor']}:{c['beta']:.4f}" for c in slope_coefs])
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        log("Intercept vs slope comparison data created")
        # Create Regression Diagnostics Data

        log("Creating regression diagnostics dataset...")
        
        # Create formatted diagnostics dataset
        diagnostics_plot_data = []
        
        for _, row in diagnostics_results.iterrows():
            # Categorize test types for better plotting organization
            test_name = row['test']
            if 'VIF' in test_name:
                test_type = 'Multicollinearity'
                formatted_test = f"VIF {test_name.split('_')[-1]}"
            elif 'Cooks' in test_name:
                test_type = 'Outliers'
                formatted_test = "Cook's Distance"
            elif 'Durbin' in test_name:
                test_type = 'Autocorrelation'
                formatted_test = "Durbin-Watson"
            elif 'Shapiro' in test_name:
                test_type = 'Normality'
                formatted_test = "Shapiro-Wilk"
            else:
                test_type = 'Other'
                formatted_test = test_name
            
            # Create status interpretation
            if row['assumption_met'] == 1:
                status = 'PASS'
            else:
                status = 'FAIL'
            
            diagnostics_plot_data.append({
                'model': row['model'],
                'test_type': test_type,
                'test_name': formatted_test,
                'statistic': row['statistic'],
                'p_value': row.get('p_value', np.nan),  # Some tests may not have p-values
                'assumption_met': row['assumption_met'],
                'status': status,
                'remedial_action': row['remedial_action']
            })
        
        diagnostics_df = pd.DataFrame(diagnostics_plot_data)
        log("Regression diagnostics data created")
        # Save Plot Data Files
        # These outputs will be used by: plotting pipeline for visualization

        log("Saving plot data files...")
        
        # Save intercept vs slope comparison data
        # Output: plots/intercept_vs_slope_comparison_data.csv
        # Contains: R² values, confidence intervals, significant predictors, key coefficients
        comparison_output_path = RQ_DIR / "plots" / "intercept_vs_slope_comparison_data.csv"
        comparison_df.to_csv(comparison_output_path, index=False, encoding='utf-8')
        log(f"intercept_vs_slope_comparison_data.csv ({len(comparison_df)} rows, {len(comparison_df.columns)} cols)")
        
        # Save regression diagnostics data
        # Output: plots/regression_diagnostics_data.csv  
        # Contains: Diagnostic test results, assumption status, remedial actions
        diagnostics_output_path = RQ_DIR / "plots" / "regression_diagnostics_data.csv"
        diagnostics_df.to_csv(diagnostics_output_path, index=False, encoding='utf-8')
        log(f"regression_diagnostics_data.csv ({len(diagnostics_df)} rows, {len(diagnostics_df.columns)} cols)")
        # Run Validation
        # Validates: Output files have expected column structure
        # Threshold: All required columns present

        log("Running validate_data_columns on outputs...")
        
        # Validate comparison data columns
        comparison_required_cols = [
            'model', 'r_squared', 'r_squared_ci_lower', 'r_squared_ci_upper', 
            'adj_r_squared', 'difference', 'difference_ci_lower', 'difference_ci_upper',
            'difference_p_value', 'significant_predictors', 'key_coefficients'
        ]
        validation_result = validate_data_columns(comparison_df, comparison_required_cols)
        
        if validation_result.get('valid', False):
            log("Comparison data columns: PASS")
        else:
            log(f"Comparison data columns: FAIL - {validation_result}")
            raise ValueError(f"Comparison data validation failed: {validation_result}")
        
        # Validate diagnostics data columns  
        diagnostics_required_cols = [
            'model', 'test_type', 'test_name', 'statistic', 'p_value',
            'assumption_met', 'status', 'remedial_action'
        ]
        validation_result = validate_data_columns(diagnostics_df, diagnostics_required_cols)
        
        if validation_result.get('valid', False):
            log("Diagnostics data columns: PASS")
        else:
            log(f"Diagnostics data columns: FAIL - {validation_result}")
            raise ValueError(f"Diagnostics data validation failed: {validation_result}")

        log("Step 08 complete - Plot data prepared")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)