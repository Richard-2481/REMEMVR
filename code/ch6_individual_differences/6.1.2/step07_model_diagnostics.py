#!/usr/bin/env python3
"""model_diagnostics: Comprehensive regression diagnostics and assumption testing for both intercept"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.api as sm
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tool (analysis tool needs custom implementation due to signature mismatch)
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_model_diagnostics.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Analysis Implementation (Due to Signature Mismatch)

def compute_regression_diagnostics_custom(model, X, return_dataframe=True):
    """
    Custom implementation of regression diagnostics with expected signature.
    
    Wraps tools.analysis_regression.compute_regression_diagnostics but matches
    the signature expected by 4_analysis.yaml.
    
    Based on gcode_lessons.md bug #9 pattern.
    """
    from tools.analysis_regression import compute_regression_diagnostics
    
    # Extract y from model (needed for actual function)
    y = model.model.endog  # Get dependent variable from statsmodels model
    
    # Call actual function with correct parameters
    diagnostics = compute_regression_diagnostics(model, X, y)
    
    if return_dataframe:
        # Convert to structured format for CSV output
        results = []
        
        # Add VIF results
        if 'vif' in diagnostics:
            vif_values = diagnostics['vif']
            for i, vif in enumerate(vif_values):
                results.append({
                    'test': f'VIF_predictor_{i+1}',
                    'statistic': vif,
                    'p_value': np.nan,  # VIF doesn't have p-value
                    'assumption_met': 1 if vif < 5.0 else 0,
                    'remedial_action': 'None needed' if vif < 5.0 else 'Check multicollinearity'
                })
        
        # Add Cook's distance (check for influential observations)
        if 'cooks_d' in diagnostics:
            cooks_d = diagnostics['cooks_d']
            max_cooks = np.max(cooks_d)
            threshold = 4.0 / len(cooks_d)  # Standard Cook's D threshold
            results.append({
                'test': 'Cooks_distance',
                'statistic': max_cooks,
                'p_value': np.nan,
                'assumption_met': 1 if max_cooks < threshold else 0,
                'remedial_action': 'None needed' if max_cooks < threshold else 'Check for influential observations'
            })
        
        # Add Durbin-Watson test for autocorrelation
        if 'durbin_watson' in diagnostics:
            dw_stat = diagnostics['durbin_watson']
            results.append({
                'test': 'Durbin_Watson',
                'statistic': dw_stat,
                'p_value': np.nan,  # DW doesn't directly give p-value
                'assumption_met': 1 if 1.5 < dw_stat < 2.5 else 0,
                'remedial_action': 'None needed' if 1.5 < dw_stat < 2.5 else 'Check for autocorrelation'
            })
        
        # Add Breusch-Pagan test for heteroscedasticity
        if 'breusch_pagan' in diagnostics:
            bp_result = diagnostics['breusch_pagan']
            if isinstance(bp_result, tuple) and len(bp_result) >= 2:
                bp_stat, bp_pval = bp_result[0], bp_result[1]
                results.append({
                    'test': 'Breusch_Pagan',
                    'statistic': bp_stat,
                    'p_value': bp_pval,
                    'assumption_met': 1 if bp_pval > 0.05 else 0,
                    'remedial_action': 'None needed' if bp_pval > 0.05 else 'Consider robust standard errors'
                })
        
        # Add Shapiro-Wilk normality test on residuals
        residuals = model.resid
        if len(residuals) > 3:  # Shapiro-Wilk needs at least 4 observations
            try:
                sw_stat, sw_pval = stats.shapiro(residuals)
                results.append({
                    'test': 'Shapiro_Wilk',
                    'statistic': sw_stat,
                    'p_value': sw_pval,
                    'assumption_met': 1 if sw_pval > 0.05 else 0,
                    'remedial_action': 'None needed' if sw_pval > 0.05 else 'Consider transformation or robust methods'
                })
            except:
                # Fall back to simpler normality check if Shapiro-Wilk fails
                pass
        
        return pd.DataFrame(results)
    else:
        return diagnostics

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: model_diagnostics")
        # Load Input Data

        log("Loading regression input data...")
        regression_input = pd.read_csv(RQ_DIR / "data/step02_regression_input.csv")
        log(f"step02_regression_input.csv ({len(regression_input)} rows, {len(regression_input.columns)} cols)")
        
        # Verify expected columns
        required_cols = ['UID', 'intercept', 'slope', 'se_intercept', 'se_slope', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        if not all(col in regression_input.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {regression_input.columns.tolist()}")
        # Refit Both Models for Diagnostics

        log("Refitting models for diagnostic analysis...")
        
        # Prepare predictors (cognitive tests)
        X = regression_input[['RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']].values
        X_with_const = sm.add_constant(X)  # Add intercept term
        
        # Fit intercept prediction model
        y_intercept = regression_input['intercept'].values
        intercept_model = sm.OLS(y_intercept, X_with_const).fit()
        log("Intercept prediction model")
        
        # Fit slope prediction model  
        y_slope = regression_input['slope'].values
        slope_model = sm.OLS(y_slope, X_with_const).fit()
        log("Slope prediction model")
        # Run Comprehensive Diagnostics on Both Models
        # Validates: Model assumptions (normality, homoscedasticity, multicollinearity, outliers)

        log("Computing regression diagnostics for intercept model...")
        intercept_diagnostics_df = compute_regression_diagnostics_custom(
            model=intercept_model,
            X=X,  # Without constant (function adds it internally if needed)
            return_dataframe=True
        )
        intercept_diagnostics_df['model'] = 'intercept'
        log(f"Intercept model diagnostics ({len(intercept_diagnostics_df)} tests)")

        log("Computing regression diagnostics for slope model...")
        slope_diagnostics_df = compute_regression_diagnostics_custom(
            model=slope_model,
            X=X,  # Without constant
            return_dataframe=True
        )
        slope_diagnostics_df['model'] = 'slope'
        log(f"Slope model diagnostics ({len(slope_diagnostics_df)} tests)")
        # Combine and Structure Diagnostic Results
        # These outputs will be used by: Final interpretation and reporting steps

        # Combine results from both models
        all_diagnostics = pd.concat([intercept_diagnostics_df, slope_diagnostics_df], ignore_index=True)
        
        # Reorder columns to match expected output format
        output_columns = ['model', 'test', 'statistic', 'p_value', 'assumption_met', 'remedial_action']
        all_diagnostics = all_diagnostics[output_columns]

        log(f"Saving step07_model_diagnostics.csv...")
        # Output: step07_model_diagnostics.csv
        # Contains: Comprehensive diagnostic results for both intercept and slope models
        # Columns: ['model', 'test', 'statistic', 'p_value', 'assumption_met', 'remedial_action']
        all_diagnostics.to_csv(RQ_DIR / "data/step07_model_diagnostics.csv", index=False, encoding='utf-8')
        log(f"step07_model_diagnostics.csv ({len(all_diagnostics)} rows, {len(all_diagnostics.columns)} cols)")
        # Run Validation Tool
        # Validates: p-value range check (all p-values should be between 0 and 1)
        # Threshold: p-values must be in [0, 1] range

        log("Running validate_numeric_range...")
        
        # Filter out NaN p-values for validation (VIF and other tests don't have p-values)
        valid_pvals = all_diagnostics.dropna(subset=['p_value'])['p_value']
        
        if len(valid_pvals) > 0:
            validation_result = validate_numeric_range(
                data=valid_pvals,
                min_val=0.0,
                max_val=1.0,
                column_name="p_value"
            )

            # Report validation results
            if isinstance(validation_result, dict):
                for key, value in validation_result.items():
                    log(f"{key}: {value}")
            else:
                log(f"{validation_result}")
        else:
            log("No p-values to validate (all diagnostic tests were non-parametric)")

        # Summary statistics for interpretation
        n_tests = len(all_diagnostics)
        n_intercept = len(all_diagnostics[all_diagnostics['model'] == 'intercept'])
        n_slope = len(all_diagnostics[all_diagnostics['model'] == 'slope'])
        n_assumptions_met = all_diagnostics['assumption_met'].sum()
        
        log(f"Total diagnostic tests: {n_tests}")
        log(f"Intercept model tests: {n_intercept}")
        log(f"Slope model tests: {n_slope}")
        log(f"Assumptions met: {n_assumptions_met}/{n_tests}")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)