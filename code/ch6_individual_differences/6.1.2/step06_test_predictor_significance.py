#!/usr/bin/env python3
"""step06_test_predictor_significance: Test individual predictor significance with dual p-value reporting (Decision D068)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_test_predictor_significance.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Test Predictor Significance")
        # Load Input Data

        log("Loading intercept regression results...")
        intercept_results = pd.read_csv(RQ_DIR / "data" / "step03_intercept_predictions.csv")
        log(f"step03_intercept_predictions.csv ({len(intercept_results)} rows, {len(intercept_results.columns)} cols)")

        log("Loading slope regression results...")
        slope_results = pd.read_csv(RQ_DIR / "data" / "step04_slope_predictions.csv")
        log(f"step04_slope_predictions.csv ({len(slope_results)} rows, {len(slope_results.columns)} cols)")
        # Filter Cognitive Test Predictors
        # Extract only the cognitive test predictors (exclude regression constant/intercept)
        # Target predictors: RAVLT_T, BVMT_T, RPM_T

        log("Extracting cognitive test predictors...")
        
        # Define cognitive test predictors (exclude constant/intercept terms)
        cognitive_predictors = ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "RPM_T"]
        
        # Filter intercept model results for cognitive predictors only
        intercept_cognitive = intercept_results[
            intercept_results['predictor'].isin(cognitive_predictors)
        ].copy()
        intercept_cognitive['outcome'] = 'intercept'
        
        log(f"Intercept model: {len(intercept_cognitive)} cognitive predictors")
        
        # Filter slope model results for cognitive predictors only  
        slope_cognitive = slope_results[
            slope_results['predictor'].isin(cognitive_predictors)
        ].copy()
        slope_cognitive['outcome'] = 'slope'
        
        log(f"Slope model: {len(slope_cognitive)} cognitive predictors")
        # Combine Results and Compute Significance Flags
        # Combine intercept and slope results, then compute significance flags
        # Decision D068: Dual p-value reporting (uncorrected + Bonferroni corrected)

        log("Combining results and computing significance flags...")
        
        # Combine both model results
        combined_results = pd.concat([intercept_cognitive, slope_cognitive], ignore_index=True)
        
        # Select relevant columns for output
        significance_results = combined_results[[
            'predictor', 'outcome', 'beta', 'p_uncorrected', 'p_bonferroni'
        ]].copy()
        
        # Add 'term' column for validation compatibility
        significance_results['term'] = significance_results['predictor']
        
        # Compute significance flags per Decision D068
        # Uncorrected: alpha = 0.05
        significance_results['sig_uncorrected'] = (
            significance_results['p_uncorrected'] < 0.05
        ).astype(int)
        
        # Bonferroni corrected: alpha = 0.05/10 = 0.005 (5 predictors x 2 outcomes)
        bonferroni_alpha = 0.05 / 10  # 0.005
        significance_results['sig_bonferroni'] = (
            significance_results['p_bonferroni'] < bonferroni_alpha
        ).astype(int)
        # Generate Effect Interpretations
        # Provide qualitative interpretation of effect direction and magnitude
        
        log("Generating effect interpretations...")
        
        def interpret_effect(row):
            """Generate effect interpretation based on beta coefficient and significance."""
            predictor = row['predictor']
            outcome = row['outcome']
            beta = row['beta']
            sig_uncorr = row['sig_uncorrected']
            sig_bonf = row['sig_bonferroni']
            
            # Effect direction
            direction = "positive" if beta > 0 else "negative"
            
            # Effect magnitude (rule of thumb for standardized predictors)
            abs_beta = abs(beta)
            if abs_beta < 0.1:
                magnitude = "negligible"
            elif abs_beta < 0.3:
                magnitude = "small"
            elif abs_beta < 0.5:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            # Significance interpretation
            if sig_bonf:
                sig_status = "significant (corrected)"
            elif sig_uncorr:
                sig_status = "significant (uncorrected only)"
            else:
                sig_status = "non-significant"
                
            return f"{magnitude} {direction} effect on {outcome} ({sig_status})"
        
        # Apply interpretation function
        significance_results['effect_interpretation'] = significance_results.apply(
            interpret_effect, axis=1
        )
        
        log(f"Effect interpretations for {len(significance_results)} predictor-outcome combinations")
        # Save Analysis Output
        # Save detailed significance test results with dual p-values per D068

        log("Saving predictor significance results...")
        output_path = RQ_DIR / "data" / "step06_predictor_significance.csv"
        significance_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step06_predictor_significance.csv ({len(significance_results)} rows, {len(significance_results.columns)} cols)")
        # Run Validation Tool
        # Validate dual p-value reporting and required predictors per Decision D068
        
        log("Running validate_hypothesis_test_dual_pvalues...")
        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=significance_results,
            required_terms=["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "RPM_T"],
            alpha_bonferroni=bonferroni_alpha  # 0.0083
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
                
            # Check validation success
            if validation_result.get('valid', False):
                log("Predictor significance analysis PASSED validation")
            else:
                log(f"FAILED: {validation_result.get('message', 'Unknown error')}")
                raise ValueError(f"Validation failed: {validation_result.get('message')}")
        else:
            log(f"{validation_result}")

        # Summary statistics
        log("Predictor significance test complete:")
        log(f"  - Total predictor-outcome combinations: {len(significance_results)}")
        log(f"  - Significant (uncorrected p<0.05): {significance_results['sig_uncorrected'].sum()}")
        log(f"  - Significant (Bonferroni p<{bonferroni_alpha:.4f}): {significance_results['sig_bonferroni'].sum()}")
        log(f"  - Dual p-value reporting: Decision D068 compliant")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)