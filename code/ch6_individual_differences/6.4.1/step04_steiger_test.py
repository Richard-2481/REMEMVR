#!/usr/bin/env python3
"""steiger_z_test: Test process-specificity: r(RAVLT, FreeRecall) > r(RAVLT, Recognition) using Steiger's Z-test"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_hypothesis_test_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_steiger_test.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Steiger's Z-test Implementation

def steiger_z_test(r12, r13, r23, n):
    """
    Compute Steiger's Z-test for comparing dependent correlations.
    
    Tests H0: r12 = r13 vs H1: r12 ≠ r13 (or r12 > r13 for one-tailed)
    where r12 and r13 share a common variable (dependent correlations).
    
    Parameters:
    -----------
    r12 : float
        Correlation between variable 1 and variable 2
    r13 : float  
        Correlation between variable 1 and variable 3
    r23 : float
        Correlation between variable 2 and variable 3
    n : int
        Sample size
        
    Returns:
    --------
    Dict with z_statistic and p_value (two-tailed)
    """
    # Steiger's formula for dependent correlations
    # Z = (r12 - r13) * sqrt((n-3) / (2 * (1 - r23^2)))
    
    numerator = r12 - r13
    
    # Avoid division by zero if r23 is exactly ±1
    if abs(r23) >= 0.999:
        # Use slightly smaller value to avoid numerical issues
        r23_adjusted = 0.999 * np.sign(r23) if r23 != 0 else 0.999
        log(f"r23 = {r23:.6f} very close to ±1, adjusting to {r23_adjusted:.6f}")
        r23 = r23_adjusted
    
    denominator_squared = 2 * (1 - r23**2)
    z_statistic = numerator * np.sqrt((n - 3) / denominator_squared)
    
    # Two-tailed p-value
    p_value_two_tailed = 2 * (1 - norm.cdf(abs(z_statistic)))
    
    return {
        'z_statistic': z_statistic,
        'p_value': p_value_two_tailed,
        'r_difference': numerator
    }

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Steiger Z-test for process-specificity")
        # Load Input Data
        
        log("Loading correlation results from step03...")
        correlation_results = pd.read_csv(RQ_DIR / "data" / "step03_correlation_results.csv")
        log(f"step03_correlation_results.csv ({len(correlation_results)} rows, {len(correlation_results.columns)} cols)")
        
        log("Loading raw correlation input data from step02...")
        raw_data = pd.read_csv(RQ_DIR / "data" / "step02_correlation_input.csv")
        log(f"step02_correlation_input.csv ({len(raw_data)} rows, {len(raw_data.columns)} cols)")
        # Extract Correlation Values and Run Steiger Tests
        # For each RAVLT predictor (total and pct_ret), extract r(pred, FR) and r(pred, RE)
        # then run Steiger's Z-test comparing the two dependent correlations.

        # Compute r23 = correlation(FreeRecall, Recognition) from raw data (shared across tests)
        log("Computing r23 = correlation(theta_free_recall, theta_recognition)...")
        r23 = raw_data['theta_free_recall'].corr(raw_data['theta_recognition'])
        log(f"r23 (FreeRecall-Recognition): {r23:.6f}")

        # Get sample size
        n = len(raw_data)
        log(f"Sample size: {n}")

        # Alpha: chapter-level Bonferroni correction
        alpha_threshold = 0.00179

        # Define predictor pairs to test
        predictor_configs = [
            {
                'predictor': 'ravlt_total',
                'fr_pair': 'RAVLTtotal-FreeRecall',
                're_pair': 'RAVLTtotal-Recognition',
            },
            {
                'predictor': 'ravlt_pct_ret',
                'fr_pair': 'RAVLTpctret-FreeRecall',
                're_pair': 'RAVLTpctret-Recognition',
            },
        ]

        steiger_rows = []
        validation_terms = []

        for config in predictor_configs:
            pred_name = config['predictor']
            log(f"\n=== Steiger test for {pred_name} ===")

            # Extract r12 (pred-FreeRecall) and r13 (pred-Recognition)
            fr_row = correlation_results[correlation_results['correlation_pair'] == config['fr_pair']]
            re_row = correlation_results[correlation_results['correlation_pair'] == config['re_pair']]

            if len(fr_row) == 0:
                raise ValueError(f"{config['fr_pair']} not found in step03 results")
            if len(re_row) == 0:
                raise ValueError(f"{config['re_pair']} not found in step03 results")

            r12 = fr_row['r_value'].iloc[0]
            r13 = re_row['r_value'].iloc[0]

            log(f"r12 ({config['fr_pair']}): {r12:.6f}")
            log(f"r13 ({config['re_pair']}): {r13:.6f}")

            # Run Steiger's Z-test
            steiger_result = steiger_z_test(r12, r13, r23, n)

            z_statistic = steiger_result['z_statistic']
            p_value_two_tailed = steiger_result['p_value']
            r_difference = steiger_result['r_difference']

            # Convert to one-tailed p-value for directional hypothesis H1: r12 > r13
            if z_statistic > 0:
                p_value_one_tailed = p_value_two_tailed / 2
            else:
                p_value_one_tailed = 1 - (p_value_two_tailed / 2)

            significant = p_value_one_tailed < alpha_threshold

            if significant:
                if r12 > r13:
                    interpretation = f"{pred_name} shows significantly greater correlation with Free Recall than Recognition (process-specific prediction supported)"
                else:
                    interpretation = f"{pred_name} shows significantly greater correlation with Recognition than Free Recall (opposite to hypothesis)"
            else:
                interpretation = f"No significant difference in {pred_name} correlation strength between Free Recall and Recognition paradigms"

            log(f"Z-statistic: {z_statistic:.6f}")
            log(f"Two-tailed p-value: {p_value_two_tailed:.8f}")
            log(f"One-tailed p-value (H1: r12 > r13): {p_value_one_tailed:.8f}")
            log(f"Correlation difference (r12 - r13): {r_difference:.6f}")
            log(f"Significant (p < {alpha_threshold}): {significant}")
            log(f"{interpretation}")

            steiger_rows.append({
                'predictor': pred_name,
                'z_statistic': z_statistic,
                'p_value': p_value_one_tailed,
                'r_difference': r_difference,
                'significant': significant,
                'interpretation': interpretation,
                'alpha_threshold': alpha_threshold
            })
            validation_terms.append(f'process_specificity_{pred_name}')
        # Save Analysis Output
        log("\nSaving Steiger test results...")

        steiger_output = pd.DataFrame(steiger_rows)

        output_path = RQ_DIR / "data" / "step04_steiger_test.csv"
        steiger_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(steiger_output)} rows, {len(steiger_output.columns)} cols)")
        # Run Validation Tool
        log("Running validate_hypothesis_test_dual_pvalues...")

        # Add compatibility columns for validation function
        validation_df = steiger_output.copy()
        validation_df['term'] = validation_terms
        validation_df['p_uncorrected'] = steiger_output['p_value']
        validation_df['p_bonferroni'] = steiger_output['p_value']

        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=validation_df,
            required_terms=validation_terms,
            alpha_bonferroni=0.00179
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 4 complete - Steiger's Z-test results generated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)