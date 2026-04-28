#!/usr/bin/env python3
"""compute_correlations_with_bootstrap_ci: Compute r(RAVLT, FreeRecall) and r(RAVLT, Recognition) with bootstrap CIs"""

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

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_compute_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: compute_correlations_with_bootstrap_ci")
        # Load Input Data

        log("Loading correlation input data...")
        input_file = RQ_DIR / "data" / "step02_correlation_input.csv"
        df = pd.read_csv(input_file)
        log(f"{input_file.name} ({len(df)} rows, {len(df.columns)} cols)")
        
        # Verify expected columns
        expected_cols = ["uid", "ravlt_total", "ravlt_pct_ret", "theta_free_recall", "theta_recognition"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

        log(f"All expected columns present: {expected_cols}")
        log(f"Data shape: {df.shape}")
        # Run Analysis Tool

        log("Computing correlations with bootstrap confidence intervals...")
        
        # Parameters for bootstrap
        n_bootstrap = 1000
        confidence = 0.95
        method = "pearson"
        seed = 42
        
        log(f"Bootstrap parameters: n={n_bootstrap}, confidence={confidence}, method={method}, seed={seed}")
        
        # Convert to numpy arrays for bootstrap function
        ravlt_total = df['ravlt_total'].values
        ravlt_pct_ret = df['ravlt_pct_ret'].values
        theta_free_recall = df['theta_free_recall'].values
        theta_recognition = df['theta_recognition'].values

        # Define all correlation pairs to compute
        corr_pairs = [
            ('RAVLTtotal-FreeRecall', ravlt_total, theta_free_recall),
            ('RAVLTtotal-Recognition', ravlt_total, theta_recognition),
            ('RAVLTpctret-FreeRecall', ravlt_pct_ret, theta_free_recall),
            ('RAVLTpctret-Recognition', ravlt_pct_ret, theta_recognition),
        ]

        corr_results = []
        for pair_name, x_arr, y_arr in corr_pairs:
            log(f"Computing r({pair_name})...")
            result = bootstrap_correlation_ci(
                x=x_arr,
                y=y_arr,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                method=method,
                seed=seed
            )
            log(f"r({pair_name}) = {result['r']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            corr_results.append((pair_name, result))
        # STEP 2b: Calculate p-values (Decision D068 requirement)
        # Decision D068: Must provide both uncorrected and Bonferroni-corrected p-values

        log("Calculating p-values for correlations...")

        n_obs = len(df)
        n_tests = len(corr_pairs)  # 4 tests for Bonferroni correction
        log(f"Bonferroni correction for {n_tests} tests")

        p_uncorrected_list = []
        for pair_name, result in corr_results:
            r_val = result['r']
            if abs(r_val) >= 1.0:
                p_unc = 0.0 if abs(r_val) > 1.0 else 1.0
            else:
                t_val = r_val * np.sqrt((n_obs - 2) / (1 - r_val**2))
                p_unc = 2 * (1 - stats.t.cdf(abs(t_val), df=n_obs-2))
            p_uncorrected_list.append(p_unc)

        p_bonferroni_list = [min(p * n_tests, 1.0) for p in p_uncorrected_list]

        for i, (pair_name, _) in enumerate(corr_results):
            log(f"{pair_name}: p_uncorrected={p_uncorrected_list[i]:.6f}, p_bonferroni={p_bonferroni_list[i]:.6f}")
        # Save Analysis Outputs
        # These outputs will be used by: Step 4 (Steiger Z-test) for process-specificity testing

        log("Compiling correlation results...")

        # Create results dataframe
        rows = []
        for i, (pair_name, result) in enumerate(corr_results):
            rows.append({
                'correlation_pair': pair_name,
                'r_value': result['r'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'p_uncorrected': p_uncorrected_list[i],
                'p_bonferroni': p_bonferroni_list[i],
                'n_obs': n_obs
            })
        correlation_results = pd.DataFrame(rows)
        
        # Save results
        output_file = RQ_DIR / "data" / "step03_correlation_results.csv"
        correlation_results.to_csv(output_file, index=False, encoding='utf-8')
        log(f"{output_file.name} ({len(correlation_results)} rows, {len(correlation_results.columns)} cols)")
        # Run Validation Tool
        # Validates: Decision D068 compliance (dual p-value reporting)
        # Threshold: Checks for required columns and proper structure

        log("Running validate_correlation_test_d068...")
        validation_result = validate_correlation_test_d068(
            correlation_df=correlation_results,
            required_cols=["correlation_pair", "r_value", "ci_lower", "ci_upper", "p_uncorrected", "p_bonferroni"]
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check if validation passed
        if validation_result.get('valid', False):
            log("D068 compliance validated")
        else:
            log("D068 compliance issues found")
            if not validation_result.get('d068_compliant', True):
                log("Missing required dual p-value columns")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)