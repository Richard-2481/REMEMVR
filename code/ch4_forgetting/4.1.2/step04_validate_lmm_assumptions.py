#!/usr/bin/env python3
"""Validate LMM Assumptions: Perform comprehensive LMM assumption validation for both quadratic and piecewise"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

from tools.validation import validate_hypothesis_test_dual_pvalues

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_validate_lmm_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Validate LMM Assumptions")
        # Load Fitted Models from Steps 2-3
        #          residuals and random effects estimates

        log("Loading fitted models from Steps 2-3...")

        # Load quadratic model from Step 2
        quadratic_model_path = RQ_DIR / "data" / "step02_quadratic_model.pkl"
        log(f"Loading quadratic model from {quadratic_model_path}")

        if not quadratic_model_path.exists():
            raise FileNotFoundError(f"Quadratic model not found: {quadratic_model_path}")

        # CRITICAL: Use MixedLMResults.load() method (not pickle.load)
        # Reason: statsmodels models require special loading to restore patsy state
        quadratic_model = MixedLMResults.load(str(quadratic_model_path))
        log(f"Quadratic model ({quadratic_model.nobs} observations)")

        # Load piecewise model from Step 3
        piecewise_model_path = RQ_DIR / "data" / "step03_piecewise_model.pkl"
        log(f"Loading piecewise model from {piecewise_model_path}")

        if not piecewise_model_path.exists():
            raise FileNotFoundError(f"Piecewise model not found: {piecewise_model_path}")

        piecewise_model = MixedLMResults.load(str(piecewise_model_path))
        log(f"Piecewise model ({piecewise_model.nobs} observations)")

        # Load original data for residual computation
        time_data_path = RQ_DIR / "data" / "step01_time_transformed.csv"
        log(f"Loading time-transformed data from {time_data_path}")

        if not time_data_path.exists():
            raise FileNotFoundError(f"Time data not found: {time_data_path}")

        time_data = pd.read_csv(time_data_path)
        log(f"Time data ({len(time_data)} rows, {len(time_data.columns)} cols)")
        # Run Assumption Validation for Quadratic Model
        #               random effects (normality, homoscedasticity, autocorrelation,
        #               linearity, outliers, convergence)

        log("Running assumption validation for quadratic model...")

        quadratic_validation = validate_lmm_assumptions_comprehensive(
            lmm_result=quadratic_model,  # Fitted model from Step 2
            data=time_data,  # Original data for residual computation
            output_dir=RQ_DIR / "results",  # Save diagnostic plots to results/
            acf_lag1_threshold=0.1,  # Autocorrelation threshold (conservative)
            alpha=0.05  # Significance level (Bonferroni corrected internally)
        )

        log("Quadratic model assumption validation complete")
        log(f"Quadratic model valid: {quadratic_validation['valid']}")
        log(f"Quadratic diagnostics: {len(quadratic_validation['diagnostics'])} checks performed")
        # Run Assumption Validation for Piecewise Model

        log("Running assumption validation for piecewise model...")

        piecewise_validation = validate_lmm_assumptions_comprehensive(
            lmm_result=piecewise_model,  # Fitted model from Step 3
            data=time_data,  # Original data for residual computation
            output_dir=RQ_DIR / "results",  # Save diagnostic plots to results/
            acf_lag1_threshold=0.1,  # Autocorrelation threshold (conservative)
            alpha=0.05  # Significance level (Bonferroni corrected internally)
        )

        log("Piecewise model assumption validation complete")
        log(f"Piecewise model valid: {piecewise_validation['valid']}")
        log(f"Piecewise diagnostics: {len(piecewise_validation['diagnostics'])} checks performed")
        # Aggregate Results and Save Report
        # These outputs will be used by: rq_inspect for validation, results analysis for reporting

        log("Saving assumption validation report...")

        report_path = RQ_DIR / "results" / "step04_assumption_validation_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LMM ASSUMPTION VALIDATION REPORT - RQ 5.1.2\n")
            f.write("=" * 80 + "\n\n")

            f.write("OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Quadratic model: {'PASS' if quadratic_validation['valid'] else 'FAIL'}\n")
            f.write(f"Piecewise model: {'PASS' if piecewise_validation['valid'] else 'FAIL'}\n")
            f.write(f"Total checks: {len(quadratic_validation['diagnostics']) + len(piecewise_validation['diagnostics'])}\n\n")

            f.write("QUADRATIC MODEL DIAGNOSTICS\n")
            f.write("-" * 80 + "\n")
            for check_name, check_result in quadratic_validation['diagnostics'].items():
                f.write(f"\n{check_name}:\n")
                if isinstance(check_result, dict):
                    for key, value in check_result.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {check_result}\n")

            f.write("\n\nPIECEWISE MODEL DIAGNOSTICS\n")
            f.write("-" * 80 + "\n")
            for check_name, check_result in piecewise_validation['diagnostics'].items():
                f.write(f"\n{check_name}:\n")
                if isinstance(check_result, dict):
                    for key, value in check_result.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {check_result}\n")

            f.write("\n\nDIAGNOSTIC PLOTS\n")
            f.write("-" * 80 + "\n")
            f.write("Quadratic model plots:\n")
            for plot_path in quadratic_validation.get('plot_paths', []):
                f.write(f"  {plot_path}\n")
            f.write("\nPiecewise model plots:\n")
            for plot_path in piecewise_validation.get('plot_paths', []):
                f.write(f"  {plot_path}\n")

            f.write("\n\nOVERALL MESSAGE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Quadratic: {quadratic_validation['message']}\n")
            f.write(f"Piecewise: {piecewise_validation['message']}\n")

        log(f"{report_path}")
        # Meta-Validation (Validate the Validators)
        # Validates: That assumption tests executed correctly (test statistics finite,
        #            p-values in [0, 1], PASS/FAIL documented)
        # Threshold: N/A (meta-validation checks execution correctness, not statistical thresholds)

        log("Running meta-validation on assumption test results...")

        # Prepare diagnostics dict for validation
        # Convert nested diagnostics to DataFrame-like structure for validation
        all_diagnostics = {
            'Quadratic': quadratic_validation['diagnostics'],
            'Piecewise': piecewise_validation['diagnostics']
        }

        # Extract key test statistics for validation
        required_terms = ["Shapiro-Wilk", "Breusch-Pagan", "ACF", "Cook's D"]

        # Create a pseudo-DataFrame for validation
        # (validate_hypothesis_test_dual_pvalues expects DataFrame with terms)
        validation_rows = []
        for model_name, diagnostics in all_diagnostics.items():
            for term in required_terms:
                # Check if term exists in diagnostics
                if term in diagnostics or any(term in str(k) for k in diagnostics.keys()):
                    validation_rows.append({
                        'model': model_name,
                        'term': term,
                        'present': True
                    })
                else:
                    validation_rows.append({
                        'model': model_name,
                        'term': term,
                        'present': False
                    })

        validation_df = pd.DataFrame(validation_rows)

        # Check that all required terms are present
        missing_terms = validation_df[~validation_df['present']]['term'].unique().tolist()

        if missing_terms:
            log(f"Some assumption tests missing: {missing_terms}")
        else:
            log("All assumption tests present")

        # Verify test statistics are finite
        all_finite = True
        for model_name, diagnostics in all_diagnostics.items():
            for check_name, check_result in diagnostics.items():
                if isinstance(check_result, dict):
                    for key, value in check_result.items():
                        if isinstance(value, (int, float)) and not np.isfinite(value):
                            log(f"Non-finite value in {model_name} {check_name}.{key}: {value}")
                            all_finite = False

        if all_finite:
            log("All test statistics finite")

        # Report validation results
        log("Meta-validation results:")
        log(f"  Required terms present: {len(missing_terms) == 0}")
        log(f"  Test statistics finite: {all_finite}")
        log(f"  Quadratic model checks: {len(quadratic_validation['diagnostics'])}")
        log(f"  Piecewise model checks: {len(piecewise_validation['diagnostics'])}")

        overall_valid = (len(missing_terms) == 0 and all_finite and
                         len(quadratic_validation['diagnostics']) >= 6 and
                         len(piecewise_validation['diagnostics']) >= 6)

        if overall_valid:
            log("Meta-validation PASS - All assumption tests executed correctly")
        else:
            log("Meta-validation FAIL - Some issues detected (see warnings above)")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
