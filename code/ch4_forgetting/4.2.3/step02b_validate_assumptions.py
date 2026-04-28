#!/usr/bin/env python3
"""Validate LMM Assumptions: Verify Linear Mixed Model assumptions before proceeding to inference. Runs"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02b_validate_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02b: Validate LMM Assumptions")
        # Load Input Data

        log("Loading fitted LMM model from Step 2...")
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"

        # Load statsmodels MixedLMResults object using statsmodels.load()
        # CRITICAL: Use MixedLMResults.load() method, NOT pickle.load()
        # Why: pickle.load() causes patsy/eval errors with statsmodels models
        from statsmodels.regression.mixed_linear_model import MixedLMResults
        lmm_model = MixedLMResults.load(str(model_path))
        log(f"LMM model from {model_path}")

        log("Loading LMM input data from Step 1...")
        lmm_input_path = RQ_DIR / "data" / "step01_lmm_input.csv"
        lmm_input = pd.read_csv(lmm_input_path, encoding='utf-8')
        log(f"LMM input data: {len(lmm_input)} rows, {len(lmm_input.columns)} columns")

        # Verify expected structure
        expected_cols = ['UID', 'composite_ID', 'test', 'domain', 'theta', 'TSVR_hours',
                        'log_TSVR', 'age', 'Age_c', 'mean_age']
        missing_cols = set(expected_cols) - set(lmm_input.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns in LMM input: {missing_cols}")
        log(f"All expected columns present in LMM input")
        # Run Assumption Validation
        #   (1) Residual normality: Shapiro-Wilk test + Q-Q plot
        #   (2) Homoscedasticity: Breusch-Pagan test + residuals vs fitted plot
        #   (3) Random effects normality: Shapiro-Wilk + Q-Q plots for intercepts/slopes
        #   (4) Independence: ACF plot + Lag-1 test (autocorrelation check)
        #   (5) Linearity: Partial residual CSVs for each predictor
        #   (6) Outliers: Cook's distance plot
        #   (7) Convergence: Model convergence status

        log("Running comprehensive LMM assumption checks...")
        log("This includes 7 diagnostics:")
        log("  (1) Residual normality (Shapiro-Wilk + Q-Q plot)")
        log("  (2) Homoscedasticity (Breusch-Pagan + residuals vs fitted)")
        log("  (3) Random effects normality (Shapiro-Wilk + Q-Q plots)")
        log("  (4) Independence (ACF plot + Lag-1 autocorrelation)")
        log("  (5) Linearity (partial residual plots)")
        log("  (6) Outliers (Cook's distance)")
        log("  (7) Convergence (model status)")

        # Create plots directory if it doesn't exist
        plots_dir = RQ_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        assumption_result = validate_lmm_assumptions_comprehensive(
            lmm_result=lmm_model,
            data=lmm_input,
            output_dir=plots_dir,
            acf_lag1_threshold=0.1,  # Autocorrelation tolerance (|ACF lag-1| < 0.1 acceptable)
            alpha=0.05  # Significance level for statistical tests
        )

        log("Assumption validation complete")
        # Save Assumption Diagnostics Report
        # Output: results/step02b_assumption_diagnostics.txt
        # Contains: Text summary of all 7 diagnostics with overall assessment
        # Format: Plain text with Pass/Conditional/Fail per diagnostic

        log("Saving assumption diagnostics report...")
        diagnostics_path = RQ_DIR / "results" / "step02b_assumption_diagnostics.txt"

        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LMM ASSUMPTION VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"RQ: ch5/5.2.3 - Age x Domain x Time Interaction (3-way)\n")
            f.write(f"Step: 02b - Validate LMM Assumptions\n")
            f.write(f"Model: 3-way Age x Domain x Time interaction with random slopes\n\n")

            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            if assumption_result['valid']:
                f.write("Status: PASS - All assumptions satisfied\n")
            else:
                f.write("Status: CONDITIONAL PASS / FAIL - See violations below\n")
            f.write(f"Message: {assumption_result['message']}\n\n")

            f.write("DIAGNOSTIC DETAILS\n")
            f.write("-" * 80 + "\n\n")

            # Write each diagnostic result
            diagnostics = assumption_result['diagnostics']
            for idx, (diag_name, diag_result) in enumerate(diagnostics.items(), 1):
                f.write(f"{idx}. {diag_name.upper()}\n")
                f.write(f"   Result: {diag_result}\n\n")

            f.write("\nDIAGNOSTIC PLOTS\n")
            f.write("-" * 80 + "\n")
            for plot_path in assumption_result.get('plot_paths', []):
                f.write(f"  - {plot_path.name}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        log(f"Assumption diagnostics report: {diagnostics_path}")
        # Verify Diagnostic Plots Generated

        log("Checking diagnostic plots...")
        plot_paths = assumption_result.get('plot_paths', [])
        if len(plot_paths) == 0:
            log("No diagnostic plots generated (unexpected)")
        else:
            log(f"{len(plot_paths)} diagnostic plots generated:")
            for plot_path in plot_paths:
                if plot_path.exists():
                    log(f"  [OK] {plot_path.name}")
                else:
                    log(f"  {plot_path.name}")
        # Report Validation Summary
        # Report overall status and any violations
        # Non-blocking: Assumption violations flagged but don't halt pipeline

        log("[VALIDATION SUMMARY]")
        log(f"  Overall valid: {assumption_result['valid']}")
        log(f"  Message: {assumption_result['message']}")

        # Report individual diagnostic statuses
        log("[DIAGNOSTIC STATUSES]")
        for diag_name, diag_result in assumption_result['diagnostics'].items():
            log(f"  {diag_name}: {diag_result}")

        # Determine exit status
        # NOTE: Assumption validation is NON-BLOCKING per 4_analysis.yaml
        # "on_failure: Flag violations in report, proceed with caution (not blocking)"
        if assumption_result['valid']:
            log("Step 02b complete - All assumptions satisfied")
            sys.exit(0)
        else:
            log("[CONDITIONAL SUCCESS] Step 02b complete - Some violations flagged")
            log("Assumption violations logged but not blocking (proceed with caution)")
            sys.exit(0)  # Exit 0 because non-blocking per specification

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
