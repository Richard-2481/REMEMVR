#!/usr/bin/env python3
"""validate_irt_assumptions: Perform comprehensive assumption checks for IRT LMM to validate statistical"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

from tools.validation import check_file_exists

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04a_validate_irt_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4a: Validate IRT LMM Assumptions")
        # Load Input Data
        #           and original LMM input data (1200 rows)

        log("Loading fitted IRT model from step03...")
        # CRITICAL: Use MixedLMResults.load() method (NOT pickle.load())
        # Reason: pickle.load() causes patsy/eval errors with statsmodels models
        irt_model_path = RQ_DIR / "data" / "step03_irt_lmm_model.pkl"
        irt_model = MixedLMResults.load(str(irt_model_path))
        log(f"IRT model from {irt_model_path}")
        log(f"Model converged: {irt_model.converged}")
        log(f"Number of observations: {irt_model.nobs}")
        log(f"Number of groups (UIDs): {len(irt_model.model.group_labels)}")

        log("Loading IRT LMM input data...")
        # Expected rows: 1200 (400 UID x test x 3 domains)
        irt_lmm_input = pd.read_csv(RQ_DIR / "data" / "step03_irt_lmm_input.csv")
        log(f"step03_irt_lmm_input.csv ({len(irt_lmm_input)} rows, {len(irt_lmm_input.columns)} cols)")
        # Create Output Directory for Diagnostic Plots

        output_dir = RQ_DIR / "plots" / "step04a_irt_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Diagnostic plots directory: {output_dir}")
        # Run Comprehensive Assumption Validation
        #   1. Residual normality (Shapiro-Wilk + Q-Q plot)
        #   2. Homoscedasticity (Breusch-Pagan + residuals vs fitted)
        #   3. Random effects normality (Shapiro-Wilk + Q-Q plots)
        #   4. Autocorrelation (ACF plot + Lag-1 test)
        #   5. Linearity (partial residual CSVs)
        #   6. Outliers (Cook's distance)
        #   7. Convergence diagnostics

        log("Running validate_lmm_assumptions_comprehensive...")
        log(f"ACF Lag-1 threshold: 0.1")
        log(f"Significance level (alpha): 0.05")

        assumptions_result = validate_lmm_assumptions_comprehensive(
            lmm_result=irt_model,  # Fitted IRT model from step03
            data=irt_lmm_input,    # Original input data (1200 rows)
            output_dir=output_dir, # plots/step04a_irt_diagnostics/
            acf_lag1_threshold=0.1,  # Autocorrelation threshold (max 10% correlation at lag 1)
            alpha=0.05                # Significance level for statistical tests
        )
        log("Assumption validation complete")
        # Save Comprehensive Diagnostics Report
        # These outputs will be used by: rq_inspect (validation) and researcher review

        log("Saving assumption diagnostics report...")
        # Output: results/step04a_irt_assumptions_report.txt
        # Contains: Text summary of all 7 diagnostics + pass/fail status
        report_path = RQ_DIR / "results" / "step04a_irt_assumptions_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("IRT LMM ASSUMPTION VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Overall Status: {'' if assumptions_result['valid'] else ''}\n\n")
            f.write(assumptions_result['message'] + "\n\n")

            f.write("DIAGNOSTICS SUMMARY:\n")
            f.write("-" * 80 + "\n")
            for key, value in assumptions_result['diagnostics'].items():
                f.write(f"{key}: {value}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write(f"DIAGNOSTIC PLOTS GENERATED ({len(assumptions_result['plot_paths'])} files):\n")
            f.write("-" * 80 + "\n")
            for plot_path in assumptions_result['plot_paths']:
                f.write(f"  - {plot_path.name}\n")

        log(f"{report_path} ({report_path.stat().st_size} bytes)")
        # Validate Outputs
        # Validates: Assumption report exists (>500 bytes) and plots directory exists
        # Threshold: Text report must be >500 characters for comprehensive diagnostics

        log("Validating assumption report file...")
        report_validation = check_file_exists(
            file_path=report_path,
            min_size_bytes=500  # Comprehensive report expected to be >500 characters
        )

        if not report_validation['valid']:
            raise FileNotFoundError(f"Assumption report validation failed: {report_validation['message']}")

        log(f"Assumption report exists: {report_validation['size_bytes']} bytes (>500 required)")

        log("Validating diagnostic plots directory...")
        # Check directory exists and contains plots
        if not output_dir.exists():
            raise FileNotFoundError(f"Diagnostic plots directory missing: {output_dir}")

        plot_files = list(output_dir.glob("*.png"))
        log(f"Found {len(plot_files)} diagnostic plots")

        if len(plot_files) < 6:
            raise FileNotFoundError(f"Expected 6 diagnostic plots, found {len(plot_files)}")

        log("All outputs validated successfully")

        # Report validation results summary
        log("\n[ASSUMPTION VALIDATION SUMMARY]")
        log(f"  Overall Status: {'' if assumptions_result['valid'] else ''}")
        log(f"  Diagnostic plots: {len(plot_files)} generated")
        log(f"  Report size: {report_validation['size_bytes']} bytes")

        if assumptions_result['valid']:
            log("IRT LMM assumptions satisfied")
        else:
            log("Some assumptions violated - review diagnostics report")

        log("Step 4a complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
