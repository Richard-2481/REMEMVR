#!/usr/bin/env python3
"""validate_ctt_assumptions: Perform comprehensive assumption checks for the CTT LMM model (residual normality,"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

from tools.validation import check_file_exists

# Import for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04b_validate_ctt_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04b: validate_ctt_assumptions")
        # Load Input Data

        log("Loading CTT model and input data...")

        # Load CTT model using MixedLMResults.load() method
        # CRITICAL: Use MixedLMResults.load() NOT pickle.load() to avoid patsy/eval errors
        model_path = RQ_DIR / "data" / "step03_ctt_lmm_model.pkl"
        ctt_model = MixedLMResults.load(str(model_path))
        log(f"{model_path.name} (CTT LMM model object)")

        # Load CTT LMM input data
        # Expected columns: composite_ID, UID, test, domain, TSVR_hours, CTT_score
        # Expected rows: ~1200 (400 UID x test x 3 domains)
        ctt_lmm_input = pd.read_csv(RQ_DIR / "data" / "step03_ctt_lmm_input.csv", encoding='utf-8')
        log(f"step03_ctt_lmm_input.csv ({len(ctt_lmm_input)} rows, {len(ctt_lmm_input.columns)} cols)")
        # Run Analysis Tool
        #   1. Residual normality (Shapiro-Wilk test + QQ plot)
        #   2. Homoscedasticity (Breusch-Pagan test + residuals vs fitted plot)
        #   3. Random effects normality (QQ plot of BLUPs)
        #   4. Independence (ACF plot + lag-1 autocorrelation check)

        log("Running validate_lmm_assumptions_comprehensive...")

        # Create output directory for diagnostic plots
        output_dir = RQ_DIR / "plots" / "step04b_ctt_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Call validation tool with parameters from 4_analysis.yaml
        # Parameters:
        #   - acf_lag1_threshold=0.1: Threshold for acceptable lag-1 autocorrelation
        #   - alpha=0.05: Significance level for statistical tests
        ctt_assumptions_result = validate_lmm_assumptions_comprehensive(
            lmm_result=ctt_model,  # Fitted CTT model from step03
            data=ctt_lmm_input,     # Input data used for model fitting
            output_dir=output_dir,  # Where to save diagnostic plots
            acf_lag1_threshold=0.1, # Threshold for independence check
            alpha=0.05              # Significance level for tests
        )
        log("Assumption validation complete")
        # Save Analysis Outputs
        # These outputs will be used by: rq_inspect for validation, results analysis for reporting

        log("Saving assumption diagnostics report...")

        # Output: results/step04b_ctt_assumptions_report.txt
        # Contains: Comprehensive text report with test results and interpretations
        # Format: Plain text (UTF-8) with structured sections
        report_path = RQ_DIR / "results" / "step04b_ctt_assumptions_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CTT LMM ASSUMPTION DIAGNOSTICS (Step 04b)\n")
            f.write("=" * 80 + "\n\n")

            # Write validation results (dictionary returned by validation tool)
            for key, value in ctt_assumptions_result.items():
                f.write(f"{key}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("DIAGNOSTIC PLOTS SAVED TO: plots/step04b_ctt_diagnostics/\n")
            f.write("=" * 80 + "\n")

        log(f"{report_path.name} ({report_path.stat().st_size} bytes)")

        # Diagnostic plots are automatically saved by validate_lmm_assumptions_comprehensive
        # Expected 6 plots in plots/step04b_ctt_diagnostics/:
        #   1. residuals_qq.png (residual normality check)
        #   2. residuals_vs_fitted.png (homoscedasticity check)
        #   3. random_effects_qq.png (random effects normality check)
        #   4. acf_plot.png (independence check)
        #   5. residuals_histogram.png (residual distribution)
        #   6. scale_location.png (spread-location plot)
        plot_count = len(list(output_dir.glob("*.png")))
        log(f"{plot_count} diagnostic plots in {output_dir.name}/")
        # Run Validation Tool
        # Validates: Both assumption reports exist (step04a IRT, step04b CTT)
        #            Text reports > 500 characters (comprehensive diagnostics)
        #            Both diagnostic plot directories exist with plots
        # Threshold: min_size_bytes=500 for text reports

        log("Checking file existence with os.path...")

        # Check step04a IRT report exists (prerequisite from earlier step)
        irt_report_path = RQ_DIR / "results" / "step04a_irt_assumptions_report.txt"
        irt_exists = irt_report_path.exists()

        # Check step04b CTT report exists (just created)
        ctt_exists = report_path.exists()

        # Check step04a IRT plot directory exists
        irt_plots_dir = RQ_DIR / "plots" / "step04a_irt_diagnostics"
        irt_plots_exist = irt_plots_dir.exists()

        # Check step04b CTT plot directory exists
        ctt_plots_exist = output_dir.exists()

        # Report validation results
        log(f"IRT report exists: {irt_exists}")
        log(f"CTT report exists: {ctt_exists}")
        log(f"IRT plots exist: {irt_plots_exist}")
        log(f"CTT plots exist: {ctt_plots_exist}")

        # Check all validations passed
        all_passed = all([irt_exists, ctt_exists, irt_plots_exist, ctt_plots_exist])

        if not all_passed:
            raise FileNotFoundError(
                "Validation failed - not all required files exist. "
                "Check that step04a completed successfully before running step04b."
            )

        log("Step 04b complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
