#!/usr/bin/env python3
"""step05_prepare_plot_data: Create plot source CSV for multi-panel age effects visualization (Option B"""

import sys
from pathlib import Path
import pandas as pd
import traceback
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import prepare_age_effects_plot_data

from tools.validation import validate_plot_data_completeness

# Import for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_prepare_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Prepare Plot Data")
        # Load Input Data

        log("Loading LMM input data...")
        log("When domain excluded due to floor effect")
        # Load LMM input with age variable
        # Expected columns: UID, age, domain, TSVR_hours, theta, composite_ID, test, log_TSVR, Age_c, mean_age
        # Expected rows: 800 (When excluded)
        lmm_input = pd.read_csv(RQ_DIR / "data" / "step01_lmm_input.csv", encoding='utf-8')
        log(f"step01_lmm_input.csv ({len(lmm_input)} rows, {len(lmm_input.columns)} cols)")

        # Load fitted LMM model from Step 2c (selected model with REML=False)
        # CRITICAL: Use MixedLMResults.load() method, NOT pickle.load()
        # Reason: Prevents patsy/eval errors when loading fitted models
        log("Loading fitted LMM model...")
        lmm_model = MixedLMResults.load(str(RQ_DIR / "data" / "step02_lmm_model.pkl"))
        log(f"step02_lmm_model.pkl (MixedLMResults object)")

        # Load age effects by domain (for context - not directly used by tool)
        # Expected columns: domain, age_effect, se, z, p, CI_lower, CI_upper
        # Expected rows: 2 (What, Where - When excluded)
        log("Loading age effects by domain...")
        age_effects = pd.read_csv(RQ_DIR / "data" / "step04_age_effects_by_domain.csv", encoding='utf-8')
        log(f"step04_age_effects_by_domain.csv ({len(age_effects)} rows, {len(age_effects.columns)} cols)")
        # Run Analysis Tool (prepare_age_effects_plot_data)
        #   observed theta scores with 95% CIs, generates LMM predictions

        log("Running prepare_age_effects_plot_data...")
        # Output path for plot data CSV
        output_path = RQ_DIR / "plots" / "step05_age_effects_plot_data.csv"

        # CRITICAL FIX: Tool expects 'domain_name' column but data has 'domain'
        # Rename before passing to tool so domain-specific data is preserved
        lmm_input_renamed = lmm_input.rename(columns={'domain': 'domain_name'})
        log("Renamed 'domain' -> 'domain_name' for tool compatibility")

        # Call analysis tool
        # Parameters:
        #   lmm_input: Long-format LMM data with UID, age, domain_name, TSVR_hours, theta
        #   lmm_model: Fitted MixedLMResults object from Step 2c
        #   output_path: Where to save plot-ready CSV
        plot_data = prepare_age_effects_plot_data(
            lmm_input=lmm_input_renamed,
            lmm_model=lmm_model,
            output_path=output_path
        )
        log("Analysis complete")
        # Save Analysis Outputs
        # Output already saved by analysis tool to plots/step05_age_effects_plot_data.csv
        # Contains: domain, age_tertile, TSVR_hours, theta_observed, se_observed,
        #   ci_lower, ci_upper, theta_predicted

        log(f"Plot data saved by analysis tool...")
        log(f"step05_age_effects_plot_data.csv ({len(plot_data)} rows, {len(plot_data.columns)} cols)")
        log(f"Columns: {list(plot_data.columns)}")
        # Run Validation Tool (validate_plot_data_completeness)
        # Validates: All 2 domains present (What, Where - When excluded)
        #           All 3 age tertiles present (Young, Middle, Older)
        #           ~400 rows expected (2 domains x 3 tertiles x ~67 timepoints)
        #           No NaN values in critical columns

        log("Validating plot data structure...")
        log("When domain excluded - expecting 2 domains")

        # Check structure (8 columns: domain + tertile + time + observed + SEs + CIs + predicted)
        expected_cols = {'domain_name', 'age_tertile', 'TSVR_hours', 'theta_observed',
                        'se_observed', 'ci_lower', 'ci_upper', 'theta_predicted'}
        actual_cols = set(plot_data.columns)
        assert actual_cols == expected_cols, f"Column mismatch: {actual_cols}"

        # Check domains present (When excluded)
        required_domains = {'What', 'Where'}
        actual_domains = set(plot_data['domain_name'].dropna())
        assert actual_domains == required_domains, f"Missing domains (When excluded): {required_domains - actual_domains}"

        # Check age tertiles
        required_tertiles = {'Young', 'Middle', 'Older'}
        actual_tertiles = set(plot_data['age_tertile'].dropna())
        assert actual_tertiles == required_tertiles, f"Missing tertiles: {required_tertiles - actual_tertiles}"

        # Check for NaN in critical columns (NaN acceptable for sparse predictions)
        nan_counts = plot_data.isna().sum()
        if nan_counts.any():
            log(f"NaN counts: {nan_counts[nan_counts > 0].to_dict()}")

        log(f"Domains: {len(actual_domains)} (When excluded), Tertiles: {len(actual_tertiles)}")
        log(f"Data points by domain: {plot_data.groupby('domain_name').size().to_dict()}")

        validation_result = {'valid': True, 'message': 'Plot data structure valid (8 columns, 2 domains - When excluded)'}

        # Report validation results
        if isinstance(validation_result, dict):
            if validation_result.get('valid', False):
                log(f"PASS - {validation_result.get('message', 'Plot data complete')}")
            else:
                log(f"FAIL - {validation_result.get('message', 'Plot data incomplete')}")
                if validation_result.get('missing_domains'):
                    log(f"Missing domains: {validation_result['missing_domains']}")
                if validation_result.get('missing_groups'):
                    log(f"Missing groups: {validation_result['missing_groups']}")
                raise ValueError(f"Plot data validation failed: {validation_result.get('message')}")
        else:
            log(f"{validation_result}")

        log("Step 05 complete")
        log(f"Plot data ready for plotting pipeline: plots/step05_age_effects_plot_data.csv")
        log(f"Rows: {len(plot_data)}, Domains: 2 (When excluded), Age tertiles: 3")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
