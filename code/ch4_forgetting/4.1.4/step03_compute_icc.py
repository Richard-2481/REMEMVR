#!/usr/bin/env python3
"""compute_icc: Quantify proportion of variance that is between-person (stable individual"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import compute_icc_from_variance_components

from tools.validation import validate_icc_bounds

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_icc_computation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Compute ICC")
        # Load Input Data

        log("Loading variance components from Step 2...")
        variance_components = pd.read_csv(RQ_DIR / "data" / "step02_variance_components.csv")
        log(f"step02_variance_components.csv ({len(variance_components)} rows, {len(variance_components.columns)} cols)")

        # Validate input format
        expected_cols = ['component', 'estimate']
        if list(variance_components.columns) != expected_cols:
            raise ValueError(f"Input columns mismatch. Expected {expected_cols}, got {list(variance_components.columns)}")

        if len(variance_components) != 5:
            raise ValueError(f"Expected 5 variance components, got {len(variance_components)} rows")

        log("[INPUT VALIDATION] Variance components format validated")
        # Run Analysis Tool
        #   1. ICC_intercept: Baseline stability (between-person variance in intercepts)
        #   2. ICC_slope_simple: Forgetting rate stability (simple ratio of slope variance)
        #   3. ICC_slope_conditional: Conditional ICC (accounts for intercept-slope correlation at timepoint)

        log("Running compute_icc_from_variance_components...")
        log(f"time_point=None (simple ICC, no conditional adjustment)")
        log(f"slope_name='TSVR_hours' (matches LMM time variable from RQ 5.7)")

        # Transform component names to match function expectations
        # Function expects: 'Intercept', 'Residual', slope_name (e.g., 'TSVR_hours')
        # We have: 'var_intercept', 'var_residual', 'var_slope', 'cov_int_slope', 'cor_int_slope'
        component_mapping = {
            'var_intercept': 'Intercept',
            'var_residual': 'Residual',
            'var_slope': 'TSVR_hours',  # slope_name parameter
            'cov_int_slope': 'Intercept:TSVR_hours',  # Covariance between intercept and slope
            'cor_int_slope': 'correlation'  # Not used by function, keep for completeness
        }

        variance_components_renamed = variance_components.copy()
        variance_components_renamed['component'] = variance_components_renamed['component'].map(component_mapping)
        variance_components_renamed = variance_components_renamed.rename(columns={'estimate': 'variance'})

        log("Mapped component names for function compatibility:")
        log(f"  var_intercept → Intercept")
        log(f"  var_residual → Residual")
        log(f"  var_slope → TSVR_hours")
        log(f"  cov_int_slope → Intercept:TSVR_hours")
        log("Renamed 'estimate' → 'variance'")

        icc_estimates = compute_icc_from_variance_components(
            variance_components_df=variance_components_renamed,
            time_point=None,  # None triggers simple ICC (ICC_intercept, ICC_slope_simple only - no conditional)
            slope_name='TSVR_hours'  # Matches LMM time variable from RQ 5.7
        )

        log("ICC computation complete")
        log(f"Computed {len(icc_estimates)} ICC estimates")
        # Save Analysis Outputs
        # These outputs answer RQ 5.13's primary question about trait stability

        log(f"Saving data/step03_icc_estimates.csv...")
        # Output: data/step03_icc_estimates.csv
        # Contains: ICC estimates with interpretations (intercept, slope_simple, slope_conditional)
        # Columns: icc_type, icc_value, interpretation
        icc_estimates.to_csv(RQ_DIR / "data" / "step03_icc_estimates.csv", index=False, encoding='utf-8')
        log(f"step03_icc_estimates.csv ({len(icc_estimates)} rows, {len(icc_estimates.columns)} cols)")

        # Generate plain text summary
        log(f"Saving results/step03_icc_summary.txt...")
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("RQ 5.13: Intraclass Correlation Coefficient (ICC) Estimates")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("PURPOSE:")
        summary_lines.append("Quantify proportion of variance in forgetting trajectories that reflects")
        summary_lines.append("stable individual differences (between-person variance) vs measurement")
        summary_lines.append("noise or within-person fluctuation (residual variance).")
        summary_lines.append("")
        summary_lines.append("METHODOLOGY:")
        summary_lines.append("ICC = Between-person variance / (Between-person + Within-person variance)")
        summary_lines.append("")
        summary_lines.append("INTERPRETATION THRESHOLDS:")
        summary_lines.append("  - ICC < 0.20: Low (forgetting rate mostly noise)")
        summary_lines.append("  - 0.20 <= ICC < 0.40: Moderate (mixed trait/state)")
        summary_lines.append("  - ICC >= 0.40: Substantial (forgetting rate is trait-like)")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("RESULTS:")
        summary_lines.append("-" * 80)
        summary_lines.append("")

        for _, row in icc_estimates.iterrows():
            icc_type = row['icc_type']
            icc_value = row['icc_value']
            interpretation = row['interpretation']

            summary_lines.append(f"{icc_type.upper()}:")
            summary_lines.append(f"  ICC = {icc_value:.3f}")
            summary_lines.append(f"  Interpretation: {interpretation}")

            # Add implications
            if icc_type == 'intercept':
                summary_lines.append(f"  Implication: {icc_value*100:.1f}% of variance in baseline memory ability")
                summary_lines.append(f"                reflects stable individual differences (trait-like)")
            elif 'slope' in icc_type:
                summary_lines.append(f"  Implication: {icc_value*100:.1f}% of variance in forgetting rate")
                summary_lines.append(f"                reflects stable individual differences (trait-like)")

            summary_lines.append("")

        summary_lines.append("-" * 80)
        summary_lines.append("HYPOTHESIS EVALUATION:")
        summary_lines.append("-" * 80)
        summary_lines.append("")

        # Find slope ICC (either slope_simple or slope_conditional)
        slope_icc_rows = icc_estimates[icc_estimates['icc_type'].str.contains('slope')]
        if len(slope_icc_rows) > 0:
            # Use first slope ICC for hypothesis test
            slope_icc = slope_icc_rows.iloc[0]['icc_value']

            if slope_icc >= 0.40:
                summary_lines.append(f"HYPOTHESIS SUPPORTED: ICC_slope = {slope_icc:.3f} >= 0.40")
                summary_lines.append("Forgetting rates reflect substantial stable individual differences")
                summary_lines.append("(trait-like property, not just measurement noise).")
            elif slope_icc >= 0.20:
                summary_lines.append(f"HYPOTHESIS PARTIALLY SUPPORTED: ICC_slope = {slope_icc:.3f}")
                summary_lines.append("Forgetting rates show moderate individual differences")
                summary_lines.append("(mixed trait/state property).")
            else:
                summary_lines.append(f"HYPOTHESIS NOT SUPPORTED: ICC_slope = {slope_icc:.3f} < 0.20")
                summary_lines.append("Forgetting rates primarily reflect measurement noise")
                summary_lines.append("(low between-person stability).")

        summary_lines.append("")
        summary_lines.append("=" * 80)

        summary_text = "\n".join(summary_lines)
        with open(RQ_DIR / "results" / "step03_icc_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)

        log(f"results/step03_icc_summary.txt")
        # Run Validation Tool
        # Validates: All ICC values in [0, 1] range, no NaN/infinite values
        # Threshold: Mathematical constraint (ICC is proportion of variance)

        log("Running validate_icc_bounds...")
        validation_result = validate_icc_bounds(
            icc_df=icc_estimates,
            icc_col='icc_value'  # Column containing ICC estimates
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check validation passed
        if not validation_result.get('valid', False):
            error_msg = validation_result.get('message', 'ICC validation failed')
            raise ValueError(f"ICC validation failed: {error_msg}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
