#!/usr/bin/env python3
"""Prepare Paradigm Plot Data: Create plot source CSV for bar plot with linear trend overlay."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.2
LOG_FILE = RQ_DIR / "logs" / "step03_prepare_paradigm_plot_data.log"

# Paradigm ordering (by retrieval support level)
PARADIGM_ORDER = {
    'free_recall': 1,
    'cued_recall': 2,
    'recognition': 3
}

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Prepare Paradigm Plot Data")
        # Load Input Data
        log("Loading input data...")

        # Load marginal means
        marginal_means_path = RQ_DIR / "data" / "step01_marginal_means.csv"
        if not marginal_means_path.exists():
            raise FileNotFoundError(f"Step 1 output not found: {marginal_means_path}")

        marginal_means = pd.read_csv(marginal_means_path)
        log(f"marginal_means: {len(marginal_means)} rows")

        # Load contrast results
        contrast_path = RQ_DIR / "data" / "step02_linear_trend_contrast.csv"
        if not contrast_path.exists():
            raise FileNotFoundError(f"Step 2 output not found: {contrast_path}")

        contrast_results = pd.read_csv(contrast_path)
        log(f"contrast_results: {len(contrast_results)} rows")

        log(f"Marginal means:\n{marginal_means.to_string()}")
        log(f"Contrast results:\n{contrast_results.to_string()}")
        # Add Paradigm Code
        log("Adding paradigm codes...")

        marginal_means['paradigm_code'] = marginal_means['paradigm'].map(PARADIGM_ORDER)

        # Verify all paradigms got codes
        if marginal_means['paradigm_code'].isnull().any():
            unknown_paradigms = marginal_means.loc[marginal_means['paradigm_code'].isnull(), 'paradigm'].tolist()
            raise ValueError(f"Unknown paradigms: {unknown_paradigms}")

        log(f"Paradigm codes assigned: {dict(zip(marginal_means['paradigm'], marginal_means['paradigm_code']))}")
        # Compute Linear Trend Line Values
        # The linear trend represents the best-fitting line through the paradigm slopes
        # Trend line: For each paradigm_code x, compute:
        #   trend_value = intercept + slope * (x - 2)
        # Where:
        #   - intercept = mean of marginal means (at paradigm_code=2, centered)
        #   - slope = contrast_estimate / 2 (since weights span -1 to +1, range is 2)

        log("Computing linear trend line values...")

        contrast_estimate = contrast_results['estimate'].iloc[0]
        log(f"Contrast estimate: {contrast_estimate:.4f}")

        # The contrast estimate IS the slope of the linear trend
        # Since contrast weights are [-1, 0, +1] with span of 2 units,
        # the actual slope per unit change in paradigm_code is contrast_estimate / 2
        trend_slope = contrast_estimate / 2.0

        # Intercept is the mean marginal mean (value at center paradigm_code = 2)
        # But since Cued_Recall (code=2) might not be exactly at the mean,
        # we compute: intercept = mean_mm - slope * (2 - 2) = mean_mm
        mean_marginal_mean = marginal_means['marginal_mean'].mean()
        trend_intercept = mean_marginal_mean

        log(f"Trend slope: {trend_slope:.4f}")
        log(f"Trend intercept (mean MM): {trend_intercept:.4f}")

        # Compute trend line values: trend = intercept + slope * (code - 2)
        marginal_means['trend_line'] = trend_intercept + trend_slope * (marginal_means['paradigm_code'] - 2)

        log(f"Trend line values:")
        for _, row in marginal_means.iterrows():
            log(f"  {row['paradigm']}: marginal_mean={row['marginal_mean']:.4f}, trend={row['trend_line']:.4f}")
        # Reorder Columns for Output
        log("Reordering columns...")

        plot_data = marginal_means[['paradigm', 'paradigm_code', 'marginal_mean', 'SE', 'CI_lower', 'CI_upper', 'trend_line']].copy()

        # Sort by paradigm_code
        plot_data = plot_data.sort_values('paradigm_code').reset_index(drop=True)

        log(f"Final plot data:\n{plot_data.to_string()}")
        # Create Annotation Text
        log("Creating contrast annotation text...")

        z_value = contrast_results['z_value'].iloc[0]
        p_uncorrected = contrast_results['p_value_uncorrected'].iloc[0]
        p_bonferroni = contrast_results['p_value_bonferroni'].iloc[0]
        sig_uncorrected = contrast_results['significant_uncorrected'].iloc[0]
        sig_bonferroni = contrast_results['significant_bonferroni'].iloc[0]

        # Format p-value for display
        if p_uncorrected < 0.001:
            p_display = "< .001"
        elif p_uncorrected < 0.01:
            p_display = f"= {p_uncorrected:.3f}"
        else:
            p_display = f"= {p_uncorrected:.2f}"

        # Create annotation (for plot subtitle or note)
        annotation_text = f"Linear trend: b = {contrast_estimate:.3f}, z = {z_value:.2f}, p {p_display}"

        # Also create a more detailed version
        detailed_annotation = f"""Plot Annotation for RQ 5.3.2

MAIN ANNOTATION (for plot subtitle):
{annotation_text}

DETAILED STATISTICS:
- Linear trend contrast estimate: {contrast_estimate:.4f}
- Standard error: {contrast_results['SE'].iloc[0]:.4f}
- z-value: {z_value:.3f}
- p-value (uncorrected): {p_uncorrected:.6f}
- p-value (Bonferroni, n=15): {p_bonferroni:.6f}
- Significant (uncorrected): {sig_uncorrected}
- Significant (Bonferroni): {sig_bonferroni}

TREND LINE PARAMETERS:
- Slope: {trend_slope:.4f}
- Intercept: {trend_intercept:.4f}

INTERPRETATION:
The linear trend {"is" if sig_uncorrected else "is NOT"} statistically significant at alpha = 0.05 (uncorrected).
The linear trend {"is" if sig_bonferroni else "is NOT"} statistically significant after Bonferroni correction.
"""

        log(f"Annotation: {annotation_text}")
        # Validate Output
        log("Checking output validity...")

        # Check row count
        if len(plot_data) != 3:
            raise ValueError(f"Expected 3 rows, got {len(plot_data)}")

        # Check all paradigms present
        expected_paradigms = {'free_recall', 'cued_recall', 'recognition'}
        actual_paradigms = set(plot_data['paradigm'])
        if actual_paradigms != expected_paradigms:
            raise ValueError(f"Expected paradigms {expected_paradigms}, got {actual_paradigms}")

        # Check paradigm codes
        expected_codes = {1, 2, 3}
        actual_codes = set(plot_data['paradigm_code'])
        if actual_codes != expected_codes:
            raise ValueError(f"Expected paradigm codes {expected_codes}, got {actual_codes}")

        # Check for NaN
        if plot_data.isnull().any().any():
            raise ValueError("Output contains NaN values")

        # Check CI ordering
        for _, row in plot_data.iterrows():
            if not row['CI_lower'] < row['CI_upper']:
                raise ValueError(f"CI ordering violated for {row['paradigm']}")

        # Check value ranges (with warning, not error)
        if not plot_data['marginal_mean'].between(-3, 3).all():
            log(f"Some marginal means outside expected range [-3, 3]")

        if not plot_data['trend_line'].between(-3, 3).all():
            log(f"Some trend line values outside expected range [-3, 3]")

        log("All validation checks passed")
        # Save Outputs
        log("Saving output files...")

        # Ensure plots directory exists
        plots_dir = RQ_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save plot data CSV
        output_csv_path = plots_dir / "step03_paradigm_forgetting_rates_data.csv"
        plot_data.to_csv(output_csv_path, index=False, encoding='utf-8')
        log(f"{output_csv_path} ({len(plot_data)} rows)")

        # Save annotation text
        output_txt_path = plots_dir / "step03_contrast_annotation.txt"
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(detailed_annotation)
        log(f"{output_txt_path}")

        log("Step 03 complete - Plot data prepared")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
