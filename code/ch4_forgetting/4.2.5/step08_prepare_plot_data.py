#!/usr/bin/env python3
"""Prepare Plot Data (Methodological Comparison Visualization): Create plot source CSVs for methodological comparison visualizations:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.2.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 8: Prepare Plot Data (Methodological Comparison Visualization)")
        # Load Input Data

        log("Loading input data...")

        # Load Step 5 correlation analysis results
        # Contains: domain, r_full_irt, r_purified_irt, delta_r, p_bonferroni, etc.
        df_corr = pd.read_csv(RQ_DIR / "data/step05_correlation_analysis.csv")
        log(f"step05_correlation_analysis.csv ({len(df_corr)} rows, {len(df_corr.columns)} cols)")
        log(f"  Columns: {df_corr.columns.tolist()}")

        # Load Step 7 LMM model comparison
        # Contains: measurement, AIC, delta_AIC, interpretation
        df_aic = pd.read_csv(RQ_DIR / "data/step07_lmm_model_comparison.csv")
        log(f"step07_lmm_model_comparison.csv ({len(df_aic)} rows, {len(df_aic.columns)} cols)")
        log(f"  Columns: {df_aic.columns.tolist()}")
        # Prepare Plot 1 Data (Correlation Comparison)

        log("[PLOT1] Preparing correlation comparison data...")

        # Reshape correlations to long format
        # Original: domain | r_full_irt | r_purified_irt | ...
        # Target:   domain | measurement_type | correlation | significance

        plot1_data = []

        for _, row in df_corr.iterrows():
            domain = row['domain']

            # Full CTT row
            plot1_data.append({
                'domain': domain,
                'measurement_type': 'Full CTT',
                'correlation': row['r_full_irt'],
                'significance': 'Not significant' if row['p_bonferroni'] >= 0.05 else 'Significant improvement'
            })

            # Purified CTT row
            plot1_data.append({
                'domain': domain,
                'measurement_type': 'Purified CTT',
                'correlation': row['r_purified_irt'],
                'significance': 'Not significant' if row['p_bonferroni'] >= 0.05 else 'Significant improvement'
            })

        df_plot1 = pd.DataFrame(plot1_data)

        log(f"[PLOT1] Created correlation comparison data ({len(df_plot1)} rows)")
        log(f"  Domains: {sorted(df_plot1['domain'].unique())}")
        log(f"  Measurement types: {sorted(df_plot1['measurement_type'].unique())}")
        # Prepare Plot 2 Data (AIC Comparison)

        log("[PLOT2] Preparing AIC comparison data...")

        # AIC comparison is already plot-ready (measurement, AIC, delta_AIC, interpretation)
        # Just select relevant columns and copy to plots/
        df_plot2 = df_aic[['measurement', 'AIC', 'delta_AIC', 'interpretation']].copy()

        log(f"[PLOT2] Created AIC comparison data ({len(df_plot2)} rows)")
        log(f"  Measurements: {sorted(df_plot2['measurement'].unique())}")
        log(f"  AIC range: [{df_plot2['AIC'].min():.2f}, {df_plot2['AIC'].max():.2f}]")
        log(f"  Delta AIC range: [{df_plot2['delta_AIC'].min():.2f}, {df_plot2['delta_AIC'].max():.2f}]")
        # Save Plot Data CSVs
        # These outputs will be used by: plotting pipeline to generate PNG visualizations

        log("Saving plot data CSVs...")

        # Save Plot 1 data (correlation comparison)
        # Output: plots/step08_correlation_comparison_data.csv
        # Contains: Grouped bar chart data (Full CTT vs Purified CTT correlations with IRT)
        plot1_path = RQ_DIR / "plots/step08_correlation_comparison_data.csv"
        df_plot1.to_csv(plot1_path, index=False, encoding='utf-8')
        log(f"step08_correlation_comparison_data.csv ({len(df_plot1)} rows, {len(df_plot1.columns)} cols)")

        # Save Plot 2 data (AIC comparison)
        # Output: plots/step08_aic_comparison_data.csv
        # Contains: Bar chart data (Full CTT vs Purified CTT vs IRT theta AIC values)
        plot2_path = RQ_DIR / "plots/step08_aic_comparison_data.csv"
        df_plot2.to_csv(plot2_path, index=False, encoding='utf-8')
        log(f"step08_aic_comparison_data.csv ({len(df_plot2)} rows, {len(df_plot2.columns)} cols)")
        # Run Validation Tool
        # Validates: All domains/measurements present in plot data
        # Threshold: No missing categories (complete factorial design)

        log("Running validate_plot_data_completeness...")

        # Validate Plot 1 data
        validation_result1 = validate_plot_data_completeness(
            plot_data=df_plot1,
            required_domains=['what', 'where'],  # When excluded
            required_groups=['Full CTT', 'Purified CTT'],
            domain_col='domain',
            group_col='measurement_type'
        )

        # Report validation results for Plot 1
        if isinstance(validation_result1, dict):
            log(f"Plot 1 result: {validation_result1.get('message', validation_result1)}")
            if not validation_result1.get('valid', False):
                log(f"Plot 1 validation failed")
                raise ValueError(f"Plot 1 validation failed: {validation_result1.get('message', 'Unknown error')}")
        else:
            log(f"Plot 1 result: {validation_result1}")

        # Validate Plot 2 data
        # Note: Plot 2 doesn't have a domain dimension, so we validate measurements only
        if len(df_plot2) != 3:
            log(f"Plot 2 should have exactly 3 rows (Full CTT, Purified CTT, IRT theta), got {len(df_plot2)}")
            raise ValueError(f"Plot 2 validation failed: Expected 3 rows, got {len(df_plot2)}")

        expected_measurements = ['Full CTT', 'Purified CTT', 'IRT theta']
        actual_measurements = set(df_plot2['measurement'].unique())
        missing_measurements = set(expected_measurements) - actual_measurements

        if missing_measurements:
            log(f"Plot 2 missing measurements: {missing_measurements}")
            raise ValueError(f"Plot 2 validation failed: Missing measurements {missing_measurements}")

        log(f"Plot 2 result: All measurements present (Full CTT, Purified CTT, IRT theta)")

        # Check for NaN values in either plot data
        if df_plot1.isnull().any().any():
            log(f"Plot 1 contains NaN values")
            raise ValueError("Plot 1 validation failed: Contains NaN values")

        if df_plot2.isnull().any().any():
            log(f"Plot 2 contains NaN values")
            raise ValueError("Plot 2 validation failed: Contains NaN values")

        log(f"No NaN values detected in plot data")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
