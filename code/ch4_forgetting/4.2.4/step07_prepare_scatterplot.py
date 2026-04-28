#!/usr/bin/env python3
"""prepare_scatterplot: Create plot source CSV for scatterplots showing IRT vs CTT correlation per domain."""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_scatterplot.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Prepare Scatterplot Data")
        # Load Input Data

        log("Loading input data...")

        # Load IRT theta scores (wide format: one row per composite_ID, columns per domain)
        irt_theta_path = RQ_DIR / "data" / "step00_irt_theta_loaded.csv"
        irt_theta = pd.read_csv(irt_theta_path)
        log(f"step00_irt_theta_loaded.csv ({len(irt_theta)} rows, {len(irt_theta.columns)} cols)")

        # Load CTT scores (long format: one row per composite_ID x domain)
        ctt_scores_path = RQ_DIR / "data" / "step01_ctt_scores.csv"
        ctt_scores = pd.read_csv(ctt_scores_path)
        log(f"step01_ctt_scores.csv ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")

        # Load correlations (one row per domain)
        correlations_path = RQ_DIR / "data" / "step02_correlations.csv"
        correlations = pd.read_csv(correlations_path)
        log(f"step02_correlations.csv ({len(correlations)} rows, {len(correlations.columns)} cols)")
        # Reshape IRT Theta to Long Format
        #               to long format (composite_ID, domain, IRT_score rows)

        log("Converting IRT theta from wide to long format (When excluded)...")

        # Domain mapping: theta column names -> domain labels
        # theta_what -> what
        # theta_where -> where
        # Note: Use lowercase to match CTT domain values
        # theta_when -> EXCLUDED due to floor effects
        domain_mapping = {
            'theta_what': 'what',
            'theta_where': 'where'
            # 'theta_when': 'when' - EXCLUDED
        }

        # Reshape using pd.melt() - NO theta_when
        irt_long = pd.melt(
            irt_theta,
            id_vars=['composite_ID'],
            value_vars=['theta_what', 'theta_where'],  # NO theta_when
            var_name='theta_column',
            value_name='IRT_score'
        )

        # Map theta column names to domain labels
        irt_long['domain'] = irt_long['theta_column'].map(domain_mapping)

        # Drop intermediate theta_column
        irt_long = irt_long[['composite_ID', 'domain', 'IRT_score']]

        log(f"IRT theta long format: {len(irt_long)} rows, {len(irt_long.columns)} cols")
        log(f"Domains: {sorted(irt_long['domain'].unique().tolist())}")
        # Merge IRT and CTT on composite_ID + domain

        log("Joining IRT and CTT scores on composite_ID + domain...")

        # Select only required columns from CTT scores
        ctt_subset = ctt_scores[['composite_ID', 'domain', 'CTT_score']].copy()

        # Merge IRT and CTT
        scatterplot_data = pd.merge(
            irt_long,
            ctt_subset,
            on=['composite_ID', 'domain'],
            how='inner'
        )

        log(f"Combined data: {len(scatterplot_data)} rows")

        # Check for any merge failures (should be 0 unmatched rows)
        n_irt_rows = len(irt_long)
        n_merged_rows = len(scatterplot_data)
        if n_merged_rows < n_irt_rows:
            log(f"Merge resulted in {n_irt_rows - n_merged_rows} unmatched IRT rows")
        # Join with Correlation Coefficients

        log("Adding correlation coefficients for plot annotations...")

        # Select only domain and r columns from correlations
        # Filter out 'Overall' domain (only need what, where for individual scatter plots - When excluded)
        correlations_subset = correlations[correlations['domain'].isin(['what', 'where'])][['domain', 'r']].copy()

        # Join correlations on domain
        scatterplot_data = pd.merge(
            scatterplot_data,
            correlations_subset,
            on='domain',
            how='left'
        )

        log(f"Data with correlations: {len(scatterplot_data)} rows, {len(scatterplot_data.columns)} cols")
        # Sort and Finalize
        # Sort by domain, then composite_ID for consistent ordering

        log("Sorting by domain and composite_ID...")

        scatterplot_data = scatterplot_data.sort_values(['domain', 'composite_ID']).reset_index(drop=True)

        # Select final column order
        final_columns = ['composite_ID', 'domain', 'IRT_score', 'CTT_score', 'r']
        scatterplot_data = scatterplot_data[final_columns]

        log(f"Final data: {len(scatterplot_data)} rows, {len(scatterplot_data.columns)} cols")
        # Save Output
        # Output will be used by plotting pipeline to create scatter plots

        output_path = RQ_DIR / "data" / "step07_scatterplot_data.csv"
        log(f"Saving to {output_path}...")

        scatterplot_data.to_csv(output_path, index=False, encoding='utf-8')

        log(f"step07_scatterplot_data.csv ({len(scatterplot_data)} rows, {len(scatterplot_data.columns)} cols)")

        # Log summary statistics
        log("Data summary:")
        log(f"  - Domains: {sorted(scatterplot_data['domain'].unique().tolist())}")
        log(f"  - IRT_score range: [{scatterplot_data['IRT_score'].min():.2f}, {scatterplot_data['IRT_score'].max():.2f}]")
        log(f"  - CTT_score range: [{scatterplot_data['CTT_score'].min():.3f}, {scatterplot_data['CTT_score'].max():.3f}]")
        log(f"  - Correlation coefficients (r): {scatterplot_data.groupby('domain')['r'].first().to_dict()}")
        # Run Validation (When EXCLUDED)
        # Validates: Both domains present (When excluded), expected row count, no missing data,
        #            IRT scores in [-3, 3], CTT scores in [0, 1]

        log("Running validate_plot_data_completeness (When excluded)...")

        validation_result = validate_plot_data_completeness(
            plot_data=scatterplot_data,
            required_domains=['what', 'where'],  # NO 'when' - excluded
            required_groups=[],  # No group stratification for this plot
            domain_col='domain'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Check validation passed
        if isinstance(validation_result, dict) and not validation_result.get('valid', False):
            raise ValueError(f"Validation failed: {validation_result.get('message', 'Unknown error')}")

        # Additional validations specific to this step
        log("Additional checks...")

        # Check row count (800 = 400 x 2 domains, When excluded)
        expected_rows = 800  # When excluded: 400 x 2 domains
        actual_rows = len(scatterplot_data)
        if actual_rows != expected_rows:
            log(f"Expected {expected_rows} rows, got {actual_rows}")
        else:
            log(f"Row count: {actual_rows} (matches expected - When excluded)")

        # Check for NaN values
        nan_irt = scatterplot_data['IRT_score'].isna().sum()
        nan_ctt = scatterplot_data['CTT_score'].isna().sum()
        if nan_irt > 0 or nan_ctt > 0:
            raise ValueError(f"Found NaN values: IRT_score={nan_irt}, CTT_score={nan_ctt}")
        else:
            log("No NaN values in IRT_score or CTT_score")

        # Check IRT score range
        irt_min, irt_max = scatterplot_data['IRT_score'].min(), scatterplot_data['IRT_score'].max()
        if irt_min < -3 or irt_max > 3:
            log(f"IRT scores outside typical range [-3, 3]: [{irt_min:.2f}, {irt_max:.2f}]")
        else:
            log(f"IRT scores in typical range: [{irt_min:.2f}, {irt_max:.2f}]")

        # Check CTT score range
        ctt_min, ctt_max = scatterplot_data['CTT_score'].min(), scatterplot_data['CTT_score'].max()
        if ctt_min < 0 or ctt_max > 1:
            raise ValueError(f"CTT scores outside [0, 1] range: [{ctt_min:.3f}, {ctt_max:.3f}]")
        else:
            log(f"CTT scores in valid range: [{ctt_min:.3f}, {ctt_max:.3f}]")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
