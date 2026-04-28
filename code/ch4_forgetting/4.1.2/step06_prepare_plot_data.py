#!/usr/bin/env python3
"""Prepare Plot Data: Aggregate observed means and model predictions for piecewise vs continuous"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.plotting import prepare_piecewise_plot_data

from tools.validation import validate_plot_data_completeness

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_prepare_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Prepare Plot Data")
        # Load Input Data
        #           Piecewise predictions (Step 3)

        log("Loading observed theta scores from Step 0...")
        observed_data = pd.read_csv(RQ_DIR / "data" / "step00_theta_tsvr.csv", encoding='utf-8')
        log(f"step00_theta_tsvr.csv ({len(observed_data)} rows, {len(observed_data.columns)} cols)")
        log(f"Observed data columns: {list(observed_data.columns)}")

        log("Loading quadratic model predictions from Step 2...")
        quadratic_preds = pd.read_csv(RQ_DIR / "data" / "step02_quadratic_predictions.csv", encoding='utf-8')
        log(f"step02_quadratic_predictions.csv ({len(quadratic_preds)} rows, {len(quadratic_preds.columns)} cols)")
        log(f"Quadratic predictions columns: {list(quadratic_preds.columns)}")

        log("Loading piecewise model predictions from Step 3...")
        piecewise_preds = pd.read_csv(RQ_DIR / "data" / "step03_piecewise_predictions.csv", encoding='utf-8')
        log(f"step03_piecewise_predictions.csv ({len(piecewise_preds)} rows, {len(piecewise_preds.columns)} cols)")
        log(f"Piecewise predictions columns: {list(piecewise_preds.columns)}")
        # Aggregate Observed Data by Test Session
        # Aggregate observed theta scores by test session (4 timepoints)
        # Compute mean theta and 95% CI per timepoint

        log("Computing observed means per test session...")

        # Group by test session, compute mean theta and SEM
        observed_agg = observed_data.groupby('test')['theta'].agg(['mean', 'sem', 'count']).reset_index()

        # Compute 95% CI: mean ± 1.96 × SEM
        observed_agg['CI_lower'] = observed_agg['mean'] - 1.96 * observed_agg['sem']
        observed_agg['CI_upper'] = observed_agg['mean'] + 1.96 * observed_agg['sem']

        # Merge with TSVR_hours mapping (median TSVR per test session)
        tsvr_mapping = observed_data.groupby('test')['TSVR_hours'].median().reset_index()
        observed_agg = observed_agg.merge(tsvr_mapping, on='test', how='left')

        # Rename columns for consistency
        observed_agg = observed_agg.rename(columns={'mean': 'theta'})

        # Add source column
        observed_agg['source'] = 'Observed'
        observed_agg['Segment'] = None  # No segment for observed data

        # Select final columns
        observed_agg = observed_agg[['source', 'TSVR_hours', 'theta', 'CI_lower', 'CI_upper', 'Segment']]

        log(f"Observed data: {len(observed_agg)} timepoints (4 test sessions)")
        log(f"Observed aggregated shape: {observed_agg.shape}")
        # Format Quadratic Predictions
        # Quadratic predictions already on prediction grid (0, 24, 48, ..., 240)
        # Just add source column and rename to match output schema

        log("Formatting quadratic model predictions...")

        quadratic_formatted = quadratic_preds.copy()
        quadratic_formatted['source'] = 'Quadratic'
        quadratic_formatted['Segment'] = None  # No segment for continuous model
        quadratic_formatted = quadratic_formatted.rename(columns={'Time': 'TSVR_hours'})

        # Select final columns (match observed_agg schema)
        quadratic_formatted = quadratic_formatted[['source', 'TSVR_hours', 'predicted_theta', 'CI_lower', 'CI_upper', 'Segment']]
        quadratic_formatted = quadratic_formatted.rename(columns={'predicted_theta': 'theta'})

        log(f"Quadratic predictions: {len(quadratic_formatted)} timepoints")
        log(f"Quadratic formatted shape: {quadratic_formatted.shape}")
        # Format Piecewise Predictions
        # Piecewise predictions have Segment column (Early/Late)
        # Add source column and select matching schema

        log("Formatting piecewise model predictions...")

        piecewise_formatted = piecewise_preds.copy()
        piecewise_formatted['source'] = 'Piecewise'
        piecewise_formatted = piecewise_formatted.rename(columns={'predicted_theta': 'theta'})

        # Select final columns (TSVR_hours already present from Step 3)
        piecewise_formatted = piecewise_formatted[['source', 'TSVR_hours', 'theta', 'CI_lower', 'CI_upper', 'Segment']]

        log(f"Piecewise predictions: {len(piecewise_formatted)} timepoints")
        log(f"Piecewise formatted shape: {piecewise_formatted.shape}")
        # Combine All Data Sources
        # Stack Observed + Quadratic + Piecewise into single DataFrame

        log("Stacking all data sources...")

        plot_data = pd.concat([observed_agg, quadratic_formatted, piecewise_formatted], ignore_index=True)

        log(f"Total plot data: {len(plot_data)} rows")
        log(f"Expected: 33 rows (4 observed + 11 quadratic + 18 piecewise)")
        log(f"Source counts:")
        for source in plot_data['source'].unique():
            count = len(plot_data[plot_data['source'] == source])
            log(f"  - {source}: {count} rows")
        # Save Combined Plot Data
        # Save to plots/ folder (plot-ready CSV, not intermediate data)

        output_path = RQ_DIR / "plots" / "step06_piecewise_comparison_data.csv"
        log(f"Saving combined plot data to {output_path.name}...")

        plot_data.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(plot_data)} rows, {len(plot_data.columns)} cols)")
        # Run Validation Tool
        # Validate all 3 sources present, expected row count, no NaN in critical columns

        log("Running validate_plot_data_completeness...")

        validation_result = validate_plot_data_completeness(
            plot_data=plot_data,
            required_domains=['Observed', 'Quadratic', 'Piecewise'],  # Using 'source' as domain
            required_groups=['Early', 'Late'],  # Segments (only Piecewise has these)
            domain_col='source',
            group_col='Segment'
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional manual validation (expected row count)
        if len(plot_data) != 33:
            log(f"Expected 33 rows, got {len(plot_data)}")
        else:
            log("Row count correct: 33 rows")

        # Check for NaN in critical columns
        critical_cols = ['source', 'TSVR_hours', 'theta', 'CI_lower', 'CI_upper']
        nan_check = plot_data[critical_cols].isna().sum()
        if nan_check.any():
            log(f"NaN values found in critical columns: {nan_check[nan_check > 0].to_dict()}")
        else:
            log("No NaN in critical columns")

        # Check Segment can be NA for non-piecewise sources
        non_piecewise_segment = plot_data[plot_data['source'] != 'Piecewise']['Segment']
        if non_piecewise_segment.isna().all():
            log("Segment correctly NA for non-piecewise sources")
        else:
            log(f"Segment should be NA for non-piecewise sources: {non_piecewise_segment.value_counts()}")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
