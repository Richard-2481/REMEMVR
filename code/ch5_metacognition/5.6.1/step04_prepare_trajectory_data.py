#!/usr/bin/env python3
"""Prepare trajectory plot data: Aggregate HCE rates by timepoint for trajectory plot with 95% confidence intervals."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.6.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_prepare_trajectory_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Prepare trajectory plot data")
        # Load Input Data

        log("Loading HCE rates from Step 1...")
        input_path = RQ_DIR / "data" / "step01_hce_rates.csv"
        df_hce_rates = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df_hce_rates)} rows, {len(df_hce_rates.columns)} cols)")
        log(f"Columns: {list(df_hce_rates.columns)}")
        log(f"Tests present: {sorted(df_hce_rates['TEST'].unique())}")
        # Aggregate by Timepoint

        log("Aggregating HCE rates by timepoint...")

        # Group by TEST and compute statistics
        grouped = df_hce_rates.groupby('TEST', as_index=False).agg({
            'TSVR': 'mean',           # Mean time per timepoint (for x-axis)
            'HCE_rate': ['mean', 'std', 'count']  # Mean, SD, N for confidence intervals
        })

        # Flatten column names from multi-level index
        grouped.columns = ['TEST', 'time', 'HCE_rate_mean', 'HCE_rate_std', 'n_participants']

        log(f"Aggregated to {len(grouped)} timepoints")
        log(f"Timepoints: {list(grouped['TEST'].values)}")
        log(f"Sample sizes per timepoint: {list(grouped['n_participants'].values)}")
        # Compute Confidence Intervals
        # Standard error: SE = SD / sqrt(N)
        # 95% CI: mean ± 1.96*SE (assumes normality via Central Limit Theorem)

        log("Computing 95% confidence intervals...")

        # Compute standard error
        grouped['SE'] = grouped['HCE_rate_std'] / np.sqrt(grouped['n_participants'])

        # Compute 95% confidence intervals
        z_critical = 1.96  # Z-score for 95% confidence level
        grouped['CI_lower'] = grouped['HCE_rate_mean'] - z_critical * grouped['SE']
        grouped['CI_upper'] = grouped['HCE_rate_mean'] + z_critical * grouped['SE']

        # Clip CI bounds to [0, 1] (proportions cannot be negative or >1)
        grouped['CI_lower'] = grouped['CI_lower'].clip(lower=0.0)
        grouped['CI_upper'] = grouped['CI_upper'].clip(upper=1.0)

        log("Confidence intervals computed")
        for idx, row in grouped.iterrows():
            log(f"{row['TEST']}: mean={row['HCE_rate_mean']:.4f}, "
                f"CI=[{row['CI_lower']:.4f}, {row['CI_upper']:.4f}], "
                f"SE={row['SE']:.4f}")
        # Prepare Plot-Ready Output
        # Format: time, HCE_rate_mean, CI_lower, CI_upper, test
        # Rename TEST to 'test' for consistency with plot specification

        log("Preparing plot-ready output...")

        # Select and rename columns
        df_plot_data = grouped[['time', 'HCE_rate_mean', 'CI_lower', 'CI_upper', 'TEST']].copy()
        df_plot_data.rename(columns={'TEST': 'test'}, inplace=True)

        # Convert test column to string (required for validation)
        df_plot_data['test'] = df_plot_data['test'].astype(int).astype(str)

        # Sort by time (chronological order)
        df_plot_data = df_plot_data.sort_values('time').reset_index(drop=True)

        log(f"Plot data prepared: {len(df_plot_data)} rows, {len(df_plot_data.columns)} cols")
        # Save Output

        output_path = RQ_DIR / "data" / "step04_hce_trajectory_data.csv"
        df_plot_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_plot_data)} rows, {len(df_plot_data.columns)} cols)")
        # Validation
        # Validates: All 4 tests present (T1, T2, T3, T4)

        log("Running validate_plot_data_completeness...")

        validation_result = validate_plot_data_completeness(
            plot_data=df_plot_data,
            required_domains=[],                    # No domain grouping
            required_groups=["1", "2", "3", "4"],   # Test sessions as strings
            domain_col="test",
            group_col="test"
        )

        log(f"{validation_result['message']}")

        if not validation_result['valid']:
            error_msg = f"Validation failed: {validation_result['message']}"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        # Additional validation checks (beyond validate_plot_data_completeness)
        log("Running additional checks...")

        # Check exactly 4 rows
        if len(df_plot_data) != 4:
            error_msg = f"Expected exactly 4 rows, found {len(df_plot_data)}"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        # Check no NaN values
        if df_plot_data.isnull().any().any():
            nan_cols = df_plot_data.columns[df_plot_data.isnull().any()].tolist()
            error_msg = f"NaN values detected in columns: {nan_cols}"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        # Check CI_upper > CI_lower
        invalid_ci = df_plot_data[df_plot_data['CI_upper'] <= df_plot_data['CI_lower']]
        if len(invalid_ci) > 0:
            error_msg = f"Invalid confidence intervals found (CI_upper <= CI_lower) for tests: {invalid_ci['test'].tolist()}"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        # Check time values monotonically increasing
        if not df_plot_data['time'].is_monotonic_increasing:
            error_msg = "Time values not monotonically increasing"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        # Check HCE_rate_mean in [0, 1]
        out_of_range = df_plot_data[(df_plot_data['HCE_rate_mean'] < 0) | (df_plot_data['HCE_rate_mean'] > 1)]
        if len(out_of_range) > 0:
            error_msg = f"HCE_rate_mean out of [0, 1] range for tests: {out_of_range['test'].tolist()}"
            log(f"{error_msg}")
            raise ValueError(error_msg)

        log("All checks passed")
        log(f"Plot data preparation complete: {len(df_plot_data)} timepoints created")
        log(f"All tests represented: {', '.join(map(str, sorted(df_plot_data['test'].values)))}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
