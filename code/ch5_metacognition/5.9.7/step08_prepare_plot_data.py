#!/usr/bin/env python3
"""
Step 08: Prepare Plot Data
RQ 6.9.7 - Paradigm-Specific Calibration Trajectory

PURPOSE: Format data for calibration trajectory plots (mean + individual trajectories)
"""

import sys
from pathlib import Path
import pandas as pd
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_plot_data.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 8: prepare_plot_data")

        # Load data
        calibration_path = RQ_DIR / "data" / "step02_calibration_scores.csv"
        df_calibration = pd.read_csv(calibration_path, encoding='utf-8')
        log(f"{calibration_path.name} ({len(df_calibration)} rows)")

        desc_path = RQ_DIR / "data" / "step06_paradigm_by_time.csv"
        df_desc = pd.read_csv(desc_path, encoding='utf-8')
        log(f"{desc_path.name} ({len(df_desc)} rows)")
        # Plot 1: Mean Trajectories with CIs

        log("Creating mean trajectory plot data...")

        plot_mean = df_desc[['paradigm', 'TSVR_hours_mean', 'mean_calibration', 'ci_lower', 'ci_upper']].copy()
        plot_mean = plot_mean.rename(columns={'TSVR_hours_mean': 'TSVR_hours'})
        plot_mean = plot_mean.sort_values(['paradigm', 'TSVR_hours'])

        out_mean = RQ_DIR / "data" / "step08_plot_mean_trajectories.csv"
        plot_mean.to_csv(out_mean, index=False, encoding='utf-8')
        log(f"{out_mean.name} ({len(plot_mean)} rows)")
        # Plot 2: Individual Trajectories (Spaghetti Plot)

        log("Creating individual trajectory plot data...")

        plot_individual = df_calibration[['UID', 'paradigm', 'TSVR_hours', 'calibration']].copy()
        plot_individual = plot_individual.sort_values(['UID', 'paradigm', 'TSVR_hours'])

        out_individual = RQ_DIR / "data" / "step08_plot_individual_trajectories.csv"
        plot_individual.to_csv(out_individual, index=False, encoding='utf-8')
        log(f"{out_individual.name} ({len(plot_individual)} rows)")
        # Plot 3: Model Predictions (Optional)

        log("Model predictions plot data (optional) - skipped")
        log("Can be generated from LMM fitted values if needed")
        # Validate Plot Data Completeness

        log("Validating plot data completeness...")

        validation_result = validate_plot_data_completeness(
            plot_data=plot_mean,
            required_domains=['free_recall', 'cued_recall', 'recognition'],
            required_groups=[],
            domain_col='paradigm',
            group_col='UID'
        )

        if not validation_result.get('valid', False):
            log(f"Plot data validation failed: {validation_result.get('message', 'Unknown')}")
            sys.exit(1)

        log("Plot data validation successful")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
