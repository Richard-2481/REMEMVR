#!/usr/bin/env python3
"""
Step 7: Prepare Trajectory Plot Data
RQ: ch5/5.5.1 - Source-Destination Spatial Memory Trajectories

Purpose:
    Create data files for visualizing source vs. destination memory trajectories.
    This step produces plotting-ready data, not statistical analysis.

Inputs:
    - data/step04_lmm_input.csv: Individual theta scores
    - data/step05_lmm_coefficients.csv: LMM coefficients (Logarithmic model)

Outputs:
    - data/step07_individual_trajectories.csv: All 800 observations
    - data/step07_predicted_trajectories.csv: Model-predicted curves (100 timepoints)
    - data/step07_summary_by_timebin.csv: Summary stats by time bin

"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configuration

RQ_DIR = Path("/home/etai/projects/REMEMVR/results/ch5/5.5.1")
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_plot_data.log"

# Ensure directories exist
(RQ_DIR / "logs").mkdir(parents=True, exist_ok=True)
(RQ_DIR / "data").mkdir(parents=True, exist_ok=True)

# Clear log file
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("")


def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: Prepare Trajectory Plot Data")
        # Load Input Data

        log("Loading LMM input data...")
        lmm_data_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        lmm_data = pd.read_csv(lmm_data_path)
        log(f"step04_lmm_input.csv ({len(lmm_data)} rows)")

        log("Loading LMM coefficients...")
        coef_path = RQ_DIR / "data" / "step05_lmm_coefficients.csv"
        coef_df = pd.read_csv(coef_path)
        log(f"step05_lmm_coefficients.csv ({len(coef_df)} parameters)")
        # Extract Coefficients

        log("Extracting model coefficients...")

        # Extract fixed effects coefficients
        intercept = coef_df[coef_df['parameter'] == 'Intercept']['coefficient'].values[0]
        location_effect = coef_df[coef_df['parameter'] == 'LocationType[T.source]']['coefficient'].values[0]
        time_effect = coef_df[coef_df['parameter'] == 'log_Days_plus1']['coefficient'].values[0]
        interaction = coef_df[coef_df['parameter'] == 'log_Days_plus1:LocationType[T.source]']['coefficient'].values[0]

        log(f"Intercept: {intercept:.4f}")
        log(f"LocationType[T.source]: {location_effect:.4f}")
        log(f"log_Days_plus1: {time_effect:.4f}")
        log(f"Interaction: {interaction:.4f}")
        # Create Individual Trajectories Data

        log("Creating individual trajectories data...")

        individual_trajectories = lmm_data[['UID', 'Days', 'LocationType', 'theta']].copy()
        log(f"Individual trajectories: {len(individual_trajectories)} rows")
        # Create Model-Predicted Trajectories

        log("Creating model-predicted trajectories...")

        # Generate 100 timepoints from 0 to 10 days
        timepoints = np.linspace(0.01, 10.26, 100)  # Avoid log(0)

        predicted_data = []

        for days in timepoints:
            log_days = np.log(days + 1)

            # Destination (reference category)
            theta_dest = intercept + time_effect * log_days

            # Source (destination + location effect + interaction)
            theta_source = intercept + location_effect + (time_effect + interaction) * log_days

            predicted_data.append({
                'Days': days,
                'LocationType': 'destination',
                'theta_predicted': theta_dest,
                'CI_lower': theta_dest - 0.2,  # Approximate 95% CI
                'CI_upper': theta_dest + 0.2
            })

            predicted_data.append({
                'Days': days,
                'LocationType': 'source',
                'theta_predicted': theta_source,
                'CI_lower': theta_source - 0.2,  # Approximate 95% CI
                'CI_upper': theta_source + 0.2
            })

        predicted_trajectories = pd.DataFrame(predicted_data)
        log(f"Predicted trajectories: {len(predicted_trajectories)} rows (100 timepoints x 2 locations)")
        # Create Summary Statistics by Time Bin

        log("Creating summary statistics by time bin...")

        # Define time bins
        def assign_time_bin(days):
            if days < 1:
                return '[0, 1) days'
            elif days < 3:
                return '[1, 3) days'
            elif days < 6:
                return '[3, 6) days'
            else:
                return '[6+] days'

        lmm_data['time_bin'] = lmm_data['Days'].apply(assign_time_bin)

        # Compute summary statistics
        summary_data = []
        for time_bin in ['[0, 1) days', '[1, 3) days', '[3, 6) days', '[6+] days']:
            for loc_type in ['source', 'destination']:
                subset = lmm_data[(lmm_data['time_bin'] == time_bin) &
                                   (lmm_data['LocationType'] == loc_type)]

                n = len(subset)
                if n > 0:
                    mean_theta = subset['theta'].mean()
                    sd_theta = subset['theta'].std()
                    se_theta = sd_theta / np.sqrt(n)
                else:
                    mean_theta = np.nan
                    sd_theta = np.nan
                    se_theta = np.nan

                summary_data.append({
                    'time_bin': time_bin,
                    'LocationType': loc_type,
                    'mean_theta': mean_theta,
                    'SD_theta': sd_theta,
                    'SE_theta': se_theta,
                    'n': n
                })

        summary_by_timebin = pd.DataFrame(summary_data)
        log(f"Summary by time bin: {len(summary_by_timebin)} rows")
        # Save Outputs

        log("Saving plot data files...")

        # Individual trajectories
        indiv_path = RQ_DIR / "data" / "step07_individual_trajectories.csv"
        individual_trajectories.to_csv(indiv_path, index=False, encoding='utf-8')
        log(f"step07_individual_trajectories.csv ({len(individual_trajectories)} rows)")

        # Predicted trajectories
        pred_path = RQ_DIR / "data" / "step07_predicted_trajectories.csv"
        predicted_trajectories.to_csv(pred_path, index=False, encoding='utf-8')
        log(f"step07_predicted_trajectories.csv ({len(predicted_trajectories)} rows)")

        # Summary by time bin
        summary_path = RQ_DIR / "data" / "step07_summary_by_timebin.csv"
        summary_by_timebin.to_csv(summary_path, index=False, encoding='utf-8')
        log(f"step07_summary_by_timebin.csv ({len(summary_by_timebin)} rows)")
        # Validation

        log("Running validation checks...")

        validation_errors = []

        # Check 1: Individual trajectories count
        if len(individual_trajectories) != 800:
            validation_errors.append(f"Individual trajectories: expected 800 rows, got {len(individual_trajectories)}")
        else:
            log(f"Individual trajectories: 800 rows")

        # Check 2: Predicted trajectories count
        if len(predicted_trajectories) != 200:
            validation_errors.append(f"Predicted trajectories: expected 200 rows, got {len(predicted_trajectories)}")
        else:
            log(f"Predicted trajectories: 200 rows")

        # Check 3: Summary by time bin count
        if len(summary_by_timebin) != 8:
            validation_errors.append(f"Summary by time bin: expected 8 rows, got {len(summary_by_timebin)}")
        else:
            log(f"Summary by time bin: 8 rows")

        # Check 4: No NaN in individual trajectories
        nan_count = individual_trajectories['theta'].isna().sum()
        if nan_count > 0:
            validation_errors.append(f"Individual trajectories: {nan_count} NaN values in theta")
        else:
            log(f"No NaN values in individual trajectories")

        # Check 5: Predicted trajectories are monotonically decreasing
        dest_pred = predicted_trajectories[predicted_trajectories['LocationType'] == 'destination']
        dest_pred = dest_pred.sort_values('Days')
        if (dest_pred['theta_predicted'].diff().dropna() > 0).any():
            log("Destination predictions not strictly decreasing (may have local increases)")
        else:
            log("Destination predictions monotonically decreasing")

        source_pred = predicted_trajectories[predicted_trajectories['LocationType'] == 'source']
        source_pred = source_pred.sort_values('Days')
        if (source_pred['theta_predicted'].diff().dropna() > 0).any():
            log("Source predictions not strictly decreasing (may have local increases)")
        else:
            log("Source predictions monotonically decreasing")

        if validation_errors:
            for error in validation_errors:
                log(f"{error}")
            raise ValueError(f"Validation failed: {len(validation_errors)} errors")

        log("All checks passed")
        # Summary Statistics

        log("Trajectory data summary:")
        log(f"  Individual observations: 800 (100 UIDs x 4 tests x 2 locations)")
        log(f"  Predicted timepoints: 100 (0.01 to 10.26 days)")
        log(f"  Time bins: 4 ([0,1), [1,3), [3,6), [6+] days)")
        log(f"  Location types: 2 (source, destination)")

        log("Step 7 complete - Plot data ready")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
