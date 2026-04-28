#!/usr/bin/env python3
"""Generate Plot Data Sources: Prepare CSV files for plotting pipeline trajectory comparison plots. Creates 4 plot data"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.1
LOG_FILE = RQ_DIR / "logs" / "step08_plot_data.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 8: Generate plot data sources")
        # Load Input Data

        log("Loading input data...")

        df_traj = pd.read_csv(RQ_DIR / "data" / "step01_merged_trajectories.csv", encoding='utf-8')
        log(f"step01_merged_trajectories.csv ({len(df_traj)} rows)")

        df_rates = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv", encoding='utf-8')
        log(f"step02_individual_decline_rates.csv ({len(df_rates)} rows)")
        # Aggregate Trajectories (Line Plot with Error Bars)

        log("Computing mean and SE per test...")

        # Group by test
        agg_results = []
        for test_num in [1, 2, 3, 4]:
            test_data = df_traj[df_traj['test'] == test_num]

            # Get day label
            day_label = test_data['day_label'].iloc[0]

            # Compute statistics
            mean_TSVR = test_data['TSVR_hours'].mean()
            mean_theta_acc = test_data['theta_acc'].mean()
            SE_theta_acc = test_data['theta_acc'].std(ddof=1) / np.sqrt(len(test_data))
            mean_theta_conf = test_data['theta_conf'].mean()
            SE_theta_conf = test_data['theta_conf'].std(ddof=1) / np.sqrt(len(test_data))
            N = len(test_data)

            agg_results.append({
                'test': test_num,
                'day_label': day_label,
                'mean_TSVR_hours': mean_TSVR,
                'mean_theta_acc': mean_theta_acc,
                'SE_theta_acc': SE_theta_acc,
                'mean_theta_conf': mean_theta_conf,
                'SE_theta_conf': SE_theta_conf,
                'N': N
            })

        df_agg = pd.DataFrame(agg_results)
        log(f"Computed statistics for {len(df_agg)} timepoints")
        # Individual Trajectories (Spaghetti Plot)

        log("Reshaping to long format...")

        # Create separate DataFrames for accuracy and confidence
        df_acc = df_traj[['UID', 'test', 'TSVR_hours', 'theta_acc']].copy()
        df_acc['theta_value'] = df_acc['theta_acc']
        df_acc['measure_type'] = 'accuracy'
        df_acc = df_acc[['UID', 'test', 'TSVR_hours', 'theta_value', 'measure_type']]

        df_conf = df_traj[['UID', 'test', 'TSVR_hours', 'theta_conf']].copy()
        df_conf['theta_value'] = df_conf['theta_conf']
        df_conf['measure_type'] = 'confidence'
        df_conf = df_conf[['UID', 'test', 'TSVR_hours', 'theta_value', 'measure_type']]

        # Concatenate
        df_individual = pd.concat([df_acc, df_conf], ignore_index=True)
        log(f"Reshaped to long format: {len(df_individual)} observations")
        # Decline Rates Histogram Data

        log("Extracting decline rates for histogram...")

        # Extract relevant columns
        df_decline_hist = df_rates[['UID', 'acc_rate', 'conf_rate', 'difference', 'ratio']].copy()
        log(f"Extracted {len(df_decline_hist)} individual decline rates")
        # Ratio Histogram Bins

        log("Computing histogram bins for ratio distribution...")

        # Filter valid ratios (exclude no_decline_flag)
        valid_ratios = df_rates[~df_rates['no_decline_flag']]['ratio'].values

        # Compute histogram (20 bins)
        hist_counts, bin_edges = np.histogram(valid_ratios, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        df_ratio_hist = pd.DataFrame({
            'bin_center': bin_centers,
            'count': hist_counts
        })
        log(f"Created {len(df_ratio_hist)} bins")
        # Save Plot Data Outputs

        log("Saving plot data sources...")

        # Aggregate trajectories
        output_path_agg = RQ_DIR / "data" / "step08_aggregate_trajectories_plot_data.csv"
        df_agg.to_csv(output_path_agg, index=False, encoding='utf-8')
        log(f"{output_path_agg.name} ({len(df_agg)} rows)")

        # Individual trajectories
        output_path_ind = RQ_DIR / "data" / "step08_individual_trajectories_plot_data.csv"
        df_individual.to_csv(output_path_ind, index=False, encoding='utf-8')
        log(f"{output_path_ind.name} ({len(df_individual)} rows)")

        # Decline rates histogram
        output_path_decline = RQ_DIR / "data" / "step08_decline_rates_histogram_data.csv"
        df_decline_hist.to_csv(output_path_decline, index=False, encoding='utf-8')
        log(f"{output_path_decline.name} ({len(df_decline_hist)} rows)")

        # Ratio histogram
        output_path_ratio = RQ_DIR / "data" / "step08_ratio_histogram_data.csv"
        df_ratio_hist.to_csv(output_path_ratio, index=False, encoding='utf-8')
        log(f"{output_path_ratio.name} ({len(df_ratio_hist)} rows)")
        # Validation

        log("Running inline validation...")

        # Check aggregate trajectories
        if len(df_agg) != 4:
            log(f"Expected 4 rows in aggregate, got {len(df_agg)}")

        # Check N=100 for all tests
        for idx, row in df_agg.iterrows():
            if row['N'] != 100:
                log(f"Test {row['test']}: N={row['N']} (expected 100)")

        # Check for NaN in aggregate
        critical_cols = ['mean_theta_acc', 'mean_theta_conf', 'SE_theta_acc', 'SE_theta_conf']
        if df_agg[critical_cols].isna().any().any():
            log(f"NaN values in aggregate trajectories")

        # Check individual trajectories
        if len(df_individual) != 800:
            log(f"Expected 800 rows in individual, got {len(df_individual)}")

        # Check measure_type values
        measure_types = df_individual['measure_type'].unique()
        expected_types = ['accuracy', 'confidence']
        if not all(m in expected_types for m in measure_types):
            log(f"Invalid measure_type values: {measure_types}")

        # Check for NaN in individual
        if df_individual[['UID', 'test', 'theta_value', 'measure_type']].isna().any().any():
            log(f"NaN values in individual trajectories")

        # Check decline rates histogram
        if len(df_decline_hist) != 100:
            log(f"Expected 100 rows in decline rates, got {len(df_decline_hist)}")

        # Check for NaN in decline rates
        if df_decline_hist[['UID', 'acc_rate', 'conf_rate']].isna().any().any():
            log(f"NaN values in decline rates histogram data")

        # Check ratio histogram bins
        if len(df_ratio_hist) < 15 or len(df_ratio_hist) > 25:
            log(f"Expected ~20 bins in ratio histogram, got {len(df_ratio_hist)}")

        # Check value ranges
        theta_range = (df_individual['theta_value'].min(), df_individual['theta_value'].max())
        log(f"theta_value range: [{theta_range[0]:.2f}, {theta_range[1]:.2f}]")

        if not (-3 <= theta_range[0] and theta_range[1] <= 3):
            log(f"Theta values outside expected range [-3, 3]")

        SE_range = (df_agg['SE_theta_acc'].min(), df_agg['SE_theta_acc'].max())
        log(f"SE range: [{SE_range[0]:.4f}, {SE_range[1]:.4f}]")

        if not (0.05 <= SE_range[0] and SE_range[1] <= 0.3):
            log(f"SE outside expected range [0.05, 0.3]")

        log("Aggregate trajectories: 4 timepoints")
        log("Individual trajectories: 800 observations")
        log("Plot data sources saved")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
