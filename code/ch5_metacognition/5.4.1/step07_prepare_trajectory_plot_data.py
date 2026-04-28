#!/usr/bin/env python3
"""Prepare Trajectory Plot Data (Dual-Scale per Decision D069): Create plot-ready trajectory data with dual-scale outputs (theta + probability scales)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.plotting import convert_theta_to_probability
from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.4.1
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_trajectory_plot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Prepare Trajectory Plot Data")
        # Load Input Data

        log("Loading input data...")

        # Load long-format theta data (has TSVR_hours)
        lmm_input_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_long = pd.read_csv(lmm_input_path, encoding='utf-8')
        log(f"step04_lmm_input.csv ({len(df_long)} rows, {len(df_long.columns)} cols)")

        # Load item parameters (for mean discrimination)
        item_params_path = RQ_DIR / "data" / "step03_pass2_item_params.csv"
        df_items = pd.read_csv(item_params_path, encoding='utf-8')
        log(f"step03_pass2_item_params.csv ({len(df_items)} rows, {len(df_items.columns)} cols)")
        # Aggregate Theta Scores by Domain × Timepoint

        log("Aggregating theta scores by domain and test...")

        # Group by domain and TEST (not TSVR_hours - those vary per participant)
        # Then compute mean TSVR_hours per test as the "time" coordinate
        grouped = df_long.groupby(['paradigm', 'test']).agg({
            'theta': ['mean', 'sem', 'count'],
            'TSVR_hours': 'mean'  # Average time per test
        }).reset_index()

        # Flatten column names
        grouped.columns = ['paradigm', 'test', 'theta', 'se', 'n', 'time']

        # Compute 95% CI using t-distribution
        # CI = mean ± t_critical * SE
        # For large N, t-distribution approaches normal (1.96)
        # For small N, use t.ppf for correct critical value
        grouped['CI_lower'] = grouped.apply(
            lambda row: row['theta'] - stats.t.ppf(0.975, row['n'] - 1) * row['se'] if row['n'] > 1 else np.nan,
            axis=1
        )
        grouped['CI_upper'] = grouped.apply(
            lambda row: row['theta'] + stats.t.ppf(0.975, row['n'] - 1) * row['se'] if row['n'] > 1 else np.nan,
            axis=1
        )

        # Select final columns in desired order
        theta_data = grouped[['time', 'theta', 'CI_lower', 'CI_upper', 'paradigm', 'n']].copy()

        # Sort by domain, then time
        theta_data = theta_data.sort_values(['paradigm', 'time']).reset_index(drop=True)

        log(f"Aggregated to {len(theta_data)} rows (domains × tests)")
        log(f"Domains present: {sorted(theta_data['paradigm'].unique())}")
        log(f"Tests per domain: {theta_data.groupby('paradigm').size().to_dict()}")
        # Save Theta-Scale Plot Data
        # Output: data/step07_trajectory_theta_data.csv
        # Contains: Theta-scale trajectory data (statistical scale)
        # Columns: ['time', 'theta', 'CI_lower', 'CI_upper', 'paradigm', 'n']

        log("Saving theta-scale plot data...")
        theta_output_path = RQ_DIR / "data" / "step07_trajectory_theta_data.csv"
        theta_data.to_csv(theta_output_path, index=False, encoding='utf-8')
        log(f"step07_trajectory_theta_data.csv ({len(theta_data)} rows, {len(theta_data.columns)} cols)")
        # Compute Mean Discrimination Per Domain
        # Uses Pass 2 item parameters (purified items only)

        log("Computing mean discrimination per domain...")

        # Item parameters have domain-specific discrimination columns
        # Discrim_IFR, Discrim_ICR, Discrim_IRE
        # Non-zero values indicate which domain each item loads on

        # Compute mean discrimination per domain
        mean_discrim = {}

        for paradigm in ['IFR', 'ICR', 'IRE']:
            discrim_col = f'Discrim_{paradigm}'

            if discrim_col in df_items.columns:
                # Filter to items loading on this domain (non-zero discrimination)
                domain_items = df_items[df_items[discrim_col] > 0]

                if len(domain_items) > 0:
                    # Compute mean discrimination for this paradigm
                    mean_discrim[paradigm] = domain_items[discrim_col].mean()
                    log(f"Mean discrimination for {paradigm}: {mean_discrim[paradigm]:.4f} ({len(domain_items)} items)")
                else:
                    log(f"No items found for {paradigm} paradigm (paradigm excluded after purification)")
            else:
                log(f"{discrim_col} column not found in item parameters")

        # Check if any domains found
        if not mean_discrim:
            raise ValueError("No domain discriminations computed - check item parameters structure")
        # Transform Theta to Probability Scale
        # Formula: P = 1 / (1 + exp(-a * (theta - b)))
        #
        # CRITICAL FIX (2025-12-11): GRM confidence theta is systematically negative
        # (mean ≈ -0.78) unlike 2PL accuracy theta which centers at 0. Using b=0
        # produces misleadingly low probabilities (2-20%). The statistically rigorous
        # solution is to use b = mean(theta) to get interpretable probabilities
        # representing "probability relative to average participant" (EAP normalization).

        log("Transforming theta to probability scale...")

        # Create copy for probability transformation
        probability_data = theta_data.copy()

        # Compute sample mean theta for centering (EAP normalization)
        # This ensures probabilities are interpretable relative to the sample
        sample_mean_theta = theta_data['theta'].mean()
        log(f"Sample mean theta: {sample_mean_theta:.4f} (used for probability centering)")

        # Transform theta and CIs to probability for each domain
        for domain in probability_data['paradigm'].unique():
            # Get mean discrimination for this domain
            if domain not in mean_discrim:
                log(f"No mean discrimination for {domain}, skipping probability transformation")
                continue

            a = mean_discrim[domain]
            # Use sample mean theta as difficulty parameter (EAP normalization)
            # This produces interpretable probabilities centered around 50% for average theta
            b = sample_mean_theta

            # Transform theta to probability
            domain_mask = probability_data['paradigm'] == domain

            probability_data.loc[domain_mask, 'probability'] = convert_theta_to_probability(
                theta=probability_data.loc[domain_mask, 'theta'].values,
                discrimination=a,
                difficulty=b
            )

            # Transform CI bounds to probability
            probability_data.loc[domain_mask, 'CI_lower_prob'] = convert_theta_to_probability(
                theta=probability_data.loc[domain_mask, 'CI_lower'].values,
                discrimination=a,
                difficulty=b
            )

            probability_data.loc[domain_mask, 'CI_upper_prob'] = convert_theta_to_probability(
                theta=probability_data.loc[domain_mask, 'CI_upper'].values,
                discrimination=a,
                difficulty=b
            )

        # Drop theta-scale columns, rename probability columns
        probability_data = probability_data.drop(columns=['theta', 'CI_lower', 'CI_upper'])
        probability_data = probability_data.rename(columns={
            'probability': 'probability',
            'CI_lower_prob': 'CI_lower',
            'CI_upper_prob': 'CI_upper'
        })

        # Reorder columns to match theta_data structure
        probability_data = probability_data[['time', 'probability', 'CI_lower', 'CI_upper', 'paradigm', 'n']]

        log("Probability transformation complete")
        # Save Probability-Scale Plot Data
        # Output: data/step07_trajectory_probability_data.csv
        # Contains: Probability-scale trajectory data (interpretable scale)
        # Columns: ['time', 'probability', 'CI_lower', 'CI_upper', 'paradigm', 'n']

        log("Saving probability-scale plot data...")
        probability_output_path = RQ_DIR / "data" / "step07_trajectory_probability_data.csv"
        probability_data.to_csv(probability_output_path, index=False, encoding='utf-8')
        log(f"step07_trajectory_probability_data.csv ({len(probability_data)} rows, {len(probability_data.columns)} cols)")
        # Validation
        # Validates: Probability values in [0, 1], CI bounds valid

        log("Running validation checks...")

        # Check 1: Both files exist and have same row count
        if len(theta_data) != len(probability_data):
            raise ValueError(f"Row count mismatch: theta_data={len(theta_data)}, probability_data={len(probability_data)}")
        log(f"Both datasets have {len(theta_data)} rows")

        # Check 2: Expected row count (8 or 12 based on domains present)
        n_domains = len(theta_data['paradigm'].unique())
        n_timepoints = len(theta_data['time'].unique())
        expected_rows = n_domains * n_timepoints
        if len(theta_data) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows ({n_domains} domains × {n_timepoints} timepoints), got {len(theta_data)}")
        log(f"Row count matches expected: {expected_rows} rows ({n_domains} domains × {n_timepoints} timepoints)")

        # Check 3: No NaN values in theta data
        if theta_data.isnull().any().any():
            raise ValueError("NaN values found in theta_data")
        log("No NaN values in theta_data")

        # Check 4: No NaN values in probability data
        if probability_data.isnull().any().any():
            raise ValueError("NaN values found in probability_data")
        log("No NaN values in probability_data")

        # Check 5: CI bounds valid (lower < upper) for theta scale
        invalid_ci_theta = theta_data[theta_data['CI_lower'] >= theta_data['CI_upper']]
        if len(invalid_ci_theta) > 0:
            raise ValueError(f"Invalid CI bounds in theta_data: {len(invalid_ci_theta)} rows with CI_lower >= CI_upper")
        log("All theta CI bounds valid (CI_lower < CI_upper)")

        # Check 6: CI bounds valid (lower < upper) for probability scale
        invalid_ci_prob = probability_data[probability_data['CI_lower'] >= probability_data['CI_upper']]
        if len(invalid_ci_prob) > 0:
            raise ValueError(f"Invalid CI bounds in probability_data: {len(invalid_ci_prob)} rows with CI_lower >= CI_upper")
        log("All probability CI bounds valid (CI_lower < CI_upper)")

        # Check 7: Probability range validation using tools.validation
        validation_result = validate_probability_range(
            probability_df=probability_data,
            prob_columns=['probability', 'CI_lower', 'CI_upper']
        )

        if not validation_result['valid']:
            raise ValueError(f"Probability range validation failed: {validation_result['message']}")
        log(f"{validation_result['message']}")

        # Final summary
        log("All validation checks passed")
        log(f"Theta range: [{theta_data['theta'].min():.4f}, {theta_data['theta'].max():.4f}]")
        log(f"Probability range: [{probability_data['probability'].min():.4f}, {probability_data['probability'].max():.4f}]")

        log("Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
