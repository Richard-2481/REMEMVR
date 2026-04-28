#!/usr/bin/env python3
"""Prepare Trajectory Plot Data (Decision D069): Aggregate theta scores by congruence and time for dual-scale trajectory"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_trajectory_plot_data.log"

# Input files
INPUT_LMM = RQ_DIR / "data" / "step04_lmm_input.csv"

# Output files
OUTPUT_THETA_PLOT = RQ_DIR / "plots" / "step07_trajectory_theta_data.csv"
OUTPUT_PROB_PLOT = RQ_DIR / "plots" / "step07_trajectory_probability_data.csv"

# IRT Transform Parameters (for probability scale)
# Using 2PL with a=1.0 (average discrimination) and b=0.0 (average difficulty)
IRT_A = 1.0
IRT_B = 0.0

# CI multiplier (1.96 for 95% CI)
CI_MULTIPLIER = 1.96

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# IRT Transform Functions

def theta_to_probability(theta, a=1.0, b=0.0):
    """
    Transform theta to probability using 2PL model.
    P(X=1|theta) = 1 / (1 + exp(-a * (theta - b)))
    """
    return 1 / (1 + np.exp(-a * (theta - b)))

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Prepare Trajectory Plot Data")
        log(f"RQ Directory: {RQ_DIR}")
        # Load LMM Input Data
        log("\nLoading LMM input data...")

        df_lmm = pd.read_csv(INPUT_LMM)
        log(f"{INPUT_LMM.name} ({len(df_lmm)} rows, {len(df_lmm.columns)} cols)")
        # Aggregate by Congruence and Test
        log("\nComputing group statistics...")

        # Group by congruence and test
        grouped = df_lmm.groupby(["congruence", "test"])

        # Compute aggregates
        agg_df = grouped.agg(
            theta_mean=("theta", "mean"),
            theta_std=("theta", "std"),
            theta_se=("theta", lambda x: x.std() / np.sqrt(len(x))),
            n=("theta", "count"),
            time=("TSVR_hours", "mean")  # Mean TSVR per test
        ).reset_index()

        log(f"{len(agg_df)} groups")
        # Compute 95% CI for Theta Scale
        log("\n[CI] Computing 95% confidence intervals (theta scale)...")

        agg_df["CI_lower_theta"] = agg_df["theta_mean"] - CI_MULTIPLIER * agg_df["theta_se"]
        agg_df["CI_upper_theta"] = agg_df["theta_mean"] + CI_MULTIPLIER * agg_df["theta_se"]

        # Create theta-scale output
        df_theta_plot = agg_df[["time", "theta_mean", "CI_lower_theta", "CI_upper_theta", "congruence"]].copy()
        df_theta_plot = df_theta_plot.rename(columns={
            "CI_lower_theta": "CI_lower",
            "CI_upper_theta": "CI_upper"
        })

        log(f"  Theta mean range: [{df_theta_plot['theta_mean'].min():.3f}, {df_theta_plot['theta_mean'].max():.3f}]")
        # Transform to Probability Scale
        log("\nTransforming to probability scale...")

        # Transform mean and CI bounds to probability
        agg_df["prob_mean"] = theta_to_probability(agg_df["theta_mean"], IRT_A, IRT_B)
        agg_df["CI_lower_prob"] = theta_to_probability(agg_df["CI_lower_theta"], IRT_A, IRT_B)
        agg_df["CI_upper_prob"] = theta_to_probability(agg_df["CI_upper_theta"], IRT_A, IRT_B)

        # Create probability-scale output
        df_prob_plot = agg_df[["time", "prob_mean", "CI_lower_prob", "CI_upper_prob", "congruence"]].copy()
        df_prob_plot = df_prob_plot.rename(columns={
            "CI_lower_prob": "CI_lower",
            "CI_upper_prob": "CI_upper"
        })

        log(f"  Probability mean range: [{df_prob_plot['prob_mean'].min():.3f}, {df_prob_plot['prob_mean'].max():.3f}]")
        # Sort and Finalize
        log("\nSorting and finalizing output...")

        # Sort by congruence and time
        df_theta_plot = df_theta_plot.sort_values(["congruence", "time"]).reset_index(drop=True)
        df_prob_plot = df_prob_plot.sort_values(["congruence", "time"]).reset_index(drop=True)

        log(f"  Theta plot: {len(df_theta_plot)} rows")
        log(f"  Probability plot: {len(df_prob_plot)} rows")
        # Save Outputs
        log("\nSaving output files...")

        # Ensure plots directory exists
        (RQ_DIR / "plots").mkdir(parents=True, exist_ok=True)

        # Save theta-scale data
        df_theta_plot.to_csv(OUTPUT_THETA_PLOT, index=False, encoding='utf-8')
        log(f"{OUTPUT_THETA_PLOT.name} ({len(df_theta_plot)} rows)")

        # Save probability-scale data
        df_prob_plot.to_csv(OUTPUT_PROB_PLOT, index=False, encoding='utf-8')
        log(f"{OUTPUT_PROB_PLOT.name} ({len(df_prob_plot)} rows)")
        # Validation
        log("\nValidating results...")

        # Check both files created
        assert OUTPUT_THETA_PLOT.exists(), f"Theta plot file not created: {OUTPUT_THETA_PLOT}"
        assert OUTPUT_PROB_PLOT.exists(), f"Probability plot file not created: {OUTPUT_PROB_PLOT}"
        log("Both plot files created")

        # Check row count
        expected_rows = 3 * 4  # 3 congruence x 4 tests
        if len(df_theta_plot) != expected_rows:
            log(f"Theta plot has {len(df_theta_plot)} rows (expected {expected_rows})")
        else:
            log(f"Row count: {len(df_theta_plot)} (3 congruence x 4 tests)")

        # Check all congruence categories present
        for cat in ["common", "congruent", "incongruent"]:
            n_cat = len(df_theta_plot[df_theta_plot["congruence"] == cat])
            if n_cat != 4:
                log(f"{cat}: {n_cat} rows (expected 4)")
            else:
                log(f"{cat}: {n_cat} rows")

        # Check theta range
        theta_min = df_theta_plot["theta_mean"].min()
        theta_max = df_theta_plot["theta_mean"].max()
        if theta_min < -3 or theta_max > 3:
            log(f"Theta mean outside [-3, 3]: [{theta_min:.3f}, {theta_max:.3f}]")
        else:
            log(f"Theta mean range: [{theta_min:.3f}, {theta_max:.3f}]")

        # Check probability range
        prob_min = df_prob_plot["prob_mean"].min()
        prob_max = df_prob_plot["prob_mean"].max()
        if prob_min < 0 or prob_max > 1:
            log(f"Probability mean outside [0, 1]: [{prob_min:.3f}, {prob_max:.3f}]")
        else:
            log(f"Probability mean range: [{prob_min:.3f}, {prob_max:.3f}]")

        # Check CI bounds logical
        ci_ok = all(df_theta_plot["CI_lower"] < df_theta_plot["theta_mean"]) and \
                all(df_theta_plot["theta_mean"] < df_theta_plot["CI_upper"])
        if ci_ok:
            log("CI bounds logical (lower < mean < upper)")
        else:
            log("CI bounds may be illogical")

        # Check no NaN
        nan_theta = df_theta_plot.isna().sum().sum()
        nan_prob = df_prob_plot.isna().sum().sum()
        if nan_theta > 0 or nan_prob > 0:
            log(f"NaN values found: theta={nan_theta}, prob={nan_prob}")
        else:
            log("No NaN values")

        # Print summary table
        log("\n  Trajectory Summary (Theta Scale):")
        log("  " + "-" * 60)
        for cat in ["common", "congruent", "incongruent"]:
            cat_data = df_theta_plot[df_theta_plot["congruence"] == cat]
            log(f"  {cat}:")
            for _, row in cat_data.iterrows():
                log(f"    time={row['time']:.1f}h: theta={row['theta_mean']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")

        log("\nStep 07 complete (Trajectory Plot Data)")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
