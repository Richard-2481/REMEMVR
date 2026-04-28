#!/usr/bin/env python3
"""Compute Standardized Decline Rates: Compute (T4-T1)/baseline_SD for accuracy and confidence, convert to rates"""

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
LOG_FILE = RQ_DIR / "logs" / "step02_compute_decline_rates.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 2: Compute Standardized Decline Rates")
        # Load Input Data

        log("Loading input data...")

        # Load merged trajectories (400 rows: 100 participants × 4 tests)
        df_traj = pd.read_csv(RQ_DIR / "data" / "step01_merged_trajectories.csv", encoding='utf-8')
        log(f"step01_merged_trajectories.csv ({len(df_traj)} rows, {len(df_traj.columns)} cols)")

        # Load baseline comparison (2 rows: accuracy, confidence)
        df_baseline = pd.read_csv(RQ_DIR / "data" / "step01_baseline_comparison.csv", encoding='utf-8')
        log(f"step01_baseline_comparison.csv ({len(df_baseline)} rows, {len(df_baseline.columns)} cols)")
        # Extract Baseline SDs

        log("Extracting baseline SDs...")

        # Extract SD for accuracy measure
        SD_acc_baseline = df_baseline.loc[df_baseline['measure'] == 'accuracy', 'SD'].values[0]
        log(f"SD_acc_baseline = {SD_acc_baseline:.6f}")

        # Extract SD for confidence measure
        SD_conf_baseline = df_baseline.loc[df_baseline['measure'] == 'confidence', 'SD'].values[0]
        log(f"SD_conf_baseline = {SD_conf_baseline:.6f}")

        # Validation: Both SDs must be > 0
        if SD_acc_baseline <= 0 or SD_conf_baseline <= 0:
            raise ValueError(f"Baseline SD = 0, cannot standardize. SD_acc={SD_acc_baseline}, SD_conf={SD_conf_baseline}")
        # Filter T1/T4 Data

        log("Filtering T1 and T4 data...")

        # Filter to test 1 (Day 0) and test 4 (Day 6)
        df_t1_t4 = df_traj.query("test in [1, 4]").copy()
        log(f"T1/T4 data: {len(df_t1_t4)} rows (expected 200)")

        # Validation: Should have exactly 200 rows
        if len(df_t1_t4) != 200:
            log(f"Expected 200 rows (100 UIDs × 2 tests), got {len(df_t1_t4)}")
        # Compute Per-Participant Decline Rates

        log("Computing per-participant decline rates...")

        results = []

        for uid, group in df_t1_t4.groupby('UID'):
            # Ensure we have exactly 2 rows (T1 and T4)
            if len(group) != 2:
                log(f"UID {uid} has {len(group)} observations (expected 2), skipping")
                continue

            # Extract T1 (test=1) and T4 (test=4) values
            t1_row = group[group['test'] == 1].iloc[0]
            t4_row = group[group['test'] == 4].iloc[0]

            # Extract theta values
            T1_theta_acc = t1_row['theta_acc']
            T4_theta_acc = t4_row['theta_acc']
            T1_theta_conf = t1_row['theta_conf']
            T4_theta_conf = t4_row['theta_conf']

            # Extract time points
            T1_TSVR_hours = t1_row['TSVR_hours']
            T4_TSVR_hours = t4_row['TSVR_hours']

            # Compute time elapsed
            delta_hours = T4_TSVR_hours - T1_TSVR_hours

            # Compute SD decline (raw theta change standardized by baseline SD)
            SD_decline_acc = (T4_theta_acc - T1_theta_acc) / SD_acc_baseline
            SD_decline_conf = (T4_theta_conf - T1_theta_conf) / SD_conf_baseline

            # Compute decline rates (SD/hour)
            acc_rate = SD_decline_acc / delta_hours
            conf_rate = SD_decline_conf / delta_hours

            # Compute difference (positive = conf declines faster)
            difference = conf_rate - acc_rate

            # Compute ratio (>1.0 = conf declines faster)
            # Flag if denominator near zero (edge case)
            no_decline_flag = False
            if abs(acc_rate) < 0.01:
                ratio = np.nan  # Undefined ratio
                no_decline_flag = True
            else:
                ratio = conf_rate / acc_rate

            results.append({
                'UID': uid,
                'acc_rate': acc_rate,
                'conf_rate': conf_rate,
                'difference': difference,
                'ratio': ratio,
                'delta_hours': delta_hours,
                'SD_decline_acc': SD_decline_acc,
                'SD_decline_conf': SD_decline_conf,
                'no_decline_flag': no_decline_flag
            })

        # Convert to DataFrame
        df_rates = pd.DataFrame(results)
        log(f"Decline rates for {len(df_rates)} participants")
        # Screen Practice Effects

        log("Screening for practice effects...")

        # Recompute with full trajectory data for practice effects screening
        df_t1 = df_traj[df_traj['test'] == 1]
        df_t4 = df_traj[df_traj['test'] == 4]

        # Merge T1 and T4
        df_comparison = df_t1.merge(df_t4, on='UID', suffixes=('_t1', '_t4'))

        # Categorize per measure
        # Accuracy
        acc_decline = np.sum(df_comparison['theta_acc_t4'] < df_comparison['theta_acc_t1'])
        acc_stable = np.sum(np.abs(df_comparison['theta_acc_t4'] - df_comparison['theta_acc_t1']) < 0.1 * SD_acc_baseline)
        acc_improvement = np.sum(df_comparison['theta_acc_t4'] > df_comparison['theta_acc_t1'])

        # Confidence
        conf_decline = np.sum(df_comparison['theta_conf_t4'] < df_comparison['theta_conf_t1'])
        conf_stable = np.sum(np.abs(df_comparison['theta_conf_t4'] - df_comparison['theta_conf_t1']) < 0.1 * SD_conf_baseline)
        conf_improvement = np.sum(df_comparison['theta_conf_t4'] > df_comparison['theta_conf_t1'])

        log(f"Accuracy: {acc_decline} decline, {acc_stable} stable, {acc_improvement} improvement")
        log(f"Confidence: {conf_decline} decline, {conf_stable} stable, {conf_improvement} improvement")

        # Flag if >10% show improvement
        acc_improvement_pct = (acc_improvement / len(df_comparison)) * 100
        conf_improvement_pct = (conf_improvement / len(df_comparison)) * 100

        if acc_improvement_pct > 10:
            log(f"Accuracy: {acc_improvement_pct:.1f}% show improvement (acknowledges practice effects)")
        if conf_improvement_pct > 10:
            log(f"Confidence: {conf_improvement_pct:.1f}% show improvement (acknowledges practice effects)")

        # Create practice effects summary text
        practice_summary = f"""Practice Effects Summary
========================

Accuracy:
  Decline: {acc_decline} ({acc_decline/len(df_comparison)*100:.1f}%)
  Stable: {acc_stable} ({acc_stable/len(df_comparison)*100:.1f}%)
  Improvement: {acc_improvement} ({acc_improvement_pct:.1f}%)

Confidence:
  Decline: {conf_decline} ({conf_decline/len(df_comparison)*100:.1f}%)
  Stable: {conf_stable} ({conf_stable/len(df_comparison)*100:.1f}%)
  Improvement: {conf_improvement} ({conf_improvement_pct:.1f}%)

Notes:
- Decline = T4 < T1 (expected forgetting)
- Stable = |T4 - T1| < 0.1 SD (minimal change)
- Improvement = T4 > T1 (rare, may indicate practice effects)
"""
        # Save Analysis Outputs
        # These outputs will be used by: Steps 3-8 (all downstream analyses)

        log("Saving analysis outputs...")

        # Save individual decline rates
        output_path_rates = RQ_DIR / "data" / "step02_individual_decline_rates.csv"
        df_rates.to_csv(output_path_rates, index=False, encoding='utf-8')
        log(f"{output_path_rates.name} ({len(df_rates)} rows, {len(df_rates.columns)} cols)")

        # Save practice effects summary
        output_path_practice = RQ_DIR / "data" / "step02_practice_effects_summary.txt"
        with open(output_path_practice, 'w', encoding='utf-8') as f:
            f.write(practice_summary)
        log(f"{output_path_practice.name}")
        # Validation
        # Validates: Output structure, value ranges, data quality

        log("Running inline validation...")

        # Check rows
        if len(df_rates) != 100:
            log(f"Expected 100 rows, got {len(df_rates)}")

        # Check for NaN in critical columns
        critical_cols = ['UID', 'acc_rate', 'conf_rate', 'difference', 'delta_hours']
        for col in critical_cols:
            nan_count = df_rates[col].isna().sum()
            if nan_count > 0:
                log(f"{nan_count} NaN values in {col}")

        # Check for duplicate UIDs
        if df_rates['UID'].duplicated().any():
            log(f"Duplicate UIDs found in decline rates")

        # Check value ranges
        acc_rate_range = (df_rates['acc_rate'].min(), df_rates['acc_rate'].max())
        conf_rate_range = (df_rates['conf_rate'].min(), df_rates['conf_rate'].max())
        diff_range = (df_rates['difference'].min(), df_rates['difference'].max())
        delta_hours_range = (df_rates['delta_hours'].min(), df_rates['delta_hours'].max())

        log(f"acc_rate range: [{acc_rate_range[0]:.6f}, {acc_rate_range[1]:.6f}]")
        log(f"conf_rate range: [{conf_rate_range[0]:.6f}, {conf_rate_range[1]:.6f}]")
        log(f"difference range: [{diff_range[0]:.6f}, {diff_range[1]:.6f}]")
        log(f"delta_hours range: [{delta_hours_range[0]:.1f}, {delta_hours_range[1]:.1f}]")

        # Check no_decline_flag count
        no_decline_count = df_rates['no_decline_flag'].sum()
        log(f"no_decline_flag count: {no_decline_count}")
        if no_decline_count >= 5:
            log(f"Too many near-zero denominators ({no_decline_count})")

        # Report improvement percentages
        log(f"Improvement: accuracy {acc_improvement_pct:.1f}%, confidence {conf_improvement_pct:.1f}%")
        if acc_improvement_pct > 10 or conf_improvement_pct > 10:
            log(f"Practice effects: >10% improvement (flag but proceed)")

        # Summary statistics
        log(f"Time elapsed: mean={df_rates['delta_hours'].mean():.1f} hours, SD={df_rates['delta_hours'].std():.1f}")
        log(f"Mean acc_rate: {df_rates['acc_rate'].mean():.6f}")
        log(f"Mean conf_rate: {df_rates['conf_rate'].mean():.6f}")
        log(f"Mean difference: {df_rates['difference'].mean():.6f}")

        log("Step 2 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
