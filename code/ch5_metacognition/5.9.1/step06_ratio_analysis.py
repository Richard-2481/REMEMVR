#!/usr/bin/env python3
"""Decline Rate Ratio Analysis (SECONDARY TEST): Compute ratio = conf_rate/acc_rate and test if mean > 1.0 using one-sample"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.1
LOG_FILE = RQ_DIR / "logs" / "step06_ratio_analysis.log"

# Bootstrap parameters
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 42

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Decline rate ratio analysis (SECONDARY TEST)")
        # Load Input Data

        log("Loading input data...")

        # Load individual decline rates
        df_rates = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv", encoding='utf-8')
        log(f"step02_individual_decline_rates.csv ({len(df_rates)} rows, {len(df_rates.columns)} cols)")
        # Handle Edge Cases

        log("Handling near-zero denominators...")

        # Count no_decline_flag (near-zero denominators)
        no_decline_count = df_rates['no_decline_flag'].sum()
        log(f"no_decline_flag count: {no_decline_count}")

        # Filter to valid ratios (exclude no_decline_flag)
        df_valid = df_rates[~df_rates['no_decline_flag']].copy()
        log(f"Valid ratios: {len(df_valid)} participants (excluded {no_decline_count})")
        # Compute Ratio Descriptives

        log("Computing ratio statistics...")

        # Extract valid ratios
        valid_ratios = df_valid['ratio'].values

        # Compute descriptives
        mean_ratio = np.mean(valid_ratios)
        median_ratio = np.median(valid_ratios)
        sd_ratio = np.std(valid_ratios, ddof=1)

        # Compute IQR
        q1 = np.percentile(valid_ratios, 25)
        q3 = np.percentile(valid_ratios, 75)
        IQR_ratio = q3 - q1

        log(f"Mean ratio: {mean_ratio:.4f}")
        log(f"Median ratio: {median_ratio:.4f}")
        log(f"SD ratio: {sd_ratio:.4f}")
        log(f"IQR ratio: {IQR_ratio:.4f}")
        # Quantify Individual Heterogeneity

        log("Quantifying individual patterns...")

        # Categorize patterns (using valid ratios only)
        hedging_count = np.sum(df_valid['ratio'] > 1.1)  # Conf declines >10% faster
        parallel_count = np.sum((df_valid['ratio'] >= 0.9) & (df_valid['ratio'] <= 1.1))  # Similar rates
        overconfidence_count = np.sum(df_valid['ratio'] < 0.9)  # Conf declines slower

        # Compute percentages (out of valid N)
        N_valid = len(df_valid)
        hedging_pct = (hedging_count / N_valid) * 100
        parallel_pct = (parallel_count / N_valid) * 100
        overconfidence_pct = (overconfidence_count / N_valid) * 100

        log(f"Hedging: {hedging_count} ({hedging_pct:.1f}%)")
        log(f"Parallel: {parallel_count} ({parallel_pct:.1f}%)")
        log(f"Overconfidence: {overconfidence_count} ({overconfidence_pct:.1f}%)")
        log(f"No decline: {no_decline_count}")

        # Flag if >25% show overconfidence
        if overconfidence_pct > 25:
            log(f"Substantial overconfidence heterogeneity ({overconfidence_pct:.1f}%) - discussion needed")
        # One-Sample t-test

        log("Running one-sample t-test (ratio vs 1.0)...")

        # One-sample t-test (two-tailed, then convert)
        t_result = stats.ttest_1samp(valid_ratios, popmean=1.0)
        t_statistic = t_result.statistic
        p_value_twotailed = t_result.pvalue

        # Convert to one-tailed (ratio > 1.0)
        if t_statistic > 0:
            p_value = p_value_twotailed / 2
        else:
            p_value = 1 - (p_value_twotailed / 2)

        df = len(valid_ratios) - 1

        log(f"t({df}) = {t_statistic:.4f}, p = {p_value:.6f}")
        log("SECONDARY TEST - provides interpretable metric but NOT for hypothesis decision")
        # Bootstrap Ratio CI

        log(f"Computing 95% CI ({BOOTSTRAP_ITERATIONS} iterations, seed={RANDOM_SEED})...")

        # Initialize storage
        boot_ratios = []

        # Set seed
        np.random.seed(RANDOM_SEED)

        # Bootstrap iterations
        for i in range(BOOTSTRAP_ITERATIONS):
            np.random.seed(RANDOM_SEED + i)
            sampled_indices = np.random.choice(len(valid_ratios), size=len(valid_ratios), replace=True)
            sampled_ratios = valid_ratios[sampled_indices]
            boot_ratios.append(np.mean(sampled_ratios))

        # Compute percentile CI
        ci_lower = np.percentile(boot_ratios, 2.5)
        ci_upper = np.percentile(boot_ratios, 97.5)

        log(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        # Save Analysis Outputs
        # These outputs will be used by: results analysis (effect quantification)

        log("Saving analysis outputs...")

        # Save decline ratio results
        ratio_results = {
            'mean_ratio': [mean_ratio],
            'median_ratio': [median_ratio],
            'sd_ratio': [sd_ratio],
            'IQR_ratio': [IQR_ratio],
            'ci_lower': [ci_lower],
            'ci_upper': [ci_upper],
            't_statistic': [t_statistic],
            'df': [df],
            'p_value': [p_value]
        }
        df_ratio = pd.DataFrame(ratio_results)

        output_path_ratio = RQ_DIR / "data" / "step06_decline_ratio.csv"
        df_ratio.to_csv(output_path_ratio, index=False, encoding='utf-8')
        log(f"{output_path_ratio.name} ({len(df_ratio)} rows, {len(df_ratio.columns)} cols)")

        # Save individual heterogeneity
        het_results = {
            'hedging_count': [hedging_count],
            'hedging_pct': [hedging_pct],
            'parallel_count': [parallel_count],
            'parallel_pct': [parallel_pct],
            'overconfidence_count': [overconfidence_count],
            'overconfidence_pct': [overconfidence_pct],
            'no_decline_count': [no_decline_count]
        }
        df_het = pd.DataFrame(het_results)

        output_path_het = RQ_DIR / "data" / "step06_individual_heterogeneity.csv"
        df_het.to_csv(output_path_het, index=False, encoding='utf-8')
        log(f"{output_path_het.name} ({len(df_het)} rows, {len(df_het.columns)} cols)")
        # Validation
        # Validates: Output structure, value ranges, heterogeneity sums

        log("Running inline validation...")

        # Check ratio file
        if len(df_ratio) != 1:
            log(f"Expected 1 row in ratio file, got {len(df_ratio)}")
        if df_ratio[['mean_ratio', 'median_ratio', 't_statistic']].isna().any().any():
            log(f"NaN values in critical columns")

        # Check heterogeneity file
        if len(df_het) != 1:
            log(f"Expected 1 row in heterogeneity file, got {len(df_het)}")
        if df_het.isna().any().any():
            log(f"NaN values in heterogeneity file")

        # Check heterogeneity counts sum to 100
        total_count = hedging_count + parallel_count + overconfidence_count + no_decline_count
        if total_count != 100:
            log(f"Heterogeneity counts don't sum to 100: {total_count}")

        # Check CI brackets mean
        if not (ci_lower < mean_ratio < ci_upper):
            log(f"CI doesn't bracket mean: [{ci_lower}, {ci_upper}] vs {mean_ratio}")

        # Check no_decline_count
        if no_decline_count >= 5:
            log(f"More than expected near-zero denominators ({no_decline_count})")

        # Value ranges
        if not (0.5 <= mean_ratio <= 2.0):
            log(f"mean_ratio outside expected range: {mean_ratio}")
        if not (0.5 <= median_ratio <= 2.0):
            log(f"median_ratio outside expected range: {median_ratio}")

        log(f"Ratio analysis: N={N_valid} (excluded {no_decline_count} near-zero)")
        log(f"Mean ratio = {mean_ratio:.4f}")
        log(f"Individual heterogeneity: {hedging_pct:.1f}% hedging, {parallel_pct:.1f}% parallel, {overconfidence_pct:.1f}% overconfidence")
        log(f"One-sample t-test: t({df}) = {t_statistic:.4f}, p = {p_value:.6f}")
        log(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
