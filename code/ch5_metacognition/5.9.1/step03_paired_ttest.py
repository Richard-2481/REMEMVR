#!/usr/bin/env python3
"""Paired t-test (PRIMARY HYPOTHESIS TEST): Test if mean(conf_rate - acc_rate) > 0 using paired t-test. This is the"""

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
LOG_FILE = RQ_DIR / "logs" / "step03_paired_ttest.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Paired t-test on decline rate difference (PRIMARY TEST)")
        # Load Input Data

        log("Loading input data...")

        # Load individual decline rates
        df_rates = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv", encoding='utf-8')
        log(f"step02_individual_decline_rates.csv ({len(df_rates)} rows, {len(df_rates.columns)} cols)")
        # Run Paired t-test

        log("Running paired t-test...")

        # Extract arrays
        conf_rate = df_rates['conf_rate'].values
        acc_rate = df_rates['acc_rate'].values
        difference = df_rates['difference'].values

        # Paired t-test (two-tailed)
        # Note: scipy.stats.ttest_rel doesn't support 'alternative' parameter directly
        # We'll compute two-tailed and convert to one-tailed
        t_result = stats.ttest_rel(conf_rate, acc_rate)
        t_statistic = t_result.statistic
        p_value_twotailed = t_result.pvalue

        # Convert to one-tailed (conf > acc)
        # If t > 0, one-tailed p = p_twotailed / 2
        # If t < 0, one-tailed p = 1 - (p_twotailed / 2)
        if t_statistic > 0:
            p_value_onetailed = p_value_twotailed / 2
        else:
            p_value_onetailed = 1 - (p_value_twotailed / 2)

        df = len(df_rates) - 1  # Degrees of freedom

        log(f"t({df}) = {t_statistic:.4f}, p (two-tailed) = {p_value_twotailed:.6f}")
        log(f"p (one-tailed, conf > acc) = {p_value_onetailed:.6f}")
        # Apply Decision D068 (Dual p-value reporting)
        # Note: Single primary test, so p_bonferroni = p_uncorrected
        # But report both for format consistency

        log("Applying Decision D068 (dual p-value reporting)...")

        p_uncorrected = p_value_onetailed
        p_bonferroni = p_uncorrected * 1  # Family size = 1 (single primary test)

        log(f"p_uncorrected = {p_uncorrected:.6f}")
        log(f"p_bonferroni = {p_bonferroni:.6f} (no correction needed, single test)")
        # Compute Descriptives

        log("Computing mean difference and 95% CI...")

        mean_difference = np.mean(difference)
        sd_difference = np.std(difference, ddof=1)
        se_difference = sd_difference / np.sqrt(len(difference))

        # 95% CI (parametric)
        ci_lower = mean_difference - 1.96 * se_difference
        ci_upper = mean_difference + 1.96 * se_difference

        log(f"Mean difference: {mean_difference:.6f}")
        log(f"SE difference: {se_difference:.6f}")
        log(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        # Save Analysis Outputs
        # These outputs will be used by: results analysis (hypothesis decision)

        log("Saving analysis outputs...")

        # Create results DataFrame
        results = {
            't_statistic': [t_statistic],
            'df': [df],
            'p_uncorrected': [p_uncorrected],
            'p_bonferroni': [p_bonferroni],
            'mean_difference': [mean_difference],
            'se_difference': [se_difference],
            'ci_lower': [ci_lower],
            'ci_upper': [ci_upper]
        }
        df_results = pd.DataFrame(results)

        # Save paired t-test results
        output_path = RQ_DIR / "data" / "step03_paired_ttest_results.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_results)} rows, {len(df_results.columns)} cols)")
        # Validation
        # Validates: Output structure, value ranges, CI bracket

        log("Running inline validation...")

        # Check rows
        if len(df_results) != 1:
            log(f"Expected 1 row, got {len(df_results)}")

        # Check columns
        if len(df_results.columns) != 8:
            log(f"Expected 8 columns, got {len(df_results.columns)}")

        # Check for NaN
        if df_results.isna().any().any():
            log(f"NaN values found in results")

        # Check df
        if df_results['df'].values[0] != 99:
            log(f"Expected df=99, got {df_results['df'].values[0]}")

        # Check CI brackets mean
        if not (ci_lower < mean_difference < ci_upper):
            log(f"CI doesn't bracket mean: [{ci_lower}, {ci_upper}] vs {mean_difference}")

        # Check p_bonferroni == p_uncorrected (single test)
        if not np.isclose(p_bonferroni, p_uncorrected):
            log(f"p_bonferroni != p_uncorrected for single test")

        # Value ranges
        if not (-10 <= t_statistic <= 10):
            log(f"t-statistic outside expected range: {t_statistic}")
        if not (0 <= p_uncorrected <= 1):
            log(f"p_uncorrected outside [0, 1]: {p_uncorrected}")
        if not (0 <= p_bonferroni <= 1):
            log(f"p_bonferroni outside [0, 1]: {p_bonferroni}")

        log("Paired t-test complete: t(99) = {:.4f}, p = {:.6f}".format(t_statistic, p_uncorrected))
        log(f"Mean difference: {mean_difference:.6f}")
        log("Dual reporting per Decision D068")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
