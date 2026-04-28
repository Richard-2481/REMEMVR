#!/usr/bin/env python3
"""statistical_comparisons_between_domains: Test pairwise difference between What and Where domain ICC estimates and compute effect size."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.analysis_stats import compute_effect_sizes
from tools.validation import validate_contrasts_dual_pvalues

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_statistical_comparisons.log"
OUTPUT_FILE = RQ_DIR / "data" / "step04_pairwise_comparisons.csv"

# Input files
INPUT_BOOT_DIST = RQ_DIR / "data" / "step03_bootstrap_distributions.csv"
INPUT_ICC = RQ_DIR / "data" / "step02_icc_estimates.csv"

# Statistical parameters
ALPHA = 0.05
N_COMPARISONS = 1  # MODIFIED: Only What-Where (When excluded)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Statistical Comparisons Between Domains")
        log(f"Single comparison only (What-Where), When domain excluded")
        # Load Data

        log(f"Reading bootstrap distributions...")
        df_boot = pd.read_csv(INPUT_BOOT_DIST)
        log(f"{len(df_boot)} rows")

        log(f"Reading ICC estimates...")
        df_icc = pd.read_csv(INPUT_ICC)
        log(f"{len(df_icc)} rows")
        # Compute Bootstrap Difference Distribution (What - Where)
        # For each bootstrap iteration, compute ICC_what - ICC_where
        # This gives distribution of differences under resampling

        log(f"Calculating bootstrap difference distribution...")

        # Pivot bootstrap distributions to wide format
        df_boot_pivot = df_boot.pivot(
            index='bootstrap_iteration',
            columns='domain',
            values='icc_bootstrap'
        )

        # Compute differences for each bootstrap sample
        diff_boot = df_boot_pivot['What'] - df_boot_pivot['Where']

        log(f"Bootstrap differences calculated: {len(diff_boot)} samples")
        # Calculate Difference Statistics

        # Mean difference
        diff_mean = diff_boot.mean()

        # 95% CI for difference (percentile method)
        ci_lower = np.percentile(diff_boot, 2.5)
        ci_upper = np.percentile(diff_boot, 97.5)

        log(f"Mean: {diff_mean:.4f}")
        log(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        # Calculate P-Value (Two-Tailed Test)
        # H0: ICC_what = ICC_where (difference = 0)
        # P-value: Proportion of bootstrap samples with |diff| >= |observed_diff|

        # Observed difference
        icc_what = df_icc[df_icc['domain'] == 'What']['icc_value'].values[0]
        icc_where = df_icc[df_icc['domain'] == 'Where']['icc_value'].values[0]
        diff_observed = icc_what - icc_where

        log(f"ICC_what = {icc_what:.4f}, ICC_where = {icc_where:.4f}")
        log(f"Difference = {diff_observed:.4f}")

        # Two-tailed p-value
        p_uncorrected = np.mean(np.abs(diff_boot) >= np.abs(diff_observed))

        log(f"[P-VALUE] Uncorrected: {p_uncorrected:.4f}")
        # Multiple Comparison Correction (Decision D068)
        # CRITICAL: Single test, so Bonferroni and FDR = uncorrected
        # Bonferroni correction: p * n_tests (but n_tests = 1)
        # FDR correction: Not applicable for single test

        p_bonferroni = min(p_uncorrected * N_COMPARISONS, 1.0)
        p_fdr = p_uncorrected  # FDR = uncorrected for single test

        log(f"[P-VALUE] Bonferroni (n=1): {p_bonferroni:.4f}")
        log(f"[P-VALUE] FDR (n=1): {p_fdr:.4f}")
        log(f"All p-values identical (single test, no correction needed)")
        # Calculate Effect Size (Cohen's d)
        # Cohen's d for ICC comparison: mean_diff / pooled_SD

        # Get What and Where bootstrap samples
        what_boot = df_boot[df_boot['domain'] == 'What']['icc_bootstrap'].values
        where_boot = df_boot[df_boot['domain'] == 'Where']['icc_bootstrap'].values

        # Compute effect size using tools.analysis_stats
        effect_size_result = compute_effect_sizes(
            group1=what_boot,
            group2=where_boot,
            test_type='cohens_d'
        )

        cohens_d = effect_size_result['cohens_d']

        log(f"[EFFECT SIZE] Cohen's d: {cohens_d:.4f}")
        # Create Results DataFrame

        comparison_results = [{
            'comparison': 'What-Where',
            'diff_mean': diff_mean,
            'diff_ci_lower': ci_lower,
            'diff_ci_upper': ci_upper,
            'cohens_d': cohens_d,
            'p_uncorrected': p_uncorrected,
            'p_bonferroni': p_bonferroni,
            'p_fdr': p_fdr
        }]

        df_comparisons = pd.DataFrame(comparison_results)

        log(f"Pairwise comparisons table: {len(df_comparisons)} row")
        # Validate Decision D068 Compliance

        log(f"Checking Decision D068 compliance...")

        validation_result = validate_contrasts_dual_pvalues(
            contrasts_df=df_comparisons,
            required_comparisons=['What-Where']
        )

        if not validation_result.get('valid', False):
            log(f"D068 validation failed: {validation_result}")
            sys.exit(1)
        else:
            log(f"Decision D068 compliance verified")
            log(f"All required comparisons present")
            log(f"Dual p-values (uncorrected + bonferroni + fdr) present")

        # Validate mathematical constraints
        if df_comparisons['p_bonferroni'].values[0] < df_comparisons['p_uncorrected'].values[0]:
            log(f"p_bonferroni < p_uncorrected (impossible)")
            sys.exit(1)
        else:
            log(f"p_bonferroni >= p_uncorrected")

        # Validate p-values in [0, 1]
        for p_col in ['p_uncorrected', 'p_bonferroni', 'p_fdr']:
            p_val = df_comparisons[p_col].values[0]
            if p_val < 0 or p_val > 1:
                log(f"{p_col} = {p_val} outside [0, 1]")
                sys.exit(1)

        log(f"All p-values in [0, 1]")

        # Validate effect size range
        if abs(cohens_d) > 3:
            log(f"Large effect size |d| = {abs(cohens_d):.4f} > 3")
        else:
            log(f"Effect size in reasonable range")
        # Save Results

        log(f"Writing pairwise comparisons...")
        df_comparisons.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_comparisons)} row)")

        log(f"Step 04 complete")
        log(f"What-Where comparison: diff={diff_mean:.4f}, p={p_uncorrected:.4f}, d={cohens_d:.4f}")
        log(f"Proceed to step05 (outlier analysis)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
