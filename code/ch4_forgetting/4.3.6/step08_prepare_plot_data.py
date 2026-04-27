#!/usr/bin/env python3
# =============================================================================
# SCRIPT METADATA
# =============================================================================
"""
Step ID: step08
Step Name: Prepare Comparison Plot Data
RQ: 5.3.6 - Purified CTT Effects (Paradigms)
Generated: 2025-12-04

PURPOSE:
Create plot-ready CSV files for correlation comparison and AIC comparison
visualizations. Reshapes correlation data to long format with Fisher-z CIs,
extracts AIC values for grouped bar charts.

EXPECTED INPUTS:
  - data/step05_correlation_analysis.csv
    Columns: ["paradigm", "r_full", "r_purified", "delta_r", "steiger_z", "p_uncorrected", "p_bonferroni", "normality_test_p", "linearity_flag"]
    Expected rows: 3 (one per paradigm)

  - data/step07_lmm_model_comparison.csv
    Columns: ["paradigm", "AIC_IRT", "AIC_full", "AIC_purified", "delta_AIC_full_purified", ...]
    Expected rows: 3 (one per paradigm)

EXPECTED OUTPUTS:
  - data/step08_correlation_comparison_data.csv
    Columns: ["paradigm", "ctt_type", "r", "CI_lower", "CI_upper"]
    Expected rows: 6 (3 paradigms x 2 CTT types)

  - data/step08_aic_comparison_data.csv
    Columns: ["paradigm", "AIC_IRT", "AIC_full", "AIC_purified", "delta_AIC_full_purified"]
    Expected rows: 3 (3 paradigms)

VALIDATION CRITERIA:
  - correlation_comparison_data: 6 rows (3 paradigms x 2 CTT types)
  - aic_comparison_data: 3 rows (3 paradigms)
  - r in [-1, 1]
  - CI_lower <= r <= CI_upper
  - AIC values positive
  - ctt_type contains only {"Full", "Purified"}
  - No NaN values

IMPLEMENTATION NOTES:
- Fisher Z-transformation for correlation CIs: z = arctanh(r), SE = 1/sqrt(n-3)
- CI = tanh(z ± 1.96*SE) for 95% confidence interval
- n = 400 observations (100 participants x 4 test sessions)
- Long format for grouped bar charts (easier for rq_plots agent)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.6
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_plot_data.log"

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Fisher Z-Transformation for Correlation CI
# =============================================================================

def correlation_ci(r, n, alpha=0.05):
    """
    Compute confidence interval for correlation using Fisher Z-transformation.

    Parameters:
    -----------
    r : float
        Correlation coefficient
    n : int
        Sample size (number of observations)
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns:
    --------
    ci_lower, ci_upper : float
        Lower and upper bounds of confidence interval

    Method:
    -------
    1. Transform r to z: z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
    2. Compute SE: SE_z = 1/sqrt(n-3)
    3. Compute z CI: z_CI = z ± z_crit * SE_z
    4. Back-transform: r_CI = tanh(z_CI)
    """
    z = np.arctanh(r)  # Fisher z-transform
    se = 1 / np.sqrt(n - 3)  # Standard error of z
    z_crit = 1.96  # Critical value for 95% CI (two-tailed)

    # Compute z-space CI
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    # Back-transform to r-space
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)

    return ci_lower, ci_upper

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 08: Prepare Comparison Plot Data")

        # =========================================================================
        # STEP 1: Load Input Data
        # =========================================================================
        log("[LOAD] Loading correlation analysis results...")
        correlation_df = pd.read_csv(RQ_DIR / "data" / "step05_correlation_analysis.csv")
        log(f"[LOADED] step05_correlation_analysis.csv ({len(correlation_df)} rows)")

        log("[LOAD] Loading LMM model comparison results...")
        lmm_df = pd.read_csv(RQ_DIR / "data" / "step07_lmm_model_comparison.csv")
        log(f"[LOADED] step07_lmm_model_comparison.csv ({len(lmm_df)} rows)")

        # =========================================================================
        # STEP 2: Create Correlation Comparison Data (Long Format)
        # =========================================================================
        # Goal: 6 rows (3 paradigms x 2 CTT types) for grouped bar chart
        # Columns: paradigm, ctt_type, r, CI_lower, CI_upper

        log("[ANALYSIS] Creating correlation comparison data (long format)...")

        n_observations = 400  # 100 participants x 4 test sessions

        # Initialize list to collect rows
        correlation_rows = []

        for _, row in correlation_df.iterrows():
            paradigm = row['paradigm']
            r_full = row['r_full']
            r_purified = row['r_purified']

            # Compute CIs using Fisher Z-transformation
            ci_lower_full, ci_upper_full = correlation_ci(r_full, n_observations)
            ci_lower_purified, ci_upper_purified = correlation_ci(r_purified, n_observations)

            # Add Full CTT row
            correlation_rows.append({
                'paradigm': paradigm,
                'ctt_type': 'Full',
                'r': r_full,
                'CI_lower': ci_lower_full,
                'CI_upper': ci_upper_full
            })

            # Add Purified CTT row
            correlation_rows.append({
                'paradigm': paradigm,
                'ctt_type': 'Purified',
                'r': r_purified,
                'CI_lower': ci_lower_purified,
                'CI_upper': ci_upper_purified
            })

        correlation_plot_data = pd.DataFrame(correlation_rows)
        log(f"[CREATED] Correlation comparison data: {len(correlation_plot_data)} rows x {len(correlation_plot_data.columns)} columns")

        # =========================================================================
        # STEP 3: Create AIC Comparison Data
        # =========================================================================
        # Goal: 3 rows (one per paradigm) with AIC values for all 3 measurement types
        # Columns: paradigm, AIC_IRT, AIC_full, AIC_purified, delta_AIC_full_purified

        log("[ANALYSIS] Creating AIC comparison data...")

        aic_plot_data = lmm_df[['paradigm', 'AIC_IRT', 'AIC_full', 'AIC_purified', 'delta_AIC_full_purified']].copy()
        log(f"[CREATED] AIC comparison data: {len(aic_plot_data)} rows x {len(aic_plot_data.columns)} columns")

        # =========================================================================
        # STEP 4: Save Output Files
        # =========================================================================
        log("[SAVE] Saving correlation comparison data...")
        correlation_plot_data.to_csv(
            RQ_DIR / "data" / "step08_correlation_comparison_data.csv",
            index=False,
            encoding='utf-8'
        )
        log(f"[SAVED] step08_correlation_comparison_data.csv ({len(correlation_plot_data)} rows)")

        log("[SAVE] Saving AIC comparison data...")
        aic_plot_data.to_csv(
            RQ_DIR / "data" / "step08_aic_comparison_data.csv",
            index=False,
            encoding='utf-8'
        )
        log(f"[SAVED] step08_aic_comparison_data.csv ({len(aic_plot_data)} rows)")

        # =========================================================================
        # STEP 5: Validation Checks
        # =========================================================================
        log("[VALIDATION] Validating correlation comparison data...")

        # Check row count (6 rows: 3 paradigms x 2 CTT types)
        assert len(correlation_plot_data) == 6, f"Expected 6 rows, got {len(correlation_plot_data)}"
        log("[PASS] Row count: 6 rows (3 paradigms x 2 CTT types)")

        # Check column names
        expected_corr_cols = ["paradigm", "ctt_type", "r", "CI_lower", "CI_upper"]
        assert list(correlation_plot_data.columns) == expected_corr_cols, \
            f"Column mismatch: expected {expected_corr_cols}, got {list(correlation_plot_data.columns)}"
        log(f"[PASS] Column names: {expected_corr_cols}")

        # Check r in [-1, 1]
        assert correlation_plot_data['r'].between(-1, 1).all(), "r values outside [-1, 1] range"
        log(f"[PASS] r values in [-1, 1]: min={correlation_plot_data['r'].min():.4f}, max={correlation_plot_data['r'].max():.4f}")

        # Check CI_lower <= r <= CI_upper
        ci_valid = (correlation_plot_data['CI_lower'] <= correlation_plot_data['r']) & \
                   (correlation_plot_data['r'] <= correlation_plot_data['CI_upper'])
        assert ci_valid.all(), "Some r values outside their confidence intervals"
        log("[PASS] All r values within their confidence intervals")

        # Check ctt_type values
        valid_ctt_types = {"Full", "Purified"}
        assert set(correlation_plot_data['ctt_type'].unique()) == valid_ctt_types, \
            f"ctt_type contains invalid values: {set(correlation_plot_data['ctt_type'].unique())}"
        log(f"[PASS] ctt_type contains only {valid_ctt_types}")

        # Check paradigms
        expected_paradigms = {"IFR", "ICR", "IRE"}
        assert set(correlation_plot_data['paradigm'].unique()) == expected_paradigms, \
            f"Paradigm mismatch: expected {expected_paradigms}, got {set(correlation_plot_data['paradigm'].unique())}"
        log(f"[PASS] All paradigms present: {expected_paradigms}")

        # Check for NaN values
        assert not correlation_plot_data.isna().any().any(), "NaN values found in correlation comparison data"
        log("[PASS] No NaN values in correlation comparison data")

        log("[VALIDATION] Validating AIC comparison data...")

        # Check row count (3 rows: one per paradigm)
        assert len(aic_plot_data) == 3, f"Expected 3 rows, got {len(aic_plot_data)}"
        log("[PASS] Row count: 3 rows (one per paradigm)")

        # Check column names
        expected_aic_cols = ["paradigm", "AIC_IRT", "AIC_full", "AIC_purified", "delta_AIC_full_purified"]
        assert list(aic_plot_data.columns) == expected_aic_cols, \
            f"Column mismatch: expected {expected_aic_cols}, got {list(aic_plot_data.columns)}"
        log(f"[PASS] Column names: {expected_aic_cols}")

        # Check AIC values positive
        aic_columns = ['AIC_IRT', 'AIC_full', 'AIC_purified']
        for col in aic_columns:
            assert (aic_plot_data[col] > 0).all(), f"{col} contains non-positive values"
        log(f"[PASS] All AIC values positive (min={aic_plot_data[aic_columns].min().min():.2f})")

        # Check paradigms
        assert set(aic_plot_data['paradigm'].unique()) == expected_paradigms, \
            f"Paradigm mismatch: expected {expected_paradigms}, got {set(aic_plot_data['paradigm'].unique())}"
        log(f"[PASS] All paradigms present: {expected_paradigms}")

        # Check for NaN values
        assert not aic_plot_data.isna().any().any(), "NaN values found in AIC comparison data"
        log("[PASS] No NaN values in AIC comparison data")

        # =========================================================================
        # STEP 6: Summary Statistics
        # =========================================================================
        log("[SUMMARY] Correlation comparison data:")
        for paradigm in ['IFR', 'ICR', 'IRE']:
            subset = correlation_plot_data[correlation_plot_data['paradigm'] == paradigm]
            r_full = subset[subset['ctt_type'] == 'Full']['r'].values[0]
            r_purified = subset[subset['ctt_type'] == 'Purified']['r'].values[0]
            delta_r = r_purified - r_full
            log(f"  {paradigm}: r_full={r_full:.4f}, r_purified={r_purified:.4f}, delta_r={delta_r:.4f}")

        log("[SUMMARY] AIC comparison data:")
        for _, row in aic_plot_data.iterrows():
            paradigm = row['paradigm']
            aic_irt = row['AIC_IRT']
            aic_full = row['AIC_full']
            aic_purified = row['AIC_purified']
            delta_aic = row['delta_AIC_full_purified']
            log(f"  {paradigm}: AIC_IRT={aic_irt:.2f}, AIC_full={aic_full:.2f}, AIC_purified={aic_purified:.2f}, delta={delta_aic:.2f}")

        log("[SUCCESS] Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
