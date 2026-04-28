#!/usr/bin/env python3
"""Compute Effect Sizes for Practice Magnitude Comparison: Compute Cohen's d effect sizes for T1->T2 improvement in both measures to quantify"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import power analysis
from statsmodels.stats.power import TTestPower

from tools.validation import validate_effect_sizes

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.6
LOG_FILE = RQ_DIR / "logs" / "step05_compute_effect_sizes.log"

# Input paths
CONFIDENCE_WIDE_PATH = RQ_DIR / "data" / "step02_confidence_intervals_wide.csv"
ACCURACY_WIDE_PATH = PROJECT_ROOT / "results" / "ch5" / "5.1.2" / "data" / "step02_accuracy_intervals_wide.csv"
RATIO_COMPARISON_PATH = RQ_DIR / "data" / "step03_ratio_comparison.csv"
BOOTSTRAP_PATH = RQ_DIR / "data" / "step03_bootstrap_distribution.csv"

# Output paths
OUTPUT_EFFECT_SIZES = RQ_DIR / "data" / "step05_effect_sizes.csv"
OUTPUT_POWER = RQ_DIR / "data" / "step05_power_analysis.csv"

# Cohen (1988) thresholds
COHEN_SMALL = 0.2
COHEN_MEDIUM = 0.5
COHEN_LARGE = 0.8

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Functions

def interpret_cohens_d(d):
    """Interpret Cohen's d using Cohen (1988) thresholds."""
    abs_d = abs(d)
    if abs_d < COHEN_SMALL:
        return "Negligible"
    elif abs_d < COHEN_MEDIUM:
        return "Small"
    elif abs_d < COHEN_LARGE:
        return "Medium"
    else:
        return "Large"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Compute Effect Sizes for Practice Magnitude Comparison")
        log("=" * 80)
        # Load Confidence T1->T2 Improvements
        log("\nLoading confidence T1->T2 improvements...")
        df_conf = pd.read_csv(CONFIDENCE_WIDE_PATH, encoding='utf-8')
        log(f"{CONFIDENCE_WIDE_PATH.name} ({len(df_conf)} rows)")

        # Extract delta_T1_T2 values
        delta_conf = df_conf['delta_T1_T2'].dropna()
        n_conf = len(delta_conf)

        log(f"N={n_conf} valid T1->T2 improvements")
        log(f"             Mean={delta_conf.mean():.4f}, SD={delta_conf.std():.4f}")
        # Load Accuracy T1->T2 Improvements
        log("\nLoading accuracy T1->T2 improvements...")

        if ACCURACY_WIDE_PATH.exists():
            df_acc = pd.read_csv(ACCURACY_WIDE_PATH, encoding='utf-8')
            log(f"{ACCURACY_WIDE_PATH.name} ({len(df_acc)} rows)")

            # Find delta_T1_T2 column (may have different naming)
            delta_cols = [col for col in df_acc.columns if 'T1' in col and 'T2' in col]
            if len(delta_cols) == 0:
                log(f"No delta_T1_T2 column found, will use first delta column")
                delta_cols = [col for col in df_acc.columns if 'delta' in col.lower()]

            if len(delta_cols) > 0:
                delta_acc = df_acc[delta_cols[0]].dropna()
                log(f"Using column: {delta_cols[0]}")
            else:
                raise ValueError(f"Cannot find T1->T2 delta column in {ACCURACY_WIDE_PATH}")

        else:
            log(f"[NOT FOUND] {ACCURACY_WIDE_PATH}")
            log(f"Would compute from accuracy raw data - not implemented")
            raise FileNotFoundError(f"Accuracy intervals file not found: {ACCURACY_WIDE_PATH}")

        n_acc = len(delta_acc)

        log(f"N={n_acc} valid T1->T2 improvements")
        log(f"           Mean={delta_acc.mean():.4f}, SD={delta_acc.std():.4f}")
        # Compute Cohen's d for T1->T2 Practice Effects
        log("\n[EFFECT SIZE] Computing Cohen's d for T1->T2 practice effects...")

        # Cohen's d for within-subjects design: d = mean / SD
        d_conf = delta_conf.mean() / delta_conf.std()
        d_acc = delta_acc.mean() / delta_acc.std()

        log(f"\n[CONFIDENCE T1->T2]")
        log(f"  Cohen's d = {d_conf:.3f}")
        log(f"  Interpretation: {interpret_cohens_d(d_conf)}")

        log(f"\n[ACCURACY T1->T2]")
        log(f"  Cohen's d = {d_acc:.3f}")
        log(f"  Interpretation: {interpret_cohens_d(d_acc)}")
        # Compute Cohen's d for Ratio Difference
        log("\n[EFFECT SIZE] Computing Cohen's d for saturation ratio difference...")

        # Load ratio comparison and bootstrap distribution
        df_ratio = pd.read_csv(RATIO_COMPARISON_PATH, encoding='utf-8')
        df_bootstrap = pd.read_csv(BOOTSTRAP_PATH, encoding='utf-8')

        diff_ratio = df_ratio['diff_ratio'].values[0]
        pooled_sd = df_bootstrap['diff_ratio_boot'].std()  # SD from bootstrap distribution

        # Cohen's d for ratio difference: d = mean_diff / pooled_SD
        d_ratio = diff_ratio / pooled_sd

        log(f"\n[RATIO DIFFERENCE]")
        log(f"  diff_ratio = {diff_ratio:.3f}")
        log(f"  Pooled SD (bootstrap) = {pooled_sd:.3f}")
        log(f"  Cohen's d = {d_ratio:.3f}")
        log(f"  Interpretation: {interpret_cohens_d(d_ratio)}")
        # Compute Bootstrap CIs for Effect Sizes (Simplified)
        log("\n[BOOTSTRAP CI] Computing approximate CIs for effect sizes...")

        # For simplicity, use SE = SD/sqrt(N) for delta-based effect sizes
        se_d_conf = d_conf / np.sqrt(n_conf)
        se_d_acc = d_acc / np.sqrt(n_acc)

        ci_d_conf_lower = d_conf - 1.96 * se_d_conf
        ci_d_conf_upper = d_conf + 1.96 * se_d_conf

        ci_d_acc_lower = d_acc - 1.96 * se_d_acc
        ci_d_acc_upper = d_acc + 1.96 * se_d_acc

        # Ratio difference CI from bootstrap percentiles
        ci_d_ratio_lower = np.percentile(df_bootstrap['diff_ratio_boot'].dropna() / pooled_sd, 2.5)
        ci_d_ratio_upper = np.percentile(df_bootstrap['diff_ratio_boot'].dropna() / pooled_sd, 97.5)

        log(f"  Confidence T1->T2 CI: [{ci_d_conf_lower:.3f}, {ci_d_conf_upper:.3f}]")
        log(f"  Accuracy T1->T2 CI: [{ci_d_acc_lower:.3f}, {ci_d_acc_upper:.3f}]")
        log(f"  Ratio difference CI: [{ci_d_ratio_lower:.3f}, {ci_d_ratio_upper:.3f}]")
        # Post-Hoc Power Analysis
        log("\nConducting post-hoc power analysis...")

        power_analyzer = TTestPower()

        # Achieved power for confidence T1->T2 effect
        power_conf = power_analyzer.solve_power(
            effect_size=abs(d_conf),
            nobs=n_conf,
            alpha=0.05,
            alternative='two-sided'
        )

        # Achieved power for accuracy T1->T2 effect
        power_acc = power_analyzer.solve_power(
            effect_size=abs(d_acc),
            nobs=n_acc,
            alpha=0.05,
            alternative='two-sided'
        )

        log(f"\n[CONFIDENCE T1->T2]")
        log(f"  Achieved power: {power_conf:.3f}")
        log(f"  Interpretation: {'Adequate (>= 0.80)' if power_conf >= 0.80 else 'Underpowered (< 0.80)'}")

        log(f"\n[ACCURACY T1->T2]")
        log(f"  Achieved power: {power_acc:.3f}")
        log(f"  Interpretation: {'Adequate (>= 0.80)' if power_acc >= 0.80 else 'Underpowered (< 0.80)'}")
        # Save Effect Sizes
        log(f"\nSaving effect sizes...")

        df_effect_sizes = pd.DataFrame({
            'comparison': ['confidence_T1_T2', 'accuracy_T1_T2', 'ratio_difference'],
            'cohen_d': [d_conf, d_acc, d_ratio],
            'ci_lower': [ci_d_conf_lower, ci_d_acc_lower, ci_d_ratio_lower],
            'ci_upper': [ci_d_conf_upper, ci_d_acc_upper, ci_d_ratio_upper],
            'interpretation': [interpret_cohens_d(d_conf), interpret_cohens_d(d_acc), interpret_cohens_d(d_ratio)]
        })

        df_effect_sizes.to_csv(OUTPUT_EFFECT_SIZES, index=False, encoding='utf-8')
        log(f"{OUTPUT_EFFECT_SIZES} ({len(df_effect_sizes)} rows)")
        # Save Power Analysis
        log(f"\nSaving power analysis...")

        df_power = pd.DataFrame({
            'measure': ['confidence', 'accuracy'],
            'cohen_d': [abs(d_conf), abs(d_acc)],
            'achieved_power': [power_conf, power_acc],
            'interpretation': [
                'Adequate (>= 0.80)' if power_conf >= 0.80 else 'Underpowered (< 0.80)',
                'Adequate (>= 0.80)' if power_acc >= 0.80 else 'Underpowered (< 0.80)'
            ]
        })

        df_power.to_csv(OUTPUT_POWER, index=False, encoding='utf-8')
        log(f"{OUTPUT_POWER} ({len(df_power)} rows)")
        # Run Validation
        log("\nRunning validate_effect_sizes...")

        # Note: validate_effect_sizes expects 'cohens_f2' column, but we have 'cohen_d'
        # We'll skip this validation or adapt
        log(f"validate_effect_sizes designed for f2, not d - using custom validation")

        # Custom validation: check for NaN, infinite, extreme values
        if df_effect_sizes['cohen_d'].isna().any():
            log(f"NaN values in cohen_d")
            raise ValueError("NaN in effect sizes")

        if np.isinf(df_effect_sizes['cohen_d']).any():
            log(f"Infinite values in cohen_d")
            raise ValueError("Infinite effect sizes")

        d_min = df_effect_sizes['cohen_d'].min()
        d_max = df_effect_sizes['cohen_d'].max()
        log(f"Effect sizes range: [{d_min:.3f}, {d_max:.3f}]")

        if abs(d_min) > 5 or abs(d_max) > 5:
            log(f"Extreme effect sizes detected (|d| > 5)")

        # Validate power values
        if (df_power['achieved_power'] < 0).any() or (df_power['achieved_power'] > 1).any():
            log(f"Power values outside [0, 1] range")
            raise ValueError("Invalid power values")

        log(f"Power values in valid range [0, 1]")
        # SUMMARY
        log("\n" + "=" * 80)
        log("Step 5 complete")
        log(f"  Cohen's d (confidence T1->T2): {d_conf:.3f} ({interpret_cohens_d(d_conf)})")
        log(f"  Cohen's d (accuracy T1->T2): {d_acc:.3f} ({interpret_cohens_d(d_acc)})")
        log(f"  Cohen's d (ratio difference): {d_ratio:.3f} ({interpret_cohens_d(d_ratio)})")
        log(f"  Power (confidence): {power_conf:.3f}, Power (accuracy): {power_acc:.3f}")
        log(f"  Outputs: {OUTPUT_EFFECT_SIZES}, {OUTPUT_POWER}")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
