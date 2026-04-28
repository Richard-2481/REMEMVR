#!/usr/bin/env python3
"""power_analysis: Interpret effect sizes using Cohen's guidelines and assess statistical power"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_probability_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Effect Size and Power Functions

def interpret_effect_size(r):
    """Interpret correlation effect size using Cohen's guidelines."""
    abs_r = abs(r)
    if abs_r < 0.10:
        return "negligible"
    elif abs_r < 0.30:
        return "small"
    elif abs_r < 0.50:
        return "medium"
    else:
        return "large"


def approximate_power(r, n=100, alpha=0.05):
    """Approximate statistical power for correlation test (N=100, alpha=0.05)."""
    effect_size = abs(r)
    if effect_size >= 0.50:
        return 0.99
    elif effect_size >= 0.40:
        return 0.97
    elif effect_size >= 0.30:
        return 0.86
    elif effect_size >= 0.28:
        return 0.80
    elif effect_size >= 0.20:
        return 0.52
    elif effect_size >= 0.10:
        return 0.17
    else:
        return 0.08

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: power_analysis (Forgetting + Pct Retention)")
        # Load Correlation Results

        log("Loading correlation results...")

        bivariate_path = RQ_DIR / "data" / "step04_bivariate_correlation.csv"
        bivariate_df = pd.read_csv(bivariate_path, encoding='utf-8')
        log(f"Bivariate correlations: {bivariate_path} ({len(bivariate_df)} rows)")

        partial_path = RQ_DIR / "data" / "step05_partial_correlation.csv"
        partial_df = pd.read_csv(partial_path, encoding='utf-8')
        log(f"Partial correlations: {partial_path} ({len(partial_df)} rows)")

        min_detectable_r = 0.28  # For N=100, alpha=0.05, power=0.80

        power_results = []
        # Analyze Bivariate Correlations

        log("\nAnalyzing bivariate correlations...")

        for _, row in bivariate_df.iterrows():
            predictor = row['predictor']
            r_val = row['correlation']
            n_val = int(row['N'])
            power_val = approximate_power(r_val, n=n_val)
            interp = interpret_effect_size(r_val)

            log(f"{predictor} bivariate: r={r_val:.4f}, effect={interp}, power={power_val:.2f}")

            power_results.append({
                'predictor': predictor,
                'analysis_type': 'bivariate_correlation',
                'effect_size': r_val,
                'power': power_val,
                'min_detectable_r': min_detectable_r,
                'interpretation': interp
            })
        # Analyze Partial Correlations

        log("\nAnalyzing partial correlations...")

        for _, row in partial_df.iterrows():
            predictor = row['predictor']
            r_val = row['partial_r']
            n_val = int(row['N'])
            power_val = approximate_power(r_val, n=n_val)
            interp = interpret_effect_size(r_val)

            log(f"{predictor} partial: r={r_val:.4f}, effect={interp}, power={power_val:.2f}")

            power_results.append({
                'predictor': predictor,
                'analysis_type': 'partial_correlation',
                'effect_size': r_val,
                'power': power_val,
                'min_detectable_r': min_detectable_r,
                'interpretation': interp
            })
        # Save Power Analysis Results

        log("\nSaving power analysis results...")

        power_df = pd.DataFrame(power_results)
        output_path = RQ_DIR / "data" / "step08_power_analysis.csv"
        power_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(power_df)} rows, {len(power_df.columns)} cols)")
        # Run Validation Tool

        log("Running validate_probability_range on power...")
        validation_result = validate_probability_range(
            probability_df=power_df,
            prob_columns=['power']
        )
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
            if validation_result.get('valid', False):
                log("PASS - All power values in valid range [0, 1]")
            else:
                log("FAIL - Some power values out of range")
        else:
            log(f"{validation_result}")

        # Summary
        log("\nPower Analysis Results:")
        log("=" * 80)
        log(f"  Minimum detectable r (80% power, N=100, alpha=0.05): {min_detectable_r:.2f}")
        log("")
        for _, row in power_df.iterrows():
            log(f"  [{row['predictor']}] {row['analysis_type']}:")
            log(f"    Effect size: {row['effect_size']:.4f} ({row['interpretation']})")
            log(f"    Approximate power: {row['power']:.2f}")
            log("")
        log("=" * 80)

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
