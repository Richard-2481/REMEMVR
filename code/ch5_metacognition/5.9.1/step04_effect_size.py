#!/usr/bin/env python3
"""Compute Effect Size (Cohen's d): Compute Cohen's d for paired samples: d = mean(difference) / SD(difference)."""

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
LOG_FILE = RQ_DIR / "logs" / "step04_effect_size.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 4: Compute Effect Size (Cohen's d)")
        # Load Input Data

        log("Loading input data...")

        # Load individual decline rates
        df_rates = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv", encoding='utf-8')
        log(f"step02_individual_decline_rates.csv ({len(df_rates)} rows, {len(df_rates.columns)} cols)")
        # Compute Cohen's d (Paired Samples)

        log("Computing Cohen's d for paired samples...")

        # Extract difference array
        difference = df_rates['difference'].values

        # Compute Cohen's d
        # Formula: d = mean(difference) / SD(difference)
        # Note: Uses SD of differences (paired design), not pooled SD (independent design)
        mean_difference = np.mean(difference)
        sd_difference = np.std(difference, ddof=1)
        cohen_d = mean_difference / sd_difference

        log(f"Mean difference: {mean_difference:.6f}")
        log(f"SD difference: {sd_difference:.6f}")
        log(f"Cohen's d: {cohen_d:.4f}")
        # Interpret Magnitude

        log("Applying Cohen (1988) conventions...")

        # Cohen (1988) thresholds
        abs_d = abs(cohen_d)

        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        log(f"|d| = {abs_d:.4f} -> {interpretation}")
        # Hypothesis Decision Integration
        # Note: Full decision made in results analysis (combines all evidence)
        # This step just notes the criteria

        log("Hedging hypothesis SUPPORTED if:")
        log("(1) p_uncorrected < 0.05 (Step 3)")
        log("AND (2) |cohen_d| > 0.3 (medium+ effect)")
        log("Full decision made in results pipeline")
        # Save Analysis Outputs
        # These outputs will be used by: results analysis (hypothesis decision)

        log("Saving analysis outputs...")

        # Create results DataFrame
        results = {
            'cohen_d': [cohen_d],
            'interpretation': [interpretation]
        }
        df_results = pd.DataFrame(results)

        # Save effect size
        output_path = RQ_DIR / "data" / "step04_effect_size.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(df_results)} rows, {len(df_results.columns)} cols)")
        # Validation
        # Validates: Output structure, value ranges, interpretation match

        log("Running inline validation...")

        # Check rows
        if len(df_results) != 1:
            log(f"Expected 1 row, got {len(df_results)}")

        # Check columns
        if len(df_results.columns) != 2:
            log(f"Expected 2 columns, got {len(df_results.columns)}")

        # Check for NaN in cohen_d
        if pd.isna(df_results['cohen_d'].values[0]):
            log(f"cohen_d is NaN")

        # Check interpretation is valid
        valid_interpretations = ['negligible', 'small', 'medium', 'large']
        if interpretation not in valid_interpretations:
            log(f"Invalid interpretation: {interpretation}")

        # Check interpretation matches magnitude
        if abs_d < 0.2 and interpretation != 'negligible':
            log(f"|d| < 0.2 but interpretation is {interpretation} (expected negligible)")
        elif 0.2 <= abs_d < 0.5 and interpretation != 'small':
            log(f"0.2 <= |d| < 0.5 but interpretation is {interpretation} (expected small)")
        elif 0.5 <= abs_d < 0.8 and interpretation != 'medium':
            log(f"0.5 <= |d| < 0.8 but interpretation is {interpretation} (expected medium)")
        elif abs_d >= 0.8 and interpretation != 'large':
            log(f"|d| >= 0.8 but interpretation is {interpretation} (expected large)")

        # Value range
        if not (-5 <= cohen_d <= 5):
            log(f"Cohen's d outside expected range: {cohen_d}")

        log(f"Cohen's d = {cohen_d:.4f}")
        log(f"interpretation: {interpretation}")

        log("Step 4 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
