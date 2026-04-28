#!/usr/bin/env python3
"""Power Analysis and Effect Size Assessment: Comprehensive post-hoc power analysis for hierarchical regression model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels power analysis
from statsmodels.stats.power import FTestAnovaPower

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"

# Power analysis parameters
N_PARTICIPANTS = 100
N_PREDICTORS_FULL = 8  # Full model (demographics + 5 cognitive)
N_PREDICTORS_DEMO = 3  # Demographics only
N_PREDICTORS_COGNITIVE = 5  # Incremental cognitive predictors

ALPHA_STANDARD = 0.05
ALPHA_CORRECTED = 0.00179  # Ch7 chapter-wide correction
TARGET_POWER = 0.80

# Effect size benchmarks (Cohen's f²)
EFFECT_SIZE_SMALL = 0.02
EFFECT_SIZE_MEDIUM = 0.15
EFFECT_SIZE_LARGE = 0.35

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)

# Effect Size Interpretation Function

def interpret_f2(f2: float) -> str:
    """
    Interpret Cohen's f² effect size.

    Benchmarks (Cohen, 1988):
    - f² = 0.02: small
    - f² = 0.15: medium
    - f² = 0.35: large

    Parameters:
        f2: Cohen's f² effect size

    Returns:
        Interpretation string
    """
    if f2 < EFFECT_SIZE_SMALL:
        return "negligible (f² < 0.02)"
    elif f2 < EFFECT_SIZE_MEDIUM:
        return "small (0.02 <= f² < 0.15)"
    elif f2 < EFFECT_SIZE_LARGE:
        return "medium (0.15 <= f² < 0.35)"
    else:
        return "large (f² >= 0.35)"

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: Power Analysis and Effect Size Assessment")
        # Load Regression Results

        log("Loading regression results...")
        input_path = RQ_DIR / "data" / "step04_regression_results.csv"
        df = pd.read_csv(input_path)
        log(f"step04_regression_results.csv ({len(df)} rows, {len(df.columns)} cols)")
        # Extract R² Values

        log("Extracting R² values from regression results...")

        # Find rows with model-level statistics
        model1_row = df[df['predictor'] == 'MODEL1_R2']
        model2_row = df[df['predictor'] == 'MODEL2_R2']
        delta_row = df[df['predictor'] == 'DELTA_R2']

        if len(model1_row) == 0 or len(model2_row) == 0 or len(delta_row) == 0:
            raise ValueError("Could not find MODEL1_R2, MODEL2_R2, or DELTA_R2 in regression results")

        r2_model1 = model1_row['beta'].values[0]
        r2_model2 = model2_row['beta'].values[0]
        delta_r2 = delta_row['beta'].values[0]

        log(f"Model 1 (Demographics) R²: {r2_model1:.4f}")
        log(f"Model 2 (Full) R²: {r2_model2:.4f}")
        log(f"Incremental R² (Cognitive): {delta_r2:.4f}")
        # Convert R² to Cohen's f²
        # Cohen's f² = R² / (1 - R²)

        log("[EFFECT SIZE] Converting R² to Cohen's f²...")

        # Overall model effect size
        f2_full = r2_model2 / (1 - r2_model2) if r2_model2 < 1.0 else np.inf
        log(f"[EFFECT SIZE] Full model f²: {f2_full:.4f} ({interpret_f2(f2_full)})")

        # Incremental cognitive effect size
        # f²_incremental = ΔR² / (1 - R²_full)
        f2_incremental = delta_r2 / (1 - r2_model2) if r2_model2 < 1.0 else np.inf
        log(f"[EFFECT SIZE] Incremental cognitive f²: {f2_incremental:.4f} ({interpret_f2(f2_incremental)})")
        # Post-Hoc Power Analysis
        # Given: observed effect size (f²), sample size (n), alpha
        # Compute: achieved power

        log("Computing post-hoc power (achieved power given observed effects)...")

        # Initialize power analysis object
        power_test = FTestAnovaPower()

        # --- Overall Model Power ---
        log(f"--- Overall Model (R² = {r2_model2:.4f}, f² = {f2_full:.4f}) ---")

        # Degrees of freedom for overall model F-test
        df_num_full = N_PREDICTORS_FULL  # k predictors
        df_denom_full = N_PARTICIPANTS - N_PREDICTORS_FULL - 1  # n - k - 1

        # Power at alpha = 0.05
        power_full_standard = power_test.solve_power(
            effect_size=f2_full,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_STANDARD
        )
        log(f"Power at alpha = {ALPHA_STANDARD:.4f}: {power_full_standard:.4f}")

        # Power at alpha = 0.00179 (chapter correction)
        power_full_corrected = power_test.solve_power(
            effect_size=f2_full,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_CORRECTED
        )
        log(f"Power at alpha = {ALPHA_CORRECTED:.5f}: {power_full_corrected:.4f}")

        adequate_full_standard = power_full_standard >= TARGET_POWER
        adequate_full_corrected = power_full_corrected >= TARGET_POWER
        log(f"Adequate power (>= {TARGET_POWER:.2f}) at alpha=0.05: {'YES' if adequate_full_standard else 'NO'}")
        log(f"Adequate power (>= {TARGET_POWER:.2f}) at alpha=0.00179: {'YES' if adequate_full_corrected else 'NO'}")

        # --- Incremental Cognitive Power ---
        log(f"--- Incremental Cognitive (ΔR² = {delta_r2:.4f}, f² = {f2_incremental:.4f}) ---")

        # Degrees of freedom for incremental F-test
        df_num_incr = N_PREDICTORS_COGNITIVE  # 3 cognitive predictors
        df_denom_incr = N_PARTICIPANTS - N_PREDICTORS_FULL - 1  # Same as full model

        # Power at alpha = 0.05
        power_incr_standard = power_test.solve_power(
            effect_size=f2_incremental,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_STANDARD
        )
        log(f"Power at alpha = {ALPHA_STANDARD:.4f}: {power_incr_standard:.4f}")

        # Power at alpha = 0.00179 (chapter correction)
        power_incr_corrected = power_test.solve_power(
            effect_size=f2_incremental,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_CORRECTED
        )
        log(f"Power at alpha = {ALPHA_CORRECTED:.5f}: {power_incr_corrected:.4f}")

        adequate_incr_standard = power_incr_standard >= TARGET_POWER
        adequate_incr_corrected = power_incr_corrected >= TARGET_POWER
        log(f"Adequate power (>= {TARGET_POWER:.2f}) at alpha=0.05: {'YES' if adequate_incr_standard else 'NO'}")
        log(f"Adequate power (>= {TARGET_POWER:.2f}) at alpha=0.00179: {'YES' if adequate_incr_corrected else 'NO'}")
        # A Priori Sensitivity Analysis
        # Given: target power = 0.80, sample size (n), alpha
        # Compute: minimum detectable effect size (f²)

        log("Computing a priori sensitivity (minimum detectable f² for power = 0.80)...")

        # --- Overall Model Sensitivity ---
        log(f"--- Overall Model (n={N_PARTICIPANTS}, k={N_PREDICTORS_FULL}) ---")

        min_f2_full_standard = power_test.solve_power(
            effect_size=None,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_STANDARD,
            power=TARGET_POWER
        )
        log(f"Min detectable f² at alpha = {ALPHA_STANDARD:.4f}: {min_f2_full_standard:.4f} ({interpret_f2(min_f2_full_standard)})")

        min_f2_full_corrected = power_test.solve_power(
            effect_size=None,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_CORRECTED,
            power=TARGET_POWER
        )
        log(f"Min detectable f² at alpha = {ALPHA_CORRECTED:.5f}: {min_f2_full_corrected:.4f} ({interpret_f2(min_f2_full_corrected)})")

        # --- Incremental Cognitive Sensitivity ---
        log(f"--- Incremental Cognitive (n={N_PARTICIPANTS}, k={N_PREDICTORS_COGNITIVE}) ---")

        min_f2_incr_standard = power_test.solve_power(
            effect_size=None,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_STANDARD,
            power=TARGET_POWER
        )
        log(f"Min detectable f² at alpha = {ALPHA_STANDARD:.4f}: {min_f2_incr_standard:.4f} ({interpret_f2(min_f2_incr_standard)})")

        min_f2_incr_corrected = power_test.solve_power(
            effect_size=None,
            nobs=N_PARTICIPANTS,
            alpha=ALPHA_CORRECTED,
            power=TARGET_POWER
        )
        log(f"Min detectable f² at alpha = {ALPHA_CORRECTED:.5f}: {min_f2_incr_corrected:.4f} ({interpret_f2(min_f2_incr_corrected)})")
        # Save Power Analysis Results

        log("Saving power analysis results...")

        power_results = [
            {
                'analysis_type': 'Overall Model',
                'n': N_PARTICIPANTS,
                'k_predictors': N_PREDICTORS_FULL,
                'r2': r2_model2,
                'f2': f2_full,
                'power_alpha_0.05': power_full_standard,
                'power_alpha_0.00179': power_full_corrected,
                'min_detectable_f2_0.80_alpha_0.05': min_f2_full_standard,
                'min_detectable_f2_0.80_alpha_0.00179': min_f2_full_corrected,
                'interpretation': interpret_f2(f2_full),
                'adequate_power_0.05': adequate_full_standard,
                'adequate_power_0.00179': adequate_full_corrected
            },
            {
                'analysis_type': 'Incremental Cognitive',
                'n': N_PARTICIPANTS,
                'k_predictors': N_PREDICTORS_COGNITIVE,
                'r2': delta_r2,
                'f2': f2_incremental,
                'power_alpha_0.05': power_incr_standard,
                'power_alpha_0.00179': power_incr_corrected,
                'min_detectable_f2_0.80_alpha_0.05': min_f2_incr_standard,
                'min_detectable_f2_0.80_alpha_0.00179': min_f2_incr_corrected,
                'interpretation': interpret_f2(f2_incremental),
                'adequate_power_0.05': adequate_incr_standard,
                'adequate_power_0.00179': adequate_incr_corrected
            }
        ]

        power_df = pd.DataFrame(power_results)
        output_path = RQ_DIR / "data" / "step08_power_analysis.csv"
        power_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step08_power_analysis.csv ({len(power_df)} rows, {len(power_df.columns)} cols)")
        # Validation

        log("Power analysis validation checks:")

        # Check power values in range [0, 1]
        power_cols = ['power_alpha_0.05', 'power_alpha_0.00179']
        power_in_range = all((power_df[col] >= 0).all() and (power_df[col] <= 1).all() for col in power_cols)
        log(f"Power values in [0, 1]: {'PASS' if power_in_range else 'FAIL'}")

        # Check effect sizes non-negative
        f2_nonnegative = (power_df['f2'] >= 0).all()
        log(f"Effect sizes (f²) >= 0: {'PASS' if f2_nonnegative else 'FAIL'}")

        # Check minimum detectable f² non-negative
        min_f2_cols = ['min_detectable_f2_0.80_alpha_0.05', 'min_detectable_f2_0.80_alpha_0.00179']
        min_f2_nonnegative = all((power_df[col] >= 0).all() for col in min_f2_cols)
        log(f"Min detectable f² >= 0: {'PASS' if min_f2_nonnegative else 'FAIL'}")

        # Summary
        log(f"Overall model adequate power at alpha=0.05: {'YES' if adequate_full_standard else 'NO'}")
        log(f"Overall model adequate power at alpha=0.00179: {'YES' if adequate_full_corrected else 'NO'}")
        log(f"Incremental cognitive adequate power at alpha=0.05: {'YES' if adequate_incr_standard else 'NO'}")
        log(f"Incremental cognitive adequate power at alpha=0.00179: {'YES' if adequate_incr_corrected else 'NO'}")

        validation_pass = power_in_range and f2_nonnegative and min_f2_nonnegative
        log(f"Overall validation: {'PASS' if validation_pass else 'FAIL'}")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
