#!/usr/bin/env python3
"""power_analysis: Post-hoc power analysis and sensitivity testing:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.stats.power import FTestAnovaPower

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_power_analysis.log"
OUTPUT_POWER = RQ_DIR / "data" / "step08_power_analysis.csv"
OUTPUT_SENS = RQ_DIR / "data" / "step08_sensitivity.csv"

# Configuration

BONFERRONI_ALPHA = 0.05 / 28 / 5  # Decision D068
CURRENT_N = 100
POWER_TARGET = 0.80

# Cohen's effect size conventions for f²
EFFECT_SIZES = {
    'small': 0.02,
    'medium': 0.15,
    'large': 0.35
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Power Calculation Function

def compute_power(f2, n, k, alpha):
    """
    Compute achieved power for given effect size, sample size, predictors, and alpha.

    Parameters:
    - f2: Cohen's f² (effect size)
    - n: Sample size
    - k: Number of predictors
    - alpha: Significance level

    Returns:
    - power: Achieved power [0, 1]
    """
    power_test = FTestAnovaPower()
    df_num = k
    df_denom = n - k - 1

    # Avoid bug #19 - don't use k_constraint parameter
    power = power_test.solve_power(
        effect_size=f2,
        nobs=n,
        alpha=alpha,
        power=None  # Solve for power
    )

    return power

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 08: Power Analysis")
        # Load Effect Sizes
        log("Loading effect sizes...")

        es_path = RQ_DIR / "data" / "step05_effect_sizes.csv"
        es_df = pd.read_csv(es_path)

        log(f"{es_path.name} ({len(es_df)} models)")
        # Load Model Results (for predictor counts)
        log("Loading model results...")

        models_path = RQ_DIR / "data" / "step04_model_results.csv"
        models_df = pd.read_csv(models_path)

        log(f"{models_path.name} ({len(models_df)} models)")
        # Compute Achieved Power for Each Model
        log("Computing achieved power for observed effects...")

        power_results = []

        # Model specifications (need k = number of predictors)
        models_k = {
            'Model_1_Total': 1,
            'Model_2_Learning': 1,
            'Model_3_LearningSlope': 1,
            'Model_4_Forgetting': 1,
            'Model_5_Recognition': 1,
            'Model_6_PctRet': 1,
            'Model_7_Combined': 2
        }

        for _, row in es_df.iterrows():
            model_name = row['model']
            f2 = row['f2']
            k = models_k[model_name]

            log(f"[{model_name}] f²={f2:.4f}, k={k}, N={CURRENT_N}")

            # Compute achieved power
            achieved_power = compute_power(f2, CURRENT_N, k, BONFERRONI_ALPHA)

            log(f"  Achieved power: {achieved_power:.4f}")

            # Interpret power
            if achieved_power >= 0.80:
                interpretation = "Adequate"
                adequate_flag = True
            elif achieved_power >= 0.50:
                interpretation = "Moderate"
                adequate_flag = False
            else:
                interpretation = "Low"
                adequate_flag = False

            power_results.append({
                'model': model_name,
                'observed_f2': f2,
                'alpha': BONFERRONI_ALPHA,
                'achieved_power': achieved_power,
                'interpretation': interpretation,
                'adequate_power_flag': adequate_flag
            })

        log("Achieved power for all models")
        # Sensitivity Analysis
        log("Computing minimum detectable effects and required N...")

        sensitivity_results = []

        # For each standard effect size, compute required N at 80% power
        for effect_label, f2_threshold in EFFECT_SIZES.items():
            log(f"{effect_label.capitalize()} effect (f²={f2_threshold})...")

            # Compute required N for 80% power (assume k=1 for simple case)
            power_test = FTestAnovaPower()

            try:
                required_n = power_test.solve_power(
                    effect_size=f2_threshold,
                    nobs=None,  # Solve for N
                    alpha=BONFERRONI_ALPHA,
                    power=POWER_TARGET
                )

                log(f"  Required N for 80% power: {required_n:.0f}")
            except:
                required_n = np.nan
                log(f"  Could not compute required N")

            # Compute power at current N
            current_power = compute_power(f2_threshold, CURRENT_N, k=1, alpha=BONFERRONI_ALPHA)

            log(f"  Power at current N={CURRENT_N}: {current_power:.4f}")

            sensitivity_results.append({
                'effect_size': effect_label,
                'f2_threshold': f2_threshold,
                'required_N_80pct': required_n,
                'current_N_power': current_power
            })

        log("Sensitivity analysis complete")
        # Save Results
        log("Saving power analysis results...")

        power_df = pd.DataFrame(power_results)
        power_df.to_csv(OUTPUT_POWER, index=False, encoding='utf-8')
        log(f"{OUTPUT_POWER} ({len(power_df)} models)")

        log("Saving sensitivity analysis results...")

        sens_df = pd.DataFrame(sensitivity_results)
        sens_df.to_csv(OUTPUT_SENS, index=False, encoding='utf-8')
        log(f"{OUTPUT_SENS} ({len(sens_df)} effect sizes)")
        # Summary
        log("Power analysis results:")

        for _, row in power_df.iterrows():
            adequate_mark = "" if row['adequate_power_flag'] else ""
            log(f"  {row['model']}: power={row['achieved_power']:.4f} {adequate_mark} ({row['interpretation']})")

        log(f"Models with adequate power: {power_df['adequate_power_flag'].sum()}/{len(power_df)}")

        log("Sensitivity analysis:")
        log(f"  At N={CURRENT_N} with alpha={BONFERRONI_ALPHA:.6f}:")
        for _, row in sens_df.iterrows():
            log(f"    {row['effect_size'].capitalize()} effect (f²={row['f2_threshold']}): "
                f"power={row['current_N_power']:.4f}, requires N={row['required_N_80pct']:.0f} for 80% power")

        log("Step 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
