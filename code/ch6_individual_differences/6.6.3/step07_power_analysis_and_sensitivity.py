#!/usr/bin/env python3
"""power_analysis_and_sensitivity: Conduct post-hoc power analysis for What-Where ICC comparison and sensitivity"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.analysis_regression import compute_post_hoc_power
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_power_analysis_and_sensitivity.log"
OUTPUT_FILE = RQ_DIR / "data" / "step07_power_analysis.csv"

# Input files
INPUT_ICC = RQ_DIR / "data" / "step02_icc_estimates.csv"
INPUT_COMPARISONS = RQ_DIR / "data" / "step04_pairwise_comparisons.csv"

# Power analysis parameters
N_PARTICIPANTS = 100
ALPHA = 0.05  # Single test
POWER_TARGET = 0.80
EFFECT_SIZE_SMALL = 0.2  # Cohen's d = 0.2 (small effect)

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Power Calculation Helper Functions

def cohen_d_to_r2(d, n1, n2):
    """
    Convert Cohen's d to R^2 for regression-based power calculation.

    Formula: R^2 = d^2 / (d^2 + (n1 + n2)^2 / (n1 * n2))
    """
    r2 = (d ** 2) / (d ** 2 + ((n1 + n2) ** 2) / (n1 * n2))
    return r2

def solve_for_effect_size(power, n, alpha, k=1):
    """
    Solve for minimum detectable effect size (R^2) given power.

    Uses binary search to find R^2 that achieves target power.
    """
    from scipy.optimize import brentq

    def power_diff(r2):
        try:
            calc_power = compute_post_hoc_power(n=n, k_predictors=k, r2=r2, alpha=alpha)
            return calc_power - power
        except:
            return -1  # Invalid R^2

    try:
        # Search for R^2 in [0.001, 0.999]
        r2_target = brentq(power_diff, 0.001, 0.999)
        return r2_target
    except:
        return np.nan

def r2_to_cohen_d(r2, n1, n2):
    """
    Convert R^2 back to Cohen's d.

    Inverse of cohen_d_to_r2 formula.
    """
    d_squared = r2 * ((n1 + n2) ** 2) / ((1 - r2) * n1 * n2)
    d = np.sqrt(d_squared) if d_squared >= 0 else 0
    return d

def solve_for_sample_size(effect_size, power, alpha):
    """
    Solve for required sample size to detect effect at target power.

    Uses binary search to find N that achieves target power.
    """
    from scipy.optimize import brentq

    # Convert Cohen's d to R^2 (assuming equal groups)
    def n_to_power(n_total):
        n_per_group = int(n_total / 2)
        r2 = cohen_d_to_r2(effect_size, n_per_group, n_per_group)
        try:
            calc_power = compute_post_hoc_power(n=int(n_total), k_predictors=1, r2=r2, alpha=alpha)
            return calc_power - power
        except:
            return -1

    try:
        # Search for N in [20, 2000]
        n_required = brentq(n_to_power, 20, 2000)
        return int(np.ceil(n_required))
    except:
        return np.nan

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 07: Power Analysis and Sensitivity")
        # Load Data

        log(f"Reading ICC estimates...")
        df_icc = pd.read_csv(INPUT_ICC)
        log(f"{len(df_icc)} rows")

        log(f"Reading pairwise comparisons...")
        df_comparisons = pd.read_csv(INPUT_COMPARISONS)
        log(f"{len(df_comparisons)} rows")
        # Extract Observed Effect Size

        comparison = df_comparisons['comparison'].values[0]
        observed_d = df_comparisons['cohens_d'].values[0]

        log(f"Comparison: {comparison}")
        log(f"Cohen's d: {observed_d:.4f}")
        # Calculate Achieved Power
        # Power to detect observed effect size with N=100, alpha=0.05

        log(f"Calculating achieved power...")
        log(f"N={N_PARTICIPANTS}, alpha={ALPHA}, d={observed_d:.4f}")

        # Convert Cohen's d to R^2 (equal groups, n=50 each)
        n_per_group = N_PARTICIPANTS // 2
        r2_observed = cohen_d_to_r2(observed_d, n_per_group, n_per_group)

        log(f"d={observed_d:.4f} -> R^2={r2_observed:.6f}")

        # Calculate power using tools.analysis_regression.compute_post_hoc_power
        achieved_power = compute_post_hoc_power(
            n=N_PARTICIPANTS,
            k_predictors=1,  # Single predictor (domain)
            r2=r2_observed,
            alpha=ALPHA
        )

        log(f"Power: {achieved_power:.4f}")
        # Calculate Minimum Detectable Effect Size
        # Effect size (Cohen's d) detectable at 80% power with N=100, alpha=0.05

        log(f"Calculating minimum detectable effect size...")
        log(f"N={N_PARTICIPANTS}, power={POWER_TARGET}, alpha={ALPHA}")

        # Solve for R^2 that gives 80% power
        r2_min = solve_for_effect_size(POWER_TARGET, N_PARTICIPANTS, ALPHA, k=1)

        # Convert back to Cohen's d
        min_detectable_d = r2_to_cohen_d(r2_min, n_per_group, n_per_group)

        log(f"Detectable d: {min_detectable_d:.4f} (at 80% power)")
        # Calculate Required Sample Size for Small Effect
        # Sample size needed to detect d=0.2 at 80% power, alpha=0.05

        log(f"Calculating required N for small effect...")
        log(f"d={EFFECT_SIZE_SMALL}, power={POWER_TARGET}, alpha={ALPHA}")

        required_n_d02 = solve_for_sample_size(EFFECT_SIZE_SMALL, POWER_TARGET, ALPHA)

        log(f"N={required_n_d02} participants (for d=0.2 at 80% power)")
        # Create Results DataFrame

        power_results = [{
            'comparison': comparison,
            'observed_d': observed_d,
            'achieved_power': achieved_power,
            'min_detectable_d': min_detectable_d,
            'required_n_d02': required_n_d02
        }]

        df_power = pd.DataFrame(power_results)
        log(f"Power analysis table: {len(df_power)} row")
        # Validate Results

        log(f"Checking power analysis results...")

        # Validate achieved_power in [0, 1]
        validation_power = validate_numeric_range(
            data=df_power['achieved_power'],
            min_val=0.0,
            max_val=1.0,
            column_name='achieved_power'
        )

        if not validation_power.get('valid', False):
            log(f"achieved_power validation failed: {validation_power}")
            sys.exit(1)
        else:
            log(f"achieved_power in [0, 1]")

        # Validate min_detectable_d > 0
        if min_detectable_d <= 0:
            log(f"min_detectable_d = {min_detectable_d} <= 0")
            sys.exit(1)
        else:
            log(f"min_detectable_d > 0")

        # Validate required_n_d02 > 100
        if required_n_d02 <= 100:
            log(f"required_n_d02 = {required_n_d02} <= 100 (current sample)")
            log(f"Small effects detectable with current sample size")
        else:
            log(f"required_n_d02 > 100 (larger sample needed for d=0.2)")
        # Save Results

        log(f"Writing power analysis...")
        df_power.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE} ({len(df_power)} row)")

        log(f"Step 07 complete")
        log(f"Achieved power: {achieved_power:.4f}, Min detectable d: {min_detectable_d:.4f}")
        log(f"All 8 analysis steps complete for RQ 7.6.3")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
