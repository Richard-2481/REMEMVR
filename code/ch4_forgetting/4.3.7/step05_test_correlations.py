#!/usr/bin/env python3
"""Test Intercept-Slope Correlation Per Paradigm: Test intercept-slope correlation for each paradigm (IFR, ICR, IRE) separately"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import test_intercept_slope_correlation_d068

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.7
LOG_FILE = RQ_DIR / "logs" / "step05_test_correlations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Function: Fisher Z-Transformation for Confidence Intervals

def compute_fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute confidence interval for Pearson r using Fisher z-transformation.

    Formula:
        z = arctanh(r)
        se = 1 / sqrt(n - 3)
        CI = tanh(z ± z_critical * se)

    Parameters
    ----------
    r : float
        Pearson correlation coefficient
    n : int
        Sample size
    alpha : float
        Significance level (default: 0.05 for 95% CI)

    Returns
    -------
    Tuple[float, float]
        (CI_lower, CI_upper) both bounded to [-1, 1]
    """
    from scipy import stats

    # Fisher z-transformation
    z = np.arctanh(r)

    # Standard error
    se = 1.0 / np.sqrt(n - 3)

    # Critical value for two-tailed test
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Confidence interval on z scale
    z_lower = z - z_critical * se
    z_upper = z + z_critical * se

    # Transform back to r scale
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)

    # Ensure bounds in [-1, 1]
    ci_lower = max(-1.0, min(1.0, ci_lower))
    ci_upper = max(-1.0, min(1.0, ci_upper))

    return ci_lower, ci_upper

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Test Intercept-Slope Correlation Per Paradigm")
        # Load Random Effects Data

        log("Loading random effects data...")
        random_effects_path = RQ_DIR / "data" / "step04_random_effects.csv"
        random_effects = pd.read_csv(random_effects_path)
        log(f"{random_effects_path.name} ({len(random_effects)} rows, {len(random_effects.columns)} cols)")

        # Verify expected structure
        expected_cols = ['UID', 'paradigm', 'Total_Intercept', 'Total_Slope']
        if list(random_effects.columns) != expected_cols:
            raise ValueError(f"Expected columns {expected_cols}, got {list(random_effects.columns)}")

        # Verify paradigms present
        paradigms = sorted(random_effects['paradigm'].unique())
        log(f"Paradigms present: {paradigms}")
        if len(paradigms) != 3 or not all(p in paradigms for p in ['IFR', 'ICR', 'IRE']):
            raise ValueError(f"Expected paradigms ['IFR', 'ICR', 'IRE'], got {paradigms}")
        # Run Correlation Tests Per Paradigm

        log("Running test_intercept_slope_correlation_d068 per paradigm...")

        correlation_results = []
        interpretations = []

        for paradigm in ['IFR', 'ICR', 'IRE']:
            log(f"Testing {paradigm}...")

            # Filter to current paradigm
            paradigm_data = random_effects[random_effects['paradigm'] == paradigm].copy()
            n_participants = len(paradigm_data)
            log(f"  N = {n_participants} participants")

            # Call correlation test tool
            result = test_intercept_slope_correlation_d068(
                random_effects_df=paradigm_data,
                family_alpha=0.05,
                n_tests=15,  # Decision D068: 15 tests in family
                intercept_col='Total_Intercept',
                slope_col='Total_Slope'
            )

            # Compute 95% CI using Fisher z-transformation
            ci_lower, ci_upper = compute_fisher_z_ci(result['r'], n_participants, alpha=0.05)

            # Build result row
            correlation_results.append({
                'paradigm': paradigm,
                'r': result['r'],
                'p_uncorrected': result['p_uncorrected'],
                'p_bonferroni': result['p_bonferroni'],
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'interpretation': result['interpretation']
            })

            # Collect interpretation
            interpretations.append(f"{paradigm}: {result['interpretation']}")

            log(f"  r = {result['r']:.3f}, p_uncorr = {result['p_uncorrected']:.4f}, p_bonf = {result['p_bonferroni']:.4f}")
            log(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        log("All correlation tests complete")

        # Create DataFrame from results
        correlation_df = pd.DataFrame(correlation_results)
        # Save Analysis Outputs
        # These outputs will be used by: Step 6 (ICC comparison + plot preparation)

        log("Saving correlation results...")

        # Output 1: Correlation results CSV (3 rows × 7 columns)
        correlation_csv_path = RQ_DIR / "data" / "step05_intercept_slope_correlation.csv"
        correlation_df.to_csv(correlation_csv_path, index=False, encoding='utf-8')
        log(f"{correlation_csv_path.name} ({len(correlation_df)} rows, {len(correlation_df.columns)} cols)")

        # Output 2: Interpretation text file
        interpretation_path = RQ_DIR / "data" / "step05_correlation_interpretation.txt"
        interpretation_text = "INTERCEPT-SLOPE CORRELATION ANALYSIS\n" + "=" * 60 + "\n\n"
        interpretation_text += "Decision D068: Dual p-value reporting (uncorrected + Bonferroni)\n"
        interpretation_text += "Family size: 15 tests (Bonferroni correction factor)\n\n"
        interpretation_text += "\n\n".join(interpretations)
        interpretation_text += "\n\n" + "=" * 60 + "\n"
        interpretation_text += "SUMMARY:\n"
        interpretation_text += f"Tested {len(paradigms)} paradigms (IFR, ICR, IRE) with N=100 per paradigm\n"
        interpretation_text += f"Confidence intervals computed via Fisher z-transformation (95% CI)\n"

        with open(interpretation_path, 'w', encoding='utf-8') as f:
            f.write(interpretation_text)
        log(f"{interpretation_path.name}")
        # Run Validation Tool
        # Validates: Decision D068 compliance (dual p-values), r bounds, CI validity
        # Threshold: r in [-1, 1], p_bonferroni >= p_uncorrected, CI_lower < CI_upper

        log("Running validate_correlation_test_d068...")
        validation_result = validate_correlation_test_d068(
            correlation_df=correlation_df,
            required_cols=['paradigm', 'r', 'p_uncorrected', 'p_bonferroni', 'CI_lower', 'CI_upper', 'interpretation']
        )

        # Report validation results
        if isinstance(validation_result, dict):
            if validation_result.get('valid', False):
                log(f"PASS: {validation_result.get('message', 'All checks passed')}")
            else:
                log(f"FAIL: {validation_result.get('message', 'Unknown error')}")
                raise ValueError(f"Validation failed: {validation_result.get('message')}")
        else:
            log(f"{validation_result}")

        log("Step 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
