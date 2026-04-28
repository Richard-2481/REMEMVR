#!/usr/bin/env python3
"""
RQ 6.2.5: Power Analysis and TOST for NULL Finding (Validation Requirement)
==========================================================================

Per improvement_taxonomy.md Section 3 (MANDATORY for NULL findings):
- Power analysis to demonstrate adequate power for meaningful effects
- TOST (Two One-Sided Tests) for equivalence testing

Primary NULL: Age × Time interaction (β = 0.000019, p = 1.000)

Equivalence bounds: ±0.002 per TSVR_hour (corresponds to ±0.4 z-score change
over 200 hours for a 20-year age difference, which is a small but meaningful
effect on calibration bias).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.stats.power as smp

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step12e_power_and_tost.log"

# Observed effect from corrected LMM
OBSERVED_BETA = 0.000019  # Age × Time interaction coefficient
OBSERVED_SE = 0.000045    # Standard error
N_OBS = 400
N_PARAMS = 5

# Equivalence bounds for TOST
EQUIV_LOWER = -0.002  # Lower equivalence bound
EQUIV_UPPER = +0.002  # Upper equivalence bound

# Power analysis effect sizes
SMALL_EFFECT = 0.0005   # Small interaction (0.1 z-score over 200h for 20y age diff)
MEDIUM_EFFECT = 0.0010  # Medium interaction (0.2 z-score)
LARGE_EFFECT = 0.002   # Large interaction (0.4 z-score, at equiv bound)


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def power_analysis():
    """
    Compute post-hoc power for detecting interaction effects.

    Uses t-test approximation: df = N - k, noncentrality parameter = effect / SE
    """
    log("\n" + "=" * 70)
    log("POWER ANALYSIS")
    log("=" * 70)

    df = N_OBS - N_PARAMS
    alpha = 0.05
    log(f"\nStudy parameters:")
    log(f"  N observations: {N_OBS}")
    log(f"  N parameters: {N_PARAMS}")
    log(f"  df: {df}")
    log(f"  alpha: {alpha}")
    log(f"  Observed SE: {OBSERVED_SE:.6f}")

    # Critical t-value for two-tailed test
    t_crit = stats.t.ppf(1 - alpha/2, df)
    log(f"  Critical t-value (two-tailed): {t_crit:.3f}")

    # Power for observed effect
    noncentrality_obs = abs(OBSERVED_BETA) / OBSERVED_SE
    power_obs = 1 - stats.nct.cdf(t_crit, df, noncentrality_obs) + stats.nct.cdf(-t_crit, df, noncentrality_obs)

    log(f"\n--- Power for Observed Effect ---")
    log(f"Observed β: {OBSERVED_BETA:.6f}")
    log(f"Noncentrality parameter: {noncentrality_obs:.3f}")
    log(f"Power: {power_obs:.3f}")

    # Power for small, medium, large effects
    effects = {
        'Small': SMALL_EFFECT,
        'Medium': MEDIUM_EFFECT,
        'Large': LARGE_EFFECT
    }

    power_results = []
    log(f"\n--- Power for Hypothetical Effects ---")

    for effect_name, effect_size in effects.items():
        noncentrality = effect_size / OBSERVED_SE
        power = 1 - stats.nct.cdf(t_crit, df, noncentrality) + stats.nct.cdf(-t_crit, df, noncentrality)

        log(f"\n{effect_name} effect (β = {effect_size:.4f}):")
        log(f"  Noncentrality: {noncentrality:.3f}")
        log(f"  Power: {power:.3f} {'✓ (>0.80)' if power >= 0.80 else '✗ (<0.80)'}")

        power_results.append({
            'effect_size_label': effect_name,
            'effect_size_beta': effect_size,
            'noncentrality': noncentrality,
            'power': power,
            'adequate': power >= 0.80
        })

    # Add observed effect
    power_results.insert(0, {
        'effect_size_label': 'Observed',
        'effect_size_beta': OBSERVED_BETA,
        'noncentrality': noncentrality_obs,
        'power': power_obs,
        'adequate': power_obs >= 0.80
    })

    power_df = pd.DataFrame(power_results)

    # Interpretation
    log("\n" + "=" * 70)
    log("POWER INTERPRETATION")
    log("=" * 70)

    adequate_count = power_df['adequate'].sum()
    if adequate_count >= 3:  # Observed + at least 2 hypothetical
        log("\n✓ Study adequately powered:")
        log(f"  - {adequate_count}/4 effect sizes have power ≥ 0.80")
        log("  - Can detect meaningful Age × Time interactions")
        log("  - NULL finding is NOT due to inadequate power")
    else:
        log("\n⚠️  Study underpowered:")
        log(f"  - Only {adequate_count}/4 effect sizes have power ≥ 0.80")
        log("  - May miss small/medium Age × Time interactions")
        log("  - NULL finding could be due to inadequate power")

    return power_df


def tost_equivalence():
    """
    Perform TOST (Two One-Sided Tests) for equivalence.

    Tests whether observed effect is statistically equivalent to zero
    (within equivalence bounds).
    """
    log("\n" + "=" * 70)
    log("TOST EQUIVALENCE TESTING")
    log("=" * 70)

    log(f"\nEquivalence bounds:")
    log(f"  Lower: {EQUIV_LOWER:.4f}")
    log(f"  Upper: {EQUIV_UPPER:.4f}")
    log(f"\nObserved effect:")
    log(f"  β: {OBSERVED_BETA:.6f}")
    log(f"  SE: {OBSERVED_SE:.6f}")

    # Compute 90% CI (for equivalence testing, use 1-2α = 0.90)
    df = N_OBS - N_PARAMS
    t_crit_90 = stats.t.ppf(0.95, df)  # One-sided 0.05 = two-sided 0.10
    ci_lower = OBSERVED_BETA - t_crit_90 * OBSERVED_SE
    ci_upper = OBSERVED_BETA + t_crit_90 * OBSERVED_SE

    log(f"\n90% Confidence Interval:")
    log(f"  [{ci_lower:.6f}, {ci_upper:.6f}]")

    # TOST: Test 1 (β > lower bound)
    t1 = (OBSERVED_BETA - EQUIV_LOWER) / OBSERVED_SE
    p1 = 1 - stats.t.cdf(t1, df)  # One-sided test

    # TOST: Test 2 (β < upper bound)
    t2 = (OBSERVED_BETA - EQUIV_UPPER) / OBSERVED_SE
    p2 = stats.t.cdf(t2, df)  # One-sided test

    # TOST p-value is the maximum of the two
    p_tost = max(p1, p2)

    log(f"\n--- TOST Test 1: β > {EQUIV_LOWER:.4f} ---")
    log(f"  t-statistic: {t1:.3f}")
    log(f"  p-value (one-sided): {p1:.4f}")

    log(f"\n--- TOST Test 2: β < {EQUIV_UPPER:.4f} ---")
    log(f"  t-statistic: {t2:.3f}")
    log(f"  p-value (one-sided): {p2:.4f}")

    log(f"\n--- TOST Result ---")
    log(f"  TOST p-value: {p_tost:.4f}")

    if p_tost < 0.05:
        conclusion = "EQUIVALENT (p < 0.05)"
        interpretation = "Effect is statistically equivalent to zero"
    elif p_tost < 0.10:
        conclusion = "MARGINALLY EQUIVALENT (p < 0.10)"
        interpretation = "Effect is marginally equivalent to zero"
    else:
        conclusion = "NOT EQUIVALENT (p ≥ 0.10)"
        interpretation = "Cannot conclude statistical equivalence to zero"

    log(f"  Conclusion: {conclusion}")
    log(f"  Interpretation: {interpretation}")

    # Visual check
    log(f"\n--- Equivalence Region Check ---")
    log(f"  90% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    log(f"  Equiv bounds: [{EQUIV_LOWER:.4f}, {EQUIV_UPPER:.4f}]")

    ci_inside = (ci_lower > EQUIV_LOWER) and (ci_upper < EQUIV_UPPER)
    log(f"  CI entirely inside bounds: {ci_inside}")

    tost_df = pd.DataFrame([{
        'observed_beta': OBSERVED_BETA,
        'observed_se': OBSERVED_SE,
        'equiv_lower': EQUIV_LOWER,
        'equiv_upper': EQUIV_UPPER,
        'ci_90_lower': ci_lower,
        'ci_90_upper': ci_upper,
        't1_statistic': t1,
        't1_p_value': p1,
        't2_statistic': t2,
        't2_p_value': p2,
        'tost_p_value': p_tost,
        'conclusion': conclusion,
        'ci_inside_bounds': ci_inside
    }])

    return tost_df


def main():
    """Execute power analysis and TOST."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.2.5: Power Analysis and TOST\n")
        f.write("=" * 70 + "\n")
        f.write("Validation Requirement: Section 3 (NULL findings)\n\n")

    log("Starting power analysis and equivalence testing...")

    try:
        # Power analysis
        power_df = power_analysis()

        # TOST equivalence testing
        tost_df = tost_equivalence()

        # Save results
        power_path = RQ_DIR / "data" / "step12e_power_analysis.csv"
        power_df.to_csv(power_path, index=False)
        log(f"\n✓ Power analysis saved: {power_path}")

        tost_path = RQ_DIR / "data" / "step12e_tost_equivalence.csv"
        tost_df.to_csv(tost_path, index=False)
        log(f"✓ TOST results saved: {tost_path}")

        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        log(f"\n✅ Power analysis and TOST complete")
        log(f"Power for medium effects: {power_df[power_df['effect_size_label']=='Medium']['power'].iloc[0]:.3f}")
        log(f"TOST p-value: {tost_df['tost_p_value'].iloc[0]:.4f}")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
