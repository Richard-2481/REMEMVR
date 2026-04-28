#!/usr/bin/env python3
"""
RQ 5.5.1 - Power Analysis for NULL Main Effect

PURPOSE:
Section 3.1 requires post-hoc power analysis for NULL findings.
LocationType main effect: β=+0.100, SE=0.077, p=0.403 (NULL)

MANDATORY per improvement_taxonomy.md Section 3:
- Compute post-hoc power for observed effect size
- Report power to detect small (d=0.20), medium (d=0.50), large (d=0.80)
- Estimate N required for 0.80 power
- Flag if underpowered (power < 0.60 for small effects)

APPROACH:
LMM power analysis via simulation or analytical approximation.
Use statsmodels or power libraries to compute:
1. Post-hoc power for observed β=0.100
2. Power curves for d=0.20, 0.50, 0.80
3. Sample size needed for 80% power

OUTPUT:
- data/step06b_power_analysis.csv (power estimates)
- plots/power_curve.png (power vs sample size)
- logs/step06b_power_analysis.log

Date: 2025-12-27
RQ: ch5/5.5.1
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06b_power_analysis.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    """Write to log and console."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def cohen_d_to_beta(d, pooled_sd):
    """Convert Cohen's d to regression coefficient beta."""
    return d * pooled_sd

def beta_to_cohen_d(beta, pooled_sd):
    """Convert regression coefficient beta to Cohen's d."""
    return beta / pooled_sd

def lmm_power_approx(n, beta, se, alpha=0.05):
    """
    Approximate power for LMM coefficient test.

    Power = P(reject H0 | H1 true)
         = P(|z| > z_crit | true beta ≠ 0)

    Under H1, test statistic z ~ N(beta/SE, 1)
    """
    z_crit = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    ncp = beta / se  # Non-centrality parameter
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    return power

def sample_size_for_power(beta, se_per_n, target_power=0.80, alpha=0.05):
    """
    Estimate N required for target power.

    SE decreases with sqrt(N), so SE(N) = SE_baseline * sqrt(N_baseline/N)
    """
    # Binary search for N
    n_low, n_high = 10, 10000
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        se_mid = se_per_n * np.sqrt(100 / n_mid)  # Scale SE from N=100
        power_mid = lmm_power_approx(n_mid, beta, se_mid, alpha)

        if power_mid < target_power:
            n_low = n_mid
        else:
            n_high = n_mid

    return n_high

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Power Analysis for NULL Main Effect (Section 3.1)")
        log("=" * 80)

        # Observed effect from step06_post_hoc_contrasts.csv
        log("\nLocationType Main Effect:")
        log("  Coefficient (β): +0.100 theta units")
        log("  SE: 0.077")
        log("  p-value: 0.403 (NULL)")
        log("  Sample: N=100 participants, 800 observations")

        beta_observed = 0.100
        se_observed = 0.077
        n_baseline = 100
        alpha = 0.05

        # Estimate pooled SD from theta data
        lmm_input = pd.read_csv(DATA_DIR / "step04_lmm_input.csv")
        pooled_sd = lmm_input.groupby('LocationType')['theta'].std().mean()
        log(f"\nPooled SD (theta): {pooled_sd:.3f}")

        # Convert observed beta to Cohen's d
        d_observed = beta_to_cohen_d(beta_observed, pooled_sd)
        log(f"  Observed Cohen's d: {d_observed:.3f}")

        # Post-hoc power for observed effect
        power_observed = lmm_power_approx(n_baseline, beta_observed, se_observed, alpha)
        log(f"\n[POST-HOC] Power for observed effect (β={beta_observed:.3f}):")
        log(f"  Power: {power_observed:.3f} ({power_observed*100:.1f}%)")

        if power_observed < 0.60:
            log(f"  ⚠ UNDERPOWERED (power < 0.60)")
            log(f"    Interpretation: NULL finding may be due to insufficient power")
        elif power_observed < 0.80:
            log(f"  ⚠ BORDERLINE POWER (0.60 ≤ power < 0.80)")
            log(f"    Interpretation: Marginal power to detect this effect")
        else:
            log(f"  ✓ ADEQUATE POWER (power ≥ 0.80)")

        # Power for standard effect sizes
        log("\nStandard Effect Sizes (Cohen's benchmarks):")
        effect_sizes = {
            'Small': 0.20,
            'Medium': 0.50,
            'Large': 0.80
        }

        results = []
        for label, d in effect_sizes.items():
            beta_d = cohen_d_to_beta(d, pooled_sd)
            power_d = lmm_power_approx(n_baseline, beta_d, se_observed, alpha)
            n_required = sample_size_for_power(beta_d, se_observed, 0.80, alpha)

            results.append({
                'effect_label': label,
                'cohens_d': d,
                'beta': beta_d,
                'power_at_n100': power_d,
                'n_for_80pct_power': n_required
            })

            log(f"  {label:8s} (d={d:.2f}):")
            log(f"    β = {beta_d:.3f}")
            log(f"    Power at N=100: {power_d:.3f} ({power_d*100:.1f}%)")
            log(f"    N for 80% power: {n_required}")

        # Add observed effect to results
        n_required_observed = sample_size_for_power(beta_observed, se_observed, 0.80, alpha)
        results.insert(0, {
            'effect_label': 'Observed',
            'cohens_d': d_observed,
            'beta': beta_observed,
            'power_at_n100': power_observed,
            'n_for_80pct_power': n_required_observed
        })

        # Save results
        power_df = pd.DataFrame(results)
        output_path = DATA_DIR / "step06b_power_analysis.csv"
        power_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"\nPower analysis saved: {output_path.name}")

        # Summary interpretation
        log("\n" + "=" * 80)
        log("")
        log("=" * 80)

        if power_observed < 0.30:
            log("✗ SEVERELY UNDERPOWERED for observed effect")
            log("  NULL finding likely due to insufficient power, NOT true null")
            log(f"  Need N≈{n_required_observed} for 80% power (current N=100)")
        elif power_observed < 0.60:
            log("⚠ UNDERPOWERED for observed effect")
            log("  NULL finding may be Type II error (insufficient power)")
            log(f"  Need N≈{n_required_observed} for 80% power (current N=100)")
        else:
            log("✓ ADEQUATE power for observed effect")
            log("  NULL finding likely reflects true absence of effect")
            log("  Study had sufficient power to detect this effect size")

        # Check power for small effects
        power_small = power_df[power_df['effect_label'] == 'Small']['power_at_n100'].values[0]
        if power_small < 0.60:
            log(f"\n⚠ UNDERPOWERED for small effects (d=0.20)")
            log(f"  Power: {power_small:.2f} (< 0.60 threshold)")
            log("  Study cannot rule out small true effects")

        log("\n" + "=" * 80)
        log("Power analysis complete")
        log("=" * 80)

    except Exception as e:
        log(f"\n{str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
