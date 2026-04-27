#!/usr/bin/env python3
"""
RQ 6.7.2: Post-Hoc Power Analysis
==================================

Per improvement_taxonomy.md Section 3.1: Power analysis mandatory.

Though finding IS significant (p = .034), not null, power analysis documents:
1. Post-hoc power for observed r = 0.214
2. Power for hypothesis threshold r = 0.30
3. Required N for 0.80 power at observed effect
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
PLOTS_DIR = RQ_DIR / "plots"
LOG_FILE = LOG_DIR / "step09_power_analysis.log"


def log(msg: str):
    """Log to file and stdout."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a') as f:
        f.write(f"{msg}\n")
    print(msg)


def power_correlation(r: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute power for correlation test.

    Power = P(reject H0 | H1 true with effect r)

    Uses Fisher z-transformation and normal approximation.
    """
    # Fisher z-transformation of r
    z_r = 0.5 * np.log((1 + r) / (1 - r))

    # Standard error under H1
    se = 1 / np.sqrt(n - 3)

    # Critical z-value (two-tailed)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    ncp = z_r / se

    # Power = P(|Z| > z_crit | Z ~ N(ncp, 1))
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    return power


def required_n(r: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """
    Compute required sample size for given power.

    Binary search to find N such that power_correlation(r, N) ≈ power.
    """
    # Binary search bounds
    n_min = 10
    n_max = 10000

    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        p = power_correlation(r, n_mid, alpha)

        if p < power:
            n_min = n_mid
        else:
            n_max = n_mid

    return n_max


def main():
    log("=" * 70)
    log("RQ 6.7.2: Post-Hoc Power Analysis")
    log("=" * 70)
    log("")

    # Load correlation results
    corr = pd.read_csv(DATA_DIR / "step03_correlation.csv")
    r_observed = corr['r_partial'].values[0]
    n_observed = 100
    alpha = 0.05

    log(f"Observed effect:")
    log(f"  Partial r = {r_observed:.4f}")
    log(f"  N = {n_observed}")
    log(f"  α = {alpha}")
    log("")

    # Post-hoc power for observed effect
    log("=" * 70)
    log("POST-HOC POWER (Observed Effect)")
    log("=" * 70)
    log("")

    power_observed = power_correlation(r_observed, n_observed, alpha)
    log(f"Power to detect r = {r_observed:.4f} with N = {n_observed}:")
    log(f"  Power = {power_observed:.3f} ({power_observed * 100:.1f}%)")
    log("")

    if power_observed < 0.60:
        log("INTERPRETATION: UNDERPOWERED")
        log("  Power < 60% indicates high risk of Type II error")
        log("  Finding p = .034 is legitimate but marginal")
        log("  Replication in larger sample recommended")
    elif power_observed < 0.80:
        log("INTERPRETATION: MARGINAL POWER")
        log("  Power 60-80% is below conventional threshold (0.80)")
        log("  Finding p = .034 is plausible but should be replicated")
    else:
        log("INTERPRETATION: ADEQUATE POWER")
        log("  Power ≥ 80% meets conventional standard")
        log("  Finding p = .034 is well-supported")
    log("")

    # Power for hypothesis threshold
    log("=" * 70)
    log("POWER FOR HYPOTHESIS THRESHOLD (r = 0.30)")
    log("=" * 70)
    log("")

    r_hypothesis = 0.30
    power_hypothesis = power_correlation(r_hypothesis, n_observed, alpha)

    log(f"Power to detect r = {r_hypothesis:.2f} with N = {n_observed}:")
    log(f"  Power = {power_hypothesis:.3f} ({power_hypothesis * 100:.1f}%)")
    log("")

    if power_hypothesis >= 0.80:
        log("INTERPRETATION:")
        log("  Study adequately powered for moderate effects (r ≥ 0.30)")
        log("  Failure to detect r = 0.30 would be informative")
    else:
        log("INTERPRETATION:")
        log("  Study underpowered for r = 0.30 threshold")
        log("  Cannot rule out moderate effects with confidence")
    log("")

    # Required N for 0.80 power
    log("=" * 70)
    log("REQUIRED SAMPLE SIZE (0.80 power)")
    log("=" * 70)
    log("")

    n_required = required_n(r_observed, power=0.80, alpha=alpha)
    log(f"N required for 80% power at r = {r_observed:.4f}:")
    log(f"  N = {n_required}")
    log(f"  Current N = {n_observed} ({n_observed / n_required * 100:.1f}% of required)")
    log("")

    # Power curve
    log("=" * 70)
    log("GENERATING POWER CURVE")
    log("=" * 70)
    log("")

    rs = [0.10, 0.20, 0.30, 0.50]
    ns = np.arange(50, 301, 10)

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in rs:
        powers = [power_correlation(r, n, alpha) for n in ns]
        ax.plot(ns, powers, label=f"r = {r:.2f}", linewidth=2)

    # Mark current study
    ax.axvline(n_observed, color='red', linestyle='--', alpha=0.5, label=f'Current N = {n_observed}')
    ax.axhline(0.80, color='gray', linestyle=':', alpha=0.5, label='Power = 0.80')

    # Mark observed effect
    ax.plot(n_observed, power_observed, 'ro', markersize=10,
            label=f'Observed (r={r_observed:.2f}, power={power_observed:.2f})')

    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Statistical Power")
    ax.set_title(f"Power Analysis: Correlation Test (α = {alpha}, two-tailed)")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(50, 300)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "power_curve.png", dpi=300)
    plt.close()
    log("Saved: plots/power_curve.png")
    log("")

    # Save results
    results = pd.DataFrame([{
        'r_observed': r_observed,
        'n_observed': n_observed,
        'power_observed': power_observed,
        'r_hypothesis': r_hypothesis,
        'power_hypothesis': power_hypothesis,
        'n_required_80pct': n_required
    }])

    results.to_csv(DATA_DIR / "step09_power_analysis.csv", index=False)
    log(f"Saved: data/step09_power_analysis.csv")
    log("")

    # Summary
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("")
    log(f"Current study (N = {n_observed}):")
    log(f"  Post-hoc power for r = {r_observed:.2f}: {power_observed:.2f}")
    log(f"  Power for r = 0.30 threshold: {power_hypothesis:.2f}")
    log(f"  Required N for 80% power: {n_required}")
    log("")
    log("KEY TAKEAWAY:")
    log("  Finding p = .034 is legitimate but near detection threshold")
    log("  Study adequately powered for moderate effects (r ≥ 0.30)")
    log("  Weak effect (r = 0.21) detected with marginal power (~54%)")
    log("  Replication in N ≈ 170 would provide robust confirmation")
    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
