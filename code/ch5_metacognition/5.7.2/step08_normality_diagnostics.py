#!/usr/bin/env python3
"""
RQ 6.7.2: Normality Diagnostics for Partial Correlation
========================================================

Validates assumption of bivariate normality for partial correlation analysis.
Per improvement_taxonomy.md Section 5: Assumption validation mandatory.

Tests:
1. Shapiro-Wilk test on residuals of SD_confidence (after controlling mean_accuracy)
2. Shapiro-Wilk test on residuals of SD_accuracy (after controlling mean_accuracy)
3. Q-Q plots for visual inspection
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
PLOTS_DIR = RQ_DIR / "plots" / "diagnostics"
LOG_FILE = LOG_DIR / "step08_normality_diagnostics.log"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """Log to file and stdout."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a') as f:
        f.write(f"{msg}\n")
    print(msg)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residualize y on x (remove linear relationship)."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = intercept + slope * x
    return y - y_pred


def main():
    log("=" * 70)
    log("RQ 6.7.2: Normality Diagnostics for Partial Correlation")
    log("=" * 70)
    log("")

    # Load person-level data
    person_level = pd.read_csv(DATA_DIR / "step03_person_level.csv")
    log(f"Loaded step03_person_level.csv: {len(person_level)} participants")
    log("")

    # Extract variables
    sd_conf = person_level['avg_SD_confidence'].values
    sd_acc = person_level['avg_SD_accuracy'].values
    mean_acc = person_level['avg_mean_accuracy'].values

    log("Variables:")
    log(f"  SD_confidence: N={len(sd_conf)}, Mean={sd_conf.mean():.4f}, SD={sd_conf.std():.4f}")
    log(f"  SD_accuracy: N={len(sd_acc)}, Mean={sd_acc.mean():.4f}, SD={sd_acc.std():.4f}")
    log(f"  mean_accuracy: N={len(mean_acc)}, Mean={mean_acc.mean():.4f}, SD={mean_acc.std():.4f}")
    log("")

    # Residualize both variables on mean_accuracy
    log("=" * 70)
    log("RESIDUALIZATION (controlling for mean_accuracy)")
    log("=" * 70)
    log("")

    sd_conf_resid = residualize(sd_conf, mean_acc)
    sd_acc_resid = residualize(sd_acc, mean_acc)

    log(f"SD_confidence residuals:")
    log(f"  Mean: {sd_conf_resid.mean():.6f} (should be ~0)")
    log(f"  SD: {sd_conf_resid.std():.4f}")
    log("")
    log(f"SD_accuracy residuals:")
    log(f"  Mean: {sd_acc_resid.mean():.6f} (should be ~0)")
    log(f"  SD: {sd_acc_resid.std():.4f}")
    log("")

    # Shapiro-Wilk tests
    log("=" * 70)
    log("SHAPIRO-WILK NORMALITY TESTS")
    log("=" * 70)
    log("")

    w_conf, p_conf = stats.shapiro(sd_conf_resid)
    w_acc, p_acc = stats.shapiro(sd_acc_resid)

    log(f"SD_confidence residuals:")
    log(f"  Shapiro-Wilk W = {w_conf:.4f}")
    log(f"  p-value = {p_conf:.4f}")
    log(f"  Result: {'Normal (p > 0.05)' if p_conf > 0.05 else 'Non-normal (p ≤ 0.05)'}")
    log("")

    log(f"SD_accuracy residuals:")
    log(f"  Shapiro-Wilk W = {w_acc:.4f}")
    log(f"  p-value = {p_acc:.4f}")
    log(f"  Result: {'Normal (p > 0.05)' if p_acc > 0.05 else 'Non-normal (p ≤ 0.05)'}")
    log("")

    # Q-Q plots
    log("=" * 70)
    log("GENERATING Q-Q PLOTS")
    log("=" * 70)
    log("")

    # Q-Q plot for SD_confidence residuals
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(sd_conf_resid, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: SD_confidence Residuals\n(after controlling mean_accuracy)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qq_plot_confidence_residuals.png", dpi=300)
    plt.close()
    log("Saved: plots/diagnostics/qq_plot_confidence_residuals.png")

    # Q-Q plot for SD_accuracy residuals
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(sd_acc_resid, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: SD_accuracy Residuals\n(after controlling mean_accuracy)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qq_plot_accuracy_residuals.png", dpi=300)
    plt.close()
    log("Saved: plots/diagnostics/qq_plot_accuracy_residuals.png")
    log("")

    # Overall assessment
    log("=" * 70)
    log("OVERALL ASSESSMENT")
    log("=" * 70)
    log("")

    both_normal = (p_conf > 0.05) and (p_acc > 0.05)

    if both_normal:
        log("STATUS: ✓ ASSUMPTIONS MET")
        log("  Both residual distributions are normally distributed (p > 0.05)")
        log("")
        log("INTERPRETATION:")
        log("  Pearson partial correlation is statistically valid")
        log("  Parametric inference appropriate (no need for rank-based alternative)")
        log("  Observed r_partial = 0.214, p = .034 can be trusted")
    else:
        log("STATUS: ⚠ ASSUMPTIONS VIOLATED")
        if p_conf <= 0.05:
            log(f"  SD_confidence residuals are non-normal (p = {p_conf:.4f})")
        if p_acc <= 0.05:
            log(f"  SD_accuracy residuals are non-normal (p = {p_acc:.4f})")
        log("")
        log("RECOMMENDATION:")
        log("  Use Spearman rank-based partial correlation as robustness check")
        log("  Report both parametric and non-parametric results")
        log("  If conclusions agree, parametric result defensible despite violation")

    # Save results
    results = pd.DataFrame([{
        'variable': 'SD_confidence_residuals',
        'shapiro_W': w_conf,
        'shapiro_p': p_conf,
        'normal': p_conf > 0.05
    }, {
        'variable': 'SD_accuracy_residuals',
        'shapiro_W': w_acc,
        'shapiro_p': p_acc,
        'normal': p_acc > 0.05
    }])

    results.to_csv(DATA_DIR / "step08_normality_diagnostics.csv", index=False)
    log("")
    log(f"Saved: data/step08_normality_diagnostics.csv")
    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
