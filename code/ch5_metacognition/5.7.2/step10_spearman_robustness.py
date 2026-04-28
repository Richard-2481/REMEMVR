#!/usr/bin/env python3
"""
RQ 6.7.2: Spearman Rank-Based Partial Correlation
==================================================

REASON: Normality diagnostics (step08) detected non-normal residuals.

Robustness check using Spearman rank correlation (non-parametric alternative).

If Spearman partial correlation confirms Pearson result, parametric finding
is defensible despite normality violation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step10_spearman_robustness.log"


def log(msg: str):
    """Log to file and stdout."""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a') as f:
        f.write(f"{msg}\n")
    print(msg)


def spearman_partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Compute Spearman partial correlation rho(x,y|z) using rank residuals.

    Method:
    1. Rank-transform all variables
    2. Residualize ranked x on ranked z
    3. Residualize ranked y on ranked z
    4. Compute Spearman correlation of residuals

    Returns: (rho_partial, p_value, df)
    """
    n = len(x)

    # Rank-transform
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    z_rank = stats.rankdata(z)

    # Residualize x_rank on z_rank
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z_rank, x_rank)
    x_resid = x_rank - (intercept_xz + slope_xz * z_rank)

    # Residualize y_rank on z_rank
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z_rank, y_rank)
    y_resid = y_rank - (intercept_yz + slope_yz * z_rank)

    # Spearman correlation of residuals
    rho_partial, p_value = spearmanr(x_resid, y_resid)

    # Degrees of freedom
    df = n - 3  # Controlling for 1 variable

    return rho_partial, p_value, df


def main():
    log("=" * 70)
    log("RQ 6.7.2: Spearman Rank-Based Partial Correlation")
    log("=" * 70)
    log("")

    # Load person-level data
    person_level = pd.read_csv(DATA_DIR / "step03_person_level.csv")
    log(f"Loaded step03_person_level.csv: {len(person_level)} participants")
    log("")

    # Extract variables
    x = person_level['avg_SD_confidence'].values
    y = person_level['avg_SD_accuracy'].values
    z = person_level['avg_mean_accuracy'].values

    # Compute zero-order Spearman correlations
    log("=" * 70)
    log("ZERO-ORDER SPEARMAN CORRELATIONS")
    log("=" * 70)
    log("")

    rho_xy, p_xy = spearmanr(x, y)
    rho_xz, p_xz = spearmanr(x, z)
    rho_yz, p_yz = spearmanr(y, z)

    log(f"rho(SD_conf, SD_acc) = {rho_xy:.4f}, p = {p_xy:.4f}")
    log(f"rho(SD_conf, mean_acc) = {rho_xz:.4f}, p = {p_xz:.4f}")
    log(f"rho(SD_acc, mean_acc) = {rho_yz:.4f}, p = {p_yz:.4f}")
    log("")

    # Compute Spearman partial correlation
    log("=" * 70)
    log("SPEARMAN PARTIAL CORRELATION")
    log("=" * 70)
    log("")

    rho_partial, p_partial, df = spearman_partial_correlation(x, y, z)

    log(f"rho(SD_conf, SD_acc | mean_acc) :")
    log(f"  Partial rho = {rho_partial:.4f}")
    log(f"  p-value = {p_partial:.4f}")
    log(f"  df = {df}")
    log("")

    # Load Pearson results for comparison
    corr = pd.read_csv(DATA_DIR / "step03_correlation.csv")
    r_pearson = corr['r_partial'].values[0]
    p_pearson = corr['p_partial'].values[0]

    log("=" * 70)
    log("COMPARISON: PEARSON vs SPEARMAN")
    log("=" * 70)
    log("")

    log("Parametric (Pearson):")
    log(f"  Partial r = {r_pearson:.4f}")
    log(f"  p-value = {p_pearson:.4f}")
    log("")

    log("Non-parametric (Spearman):")
    log(f"  Partial rho = {rho_partial:.4f}")
    log(f"  p-value = {p_partial:.4f}")
    log("")

    # Agreement analysis
    log("=" * 70)
    log("ROBUSTNESS ASSESSMENT")
    log("=" * 70)
    log("")

    sign_agreement = np.sign(r_pearson) == np.sign(rho_partial)
    sig_pearson = p_pearson < 0.05
    sig_spearman = p_partial < 0.05
    both_sig = sig_pearson and sig_spearman

    log(f"Sign agreement: {'Yes' if sign_agreement else 'No'}")
    log(f"Pearson significant: {'Yes' if sig_pearson else 'No'} (p = {p_pearson:.4f})")
    log(f"Spearman significant: {'Yes' if sig_spearman else 'No'} (p = {p_partial:.4f})")
    log("")

    if sign_agreement and both_sig:
        log("CONCLUSION: ROBUST")
        log("  Both methods agree on direction AND significance")
        log("  Parametric result defensible despite normality violation")
        log("  Relationship present regardless of distributional assumptions")
        status = "robust"
    elif sign_agreement and not both_sig:
        log("CONCLUSION: MARGINAL")
        log("  Methods agree on direction but not both significant")
        log("  Normality violation may affect p-value reliability")
        log("  Report both results and interpret cautiously")
        status = "marginal"
    else:
        log("CONCLUSION: INCONSISTENT")
        log("  Methods disagree on direction or significance")
        log("  Normality violation undermines parametric result")
        log("  Use Spearman result as primary finding")
        status = "inconsistent"

    log("")

    # Save results
    results = pd.DataFrame([{
        'method': 'Pearson',
        'partial_correlation': r_pearson,
        'p_value': p_pearson,
        'significant': sig_pearson
    }, {
        'method': 'Spearman',
        'partial_correlation': rho_partial,
        'p_value': p_partial,
        'significant': sig_spearman
    }])

    results.to_csv(DATA_DIR / "step10_spearman_robustness.csv", index=False)
    log(f"Saved: data/step10_spearman_robustness.csv")
    log("")

    # Summary
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("")
    log(f"Robustness status: {status.upper()}")
    log("")
    log("RECOMMENDATION:")
    if status == "robust":
        log("  Report both Pearson and Spearman results")
        log("  Emphasize agreement despite normality violation")
        log("  Primary conclusion stands (partial r/rho ≈ 0.21, p ≈ .03)")
    elif status == "marginal":
        log("  Report both results with caution")
        log("  Note that normality violation creates uncertainty")
        log("  Recommend replication before strong claims")
    else:
        log("  Use Spearman as primary result")
        log("  Report Pearson as supplementary")
        log("  Acknowledge normality violation as limitation")

    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
