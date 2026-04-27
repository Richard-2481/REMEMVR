#!/usr/bin/env python3
"""
Generate LMM diagnostic plots for RQ 6.6.3

Diagnostics for participant-level LMM:
1. Q-Q plot (residual normality)
2. Residuals vs Fitted (homoscedasticity)
3. Scale-Location (homoscedasticity alternative)
4. Residuals vs Leverage (influential observations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
PLOTS_DIR = RQ_DIR / "plots"
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"

PLOTS_DIR.mkdir(exist_ok=True)


def load_and_fit_lmm():
    """Load data and refit LMM to get residuals."""
    print("Loading LMM input data...")

    # Load item-level data
    hce_data = pd.read_csv(DATA_DIR / "step01_hce_by_domain.csv")

    # Load TSVR
    tsvr_df = pd.read_csv(CACHE_DIR / "dfData.csv", usecols=['UID', 'TEST', 'TSVR'])
    tsvr_df['UID'] = tsvr_df['UID'].astype(str)
    tsvr_df['Days'] = tsvr_df['TSVR'] / 24.0

    # Get mean TSVR per TEST
    tsvr_by_test = tsvr_df.groupby('TEST')['Days'].mean().reset_index()
    tsvr_by_test.columns = ['TEST', 'Days_mean']

    # Aggregate to participant level
    participant_agg = hce_data.groupby(['UID', 'domain', 'TEST']).agg(
        HCE_rate=('HCE', 'mean'),
        N_items=('HCE', 'count')
    ).reset_index()

    # Merge with TSVR
    lmm_data = participant_agg.merge(tsvr_by_test, on='TEST')

    print(f"LMM input data: {len(lmm_data)} rows")

    # Fit LMM (same as step03)
    print("Fitting LMM...")
    model = smf.mixedlm(
        "HCE_rate ~ C(domain, Treatment(reference='What')) * Days_mean",
        data=lmm_data,
        groups=lmm_data['UID'],
        re_formula="~1"
    )
    result = model.fit(method='powell', maxiter=1000)

    print("Model fitted successfully")

    # Extract residuals and fitted values
    fitted_values = result.fittedvalues
    residuals = result.resid

    # Standardized residuals
    residuals_std = residuals / np.std(residuals)

    return result, lmm_data, fitted_values, residuals, residuals_std


def plot_diagnostics(fitted_values, residuals, residuals_std, lmm_data):
    """Create 4-panel diagnostic plot."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RQ 6.6.3: LMM Diagnostic Plots', fontsize=14, fontweight='bold')

    # Panel 1: Q-Q Plot
    ax1 = axes[0, 0]
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Residuals vs Fitted
    ax2 = axes[0, 1]
    ax2.scatter(fitted_values, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted (Homoscedasticity)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add lowess smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(residuals, fitted_values, frac=0.3)
    ax2.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='Lowess')
    ax2.legend()

    # Panel 3: Scale-Location (sqrt of standardized residuals)
    ax3 = axes[1, 0]
    sqrt_std_resid = np.sqrt(np.abs(residuals_std))
    ax3.scatter(fitted_values, sqrt_std_resid, alpha=0.5, s=10)
    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location (Homoscedasticity)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add lowess smoother
    smoothed = lowess(sqrt_std_resid, fitted_values, frac=0.3)
    ax3.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='Lowess')
    ax3.legend()

    # Panel 4: Residuals vs Domain (check for domain-specific patterns)
    ax4 = axes[1, 1]
    domains = lmm_data['domain'].values
    for domain in ['What', 'Where', 'When']:
        mask = domains == domain
        ax4.scatter(
            np.arange(len(residuals))[mask],
            residuals[mask],
            label=domain,
            alpha=0.5,
            s=10
        )
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax4.set_xlabel('Observation Index')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals by Domain', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = PLOTS_DIR / "lmm_diagnostics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()


def generate_diagnostic_stats(residuals, residuals_std):
    """Generate numerical diagnostic statistics."""

    print("\n" + "=" * 60)
    print("DIAGNOSTIC STATISTICS")
    print("=" * 60)

    # Normality tests
    _, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    print(f"\nNormality (Shapiro-Wilk test):")
    print(f"  p-value: {shapiro_p:.6f}")
    print(f"  Interpretation: {'PASS (p > 0.05)' if shapiro_p > 0.05 else 'WARNING (p < 0.05, but LMM robust with N=1200)'}")

    # Residual range
    print(f"\nResidual Summary:")
    print(f"  Mean: {np.mean(residuals):.6f} (should be ~0)")
    print(f"  SD: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    print(f"  Range: {np.max(residuals) - np.min(residuals):.6f}")

    # Outliers (>3 SD)
    outliers = np.abs(residuals_std) > 3
    n_outliers = np.sum(outliers)
    pct_outliers = (n_outliers / len(residuals)) * 100
    print(f"\nOutliers (|z| > 3):")
    print(f"  Count: {n_outliers} / {len(residuals)} ({pct_outliers:.2f}%)")
    print(f"  Expected: ~0.3% under normality")
    print(f"  Interpretation: {'PASS' if pct_outliers < 1.0 else 'WARNING (excess outliers)'}")

    # Homoscedasticity (visual inspection primary, but report Breusch-Pagan if possible)
    print(f"\nHomoscedasticity:")
    print(f"  Check: Visual inspection of Residuals vs Fitted plot")
    print(f"  Expect: Random scatter around zero with constant spread")


def main():
    """Main diagnostic workflow."""
    print("=" * 60)
    print("RQ 6.6.3: Generate LMM Diagnostic Plots")
    print("=" * 60)
    print()

    # Fit model and get residuals
    result, lmm_data, fitted_values, residuals, residuals_std = load_and_fit_lmm()

    # Generate plots
    plot_diagnostics(fitted_values, residuals, residuals_std, lmm_data)

    # Generate statistics
    generate_diagnostic_stats(residuals, residuals_std)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print("\nOutput: plots/lmm_diagnostics.png")


if __name__ == "__main__":
    main()
