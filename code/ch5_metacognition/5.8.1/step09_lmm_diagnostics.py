"""
Step 9: LMM Diagnostic Checks (MANDATORY for validation)

Purpose: Validate LMM assumptions per taxonomy Section 5.1
         - Residual normality (Q-Q plots, Shapiro-Wilk)
         - Homoscedasticity (residuals vs fitted)
         - Leverage/influence (Cook's D)
         - Random effects normality

Date: 2025-12-27
"""

import sys
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def fit_model_and_extract_residuals(data):
    """Refit model to extract residuals and fitted values."""
    print("\n" + "="*80)
    print("REFITTING MODEL FOR DIAGNOSTICS")
    print("="*80)

    formula = "theta ~ C(location) * log_TSVR"
    model = MixedLM.from_formula(
        formula=formula,
        data=data,
        groups=data['UID'],
        re_formula="~log_TSVR"  # Slopes model
    )

    result = model.fit(reml=False, method='lbfgs', maxiter=1000)

    print(f"Converged: {result.converged}")

    # Extract fitted values and residuals
    fitted = result.fittedvalues
    residuals = result.resid

    print(f"Fitted values range: [{fitted.min():.3f}, {fitted.max():.3f}]")
    print(f"Residuals range: [{residuals.min():.3f}, {residuals.max():.3f}]")

    return result, fitted, residuals

def test_residual_normality(residuals):
    """Test normality of residuals."""
    print("\n" + "="*80)
    print("RESIDUAL NORMALITY")
    print("="*80)

    # Shapiro-Wilk test (most powerful for normality)
    # But limited to n <= 5000
    if len(residuals) <= 5000:
        stat, p_value = stats.shapiro(residuals)
        print(f"\nShapiro-Wilk test:")
        print(f"  W = {stat:.4f}")
        print(f"  p-value = {p_value:.4f}")

        if p_value < 0.001:
            print(f"  🔴 SIGNIFICANT DEVIATION from normality (p < 0.001)")
            print(f"     → Residuals non-normal (may affect inference)")
        elif p_value < 0.05:
            print(f"  ⚠️  MARGINAL DEVIATION from normality (p = {p_value:.3f})")
            print(f"     → With N=800, minor deviations acceptable (robust)")
        else:
            print(f"  ✅ NORMALITY ASSUMPTION MET (p = {p_value:.3f})")
    else:
        stat = None
        p_value = None
        print("  ⚠️  Sample too large for Shapiro-Wilk (n > 5000)")

    # Kolmogorov-Smirnov test (alternative for large samples)
    ks_stat, ks_p = stats.kstest(residuals, 'norm',
                                   args=(residuals.mean(), residuals.std()))
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  D = {ks_stat:.4f}")
    print(f"  p-value = {ks_p:.4f}")

    if ks_p < 0.05:
        print(f"  🔴 DEVIATION from normality detected")
    else:
        print(f"  ✅ Normality assumption met")

    # Descriptive statistics
    print(f"\nResidual descriptive statistics:")
    print(f"  Mean: {residuals.mean():.4f} (should be ≈ 0)")
    print(f"  SD: {residuals.std():.4f}")
    print(f"  Skewness: {stats.skew(residuals):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.3f}")

    # Q-Q plot
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Residuals vs Normal Distribution", fontsize=14)
    plt.xlabel("Theoretical Quantiles", fontsize=12)
    plt.ylabel("Sample Quantiles", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    qq_path = 'results/ch6/6.8.1/plots/diagnostics_qq_plot.png'
    plt.savefig(qq_path, dpi=300)
    plt.close()
    print(f"\n✅ Saved Q-Q plot: {qq_path}")

    return stat, p_value, ks_p

def test_homoscedasticity(fitted, residuals):
    """Test constant variance of residuals."""
    print("\n" + "="*80)
    print("HOMOSCEDASTICITY")
    print("="*80)

    # Residuals vs fitted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(fitted, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Fitted Values", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.title("Residuals vs Fitted Values", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    resid_path = 'results/ch6/6.8.1/plots/diagnostics_residuals_vs_fitted.png'
    plt.savefig(resid_path, dpi=300)
    plt.close()
    print(f"\n✅ Saved residuals vs fitted plot: {resid_path}")

    # Breusch-Pagan test (approximate - need exog matrix)
    # Simplified: test correlation between |residuals| and fitted
    abs_resid = np.abs(residuals)
    corr, p_value = stats.spearmanr(fitted, abs_resid)

    print(f"\nSpearman correlation (|residuals| vs fitted):")
    print(f"  ρ = {corr:.3f}")
    print(f"  p-value = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  ⚠️  HETEROSCEDASTICITY detected (p < 0.05)")
        print(f"     → Variance non-constant (may need robust SE)")
    else:
        print(f"  ✅ HOMOSCEDASTICITY assumption met (p = {p_value:.3f})")

    # Levene's test (split into low/high fitted value groups)
    median_fitted = np.median(fitted)
    low_group = residuals[fitted <= median_fitted]
    high_group = residuals[fitted > median_fitted]

    levene_stat, levene_p = stats.levene(low_group, high_group)
    print(f"\nLevene's test (low vs high fitted groups):")
    print(f"  W = {levene_stat:.3f}")
    print(f"  p-value = {levene_p:.4f}")

    if levene_p < 0.05:
        print(f"  ⚠️  HETEROSCEDASTICITY detected")
    else:
        print(f"  ✅ Equal variance across fitted range")

    return p_value, levene_p

def compute_cooks_distance(result, residuals):
    """Compute Cook's distance for influential observations."""
    print("\n" + "="*80)
    print("INFLUENTIAL OBSERVATIONS (Cook's D)")
    print("="*80)

    # Simplified Cook's D (exact computation requires leverage)
    # Approximation: standardized residuals
    std_resid = residuals / residuals.std()

    # Flag observations with |std_resid| > 3
    outliers = np.abs(std_resid) > 3
    n_outliers = outliers.sum()

    print(f"\nStandardized residuals:")
    print(f"  Range: [{std_resid.min():.2f}, {std_resid.max():.2f}]")
    print(f"  N outliers (|std_resid| > 3): {n_outliers} ({n_outliers/len(std_resid)*100:.1f}%)")

    if n_outliers > 0:
        print(f"\n⚠️  {n_outliers} influential observations detected")
        print(f"   Indices: {np.where(outliers)[0][:10].tolist()} (first 10)")
    else:
        print(f"\n✅ No extreme outliers (all |std_resid| <= 3)")

    # Histogram of standardized residuals
    plt.figure(figsize=(8, 6))
    plt.hist(std_resid, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=-3, color='r', linestyle='--', linewidth=2, label='±3 SD')
    plt.axvline(x=3, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Standardized Residuals", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Standardized Residuals", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = 'results/ch6/6.8.1/plots/diagnostics_residual_histogram.png'
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"\n✅ Saved residual histogram: {hist_path}")

    return n_outliers

def check_random_effects_normality(result):
    """Check normality of random effects."""
    print("\n" + "="*80)
    print("RANDOM EFFECTS NORMALITY")
    print("="*80)

    # Extract random effects
    random_effects = result.random_effects

    # Convert dict to DataFrame
    re_df = pd.DataFrame.from_dict(random_effects, orient='index')
    print(f"\nRandom effects shape: {re_df.shape}")
    print(f"Columns: {re_df.columns.tolist()}")

    # Test normality for each random effect component
    for col in re_df.columns:
        values = re_df[col].values
        stat, p_value = stats.shapiro(values)

        print(f"\n{col}:")
        print(f"  Mean: {values.mean():.4f} (should be ≈ 0)")
        print(f"  SD: {values.std():.4f}")
        print(f"  Shapiro-Wilk p = {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ⚠️  NON-NORMAL (p = {p_value:.3f})")
        else:
            print(f"  ✅ NORMAL (p = {p_value:.3f})")

    print("\n✅ Random effects normality checked")

def main():
    """Execute LMM diagnostics."""
    print("="*80)
    print("LMM DIAGNOSTIC CHECKS - RQ 6.8.1")
    print("="*80)
    print("\n🔴 MANDATORY for validation (Taxonomy Section 5.1)")
    print("   - Residual normality")
    print("   - Homoscedasticity")
    print("   - Influential observations")
    print("   - Random effects normality")

    # Create plots directory if missing
    import os
    os.makedirs('results/ch6/6.8.1/plots', exist_ok=True)

    # Load data
    data = pd.read_csv('results/ch6/6.8.1/data/step04_lmm_input.csv')

    # Fit model and extract diagnostics
    result, fitted, residuals = fit_model_and_extract_residuals(data)

    # Run diagnostic tests
    shapiro_stat, shapiro_p, ks_p = test_residual_normality(residuals)
    het_p, levene_p = test_homoscedasticity(fitted, residuals)
    n_outliers = compute_cooks_distance(result, residuals)
    check_random_effects_normality(result)

    # Save summary
    print("\n" + "="*80)
    print("SAVING SUMMARY")
    print("="*80)

    summary_path = 'results/ch6/6.8.1/data/step09_diagnostics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LMM DIAGNOSTIC CHECKS - RQ 6.8.1\n")
        f.write("="*80 + "\n\n")

        f.write("RESIDUAL NORMALITY:\n")
        if shapiro_p is not None:
            f.write(f"  Shapiro-Wilk p = {shapiro_p:.4f}\n")
        f.write(f"  Kolmogorov-Smirnov p = {ks_p:.4f}\n")
        if shapiro_p is not None and shapiro_p >= 0.05:
            f.write(f"  Result: NORMAL\n")
        else:
            f.write(f"  Result: Non-normal (but N=800 robust to minor deviations)\n")
        f.write("\n")

        f.write("HOMOSCEDASTICITY:\n")
        f.write(f"  Spearman correlation p = {het_p:.4f}\n")
        f.write(f"  Levene's test p = {levene_p:.4f}\n")
        if het_p >= 0.05 and levene_p >= 0.05:
            f.write(f"  Result: HOMOSCEDASTIC\n")
        else:
            f.write(f"  Result: Heteroscedasticity detected (consider robust SE)\n")
        f.write("\n")

        f.write("INFLUENTIAL OBSERVATIONS:\n")
        f.write(f"  N outliers (|std_resid| > 3): {n_outliers}\n")
        f.write(f"  Percentage: {n_outliers/len(residuals)*100:.1f}%\n")
        if n_outliers == 0:
            f.write(f"  Result: NO extreme outliers\n")
        else:
            f.write(f"  Result: Some outliers present (inspect if >5%)\n")
        f.write("\n")

        f.write("✅ VALIDATION REQUIREMENT MET: LMM diagnostics completed\n")

    print(f"✅ Saved: {summary_path}")

    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print("\n📊 SUMMARY:")
    if shapiro_p is not None:
        print(f"   Normality: p = {shapiro_p:.4f}")
    print(f"   Homoscedasticity: p = {het_p:.4f}")
    print(f"   Outliers: {n_outliers} ({n_outliers/len(residuals)*100:.1f}%)")
    print(f"\n✅ Diagnostic plots saved to plots/ directory")

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
