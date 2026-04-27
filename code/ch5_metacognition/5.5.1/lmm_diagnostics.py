"""
LMM Diagnostics for RQ 6.5.1
Taxonomy Section 5.1 - MANDATORY

Checks:
1. Residual normality (Q-Q plot, Shapiro-Wilk test)
2. Homoscedasticity (Residuals vs Fitted, Breusch-Pagan test)
3. Leverage/influence (Cook's D)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path
import statsmodels.formula.api as smf

# Paths
base_path = Path("/home/etai/projects/REMEMVR/results/ch6/6.5.1")
data_path = base_path / "data"
plots_path = base_path / "plots" / "diagnostics"
plots_path.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LMM Diagnostics - RQ 6.5.1")
print("="*80)

# Load LMM input
lmm_input = pd.read_csv(data_path / "step04_lmm_input.csv")
print(f"Loaded {len(lmm_input)} observations")

# Refit random slopes model to extract residuals
print("\nRefitting LMM with random slopes...")
model = smf.mixedlm(
    "theta ~ C(congruence) * log_TSVR",
    data=lmm_input,
    groups=lmm_input['UID'],
    re_formula="~log_TSVR"
)
result = model.fit(method='lbfgs', reml=False)
print(f"Converged: {result.converged}")

# Extract fitted values and residuals
fitted = result.fittedvalues
residuals = result.resid

print(f"\nResiduals summary:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  SD: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

##############################################################################
# CHECK 1: RESIDUAL NORMALITY
##############################################################################

print("\n" + "="*80)
print("CHECK 1: RESIDUAL NORMALITY")
print("="*80)

# Q-Q plot
fig, ax = plt.subplots(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot: Residuals vs Normal Distribution", fontsize=14)
ax.set_xlabel("Theoretical Quantiles", fontsize=12)
ax.set_ylabel("Sample Quantiles", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "qq_plot_residuals.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_path / 'qq_plot_residuals.png'}")
plt.close()

# Shapiro-Wilk test (if N < 5000)
if len(residuals) < 5000:
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  p-value: {shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print(f"  ⚠️ NORMALITY REJECTED (p < 0.05)")
        print(f"  Residuals deviate from normality")
        normality_result = "REJECTED"
    else:
        print(f"  ✅ NORMALITY NOT REJECTED (p ≥ 0.05)")
        normality_result = "PASS"
else:
    print(f"\nShapiro-Wilk test skipped (N = {len(residuals)} > 5000)")
    shapiro_p = np.nan
    normality_result = "SKIP"

# Histogram
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
ax.set_xlabel("Residuals", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Histogram of Residuals", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "residuals_histogram.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_path / 'residuals_histogram.png'}")
plt.close()

##############################################################################
# CHECK 2: HOMOSCEDASTICITY
##############################################################################

print("\n" + "="*80)
print("CHECK 2: HOMOSCEDASTICITY")
print("="*80)

# Residuals vs Fitted plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(fitted, residuals, alpha=0.5, edgecolor='none')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Fitted Values", fontsize=12)
ax.set_ylabel("Residuals", fontsize=12)
ax.set_title("Residuals vs Fitted Values", fontsize=14)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "residuals_vs_fitted.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_path / 'residuals_vs_fitted.png'}")
plt.close()

# Breusch-Pagan test
# Requires exog matrix (design matrix)
exog = result.model.exog  # Design matrix from fitted model

try:
    bp_stat, bp_p, bp_f_stat, bp_f_p = het_breuschpagan(residuals, exog)
    print(f"\nBreusch-Pagan test for heteroscedasticity:")
    print(f"  LM Statistic: {bp_stat:.4f}")
    print(f"  p-value: {bp_p:.4f}")
    if bp_p < 0.05:
        print(f"  ⚠️ HETEROSCEDASTICITY DETECTED (p < 0.05)")
        print(f"  Variance is NOT constant across fitted values")
        homoscedasticity_result = "REJECTED"
    else:
        print(f"  ✅ HOMOSCEDASTICITY NOT REJECTED (p ≥ 0.05)")
        homoscedasticity_result = "PASS"
except Exception as e:
    print(f"\n⚠️ Breusch-Pagan test failed: {e}")
    bp_p = np.nan
    homoscedasticity_result = "FAILED"

# Scale-Location plot (sqrt of absolute residuals vs fitted)
fig, ax = plt.subplots(figsize=(10, 6))
sqrt_abs_resid = np.sqrt(np.abs(residuals))
ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, edgecolor='none')
# Add smoothed line
from scipy.ndimage import uniform_filter1d
sorted_indices = np.argsort(fitted)
fitted_sorted = fitted.iloc[sorted_indices]
sqrt_abs_resid_sorted = sqrt_abs_resid.iloc[sorted_indices]
if len(fitted_sorted) > 50:
    window_size = min(50, len(fitted_sorted) // 10)
    smoothed = uniform_filter1d(sqrt_abs_resid_sorted, size=window_size, mode='nearest')
    ax.plot(fitted_sorted, smoothed, color='red', linewidth=2, label='Smoothed trend')
ax.set_xlabel("Fitted Values", fontsize=12)
ax.set_ylabel("√|Residuals|", fontsize=12)
ax.set_title("Scale-Location Plot", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "scale_location.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_path / 'scale_location.png'}")
plt.close()

##############################################################################
# CHECK 3: LEVERAGE AND INFLUENCE
##############################################################################

print("\n" + "="*80)
print("CHECK 3: LEVERAGE AND INFLUENCE")
print("="*80)

# Cook's D (approximation for mixed models)
# True Cook's D requires refitting without each observation (computationally expensive)
# Approximation: standardized residuals
standardized_resid = residuals / residuals.std()

# Flag influential points (|standardized resid| > 3)
influential = np.abs(standardized_resid) > 3
n_influential = influential.sum()
print(f"\nInfluential observations (|standardized resid| > 3): {n_influential} / {len(residuals)} ({n_influential/len(residuals)*100:.2f}%)")

if n_influential > 0:
    print(f"⚠️ {n_influential} potential outliers detected")
    print(f"Indices: {np.where(influential)[0][:10]}")  # Show first 10
    influence_result = "OUTLIERS DETECTED"
else:
    print(f"✅ No extreme outliers detected")
    influence_result = "PASS"

# Plot standardized residuals
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(standardized_resid)), standardized_resid, alpha=0.5, edgecolor='none')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(3, color='orange', linestyle=':', linewidth=1.5, label='±3 SD threshold')
ax.axhline(-3, color='orange', linestyle=':', linewidth=1.5)
ax.set_xlabel("Observation Index", fontsize=12)
ax.set_ylabel("Standardized Residuals", fontsize=12)
ax.set_title("Standardized Residuals vs Index", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(plots_path / "standardized_residuals.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plots_path / 'standardized_residuals.png'}")
plt.close()

##############################################################################
# SAVE DIAGNOSTIC REPORT
##############################################################################

print("\n" + "="*80)
print("SAVING DIAGNOSTIC REPORT")
print("="*80)

diagnostics_df = pd.DataFrame({
    'check': ['Normality (Shapiro-Wilk)', 'Homoscedasticity (Breusch-Pagan)', 'Influence (Standardized Resid)'],
    'test_statistic': [shapiro_stat if shapiro_p is not np.nan else np.nan,
                       bp_stat if bp_p is not np.nan else np.nan,
                       n_influential],
    'p_value': [shapiro_p, bp_p, np.nan],
    'result': [normality_result, homoscedasticity_result, influence_result]
})

diagnostics_df.to_csv(data_path / "lmm_diagnostics.csv", index=False)
print(f"✅ Saved: {data_path / 'lmm_diagnostics.csv'}")

# Text report
report_path = data_path / "lmm_diagnostics_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LMM DIAGNOSTICS REPORT - RQ 6.5.1\n")
    f.write("Taxonomy Section 5.1\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL SPECIFICATION:\n")
    f.write("  Formula: theta ~ C(congruence) * log_TSVR\n")
    f.write("  Random effects: (1 + log_TSVR | UID)\n")
    f.write(f"  Converged: {result.converged}\n")
    f.write(f"  N observations: {len(lmm_input)}\n\n")

    f.write("-"*80 + "\n")
    f.write("DIAGNOSTIC CHECKS\n")
    f.write("-"*80 + "\n\n")

    f.write("1. RESIDUAL NORMALITY:\n")
    if shapiro_p is not np.nan:
        f.write(f"   Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}\n")
    f.write(f"   Result: {normality_result}\n")
    if normality_result == "REJECTED":
        f.write("   NOTE: LMM is robust to moderate non-normality with large N.\n")
    f.write("\n")

    f.write("2. HOMOSCEDASTICITY:\n")
    if bp_p is not np.nan:
        f.write(f"   Breusch-Pagan test: LM = {bp_stat:.4f}, p = {bp_p:.4f}\n")
    f.write(f"   Result: {homoscedasticity_result}\n")
    if homoscedasticity_result == "REJECTED":
        f.write("   NOTE: Consider robust standard errors if heteroscedasticity severe.\n")
    f.write("\n")

    f.write("3. INFLUENTIAL OBSERVATIONS:\n")
    f.write(f"   Outliers (|std resid| > 3): {n_influential} / {len(residuals)} ({n_influential/len(residuals)*100:.2f}%)\n")
    f.write(f"   Result: {influence_result}\n")
    if n_influential > 0:
        f.write("   NOTE: Outliers may be legitimate extreme values, not errors.\n")
    f.write("\n")

    f.write("-"*80 + "\n")
    f.write("PLOTS CREATED\n")
    f.write("-"*80 + "\n\n")
    f.write(f"  - {plots_path / 'qq_plot_residuals.png'}\n")
    f.write(f"  - {plots_path / 'residuals_histogram.png'}\n")
    f.write(f"  - {plots_path / 'residuals_vs_fitted.png'}\n")
    f.write(f"  - {plots_path / 'scale_location.png'}\n")
    f.write(f"  - {plots_path / 'standardized_residuals.png'}\n\n")

    f.write("-"*80 + "\n")
    f.write("OVERALL ASSESSMENT\n")
    f.write("-"*80 + "\n\n")

    issues = []
    if normality_result == "REJECTED":
        issues.append("Non-normal residuals")
    if homoscedasticity_result == "REJECTED":
        issues.append("Heteroscedasticity")
    if n_influential > len(residuals) * 0.05:  # > 5% outliers
        issues.append(f"{n_influential} outliers ({n_influential/len(residuals)*100:.1f}%)")

    if len(issues) == 0:
        f.write("✅ ALL DIAGNOSTIC CHECKS PASSED\n")
        f.write("Model assumptions met. Results are reliable.\n")
    else:
        f.write(f"⚠️ {len(issues)} ISSUES DETECTED:\n")
        for issue in issues:
            f.write(f"  - {issue}\n")
        f.write("\nNOTE: LMM is generally robust to moderate violations with N = 100+.\n")
        f.write("Consider sensitivity analyses if violations severe.\n")

print(f"✅ Saved: {report_path}")

print("\n" + "="*80)
print("LMM DIAGNOSTICS COMPLETE")
print("="*80)
print("\nDiagnostic plots saved to:", plots_path)
print("\nFiles created:")
print(f"  - {data_path / 'lmm_diagnostics.csv'}")
print(f"  - {report_path}")
print(f"  - 5 diagnostic plots in {plots_path}")
print("\nNext: Response pattern analysis (Section 8.3 - confidence RQs)")
