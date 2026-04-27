"""
LMM Diagnostic Checks for RQ 6.2.3
Section 5 Requirement from improvement_taxonomy.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load LMM input data
data = pd.read_csv('data/step02_gamma_lmm_input.csv')

# Refit model to extract residuals
model = smf.mixedlm(
    "gamma ~ TSVR_days",
    data=data,
    groups=data['UID'],
    re_formula="~TSVR_days"
)
result = model.fit(reml=True, method='powell')

# Extract fitted values and residuals
fitted = result.fittedvalues
residuals = result.resid

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title('Q-Q Plot (Residual Normality)', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# 2. Residuals vs Fitted
axes[0, 1].scatter(fitted, residuals, alpha=0.5, s=30)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted Values', fontsize=10)
axes[0, 1].set_ylabel('Residuals', fontsize=10)
axes[0, 1].set_title('Residuals vs Fitted (Homoscedasticity)', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# 3. Scale-Location (sqrt absolute residuals vs fitted)
residuals_abs_sqrt = np.sqrt(np.abs(residuals))
axes[1, 0].scatter(fitted, residuals_abs_sqrt, alpha=0.5, s=30)
axes[1, 0].set_xlabel('Fitted Values', fontsize=10)
axes[1, 0].set_ylabel('√|Residuals|', fontsize=10)
axes[1, 0].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 4. Histogram of residuals
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Residuals', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/lmm_diagnostics.png', dpi=300, bbox_inches='tight')
print("✅ Diagnostic plots saved to: plots/lmm_diagnostics.png")

# Formal tests
print("\n" + "=" * 60)
print("LMM DIAGNOSTIC TESTS")
print("=" * 60)

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\n1. Shapiro-Wilk Test (Residual Normality):")
print(f"   Statistic: {shapiro_stat:.4f}")
print(f"   p-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("   ✅ PASS: Residuals normally distributed (p > 0.05)")
else:
    print("   ⚠️ FLAG: Residuals may violate normality (p < 0.05)")
    print("   NOTE: LMM robust to moderate non-normality with N=400")

# Breusch-Pagan test for homoscedasticity
# Create design matrix for BP test
X = sm.add_constant(data['TSVR_days'])
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
print(f"\n2. Breusch-Pagan Test (Homoscedasticity):")
print(f"   Statistic: {bp_stat:.4f}")
print(f"   p-value: {bp_p:.4f}")
if bp_p > 0.05:
    print("   ✅ PASS: Homoscedasticity assumption met (p > 0.05)")
else:
    print("   ⚠️ FLAG: Heteroscedasticity detected (p < 0.05)")
    print("   NOTE: Consider robust SEs or weighted LMM")

# Autocorrelation check (Durbin-Watson)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"\n3. Durbin-Watson Test (Autocorrelation):")
print(f"   Statistic: {dw:.4f}")
if 1.5 < dw < 2.5:
    print("   ✅ PASS: No significant autocorrelation (1.5 < DW < 2.5)")
else:
    print("   ⚠️ FLAG: Potential autocorrelation (DW outside [1.5, 2.5])")

# Leverage analysis (Cook's D equivalent)
# For LMM, use standardized residuals
residuals_std = residuals / residuals.std()
high_leverage = (np.abs(residuals_std) > 3).sum()
pct_high_leverage = (high_leverage / len(residuals)) * 100
print(f"\n4. Leverage Analysis (Standardized Residuals):")
print(f"   Observations |std resid| > 3: {high_leverage} ({pct_high_leverage:.1f}%)")
if pct_high_leverage < 1:
    print("   ✅ PASS: Minimal influential observations (<1%)")
else:
    print("   ⚠️ FLAG: Some high-leverage observations detected")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
all_pass = (shapiro_p > 0.05) and (bp_p > 0.05) and (1.5 < dw < 2.5) and (pct_high_leverage < 1)
if all_pass:
    print("✅ ALL DIAGNOSTICS PASS - LMM assumptions met")
else:
    print("⚠️ SOME FLAGS RAISED - Review diagnostic plots")
    print("   LMM generally robust to moderate violations with N=400")
print("=" * 60)
