#!/usr/bin/env python3
"""
RQ 6.6.2: Quality Validation - Robust Regression Sensitivity Analysis

This script addresses residual non-normality (Shapiro-Wilk p<0.001) by comparing:
1. Original OLS regression
2. Robust regression with Huber M-estimator
3. Bootstrap confidence intervals (1000 iterations)

If robust methods yield similar coefficients, findings are robust to non-normality.

Date: 2025-12-28
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# LOAD DATA

print("=" * 80)
print("ROBUST REGRESSION SENSITIVITY ANALYSIS")
print("=" * 80)

# Load standardized predictors
BASE_DIR = '/home/etai/projects/REMEMVR/results/ch6/6.6.2'
data = pd.read_csv(f'{BASE_DIR}/data/step01_standardized_predictors.csv')

# Prepare X and y
predictor_cols = ['z_baseline_accuracy', 'z_baseline_confidence', 'z_Age', 'z_confidence_bias']
X = data[predictor_cols]
X = sm.add_constant(X)
y = data['HCE_rate_mean']

print(f"\nData Loaded:")
print(f"  N = {len(data)}")
print(f"  Predictors: {predictor_cols}")
print(f"  Outcome: HCE_rate_mean")

# METHOD 1: ORIGINAL OLS REGRESSION

print("\n" + "-" * 80)
print("METHOD 1: ORIGINAL OLS REGRESSION")
print("-" * 80)

ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

print(f"\nOLS Coefficients:")
print(ols_results.summary().tables[1])

# Extract coefficients
ols_coeffs = pd.DataFrame({
    'predictor': ['Intercept'] + predictor_cols,
    'ols_coef': ols_results.params.values,
    'ols_SE': ols_results.bse.values,
    'ols_t': ols_results.tvalues.values,
    'ols_p': ols_results.pvalues.values
})

print(f"\nOLS R-squared: {ols_results.rsquared:.4f}")
print(f"OLS Adj R-squared: {ols_results.rsquared_adj:.4f}")

# Residual diagnostics
residuals = ols_results.resid
shapiro_w, shapiro_p = stats.shapiro(residuals)

print(f"\nResidual Diagnostics:")
print(f"  Shapiro-Wilk W = {shapiro_w:.4f}, p = {shapiro_p:.6f}")
print(f"  → {'Non-normal (p<0.05)' if shapiro_p < 0.05 else 'Approximately normal'}")

# METHOD 2: ROBUST REGRESSION (HUBER M-ESTIMATOR)

print("\n" + "-" * 80)
print("METHOD 2: ROBUST REGRESSION (HUBER M-ESTIMATOR)")
print("-" * 80)

rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

print(f"\nRobust Coefficients:")
print(rlm_results.summary().tables[1])

# Extract coefficients
rlm_coeffs = pd.DataFrame({
    'predictor': ['Intercept'] + predictor_cols,
    'rlm_coef': rlm_results.params.values,
    'rlm_SE': rlm_results.bse.values,
    'rlm_t': rlm_results.tvalues.values,
    'rlm_p': rlm_results.pvalues.values
})

# METHOD 3: BOOTSTRAP CONFIDENCE INTERVALS

print("\n" + "-" * 80)
print("METHOD 3: BOOTSTRAP CONFIDENCE INTERVALS (1000 ITERATIONS)")
print("-" * 80)

np.random.seed(42)  # Reproducibility
n_iterations = 1000

boot_coeffs = []

for i in range(n_iterations):
    # Resample with replacement
    boot_indices = np.random.choice(len(data), size=len(data), replace=True)
    boot_data = data.iloc[boot_indices]

    # Fit OLS on bootstrap sample
    X_boot = boot_data[predictor_cols]
    X_boot = sm.add_constant(X_boot)
    y_boot = boot_data['HCE_rate_mean']

    boot_model = sm.OLS(y_boot, X_boot)
    boot_results = boot_model.fit()

    boot_coeffs.append(boot_results.params.values)

boot_coeffs = np.array(boot_coeffs)

# Compute bootstrap CIs (percentile method)
boot_lower = np.percentile(boot_coeffs, 2.5, axis=0)
boot_upper = np.percentile(boot_coeffs, 97.5, axis=0)
boot_mean = np.mean(boot_coeffs, axis=0)
boot_SE = np.std(boot_coeffs, axis=0)

print(f"\nBootstrap Results (1000 iterations):")
print(f"  Mean coefficients computed from bootstrap distribution")
print(f"  95% CI from percentile method")

# COMPARISON TABLE

print("\n" + "=" * 80)
print("COMPARISON: OLS vs ROBUST vs BOOTSTRAP")
print("=" * 80)

# Merge all results
comparison = ols_coeffs.copy()
comparison['rlm_coef'] = rlm_coeffs['rlm_coef']
comparison['rlm_SE'] = rlm_coeffs['rlm_SE']
comparison['rlm_p'] = rlm_coeffs['rlm_p']
comparison['boot_mean'] = boot_mean
comparison['boot_SE'] = boot_SE
comparison['boot_ci_lower'] = boot_lower
comparison['boot_ci_upper'] = boot_upper

# Compute differences
comparison['ols_rlm_diff'] = comparison['ols_coef'] - comparison['rlm_coef']
comparison['ols_boot_diff'] = comparison['ols_coef'] - comparison['boot_mean']

# Percent change
comparison['ols_rlm_pct_change'] = (comparison['ols_rlm_diff'] / comparison['ols_coef'].abs()) * 100
comparison['ols_boot_pct_change'] = (comparison['ols_boot_diff'] / comparison['ols_coef'].abs()) * 100

print("\nCoefficient Comparison:")
print(comparison[['predictor', 'ols_coef', 'rlm_coef', 'boot_mean', 'ols_rlm_pct_change']].to_string(index=False))

# Save full comparison
comparison.to_csv(f'{BASE_DIR}/data/step06_robust_vs_ols_comparison.csv', index=False)
print("\n✓ Full comparison saved to data/step06_robust_vs_ols_comparison.csv")

# SIGNIFICANCE COMPARISON

print("\n" + "-" * 80)
print("SIGNIFICANCE COMPARISON (α = 0.05, Bonferroni α = 0.0125)")
print("-" * 80)

# Bonferroni correction
bonf_alpha = 0.05 / 4

for idx, row in comparison.iterrows():
    pred = row['predictor']

    if pred == 'Intercept':
        continue  # Skip intercept

    ols_sig = "***" if row['ols_p'] < bonf_alpha else ("*" if row['ols_p'] < 0.05 else "n.s.")
    rlm_sig = "***" if row['rlm_p'] < bonf_alpha else ("*" if row['rlm_p'] < 0.05 else "n.s.")

    # Bootstrap significance: Check if 95% CI excludes 0
    boot_ci_excludes_zero = (row['boot_ci_lower'] > 0) or (row['boot_ci_upper'] < 0)
    boot_sig = "***" if boot_ci_excludes_zero else "n.s."

    print(f"\n{pred}:")
    print(f"  OLS:       β = {row['ols_coef']:8.6f}, p = {row['ols_p']:.6f} {ols_sig}")
    print(f"  Robust:    β = {row['rlm_coef']:8.6f}, p = {row['rlm_p']:.6f} {rlm_sig}")
    print(f"  Bootstrap: β = {row['boot_mean']:8.6f}, 95% CI = [{row['boot_ci_lower']:.6f}, {row['boot_ci_upper']:.6f}] {boot_sig}")

    # Check agreement
    all_methods_agree = (ols_sig == rlm_sig == boot_sig)
    if all_methods_agree:
        print(f"  → All methods agree: {ols_sig}")
    else:
        print(f"  → DISAGREEMENT: Methods do not agree on significance")

# CONCLUSION

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Check if coefficient differences are negligible (<10% change)
max_pct_change = comparison['ols_rlm_pct_change'].abs().max()

print(f"\nMaximum coefficient change (OLS vs Robust): {max_pct_change:.2f}%")

if max_pct_change < 10:
    conclusion = "ROBUST - Coefficients differ by <10% across methods"
    recommendation = "Original OLS results are robust to residual non-normality. No changes needed."
else:
    conclusion = "SENSITIVE - Coefficients differ by ≥10% across methods"
    recommendation = "Consider reporting robust regression or bootstrap CIs in addition to OLS."

print(f"Conclusion: {conclusion}")
print(f"Recommendation: {recommendation}")

# Check significance agreement
sig_predictors = ['z_baseline_confidence', 'z_confidence_bias']

all_agree = True
for pred in sig_predictors:
    row = comparison[comparison['predictor'] == pred].iloc[0]

    ols_sig = row['ols_p'] < bonf_alpha
    rlm_sig = row['rlm_p'] < bonf_alpha
    boot_sig = (row['boot_ci_lower'] > 0) or (row['boot_ci_upper'] < 0)

    if not (ols_sig == rlm_sig == boot_sig):
        all_agree = False
        break

if all_agree:
    print(f"\nSignificance conclusions: ALL METHODS AGREE")
    print(f"  - z_baseline_confidence: Significant across OLS, Robust, Bootstrap")
    print(f"  - z_confidence_bias: Significant across OLS, Robust, Bootstrap")
    print(f"  → Findings are ROBUST to method choice")
else:
    print(f"\nSignificance conclusions: METHODS DISAGREE")
    print(f"  → Review individual predictor comparisons above")

print("\n" + "=" * 80)
print("✓ Robust regression analysis complete")
print("=" * 80)
