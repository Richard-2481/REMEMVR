#!/usr/bin/env python3
"""
RQ 6.6.2: PLATINUM Certification - Power Analysis, TOST, and Correlation

This script performs three critical analyses for PLATINUM status:
1. Power analysis for baseline accuracy NULL finding
2. TOST equivalence test (establishes "true null" vs "underpowered")
3. Baseline confidence × baseline accuracy correlation (clarifies unexpected positive effect)

Author: rq_platinum agent
Date: 2025-12-28
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# ANALYSIS 1: POWER ANALYSIS FOR BASELINE ACCURACY NULL
# ==============================================================================

print("=" * 80)
print("ANALYSIS 1: POWER ANALYSIS FOR BASELINE ACCURACY NULL")
print("=" * 80)

# Load regression coefficients
BASE_DIR = '/home/etai/projects/REMEMVR/results/ch6/6.6.2'
coeffs = pd.read_csv(f'{BASE_DIR}/data/step03_regression_coefficients.csv')

# Extract baseline_accuracy effect
baseline_acc_row = coeffs[coeffs['predictor'] == 'z_baseline_accuracy'].iloc[0]
observed_beta = baseline_acc_row['coefficient']  # -0.001
SE = baseline_acc_row['SE']  # 0.002

print(f"\nObserved Effect:")
print(f"  β = {observed_beta:.6f}")
print(f"  SE = {SE:.6f}")
print(f"  p = {baseline_acc_row['p_uncorrected']:.3f}")

# Load effect sizes for R-squared
effect_sizes = pd.read_csv(f'{BASE_DIR}/data/step04_effect_sizes.csv')
R2_full = effect_sizes[effect_sizes['metric'] == 'R_squared']['value'].values[0]
partial_R2 = effect_sizes[effect_sizes['metric'] == 'partial_R2_baseline_accuracy']['value'].values[0]

print(f"\nVariance Explained:")
print(f"  R² (full model) = {R2_full:.4f}")
print(f"  Partial R² (baseline_accuracy) = {partial_R2:.6f}")

# Convert to Cohen's f²
# f² = partial R² / (1 - R² full)
f2_observed = partial_R2 / (1 - R2_full)

print(f"\nEffect Size (Cohen's f²):")
print(f"  f² = {f2_observed:.6f}")
print(f"  Interpretation: Negligible (Cohen's thresholds: small=0.02, medium=0.15, large=0.35)")

# Manual power analysis using non-central F distribution
# Since observed effect is essentially zero, use manual calculation

N = 100
k = 4  # number of predictors
df_num = 1  # testing 1 predictor
df_denom = N - k - 1  # 95
alpha = 0.05

# Critical F value
F_crit = stats.f.ppf(1 - alpha, df_num, df_denom)

print(f"\nPost-Hoc Power (Manual Calculation):")
print(f"  Observed effect: f² = {f2_observed:.6f}")
print(f"  Since effect is essentially zero, power ≈ alpha = {alpha:.4f}")

# Power for standard effect sizes
f2_small = 0.02
f2_medium = 0.15
f2_large = 0.35

# Convert f² to lambda (non-centrality parameter)
# λ = N * f²
lambda_small = N * f2_small
lambda_medium = N * f2_medium
lambda_large = N * f2_large

# Power = 1 - F_nc.cdf(F_crit, df_num, df_denom, λ)
from scipy.stats import ncf

power_small = 1 - ncf.cdf(F_crit, df_num, df_denom, lambda_small)
power_medium = 1 - ncf.cdf(F_crit, df_num, df_denom, lambda_medium)
power_large = 1 - ncf.cdf(F_crit, df_num, df_denom, lambda_large)

print(f"\nPower for Cohen's Standard Effect Sizes:")
print(f"  Small (f²=0.02):   Power = {power_small:.4f}")
print(f"  Medium (f²=0.15):  Power = {power_medium:.4f}")
print(f"  Large (f²=0.35):   Power = {power_large:.4f}")

# N required for 80% power (iterative search)
target_power = 0.80

def find_N_for_power(f2, target_power, df_num, alpha):
    """Find N required for target power"""
    for N_test in range(50, 100000, 10):
        df_denom_test = N_test - 5  # k+1 predictors
        F_crit_test = stats.f.ppf(1 - alpha, df_num, df_denom_test)
        lambda_test = N_test * f2
        power_test = 1 - ncf.cdf(F_crit_test, df_num, df_denom_test, lambda_test)
        if power_test >= target_power:
            return N_test
    return None

N_small = find_N_for_power(f2_small, 0.80, df_num, alpha)
N_medium = find_N_for_power(f2_medium, 0.80, df_num, alpha)
N_large = find_N_for_power(f2_large, 0.80, df_num, alpha)

print(f"\nSample Size for 80% Power:")
print(f"  Small effect (f²=0.02):   N = {N_small}")
print(f"  Medium effect (f²=0.15):  N = {N_medium}")
print(f"  Large effect (f²=0.35):   N = {N_large}")
print(f"  Current N:                N = {N}")

# Assessment
if power_small < 0.80:
    assessment = f"UNDERPOWERED for small effects (power={power_small:.3f} < 0.80)"
else:
    assessment = f"ADEQUATELY POWERED for small effects (power={power_small:.3f} ≥ 0.80)"

print(f"\nAssessment: {assessment}")

# Save results
power_results = pd.DataFrame({
    'metric': [
        'observed_f2',
        'power_observed',
        'power_small_f2_0.02',
        'power_medium_f2_0.15',
        'power_large_f2_0.35',
        'N_for_80pct_power_small',
        'N_for_80pct_power_medium',
        'N_for_80pct_power_large',
        'current_N'
    ],
    'value': [
        f2_observed,
        alpha,  # Power for zero effect = alpha
        power_small,
        power_medium,
        power_large,
        N_small if N_small else 99999,
        N_medium if N_medium else 99999,
        N_large if N_large else 99999,
        N
    ]
})

power_results.to_csv(f'{BASE_DIR}/data/step05_power_analysis.csv', index=False)
print("\n✓ Power analysis saved to data/step05_power_analysis.csv")

# ==============================================================================
# ANALYSIS 2: TOST EQUIVALENCE TEST
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: TOST EQUIVALENCE TEST")
print("=" * 80)

# Equivalence bounds
# Use f² < 0.02 (small effect per Cohen)
# Convert to beta: beta_bound = sqrt(f² * (1 - R²))
f2_bound = 0.02
beta_bound = np.sqrt(f2_bound * (1 - R2_full))

print(f"\nEquivalence Bounds:")
print(f"  f² threshold: {f2_bound} (small effect per Cohen)")
print(f"  β threshold: ±{beta_bound:.6f}")

# TOST (Two One-Sided Tests)
# H0: |β| ≥ beta_bound (effect is NOT negligible)
# H1: |β| < beta_bound (effect IS negligible)

# Test 1: β > -beta_bound
t1 = (observed_beta - (-beta_bound)) / SE
p1 = 1 - stats.t.cdf(t1, df=df_denom)

# Test 2: β < +beta_bound
t2 = (beta_bound - observed_beta) / SE
p2 = 1 - stats.t.cdf(t2, df=df_denom)

# TOST p-value is the MAXIMUM of the two p-values
tost_p = max(p1, p2)

print(f"\nTOST Results:")
print(f"  Test 1 (β > -{beta_bound:.6f}): t = {t1:.4f}, p = {p1:.6f}")
print(f"  Test 2 (β < +{beta_bound:.6f}): t = {t2:.4f}, p = {p2:.6f}")
print(f"  TOST p-value: {tost_p:.6f}")

# Interpretation
if tost_p < 0.05:
    interpretation = "SIGNIFICANT - Effect is statistically equivalent to zero (true null)"
    conclusion = "Baseline accuracy effect is negligible. Dunning-Kruger hypothesis NOT SUPPORTED."
else:
    interpretation = "NON-SIGNIFICANT - Cannot conclude equivalence (may be underpowered)"
    conclusion = "Equivalence test inconclusive. Increase N or use wider bounds."

print(f"\nInterpretation:")
print(f"  {interpretation}")
print(f"  Conclusion: {conclusion}")

# 90% CI for TOST (convention is 90% for equivalence testing)
ci_90_lower = observed_beta - 1.645 * SE
ci_90_upper = observed_beta + 1.645 * SE

print(f"\n90% Confidence Interval:")
print(f"  [{ci_90_lower:.6f}, {ci_90_upper:.6f}]")
print(f"  Equivalence bounds: [{-beta_bound:.6f}, {+beta_bound:.6f}]")

if ci_90_lower > -beta_bound and ci_90_upper < beta_bound:
    ci_interpretation = "✓ 90% CI entirely within equivalence bounds (supports equivalence)"
else:
    ci_interpretation = "✗ 90% CI not entirely within bounds (equivalence uncertain)"

print(f"  {ci_interpretation}")

# Save results
tost_results = pd.DataFrame({
    'metric': [
        'f2_bound',
        'beta_bound',
        'observed_beta',
        'SE',
        't1',
        'p1',
        't2',
        'p2',
        'tost_p',
        'ci_90_lower',
        'ci_90_upper',
        'equivalent',
        'interpretation'
    ],
    'value': [
        f2_bound,
        beta_bound,
        observed_beta,
        SE,
        t1,
        p1,
        t2,
        p2,
        tost_p,
        ci_90_lower,
        ci_90_upper,
        1 if tost_p < 0.05 else 0,
        conclusion
    ]
})

tost_results.to_csv(f'{BASE_DIR}/data/step05_tost_equivalence.csv', index=False)
print("\n✓ TOST results saved to data/step05_tost_equivalence.csv")

# ==============================================================================
# ANALYSIS 3: BASELINE CONFIDENCE × BASELINE ACCURACY CORRELATION
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: BASELINE CONFIDENCE × BASELINE ACCURACY CORRELATION")
print("=" * 80)

# Load predictor data (ORIGINAL scales, not z-scores)
data = pd.read_csv(f'{BASE_DIR}/data/step00_predictor_data.csv')

# Pearson correlation
r, p = stats.pearsonr(data['baseline_confidence'], data['baseline_accuracy'])

print(f"\nCorrelation (Day 0):")
print(f"  Pearson r = {r:.4f}")
print(f"  p-value = {p:.6f}")
print(f"  N = {len(data)}")

# Interpretation
if abs(r) < 0.10:
    r_interpretation = "Negligible (r < 0.10)"
    overconfidence_support = "STRONG - Baseline confidence uncalibrated to accuracy (overconfidence)"
elif abs(r) < 0.30:
    r_interpretation = "Small (0.10 ≤ r < 0.30)"
    overconfidence_support = "MODERATE - Weak calibration suggests overconfidence tendency"
elif abs(r) < 0.50:
    r_interpretation = "Medium (0.30 ≤ r < 0.50)"
    overconfidence_support = "WEAK - Moderate calibration, but positive effect suggests overconfidence post-encoding"
else:
    r_interpretation = "Large (r ≥ 0.50)"
    overconfidence_support = "MINIMAL - Well-calibrated at encoding, positive effect may reflect other mechanisms"

print(f"\nInterpretation:")
print(f"  Effect size: {r_interpretation}")
print(f"  Overconfidence explanation: {overconfidence_support}")

# Variance explained
r_squared = r**2
print(f"\nVariance Explained:")
print(f"  R² = {r_squared:.4f} ({r_squared*100:.2f}% shared variance)")

# 95% CI for correlation
# Fisher z-transformation
z = 0.5 * np.log((1 + r) / (1 - r))
SE_z = 1 / np.sqrt(len(data) - 3)
z_lower = z - 1.96 * SE_z
z_upper = z + 1.96 * SE_z

# Back-transform to r
r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

print(f"\n95% Confidence Interval:")
print(f"  [{r_lower:.4f}, {r_upper:.4f}]")

# Save results
correlation_results = pd.DataFrame({
    'metric': [
        'pearson_r',
        'p_value',
        'r_squared',
        'N',
        'ci_95_lower',
        'ci_95_upper',
        'interpretation',
        'overconfidence_support'
    ],
    'value': [
        r,
        p,
        r_squared,
        len(data),
        r_lower,
        r_upper,
        r_interpretation,
        overconfidence_support
    ]
})

correlation_results.to_csv(f'{BASE_DIR}/data/step05_correlation_confidence_accuracy.csv', index=False)
print("\n✓ Correlation analysis saved to data/step05_correlation_confidence_accuracy.csv")

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("PLATINUM CERTIFICATION - TIER 1 ANALYSES COMPLETE")
print("=" * 80)

print(f"\n1. POWER ANALYSIS:")
print(f"   - Observed effect: f² = {f2_observed:.6f} (negligible)")
print(f"   - Post-hoc power: {alpha:.4f} (effect is zero, power = alpha)")
print(f"   - Power for small effect (f²=0.02): {power_small:.4f}")
print(f"   - N for 80% power (small): {N_small} (current N={N})")
print(f"   → {assessment}")

print(f"\n2. TOST EQUIVALENCE:")
print(f"   - TOST p-value: {tost_p:.6f}")
print(f"   - Significant: {'YES' if tost_p < 0.05 else 'NO'}")
print(f"   - {ci_interpretation}")
print(f"   → Conclusion: {conclusion}")

print(f"\n3. CORRELATION:")
print(f"   - Baseline confidence × accuracy: r = {r:.4f}, p = {p:.6f}")
print(f"   - Interpretation: {r_interpretation}")
print(f"   → Overconfidence explanation: {overconfidence_support}")

print("\n" + "=" * 80)
print("✓ All TIER 1 analyses complete. Outputs saved to data/step05_*.csv")
print("=" * 80)
