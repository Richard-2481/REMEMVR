"""
TOST Equivalence Testing for NULL 3-Way Interaction - RQ 5.4.2
Purpose: Establish "true null" vs "underpowered" for p=.938 finding
Required by: improvement_taxonomy.md Section 3.2 (Recommended for NULL claims)
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

print("=" * 80)
print("TOST EQUIVALENCE TESTING - RQ 5.4.2")
print("=" * 80)

# OBSERVED EFFECT EXTRACTION

print("\n1. OBSERVED EFFECT")
print("-" * 80)

# From step04_hypothesis_tests.csv
hypothesis_file = 'results/step04_hypothesis_tests.csv'
hyp_tests = pd.read_csv(hypothesis_file)

# Extract 3-way interaction for Congruent
target_test = "Days_within:C(Segment, Treatment('Early'))[T.Late]:C(Congruence, Treatment('Common'))[T.Congruent]"
threeway = hyp_tests[hyp_tests['Test_Name'] == target_test]

beta = threeway['Coefficient'].values[0]
SE = threeway['SE'].values[0]
z_val = threeway['z_value'].values[0]
p_val = threeway['p_uncorrected'].values[0]

print(f"3-Way Interaction (Days_within × Segment[Late] × Congruence[Congruent]):")
print(f"  β = {beta:.4f}")
print(f"  SE = {SE:.4f}")
print(f"  z = {z_val:.4f}")
print(f"  p (NHST) = {p_val:.4f}")

# Sample parameters
N = 1200  # Total observations
df = N - 12  # Degrees of freedom (12 parameters in model)

# EQUIVALENCE BOUNDS

print("\n2. EQUIVALENCE BOUNDS")
print("-" * 80)

# Set equivalence bounds based on Cohen's d thresholds
# Using d = 0.20 (small effect) as meaningful threshold
d_bound = 0.20
print(f"Meaningful effect threshold: Cohen's d = {d_bound}")

# Convert Cohen's d to raw beta (theta SD ≈ 1.0)
beta_bound = d_bound  # For theta scale with SD≈1

print(f"Equivalence bounds: β in [{-beta_bound:.4f}, {beta_bound:.4f}]")
print(f"  Lower bound: {-beta_bound:.4f}")
print(f"  Upper bound: {beta_bound:.4f}")

# TWO ONE-SIDED TESTS (TOST)

print("\n3. TWO ONE-SIDED TESTS (TOST)")
print("-" * 80)

# Test 1: β > lower_bound (-0.20)
# H0: β <= -0.20, H1: β > -0.20
t1 = (beta - (-beta_bound)) / SE
p1 = 1 - stats.norm.cdf(t1)  # One-sided upper tail

print(f"\nTest 1: β > lower bound ({-beta_bound:.4f})")
print(f"  H0: β <= {-beta_bound:.4f}")
print(f"  H1: β > {-beta_bound:.4f}")
print(f"  t1 = {t1:.4f}")
print(f"  p1 (one-sided) = {p1:.4f}")
if p1 < 0.05:
    print(f"  ✓ REJECT H0: Effect is greater than lower bound")
else:
    print(f"  ✗ FAIL TO REJECT H0: Cannot rule out effect below lower bound")

# Test 2: β < upper_bound (0.20)
# H0: β >= 0.20, H1: β < 0.20
t2 = (beta - beta_bound) / SE
p2 = stats.norm.cdf(t2)  # One-sided lower tail

print(f"\nTest 2: β < upper bound ({beta_bound:.4f})")
print(f"  H0: β >= {beta_bound:.4f}")
print(f"  H1: β < {beta_bound:.4f}")
print(f"  t2 = {t2:.4f}")
print(f"  p2 (one-sided) = {p2:.4f}")
if p2 < 0.05:
    print(f"  ✓ REJECT H0: Effect is less than upper bound")
else:
    print(f"  ✗ FAIL TO REJECT H0: Cannot rule out effect above upper bound")

# EQUIVALENCE CONCLUSION

print("\n4. EQUIVALENCE CONCLUSION")
print("-" * 80)

# TOST p-value = max(p1, p2)
tost_p = max(p1, p2)

print(f"TOST p-value: {tost_p:.4f}")
print(f"Significance threshold: α = 0.05")

if p1 < 0.05 and p2 < 0.05:
    conclusion = "EQUIVALENT"
    interpretation = "TRUE NULL - Effect is statistically equivalent to zero"
    print(f"\n✓ EQUIVALENCE ESTABLISHED (both tests p < 0.05)")
    print(f"  Observed effect (β = {beta:.4f}) is within equivalence bounds")
    print(f"  Conclusion: {interpretation}")
    print(f"\n  Effect is smaller than Cohen's d = {d_bound} (meaningful threshold)")
    print(f"  This is a TRUE NULL, not underpowered")
else:
    conclusion = "NOT EQUIVALENT"
    interpretation = "INDETERMINATE - Cannot establish equivalence"
    print(f"\n✗ EQUIVALENCE NOT ESTABLISHED")
    if p1 >= 0.05:
        print(f"  Test 1 failed: Cannot rule out effect below {-beta_bound:.4f}")
    if p2 >= 0.05:
        print(f"  Test 2 failed: Cannot rule out effect above {beta_bound:.4f}")
    print(f"  Conclusion: {interpretation}")
    print(f"\n  Effect may be within or outside equivalence bounds")
    print(f"  Recommendation: Increase sample size or revise equivalence bounds")

# CONFIDENCE INTERVAL FOR EQUIVALENCE

print("\n5. CONFIDENCE INTERVAL FOR EQUIVALENCE")
print("-" * 80)

# 90% CI for TOST (corresponds to two one-sided 5% tests)
ci_level = 0.90
z_crit = stats.norm.ppf(1 - (1 - ci_level)/2)

ci_lower = beta - z_crit * SE
ci_upper = beta + z_crit * SE

print(f"{int(ci_level*100)}% Confidence Interval for β:")
print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")

print(f"\nEquivalence bounds:")
print(f"  [{-beta_bound:.4f}, {beta_bound:.4f}]")

if ci_lower > -beta_bound and ci_upper < beta_bound:
    print(f"\n✓ ENTIRE 90% CI is within equivalence bounds")
    print(f"  Strong evidence for equivalence")
elif ci_lower < -beta_bound or ci_upper > beta_bound:
    print(f"\n✗ 90% CI extends outside equivalence bounds")
    overlap_lower = max(ci_lower, -beta_bound)
    overlap_upper = min(ci_upper, beta_bound)
    if overlap_lower < overlap_upper:
        overlap_pct = ((overlap_upper - overlap_lower) / (ci_upper - ci_lower)) * 100
        print(f"  {overlap_pct:.1f}% of CI overlaps with equivalence region")

# ALTERNATIVE EQUIVALENCE BOUNDS

print("\n6. SENSITIVITY TO EQUIVALENCE BOUNDS")
print("-" * 80)

bounds_to_test = {
    'Very Small (d=0.10)': 0.10,
    'Small (d=0.20)': 0.20,
    'Medium (d=0.50)': 0.50
}

print("Testing equivalence with different thresholds:")
results_by_bound = []

for label, d in bounds_to_test.items():
    beta_b = d
    t1_b = (beta - (-beta_b)) / SE
    p1_b = 1 - stats.norm.cdf(t1_b)
    
    t2_b = (beta - beta_b) / SE
    p2_b = stats.norm.cdf(t2_b)
    
    tost_p_b = max(p1_b, p2_b)
    equiv = "YES" if (p1_b < 0.05 and p2_b < 0.05) else "NO"
    
    results_by_bound.append({
        'Threshold': label,
        'Cohen_d': d,
        'TOST_p': tost_p_b,
        'Equivalent': equiv
    })
    
    print(f"  {label}: TOST p = {tost_p_b:.4f}, Equivalent: {equiv}")

# SAVE RESULTS

print("\n7. SAVING RESULTS")
print("-" * 80)

# Main results
main_results = pd.DataFrame({
    'Metric': [
        'Observed Beta',
        'Standard Error',
        'Equivalence Bound (d)',
        'Equivalence Bound (beta)',
        'Test 1: β > lower bound',
        'Test 1: p-value',
        'Test 2: β < upper bound',
        'Test 2: p-value',
        'TOST p-value',
        'Equivalence Established',
        'Conclusion',
        '90% CI Lower',
        '90% CI Upper',
        'CI within bounds'
    ],
    'Value': [
        f"{beta:.4f}",
        f"{SE:.4f}",
        f"{d_bound:.2f}",
        f"{beta_bound:.4f}",
        f"t = {t1:.4f}",
        f"{p1:.4f}",
        f"t = {t2:.4f}",
        f"{p2:.4f}",
        f"{tost_p:.4f}",
        conclusion,
        interpretation,
        f"{ci_lower:.4f}",
        f"{ci_upper:.4f}",
        "YES" if (ci_lower > -beta_bound and ci_upper < beta_bound) else "NO"
    ]
})

output_file1 = 'results/step08_tost_main_results.csv'
main_results.to_csv(output_file1, index=False)
print(f"Saved main TOST results to {output_file1}")

# Sensitivity results
sensitivity_df = pd.DataFrame(results_by_bound)
output_file2 = 'results/step08_tost_sensitivity.csv'
sensitivity_df.to_csv(output_file2, index=False)
print(f"Saved sensitivity analysis to {output_file2}")

print("\n" + "=" * 80)
print("TOST EQUIVALENCE TESTING COMPLETE")
print("=" * 80)
