"""
Power Analysis for NULL 3-Way Interaction - RQ 5.4.2
Purpose: Determine if NULL finding (p=.938) is true null or underpowered
Required by: improvement_taxonomy.md Section 3.1 (MANDATORY for NULL findings)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from scipy import stats
import sys
import os

print("=" * 80)
print("POWER ANALYSIS FOR NULL 3-WAY INTERACTION - RQ 5.4.2")
print("=" * 80)

# OBSERVED EFFECT EXTRACTION

print("\n1. OBSERVED EFFECT SIZE")
print("-" * 80)

# From step04_hypothesis_tests.csv
hypothesis_file = 'results/step04_hypothesis_tests.csv'
if not os.path.exists(hypothesis_file):
    print(f"ERROR: {hypothesis_file} not found")
    sys.exit(1)

hyp_tests = pd.read_csv(hypothesis_file)
print(f"Loaded {len(hyp_tests)} hypothesis tests from step04")

# Extract 3-way interaction for Congruent
target_test = "Days_within:C(Segment, Treatment('Early'))[T.Late]:C(Congruence, Treatment('Common'))[T.Congruent]"
threeway = hyp_tests[hyp_tests['Test_Name'] == target_test]

if len(threeway) == 0:
    print(f"ERROR: 3-way interaction test not found")
    print(f"Available tests:")
    print(hyp_tests['Test_Name'].values)
    sys.exit(1)

beta = threeway['Coefficient'].values[0]
SE = threeway['SE'].values[0]
z_val = threeway['z_value'].values[0]
p_val = threeway['p_uncorrected'].values[0]

print(f"3-Way Interaction (Days_within × Segment[Late] × Congruence[Congruent]):")
print(f"  β = {beta:.4f}")
print(f"  SE = {SE:.4f}")
print(f"  z = {z_val:.4f}")
print(f"  p (uncorrected) = {p_val:.4f}")

# Convert to Cohen's d (standardized effect size)
# For interaction term: d ≈ 2 * β / σ_residual
# Estimate σ_residual from SE and sample size
N = 1200  # Total observations (100 participants × 4 tests × 3 congruence types)
n_groups = 3 # Congruence types
df = N - 12  # 12 parameters in full piecewise model

# Approximate Cohen's d: effect / pooled SD
# Theta SD ≈ 1.0 (typical IRT scale), so d ≈ β
cohens_d = beta  # Conservative approximation
print(f"\nStandardized effect size:")
print(f"  Cohen's d ≈ {cohens_d:.4f}")
print(f"  Interpretation: {abs(cohens_d):.4f} SD units")

# Convert to f² (effect size for ANOVA)
# f² = d² / 4 for 2-group comparison
# For 3-way interaction, approximate as f² = (β/SE)² / N
f_squared = (z_val ** 2) / N
print(f"  Cohen's f² ≈ {f_squared:.6f}")
print(f"  Interpretation: {f_squared*100:.4f}% variance explained")

# POST-HOC POWER FOR OBSERVED EFFECT

print("\n2. POST-HOC POWER FOR OBSERVED EFFECT")
print("-" * 80)

# Using F-test power (appropriate for LMM interaction terms)
power_calc = FTestAnovaPower()

# Post-hoc power: Given observed f², N, alpha → solve for power
alpha = 0.05
k_groups = 3  # Congruence types

try:
    observed_power = power_calc.solve_power(
        effect_size=f_squared,
        nobs=N,
        alpha=alpha,
        k_groups=k_groups
    )
    print(f"Post-hoc power for observed effect (f² = {f_squared:.6f}):")
    print(f"  N = {N}")
    print(f"  Alpha = {alpha}")
    print(f"  Power = {observed_power:.4f}")
    print(f"  Interpretation: {observed_power*100:.2f}% chance to detect this effect")
except Exception as e:
    print(f"WARNING: Could not compute power (effect too small)")
    print(f"  f² = {f_squared:.6f} (near zero)")
    observed_power = np.nan

# POWER FOR MEANINGFUL THRESHOLDS

print("\n3. POWER FOR MEANINGFUL EFFECT THRESHOLDS")
print("-" * 80)

# Test power for conventional effect sizes
thresholds = {
    'Small (d=0.20)': 0.20,
    'Medium (d=0.50)': 0.50,
    'Large (d=0.80)': 0.80
}

print("Power to detect conventional effect sizes:")
for label, d_threshold in thresholds.items():
    # Convert d to f²
    f2_threshold = (d_threshold ** 2) / 4
    
    try:
        power_threshold = power_calc.solve_power(
            effect_size=f2_threshold,
            nobs=N,
            alpha=alpha,
            k_groups=k_groups
        )
        print(f"  {label}: f² = {f2_threshold:.4f}, Power = {power_threshold:.4f}")
    except Exception as e:
        print(f"  {label}: f² = {f2_threshold:.4f}, Power = [computational error]")

# SAMPLE SIZE FOR 0.80 POWER

print("\n4. SAMPLE SIZE REQUIRED FOR 0.80 POWER")
print("-" * 80)

target_power = 0.80

print(f"N required for 0.80 power to detect:")

# For observed effect
if not np.isnan(observed_power) and f_squared > 0:
    try:
        n_required_observed = power_calc.solve_power(
            effect_size=f_squared,
            power=target_power,
            alpha=alpha,
            k_groups=k_groups
        )
        print(f"  Observed effect (f² = {f_squared:.6f}): N = {n_required_observed:.0f}")
        print(f"    Current N = {N}, would need {n_required_observed - N:.0f} more observations")
    except Exception as e:
        print(f"  Observed effect: N > 1,000,000 (effect too small)")
        n_required_observed = np.nan

# For small effect (d=0.20)
f2_small = (0.20 ** 2) / 4
try:
    n_required_small = power_calc.solve_power(
        effect_size=f2_small,
        power=target_power,
        alpha=alpha,
        k_groups=k_groups
    )
    print(f"  Small effect (d=0.20, f²={f2_small:.4f}): N = {n_required_small:.0f}")
except Exception as e:
    print(f"  Small effect (d=0.20): [computational error]")
    n_required_small = np.nan

# INTERPRETATION & CONCLUSION

print("\n5. INTERPRETATION & CONCLUSION")
print("-" * 80)

print("\nFinding Classification:")
if not np.isnan(observed_power):
    if observed_power < 0.20:
        classification = "SEVERELY UNDERPOWERED"
    elif observed_power < 0.60:
        classification = "UNDERPOWERED"
    elif observed_power < 0.80:
        classification = "MARGINALLY POWERED"
    else:
        classification = "ADEQUATELY POWERED"
else:
    classification = "EFFECT TOO SMALL TO COMPUTE"

print(f"  {classification}")

if abs(cohens_d) < 0.10:
    print(f"  Observed effect (d={cohens_d:.4f}) is NEGLIGIBLE (< 0.10)")
    print(f"  Even with infinite sample, effect is trivial")
    print(f"  Conclusion: TRUE NULL (not underpowered)")
elif abs(cohens_d) < 0.20:
    print(f"  Observed effect (d={cohens_d:.4f}) is VERY SMALL (< 0.20)")
    if not np.isnan(n_required_small):
        print(f"  Would require N > {int(n_required_small)} to detect reliably")
        if N < n_required_small:
            print(f"  Current N = {N} is insufficient")
            print(f"  Conclusion: LIKELY TRUE NULL (effect below meaningful threshold)")
        else:
            print(f"  Current N = {N} is sufficient for small effects")
            print(f"  Conclusion: UNDERPOWERED (increase N)")
    else:
        print(f"  Conclusion: TRUE NULL (effect below detectable threshold)")
else:
    print(f"  Observed effect (d={cohens_d:.4f}) is SMALL or larger")
    if not np.isnan(observed_power) and observed_power < 0.80:
        if not np.isnan(n_required_observed):
            print(f"  Power = {observed_power:.4f} < 0.80")
            print(f"  Conclusion: UNDERPOWERED (increase N to {int(n_required_observed)})")
        else:
            print(f"  Conclusion: UNDERPOWERED (effect size requires very large N)")
    else:
        print(f"  Power = {observed_power:.4f} >= 0.80")
        print(f"  Conclusion: ADEQUATELY POWERED NULL")

# SAVE RESULTS

print("\n6. SAVING RESULTS")
print("-" * 80)

results = pd.DataFrame({
    'Metric': [
        'Observed Beta',
        'Standard Error',
        'Z-value',
        'P-value (uncorrected)',
        "Cohen's d",
        "Cohen's f²",
        'Post-hoc Power',
        'Classification',
        'Sample Size (N)',
        'N for 0.80 Power (observed)',
        'N for 0.80 Power (small d=0.20)'
    ],
    'Value': [
        f"{beta:.4f}",
        f"{SE:.4f}",
        f"{z_val:.4f}",
        f"{p_val:.4f}",
        f"{cohens_d:.4f}",
        f"{f_squared:.6f}",
        f"{observed_power:.4f}" if not np.isnan(observed_power) else "N/A (effect too small)",
        classification,
        f"{N}",
        f"{int(n_required_observed)}" if not np.isnan(n_required_observed) and not np.isnan(observed_power) and f_squared > 0 else "> 1,000,000",
        f"{int(n_required_small)}" if not np.isnan(n_required_small) else "N/A"
    ]
})

output_file = 'results/step07_power_analysis.csv'
results.to_csv(output_file, index=False)
print(f"Saved power analysis results to {output_file}")

print("\n" + "=" * 80)
print("POWER ANALYSIS COMPLETE")
print("=" * 80)
