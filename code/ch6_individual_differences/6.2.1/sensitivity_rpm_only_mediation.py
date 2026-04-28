#!/usr/bin/env python3
"""
Sensitivity Analysis: RPM-Only Mediation
=========================================
Tests whether fluid reasoning (RPM) alone mediates the age-memory relationship,
addressing the circularity concern that RAVLT/BVMT-R are memory tests mediating
a memory outcome.

Design: Age -> RPM -> REMEMVR theta (accuracy)
Method: Bootstrap mediation (5,000 iterations, percentile CIs)
Comparison: Full-battery mediation (indirect = -0.156, CI [-0.256, -0.072])

This is a standalone sensitivity analysis. Does NOT modify existing files.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]

# Load data

dataset = pd.read_csv(RQ_DIR / "data" / "step01_analysis_dataset.csv")
print(f"Loaded {len(dataset)} participants")
print(f"Age range: {dataset['Age'].min()}-{dataset['Age'].max()} (M={dataset['Age'].mean():.1f}, SD={dataset['Age'].std():.1f})")
print(f"RPM_T range: {dataset['RPM_T'].min():.1f}-{dataset['RPM_T'].max():.1f}")
print(f"theta_all range: {dataset['theta_all'].min():.3f}-{dataset['theta_all'].max():.3f}")
print()

# Point estimates: Baron & Kenny paths

y = dataset['theta_all'].values
age = dataset['Age_std'].values
rpm = dataset['RPM_T_std'].values

# Path c: Age -> theta (total effect)
X_c = sm.add_constant(age)
model_c = sm.OLS(y, X_c).fit()
c_total = model_c.params[1]
c_total_se = model_c.bse[1]
c_total_p = model_c.pvalues[1]

# Path a: Age -> RPM
X_a = sm.add_constant(age)
model_a = sm.OLS(rpm, X_a).fit()
a_path = model_a.params[1]
a_path_se = model_a.bse[1]
a_path_p = model_a.pvalues[1]

# Paths b and c': Age + RPM -> theta
X_bc = sm.add_constant(np.column_stack([age, rpm]))
model_bc = sm.OLS(y, X_bc).fit()
c_prime = model_bc.params[1]  # direct effect of Age
b_path = model_bc.params[2]   # RPM -> theta controlling for Age
c_prime_se = model_bc.bse[1]
c_prime_p = model_bc.pvalues[1]
b_path_se = model_bc.bse[2]
b_path_p = model_bc.pvalues[2]

# Indirect effect = a * b
indirect = a_path * b_path

# Proportion mediated
prop_mediated = indirect / c_total if c_total != 0 else np.nan

print("=" * 70)
print("POINT ESTIMATES (Baron & Kenny / Product of Coefficients)")
print("=" * 70)
print(f"\nPath a (Age -> RPM):        beta = {a_path:.4f}, SE = {a_path_se:.4f}, p = {a_path_p:.4f}")
print(f"Path b (RPM -> theta|Age):  beta = {b_path:.4f}, SE = {b_path_se:.4f}, p = {b_path_p:.4f}")
print(f"Path c  (total effect):     beta = {c_total:.4f}, SE = {c_total_se:.4f}, p = {c_total_p:.4f}")
print(f"Path c' (direct effect):    beta = {c_prime:.4f}, SE = {c_prime_se:.4f}, p = {c_prime_p:.4f}")
print(f"\nIndirect effect (a*b):      {indirect:.4f}")
print(f"Proportion mediated:        {prop_mediated:.4f} ({prop_mediated*100:.1f}%)")

# Model fit for c' model
print(f"\nModel R² (Age only):        {model_c.rsquared:.4f}")
print(f"Model R² (Age + RPM):       {model_bc.rsquared:.4f}")
print(f"Delta R² (RPM added):       {model_bc.rsquared - model_c.rsquared:.4f}")

# Bootstrap: 5,000 iterations for indirect effect CI

print("\n" + "=" * 70)
print("BOOTSTRAP MEDIATION (5,000 iterations)")
print("=" * 70)

np.random.seed(42)
n_boot = 5000
n_obs = len(dataset)

boot_indirect = np.zeros(n_boot)
boot_a = np.zeros(n_boot)
boot_b = np.zeros(n_boot)
boot_c = np.zeros(n_boot)
boot_c_prime = np.zeros(n_boot)
boot_proportion = np.zeros(n_boot)

for i in range(n_boot):
    idx = np.random.choice(n_obs, size=n_obs, replace=True)
    y_b = y[idx]
    age_b = age[idx]
    rpm_b = rpm[idx]

    # Path a
    X_a_b = sm.add_constant(age_b)
    m_a = sm.OLS(rpm_b, X_a_b).fit()
    a_b = m_a.params[1]

    # Path c (total)
    X_c_b = sm.add_constant(age_b)
    m_c = sm.OLS(y_b, X_c_b).fit()
    c_b = m_c.params[1]

    # Paths b and c'
    X_bc_b = sm.add_constant(np.column_stack([age_b, rpm_b]))
    m_bc = sm.OLS(y_b, X_bc_b).fit()
    cp_b = m_bc.params[1]
    b_b = m_bc.params[2]

    boot_a[i] = a_b
    boot_b[i] = b_b
    boot_c[i] = c_b
    boot_c_prime[i] = cp_b
    boot_indirect[i] = a_b * b_b
    boot_proportion[i] = (a_b * b_b) / c_b if c_b != 0 else np.nan

# Percentile CIs
def pct_ci(arr, alpha=0.05):
    arr_clean = arr[~np.isnan(arr)]
    return np.percentile(arr_clean, [100 * alpha / 2, 100 * (1 - alpha / 2)])

ci_indirect = pct_ci(boot_indirect)
ci_a = pct_ci(boot_a)
ci_b = pct_ci(boot_b)
ci_c = pct_ci(boot_c)
ci_c_prime = pct_ci(boot_c_prime)
ci_proportion = pct_ci(boot_proportion)

indirect_sig = "SIGNIFICANT" if (ci_indirect[0] > 0 and ci_indirect[1] > 0) or (ci_indirect[0] < 0 and ci_indirect[1] < 0) else "NOT SIGNIFICANT"
direct_sig = "SIGNIFICANT" if (ci_c_prime[0] > 0 and ci_c_prime[1] > 0) or (ci_c_prime[0] < 0 and ci_c_prime[1] < 0) else "NOT SIGNIFICANT"

print(f"\nPath a  (Age -> RPM):       {a_path:.4f}  95% CI [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
print(f"Path b  (RPM -> theta|Age): {b_path:.4f}  95% CI [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
print(f"Path c  (total effect):     {c_total:.4f}  95% CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]")
print(f"Path c' (direct effect):    {c_prime:.4f}  95% CI [{ci_c_prime[0]:.4f}, {ci_c_prime[1]:.4f}]")
print(f"\nIndirect effect (a*b):      {indirect:.4f}  95% CI [{ci_indirect[0]:.4f}, {ci_indirect[1]:.4f}]  {indirect_sig}")
print(f"Direct effect (c'):         {c_prime:.4f}  95% CI [{ci_c_prime[0]:.4f}, {ci_c_prime[1]:.4f}]  {direct_sig}")
print(f"Proportion mediated:        {prop_mediated:.4f}  95% CI [{ci_proportion[0]:.4f}, {ci_proportion[1]:.4f}]")

# Comparison with full-battery mediation

print("\n" + "=" * 70)
print("COMPARISON: RPM-Only vs Full Battery Mediation")
print("=" * 70)
print(f"\n{'Metric':<30} {'RPM Only':>15} {'Full Battery':>15}")
print("-" * 62)
print(f"{'Indirect effect':<30} {indirect:>15.4f} {-0.156:>15.4f}")
print(f"{'Indirect CI lower':<30} {ci_indirect[0]:>15.4f} {-0.256:>15.4f}")
print(f"{'Indirect CI upper':<30} {ci_indirect[1]:>15.4f} {-0.072:>15.4f}")
print(f"{'CI excludes zero?':<30} {indirect_sig:>15} {'SIGNIFICANT':>15}")
label_cp = "Direct effect (c')"
print(f"{label_cp:<30} {c_prime:>15.4f} {0.026:>15.4f}")
print(f"{'Proportion mediated':<30} {prop_mediated*100:>14.1f}% {120.5:>14.1f}%")

# Interpretation

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if indirect_sig == "SIGNIFICANT":
    print("""
RESULT: The indirect effect through RPM alone is SIGNIFICANT.
The mediation finding is NOT circular. Fluid reasoning (a non-memory
construct) carries a significant portion of the age-memory mediation.
This supports the claim that age effects on VR memory are mediated
by fluid cognitive capacity, not merely by the tautological inclusion
of memory tests as mediators.""")
else:
    print("""
RESULT: The indirect effect through RPM alone is NOT SIGNIFICANT.
The mediation finding may be partly driven by the memory tests
(RAVLT/BVMT-R) in the mediator block. The age-VR memory mediation
claim needs qualifying: it appears to require memory-specific
mediators, not just fluid reasoning.""")

if direct_sig == "NOT SIGNIFICANT":
    print(f"""
The direct effect of age (c' = {c_prime:.4f}) remains non-significant
when controlling for RPM only, consistent with the full-battery finding
(c' = +0.026, p = .751).""")
else:
    print(f"""
NOTE: The direct effect of age (c' = {c_prime:.4f}) becomes significant
when only RPM is controlled, unlike the full-battery model. This suggests
the memory tests were absorbing additional age-related variance.""")

# Save results

results = pd.DataFrame({
    'analysis': ['RPM_only_mediation'],
    'path_a_age_rpm': [a_path],
    'path_a_p': [a_path_p],
    'path_b_rpm_theta': [b_path],
    'path_b_p': [b_path_p],
    'path_c_total': [c_total],
    'path_c_p': [c_total_p],
    'path_c_prime_direct': [c_prime],
    'path_c_prime_p': [c_prime_p],
    'indirect_effect': [indirect],
    'indirect_ci_lower': [ci_indirect[0]],
    'indirect_ci_upper': [ci_indirect[1]],
    'indirect_significant': [indirect_sig == "SIGNIFICANT"],
    'proportion_mediated': [prop_mediated],
    'proportion_ci_lower': [ci_proportion[0]],
    'proportion_ci_upper': [ci_proportion[1]],
    'r2_age_only': [model_c.rsquared],
    'r2_age_rpm': [model_bc.rsquared],
    'n_bootstrap': [n_boot],
    'n_participants': [n_obs],
    'seed': [42]
})

out_path = RQ_DIR / "data" / "sensitivity_rpm_only_mediation.csv"
results.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
