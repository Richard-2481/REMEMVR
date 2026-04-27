"""
Intercepts-Only Sensitivity Check
Addresses convergence warning by testing simpler model
"""

import pandas as pd
import statsmodels.formula.api as smf

# Load data
data = pd.read_csv('data/step02_gamma_lmm_input.csv')

print("=" * 60)
print("RANDOM EFFECTS STRUCTURE COMPARISON")
print("=" * 60)

# Model 1: Intercepts + Slopes (original, boundary estimate)
print("\nModel 1: Random Intercepts + Slopes")
print("-" * 60)
model1 = smf.mixedlm(
    "gamma ~ TSVR_days",
    data=data,
    groups=data['UID'],
    re_formula="~TSVR_days"
)
result1 = model1.fit(reml=True, method='powell')
print(f"Converged: {result1.converged}")
print(f"Time effect: β = {result1.params['TSVR_days']:.6f}")
print(f"SE: {result1.bse['TSVR_days']:.6f}")
print(f"z: {result1.tvalues['TSVR_days']:.3f}")
print(f"p: {result1.pvalues['TSVR_days']:.6f}")
print(f"AIC: {result1.aic:.2f}")
print(f"BIC: {result1.bic:.2f}")
print(f"\nRandom Effects:")
print(f"  Intercept Var: {result1.cov_re.iloc[0,0]:.6f}")
print(f"  Slope Var: {result1.cov_re.iloc[1,1]:.6f}")
print(f"  Covariance: {result1.cov_re.iloc[0,1]:.6f}")

# Model 2: Intercepts only (simpler)
print("\nModel 2: Random Intercepts Only")
print("-" * 60)
model2 = smf.mixedlm(
    "gamma ~ TSVR_days",
    data=data,
    groups=data['UID'],
    re_formula="~1"
)
result2 = model2.fit(reml=True)
print(f"Converged: {result2.converged}")
print(f"Time effect: β = {result2.params['TSVR_days']:.6f}")
print(f"SE: {result2.bse['TSVR_days']:.6f}")
print(f"z: {result2.tvalues['TSVR_days']:.3f}")
print(f"p: {result2.pvalues['TSVR_days']:.6f}")
print(f"AIC: {result2.aic:.2f}")
print(f"BIC: {result2.bic:.2f}")
print(f"\nRandom Effects:")
print(f"  Intercept Var: {result2.cov_re.iloc[0,0]:.6f}")

# Comparison
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
delta_aic = result1.aic - result2.aic
print(f"\nΔAIC (Slopes - Intercepts): {delta_aic:.2f}")
if abs(delta_aic) < 2:
    print("   → No substantial difference (|ΔAIC| < 2)")
    print("   → Simpler model (intercepts-only) preferred")
elif delta_aic < -2:
    print("   → Slopes model preferred (ΔAIC < -2)")
else:
    print("   → Intercepts model preferred (ΔAIC > 2)")

# Check if time effect robust
beta_diff = abs(result1.params['TSVR_days'] - result2.params['TSVR_days'])
beta_diff_pct = (beta_diff / abs(result2.params['TSVR_days'])) * 100
print(f"\nTime Effect Difference:")
print(f"  Slopes:     β = {result1.params['TSVR_days']:.6f}")
print(f"  Intercepts: β = {result2.params['TSVR_days']:.6f}")
print(f"  |Difference|: {beta_diff:.6f} ({beta_diff_pct:.1f}%)")

if beta_diff_pct < 5:
    print("   ✅ Time effect ROBUST (<5% difference)")
else:
    print("   ⚠️ Time effect differs between models")

# Check significance consistency
both_sig = (result1.pvalues['TSVR_days'] < 0.05) and (result2.pvalues['TSVR_days'] < 0.05)
print(f"\nSignificance Consistency:")
print(f"  Slopes:     p = {result1.pvalues['TSVR_days']:.4f} {'*' if result1.pvalues['TSVR_days'] < 0.05 else 'n.s.'}")
print(f"  Intercepts: p = {result2.pvalues['TSVR_days']:.4f} {'*' if result2.pvalues['TSVR_days'] < 0.05 else 'n.s.'}")

if both_sig:
    print("   ✅ SIGNIFICANT in BOTH models")
elif not both_sig:
    print("   ⚠️ Significance differs between models")
else:
    print("   ⚠️ Mixed significance results")

# Recommendation
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
if abs(delta_aic) < 2 and beta_diff_pct < 5 and both_sig:
    print("✅ FINDING ROBUST:")
    print("   - Time effect consistent across models")
    print("   - Random slopes variance negligible (boundary estimate)")
    print("   - Intercepts-only model adequate")
    print("   - CONCLUSION: Homogeneous decline rate confirmed")
else:
    print("⚠️ MODEL SENSITIVITY DETECTED:")
    print("   - Consider reporting both model results")

print("=" * 60)

# Save comparison
comparison = pd.DataFrame({
    'model': ['Random Slopes', 'Random Intercepts'],
    'converged': [result1.converged, result2.converged],
    'beta': [result1.params['TSVR_days'], result2.params['TSVR_days']],
    'se': [result1.bse['TSVR_days'], result2.bse['TSVR_days']],
    'z': [result1.tvalues['TSVR_days'], result2.tvalues['TSVR_days']],
    'p': [result1.pvalues['TSVR_days'], result2.pvalues['TSVR_days']],
    'AIC': [result1.aic, result2.aic],
    'BIC': [result1.bic, result2.bic]
})
comparison.to_csv('data/random_effects_comparison.csv', index=False)
print(f"\nComparison saved to: data/random_effects_comparison.csv")
