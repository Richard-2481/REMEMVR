"""
Re-fit LMM with Random Slopes for RQ 6.5.1
CRITICAL UPDATE per random slopes comparison

Random slopes model (ΔAIC = 199.14) is vastly superior to intercepts-only.
Must re-fit LMM and check if schema × time interaction conclusions change.
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from pathlib import Path

# Paths
base_path = Path("/home/etai/projects/REMEMVR/results/ch6/6.5.1")
data_path = base_path / "data"
output_path = base_path / "data"

print("="*80)
print("LMM with Random Slopes - RQ 6.5.1")
print("Re-fitting with (1 + log_TSVR | UID)")
print("="*80)

# Load LMM input data
lmm_input = pd.read_csv(data_path / "step04_lmm_input.csv")
print(f"\nLoaded {len(lmm_input)} observations")

# Fit LMM with random slopes
print("\n" + "="*80)
print("FITTING LMM WITH RANDOM SLOPES")
print("="*80)
print("Formula: theta ~ C(congruence) * log_TSVR")
print("Random effects: (1 + log_TSVR | UID)")

model = smf.mixedlm(
    "theta ~ C(congruence) * log_TSVR",
    data=lmm_input,
    groups=lmm_input['UID'],
    re_formula="~log_TSVR"  # Random intercept + slope
)

result = model.fit(method='lbfgs', reml=False)

print(f"\nConverged: {result.converged}")
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")
print(f"Log-Likelihood: {result.llf:.4f}")

print("\n" + "="*80)
print("FIXED EFFECTS")
print("="*80)
print(result.summary().tables[1])

# Extract fixed effects for comparison
fixed_effects = pd.DataFrame({
    'effect': result.params.index,
    'beta': result.params.values,
    'se': result.bse.values,
    'z': result.tvalues.values,
    'p_value': result.pvalues.values,
    'ci_lower': result.conf_int()[0].values,
    'ci_upper': result.conf_int()[1].values
})

print("\n" + "="*80)
print("PRIMARY HYPOTHESIS TEST (Schema × Time Interaction)")
print("="*80)

# Extract interaction terms
congruent_interaction = fixed_effects[fixed_effects['effect'] == 'C(congruence)[T.Congruent]:log_TSVR']
incongruent_interaction = fixed_effects[fixed_effects['effect'] == 'C(congruence)[T.Incongruent]:log_TSVR']

if len(congruent_interaction) > 0:
    print(f"\nCongruent × Time:")
    print(f"  β = {congruent_interaction['beta'].values[0]:.4f}")
    print(f"  SE = {congruent_interaction['se'].values[0]:.4f}")
    print(f"  p = {congruent_interaction['p_value'].values[0]:.3f}")
    print(f"  95% CI = [{congruent_interaction['ci_lower'].values[0]:.4f}, {congruent_interaction['ci_upper'].values[0]:.4f}]")
    congruent_p = congruent_interaction['p_value'].values[0]
else:
    congruent_p = np.nan

if len(incongruent_interaction) > 0:
    print(f"\nIncongruent × Time:")
    print(f"  β = {incongruent_interaction['beta'].values[0]:.4f}")
    print(f"  SE = {incongruent_interaction['se'].values[0]:.4f}")
    print(f"  p = {incongruent_interaction['p_value'].values[0]:.3f}")
    print(f"  95% CI = [{incongruent_interaction['ci_lower'].values[0]:.4f}, {incongruent_interaction['ci_upper'].values[0]:.4f}]")
    incongruent_p = incongruent_interaction['p_value'].values[0]
else:
    incongruent_p = np.nan

# Decision based on p-values
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if (not pd.isna(congruent_p) and congruent_p < 0.05) or (not pd.isna(incongruent_p) and incongruent_p < 0.05):
    print("\n❌ INTERACTION SIGNIFICANT (p < 0.05)")
    print("Schema congruence AFFECTS confidence decline rate")
    print("\n🔴 CRITICAL: This CONTRADICTS intercepts-only model (p > 0.3)")
    print("Random slopes model reveals schema × time interaction previously missed!")
    conclusion = "SIGNIFICANT interaction"
else:
    print("\n✅ INTERACTION NON-SIGNIFICANT (p > 0.05)")
    print("Schema congruence does NOT affect confidence decline rate")
    print("\n✅ Conclusion UNCHANGED from intercepts-only model")
    print("NULL finding robust to random effects specification")
    conclusion = "NULL interaction (robust)"

print("\n" + "="*80)
print("RANDOM EFFECTS")
print("="*80)
print("\nVariance-Covariance Matrix:")
print(result.cov_re)
print(f"\nIntercept variance: {result.cov_re.iloc[0,0]:.4f}")
print(f"Slope variance: {result.cov_re.iloc[1,1]:.4f}")
print(f"Slope SD: {np.sqrt(result.cov_re.iloc[1,1]):.4f}")
slope_intercept_corr = result.cov_re.iloc[0,1] / np.sqrt(result.cov_re.iloc[0,0] * result.cov_re.iloc[1,1])
print(f"Intercept-Slope correlation: {slope_intercept_corr:.3f}")

# Save updated results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

fixed_effects.to_csv(output_path / "lmm_random_slopes_fixed_effects.csv", index=False)
print(f"✅ Saved: {output_path / 'lmm_random_slopes_fixed_effects.csv'}")

# Save full summary
summary_path = output_path / "lmm_random_slopes_summary.txt"
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LMM WITH RANDOM SLOPES - RQ 6.5.1\n")
    f.write("="*80 + "\n\n")
    f.write("Formula: theta ~ C(congruence) * log_TSVR\n")
    f.write("Random effects: (1 + log_TSVR | UID)\n\n")
    f.write(f"Converged: {result.converged}\n")
    f.write(f"AIC: {result.aic:.2f}\n")
    f.write(f"BIC: {result.bic:.2f}\n")
    f.write(f"Log-Likelihood: {result.llf:.4f}\n\n")
    f.write("-"*80 + "\n")
    f.write("FIXED EFFECTS\n")
    f.write("-"*80 + "\n\n")
    f.write(str(result.summary().tables[1]))
    f.write("\n\n")
    f.write("-"*80 + "\n")
    f.write("RANDOM EFFECTS\n")
    f.write("-"*80 + "\n\n")
    f.write("Variance-Covariance Matrix:\n")
    f.write(str(result.cov_re))
    f.write(f"\n\nIntercept variance: {result.cov_re.iloc[0,0]:.4f}\n")
    f.write(f"Slope variance: {result.cov_re.iloc[1,1]:.4f}\n")
    f.write(f"Slope SD: {np.sqrt(result.cov_re.iloc[1,1]):.4f}\n")
    f.write(f"Intercept-Slope correlation: {slope_intercept_corr:.3f}\n\n")
    f.write("-"*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n\n")
    f.write(f"Conclusion: {conclusion}\n\n")
    if conclusion.startswith("SIGNIFICANT"):
        f.write("🔴 CRITICAL FINDING:\n")
        f.write("Random slopes model reveals schema × time interaction that was\n")
        f.write("MISSED by intercepts-only model. Incorrect random effects structure\n")
        f.write("led to TYPE II error (false NULL).\n\n")
        f.write("RECOMMENDATION: Update all RQ results with random slopes model.\n")
    else:
        f.write("✅ NULL FINDING ROBUST:\n")
        f.write("Schema × time interaction remains NON-SIGNIFICANT even with\n")
        f.write("random slopes. Original conclusion (intercepts-only) was correct,\n")
        f.write("but now has stronger empirical support (tested for heterogeneity).\n\n")
        f.write("RECOMMENDATION: Document random slopes test in validation.md.\n")
        f.write("Original results valid, no need to regenerate.\n")

print(f"✅ Saved: {summary_path}")

# Comparison with intercepts-only model
print("\n" + "="*80)
print("COMPARISON WITH INTERCEPTS-ONLY MODEL")
print("="*80)

# Load original fixed effects (if available)
try:
    original_fx = pd.read_csv(data_path / "step05_lmm_coefficients.csv")
    print("\nLoaded original intercepts-only fixed effects")

    print("\nCongruent × Time interaction:")
    orig_congruent = original_fx[original_fx['term'].str.contains('Congruent.*log_TSVR', regex=True)]
    if len(orig_congruent) > 0:
        print(f"  Intercepts-only: p = {orig_congruent['p_value'].values[0]:.3f}")
    if len(congruent_interaction) > 0:
        print(f"  Random slopes:   p = {congruent_p:.3f}")

    print("\nIncongruent × Time interaction:")
    orig_incongruent = original_fx[original_fx['term'].str.contains('Incongruent.*log_TSVR', regex=True)]
    if len(orig_incongruent) > 0:
        print(f"  Intercepts-only: p = {orig_incongruent['p_value'].values[0]:.3f}")
    if len(incongruent_interaction) > 0:
        print(f"  Random slopes:   p = {incongruent_p:.3f}")

except FileNotFoundError:
    print("\nOriginal coefficients file not found (step05_lmm_coefficients.csv)")

print("\n" + "="*80)
print("RANDOM SLOPES LMM COMPLETE")
print("="*80)
print(f"\nConclusion: {conclusion}")
print("\nFiles created:")
print(f"  - {output_path / 'lmm_random_slopes_fixed_effects.csv'}")
print(f"  - {summary_path}")
print("\nNext step: Compare conclusions, update validation.md if needed")
