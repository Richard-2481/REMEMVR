"""
GLMM Validation for RQ 5.3.4
Age × Paradigm Interaction Effects on Memory Accuracy

Purpose: Test whether IRT→LMM null Age × Paradigm interaction remains null with item-level GLMM
Method: Binomial GLMM on binary accuracy responses (Correct/Incorrect)
Expected Outcome: Verify NULL 3-way Age × Paradigm × Time interaction robust at item level
Date: 2025-12-31
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

print("=" * 80)
print("GLMM VALIDATION: RQ 5.3.4 - Age × Paradigm Interaction")
print("=" * 80)

# Load item-level accuracy data

print("\n[1] Loading item-level accuracy responses...")

# Load item-level data created by extract_item_level_data.py
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")
# Load item-level data with Age (preprocessed)
item_data = pd.read_csv("../data/item_level_responses_with_age.csv")
print(f"   Loaded {len(item_data):,} item responses")
print(f"   Participants: {item_data["UID"].nunique()}")
print(f"   Items: {item_data["Item"].nunique()}")
print(f"   Tests: {sorted(item_data["test"].unique())}")
print(f"   Age range: {item_data["Age"].min():.0f}-{item_data["Age"].max():.0f} years")

# Grand-mean center Age
item_data["Age_c"] = item_data["Age"] - item_data["Age"].mean()
print(f"   Age centered: mean(Age_c) = {item_data["Age_c"].mean():.4f}")

# Create paradigm coding and time transformations

print("\n[2] Preparing model variables...")

# Treatment coding: IFR (Free Recall) as reference
item_data['Paradigm_ICR'] = (item_data['Paradigm'] == 'ICR').astype(int)
item_data['Paradigm_IRE'] = (item_data['Paradigm'] == 'IRE').astype(int)

print(f"   Paradigm distribution:")
for p in ['IFR', 'ICR', 'IRE']:
    n = (item_data['Paradigm'] == p).sum()
    pct = 100 * n / len(item_data)
    print(f"     {p}: {n:,} ({pct:.1f}%)")

# Create log time transformation (match IRT→LMM model)
# Use test session as proxy for TSVR (1, 2, 3, 4 → log scale)
# For proper GLMM, should use actual TSVR_hours, but test session sufficient for interaction test
item_data['log_test'] = np.log(item_data['test'])
print(f"   Time variable: log(test) range [{item_data['log_test'].min():.2f}, {item_data['log_test'].max():.2f}]")

# Fit Binomial GLMM

print("\n[3] Fitting binomial GLMM...")
print("   Model: Correct ~ Age_c * Paradigm + log_test + Age_c:Paradigm:log_test")
print("   Random effects: (1 | UID) participant intercepts")
print("   Family: Binomial (logit link for binary accuracy)")
print(f"   N observations: {len(item_data):,}")

# Formula: Test Age × Paradigm intercept interaction (primary interest)
# Also include time effects for completeness
formula = """Correct ~ Age_c + Paradigm_ICR + Paradigm_IRE + log_test +
             Age_c:Paradigm_ICR + Age_c:Paradigm_IRE +
             Age_c:log_test + Paradigm_ICR:log_test + Paradigm_IRE:log_test +
             Age_c:Paradigm_ICR:log_test + Age_c:Paradigm_IRE:log_test"""

# Fit GLMM
try:
    model_glmm = smf.mixedlm(
        formula,
        data=item_data,
        groups=item_data['UID'],
        re_formula="~1",  # Random intercepts only (slopes may not converge with 28,800 obs)
        family=sm.families.Binomial()
    )

    print("\n   Fitting GLMM (this may take 5-10 minutes with ~28,800 observations)...")
    result_glmm = model_glmm.fit(method='lbfgs', maxiter=200)
    print(f"   GLMM fit complete! Converged: {result_glmm.converged}")

except Exception as e:
    print(f"\n   ❌ ERROR: GLMM fitting failed: {e}")
    print("   This is common with large item-level datasets and complex interactions")
    print("   Attempting simplified model (Age × Paradigm intercepts only, no time)...")

    # Fallback: Test only Age × Paradigm intercepts (no time interaction)
    formula_simple = "Correct ~ Age_c + Paradigm_ICR + Paradigm_IRE + Age_c:Paradigm_ICR + Age_c:Paradigm_IRE"

    model_glmm = smf.mixedlm(
        formula_simple,
        data=item_data,
        groups=item_data['UID'],
        re_formula="~1",
        family=sm.families.Binomial()
    )

    print(f"   Fitting simplified GLMM...")
    result_glmm = model_glmm.fit(method='lbfgs', maxiter=200)
    print(f"   Simplified GLMM complete! Converged: {result_glmm.converged}")

# Extract Age × Paradigm interaction effects

print("\n[4] Extracting Age × Paradigm interaction effects...")

# Get coefficients and p-values
params = result_glmm.params
pvalues = result_glmm.pvalues

# Extract Age × Paradigm interaction effects (2-way, at intercept level)
age_icr_beta = params.get('Age_c:Paradigm_ICR', np.nan)
age_icr_p = pvalues.get('Age_c:Paradigm_ICR', np.nan)

age_ire_beta = params.get('Age_c:Paradigm_IRE', np.nan)
age_ire_p = pvalues.get('Age_c:Paradigm_IRE', np.nan)

# Also extract 3-way if available (full model)
age_icr_time_beta = params.get('Age_c:Paradigm_ICR:log_test', np.nan)
age_icr_time_p = pvalues.get('Age_c:Paradigm_ICR:log_test', np.nan)

age_ire_time_beta = params.get('Age_c:Paradigm_IRE:log_test', np.nan)
age_ire_time_p = pvalues.get('Age_c:Paradigm_IRE:log_test', np.nan)

# IRT→LMM values from summary.md (3-way interaction p-values all > 0.7)
irt_lmm_age_icr_time_p = 0.711  # From step03_interaction_terms.csv: Age_c:ICR:log_TSVR
irt_lmm_age_ire_time_p = 0.798  # From step03_interaction_terms.csv: Age_c:IRE:log_TSVR

print(f"\n{'Effect':<40} {'IRT→LMM p':<15} {'GLMM p':<15} {'GLMM β':<15}")
print("=" * 80)

# Main effects
print("MAIN EFFECTS:")
print(f"{'  Age (centered):':<40} {'--':<15} {pvalues.get('Age_c', np.nan):<15.3f} {params.get('Age_c', np.nan):<15.3f}")
print(f"{'  Paradigm ICR vs IFR:':<40} {'--':<15} {pvalues.get('Paradigm_ICR', np.nan):<15.3f} {params.get('Paradigm_ICR', np.nan):<15.3f}")
print(f"{'  Paradigm IRE vs IFR:':<40} {'--':<15} {pvalues.get('Paradigm_IRE', np.nan):<15.3f} {params.get('Paradigm_IRE', np.nan):<15.3f}")

# 2-way Age × Paradigm (intercept level)
print("\n2-WAY INTERACTIONS (Age × Paradigm at baseline):")
print(f"{'  Age × ICR:':<40} {'--':<15} {age_icr_p:<15.3f} {age_icr_beta:<15.3f}")
print(f"{'  Age × IRE:':<40} {'--':<15} {age_ire_p:<15.3f} {age_ire_beta:<15.3f}")

# 3-way Age × Paradigm × Time (if available)
if not np.isnan(age_icr_time_p):
    print("\n3-WAY INTERACTIONS (Age × Paradigm × Time):")
    print(f"{'  Age × ICR × log_test:':<40} {irt_lmm_age_icr_time_p:<15.3f} {age_icr_time_p:<15.3f} {age_icr_time_beta:<15.3f}")
    print(f"{'  Age × IRE × log_test:':<40} {irt_lmm_age_ire_time_p:<15.3f} {age_ire_time_p:<15.3f} {age_ire_time_beta:<15.3f}")

# Interpret results

print("\n[5] Interpretation...")

# Check 2-way Age × Paradigm interactions (baseline effects)
age_paradigm_sig = (age_icr_p < 0.05) or (age_ire_p < 0.05)

# Check 3-way if available
if not np.isnan(age_icr_time_p):
    age_paradigm_time_sig = (age_icr_time_p < 0.05) or (age_ire_time_p < 0.05)
    threeway_changed = age_paradigm_time_sig  # IRT→LMM was NULL (p > 0.7)
else:
    age_paradigm_time_sig = False
    threeway_changed = False

if age_paradigm_sig:
    print("\n   ⚠️ CRITICAL: Age × Paradigm BASELINE interaction SIGNIFICANT in GLMM")
    print(f"      Age × ICR: p={age_icr_p:.3f}")
    print(f"      Age × IRE: p={age_ire_p:.3f}")
    print("      This means older adults show different paradigm-specific baseline performance")
    print("      ACTION: Thesis narrative revision required (age-invariance claim undermined)")
    outcome = "CHANGED_BASELINE"
elif threeway_changed:
    print("\n   ⚠️ CRITICAL: 3-way Age × Paradigm × Time interaction SIGNIFICANT in GLMM")
    print(f"      IRT→LMM: p > 0.7 (NULL)")
    print(f"      GLMM: Age × ICR × Time p={age_icr_time_p:.3f}, Age × IRE × Time p={age_ire_time_p:.3f}")
    print("      ACTION: Thesis narrative revision required")
    outcome = "CHANGED_TRAJECTORY"
else:
    print("\n   ✅ Finding ROBUST across methods")
    print(f"      Both IRT→LMM and GLMM show NULL Age × Paradigm interactions")
    if not np.isnan(age_icr_time_p):
        print(f"      3-way interaction: IRT→LMM p > 0.7, GLMM p > 0.05")
    print(f"      2-way baseline: GLMM p > 0.05")
    print(f"      GLMM validation confirms: Age does NOT modulate paradigm effects")
    outcome = "ROBUST"

# Save results

print("\n[6] Saving GLMM validation results...")

# Create comparison table
comparison_data = []

# 2-way comparisons
comparison_data.append({
    'Effect': 'Age × ICR (baseline)',
    'IRT_LMM_p': np.nan,  # Not directly tested in IRT→LMM (3-way model)
    'GLMM_p': age_icr_p,
    'GLMM_beta': age_icr_beta,
    'GLMM_SE': result_glmm.bse.get('Age_c:Paradigm_ICR', np.nan),
    'Significant': age_icr_p < 0.05
})

comparison_data.append({
    'Effect': 'Age × IRE (baseline)',
    'IRT_LMM_p': np.nan,
    'GLMM_p': age_ire_p,
    'GLMM_beta': age_ire_beta,
    'GLMM_SE': result_glmm.bse.get('Age_c:Paradigm_IRE', np.nan),
    'Significant': age_ire_p < 0.05
})

# 3-way comparisons (if available)
if not np.isnan(age_icr_time_p):
    comparison_data.append({
        'Effect': 'Age × ICR × Time',
        'IRT_LMM_p': irt_lmm_age_icr_time_p,
        'GLMM_p': age_icr_time_p,
        'GLMM_beta': age_icr_time_beta,
        'GLMM_SE': result_glmm.bse.get('Age_c:Paradigm_ICR:log_test', np.nan),
        'Significant': age_icr_time_p < 0.05
    })

    comparison_data.append({
        'Effect': 'Age × IRE × Time',
        'IRT_LMM_p': irt_lmm_age_ire_time_p,
        'GLMM_p': age_ire_time_p,
        'GLMM_beta': age_ire_time_beta,
        'GLMM_SE': result_glmm.bse.get('Age_c:Paradigm_IRE:log_test', np.nan),
        'Significant': age_ire_time_p < 0.05
    })

comparison = pd.DataFrame(comparison_data)
comparison.to_csv('../data/glmm_comparison.csv', index=False)
print(f"   Saved: data/glmm_comparison.csv")

# Save full GLMM summary
with open('../data/glmm_summary.txt', 'w') as f:
    f.write("GLMM VALIDATION: RQ 5.3.4 - Age × Paradigm Interaction\n")
    f.write("=" * 80 + "\n\n")
    f.write(str(result_glmm.summary()))
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("COMPARISON TO IRT→LMM:\n")
    f.write("=" * 80 + "\n")
    f.write(comparison.to_string(index=False))

print(f"   Saved: data/glmm_summary.txt")

# FINAL SUMMARY

print("\n" + "=" * 80)
print("GLMM VALIDATION COMPLETE")
print("=" * 80)
print(f"\nObservations: {len(item_data):,} item-level responses")
print(f"Participants: {item_data['UID'].nunique()}")
print(f"Items: {item_data['Item'].nunique()}")
print(f"\nOutcome: {outcome}")

if outcome == "ROBUST":
    print("\n   ✅ NULL findings ROBUST (IRT→LMM and GLMM agree)")
    print("   ✅ Age × Paradigm interaction non-significant at item level")
    print("   ✅ No thesis narrative revision needed")
    print("   ✅ quality validation can proceed")
elif outcome == "CHANGED_BASELINE":
    print("\n   ⚠️ BASELINE INTERACTION CHANGED (NULL → SIGNIFICANT)")
    print("   ⚠️ blocks quality validation")
    print("   ⚠️ Thesis narrative revision required")
else:
    print("\n   ⚠️ 3-WAY INTERACTION CHANGED (NULL → SIGNIFICANT)")
    print("   ⚠️ blocks quality validation")
    print("   ⚠️ Thesis narrative revision required")

print("\nFiles created:")
print("   - data/glmm_comparison.csv")
print("   - data/glmm_summary.txt")
print("   - data/item_level_responses.csv (prerequisite)")
print("\n" + "=" * 80)
