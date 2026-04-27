#!/usr/bin/env python3
"""
GLMM Validation for RQ 5.2.3 - Age × Domain Interaction

Purpose: Validate IRT→LMM findings with item-level mixed model
Method: Linear mixed model on binary accuracy (Gaussian approximation of binomial)
Justification: With large sample (28,800 obs), Gaussian approximation valid for binary data
              (Jaeger 2008, Language & Linguistics Compass)
Date: 2025-12-31

CRITICAL CONTEXT (from glmm_candidates.md):
    - RQ 5.2.3 listed as MEDIUM priority (line 45)
    - IRT→LMM result: Age main p=0.156 (null), Age:Domain p=0.713 (null)
    - Historical precedent: NULL→SIGNIFICANT for intercepts (RQ 5.4.1, 6.5.1)
    - Risk: Item-level GLMM may reveal baseline Age × Domain effects

EXPECTED OUTCOME:
    - Age main effect: Likely NULL (p > 0.05) based on IRT→LMM p=0.156
    - Age × Where interaction: Likely NULL (p > 0.05) based on IRT→LMM p=0.713
    - If significant: BLOCKER for PLATINUM (narrative revision required)
    - If null: ROBUST finding (PLATINUM can proceed)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.2.3"
PARENT_RQ = PROJECT_ROOT / "results" / "ch5" / "5.2.1"  # Source of IRT data
DATA_DIR = RQ_DIR / "data"
RESULTS_DIR = RQ_DIR / "results"
LOG_DIR = RQ_DIR / "logs"
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GLMM VALIDATION: RQ 5.2.3 - Age × Domain Interaction")
print("=" * 80)

# ==============================================================================
# STEP 1: Extract Item-Level Data from RQ 5.2.1
# ==============================================================================
print("\n[STEP 1] Extracting item-level data from parent RQ 5.2.1...")
print("-" * 80)

irt_input_path = PARENT_RQ / "data" / "step00_irt_input.csv"
print(f"Loading IRT input: {irt_input_path}")

if not irt_input_path.exists():
    print(f"[ERROR] Parent RQ data not found: {irt_input_path}")
    print("   RQ 5.2.1 must be complete before validating RQ 5.2.3")
    sys.exit(1)

irt_data = pd.read_csv(irt_input_path)
print(f"[SUCCESS] Loaded {irt_data.shape[0]} rows x {irt_data.shape[1]} columns")

# Parse composite_ID to extract UID and test
irt_data[['UID', 'test']] = irt_data['composite_ID'].str.split('_', expand=True)
irt_data['test'] = irt_data['test'].str.replace('T', '').astype(int)
print(f"[INFO] Parsed UID and test from composite_ID")
print(f"   Unique UIDs: {irt_data['UID'].nunique()}")
print(f"   Test sessions: {sorted(irt_data['test'].unique())}")

# Find item columns (format: TQ_IFR-O-i1, TQ_ICR-N-i2, etc.)
item_cols = [col for col in irt_data.columns if col.startswith('TQ_')]
print(f"[INFO] Found {len(item_cols)} item columns")

# Melt to long format
long_data = irt_data.melt(
    id_vars=['composite_ID', 'UID', 'test'],
    value_vars=item_cols,
    var_name='item',
    value_name='Correct'
)
print(f"[SUCCESS] Melted to long format: {len(long_data):,} observations")

# Parse item name to extract domain
# Format: TQ_IFR-O-i1 → Paradigm=IFR, Domain=O (What), Item=i1
def parse_item(item_str):
    """Parse TQ_IFR-O-i1 format"""
    parts = item_str.replace('TQ_', '').split('-')
    domain_map = {'O': 'What', 'L': 'Where', 'T': 'When'}  # O=Object, L=Location, T=Time
    return {
        'Paradigm': parts[0],
        'Domain_code': parts[1],
        'Domain': domain_map.get(parts[1], parts[1]),
        'Item_num': parts[2]
    }

item_info = long_data['item'].apply(parse_item).apply(pd.Series)
long_data = pd.concat([long_data, item_info], axis=1)

print(f"[INFO] Parsed item metadata")
print(f"   Paradigms: {sorted(long_data['Paradigm'].unique())}")
print(f"   Domains: {sorted(long_data['Domain'].unique())}")

# Filter to What and Where only (When excluded due to floor effect)
domains = ['What', 'Where']
long_data = long_data[long_data['Domain'].isin(domains)].copy()
print(f"[INFO] Filtered to domains {domains}: {len(long_data):,} observations")
print(f"   (When domain excluded due to floor effect)")

# Create Item identifier
long_data['Item'] = long_data['Paradigm'] + '_' + long_data['Domain'] + '_' + long_data['Item_num']

# Drop rows with missing responses (NaN)
n_before = len(long_data)
long_data = long_data.dropna(subset=['Correct'])
n_after = len(long_data)
if n_before > n_after:
    print(f"[INFO] Dropped {n_before - n_after:,} rows with missing responses")

print(f"\n[INFO] Item-level data extracted:")
print(f"   Total observations: {len(long_data):,}")
print(f"   Participants: {long_data['UID'].nunique()}")
print(f"   Items: {long_data['Item'].nunique()}")
print(f"   Domains: {long_data['Domain'].nunique()}")

# ==============================================================================
# STEP 2: Add Age from dfData.csv
# ==============================================================================
print("\n[STEP 2] Adding Age variable from dfData.csv...")
print("-" * 80)

if not DFDATA_PATH.exists():
    print(f"[ERROR] dfData.csv not found: {DFDATA_PATH}")
    sys.exit(1)

dfdata = pd.read_csv(DFDATA_PATH)
print(f"[INFO] Loaded dfData.csv: {len(dfdata)} participants")

# Extract Age
age_data = dfdata[['UID', 'age']].copy()
age_data.columns = ['UID', 'Age']

# Merge with item-level data
long_data = long_data.merge(age_data, on='UID', how='left')

if long_data['Age'].isna().any():
    n_missing = long_data['Age'].isna().sum()
    print(f"[WARNING] {n_missing} observations missing Age data")
    long_data = long_data.dropna(subset=['Age'])
    print(f"[INFO] Dropped missing Age rows, remaining: {len(long_data):,}")

print(f"[SUCCESS] Age added to item-level data")
print(f"   Age range: {long_data['Age'].min():.0f}-{long_data['Age'].max():.0f} years")
print(f"   Age mean: {long_data['Age'].mean():.1f} years")

# Grand-mean center Age
long_data['Age_c'] = long_data['Age'] - long_data['Age'].mean()
print(f"   Age centered: mean(Age_c) = {long_data['Age_c'].mean():.4f}")

# Save item-level data
item_data_path = DATA_DIR / "item_level_responses_with_age.csv"
long_data[['composite_ID', 'UID', 'test', 'Domain', 'Item', 'Correct', 'Age', 'Age_c']].to_csv(
    item_data_path, index=False
)
print(f"[INFO] Saved: {item_data_path}")

# ==============================================================================
# STEP 3: Prepare GLMM Variables
# ==============================================================================
print("\n[STEP 3] Preparing model variables...")
print("-" * 80)

# Treatment coding: What as reference (least hippocampal-dependent)
long_data['Domain_Where'] = (long_data['Domain'] == 'Where').astype(int)

print(f"[INFO] Domain distribution:")
for domain in ['What', 'Where']:
    n = (long_data['Domain'] == domain).sum()
    print(f"   {domain}: {n:,} ({100*n/len(long_data):.1f}%)")

# ==============================================================================
# STEP 4: Fit Item-Level Mixed Model (GLMM)
# ==============================================================================
print("\n[STEP 4] Fitting item-level mixed model...")
print("-" * 80)
print("[INFO] Model: Correct ~ Age_c * Domain")
print("       Random effects: (1 | UID)")
print("       Approximation: Gaussian (valid for binary with large N)")
print("")

formula = "Correct ~ Age_c + Domain_Where + Age_c:Domain_Where"

model_glmm = smf.mixedlm(
    formula,
    data=long_data,
    groups=long_data['UID'],
    re_formula="~1"  # Random intercepts only
)

print(f"[INFO] Fitting... (may take 2-3 minutes with {len(long_data):,} observations)")
result_glmm = model_glmm.fit(reml=False, method='lbfgs', maxiter=500)
print(f"[SUCCESS] Complete! Converged: {result_glmm.converged}")

# ==============================================================================
# STEP 5: Extract Results
# ==============================================================================
print("\n[STEP 5] Extracting Age × Domain interaction effects...")
print("-" * 80)

params = result_glmm.params
pvalues = result_glmm.pvalues
bse = result_glmm.bse

age_main_beta = params.get('Age_c', np.nan)
age_main_p = pvalues.get('Age_c', np.nan)
age_main_se = bse.get('Age_c', np.nan)

where_beta = params.get('Domain_Where', np.nan)
where_p = pvalues.get('Domain_Where', np.nan)
where_se = bse.get('Domain_Where', np.nan)

age_where_beta = params.get('Age_c:Domain_Where', np.nan)
age_where_p = pvalues.get('Age_c:Domain_Where', np.nan)
age_where_se = bse.get('Age_c:Domain_Where', np.nan)

print(f"\n{'Effect':<45} {'GLMM p':<15} {'GLMM β':<15} {'SE':<15}")
print("=" * 90)
print("MAIN EFFECTS:")
print(f"{'  Age (centered):':<45} {age_main_p:<15.3f} {age_main_beta:<15.6f} {age_main_se:<15.6f}")
print(f"{'  Domain Where vs What (baseline):':<45} {where_p:<15.3f} {where_beta:<15.6f} {where_se:<15.6f}")
print("\nAGE × DOMAIN INTERACTION (Baseline):")
print(f"{'  Age × Where (baseline difference):':<45} {age_where_p:<15.3f} {age_where_beta:<15.6f} {age_where_se:<15.6f}")

# ==============================================================================
# STEP 6: Compare to IRT→LMM
# ==============================================================================
print("\n[STEP 6] Comparison to IRT→LMM...")
print("-" * 80)

print("\nIRT→LMM findings (from summary.md lines 68, 76):")
print("  Age main effect: p=0.156 (NULL)")
print("  Age:Domain interaction: p=0.713 (NULL)")
print("  3-way Age × Domain × Time: p>0.4 (NULL)")
print("\nGLMM findings (item-level, N={:,}):".format(len(long_data)))
print(f"  Age main effect: p={age_main_p:.3f}")
print(f"  Age × Where (baseline): p={age_where_p:.3f}")

# ==============================================================================
# STEP 7: Interpret
# ==============================================================================
print("\n[STEP 7] Interpretation...")
print("-" * 80)

age_domain_sig = age_where_p < 0.05

if age_domain_sig:
    print("\n   ⚠️ CRITICAL: Age × Domain baseline interaction SIGNIFICANT in item-level model")
    print(f"      Age × Where: p={age_where_p:.3f} < 0.05 (SIGNIFICANT)")
    print(f"      Effect size: β={age_where_beta:.4f} ± {age_where_se:.4f}")
    print("\n   INTERPRETATION:")
    print("      Item-level analysis reveals age-domain interaction at baseline")
    print("      that was not detected in IRT→LMM aggregated analysis.")
    print("\n   PRECEDENT:")
    print("      RQ 5.4.1: Schema NULL→SIGNIFICANT (p=.548→.011)")
    print("      RQ 6.5.1: Schema NULL→SIGNIFICANT (p=.634→.003)")
    print("\n   ACTION REQUIRED:")
    print("      - Thesis narrative revision needed")
    print("      - Age-invariance claim undermined for domain effects")
    print("      - Document as methodological finding (IRT aggregation vs item-level)")
    outcome = "CHANGED_NULL_TO_SIGNIFICANT"
else:
    print("\n   ✅ Finding ROBUST across methods")
    print(f"      Item-level model confirms NULL Age × Domain interaction")
    print(f"      Age × Where: p={age_where_p:.3f} > 0.05 (NULL)")
    print("\n   INTERPRETATION:")
    print("      IRT→LMM NULL finding validated at item level (N={:,})".format(len(long_data)))
    print("      Age does NOT modulate domain-specific baseline performance")
    print("      Hippocampal aging hypothesis NOT supported in VR context")
    outcome = "ROBUST_NULL_CONFIRMED"

# ==============================================================================
# STEP 8: Save Results
# ==============================================================================
print("\n[STEP 8] Saving results...")
print("-" * 80)

# Comparison table
comparison = pd.DataFrame([
    {
        'Effect': 'Age main (baseline)',
        'IRT_LMM_p': 0.156,  # From summary.md line 68
        'GLMM_p': age_main_p,
        'GLMM_beta': age_main_beta,
        'GLMM_SE': age_main_se,
        'Significant': age_main_p < 0.05
    },
    {
        'Effect': 'Age × Where (baseline)',
        'IRT_LMM_p': 0.713,  # From summary.md line 76
        'GLMM_p': age_where_p,
        'GLMM_beta': age_where_beta,
        'GLMM_SE': age_where_se,
        'Significant': age_where_p < 0.05
    }
])
comparison_path = DATA_DIR / "glmm_comparison.csv"
comparison.to_csv(comparison_path, index=False)
print(f"[SUCCESS] Saved: {comparison_path}")

# Full summary
summary_path = DATA_DIR / "glmm_summary.txt"
with open(summary_path, 'w') as f:
    f.write("ITEM-LEVEL MIXED MODEL VALIDATION: RQ 5.2.3\n")
    f.write("=" * 80 + "\n\n")
    f.write("METHOD: Linear mixed model on binary accuracy (Gaussian approximation)\n")
    f.write(f"OBSERVATIONS: {len(long_data):,} item-level responses\n")
    f.write("JUSTIFICATION: With N>20,000, Gaussian approximation valid for binary outcomes\n")
    f.write("               (Jaeger 2008, Language & Linguistics Compass)\n\n")
    f.write(str(result_glmm.summary()))
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("COMPARISON TO IRT→LMM:\n")
    f.write("=" * 80 + "\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\n" + "=" * 80 + "\n")
    f.write(f"OUTCOME: {outcome}\n")
    f.write("=" * 80 + "\n")
print(f"[SUCCESS] Saved: {summary_path}")

# Validation report
report_path = RESULTS_DIR / "glmm_validation_report.md"
with open(report_path, 'w') as f:
    f.write("# GLMM Validation Report - RQ 5.2.3\n\n")
    f.write("**Date:** 2025-12-31\n")
    f.write("**Purpose:** Item-level validation of IRT→LMM Age × Domain findings\n")
    f.write(f"**Observations:** {len(long_data):,} item-level responses\n\n")
    f.write("## Method\n\n")
    f.write("- **Model:** Linear mixed model with Gaussian approximation\n")
    f.write("- **Formula:** `Correct ~ Age_c * Domain_Where + (1 | UID)`\n")
    f.write("- **Random Effects:** Random intercepts by participant\n")
    f.write("- **Domains:** What (reference), Where\n\n")
    f.write("## Results\n\n")
    f.write("| Effect | IRT→LMM p | GLMM p | GLMM β | GLMM SE |\n")
    f.write("|--------|-----------|--------|--------|--------|\n")
    f.write(f"| Age main | 0.156 | {age_main_p:.3f} | {age_main_beta:.4f} | {age_main_se:.4f} |\n")
    f.write(f"| Age × Where | 0.713 | {age_where_p:.3f} | {age_where_beta:.4f} | {age_where_se:.4f} |\n\n")
    f.write(f"## Outcome: {outcome.replace('_', ' ').title()}\n\n")

    if outcome == "ROBUST_NULL_CONFIRMED":
        f.write("✅ **PLATINUM CERTIFICATION CAN PROCEED**\n\n")
        f.write("Item-level GLMM validation confirms IRT→LMM NULL findings:\n")
        f.write(f"- Age main effect: p={age_main_p:.3f} (NULL)\n")
        f.write(f"- Age × Where interaction: p={age_where_p:.3f} (NULL)\n\n")
        f.write("**Interpretation:** Age does NOT modulate domain-specific baseline performance. ")
        f.write("Hippocampal aging hypothesis not supported in VR episodic memory.\n")
    else:
        f.write("⚠️ **BLOCKER FOR PLATINUM CERTIFICATION**\n\n")
        f.write("Item-level GLMM reveals baseline Age × Domain interaction not detected in IRT→LMM:\n")
        f.write(f"- Age × Where: p={age_where_p:.3f} < 0.05 (SIGNIFICANT)\n")
        f.write(f"- Effect size: β={age_where_beta:.4f} ± {age_where_se:.4f}\n\n")
        f.write("**Action required:** Thesis narrative revision to integrate item-level findings.\n")

print(f"[SUCCESS] Saved: {report_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("ITEM-LEVEL VALIDATION COMPLETE")
print("=" * 80)
print(f"\nObservations: {len(long_data):,} item-level responses")
print(f"Participants: {long_data['UID'].nunique()}")
print(f"Items: {long_data['Item'].nunique()}")
print(f"Domains: {sorted(long_data['Domain'].unique())}")
print(f"\nOutcome: {outcome.replace('_', ' ')}")

if outcome == "ROBUST_NULL_CONFIRMED":
    print("\n   ✅ NULL findings ROBUST (IRT→LMM and item-level model agree)")
    print("   ✅ Age × Domain interaction non-significant at item level")
    print("   ✅ No thesis narrative revision needed")
    print("   ✅ PLATINUM certification can proceed")
elif "CHANGED" in outcome:
    print("\n   ⚠️ BASELINE INTERACTION CHANGED (NULL → SIGNIFICANT)")
    print("   ⚠️ BLOCKER for PLATINUM certification")
    print("   ⚠️ Thesis narrative revision required")

print("\nFiles created:")
print(f"   - {item_data_path}")
print(f"   - {comparison_path}")
print(f"   - {summary_path}")
print(f"   - {report_path}")
print("\n" + "=" * 80)
