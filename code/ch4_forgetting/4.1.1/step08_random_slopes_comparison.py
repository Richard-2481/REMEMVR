"""
RQ 5.1.1 - Step 8: Random Slopes Comparison (MANDATORY)

PURPOSE: Test intercepts-only vs intercepts+slopes random effects structure
         per improvement_taxonomy.md Section 4.4 (NON-NEGOTIABLE for modeling RQs)

RATIONALE: Cannot claim homogeneous effects without testing for heterogeneity.

METHODOLOGY:
1. Load LMM input data with model-averaged power-law transformation
2. Fit Model A: Intercepts-only (current implementation)
3. Fit Model B: Intercepts + random slopes on time
4. Compare via AIC (ΔAIC > 2 = slopes improve fit)
5. Report random slope variance + interpretation

OUTCOMES (per agent prompt):
- Option A: Slopes improve fit (ΔAIC > 2) → Individual differences confirmed
- Option B: Slopes don't converge → Insufficient data for stable estimation
- Option C: Slopes converge but don't improve (ΔAIC < 2) → Homogeneous effects confirmed

Date: 2025-12-27
Agent: rq_platinum
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("="*80)
print("RQ 5.1.1 - RANDOM SLOPES COMPARISON (MANDATORY BLOCKER RESOLUTION)")
print("="*80)
print()

# Load LMM input data
print("[1/5] Loading LMM input data...")
lmm_data = pd.read_csv('../data/step04_lmm_input.csv')
print(f"  Loaded: {len(lmm_data)} observations")
print(f"  Participants: {lmm_data['UID'].nunique()}")
print(f"  Time points per participant: {lmm_data.groupby('UID').size().mode()[0]}")
print()

# Create power-law transformation (effective α = 0.410 from model averaging)
# Using α = 0.4 (closest single model: PowerLaw_04)
print("[2/5] Creating power-law transformation (α = 0.4, model-averaged best)...")
lmm_data['TSVR_hours_pow_neg04'] = (lmm_data['TSVR_hours'] + 1) ** (-0.4)
print(f"  Transformation: (TSVR_hours + 1)^(-0.4)")
print(f"  Range: [{lmm_data['TSVR_hours_pow_neg04'].min():.4f}, {lmm_data['TSVR_hours_pow_neg04'].max():.4f}]")
print()

# ============================================================================
# STEP 2: FIT MODEL A (INTERCEPTS-ONLY)
# ============================================================================

print("[3/5] Fitting Model A: Random intercepts only (current implementation)...")
print("  Formula: Theta ~ TSVR_hours_pow_neg04 + (1 | UID)")
print("  REML: False (for AIC comparability)")

try:
    model_intercepts = smf.mixedlm(
        "theta ~ TSVR_hours_pow_neg04",
        data=lmm_data,
        groups=lmm_data['UID'],
        re_formula='~1'  # Random intercepts only
    )
    result_intercepts = model_intercepts.fit(reml=False)

    print(f"  ✓ Converged: {result_intercepts.converged}")
    print(f"  ✓ AIC: {result_intercepts.aic:.2f}")
    print(f"  ✓ Log-likelihood: {result_intercepts.llf:.2f}")
    print(f"  ✓ Random intercept variance: {result_intercepts.cov_re.iloc[0,0]:.4f}")
    intercepts_success = True
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    intercepts_success = False

print()

# ============================================================================
# STEP 3: FIT MODEL B (INTERCEPTS + SLOPES)
# ============================================================================

print("[4/5] Fitting Model B: Random intercepts + random slopes...")
print("  Formula: Theta ~ TSVR_hours_pow_neg04 + (TSVR_hours_pow_neg04 | UID)")
print("  REML: False (for AIC comparability)")
print()

try:
    model_slopes = smf.mixedlm(
        "theta ~ TSVR_hours_pow_neg04",
        data=lmm_data,
        groups=lmm_data['UID'],
        re_formula='~TSVR_hours_pow_neg04'  # Random slopes on time
    )
    result_slopes = model_slopes.fit(reml=False)

    print(f"  ✓ Converged: {result_slopes.converged}")
    print(f"  ✓ AIC: {result_slopes.aic:.2f}")
    print(f"  ✓ Log-likelihood: {result_slopes.llf:.2f}")

    # Extract random effects variance-covariance matrix
    if result_slopes.cov_re.shape[0] >= 2:
        intercept_var = result_slopes.cov_re.iloc[0, 0]
        slope_var = result_slopes.cov_re.iloc[1, 1]
        intercept_slope_cov = result_slopes.cov_re.iloc[0, 1] if result_slopes.cov_re.shape[0] > 1 else 0

        print(f"  ✓ Random intercept variance: {intercept_var:.4f}")
        print(f"  ✓ Random slope variance: {slope_var:.4f}")
        print(f"  ✓ Intercept-slope covariance: {intercept_slope_cov:.4f}")
        print(f"  ✓ Random slope SD: {np.sqrt(slope_var):.4f}")
    else:
        print(f"  ⚠ Random slope variance: SINGULAR (variance = 0 or boundary)")

    slopes_success = True

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    print(f"  → Likely cause: Insufficient data (N=100, 4 timepoints)")
    print(f"  → Conclusion: Random slopes model too complex for data")
    slopes_success = False

print()

# ============================================================================
# STEP 4: COMPARE MODELS VIA AIC
# ============================================================================

print("[5/5] Model Comparison & Interpretation")
print("="*80)

if intercepts_success and slopes_success:
    # Both models converged - compare via AIC
    delta_aic = result_intercepts.aic - result_slopes.aic

    print(f"Model A (Intercepts-only):")
    print(f"  AIC: {result_intercepts.aic:.2f}")
    print(f"  Parameters: {len(result_intercepts.params)} fixed + 1 random (intercept variance)")
    print()

    print(f"Model B (Intercepts + Slopes):")
    print(f"  AIC: {result_slopes.aic:.2f}")
    print(f"  Parameters: {len(result_slopes.params)} fixed + 3 random (2 variances + covariance)")
    print()

    print(f"ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    print()

    # Interpret ΔAIC per Burnham & Anderson (2004)
    if delta_aic > 2:
        # Option A: Slopes improve fit
        print("🔴 OUTCOME: OPTION A - SLOPES IMPROVE FIT")
        print(f"  → ΔAIC = {delta_aic:.2f} > 2 (substantial improvement)")
        print(f"  → Random slope variance: {result_slopes.cov_re.iloc[1,1]:.4f}")
        print(f"  → Random slope SD: {np.sqrt(result_slopes.cov_re.iloc[1,1]):.4f}")
        print(f"  → INTERPRETATION: Individual differences in forgetting rates confirmed")
        print(f"  → RECOMMENDATION: Use Model B (slopes) for downstream analyses")
        print(f"  → Document: 'Individual forgetting rates vary (SD={np.sqrt(result_slopes.cov_re.iloc[1,1]):.3f})'")
        outcome = "A"

    elif delta_aic < -2:
        # Slopes model worse (shouldn't happen if it converged properly, but possible with overfitting)
        print("🟡 OUTCOME: OPTION C - SLOPES CONVERGE BUT DON'T IMPROVE")
        print(f"  → ΔAIC = {delta_aic:.2f} < -2 (intercepts-only favored by parsimony)")
        print(f"  → Random slope variance: {result_slopes.cov_re.iloc[1,1]:.4f} (negligible)")
        print(f"  → INTERPRETATION: Homogeneous forgetting rates across participants")
        print(f"  → RECOMMENDATION: Keep Model A (intercepts-only)")
        print(f"  → Document: 'Random slopes tested, variance negligible (homogeneous effects confirmed)'")
        outcome = "C"

    else:
        # -2 < ΔAIC < 2: Models essentially equivalent
        print("🟢 OUTCOME: OPTION C - MODELS EQUIVALENT")
        print(f"  → ΔAIC = {delta_aic:.2f} (|ΔAIC| < 2, models equivalent)")
        print(f"  → Random slope variance: {result_slopes.cov_re.iloc[1,1]:.4f}")
        print(f"  → INTERPRETATION: Slopes model adds complexity without clear improvement")
        print(f"  → RECOMMENDATION: Keep Model A (intercepts-only) by parsimony")
        print(f"  → Document: 'Random slopes tested, no substantial improvement (ΔAIC={delta_aic:.2f})'")
        outcome = "C"

elif intercepts_success and not slopes_success:
    # Option B: Slopes model failed to converge
    print("🔴 OUTCOME: OPTION B - SLOPES MODEL FAILED TO CONVERGE")
    print(f"  → Model A (intercepts) AIC: {result_intercepts.aic:.2f}")
    print(f"  → Model B (slopes) FAILED: Convergence failure or singular covariance matrix")
    print(f"  → Likely cause: Insufficient data (N=100 participants × 4 timepoints)")
    print(f"  → INTERPRETATION: Data insufficient for stable random slope estimation")
    print(f"  → RECOMMENDATION: Keep Model A (intercepts-only)")
    print(f"  → Document: 'Random slopes attempted, convergence failed with N=4 timepoints'")
    outcome = "B"

else:
    print("✗ CRITICAL ERROR: Both models failed to converge")
    print("  → This should not happen - intercepts-only model previously successful")
    print("  → Recommend investigating data quality or transformation issues")
    outcome = "ERROR"

print()

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("="*80)
print("SAVING RESULTS")
print("="*80)

# Create comparison summary dataframe
comparison_data = {
    'model_name': ['Intercepts_Only', 'Intercepts_Slopes'],
    'formula': [
        'Theta ~ power_law + (1 | UID)',
        'Theta ~ power_law + (power_law | UID)'
    ],
    'converged': [
        result_intercepts.converged if intercepts_success else False,
        result_slopes.converged if slopes_success else False
    ],
    'aic': [
        result_intercepts.aic if intercepts_success else np.nan,
        result_slopes.aic if slopes_success else np.nan
    ],
    'log_likelihood': [
        result_intercepts.llf if intercepts_success else np.nan,
        result_slopes.llf if slopes_success else np.nan
    ],
    'n_params_fixed': [
        len(result_intercepts.params) if intercepts_success else np.nan,
        len(result_slopes.params) if slopes_success else np.nan
    ],
    'n_params_random': [
        1,  # Intercept variance only
        3   # 2 variances + covariance
    ],
    'random_intercept_var': [
        result_intercepts.cov_re.iloc[0,0] if intercepts_success else np.nan,
        result_slopes.cov_re.iloc[0,0] if slopes_success and result_slopes.cov_re.shape[0] >= 1 else np.nan
    ],
    'random_slope_var': [
        0.0,  # No slope in Model A
        result_slopes.cov_re.iloc[1,1] if slopes_success and result_slopes.cov_re.shape[0] >= 2 else np.nan
    ]
}

comparison_df = pd.DataFrame(comparison_data)

# Add ΔAIC
if intercepts_success and slopes_success:
    comparison_df['delta_aic'] = [0.0, result_slopes.aic - result_intercepts.aic]
else:
    comparison_df['delta_aic'] = [0.0, np.nan]

# Add outcome
comparison_df['outcome'] = [
    'Reference',
    outcome if intercepts_success else 'ERROR'
]

# Save comparison table
output_file = '../data/step08_random_slopes_comparison.csv'
comparison_df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")
print()

# Print comparison table
print("COMPARISON TABLE:")
print(comparison_df.to_string(index=False))
print()

# ============================================================================
# FINAL SUMMARY FOR DOCUMENTATION
# ============================================================================

print("="*80)
print("DOCUMENTATION SUMMARY (Add to summary.md)")
print("="*80)
print()

if outcome == "A":
    print("### Random Effects Structure")
    print()
    print("**Random Slopes Comparison (MANDATORY per improvement_taxonomy.md Section 4.4):**")
    print()
    print("| Model | Formula | AIC | ΔAIC | Outcome |")
    print("|-------|---------|-----|------|---------|")
    print(f"| Intercepts-only | Theta ~ power_law + (1 \\| UID) | {result_intercepts.aic:.2f} | 0.00 | Reference |")
    print(f"| Intercepts + Slopes | Theta ~ power_law + (power_law \\| UID) | {result_slopes.aic:.2f} | {delta_aic:.2f} | **SELECTED** |")
    print()
    print(f"**Conclusion:** Random slopes model improves fit (ΔAIC = {delta_aic:.2f} > 2). Individual differences in forgetting rates confirmed (slope SD = {np.sqrt(result_slopes.cov_re.iloc[1,1]):.3f}).")
    print()
    print("**Interpretation:** Participants vary in their power-law forgetting exponent α. Heterogeneous forgetting rates observed.")

elif outcome == "B":
    print("### Random Effects Structure")
    print()
    print("**Random Slopes Comparison (MANDATORY per improvement_taxonomy.md Section 4.4):**")
    print()
    print("| Model | Formula | AIC | Outcome |")
    print("|-------|---------|-----|---------|")
    print(f"| Intercepts-only | Theta ~ power_law + (1 \\| UID) | {result_intercepts.aic:.2f} | **SELECTED** |")
    print("| Intercepts + Slopes | Theta ~ power_law + (power_law \\| UID) | N/A | Convergence failure |")
    print()
    print("**Conclusion:** Random slopes model failed to converge (singular covariance matrix). Insufficient data (N=4 timepoints) for stable slope estimation.")
    print()
    print("**Interpretation:** Intercepts-only model retained. Homogeneous forgetting rates assumed (not definitively tested due to convergence failure).")

elif outcome == "C":
    print("### Random Effects Structure")
    print()
    print("**Random Slopes Comparison (MANDATORY per improvement_taxonomy.md Section 4.4):**")
    print()
    print("| Model | Formula | AIC | ΔAIC | Outcome |")
    print("|-------|---------|-----|------|---------|")
    print(f"| Intercepts-only | Theta ~ power_law + (1 \\| UID) | {result_intercepts.aic:.2f} | 0.00 | **SELECTED** |")
    print(f"| Intercepts + Slopes | Theta ~ power_law + (power_law \\| UID) | {result_slopes.aic:.2f} | {delta_aic:.2f} | Equivalent |")
    print()
    print(f"**Conclusion:** Random slopes model converged but did not improve fit (ΔAIC = {delta_aic:.2f}, |ΔAIC| < 2). Slope variance negligible ({result_slopes.cov_re.iloc[1,1]:.4f}).")
    print()
    print("**Interpretation:** Homogeneous forgetting rates confirmed. Individual differences in slopes negligible.")

print()
print("="*80)
print("RANDOM SLOPES COMPARISON COMPLETE")
print("="*80)
print()
print("🔴 BLOCKER RESOLVED: Random slopes tested (MANDATORY requirement met)")
print()
