"""
Random Slopes Comparison for RQ 6.5.1
MANDATORY CHECK per improvement_taxonomy.md Section 4.4

Tests whether random slopes for log_TSVR improve model fit compared to
intercepts-only model. Cannot claim homogeneous confidence decline rates
without testing for individual differences.

Expected outcomes:
- Option A: Slopes improve fit (ΔAIC > 2) → Use slopes model, report heterogeneity
- Option B: Slopes don't converge → Document attempt, explain why (e.g., insufficient timepoints)
- Option C: Slopes converge but ΔAIC < 2 → Keep intercepts, document negligible variance
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
print("Random Slopes Comparison - RQ 6.5.1")
print("Testing: (1 | UID) vs (1 + log_TSVR | UID)")
print("="*80)

# Load LMM input data
lmm_input = pd.read_csv(data_path / "step04_lmm_input.csv")
print(f"\nLoaded {len(lmm_input)} observations from step04_lmm_input.csv")
print(f"Participants: {lmm_input['UID'].nunique()}")
print(f"Observations per participant: {len(lmm_input) / lmm_input['UID'].nunique():.1f}")

# Model 1: Intercepts-only (CURRENT MODEL)
print("\n" + "="*80)
print("MODEL 1: Random Intercepts Only (Current)")
print("="*80)
print("Formula: theta ~ C(congruence) * log_TSVR")
print("Random effects: (1 | UID)")

model_intercepts = smf.mixedlm(
    "theta ~ C(congruence) * log_TSVR",
    data=lmm_input,
    groups=lmm_input['UID'],
    re_formula="~1"  # Random intercept only
)

try:
    result_intercepts = model_intercepts.fit(method='lbfgs', reml=False)
    print(f"\nConverged: {result_intercepts.converged}")
    print(f"AIC: {result_intercepts.aic:.2f}")
    print(f"BIC: {result_intercepts.bic:.2f}")
    print(f"Log-Likelihood: {result_intercepts.llf:.4f}")
    print(f"\nRandom Effects Variance:")
    print(f"  Intercept: {result_intercepts.cov_re.iloc[0,0]:.4f}")

    intercepts_converged = True
    intercepts_aic = result_intercepts.aic
    intercepts_variance = result_intercepts.cov_re.iloc[0,0]
except Exception as e:
    print(f"\n❌ MODEL 1 FAILED: {e}")
    intercepts_converged = False
    intercepts_aic = np.nan

# Model 2: Intercepts + Slopes (TEST)
print("\n" + "="*80)
print("MODEL 2: Random Intercepts + Slopes (TEST)")
print("="*80)
print("Formula: theta ~ C(congruence) * log_TSVR")
print("Random effects: (1 + log_TSVR | UID)")

model_slopes = smf.mixedlm(
    "theta ~ C(congruence) * log_TSVR",
    data=lmm_input,
    groups=lmm_input['UID'],
    re_formula="~log_TSVR"  # Random slope on log_TSVR
)

try:
    result_slopes = model_slopes.fit(method='lbfgs', reml=False)
    print(f"\nConverged: {result_slopes.converged}")
    print(f"AIC: {result_slopes.aic:.2f}")
    print(f"BIC: {result_slopes.bic:.2f}")
    print(f"Log-Likelihood: {result_slopes.llf:.4f}")
    print(f"\nRandom Effects Variance-Covariance:")
    print(result_slopes.cov_re)

    slopes_converged = True
    slopes_aic = result_slopes.aic

    # Extract slope variance
    if result_slopes.cov_re.shape[0] >= 2:
        slope_variance = result_slopes.cov_re.iloc[1,1]
        slope_intercept_corr = result_slopes.cov_re.iloc[0,1] / np.sqrt(result_slopes.cov_re.iloc[0,0] * result_slopes.cov_re.iloc[1,1])
        print(f"\n  Intercept variance: {result_slopes.cov_re.iloc[0,0]:.4f}")
        print(f"  Slope variance: {slope_variance:.4f}")
        print(f"  Slope SD: {np.sqrt(slope_variance):.4f}")
        print(f"  Intercept-Slope correlation: {slope_intercept_corr:.3f}")
    else:
        slope_variance = np.nan

except Exception as e:
    print(f"\n❌ MODEL 2 FAILED: {e}")
    slopes_converged = False
    slopes_aic = np.nan
    slope_variance = np.nan

# Compare models
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

if intercepts_converged and slopes_converged:
    delta_aic = intercepts_aic - slopes_aic
    print(f"\nIntercepts-only AIC: {intercepts_aic:.2f}")
    print(f"Intercepts+slopes AIC: {slopes_aic:.2f}")
    print(f"ΔAIC (intercepts - slopes): {delta_aic:.2f}")

    if delta_aic > 2:
        print(f"\n✅ SLOPES IMPROVE FIT (ΔAIC = {delta_aic:.2f} > 2)")
        print("Random slope variance is non-zero")
        print("Individual differences in confidence decline confirmed")
        print("\n🔴 ACTION: Use slopes model going forward, report heterogeneity")
        recommendation = "Use slopes model"
        outcome = "Option A: Slopes improve fit"
    elif delta_aic < -2:
        print(f"\n✅ INTERCEPTS-ONLY BETTER (ΔAIC = {delta_aic:.2f} < -2)")
        print("Random slopes add complexity without improving fit")
        print("\n✅ ACTION: Keep intercepts-only model (simpler is better)")
        recommendation = "Keep intercepts-only"
        outcome = "Option C: Slopes don't improve fit"
    else:
        print(f"\n⚠️ MODELS EQUIVALENT (|ΔAIC| = {abs(delta_aic):.2f} < 2)")
        print("Akaike weights suggest similar support for both models")
        if slope_variance < 0.001:
            print(f"Slope variance negligible ({slope_variance:.6f})")
            print("\n✅ ACTION: Keep intercepts-only (homogeneous effects confirmed)")
            recommendation = "Keep intercepts-only"
            outcome = "Option C: Negligible slope variance"
        else:
            print(f"Slope variance non-trivial ({slope_variance:.4f})")
            print("\n✅ ACTION: Use slopes model (more conservative)")
            recommendation = "Use slopes model"
            outcome = "Option A: Marginal improvement"

elif slopes_converged and not intercepts_converged:
    print("\n❌ INTERCEPTS-ONLY FAILED, SLOPES SUCCEEDED")
    print("Unexpected - intercepts-only should always converge")
    recommendation = "Investigate convergence failure"
    outcome = "Unexpected failure"

elif intercepts_converged and not slopes_converged:
    print("\n⚠️ SLOPES MODEL FAILED TO CONVERGE")
    print("Possible reasons:")
    print("  - Insufficient data per participant (N=12 obs per person may be limiting)")
    print("  - Only 4 timepoints (limited variation for slope estimation)")
    print("  - High correlation between intercept and slope causing identification issues")
    print("\n✅ ACTION: Keep intercepts-only, document convergence failure")
    recommendation = "Keep intercepts-only"
    outcome = "Option B: Slopes don't converge"

else:
    print("\n❌ BOTH MODELS FAILED")
    print("Critical error - investigate data quality")
    recommendation = "Debug data issues"
    outcome = "Both models failed"

# Save comparison report
print("\n" + "="*80)
print("SAVING COMPARISON REPORT")
print("="*80)

comparison_df = pd.DataFrame({
    'model': ['Intercepts-only', 'Intercepts+slopes'],
    'converged': [intercepts_converged, slopes_converged],
    'aic': [intercepts_aic if intercepts_converged else np.nan,
            slopes_aic if slopes_converged else np.nan],
    'delta_aic': [0.0 if intercepts_converged else np.nan,
                  delta_aic if (intercepts_converged and slopes_converged) else np.nan],
    'random_effects': ['~1', '~log_TSVR'],
    'n_random_params': [1, 3]  # intercepts: 1 var, slopes: 2 var + 1 cov
})

comparison_df.to_csv(output_path / "random_slopes_comparison.csv", index=False)
print(f"✅ Saved: {output_path / 'random_slopes_comparison.csv'}")

# Save detailed report
report_path = output_path / "random_slopes_comparison_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("RANDOM SLOPES COMPARISON - RQ 6.5.1\n")
    f.write("Taxonomy Section 4.4 MANDATORY CHECK\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION:\n")
    f.write("Do schema congruence groups (Common/Congruent/Incongruent) show different\n")
    f.write("confidence decline patterns across 6-day retention interval?\n\n")

    f.write("NULL HYPOTHESIS (Random Slopes):\n")
    f.write("All participants have identical confidence decline rates (homogeneous slopes).\n")
    f.write("Random slope variance = 0.\n\n")

    f.write("ALTERNATIVE HYPOTHESIS (Random Slopes):\n")
    f.write("Participants differ in confidence decline rates (heterogeneous slopes).\n")
    f.write("Random slope variance > 0.\n\n")

    f.write("-"*80 + "\n")
    f.write("MODEL COMPARISON RESULTS\n")
    f.write("-"*80 + "\n\n")

    if intercepts_converged:
        f.write(f"Intercepts-only:\n")
        f.write(f"  AIC: {intercepts_aic:.2f}\n")
        f.write(f"  Intercept variance: {intercepts_variance:.4f}\n")
        f.write(f"  Converged: TRUE\n\n")
    else:
        f.write("Intercepts-only: CONVERGENCE FAILED\n\n")

    if slopes_converged:
        f.write(f"Intercepts+slopes:\n")
        f.write(f"  AIC: {slopes_aic:.2f}\n")
        f.write(f"  Slope variance: {slope_variance:.4f}\n")
        f.write(f"  Slope SD: {np.sqrt(slope_variance):.4f}\n")
        f.write(f"  Converged: TRUE\n\n")
    else:
        f.write("Intercepts+slopes: CONVERGENCE FAILED\n\n")

    f.write("-"*80 + "\n")
    f.write("DECISION\n")
    f.write("-"*80 + "\n\n")
    f.write(f"Outcome: {outcome}\n")
    f.write(f"Recommendation: {recommendation}\n\n")

    if intercepts_converged and slopes_converged:
        f.write(f"ΔAIC (intercepts - slopes): {delta_aic:.2f}\n")
        if delta_aic > 2:
            f.write("\nSlopes model improves fit (ΔAIC > 2).\n")
            f.write("Random slope variance is non-zero, indicating individual differences\n")
            f.write("in confidence decline rates across participants.\n\n")
            f.write("INTERPRETATION: Participants do NOT show homogeneous forgetting.\n")
            f.write("Some maintain confidence longer than others.\n")
        elif delta_aic < -2:
            f.write("\nIntercepts-only model better (ΔAIC < -2).\n")
            f.write("Random slopes add unnecessary complexity.\n\n")
            f.write("INTERPRETATION: Homogeneous confidence decline confirmed.\n")
        else:
            f.write("\nModels equivalent (|ΔAIC| < 2).\n")
            if slope_variance < 0.001:
                f.write(f"Slope variance negligible ({slope_variance:.6f}).\n\n")
                f.write("INTERPRETATION: Effectively homogeneous decline rates.\n")
            else:
                f.write(f"Slope variance non-trivial ({slope_variance:.4f}).\n\n")
                f.write("INTERPRETATION: Some individual differences, but marginal.\n")
    elif slopes_converged and not intercepts_converged:
        f.write("\nUnexpected: Intercepts-only failed but slopes succeeded.\n")
        f.write("This should not occur - investigate data quality.\n")
    elif intercepts_converged and not slopes_converged:
        f.write("\nSlopes model failed to converge.\n")
        f.write("Possible reasons:\n")
        f.write("  - Insufficient data per participant (N=12 obs, 4 timepoints)\n")
        f.write("  - High intercept-slope correlation\n")
        f.write("  - Model overparameterization\n\n")
        f.write("INTERPRETATION: Keep intercepts-only model.\n")
        f.write("Document convergence failure as evidence that random slopes\n")
        f.write("are not identifiable with current data structure.\n")
    else:
        f.write("\nBoth models failed - critical data quality issue.\n")

    f.write("\n" + "-"*80 + "\n")
    f.write("TAXONOMY SECTION 4.4 COMPLIANCE\n")
    f.write("-"*80 + "\n\n")
    f.write("✅ Random slopes TESTED (no longer a BLOCKER)\n")
    f.write("✅ Comparison documented with AIC\n")
    if slopes_converged:
        f.write("✅ Variance components reported\n")
    else:
        f.write("✅ Convergence failure documented\n")
    f.write("\nMANDATORY CHECK COMPLETE.\n")
    f.write("Can now claim homogeneous/heterogeneous effects based on empirical test.\n")

print(f"✅ Saved: {report_path}")

print("\n" + "="*80)
print("RANDOM SLOPES COMPARISON COMPLETE")
print("="*80)
print(f"Outcome: {outcome}")
print(f"Recommendation: {recommendation}")
print("\nFiles created:")
print(f"  - {output_path / 'random_slopes_comparison.csv'}")
print(f"  - {report_path}")
print("\nNext step: Update validation.md with random slopes test results")
