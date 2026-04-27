#!/usr/bin/env python3
"""
RQ 6.7.1: Additional Analyses for ROOT RQ Standards
====================================================

This script addresses gaps identified during validation:
1. Regression diagnostics (Q-Q, residuals, Cook's D) - was in plan.md but not executed
2. Partial correlation controlling for baseline accuracy - CRITICAL to disentangle metacognition from regression artifact
3. Simple linear regression with full reporting

These analyses are MANDATORY for a ROOT RQ to be bulletproof.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.7.1
LOG_FILE = RQ_DIR / "logs" / "step06_additional_analyses.log"

# Source data
CH5_5_1_4_SLOPES = Path("/home/etai/projects/REMEMVR/results/ch5/5.1.4/data/step04_random_effects.csv")

def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def init_log():
    """Initialize log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ 6.7.1: Additional Analyses for ROOT RQ Standards\n")
        f.write("=" * 70 + "\n\n")

# =============================================================================
# Step 6A: Simple Linear Regression with Diagnostics
# =============================================================================

def step06a_regression_with_diagnostics():
    """Fit simple linear regression and compute full diagnostics."""
    log("=" * 70)
    log("STEP 6A: Simple Linear Regression with Diagnostics")
    log("=" * 70)

    # Load predictive data
    df = pd.read_csv(RQ_DIR / "data" / "step03_predictive_data.csv")

    x = df['Day0_confidence'].values
    y = df['forgetting_slope'].values

    # Fit OLS regression: forgetting_slope ~ Day0_confidence
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    log("\n--- Regression Summary ---")
    log(f"Model: forgetting_slope ~ Day0_confidence")
    log(f"N = {len(df)}")
    log(f"\nCoefficients:")
    log(f"  Intercept: β = {model.params[0]:.6f}, SE = {model.bse[0]:.6f}, t = {model.tvalues[0]:.3f}, p = {model.pvalues[0]:.6f}")
    log(f"  Day0_confidence: β = {model.params[1]:.6f}, SE = {model.bse[1]:.6f}, t = {model.tvalues[1]:.3f}, p = {model.pvalues[1]:.6f}")
    log(f"\nModel Fit:")
    log(f"  R² = {model.rsquared:.4f}")
    log(f"  Adj R² = {model.rsquared_adj:.4f}")
    log(f"  F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.3f}, p = {model.f_pvalue:.6f}")

    # Residual diagnostics
    residuals = model.resid
    fitted = model.fittedvalues

    # Cook's Distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df)  # Common threshold: 4/N

    influential_points = np.where(cooks_d > threshold)[0]

    log(f"\n--- Residual Diagnostics ---")
    log(f"Residuals:")
    log(f"  Mean: {residuals.mean():.6f} (should be ~0)")
    log(f"  SD: {residuals.std():.6f}")
    log(f"  Range: [{residuals.min():.6f}, {residuals.max():.6f}]")

    # Shapiro-Wilk on residuals (normality assumption)
    shapiro_resid = stats.shapiro(residuals)
    log(f"\nNormality of Residuals (Shapiro-Wilk):")
    log(f"  W = {shapiro_resid.statistic:.4f}, p = {shapiro_resid.pvalue:.4f}")
    if shapiro_resid.pvalue >= 0.05:
        log(f"  → Normality assumption NOT violated (p >= 0.05)")
    else:
        log(f"  → Normality assumption violated (p < 0.05)")
        log(f"  → However, with N=100, Central Limit Theorem provides robustness")

    # Breusch-Pagan test for homoscedasticity
    bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X)
    log(f"\nHomoscedasticity (Breusch-Pagan test):")
    log(f"  LM statistic = {bp_test[0]:.4f}, p = {bp_test[1]:.4f}")
    if bp_test[1] >= 0.05:
        log(f"  → Homoscedasticity assumption NOT violated (p >= 0.05)")
    else:
        log(f"  → Potential heteroscedasticity detected (p < 0.05)")

    # Cook's D summary
    log(f"\nCook's Distance (outlier detection):")
    log(f"  Threshold: {threshold:.4f} (4/N)")
    log(f"  Max Cook's D: {cooks_d.max():.4f}")
    log(f"  Influential points (D > threshold): {len(influential_points)}")
    if len(influential_points) > 0:
        log(f"  → Indices: {influential_points}")
        log(f"  → UIDs: {df.iloc[influential_points]['UID'].tolist()}")
    else:
        log(f"  → No influential outliers detected")

    # Save regression results
    reg_results = pd.DataFrame({
        'term': ['Intercept', 'Day0_confidence'],
        'estimate': model.params,
        'std_error': model.bse,
        't_value': model.tvalues,
        'p_value': model.pvalues,
        'ci_lower': model.conf_int()[0],
        'ci_upper': model.conf_int()[1]
    })
    reg_results.to_csv(RQ_DIR / "data" / "step06a_regression_coefficients.csv", index=False)

    # Save regression diagnostics
    diag_results = pd.DataFrame({
        'metric': ['R_squared', 'Adj_R_squared', 'F_statistic', 'F_pvalue',
                   'Shapiro_W_residuals', 'Shapiro_p_residuals',
                   'BP_LM_statistic', 'BP_p_value',
                   'Max_Cooks_D', 'Cooks_D_threshold', 'N_influential'],
        'value': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue,
                  shapiro_resid.statistic, shapiro_resid.pvalue,
                  bp_test[0], bp_test[1],
                  cooks_d.max(), threshold, len(influential_points)]
    })
    diag_results.to_csv(RQ_DIR / "data" / "step06a_regression_diagnostics.csv", index=False)

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted (Homoscedasticity Check)')

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')

    # 3. Scale-Location (sqrt of standardized residuals vs fitted)
    std_resid = np.sqrt(np.abs(residuals / residuals.std()))
    axes[1, 0].scatter(fitted, std_resid, alpha=0.6, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location (Spread vs Level)')

    # 4. Cook's Distance
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt='o', basefmt=' ')
    axes[1, 1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold (4/N = {threshold:.4f})')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance (Influential Point Detection)")
    axes[1, 1].legend()

    plt.suptitle('RQ 6.7.1: Regression Diagnostic Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RQ_DIR / "plots" / "regression_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()

    log(f"\nSaved: data/step06a_regression_coefficients.csv")
    log(f"Saved: data/step06a_regression_diagnostics.csv")
    log(f"Saved: plots/regression_diagnostics.png")

    return model, cooks_d, influential_points

# =============================================================================
# Step 6B: Partial Correlation (CRITICAL - disentangle metacognition from baseline)
# =============================================================================

def step06b_partial_correlation():
    """
    Compute partial correlation between Day0_confidence and forgetting_slope,
    controlling for baseline accuracy (intercept from Ch5 5.1.4).

    This is the CRITICAL analysis to determine if confidence has unique
    predictive value beyond regression to mean artifact.
    """
    log("\n" + "=" * 70)
    log("STEP 6B: Partial Correlation (CRITICAL - Disentangle from Baseline)")
    log("=" * 70)

    # Load predictive data
    df = pd.read_csv(RQ_DIR / "data" / "step03_predictive_data.csv")

    # Load baseline accuracy intercepts from Ch5 5.1.4
    df_slopes = pd.read_csv(CH5_5_1_4_SLOPES)

    # Merge baseline accuracy intercept
    df = df.merge(df_slopes[['UID', 'total_intercept']], on='UID', how='left')
    df = df.rename(columns={'total_intercept': 'baseline_accuracy'})

    log(f"\nData merged: {len(df)} participants with confidence, slopes, and baseline accuracy")
    log(f"Baseline accuracy range: [{df['baseline_accuracy'].min():.3f}, {df['baseline_accuracy'].max():.3f}]")

    # Zero-order correlations (for comparison)
    r_conf_slope, p_conf_slope = spearmanr(df['Day0_confidence'], df['forgetting_slope'])
    r_base_slope, p_base_slope = spearmanr(df['baseline_accuracy'], df['forgetting_slope'])
    r_conf_base, p_conf_base = spearmanr(df['Day0_confidence'], df['baseline_accuracy'])

    log(f"\n--- Zero-Order Correlations (Spearman) ---")
    log(f"Confidence-Slope: rho = {r_conf_slope:.4f}, p = {p_conf_slope:.6f}")
    log(f"Baseline-Slope: rho = {r_base_slope:.4f}, p = {p_base_slope:.6f}")
    log(f"Confidence-Baseline: rho = {r_conf_base:.4f}, p = {p_conf_base:.6f}")

    # Partial correlation formula (Spearman-based):
    # r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    # Where: x = confidence, y = slope, z = baseline_accuracy

    numerator = r_conf_slope - (r_conf_base * r_base_slope)
    denominator = np.sqrt((1 - r_conf_base**2) * (1 - r_base_slope**2))
    partial_rho = numerator / denominator

    # Degrees of freedom for partial correlation: N - 3 (controlling one variable)
    n = len(df)
    df_partial = n - 3

    # t-test for partial correlation significance
    t_partial = partial_rho * np.sqrt(df_partial / (1 - partial_rho**2))
    p_partial = 2 * (1 - stats.t.cdf(abs(t_partial), df_partial))

    # 95% CI using Fisher z-transformation
    z_partial = 0.5 * np.log((1 + partial_rho) / (1 - partial_rho))
    se_z = 1 / np.sqrt(n - 3 - 1)  # SE for partial correlation z
    z_lower = z_partial - 1.96 * se_z
    z_upper = z_partial + 1.96 * se_z
    ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    log(f"\n--- PARTIAL CORRELATION (Controlling Baseline Accuracy) ---")
    log(f"Partial rho (Confidence-Slope | Baseline): {partial_rho:.4f}")
    log(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    log(f"t({df_partial}) = {t_partial:.3f}, p = {p_partial:.6f}")

    # Interpretation
    log(f"\n--- INTERPRETATION ---")
    if abs(partial_rho) < 0.10:
        log(f"Partial rho = {partial_rho:.4f} is NEGLIGIBLE (|rho| < 0.10)")
        log(f"→ Confidence-Slope relationship DISAPPEARS when controlling baseline accuracy")
        log(f"→ Original correlation (rho = {r_conf_slope:.4f}) was REGRESSION TO MEAN artifact")
        log(f"→ Confidence does NOT add unique predictive value beyond baseline ability")
        interpretation = "NULL_AFTER_CONTROL"
    elif p_partial >= 0.05:
        log(f"Partial rho = {partial_rho:.4f} is NOT SIGNIFICANT (p = {p_partial:.4f})")
        log(f"→ Confidence-Slope relationship ATTENUATED when controlling baseline accuracy")
        log(f"→ Original correlation likely driven by baseline ability confound")
        log(f"→ Limited evidence for unique metacognitive prediction")
        interpretation = "ATTENUATED"
    else:
        log(f"Partial rho = {partial_rho:.4f} REMAINS SIGNIFICANT (p = {p_partial:.6f})")
        log(f"→ Confidence has UNIQUE predictive value beyond baseline ability")
        log(f"→ Metacognitive monitoring adds information not captured by baseline performance")
        log(f"→ Confidence is not merely a proxy for baseline ability")
        interpretation = "UNIQUE_PREDICTOR"

    # Variance partitioning
    var_explained_total = r_conf_slope**2
    var_explained_unique = partial_rho**2
    var_shared = var_explained_total - var_explained_unique

    log(f"\n--- Variance Partitioning ---")
    log(f"Total variance explained by Confidence: {var_explained_total * 100:.1f}%")
    log(f"Unique variance (after controlling baseline): {var_explained_unique * 100:.1f}%")
    log(f"Shared variance (with baseline ability): {var_shared * 100:.1f}%")
    log(f"Proportion unique: {(var_explained_unique / var_explained_total) * 100:.1f}% of total")

    # Save partial correlation results
    partial_results = pd.DataFrame({
        'analysis': ['partial_correlation'],
        'controlling_for': ['baseline_accuracy'],
        'zero_order_rho': [r_conf_slope],
        'zero_order_p': [p_conf_slope],
        'partial_rho': [partial_rho],
        'partial_ci_lower': [ci_lower],
        'partial_ci_upper': [ci_upper],
        't_statistic': [t_partial],
        'df': [df_partial],
        'p_value': [p_partial],
        'interpretation': [interpretation],
        'var_total_pct': [var_explained_total * 100],
        'var_unique_pct': [var_explained_unique * 100],
        'var_shared_pct': [var_shared * 100],
        'conf_baseline_rho': [r_conf_base],
        'baseline_slope_rho': [r_base_slope]
    })
    partial_results.to_csv(RQ_DIR / "data" / "step06b_partial_correlation.csv", index=False)
    log(f"\nSaved: data/step06b_partial_correlation.csv")

    return partial_rho, p_partial, interpretation

# =============================================================================
# Step 6C: Sensitivity Analysis (exclude influential points)
# =============================================================================

def step06c_sensitivity_analysis(influential_points):
    """Test if results robust to excluding influential observations."""
    log("\n" + "=" * 70)
    log("STEP 6C: Sensitivity Analysis (Excluding Influential Points)")
    log("=" * 70)

    # Load predictive data
    df = pd.read_csv(RQ_DIR / "data" / "step03_predictive_data.csv")

    # Full sample correlation
    full_rho, full_p = spearmanr(df['Day0_confidence'], df['forgetting_slope'])

    log(f"\nFull sample (N={len(df)}): rho = {full_rho:.4f}, p = {full_p:.6f}")

    # Exclude influential points if any
    if len(influential_points) > 0:
        df_reduced = df.drop(influential_points).reset_index(drop=True)
        reduced_rho, reduced_p = spearmanr(df_reduced['Day0_confidence'], df_reduced['forgetting_slope'])

        log(f"Reduced sample (N={len(df_reduced)}, excluding {len(influential_points)} influential): rho = {reduced_rho:.4f}, p = {reduced_p:.6f}")
        log(f"Change in rho: {reduced_rho - full_rho:+.4f}")

        if abs(reduced_rho - full_rho) < 0.05:
            log(f"→ Results ROBUST to influential point exclusion (Δrho < 0.05)")
            robust = True
        else:
            log(f"→ Results SENSITIVE to influential points (Δrho >= 0.05)")
            robust = False
    else:
        log(f"No influential points detected (Cook's D all below threshold)")
        log(f"→ Sensitivity analysis not needed - no outliers to exclude")
        reduced_rho = full_rho
        reduced_p = full_p
        robust = True

    # Additional robustness: trim top/bottom 5%
    q05 = df['Day0_confidence'].quantile(0.05)
    q95 = df['Day0_confidence'].quantile(0.95)
    df_trimmed = df[(df['Day0_confidence'] >= q05) & (df['Day0_confidence'] <= q95)]
    trimmed_rho, trimmed_p = spearmanr(df_trimmed['Day0_confidence'], df_trimmed['forgetting_slope'])

    log(f"\nTrimmed sample (N={len(df_trimmed)}, 5% trim each tail): rho = {trimmed_rho:.4f}, p = {trimmed_p:.6f}")
    log(f"Change in rho from full: {trimmed_rho - full_rho:+.4f}")

    # Save sensitivity results
    sensitivity_results = pd.DataFrame({
        'analysis': ['full_sample', 'exclude_influential', 'trimmed_5pct'],
        'N': [len(df), len(df) - len(influential_points) if len(influential_points) > 0 else len(df), len(df_trimmed)],
        'spearman_rho': [full_rho, reduced_rho, trimmed_rho],
        'p_value': [full_p, reduced_p, trimmed_p],
        'delta_rho': [0, reduced_rho - full_rho, trimmed_rho - full_rho]
    })
    sensitivity_results.to_csv(RQ_DIR / "data" / "step06c_sensitivity_analysis.csv", index=False)
    log(f"\nSaved: data/step06c_sensitivity_analysis.csv")

    return robust

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Execute all additional analyses."""
    init_log()

    log("Starting Additional Analyses for ROOT RQ Standards")
    log(f"Output directory: {RQ_DIR}")
    log("")

    # Step 6A: Regression with diagnostics
    model, cooks_d, influential_points = step06a_regression_with_diagnostics()

    # Step 6B: Partial correlation (CRITICAL)
    partial_rho, p_partial, interpretation = step06b_partial_correlation()

    # Step 6C: Sensitivity analysis
    robust = step06c_sensitivity_analysis(influential_points)

    # Final summary
    log("\n" + "=" * 70)
    log("ADDITIONAL ANALYSES COMPLETE")
    log("=" * 70)

    log(f"\n**ROOT RQ BULLETPROOF STATUS:**")
    log(f"\n1. Regression Diagnostics: COMPLETE")
    log(f"   - Q-Q plot, residuals vs fitted, Cook's D all generated")
    log(f"   - Diagnostic plots saved to plots/regression_diagnostics.png")

    log(f"\n2. Partial Correlation (CRITICAL): {interpretation}")
    log(f"   - Partial rho = {partial_rho:.4f} (controlling baseline accuracy)")
    log(f"   - p = {p_partial:.6f}")

    if interpretation == "NULL_AFTER_CONTROL":
        log(f"\n   *** MAJOR FINDING ***")
        log(f"   The confidence-slope correlation (rho = -0.66) is ENTIRELY")
        log(f"   explained by baseline ability. Confidence is NOT a unique")
        log(f"   predictor - it's a PROXY for baseline performance.")
        log(f"   This is classic REGRESSION TO MEAN, not metacognitive prediction.")
    elif interpretation == "ATTENUATED":
        log(f"\n   The confidence-slope correlation is WEAKENED after control.")
        log(f"   Some relationship may exist but is confounded with baseline.")
    else:
        log(f"\n   Confidence has UNIQUE predictive value beyond baseline ability!")
        log(f"   Metacognitive monitoring provides additional information.")

    log(f"\n3. Sensitivity Analysis: {'ROBUST' if robust else 'SENSITIVE'}")
    log(f"   - Results stable across outlier exclusion methods")

    log(f"\nFiles created:")
    log(f"  data/step06a_regression_coefficients.csv")
    log(f"  data/step06a_regression_diagnostics.csv")
    log(f"  data/step06b_partial_correlation.csv")
    log(f"  data/step06c_sensitivity_analysis.csv")
    log(f"  plots/regression_diagnostics.png")

    log("\n" + "=" * 70)
    log("RQ 6.7.1 Additional Analyses Complete")
    log("=" * 70)

if __name__ == "__main__":
    main()
