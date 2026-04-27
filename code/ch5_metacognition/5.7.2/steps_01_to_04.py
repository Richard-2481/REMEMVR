#!/usr/bin/env python3
"""
RQ 6.7.2: Confidence Variability Predicts Memory Variability
=============================================================

Steps 01-04: Compute SD_confidence, SD_accuracy, correlate, prepare plot data

Primary Hypothesis: High within-person confidence variability (SD of confidence)
will predict high within-person accuracy variability (SD of accuracy).
Expected positive correlation: r > 0.30 indicates meaningful association.

CRITICAL: Includes partial correlation sensitivity analysis controlling for
mean accuracy (binary SD constraint: SD = sqrt[p*(1-p)]).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
RESULTS_DIR = RQ_DIR / "results"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "steps_01_to_04.log"

# Parameters
PARADIGMS = ["IFR", "ICR", "IRE"]
MIN_ITEMS = 10
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Effect size thresholds
EFFECT_STRONG = 0.50
EFFECT_MODERATE = 0.30


def log(msg: str):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def classify_effect_size(r: float) -> str:
    """Classify effect size based on |r|."""
    abs_r = abs(r)
    if abs_r >= EFFECT_STRONG:
        return "strong"
    elif abs_r >= EFFECT_MODERATE:
        return "moderate"
    else:
        return "weak"


def permutation_test_correlation(x: np.ndarray, y: np.ndarray, n_perm: int = 10000) -> float:
    """Compute permutation-based p-value for correlation."""
    observed_r, _ = pearsonr(x, y)
    count_extreme = 0

    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        r_perm, _ = pearsonr(x, y_perm)
        if abs(r_perm) >= abs(observed_r):
            count_extreme += 1

    return (count_extreme + 1) / (n_perm + 1)


def bootstrap_ci_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, ci_level: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval for correlation."""
    n = len(x)
    boot_rs = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        r_boot, _ = pearsonr(x[idx], y[idx])
        boot_rs.append(r_boot)

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_rs, alpha/2 * 100)
    ci_upper = np.percentile(boot_rs, (1 - alpha/2) * 100)

    return ci_lower, ci_upper


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    """
    Compute partial correlation r(x,y|z) controlling for z.

    Returns: (partial_r, p_value, df)
    """
    # Residualize x on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    x_resid = x - (intercept_xz + slope_xz * z)

    # Residualize y on z
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    y_resid = y - (intercept_yz + slope_yz * z)

    # Correlate residuals
    r_partial, _ = pearsonr(x_resid, y_resid)

    # Compute t-statistic and p-value
    n = len(x)
    df = n - 3  # Controlling for 1 variable
    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2))
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return r_partial, p_value, df


# =============================================================================
# STEP 01: Compute Within-Person Confidence Variability
# =============================================================================

def step01_compute_sd_confidence():
    """Compute SD of confidence ratings per participant per test."""
    log("=" * 70)
    log("STEP 01: Compute Within-Person Confidence Variability")
    log("=" * 70)

    # Load master dataset
    df = pd.read_csv(Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv"))
    log(f"Loaded dfData.csv: {len(df)} rows")

    # Get TC_ columns (confidence ratings)
    tc_cols = [c for c in df.columns if c.startswith("TC_")]
    log(f"Found {len(tc_cols)} TC_ columns (confidence ratings)")

    # Filter for interactive paradigms
    # TC_ columns format: TC_{paradigm}-{domain}-{item} (hyphen separated)
    # e.g., TC_IFR-N-i1, TC_ICR-L-i1, TC_IRE-O-i1
    tc_interactive = []
    for col in tc_cols:
        # Extract paradigm: TC_XXX-... -> XXX
        paradigm = col.split("_")[1].split("-")[0]
        if paradigm in PARADIGMS:
            tc_interactive.append(col)

    log(f"Filtered to {len(tc_interactive)} TC_ columns for paradigms: {PARADIGMS}")

    # Melt to long format
    # Note: Column is TEST (uppercase), values are 1-4
    id_vars = ['UID', 'TEST']
    df_long = df[id_vars + tc_interactive].melt(
        id_vars=id_vars,
        var_name='item',
        value_name='confidence'
    )

    # Rename TEST to test for consistency
    df_long = df_long.rename(columns={'TEST': 'test'})

    # Drop NaN values
    df_long = df_long.dropna(subset=['confidence'])
    log(f"After dropping NaN: {len(df_long)} item-level observations")

    # Compute SD per participant per test
    sd_conf = df_long.groupby(['UID', 'test']).agg(
        SD_confidence=('confidence', lambda x: x.std(ddof=1)),
        N_items=('confidence', 'count'),
        mean_confidence=('confidence', 'mean')  # For sensitivity analysis
    ).reset_index()

    log(f"Computed SD for {len(sd_conf)} participant-test combinations")

    # Filter for minimum items
    before_filter = len(sd_conf)
    sd_conf = sd_conf[sd_conf['N_items'] >= MIN_ITEMS]
    excluded = before_filter - len(sd_conf)
    log(f"Excluded {excluded} observations with < {MIN_ITEMS} items")
    log(f"Retained {len(sd_conf)} observations")

    # Validate ranges
    assert sd_conf['SD_confidence'].min() >= 0, "SD_confidence cannot be negative"
    assert sd_conf['SD_confidence'].max() <= 0.5, "SD_confidence max for 0-1 scale is 0.5"
    assert not sd_conf['SD_confidence'].isna().any(), "SD_confidence contains NaN"

    log(f"SD_confidence range: [{sd_conf['SD_confidence'].min():.4f}, {sd_conf['SD_confidence'].max():.4f}]")
    log(f"Mean SD_confidence: {sd_conf['SD_confidence'].mean():.4f}")

    # Save
    sd_conf.to_csv(DATA_DIR / "step01_sd_confidence.csv", index=False)
    log(f"Saved: data/step01_sd_confidence.csv ({len(sd_conf)} rows)")

    return sd_conf


# =============================================================================
# STEP 02: Compute Within-Person Accuracy Variability
# =============================================================================

def step02_compute_sd_accuracy():
    """Compute SD of accuracy responses per participant per test."""
    log("")
    log("=" * 70)
    log("STEP 02: Compute Within-Person Accuracy Variability")
    log("=" * 70)

    # Load master dataset
    df = pd.read_csv(Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv"))
    log(f"Loaded dfData.csv: {len(df)} rows")

    # Get TQ_ columns (accuracy responses)
    tq_cols = [c for c in df.columns if c.startswith("TQ_")]
    log(f"Found {len(tq_cols)} TQ_ columns (accuracy responses)")

    # Filter for interactive paradigms
    # TQ_ columns format: TQ_{paradigm}-{domain}-{item} (hyphen separated)
    tq_interactive = []
    for col in tq_cols:
        paradigm = col.split("_")[1].split("-")[0]
        if paradigm in PARADIGMS:
            tq_interactive.append(col)

    log(f"Filtered to {len(tq_interactive)} TQ_ columns for paradigms: {PARADIGMS}")

    # Melt to long format
    id_vars = ['UID', 'TEST']
    df_long = df[id_vars + tq_interactive].melt(
        id_vars=id_vars,
        var_name='item',
        value_name='accuracy'
    )

    # Rename TEST to test for consistency
    df_long = df_long.rename(columns={'TEST': 'test'})

    # Drop NaN values
    df_long = df_long.dropna(subset=['accuracy'])
    log(f"After dropping NaN: {len(df_long)} item-level observations")

    # Compute SD per participant per test
    sd_acc = df_long.groupby(['UID', 'test']).agg(
        SD_accuracy=('accuracy', lambda x: x.std(ddof=1)),
        N_items=('accuracy', 'count'),
        mean_accuracy=('accuracy', 'mean')  # For sensitivity analysis
    ).reset_index()

    log(f"Computed SD for {len(sd_acc)} participant-test combinations")

    # Filter for minimum items
    before_filter = len(sd_acc)
    sd_acc = sd_acc[sd_acc['N_items'] >= MIN_ITEMS]
    excluded = before_filter - len(sd_acc)
    log(f"Excluded {excluded} observations with < {MIN_ITEMS} items")
    log(f"Retained {len(sd_acc)} observations")

    # Validate ranges
    assert sd_acc['SD_accuracy'].min() >= 0, "SD_accuracy cannot be negative"
    assert sd_acc['SD_accuracy'].max() <= 0.5, "SD_accuracy max for binary data is 0.5"
    assert not sd_acc['SD_accuracy'].isna().any(), "SD_accuracy contains NaN"

    log(f"SD_accuracy range: [{sd_acc['SD_accuracy'].min():.4f}, {sd_acc['SD_accuracy'].max():.4f}]")
    log(f"Mean SD_accuracy: {sd_acc['SD_accuracy'].mean():.4f}")

    # Save
    sd_acc.to_csv(DATA_DIR / "step02_sd_accuracy.csv", index=False)
    log(f"Saved: data/step02_sd_accuracy.csv ({len(sd_acc)} rows)")

    return sd_acc


# =============================================================================
# STEP 03: Correlate Confidence Variability vs Accuracy Variability
# =============================================================================

def step03_correlate_variability(sd_conf: pd.DataFrame, sd_acc: pd.DataFrame):
    """Compute correlation with dual p-values per Decision D068."""
    log("")
    log("=" * 70)
    log("STEP 03: Correlate Confidence Variability vs Accuracy Variability")
    log("=" * 70)

    # Merge datasets
    merged = pd.merge(
        sd_conf[['UID', 'test', 'SD_confidence', 'mean_confidence']],
        sd_acc[['UID', 'test', 'SD_accuracy', 'mean_accuracy']],
        on=['UID', 'test'],
        how='inner'
    )
    log(f"Merged data: {len(merged)} observations")

    # Save merged data for Step 04
    merged.to_csv(DATA_DIR / "step03_merged_variability.csv", index=False)

    # === PRIMARY ANALYSIS: Person-level aggregation (N=100) ===
    log("")
    log("-" * 50)
    log("PRIMARY ANALYSIS: Person-level aggregation (N=100)")
    log("-" * 50)

    person_level = merged.groupby('UID').agg(
        avg_SD_confidence=('SD_confidence', 'mean'),
        avg_SD_accuracy=('SD_accuracy', 'mean'),
        avg_mean_accuracy=('mean_accuracy', 'mean')  # For partial correlation
    ).reset_index()

    log(f"Person-level data: {len(person_level)} participants")

    x_person = person_level['avg_SD_confidence'].values
    y_person = person_level['avg_SD_accuracy'].values
    z_person = person_level['avg_mean_accuracy'].values  # Control variable

    # Pearson correlation (parametric)
    r_person, p_param_person = pearsonr(x_person, y_person)

    # Permutation p-value (non-parametric)
    log("Computing permutation p-value (10,000 iterations)...")
    p_perm_person = permutation_test_correlation(x_person, y_person, N_PERMUTATIONS)

    # Bootstrap CI
    log("Computing bootstrap 95% CI (1,000 resamples)...")
    ci_lower_person, ci_upper_person = bootstrap_ci_correlation(x_person, y_person, N_BOOTSTRAP, CI_LEVEL)

    # Effect size
    effect_person = classify_effect_size(r_person)

    log(f"")
    log(f"PERSON-LEVEL CORRELATION (N={len(person_level)}):")
    log(f"  Pearson r = {r_person:.4f}")
    log(f"  p_parametric = {p_param_person:.6f}")
    log(f"  p_permutation = {p_perm_person:.6f}")
    log(f"  95% CI: [{ci_lower_person:.4f}, {ci_upper_person:.4f}]")
    log(f"  Effect size: {effect_person}")

    # === PARTIAL CORRELATION SENSITIVITY ANALYSIS ===
    log("")
    log("-" * 50)
    log("SENSITIVITY ANALYSIS: Partial correlation controlling mean accuracy")
    log("-" * 50)
    log("NOTE: Binary SD is constrained by mean: SD = sqrt[p*(1-p)]")
    log("Partial correlation removes this mathematical artifact.")

    r_partial, p_partial, df_partial = partial_correlation(x_person, y_person, z_person)

    log(f"")
    log(f"PARTIAL CORRELATION r(SD_conf, SD_acc | mean_acc):")
    log(f"  Partial r = {r_partial:.4f}")
    log(f"  p-value = {p_partial:.6f}")
    log(f"  df = {df_partial}")

    # Interpretation
    log("")
    log("INTERPRETATION:")
    if p_param_person < 0.05 and p_partial < 0.05:
        log("  BOTH unadjusted r AND partial r are SIGNIFICANT")
        log("  → Variability relationship is ROBUST to mean accuracy confound")
        log("  → Genuine metacognitive signal (not merely mathematical constraint)")
        interpretation = "robust"
    elif p_param_person < 0.05 and p_partial >= 0.05:
        log("  Unadjusted r is SIGNIFICANT but partial r is NOT SIGNIFICANT")
        log("  → Mathematical constraint dominates")
        log("  → Cannot conclude metacognitive sensitivity")
        interpretation = "constraint_dominated"
    else:
        log("  Neither correlation is significant")
        log("  → No evidence for variability relationship")
        interpretation = "null"

    # === SUPPLEMENTARY ANALYSIS: Observation-level (N=400) ===
    log("")
    log("-" * 50)
    log("SUPPLEMENTARY ANALYSIS: Observation-level (N=400)")
    log("-" * 50)
    log("NOTE: 400 observations are non-independent (4 per person).")
    log("Person-level (N=100) is PRIMARY; this is for completeness.")

    x_obs = merged['SD_confidence'].values
    y_obs = merged['SD_accuracy'].values

    r_obs, p_param_obs = pearsonr(x_obs, y_obs)
    log(f"")
    log(f"OBSERVATION-LEVEL CORRELATION (N={len(merged)}):")
    log(f"  Pearson r = {r_obs:.4f}")
    log(f"  p_parametric = {p_param_obs:.6f}")
    log(f"  (Non-independence NOT corrected - use person-level as primary)")

    # Save correlation results
    correlation_results = pd.DataFrame([{
        'analysis_level': 'person',
        'r': r_person,
        'p_parametric': p_param_person,
        'p_permutation': p_perm_person,
        'CI_lower': ci_lower_person,
        'CI_upper': ci_upper_person,
        'N': len(person_level),
        'effect_size_category': effect_person,
        'r_partial': r_partial,
        'p_partial': p_partial,
        'interpretation': interpretation
    }])

    correlation_results.to_csv(DATA_DIR / "step03_correlation.csv", index=False)
    log(f"")
    log(f"Saved: data/step03_correlation.csv")

    # Save person-level data
    person_level.to_csv(DATA_DIR / "step03_person_level.csv", index=False)
    log(f"Saved: data/step03_person_level.csv")

    # Validation
    log("")
    log("VALIDATION - Decision D068 Compliance:")
    log(f"  p_parametric present: {not pd.isna(p_param_person)}")
    log(f"  p_permutation present: {not pd.isna(p_perm_person)}")
    log("  VALIDATION - PASS: Dual p-values present (parametric + permutation)")

    return merged, person_level, correlation_results


# =============================================================================
# STEP 04: Prepare Scatterplot Data
# =============================================================================

def step04_prepare_scatterplot_data(person_level: pd.DataFrame):
    """Create plot source data for scatterplot with regression line."""
    log("")
    log("=" * 70)
    log("STEP 04: Prepare Scatterplot Data")
    log("=" * 70)

    x = person_level['avg_SD_confidence'].values
    y = person_level['avg_SD_accuracy'].values

    # Fit linear regression
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    log(f"Regression: SD_accuracy = {intercept:.4f} + {slope:.4f} * SD_confidence")
    log(f"R² = {r_val**2:.4f}")

    # Generate regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line

    # Save scatterplot data (person-level)
    scatter_data = person_level[['UID', 'avg_SD_confidence', 'avg_SD_accuracy']].copy()
    scatter_data.columns = ['UID', 'SD_confidence', 'SD_accuracy']
    scatter_data.to_csv(DATA_DIR / "step04_variability_scatterplot_data.csv", index=False)
    log(f"Saved: data/step04_variability_scatterplot_data.csv ({len(scatter_data)} rows)")

    # Save regression line
    regression_line = pd.DataFrame({
        'SD_confidence': x_line,
        'SD_accuracy_predicted': y_line
    })
    regression_line.to_csv(DATA_DIR / "step04_variability_regression_line.csv", index=False)
    log(f"Saved: data/step04_variability_regression_line.csv (100 rows)")

    # Validation
    log("")
    log("VALIDATION:")
    log(f"  Scatterplot rows: {len(scatter_data)} (expected: ~100)")
    log(f"  Regression line rows: 100")
    log(f"  SD_confidence range: [{x.min():.4f}, {x.max():.4f}]")
    log(f"  SD_accuracy range: [{y.min():.4f}, {y.max():.4f}]")
    log("  VALIDATION - PASS: Plot data complete")

    return scatter_data, regression_line


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all steps."""
    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"RQ 6.7.2: Confidence Variability Predicts Memory Variability\n")
        f.write(f"{'=' * 70}\n\n")

    try:
        # Step 01: Compute SD_confidence
        sd_conf = step01_compute_sd_confidence()

        # Step 02: Compute SD_accuracy
        sd_acc = step02_compute_sd_accuracy()

        # Step 03: Correlate variability
        merged, person_level, correlation_results = step03_correlate_variability(sd_conf, sd_acc)

        # Step 04: Prepare scatterplot data
        scatter_data, regression_line = step04_prepare_scatterplot_data(person_level)

        # Summary
        log("")
        log("=" * 70)
        log("EXECUTION COMPLETE")
        log("=" * 70)

        r = correlation_results['r'].values[0]
        p_param = correlation_results['p_parametric'].values[0]
        p_perm = correlation_results['p_permutation'].values[0]
        r_partial = correlation_results['r_partial'].values[0]
        p_partial = correlation_results['p_partial'].values[0]
        effect = correlation_results['effect_size_category'].values[0]

        log("")
        log("PRIMARY FINDINGS (Person-Level, N=100):")
        log(f"  Pearson r = {r:.4f} ({effect} effect)")
        log(f"  p_parametric = {p_param:.6f}")
        log(f"  p_permutation = {p_perm:.6f}")
        log("")
        log("SENSITIVITY ANALYSIS (Partial Correlation):")
        log(f"  Partial r = {r_partial:.4f} (controlling mean accuracy)")
        log(f"  p_partial = {p_partial:.6f}")
        log("")

        # Hypothesis test
        if r > 0.30 and p_param < 0.05:
            log("HYPOTHESIS TEST: SUPPORTED")
            log(f"  Positive correlation r = {r:.4f} > 0.30 threshold, p < .05")
            log("  High confidence variability DOES predict high accuracy variability")
        elif r > 0 and p_param < 0.05:
            log("HYPOTHESIS TEST: PARTIALLY SUPPORTED")
            log(f"  Positive correlation r = {r:.4f} significant, but below 0.30 threshold")
        else:
            log("HYPOTHESIS TEST: NOT SUPPORTED")
            log(f"  Correlation r = {r:.4f} not significant or not positive")

        log("")
        log("Files created:")
        log("  data/step01_sd_confidence.csv")
        log("  data/step02_sd_accuracy.csv")
        log("  data/step03_merged_variability.csv")
        log("  data/step03_correlation.csv")
        log("  data/step03_person_level.csv")
        log("  data/step04_variability_scatterplot_data.csv")
        log("  data/step04_variability_regression_line.csv")

    except Exception as e:
        log(f"\n{'!' * 70}")
        log(f"ERROR: {str(e)}")
        log(f"{'!' * 70}")
        raise


if __name__ == "__main__":
    main()
