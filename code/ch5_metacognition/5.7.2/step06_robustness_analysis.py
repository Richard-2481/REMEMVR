"""
RQ 6.7.2 - Step 06: Bootstrap Robustness Analysis for Partial Correlation

PURPOSE:
Validate the marginal p=0.034 partial correlation finding using:
1. Bootstrap 95% CI (10,000 resamples)
2. Leave-one-out cross-validation (100 iterations)
3. Outlier sensitivity analysis
4. Permutation test (1,000 permutations) for non-parametric p-value

FINDING TO VALIDATE:
- Partial r = 0.21 (SD_confidence → SD_accuracy | mean_accuracy controlled)
- p_parametric = 0.034 (marginal, needs validation)

This is a suppression effect analysis: zero-order r ≈ 0, but partial r = 0.21 after
controlling for mean_accuracy (which constrains binary SD mathematically).

INPUT:
- results/ch6/6.7.2/data/step03_person_level.csv
  - Columns: UID, avg_SD_confidence, avg_SD_accuracy, avg_mean_accuracy

OUTPUT:
- data/step06_bootstrap_results.csv
- data/step06_loo_results.csv
- data/step06_permutation_results.csv
- data/step06_outlier_sensitivity.csv
- results/robustness_analysis.md

Author: Claude Code
Date: 2025-12-14
RQ: ch6/6.7.2
Task: T1.2 from rq_rework.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = RQ_DIR / "logs" / "step06_robustness_analysis.log"
DATA_DIR = RQ_DIR / "data"
RESULTS_DIR = RQ_DIR / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)

# Clear log file
with open(LOG_FILE, 'w') as f:
    f.write("")

def log(msg):
    """Log message to file and stdout with flush"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def compute_partial_r(x, y, z):
    """
    Compute partial correlation r(x, y | z).

    Uses regression residuals method:
    1. Residualize x on z: resid_x = x - predicted_x from z
    2. Residualize y on z: resid_y = y - predicted_y from z
    3. Correlate residuals: r(resid_x, resid_y)
    """
    # Handle edge cases
    if len(x) < 4:
        return np.nan, np.nan

    # Residualize x on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)

    # Residualize y on z
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)

    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)

    return r, p


def bootstrap_partial_r(x, y, z, n_bootstrap=10000, random_state=42):
    """
    Bootstrap 95% CI for partial correlation.

    Returns:
    - ci_lower, ci_upper: 95% percentile CI bounds
    - bootstrap_rs: all bootstrap correlation values
    """
    np.random.seed(random_state)
    n = len(x)
    bootstrap_rs = []

    for i in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        z_boot = z[idx]

        # Compute partial r for this sample
        r, _ = compute_partial_r(x_boot, y_boot, z_boot)

        if not np.isnan(r):
            bootstrap_rs.append(r)

    bootstrap_rs = np.array(bootstrap_rs)

    # Percentile CI (not bias-corrected, but simple and interpretable)
    ci_lower = np.percentile(bootstrap_rs, 2.5)
    ci_upper = np.percentile(bootstrap_rs, 97.5)

    return ci_lower, ci_upper, bootstrap_rs


def leave_one_out_partial_r(x, y, z):
    """
    Leave-one-out cross-validation for stability.

    Returns:
    - loo_rs: list of N partial correlations, each with one participant removed
    """
    n = len(x)
    loo_rs = []

    for i in range(n):
        # Create mask to exclude participant i
        mask = np.arange(n) != i
        x_loo = x[mask]
        y_loo = y[mask]
        z_loo = z[mask]

        # Compute partial r without this participant
        r, _ = compute_partial_r(x_loo, y_loo, z_loo)
        loo_rs.append(r)

    return np.array(loo_rs)


def permutation_test_partial_r(x, y, z, n_permutations=1000, random_state=42):
    """
    Permutation test for non-parametric p-value.

    Null hypothesis: no partial relationship between x and y controlling for z.
    Test: permute x, keeping y and z fixed, compute partial r distribution.
    """
    np.random.seed(random_state)

    # Observed partial r
    r_observed, _ = compute_partial_r(x, y, z)

    # Generate null distribution by permuting x
    null_rs = []
    for i in range(n_permutations):
        x_perm = np.random.permutation(x)
        r_perm, _ = compute_partial_r(x_perm, y, z)
        null_rs.append(r_perm)

    null_rs = np.array(null_rs)

    # Two-tailed p-value
    p_permutation = np.mean(np.abs(null_rs) >= np.abs(r_observed))

    return p_permutation, null_rs


def outlier_sensitivity_analysis(x, y, z, threshold_sd=2.5):
    """
    Test sensitivity to outliers.

    1. Identify outliers (> threshold_sd from mean) in x, y, or z
    2. Recompute partial r without outliers
    3. Compare to original
    """
    # Compute z-scores
    z_x = np.abs(stats.zscore(x))
    z_y = np.abs(stats.zscore(y))
    z_z = np.abs(stats.zscore(z))

    # Identify outliers
    outliers = (z_x > threshold_sd) | (z_y > threshold_sd) | (z_z > threshold_sd)
    n_outliers = np.sum(outliers)
    outlier_indices = np.where(outliers)[0]

    # Original partial r
    r_original, p_original = compute_partial_r(x, y, z)

    # Partial r without outliers
    if n_outliers > 0 and n_outliers < len(x) - 3:
        x_clean = x[~outliers]
        y_clean = y[~outliers]
        z_clean = z[~outliers]
        r_clean, p_clean = compute_partial_r(x_clean, y_clean, z_clean)
    else:
        r_clean = r_original
        p_clean = p_original

    return {
        'n_outliers': n_outliers,
        'outlier_indices': outlier_indices,
        'r_original': r_original,
        'p_original': p_original,
        'r_clean': r_clean,
        'p_clean': p_clean,
        'delta_r': r_clean - r_original,
        'threshold_sd': threshold_sd
    }


def main():
    log("=" * 80)
    log("RQ 6.7.2 - Step 06: Bootstrap Robustness Analysis")
    log(f"Started: {datetime.now().isoformat()}")
    log("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    log("\n[STEP 1] Load Person-Level Data")
    log("-" * 60)

    data = pd.read_csv(DATA_DIR / "step03_person_level.csv")
    log(f"  ✓ Loaded {len(data)} participants")
    log(f"  ✓ Columns: {list(data.columns)}")

    # Extract arrays
    x = data['avg_SD_confidence'].values  # Predictor
    y = data['avg_SD_accuracy'].values    # Outcome
    z = data['avg_mean_accuracy'].values  # Covariate

    log(f"\n  Variable Stats:")
    log(f"    SD_confidence: M={np.mean(x):.3f}, SD={np.std(x):.3f}, range=[{np.min(x):.3f}, {np.max(x):.3f}]")
    log(f"    SD_accuracy: M={np.mean(y):.3f}, SD={np.std(y):.3f}, range=[{np.min(y):.3f}, {np.max(y):.3f}]")
    log(f"    mean_accuracy: M={np.mean(z):.3f}, SD={np.std(z):.3f}, range=[{np.min(z):.3f}, {np.max(z):.3f}]")

    # Compute original partial r
    r_original, p_original = compute_partial_r(x, y, z)
    log(f"\n  Original Partial Correlation:")
    log(f"    r_partial = {r_original:.4f}")
    log(f"    p_parametric = {p_original:.4f}")

    # =========================================================================
    # STEP 2: Bootstrap Analysis
    # =========================================================================
    log("\n[STEP 2] Bootstrap 95% CI (10,000 resamples)")
    log("-" * 60)

    ci_lower, ci_upper, bootstrap_rs = bootstrap_partial_r(x, y, z, n_bootstrap=10000)

    log(f"  N bootstrap samples: {len(bootstrap_rs)}")
    log(f"  Bootstrap distribution:")
    log(f"    Mean: {np.mean(bootstrap_rs):.4f}")
    log(f"    SD: {np.std(bootstrap_rs):.4f}")
    log(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    log(f"\n  CI excludes 0? {ci_excludes_zero}")
    if ci_excludes_zero:
        log(f"  ✓ ROBUST: 95% CI does not contain zero")
    else:
        log(f"  ⚠️ MARGINAL: 95% CI contains zero")

    # Save bootstrap results
    bootstrap_df = pd.DataFrame({
        'metric': ['r_original', 'ci_lower', 'ci_upper', 'bootstrap_mean', 'bootstrap_sd',
                   'ci_excludes_zero', 'n_bootstrap'],
        'value': [r_original, ci_lower, ci_upper, np.mean(bootstrap_rs), np.std(bootstrap_rs),
                  ci_excludes_zero, len(bootstrap_rs)]
    })
    bootstrap_df.to_csv(DATA_DIR / "step06_bootstrap_results.csv", index=False)
    log(f"  ✓ Saved: step06_bootstrap_results.csv")

    # =========================================================================
    # STEP 3: Leave-One-Out Cross-Validation
    # =========================================================================
    log("\n[STEP 3] Leave-One-Out Cross-Validation (100 iterations)")
    log("-" * 60)

    loo_rs = leave_one_out_partial_r(x, y, z)

    n_positive = np.sum(loo_rs > 0)
    n_negative = np.sum(loo_rs < 0)
    all_same_direction = (n_positive == len(loo_rs)) or (n_negative == len(loo_rs))

    log(f"  N LOO iterations: {len(loo_rs)}")
    log(f"  LOO distribution:")
    log(f"    Mean: {np.mean(loo_rs):.4f}")
    log(f"    SD: {np.std(loo_rs):.4f}")
    log(f"    Range: [{np.min(loo_rs):.4f}, {np.max(loo_rs):.4f}]")
    log(f"    N positive: {n_positive}")
    log(f"    N negative: {n_negative}")
    log(f"\n  All same direction? {all_same_direction}")

    if all_same_direction:
        log(f"  ✓ STABLE: All LOO correlations have same sign")
    else:
        log(f"  ⚠️ UNSTABLE: Some LOO correlations flip direction")

    # Identify influential observations
    deviation_from_original = np.abs(loo_rs - r_original)
    most_influential_idx = np.argmax(deviation_from_original)
    most_influential_r = loo_rs[most_influential_idx]
    most_influential_uid = data.iloc[most_influential_idx]['UID']

    log(f"\n  Most influential observation:")
    log(f"    UID: {most_influential_uid}")
    log(f"    LOO r without this participant: {most_influential_r:.4f}")
    log(f"    Deviation from original: {deviation_from_original[most_influential_idx]:.4f}")

    # Save LOO results
    loo_df = pd.DataFrame({
        'UID': data['UID'],
        'loo_r': loo_rs,
        'deviation_from_original': deviation_from_original
    })
    loo_df.to_csv(DATA_DIR / "step06_loo_results.csv", index=False)
    log(f"  ✓ Saved: step06_loo_results.csv")

    # =========================================================================
    # STEP 4: Permutation Test
    # =========================================================================
    log("\n[STEP 4] Permutation Test (1,000 permutations)")
    log("-" * 60)

    p_permutation, null_rs = permutation_test_partial_r(x, y, z, n_permutations=1000)

    log(f"  N permutations: {len(null_rs)}")
    log(f"  Null distribution:")
    log(f"    Mean: {np.mean(null_rs):.4f} (expected ~0)")
    log(f"    SD: {np.std(null_rs):.4f}")
    log(f"    Range: [{np.min(null_rs):.4f}, {np.max(null_rs):.4f}]")
    log(f"\n  Observed r: {r_original:.4f}")
    log(f"  p_permutation (two-tailed): {p_permutation:.4f}")
    log(f"  p_parametric: {p_original:.4f}")

    perm_confirms_parametric = (p_permutation < 0.05) == (p_original < 0.05)
    log(f"\n  Permutation confirms parametric? {perm_confirms_parametric}")
    if p_permutation < 0.05:
        log(f"  ✓ SIGNIFICANT: Permutation p < 0.05")
    else:
        log(f"  ⚠️ NOT SIGNIFICANT: Permutation p >= 0.05")

    # Save permutation results
    perm_df = pd.DataFrame({
        'metric': ['r_observed', 'p_parametric', 'p_permutation', 'n_permutations',
                   'null_mean', 'null_sd', 'perm_confirms_parametric'],
        'value': [r_original, p_original, p_permutation, len(null_rs),
                  np.mean(null_rs), np.std(null_rs), perm_confirms_parametric]
    })
    perm_df.to_csv(DATA_DIR / "step06_permutation_results.csv", index=False)
    log(f"  ✓ Saved: step06_permutation_results.csv")

    # =========================================================================
    # STEP 5: Outlier Sensitivity Analysis
    # =========================================================================
    log("\n[STEP 5] Outlier Sensitivity Analysis")
    log("-" * 60)

    outlier_results = outlier_sensitivity_analysis(x, y, z, threshold_sd=2.5)

    log(f"  Threshold: {outlier_results['threshold_sd']} SD from mean")
    log(f"  N outliers detected: {outlier_results['n_outliers']}")
    if outlier_results['n_outliers'] > 0:
        log(f"  Outlier indices: {list(outlier_results['outlier_indices'])}")
        outlier_uids = data.iloc[outlier_results['outlier_indices']]['UID'].tolist()
        log(f"  Outlier UIDs: {outlier_uids}")

    log(f"\n  Original r: {outlier_results['r_original']:.4f} (p={outlier_results['p_original']:.4f})")
    log(f"  Without outliers r: {outlier_results['r_clean']:.4f} (p={outlier_results['p_clean']:.4f})")
    log(f"  Delta r: {outlier_results['delta_r']:+.4f}")

    robust_to_outliers = np.abs(outlier_results['delta_r']) < 0.05
    if robust_to_outliers:
        log(f"  ✓ ROBUST: Change < 0.05 with outliers removed")
    else:
        log(f"  ⚠️ SENSITIVE: Change >= 0.05 with outliers removed")

    # Save outlier results
    outlier_df = pd.DataFrame({
        'metric': ['n_outliers', 'threshold_sd', 'r_original', 'p_original',
                   'r_clean', 'p_clean', 'delta_r', 'robust_to_outliers'],
        'value': [outlier_results['n_outliers'], outlier_results['threshold_sd'],
                  outlier_results['r_original'], outlier_results['p_original'],
                  outlier_results['r_clean'], outlier_results['p_clean'],
                  outlier_results['delta_r'], robust_to_outliers]
    })
    outlier_df.to_csv(DATA_DIR / "step06_outlier_sensitivity.csv", index=False)
    log(f"  ✓ Saved: step06_outlier_sensitivity.csv")

    # =========================================================================
    # STEP 6: Overall Robustness Assessment
    # =========================================================================
    log("\n" + "=" * 80)
    log("[SUMMARY] Robustness Assessment")
    log("=" * 80)

    # Compile robustness criteria
    criteria = {
        'Bootstrap CI excludes 0': ci_excludes_zero,
        'LOO all same direction': all_same_direction,
        'Permutation p < 0.05': p_permutation < 0.05,
        'Robust to outliers': robust_to_outliers
    }

    n_passed = sum(criteria.values())
    n_total = len(criteria)

    log(f"\n  Robustness Criteria: {n_passed}/{n_total} passed")
    for criterion, passed in criteria.items():
        status = "✓" if passed else "✗"
        log(f"    {status} {criterion}")

    # Overall assessment
    if n_passed == 4:
        overall = "FULLY ROBUST"
        interpretation = "Finding passes all robustness checks. p=0.034 can be trusted."
    elif n_passed >= 3:
        overall = "SUBSTANTIALLY ROBUST"
        interpretation = "Finding passes most checks. p=0.034 is reasonably trustworthy."
    elif n_passed >= 2:
        overall = "MARGINALLY ROBUST"
        interpretation = "Finding passes some checks. p=0.034 should be interpreted with caution."
    else:
        overall = "NOT ROBUST"
        interpretation = "Finding fails most checks. p=0.034 may be spurious."

    log(f"\n  Overall Assessment: {overall}")
    log(f"  Interpretation: {interpretation}")

    # =========================================================================
    # STEP 7: Create Results Summary
    # =========================================================================
    log("\n[STEP 7] Create Results Summary")
    log("-" * 60)

    summary_content = f"""# RQ 6.7.2 Robustness Analysis

**Generated:** {datetime.now().isoformat()}
**Task:** T1.2 from rq_rework.md - Bootstrap robustness for marginal p-value

---

## Original Finding

| Metric | Value |
|--------|-------|
| Partial r | {r_original:.4f} |
| p-value | {p_original:.4f} |
| Finding | SD_confidence → SD_accuracy \\| mean_accuracy |

---

## Robustness Tests

### 1. Bootstrap 95% CI (N=10,000)

| Metric | Value |
|--------|-------|
| CI Lower | {ci_lower:.4f} |
| CI Upper | {ci_upper:.4f} |
| CI Excludes 0 | {'Yes ✓' if ci_excludes_zero else 'No ✗'} |
| Bootstrap Mean | {np.mean(bootstrap_rs):.4f} |
| Bootstrap SD | {np.std(bootstrap_rs):.4f} |

### 2. Leave-One-Out (N=100)

| Metric | Value |
|--------|-------|
| LOO Mean | {np.mean(loo_rs):.4f} |
| LOO SD | {np.std(loo_rs):.4f} |
| LOO Range | [{np.min(loo_rs):.4f}, {np.max(loo_rs):.4f}] |
| N Positive | {n_positive} |
| N Negative | {n_negative} |
| All Same Direction | {'Yes ✓' if all_same_direction else 'No ✗'} |

Most influential observation: {most_influential_uid} (LOO r = {most_influential_r:.4f})

### 3. Permutation Test (N=1,000)

| Metric | Value |
|--------|-------|
| p_permutation | {p_permutation:.4f} |
| p_parametric | {p_original:.4f} |
| Confirms Parametric | {'Yes ✓' if perm_confirms_parametric else 'No ✗'} |
| Null Mean | {np.mean(null_rs):.4f} |
| Null SD | {np.std(null_rs):.4f} |

### 4. Outlier Sensitivity

| Metric | Value |
|--------|-------|
| Threshold | {outlier_results['threshold_sd']} SD |
| N Outliers | {outlier_results['n_outliers']} |
| Original r | {outlier_results['r_original']:.4f} |
| Clean r | {outlier_results['r_clean']:.4f} |
| Delta r | {outlier_results['delta_r']:+.4f} |
| Robust | {'Yes ✓' if robust_to_outliers else 'No ✗'} |

---

## Overall Assessment

**Robustness Score:** {n_passed}/4 criteria passed

| Criterion | Status |
|-----------|--------|
| Bootstrap CI excludes 0 | {'✓' if criteria['Bootstrap CI excludes 0'] else '✗'} |
| LOO all same direction | {'✓' if criteria['LOO all same direction'] else '✗'} |
| Permutation p < 0.05 | {'✓' if criteria['Permutation p < 0.05'] else '✗'} |
| Robust to outliers | {'✓' if criteria['Robust to outliers'] else '✗'} |

**Overall:** {overall}

**Interpretation:** {interpretation}

---

## Files Created

- `data/step06_bootstrap_results.csv`
- `data/step06_loo_results.csv`
- `data/step06_permutation_results.csv`
- `data/step06_outlier_sensitivity.csv`
- `results/robustness_analysis.md` (this file)
"""

    with open(RESULTS_DIR / "robustness_analysis.md", 'w') as f:
        f.write(summary_content)
    log(f"  ✓ Saved: results/robustness_analysis.md")

    log(f"\nCompleted: {datetime.now().isoformat()}")

    return {
        'r_original': r_original,
        'p_original': p_original,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_excludes_zero': ci_excludes_zero,
        'all_same_direction': all_same_direction,
        'p_permutation': p_permutation,
        'robust_to_outliers': robust_to_outliers,
        'overall': overall,
        'n_passed': n_passed
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        log(f"\n[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
