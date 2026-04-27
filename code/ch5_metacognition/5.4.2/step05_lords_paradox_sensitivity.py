"""
RQ 6.4.2 - Step 05: Lord's Paradox Sensitivity Check

PURPOSE:
Paradigm calibration differences may be regression artifacts if paradigms differ
in baseline accuracy. Lord's paradox: group differences in change/difference scores
can be artifacts of baseline differences.

CONCERN:
- Calibration = z(confidence) - z(accuracy) (difference score)
- If accuracy differs by paradigm (IRE > IFR > ICR), z-standardization may create
  spurious calibration differences
- Need to verify that paradigm calibration differences are genuine, not artifacts

METHODS:
1. ANCOVA: calibration ~ paradigm + baseline_accuracy (partial out accuracy)
2. Within-paradigm z-standardization (z-score separately per paradigm)
3. Compare calibration differences: Original vs ANCOVA vs Within-paradigm

INPUT:
- results/ch6/6.4.2/data/step00_calibration_by_paradigm.csv
  - Columns: UID, TEST, Paradigm, theta_accuracy, theta_confidence, TSVR_hours,
             theta_accuracy_z, theta_confidence_z, calibration, abs_calibration

OUTPUT:
- data/step05_lords_paradox_check.csv
- results/sensitivity_analysis.md

Author: Claude Code
Date: 2025-12-14
RQ: ch6/6.4.2
Task: T1.3 from rq_rework.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = RQ_DIR / "logs" / "step05_lords_paradox_sensitivity.log"
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


def main():
    log("=" * 80)
    log("RQ 6.4.2 - Step 05: Lord's Paradox Sensitivity Check")
    log(f"Started: {datetime.now().isoformat()}")
    log("=" * 80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    log("\n[STEP 1] Load Calibration Data")
    log("-" * 60)

    df = pd.read_csv(DATA_DIR / "step00_calibration_by_paradigm.csv")
    log(f"  ✓ Loaded {len(df)} observations")
    log(f"  ✓ Paradigms: {df['Paradigm'].unique().tolist()}")
    log(f"  ✓ N per paradigm: {df.groupby('Paradigm').size().to_dict()}")

    # =========================================================================
    # STEP 2: Document Baseline Accuracy Differences
    # =========================================================================
    log("\n[STEP 2] Document Baseline Accuracy Differences by Paradigm")
    log("-" * 60)

    accuracy_by_paradigm = df.groupby('Paradigm')['theta_accuracy'].agg(['mean', 'std'])
    log(f"\n  Accuracy (theta) by Paradigm:")
    for paradigm in ['IFR', 'ICR', 'IRE']:
        if paradigm in accuracy_by_paradigm.index:
            m = accuracy_by_paradigm.loc[paradigm, 'mean']
            s = accuracy_by_paradigm.loc[paradigm, 'std']
            log(f"    {paradigm}: M={m:.3f}, SD={s:.3f}")

    # ANOVA for accuracy differences
    accuracy_groups = [df[df['Paradigm'] == p]['theta_accuracy'].values
                       for p in ['IFR', 'ICR', 'IRE']]
    F_acc, p_acc = stats.f_oneway(*accuracy_groups)
    log(f"\n  ANOVA for accuracy ~ paradigm: F={F_acc:.2f}, p={p_acc:.4f}")

    if p_acc < 0.05:
        log(f"  ⚠️ Significant accuracy differences between paradigms")
        log(f"     This raises Lord's paradox concern for calibration differences")
    else:
        log(f"  ✓ No significant accuracy differences - Lord's paradox less likely")

    # =========================================================================
    # STEP 3: Original Calibration Analysis (Pooled Z-Standardization)
    # =========================================================================
    log("\n[STEP 3] Original Calibration (Pooled Z-Standardization)")
    log("-" * 60)

    original_calibration = df.groupby('Paradigm')['calibration'].agg(['mean', 'std'])
    log(f"\n  Original Calibration by Paradigm:")
    for paradigm in ['IFR', 'ICR', 'IRE']:
        if paradigm in original_calibration.index:
            m = original_calibration.loc[paradigm, 'mean']
            s = original_calibration.loc[paradigm, 'std']
            log(f"    {paradigm}: M={m:+.4f}, SD={s:.4f}")

    # ANOVA for original calibration
    calib_groups = [df[df['Paradigm'] == p]['calibration'].values
                    for p in ['IFR', 'ICR', 'IRE']]
    F_orig, p_orig = stats.f_oneway(*calib_groups)
    log(f"\n  ANOVA for calibration ~ paradigm: F={F_orig:.2f}, p={p_orig:.4f}")

    # =========================================================================
    # STEP 4: Method 1 - ANCOVA (Partial Out Accuracy)
    # =========================================================================
    log("\n[STEP 4] Method 1: ANCOVA (Partial Out Accuracy)")
    log("-" * 60)

    # Create centered accuracy covariate
    df['accuracy_centered'] = df['theta_accuracy'] - df['theta_accuracy'].mean()

    # Fit ANCOVA model
    ancova_model = smf.ols('calibration ~ C(Paradigm) + accuracy_centered', data=df).fit()

    log(f"\n  ANCOVA Model: calibration ~ paradigm + accuracy")
    log(f"  R² = {ancova_model.rsquared:.4f}")

    # Extract paradigm effects
    log(f"\n  ANCOVA Coefficients:")
    for term in ancova_model.params.index:
        coef = ancova_model.params[term]
        pval = ancova_model.pvalues[term]
        log(f"    {term}: β={coef:+.4f}, p={pval:.4f}")

    # Test paradigm effect after controlling for accuracy
    from scipy.stats import f as f_dist

    # Reduced model (accuracy only)
    reduced_model = smf.ols('calibration ~ accuracy_centered', data=df).fit()

    # Likelihood ratio test
    lr_stat = 2 * (ancova_model.llf - reduced_model.llf)
    df_diff = ancova_model.df_model - reduced_model.df_model
    p_ancova = 1 - stats.chi2.cdf(lr_stat, df_diff)

    log(f"\n  LR test for paradigm effect (controlling accuracy):")
    log(f"    χ²({df_diff}) = {lr_stat:.2f}, p = {p_ancova:.4f}")

    # Get paradigm means from ANCOVA (adjusted means)
    adjusted_means = {}
    for paradigm in ['IFR', 'ICR', 'IRE']:
        if paradigm == 'IFR':
            # Reference category
            adjusted_means[paradigm] = ancova_model.params['Intercept']
        else:
            # Add coefficient to intercept
            coef_name = f'C(Paradigm)[T.{paradigm}]'
            if coef_name in ancova_model.params:
                adjusted_means[paradigm] = ancova_model.params['Intercept'] + ancova_model.params[coef_name]
            else:
                adjusted_means[paradigm] = np.nan

    log(f"\n  ANCOVA-Adjusted Calibration Means:")
    for paradigm, mean in adjusted_means.items():
        log(f"    {paradigm}: {mean:+.4f}")

    # =========================================================================
    # STEP 5: Method 2 - Within-Paradigm Z-Standardization
    # =========================================================================
    log("\n[STEP 5] Method 2: Within-Paradigm Z-Standardization")
    log("-" * 60)

    # Z-standardize accuracy and confidence WITHIN each paradigm
    df['z_acc_within'] = np.nan
    df['z_conf_within'] = np.nan

    for paradigm in ['IFR', 'ICR', 'IRE']:
        mask = df['Paradigm'] == paradigm
        df.loc[mask, 'z_acc_within'] = stats.zscore(df.loc[mask, 'theta_accuracy'])
        df.loc[mask, 'z_conf_within'] = stats.zscore(df.loc[mask, 'theta_confidence'])

    # Compute within-paradigm calibration
    df['calibration_within'] = df['z_conf_within'] - df['z_acc_within']

    within_calibration = df.groupby('Paradigm')['calibration_within'].agg(['mean', 'std'])
    log(f"\n  Within-Paradigm Calibration:")
    for paradigm in ['IFR', 'ICR', 'IRE']:
        if paradigm in within_calibration.index:
            m = within_calibration.loc[paradigm, 'mean']
            s = within_calibration.loc[paradigm, 'std']
            log(f"    {paradigm}: M={m:+.4f}, SD={s:.4f}")

    # ANOVA for within-paradigm calibration
    within_groups = [df[df['Paradigm'] == p]['calibration_within'].values
                     for p in ['IFR', 'ICR', 'IRE']]
    F_within, p_within = stats.f_oneway(*within_groups)
    log(f"\n  ANOVA for within-paradigm calibration: F={F_within:.2f}, p={p_within:.4f}")

    # =========================================================================
    # STEP 6: Compare Methods
    # =========================================================================
    log("\n[STEP 6] Compare Methods")
    log("-" * 60)

    comparison = pd.DataFrame({
        'Method': ['Original (Pooled Z)', 'ANCOVA (Acc Controlled)', 'Within-Paradigm Z'],
        'IFR_mean': [
            original_calibration.loc['IFR', 'mean'] if 'IFR' in original_calibration.index else np.nan,
            adjusted_means.get('IFR', np.nan),
            within_calibration.loc['IFR', 'mean'] if 'IFR' in within_calibration.index else np.nan
        ],
        'ICR_mean': [
            original_calibration.loc['ICR', 'mean'] if 'ICR' in original_calibration.index else np.nan,
            adjusted_means.get('ICR', np.nan),
            within_calibration.loc['ICR', 'mean'] if 'ICR' in within_calibration.index else np.nan
        ],
        'IRE_mean': [
            original_calibration.loc['IRE', 'mean'] if 'IRE' in original_calibration.index else np.nan,
            adjusted_means.get('IRE', np.nan),
            within_calibration.loc['IRE', 'mean'] if 'IRE' in within_calibration.index else np.nan
        ],
        'F_stat': [F_orig, lr_stat/df_diff, F_within],  # Approximate for ANCOVA
        'p_value': [p_orig, p_ancova, p_within]
    })

    log(f"\n  Method Comparison Table:")
    log(f"  {'Method':<25} {'IFR':>8} {'ICR':>8} {'IRE':>8} {'F':>8} {'p':>8}")
    log(f"  {'-'*72}")
    for _, row in comparison.iterrows():
        log(f"  {row['Method']:<25} {row['IFR_mean']:>+8.4f} {row['ICR_mean']:>+8.4f} {row['IRE_mean']:>+8.4f} {row['F_stat']:>8.2f} {row['p_value']:>8.4f}")

    # =========================================================================
    # STEP 7: Assess Robustness
    # =========================================================================
    log("\n[STEP 7] Assess Robustness to Lord's Paradox")
    log("-" * 60)

    # Check if all methods agree on significance
    all_significant = all([p_orig < 0.05, p_ancova < 0.05, p_within < 0.05])
    all_nonsignificant = all([p_orig >= 0.05, p_ancova >= 0.05, p_within >= 0.05])
    methods_agree = all_significant or all_nonsignificant

    # Check if effect direction is consistent
    # Compute ICR - IFR difference for each method
    diff_orig = (original_calibration.loc['ICR', 'mean'] - original_calibration.loc['IFR', 'mean']) if 'ICR' in original_calibration.index and 'IFR' in original_calibration.index else np.nan
    diff_ancova = adjusted_means.get('ICR', np.nan) - adjusted_means.get('IFR', np.nan) if adjusted_means.get('ICR') is not None and adjusted_means.get('IFR') is not None else np.nan
    diff_within = (within_calibration.loc['ICR', 'mean'] - within_calibration.loc['IFR', 'mean']) if 'ICR' in within_calibration.index and 'IFR' in within_calibration.index else np.nan

    directions = [np.sign(diff_orig), np.sign(diff_ancova), np.sign(diff_within)]
    direction_consistent = len(set([d for d in directions if not np.isnan(d)])) <= 1

    log(f"\n  ICR - IFR Difference by Method:")
    log(f"    Original: {diff_orig:+.4f}")
    log(f"    ANCOVA: {diff_ancova:+.4f}")
    log(f"    Within-Paradigm: {diff_within:+.4f}")

    log(f"\n  Robustness Checks:")
    log(f"    Methods agree on significance? {methods_agree}")
    log(f"    Effect direction consistent? {direction_consistent}")

    # Interpretation
    if methods_agree and direction_consistent:
        robustness = "ROBUST"
        interpretation = "Paradigm calibration differences are genuine, not artifacts of baseline accuracy differences. All three approaches yield consistent conclusions."
    elif direction_consistent:
        robustness = "PARTIALLY ROBUST"
        interpretation = "Effect direction is consistent across methods, but significance differs. Finding may be genuine but effect size is uncertain."
    else:
        robustness = "NOT ROBUST - LORD'S PARADOX DETECTED"
        interpretation = "Methods yield inconsistent conclusions. Calibration differences may be artifacts of baseline accuracy differences. Exercise caution in interpretation."

    log(f"\n  Robustness Assessment: {robustness}")
    log(f"  Interpretation: {interpretation}")

    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    log("\n[STEP 8] Save Results")
    log("-" * 60)

    # Compile results
    results_df = pd.DataFrame({
        'metric': [
            'accuracy_F', 'accuracy_p',
            'original_F', 'original_p',
            'original_IFR', 'original_ICR', 'original_IRE',
            'ancova_p', 'ancova_IFR', 'ancova_ICR', 'ancova_IRE',
            'within_F', 'within_p',
            'within_IFR', 'within_ICR', 'within_IRE',
            'diff_orig_ICR_IFR', 'diff_ancova_ICR_IFR', 'diff_within_ICR_IFR',
            'methods_agree', 'direction_consistent', 'robustness'
        ],
        'value': [
            F_acc, p_acc,
            F_orig, p_orig,
            original_calibration.loc['IFR', 'mean'] if 'IFR' in original_calibration.index else np.nan,
            original_calibration.loc['ICR', 'mean'] if 'ICR' in original_calibration.index else np.nan,
            original_calibration.loc['IRE', 'mean'] if 'IRE' in original_calibration.index else np.nan,
            p_ancova, adjusted_means.get('IFR', np.nan), adjusted_means.get('ICR', np.nan), adjusted_means.get('IRE', np.nan),
            F_within, p_within,
            within_calibration.loc['IFR', 'mean'] if 'IFR' in within_calibration.index else np.nan,
            within_calibration.loc['ICR', 'mean'] if 'ICR' in within_calibration.index else np.nan,
            within_calibration.loc['IRE', 'mean'] if 'IRE' in within_calibration.index else np.nan,
            diff_orig, diff_ancova, diff_within,
            methods_agree, direction_consistent, robustness
        ]
    })

    results_df.to_csv(DATA_DIR / "step05_lords_paradox_check.csv", index=False)
    log(f"  ✓ Saved: step05_lords_paradox_check.csv")

    # Save updated data with within-paradigm calibration
    df.to_csv(DATA_DIR / "step05_calibration_with_within.csv", index=False)
    log(f"  ✓ Saved: step05_calibration_with_within.csv")

    # Create markdown summary
    summary_content = f"""# RQ 6.4.2 Lord's Paradox Sensitivity Analysis

**Generated:** {datetime.now().isoformat()}
**Task:** T1.3 from rq_rework.md

---

## Background

**Concern:** Calibration = z(confidence) - z(accuracy). If accuracy differs by paradigm,
z-standardization pooled across paradigms may create spurious calibration differences
(Lord's paradox).

---

## Baseline Accuracy Differences

| Paradigm | Accuracy (theta) Mean | SD |
|----------|----------------------|-----|
| IFR | {accuracy_by_paradigm.loc['IFR', 'mean'] if 'IFR' in accuracy_by_paradigm.index else 'N/A':.3f} | {accuracy_by_paradigm.loc['IFR', 'std'] if 'IFR' in accuracy_by_paradigm.index else 'N/A':.3f} |
| ICR | {accuracy_by_paradigm.loc['ICR', 'mean'] if 'ICR' in accuracy_by_paradigm.index else 'N/A':.3f} | {accuracy_by_paradigm.loc['ICR', 'std'] if 'ICR' in accuracy_by_paradigm.index else 'N/A':.3f} |
| IRE | {accuracy_by_paradigm.loc['IRE', 'mean'] if 'IRE' in accuracy_by_paradigm.index else 'N/A':.3f} | {accuracy_by_paradigm.loc['IRE', 'std'] if 'IRE' in accuracy_by_paradigm.index else 'N/A':.3f} |

ANOVA: F = {F_acc:.2f}, p = {p_acc:.4f} {'⚠️ Significant' if p_acc < 0.05 else '✓ Not significant'}

---

## Method Comparison

| Method | IFR | ICR | IRE | F/χ² | p |
|--------|-----|-----|-----|------|---|
| Original (Pooled Z) | {original_calibration.loc['IFR', 'mean'] if 'IFR' in original_calibration.index else 'N/A':+.4f} | {original_calibration.loc['ICR', 'mean'] if 'ICR' in original_calibration.index else 'N/A':+.4f} | {original_calibration.loc['IRE', 'mean'] if 'IRE' in original_calibration.index else 'N/A':+.4f} | {F_orig:.2f} | {p_orig:.4f} |
| ANCOVA (Acc Controlled) | {adjusted_means.get('IFR', np.nan):+.4f} | {adjusted_means.get('ICR', np.nan):+.4f} | {adjusted_means.get('IRE', np.nan):+.4f} | {lr_stat:.2f} | {p_ancova:.4f} |
| Within-Paradigm Z | {within_calibration.loc['IFR', 'mean'] if 'IFR' in within_calibration.index else 'N/A':+.4f} | {within_calibration.loc['ICR', 'mean'] if 'ICR' in within_calibration.index else 'N/A':+.4f} | {within_calibration.loc['IRE', 'mean'] if 'IRE' in within_calibration.index else 'N/A':+.4f} | {F_within:.2f} | {p_within:.4f} |

---

## ICR - IFR Contrast

| Method | Difference | Direction |
|--------|------------|-----------|
| Original | {diff_orig:+.4f} | {'Negative' if diff_orig < 0 else 'Positive'} |
| ANCOVA | {diff_ancova:+.4f} | {'Negative' if diff_ancova < 0 else 'Positive'} |
| Within-Paradigm | {diff_within:+.4f} | {'Negative' if diff_within < 0 else 'Positive'} |

---

## Robustness Assessment

| Check | Result |
|-------|--------|
| Methods agree on significance | {'Yes ✓' if methods_agree else 'No ✗'} |
| Effect direction consistent | {'Yes ✓' if direction_consistent else 'No ✗'} |

**Overall:** {robustness}

**Interpretation:** {interpretation}

---

## Files Created

- `data/step05_lords_paradox_check.csv`
- `data/step05_calibration_with_within.csv`
- `results/sensitivity_analysis.md` (this file)
"""

    with open(RESULTS_DIR / "sensitivity_analysis.md", 'w') as f:
        f.write(summary_content)
    log(f"  ✓ Saved: results/sensitivity_analysis.md")

    log(f"\nCompleted: {datetime.now().isoformat()}")

    return {
        'robustness': robustness,
        'methods_agree': methods_agree,
        'direction_consistent': direction_consistent
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        log(f"\n[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
