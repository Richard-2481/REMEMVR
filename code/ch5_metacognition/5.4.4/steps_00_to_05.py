"""
RQ 6.4.4 - ICC Decomposition by Paradigm (Steps 00-05)

PURPOSE:
Decompose variance in confidence trajectories into trait-like (intercept: baseline confidence)
and state-like (residual: within-person fluctuation) components SEPARATELY for each paradigm.
Tests whether Free Recall shows highest ICC_slope (individual differences in forgetting rate)
due to higher cognitive demand, or whether all paradigms show minimal slope variance.

KEY QUESTION:
Is confidence decline more trait-like for some paradigms than others?

INPUT:
- results/ch6/6.4.1/data/step04_lmm_input.csv (1200 rows: 100 participants × 4 tests × 3 paradigms)
  - Columns: composite_ID, UID, test, TSVR_hours, log_TSVR, paradigm, theta

OUTPUT:
- data/step00_lmm_input.csv (verified input data)
- data/step01_lmm_ifr_summary.txt, step01_lmm_icr_summary.txt, step01_lmm_ire_summary.txt
- data/step02_variance_components.csv (variance components for all 3 paradigms)
- data/step03_icc_estimates.csv (ICC values for all 3 paradigms)
- data/step04_paradigm_icc_comparison.csv (pairwise ICC comparisons)
- data/step04_paradigm_summary.txt (paradigm ranking and interpretation)
- data/step05_ch5_comparison.csv (comparison to Ch5 5.3.7 accuracy ICC)
- data/step05_ch5_summary.txt (accuracy vs confidence interpretation)

METHODOLOGY:
- Fit 3 paradigm-stratified LMMs with random slopes (log_TSVR | UID)
- Extract variance components from cov_re matrix per paradigm
- Compute ICCs following Hoffman & Stawski (2009) per paradigm
- Compare ICC_slope across paradigms (test if Free Recall highest)
- Compare to Ch5 5.3.7 (test if confidence reveals more variance than accuracy)

Author: Claude Code
Date: 2025-12-12
RQ: ch6/6.4.4
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"
DATA_DIR = RQ_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
(RQ_DIR / "logs").mkdir(exist_ok=True)

# Paradigm mapping for display
PARADIGM_DISPLAY = {
    'IFR': 'Free Recall',
    'ICR': 'Cued Recall',
    'IRE': 'Recognition'
}

# Ch5 paradigm names (lowercase)
CH5_PARADIGM_MAP = {
    'IFR': 'free_recall',
    'ICR': 'cued_recall',
    'IRE': 'recognition'
}


def log(msg):
    """Log message to file and stdout with flush"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_import_data():
    """
    Step 00: Import theta_confidence data from RQ 6.4.1, verify structure

    Input: results/ch6/6.4.1/data/step04_lmm_input.csv
    Output: data/step00_lmm_input.csv (verified copy)
    """
    log("=" * 80)
    log("[STEP 00] Import Theta Confidence Data from RQ 6.4.1")
    log("=" * 80)

    # Load LMM input from RQ 6.4.1
    input_path = PROJECT_ROOT / "results" / "ch6" / "6.4.1" / "data" / "step04_lmm_input.csv"
    log(f"[LOAD] Loading LMM input from RQ 6.4.1: {input_path}")

    df = pd.read_csv(input_path)
    log(f"  ✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    log(f"  ✓ Columns: {list(df.columns)}")

    # Verify expected structure
    expected_rows = 1200  # 100 participants × 4 tests × 3 paradigms
    if len(df) != expected_rows:
        log(f"[WARNING] Expected {expected_rows} rows, found {len(df)}")

    # Check paradigms
    paradigms = df['paradigm'].unique()
    log(f"  ✓ Paradigms: {list(paradigms)}")

    if set(paradigms) != {'IFR', 'ICR', 'IRE'}:
        log(f"[ERROR] Expected paradigms IFR, ICR, IRE, found {set(paradigms)}")
        raise ValueError("Missing paradigms")

    # Count per paradigm
    paradigm_counts = df.groupby('paradigm').size()
    log(f"  ✓ Rows per paradigm:")
    for p, count in paradigm_counts.items():
        log(f"      {p}: {count} rows ({count/4:.0f} participants × 4 tests)")

    # Verify UID count
    n_uids = df['UID'].nunique()
    log(f"  ✓ N participants: {n_uids}")

    # Check TSVR range
    log(f"  ✓ TSVR_hours range: [{df['TSVR_hours'].min():.2f}, {df['TSVR_hours'].max():.2f}] hours")
    log(f"  ✓ log_TSVR range: [{df['log_TSVR'].min():.4f}, {df['log_TSVR'].max():.4f}]")

    # Check theta range (column is 'theta' in 6.4.1)
    theta_col = 'theta'
    log(f"  ✓ {theta_col} range: [{df[theta_col].min():.4f}, {df[theta_col].max():.4f}]")
    log(f"  ✓ {theta_col} mean: {df[theta_col].mean():.4f}")

    # Save verified copy
    output_path = DATA_DIR / "step00_lmm_input.csv"
    df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return df


def step01_fit_paradigm_lmms(df):
    """
    Step 01: Fit 3 separate LMMs (one per paradigm) with random slopes

    Model: theta ~ log_TSVR + (log_TSVR | UID)

    NOTE: Using log_TSVR (already in data from 6.4.1) per Decision D070
    """
    import statsmodels.formula.api as smf

    log("\n" + "=" * 80)
    log("[STEP 01] Fit Paradigm-Stratified LMMs with Random Slopes")
    log("=" * 80)

    models = {}
    paradigms = ['IFR', 'ICR', 'IRE']

    for paradigm in paradigms:
        log(f"\n[{paradigm}] Fitting LMM for {PARADIGM_DISPLAY[paradigm]}...")

        # Filter to this paradigm
        df_p = df[df['paradigm'] == paradigm].copy()
        log(f"  ✓ N rows: {len(df_p)} ({df_p['UID'].nunique()} participants × {df_p['test'].nunique()} tests)")

        # Fit LMM: theta ~ log_TSVR + (log_TSVR | UID)
        formula = "theta ~ log_TSVR"
        re_formula = "~log_TSVR"  # Random intercept + random slope on log_TSVR

        log(f"  Formula: {formula}")
        log(f"  Random effects: (1 + log_TSVR | UID)")

        model = smf.mixedlm(formula, df_p, groups=df_p["UID"], re_formula=re_formula)

        # Fit with method='powell' for better convergence (learned from previous RQs)
        try:
            result = model.fit(reml=False, method='powell')
        except Exception as e:
            log(f"  [FALLBACK] Powell failed, trying default optimizer")
            result = model.fit(reml=False)

        models[paradigm] = result

        log(f"  ✓ Converged: {result.converged}")
        log(f"  ✓ AIC: {result.aic:.4f}")
        log(f"  ✓ BIC: {result.bic:.4f}")
        log(f"  ✓ Log-likelihood: {result.llf:.4f}")

        # Save model summary
        summary_path = DATA_DIR / f"step01_lmm_{paradigm.lower()}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"RQ 6.4.4 - Step 01: LMM Summary for {PARADIGM_DISPLAY[paradigm]} ({paradigm})\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"N observations: {result.nobs}\n")
            f.write(f"N groups: {result.model.n_groups}\n")
            f.write(f"Formula: theta ~ log_TSVR + (log_TSVR | UID)\n")
            f.write(f"Method: ML (REML=False)\n\n")
            f.write("Model Summary:\n")
            f.write(str(result.summary()) + "\n\n")
            f.write("Random Effects Covariance:\n")
            f.write(str(result.cov_re) + "\n")

        log(f"  ✓ Saved: {summary_path}")

    return models


def step02_extract_variance_components(models):
    """
    Step 02: Extract variance components from each fitted LMM

    Components per paradigm:
    - var_intercept: Random intercept variance (baseline confidence differences)
    - var_slope: Random slope variance (forgetting rate differences)
    - cov_int_slope: Covariance between intercepts and slopes
    - var_residual: Residual variance (within-person fluctuation)
    """
    log("\n" + "=" * 80)
    log("[STEP 02] Extract Variance Components Per Paradigm")
    log("=" * 80)

    variance_data = []

    for paradigm, result in models.items():
        log(f"\n[{paradigm}] Extracting variance components...")

        # Extract random effects covariance matrix
        cov_re = result.cov_re
        log(f"  Random effects covariance matrix:")
        log(f"{cov_re}")

        # Get variance components
        re_names = list(cov_re.columns)
        log(f"  Random effect names: {re_names}")

        var_intercept = cov_re.iloc[0, 0]  # Group variance (intercept)
        var_slope = cov_re.iloc[1, 1] if len(re_names) > 1 else 0.0  # Slope variance
        cov_int_slope = cov_re.iloc[0, 1] if len(re_names) > 1 else 0.0  # Covariance
        var_residual = result.scale  # Residual variance

        # Compute correlation
        if var_intercept > 0 and var_slope > 0:
            cor_int_slope = cov_int_slope / np.sqrt(var_intercept * var_slope)
        else:
            cor_int_slope = 0.0

        log(f"  var_intercept: {var_intercept:.6f}")
        log(f"  var_slope: {var_slope:.6f}")
        log(f"  cov_int_slope: {cov_int_slope:.6f}")
        log(f"  cor_int_slope: {cor_int_slope:.4f}")
        log(f"  var_residual: {var_residual:.6f}")

        # Validate
        if var_intercept < 0:
            log(f"[ERROR] Negative intercept variance for {paradigm}")
            raise ValueError(f"Negative intercept variance for {paradigm}")
        if var_slope < 0:
            log(f"[WARNING] Negative slope variance for {paradigm} - setting to 0 (boundary)")
            var_slope = 0.0

        var_total = var_intercept + var_slope + var_residual

        variance_data.append({
            'paradigm': paradigm,
            'var_intercept': var_intercept,
            'var_slope': var_slope,
            'cov_int_slope': cov_int_slope,
            'cor_int_slope': cor_int_slope,
            'var_residual': var_residual,
            'var_total': var_total
        })

    # Save variance components
    variance_df = pd.DataFrame(variance_data)
    output_path = DATA_DIR / "step02_variance_components.csv"
    variance_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    return variance_df


def step03_compute_icc(variance_df, df):
    """
    Step 03: Compute ICC estimates per paradigm following Hoffman & Stawski (2009)

    ICCs:
    - ICC_intercept: Proportion of variance attributable to stable baseline differences
    - ICC_slope_simple: Proportion of slope variance relative to total change variance
    - ICC_slope_conditional: Slope variance at final timepoint (Day 6)
    """
    log("\n" + "=" * 80)
    log("[STEP 03] Compute ICC Estimates Per Paradigm")
    log("=" * 80)

    # Get time parameters
    mean_time = df['log_TSVR'].mean()
    max_time = df['log_TSVR'].max()
    log(f"\n[PARAMETERS]:")
    log(f"  Mean log_TSVR: {mean_time:.4f}")
    log(f"  Max log_TSVR (Day 6): {max_time:.4f}")

    icc_data = []

    for _, row in variance_df.iterrows():
        paradigm = row['paradigm']
        var_int = row['var_intercept']
        var_slope = row['var_slope']
        cov_is = row['cov_int_slope']
        var_res = row['var_residual']

        log(f"\n[{paradigm}] Computing ICCs...")

        # ICC_intercept: Proportion of total variance at mean time
        total_var_mean = (var_int +
                         var_slope * mean_time**2 +
                         2 * cov_is * mean_time +
                         var_res)
        ICC_intercept = var_int / total_var_mean if total_var_mean > 0 else 0

        # ICC_slope_simple: Proportion of slope variance
        ICC_slope_simple = var_slope / (var_slope + var_res) if (var_slope + var_res) > 0 else 0

        # ICC_slope_conditional: Slope variance at final timepoint
        total_var_max = (var_int +
                        var_slope * max_time**2 +
                        2 * cov_is * max_time +
                        var_res)
        ICC_slope_conditional = var_slope * max_time**2 / total_var_max if total_var_max > 0 else 0

        log(f"  ICC_intercept: {ICC_intercept:.4f}")
        log(f"  ICC_slope_simple: {ICC_slope_simple:.6f}")
        log(f"  ICC_slope_conditional: {ICC_slope_conditional:.4f}")

        # Interpret
        def interpret_icc(icc):
            if icc < 0.05:
                return "Negligible"
            elif icc < 0.10:
                return "Small"
            elif icc < 0.30:
                return "Moderate"
            else:
                return "Substantial"

        icc_data.append({
            'paradigm': paradigm,
            'ICC_intercept': ICC_intercept,
            'ICC_slope_simple': ICC_slope_simple,
            'ICC_slope_conditional': ICC_slope_conditional,
            'interpretation_intercept': interpret_icc(ICC_intercept),
            'interpretation_slope': interpret_icc(ICC_slope_simple)
        })

    # Save ICC estimates
    icc_df = pd.DataFrame(icc_data)
    output_path = DATA_DIR / "step03_icc_estimates.csv"
    icc_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    # Summary table
    log(f"\n[SUMMARY TABLE]:")
    log(f"{'Paradigm':<12} {'ICC_intercept':>14} {'ICC_slope_simple':>18} {'interpretation':>15}")
    log("-" * 60)
    for _, row in icc_df.iterrows():
        log(f"{row['paradigm']:<12} {row['ICC_intercept']:>14.4f} {row['ICC_slope_simple']:>18.6f} {row['interpretation_slope']:>15}")

    return icc_df


def step04_compare_paradigms(icc_df):
    """
    Step 04: Compare ICC_slope across paradigms

    Test hypothesis: Free Recall shows highest ICC_slope (most trait variance)
    """
    log("\n" + "=" * 80)
    log("[STEP 04] Compare ICC Across Paradigms")
    log("=" * 80)

    # Order paradigms by ICC_slope
    icc_sorted = icc_df.sort_values('ICC_slope_simple', ascending=False)
    log(f"\n[RANKING by ICC_slope_simple]:")
    for rank, (_, row) in enumerate(icc_sorted.iterrows(), 1):
        log(f"  {rank}. {row['paradigm']}: {row['ICC_slope_simple']:.6f}")

    # Get values for comparisons
    icc_dict = {row['paradigm']: row['ICC_slope_simple'] for _, row in icc_df.iterrows()}
    icc_cond_dict = {row['paradigm']: row['ICC_slope_conditional'] for _, row in icc_df.iterrows()}

    # Pairwise comparisons
    comparisons = []
    pairs = [('IFR', 'ICR'), ('IFR', 'IRE'), ('ICR', 'IRE')]

    log(f"\n[PAIRWISE COMPARISONS]:")
    for p1, p2 in pairs:
        diff_simple = icc_dict[p1] - icc_dict[p2]
        diff_cond = icc_cond_dict[p1] - icc_cond_dict[p2]

        if diff_simple > 0.001:
            direction = f"{p1} higher"
        elif diff_simple < -0.001:
            direction = f"{p2} higher"
        else:
            direction = "Equivalent"

        log(f"  {p1} - {p2}: Δ_simple = {diff_simple:+.6f}, Δ_conditional = {diff_cond:+.4f} ({direction})")

        comparisons.append({
            'comparison': f"{p1} - {p2}",
            'ICC_slope_diff_simple': diff_simple,
            'ICC_slope_diff_conditional': diff_cond,
            'direction': direction
        })

    # Save comparisons
    comparison_df = pd.DataFrame(comparisons)
    output_path = DATA_DIR / "step04_paradigm_icc_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    # Determine hypothesis outcome
    ifr_highest = (icc_dict['IFR'] >= icc_dict['ICR']) and (icc_dict['IFR'] >= icc_dict['IRE'])
    all_near_zero = all(v < 0.05 for v in icc_dict.values())

    if all_near_zero:
        pattern = "All paradigms show ICC_slope ≈ 0: Confidence decline is STATE-LIKE across all paradigms"
        hypothesis_result = "REFUTED - No paradigm shows trait-like slope variance"
    elif ifr_highest:
        pattern = f"Free Recall shows highest ICC_slope ({icc_dict['IFR']:.6f}): Some paradigm-specific trait variance"
        hypothesis_result = "SUPPORTED - Free Recall shows highest ICC_slope (as predicted)"
    else:
        highest = max(icc_dict, key=icc_dict.get)
        pattern = f"{highest} shows highest ICC_slope ({icc_dict[highest]:.6f}): Unexpected pattern"
        hypothesis_result = f"REFUTED - {highest} has highest ICC_slope, not Free Recall"

    log(f"\n[PATTERN]:")
    log(f"  {pattern}")
    log(f"\n[HYPOTHESIS TEST]:")
    log(f"  {hypothesis_result}")

    # Save summary
    summary_path = DATA_DIR / "step04_paradigm_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RQ 6.4.4 - Step 04: Paradigm ICC Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("ICC_slope Ranking (highest to lowest):\n")
        for rank, (_, row) in enumerate(icc_sorted.iterrows(), 1):
            f.write(f"  {rank}. {row['paradigm']} ({PARADIGM_DISPLAY[row['paradigm']]}): {row['ICC_slope_simple']:.6f}\n")
        f.write(f"\nPattern: {pattern}\n")
        f.write(f"\nHypothesis Test: {hypothesis_result}\n")
        f.write(f"\nInterpretation:\n")
        if all_near_zero:
            f.write("  All three paradigms show minimal slope variance (ICC_slope < 0.05).\n")
            f.write("  Confidence decline rates are state-like (within-person fluctuation)\n")
            f.write("  regardless of retrieval support level. This parallels Chapter 5\n")
            f.write("  accuracy findings where ICC_slope ≈ 0 across all paradigms.\n")
        else:
            f.write("  Some paradigm-specific slope variance detected.\n")
            f.write("  Individual differences in confidence decline may vary by paradigm.\n")

    log(f"  ✓ Saved: {summary_path}")

    return comparison_df, pattern, hypothesis_result


def step05_compare_ch5(icc_df):
    """
    Step 05: Compare to Ch5 5.3.7 accuracy ICC estimates

    Test if confidence reveals more slope variance than accuracy
    """
    log("\n" + "=" * 80)
    log("[STEP 05] Compare to Ch5 5.3.7 Accuracy ICC")
    log("=" * 80)

    # Load Ch5 5.3.7 ICC estimates
    ch5_path = PROJECT_ROOT / "results" / "ch5" / "5.3.7" / "data" / "step03_icc_estimates.csv"

    if not ch5_path.exists():
        log(f"[WARNING] Ch5 5.3.7 file not found: {ch5_path}")
        log("  Skipping Ch5 comparison (non-fatal)")
        return None

    ch5_df = pd.read_csv(ch5_path)
    log(f"  ✓ Loaded Ch5 5.3.7 ICC estimates: {len(ch5_df)} rows")

    # Reshape Ch5 data for comparison (wide format)
    ch5_wide = ch5_df.pivot(index='paradigm', columns='icc_type', values='icc_value').reset_index()
    log(f"  ✓ Ch5 paradigms: {list(ch5_wide['paradigm'].unique())}")

    # Create comparison
    comparison_data = []

    for _, row in icc_df.iterrows():
        paradigm = row['paradigm']
        ch5_paradigm = CH5_PARADIGM_MAP[paradigm]

        # Get Ch5 values (match paradigm name)
        ch5_row = ch5_wide[ch5_wide['paradigm'] == ch5_paradigm]

        if len(ch5_row) == 0:
            log(f"[WARNING] Ch5 paradigm '{ch5_paradigm}' not found")
            continue

        ch5_row = ch5_row.iloc[0]

        ICC_int_conf = row['ICC_intercept']
        ICC_int_acc = ch5_row['intercept']
        ICC_slope_conf = row['ICC_slope_simple']
        ICC_slope_acc = ch5_row['slope_simple']

        int_diff = ICC_int_conf - ICC_int_acc
        slope_diff = ICC_slope_conf - ICC_slope_acc

        # Interpretation
        if slope_diff > 0.05:
            interp = "Confidence reveals MORE slope variance than accuracy"
        elif slope_diff < -0.05:
            interp = "Confidence reveals LESS slope variance than accuracy"
        else:
            interp = "Confidence and accuracy show SIMILAR slope variance"

        log(f"\n[{paradigm}]:")
        log(f"  ICC_intercept: confidence={ICC_int_conf:.4f}, accuracy={ICC_int_acc:.4f}, diff={int_diff:+.4f}")
        log(f"  ICC_slope: confidence={ICC_slope_conf:.6f}, accuracy={ICC_slope_acc:.6f}, diff={slope_diff:+.6f}")
        log(f"  Interpretation: {interp}")

        comparison_data.append({
            'paradigm': paradigm,
            'ICC_intercept_confidence': ICC_int_conf,
            'ICC_intercept_accuracy': ICC_int_acc,
            'ICC_intercept_diff': int_diff,
            'ICC_slope_confidence': ICC_slope_conf,
            'ICC_slope_accuracy': ICC_slope_acc,
            'ICC_slope_diff': slope_diff,
            'interpretation': interp
        })

    # Save comparison
    comparison_df = pd.DataFrame(comparison_data)
    output_path = DATA_DIR / "step05_ch5_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n  ✓ Saved: {output_path}")

    # Overall pattern
    avg_slope_diff = comparison_df['ICC_slope_diff'].mean()
    if avg_slope_diff > 0.05:
        overall_pattern = "5-level confidence data reveals MORE slope variance than dichotomous accuracy"
    elif avg_slope_diff < -0.05:
        overall_pattern = "Unexpectedly, accuracy shows MORE slope variance than confidence"
    else:
        overall_pattern = "Confidence and accuracy show SIMILAR slope variance patterns"

    log(f"\n[OVERALL PATTERN]:")
    log(f"  Average ICC_slope difference: {avg_slope_diff:+.6f}")
    log(f"  {overall_pattern}")

    # Save summary
    summary_path = DATA_DIR / "step05_ch5_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RQ 6.4.4 - Step 05: Ch5 Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write("Comparison: Confidence (5-level ordinal) vs Accuracy (dichotomous)\n\n")
        f.write("Per-Paradigm ICC_slope Comparison:\n")
        for _, row in comparison_df.iterrows():
            f.write(f"  {row['paradigm']}: conf={row['ICC_slope_confidence']:.6f}, acc={row['ICC_slope_accuracy']:.6f}, diff={row['ICC_slope_diff']:+.6f}\n")
        f.write(f"\nAverage ICC_slope Difference: {avg_slope_diff:+.6f}\n")
        f.write(f"\nOverall Pattern: {overall_pattern}\n")
        f.write(f"\nTheoretical Interpretation:\n")
        if avg_slope_diff > 0.05:
            f.write("  5-level ordinal confidence data is more sensitive to individual differences\n")
            f.write("  in forgetting rates than dichotomous accuracy data. Ch5 finding of\n")
            f.write("  ICC_slope ≈ 0 was partially a measurement limitation, not purely\n")
            f.write("  a substantive finding about universal forgetting dynamics.\n")
        else:
            f.write("  Both confidence and accuracy show minimal slope variance.\n")
            f.write("  This strengthens the conclusion that forgetting trajectories\n")
            f.write("  are fundamentally state-like (universal decline pattern)\n")
            f.write("  regardless of measurement precision.\n")

    log(f"  ✓ Saved: {summary_path}")

    return comparison_df


if __name__ == "__main__":
    try:
        log("=" * 80)
        log(f"RQ 6.4.4 - ICC Decomposition by Paradigm")
        log(f"Started: {datetime.now().isoformat()}")
        log("=" * 80)

        # Step 0: Import data
        df = step00_import_data()

        # Step 1: Fit paradigm-stratified LMMs
        models = step01_fit_paradigm_lmms(df)

        # Step 2: Extract variance components
        variance_df = step02_extract_variance_components(models)

        # Step 3: Compute ICC estimates
        icc_df = step03_compute_icc(variance_df, df)

        # Step 4: Compare across paradigms
        comparison_df, pattern, hypothesis_result = step04_compare_paradigms(icc_df)

        # Step 5: Compare to Ch5
        ch5_comparison = step05_compare_ch5(icc_df)

        log("\n" + "=" * 80)
        log("[SUCCESS] RQ 6.4.4 Complete")
        log("=" * 80)
        log(f"\n[SUMMARY]:")
        log(f"  Pattern: {pattern}")
        log(f"  Hypothesis: {hypothesis_result}")
        log(f"\n  Completed: {datetime.now().isoformat()}")

    except Exception as e:
        log(f"\n[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        raise
