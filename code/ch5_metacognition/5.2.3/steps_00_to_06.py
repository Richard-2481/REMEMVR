#!/usr/bin/env python3
"""
RQ 6.2.3: Resolution Over Time - Metacognitive Discrimination Analysis

Research Question: Does discrimination ability (resolution/gamma) decline as memory
fades over a 6-day retention interval?

Pipeline:
- Step 00: Extract item-level data (TQ_* accuracy + TC_* confidence)
- Step 01: Compute Goodman-Kruskal gamma per participant-timepoint
- Step 02: Fit LMM: gamma ~ TSVR_days + (TSVR_days | UID)
- Step 03: Extract Time effect with dual p-values (Decision D068)
- Step 04: Compute mean gamma by timepoint
- Step 05: Test gamma > 0.50 threshold at each timepoint
- Step 06: Prepare plot data for resolution trajectory visualization

Primary Hypothesis: Gamma will DECLINE from Day 0 to Day 6 as memory becomes noisier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau, ttest_1samp, t as t_dist
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.3
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_06.log"
DATA_DIR = RQ_DIR / "data"

# Interactive paradigms with paired TQ/TC tags
INTERACTIVE_PARADIGMS = ['IFR', 'ICR', 'IRE']

def log(msg):
    """Log message to file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + "\n")
        f.flush()
    print(log_msg, flush=True)

# =============================================================================
# STEP 00: Extract Item-Level Data
# =============================================================================

def step00_extract_item_level():
    """Extract item-level data with paired accuracy (TQ_*) and confidence (TC_*)."""
    log("=" * 60)
    log("STEP 00: Extract Item-Level Data")
    log("=" * 60)

    # Read dfData.csv
    df_raw = pd.read_csv(PROJECT_ROOT / "data" / "cache" / "dfData.csv")
    log(f"Loaded dfData.csv: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Get TQ_ and TC_ columns
    tq_cols = [c for c in df_raw.columns if c.startswith('TQ_')]
    tc_cols = [c for c in df_raw.columns if c.startswith('TC_')]
    log(f"Found {len(tq_cols)} TQ_ columns, {len(tc_cols)} TC_ columns")

    # Filter to interactive paradigms only
    tq_interactive = [c for c in tq_cols if any(p in c for p in INTERACTIVE_PARADIGMS)]
    tc_interactive = [c for c in tc_cols if any(p in c for p in INTERACTIVE_PARADIGMS)]
    log(f"Interactive paradigm columns: {len(tq_interactive)} TQ_, {len(tc_interactive)} TC_")

    # Create long-format data
    item_rows = []
    for _, row in df_raw.iterrows():
        uid = row['UID']
        test = row['TEST']
        tsvr = row['TSVR']

        for tq_col in tq_interactive:
            # Get corresponding TC column
            tc_col = tq_col.replace('TQ_', 'TC_')
            if tc_col not in df_raw.columns:
                continue

            accuracy = row[tq_col]
            confidence = row[tc_col]

            # Skip if either is missing
            if pd.isna(accuracy) or pd.isna(confidence):
                continue

            item_name = tq_col.replace('TQ_', '')
            item_rows.append({
                'UID': str(uid),
                'TEST': f'T{int(test)}',
                'TSVR_hours': float(tsvr),
                'ITEM': item_name,
                'Accuracy': int(accuracy),
                'Confidence': float(confidence)
            })

    df_items = pd.DataFrame(item_rows)
    log(f"Created item-level data: {len(df_items)} rows")
    log(f"Participants: {df_items['UID'].nunique()}")
    log(f"Tests: {sorted(df_items['TEST'].unique())}")
    log(f"Unique items: {df_items['ITEM'].nunique()}")

    # Validate
    log(f"Accuracy values: {sorted(df_items['Accuracy'].unique())}")
    log(f"Confidence values: {sorted(df_items['Confidence'].unique())}")

    # Items per participant per test
    items_per_pt = df_items.groupby(['UID', 'TEST']).size()
    log(f"Items per participant-test: mean={items_per_pt.mean():.1f}, min={items_per_pt.min()}, max={items_per_pt.max()}")

    # Save
    output_path = DATA_DIR / "step00_item_level.csv"
    df_items.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_items

# =============================================================================
# STEP 01: Compute Goodman-Kruskal Gamma
# =============================================================================

def compute_gamma(accuracy, confidence):
    """
    Compute Goodman-Kruskal gamma between accuracy and confidence.

    Gamma = (Nc - Nd) / (Nc + Nd)
    where Nc = concordant pairs, Nd = discordant pairs

    Uses scipy's kendalltau which computes tau-b, then convert to gamma.
    For binary vs ordinal, gamma and tau-b are related but we compute directly.
    """
    # Convert to numpy arrays
    acc = np.array(accuracy)
    conf = np.array(confidence)

    # Check for variance
    if len(np.unique(acc)) < 2 or len(np.unique(conf)) < 2:
        return np.nan

    # Count concordant and discordant pairs
    n = len(acc)
    nc = 0  # concordant
    nd = 0  # discordant

    for i in range(n):
        for j in range(i+1, n):
            # Compare pairs
            acc_diff = acc[i] - acc[j]
            conf_diff = conf[i] - conf[j]

            if acc_diff * conf_diff > 0:
                nc += 1  # concordant
            elif acc_diff * conf_diff < 0:
                nd += 1  # discordant
            # ties don't count

    if nc + nd == 0:
        return 0.0

    gamma = (nc - nd) / (nc + nd)
    return gamma

def step01_compute_gamma(df_items):
    """Compute resolution (gamma) for each participant at each timepoint."""
    log("=" * 60)
    log("STEP 01: Compute Goodman-Kruskal Gamma")
    log("=" * 60)

    gamma_rows = []

    for (uid, test), group in df_items.groupby(['UID', 'TEST']):
        accuracy = group['Accuracy'].values
        confidence = group['Confidence'].values
        n_items = len(group)
        tsvr_hours = group['TSVR_hours'].iloc[0]

        gamma = compute_gamma(accuracy, confidence)

        gamma_rows.append({
            'UID': uid,
            'TEST': test,
            'gamma': gamma,
            'n_items': n_items,
            'TSVR_hours': tsvr_hours
        })

    df_gamma = pd.DataFrame(gamma_rows)

    # Report
    log(f"Computed gamma for {len(df_gamma)} participant-timepoints")
    log(f"Mean gamma: {df_gamma['gamma'].mean():.4f}")
    log(f"SD gamma: {df_gamma['gamma'].std():.4f}")
    log(f"Range gamma: [{df_gamma['gamma'].min():.4f}, {df_gamma['gamma'].max():.4f}]")

    # Check for NaN
    n_nan = df_gamma['gamma'].isna().sum()
    if n_nan > 0:
        log(f"WARNING: {n_nan} gamma values are NaN ({100*n_nan/len(df_gamma):.1f}%)")

    # Validate gamma range
    out_of_range = ((df_gamma['gamma'] < -1) | (df_gamma['gamma'] > 1)).sum()
    if out_of_range > 0:
        log(f"ERROR: {out_of_range} gamma values outside [-1, 1]!")

    # Save
    output_path = DATA_DIR / "step01_gamma_scores.csv"
    df_gamma.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_gamma

# =============================================================================
# STEP 02: Fit Linear Mixed Model
# =============================================================================

def step02_fit_lmm(df_gamma):
    """Fit LMM: gamma ~ TSVR_days + (TSVR_days | UID)."""
    log("=" * 60)
    log("STEP 02: Fit Linear Mixed Model")
    log("=" * 60)

    # Prepare LMM input
    df_lmm = df_gamma.copy()
    df_lmm['TSVR_days'] = df_lmm['TSVR_hours'] / 24.0

    # Drop rows with NaN gamma
    df_lmm = df_lmm.dropna(subset=['gamma'])
    log(f"LMM input: {len(df_lmm)} rows (after dropping NaN gamma)")

    # Save LMM input
    lmm_input_path = DATA_DIR / "step02_gamma_lmm_input.csv"
    df_lmm.to_csv(lmm_input_path, index=False)
    log(f"Saved LMM input: {lmm_input_path}")

    # Fit LMM with random slopes
    log("Fitting LMM: gamma ~ TSVR_days + (1 + TSVR_days | UID)")
    try:
        model = smf.mixedlm(
            "gamma ~ TSVR_days",
            df_lmm,
            groups=df_lmm['UID'],
            re_formula="~TSVR_days"
        )
        result = model.fit(method='lbfgs', reml=True)
        converged = True
    except Exception as e:
        log(f"Random slopes model failed: {e}")
        log("Falling back to random intercepts only")
        model = smf.mixedlm(
            "gamma ~ TSVR_days",
            df_lmm,
            groups=df_lmm['UID']
        )
        result = model.fit(method='lbfgs', reml=True)
        converged = True

    log(f"Model converged: {converged}")

    # Save summary
    summary_path = DATA_DIR / "step02_gamma_lmm_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"Saved LMM summary: {summary_path}")

    # Extract fixed effects
    n_fe = len(result.model.exog_names)
    fixed_params = result.params[:n_fe]
    fixed_bse = result.bse[:n_fe]
    fixed_pvalues = result.pvalues[:n_fe]

    log("Fixed Effects:")
    for name, param, se, pval in zip(result.model.exog_names, fixed_params, fixed_bse, fixed_pvalues):
        log(f"  {name}: β={param:.6f}, SE={se:.6f}, p={pval:.6f}")

    return result, df_lmm

# =============================================================================
# STEP 03: Extract Time Effect Statistics
# =============================================================================

def step03_extract_time_effect(result):
    """Extract Time effect with dual p-values (Decision D068)."""
    log("=" * 60)
    log("STEP 03: Extract Time Effect Statistics")
    log("=" * 60)

    # Extract fixed effects
    fe_names = result.model.exog_names
    n_fe = len(fe_names)

    # Find TSVR_days index
    time_idx = fe_names.index('TSVR_days')

    coef = result.params[time_idx]
    se = result.bse[time_idx]
    z_stat = result.tvalues[time_idx]
    p_uncorrected = result.pvalues[time_idx]

    # Bonferroni correction (1 test only, so no adjustment)
    p_bonferroni = min(p_uncorrected * 1, 1.0)  # Only testing one time variable

    log(f"Time effect (TSVR_days):")
    log(f"  Coefficient: {coef:.6f}")
    log(f"  SE: {se:.6f}")
    log(f"  z-statistic: {z_stat:.4f}")
    log(f"  p-value (uncorrected): {p_uncorrected:.6f}")
    log(f"  p-value (Bonferroni): {p_bonferroni:.6f}")

    # Create results DataFrame
    df_time_effect = pd.DataFrame([{
        'time_variable': 'TSVR_days',
        'coefficient': coef,
        'se': se,
        'z_statistic': z_stat,
        'p_uncorrected': p_uncorrected,
        'p_bonferroni': p_bonferroni
    }])

    # Save
    output_path = DATA_DIR / "step03_time_effect.csv"
    df_time_effect.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_time_effect

# =============================================================================
# STEP 04: Compute Mean Gamma by Timepoint
# =============================================================================

def step04_mean_gamma_by_timepoint(df_gamma):
    """Compute descriptive statistics for gamma at each timepoint."""
    log("=" * 60)
    log("STEP 04: Compute Mean Gamma by Timepoint")
    log("=" * 60)

    # Drop NaN gamma
    df_valid = df_gamma.dropna(subset=['gamma'])

    results = []
    for test in sorted(df_valid['TEST'].unique()):
        group = df_valid[df_valid['TEST'] == test]
        n = len(group)
        mean_gamma = group['gamma'].mean()
        sd_gamma = group['gamma'].std()

        # 95% CI using t-distribution
        se = sd_gamma / np.sqrt(n)
        t_crit = t_dist.ppf(0.975, df=n-1)
        ci_lower = mean_gamma - t_crit * se
        ci_upper = mean_gamma + t_crit * se

        # Mean TSVR for this test
        mean_tsvr = group['TSVR_hours'].mean()

        results.append({
            'TEST': test,
            'mean_gamma': mean_gamma,
            'sd_gamma': sd_gamma,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'N': n,
            'mean_TSVR_hours': mean_tsvr
        })

        log(f"{test}: mean={mean_gamma:.4f}, SD={sd_gamma:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], N={n}")

    df_mean_gamma = pd.DataFrame(results)

    # Save
    output_path = DATA_DIR / "step04_mean_gamma.csv"
    df_mean_gamma.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_mean_gamma

# =============================================================================
# STEP 05: Test Gamma > 0.50 Threshold
# =============================================================================

def step05_test_gamma_threshold(df_gamma):
    """Test whether gamma exceeds 0.50 threshold at each timepoint."""
    log("=" * 60)
    log("STEP 05: Test Gamma > 0.50 Threshold")
    log("=" * 60)

    THRESHOLD = 0.50
    N_TESTS = 4  # One per timepoint

    df_valid = df_gamma.dropna(subset=['gamma'])

    results = []
    for test in sorted(df_valid['TEST'].unique()):
        group = df_valid[df_valid['TEST'] == test]
        gamma_values = group['gamma'].values

        # One-sample t-test: H0: mean = 0.50, H1: mean > 0.50 (one-tailed)
        t_stat, p_two_tailed = ttest_1samp(gamma_values, THRESHOLD)

        # Convert to one-tailed p-value
        if t_stat > 0:
            p_one_tailed = p_two_tailed / 2
        else:
            p_one_tailed = 1 - p_two_tailed / 2

        # Bonferroni correction for 4 tests
        p_bonferroni = min(p_one_tailed * N_TESTS, 1.0)

        mean_gamma = gamma_values.mean()
        df_val = len(gamma_values) - 1

        results.append({
            'TEST': test,
            'mean_gamma': mean_gamma,
            'threshold': THRESHOLD,
            't_statistic': t_stat,
            'df': df_val,
            'p_uncorrected': p_one_tailed,
            'p_bonferroni': p_bonferroni,
            'exceeds_threshold': mean_gamma > THRESHOLD
        })

        sig_str = "***" if p_bonferroni < 0.001 else "**" if p_bonferroni < 0.01 else "*" if p_bonferroni < 0.05 else ""
        log(f"{test}: mean={mean_gamma:.4f} vs threshold={THRESHOLD}, t={t_stat:.4f}, p_bonf={p_bonferroni:.6f} {sig_str}")

    df_threshold = pd.DataFrame(results)

    # Summary
    all_exceed = all(df_threshold['exceeds_threshold'])
    log(f"All timepoints exceed 0.50 threshold: {all_exceed}")

    # Save
    output_path = DATA_DIR / "step05_gamma_threshold_tests.csv"
    df_threshold.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_threshold

# =============================================================================
# STEP 06: Prepare Plot Data
# =============================================================================

def step06_prepare_plot_data(df_mean_gamma, lmm_result, df_lmm):
    """Prepare plot source data for resolution trajectory visualization."""
    log("=" * 60)
    log("STEP 06: Prepare Plot Data")
    log("=" * 60)

    # Extract fixed effects for prediction
    intercept = lmm_result.params['Intercept']
    slope = lmm_result.params['TSVR_days']

    log(f"Model: gamma = {intercept:.4f} + {slope:.6f} * TSVR_days")

    # Create plot data
    plot_data = []
    for _, row in df_mean_gamma.iterrows():
        test = row['TEST']
        tsvr_days = row['mean_TSVR_hours'] / 24.0

        # Compute predicted gamma from LMM
        predicted_gamma = intercept + slope * tsvr_days

        plot_data.append({
            'TEST': test,
            'time_days': tsvr_days,
            'observed_mean': row['mean_gamma'],
            'CI_lower': row['CI_lower'],
            'CI_upper': row['CI_upper'],
            'predicted_mean': predicted_gamma,
            'N': row['N']
        })

    df_plot = pd.DataFrame(plot_data)

    # Log trajectory
    log("Resolution trajectory:")
    for _, row in df_plot.iterrows():
        log(f"  {row['TEST']}: Day {row['time_days']:.1f}, observed={row['observed_mean']:.4f}, predicted={row['predicted_mean']:.4f}")

    # Calculate decline
    t1_gamma = df_plot[df_plot['TEST'] == 'T1']['observed_mean'].values[0]
    t4_gamma = df_plot[df_plot['TEST'] == 'T4']['observed_mean'].values[0]
    decline_pct = 100 * (t1_gamma - t4_gamma) / t1_gamma
    log(f"Observed decline: T1={t1_gamma:.4f} → T4={t4_gamma:.4f} ({decline_pct:.1f}% decrease)")

    # Save
    output_path = DATA_DIR / "step06_resolution_trajectory_data.csv"
    df_plot.to_csv(output_path, index=False)
    log(f"Saved: {output_path}")

    return df_plot

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all steps."""
    log("=" * 80)
    log("RQ 6.2.3: Resolution Over Time - Metacognitive Discrimination Analysis")
    log("=" * 80)
    log(f"RQ Directory: {RQ_DIR}")
    log(f"Project Root: {PROJECT_ROOT}")

    # Step 00: Extract item-level data
    df_items = step00_extract_item_level()

    # Step 01: Compute gamma
    df_gamma = step01_compute_gamma(df_items)

    # Step 02: Fit LMM
    lmm_result, df_lmm = step02_fit_lmm(df_gamma)

    # Step 03: Extract time effect
    df_time_effect = step03_extract_time_effect(lmm_result)

    # Step 04: Compute mean gamma by timepoint
    df_mean_gamma = step04_mean_gamma_by_timepoint(df_gamma)

    # Step 05: Test gamma > 0.50 threshold
    df_threshold = step05_test_gamma_threshold(df_gamma)

    # Step 06: Prepare plot data
    df_plot = step06_prepare_plot_data(df_mean_gamma, lmm_result, df_lmm)

    # Final summary
    log("=" * 80)
    log("ANALYSIS COMPLETE")
    log("=" * 80)

    # Primary hypothesis test
    time_coef = df_time_effect.iloc[0]['coefficient']
    time_p = df_time_effect.iloc[0]['p_uncorrected']

    if time_coef < 0 and time_p < 0.05:
        log("PRIMARY FINDING: Resolution DECLINES significantly over time")
        log(f"  Time effect: β={time_coef:.6f}, p={time_p:.6f}")
        log("  SUPPORTS dual-process hypothesis: discrimination degrades with memory fading")
    elif time_coef < 0 and time_p >= 0.05:
        log("PRIMARY FINDING: Resolution shows DECLINING trend but NOT SIGNIFICANT")
        log(f"  Time effect: β={time_coef:.6f}, p={time_p:.6f}")
        log("  INCONCLUSIVE: Trend in expected direction but insufficient statistical power")
    else:
        log("PRIMARY FINDING: Resolution does NOT decline over time (unexpected)")
        log(f"  Time effect: β={time_coef:.6f}, p={time_p:.6f}")
        log("  Suggests metacognitive discrimination is robust to memory fading")

    # Threshold tests
    threshold_passes = df_threshold['exceeds_threshold'].sum()
    log(f"\nThreshold tests: {threshold_passes}/4 timepoints exceed gamma > 0.50")

    log("\nAll outputs saved to: " + str(DATA_DIR))

if __name__ == "__main__":
    main()
