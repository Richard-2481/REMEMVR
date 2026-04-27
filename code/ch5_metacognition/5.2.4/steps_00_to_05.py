"""
RQ 6.2.4: Calibration by Accuracy Level (Dunning-Kruger Test)
=============================================================

Tests whether high vs low baseline performers differ in calibration quality.
Uses derived data from Ch5 5.1.1 (accuracy), RQ 6.1.1 (confidence),
RQ 6.2.1 (calibration), RQ 6.2.3 (gamma/resolution).

Steps:
0. Merge metrics from 4 source RQs (N=100 baseline)
1. Create accuracy tertiles (Low/Med/High)
2. Tertile comparison (ANOVA/Kruskal-Wallis)
3. Dunning-Kruger test (one-sample t-tests per tertile)
4. Correlations (baseline accuracy vs calibration metrics)
5. Prepare plot data

Author: Claude Code (automated execution)
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.4
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"

# Source file paths (actual column names from inspection)
SOURCE_FILES = {
    'accuracy': PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv",
    'confidence': PROJECT_ROOT / "results/ch6/6.1.1/data/step03_theta_confidence.csv",
    'calibration': PROJECT_ROOT / "results/ch6/6.2.1/data/step02_calibration_scores.csv",
    'gamma': PROJECT_ROOT / "results/ch6/6.2.3/data/step01_gamma_scores.csv"
}

# Tertile colors (per specification)
TERTILE_COLORS = {
    'Low': '#D62728',
    'Med': '#FF7F0E',
    'High': '#2CA02C'
}


def log(msg: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + "\n")
        f.flush()
    print(log_msg, flush=True)


def validate_range(value: float, min_val: float, max_val: float, name: str) -> bool:
    """Validate value is within expected range."""
    if pd.isna(value):
        log(f"WARNING: {name} is NaN")
        return False
    if value < min_val or value > max_val:
        log(f"WARNING: {name}={value:.4f} outside [{min_val}, {max_val}]")
        return False
    return True


# ============================================================================
# STEP 0: Merge Calibration Metrics from Prior RQs
# ============================================================================

def step00_merge_metrics():
    """Load and merge baseline metrics from 4 source RQs."""
    log("=" * 60)
    log("STEP 00: Merge Calibration Metrics from Prior RQs")
    log("=" * 60)

    # Check all source files exist
    for name, path in SOURCE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Source file missing: {path}")
        log(f"Source file found: {name} -> {path}")

    # --- Load Ch5 5.1.1 accuracy (baseline = test 1) ---
    df_acc = pd.read_csv(SOURCE_FILES['accuracy'])
    log(f"Loaded accuracy: {len(df_acc)} rows, columns: {list(df_acc.columns)}")

    # Filter to test=1 (baseline)
    df_acc_baseline = df_acc[df_acc['test'] == 1].copy()
    df_acc_baseline['UID'] = df_acc_baseline['UID'].astype(str)
    df_acc_baseline = df_acc_baseline[['UID', 'Theta_All']].rename(
        columns={'Theta_All': 'baseline_accuracy'}
    )
    log(f"Accuracy baseline (test=1): {len(df_acc_baseline)} rows")

    # --- Load RQ 6.1.1 confidence (baseline = T1) ---
    df_conf = pd.read_csv(SOURCE_FILES['confidence'])
    log(f"Loaded confidence: {len(df_conf)} rows, columns: {list(df_conf.columns)}")

    # Extract UID from composite_ID (format: UID_T1)
    df_conf['UID'] = df_conf['composite_ID'].str.split('_').str[0]
    df_conf['test'] = df_conf['composite_ID'].str.split('_').str[1]
    df_conf_baseline = df_conf[df_conf['test'] == 'T1'].copy()
    df_conf_baseline = df_conf_baseline[['UID', 'theta_All']].rename(
        columns={'theta_All': 'baseline_confidence'}
    )
    log(f"Confidence baseline (T1): {len(df_conf_baseline)} rows")

    # --- Load RQ 6.2.1 calibration (compute mean across all tests) ---
    df_cal = pd.read_csv(SOURCE_FILES['calibration'])
    log(f"Loaded calibration: {len(df_cal)} rows, columns: {list(df_cal.columns)}")

    # Compute mean calibration per participant
    df_cal['UID'] = df_cal['UID'].astype(str)
    df_cal_mean = df_cal.groupby('UID').agg(
        mean_calibration=('calibration', 'mean')
    ).reset_index()
    log(f"Calibration mean per UID: {len(df_cal_mean)} rows")

    # --- Load RQ 6.2.3 gamma (compute mean across all tests) ---
    df_gamma = pd.read_csv(SOURCE_FILES['gamma'])
    log(f"Loaded gamma: {len(df_gamma)} rows, columns: {list(df_gamma.columns)}")

    # Compute mean gamma per participant
    df_gamma['UID'] = df_gamma['UID'].astype(str)
    df_gamma_mean = df_gamma.groupby('UID').agg(
        mean_gamma=('gamma', 'mean')
    ).reset_index()
    log(f"Gamma mean per UID: {len(df_gamma_mean)} rows")

    # --- Merge all sources ---
    merged = df_acc_baseline.merge(df_conf_baseline, on='UID', how='inner')
    log(f"After merge accuracy + confidence: {len(merged)} rows")

    merged = merged.merge(df_cal_mean, on='UID', how='inner')
    log(f"After merge + calibration: {len(merged)} rows")

    merged = merged.merge(df_gamma_mean, on='UID', how='inner')
    log(f"After merge + gamma: {len(merged)} rows")

    # Validate
    if len(merged) != 100:
        raise ValueError(f"Expected 100 rows after merge, got {len(merged)}")

    if merged.isna().any().any():
        nan_cols = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"NaN values in columns: {nan_cols}")

    # Validate ranges
    log("\nValue ranges:")
    log(f"  baseline_accuracy: [{merged['baseline_accuracy'].min():.3f}, {merged['baseline_accuracy'].max():.3f}]")
    log(f"  baseline_confidence: [{merged['baseline_confidence'].min():.3f}, {merged['baseline_confidence'].max():.3f}]")
    log(f"  mean_calibration: [{merged['mean_calibration'].min():.3f}, {merged['mean_calibration'].max():.3f}]")
    log(f"  mean_gamma: [{merged['mean_gamma'].min():.3f}, {merged['mean_gamma'].max():.3f}]")

    # Save
    output_path = RQ_DIR / "data" / "step00_merged_metrics.csv"
    merged.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"Merged 100 participants successfully")
    log(f"Source: Ch5 5.1.1 ({len(df_acc_baseline)} rows), RQ 6.1.1 ({len(df_conf_baseline)} rows), "
        f"RQ 6.2.1 ({len(df_cal_mean)} rows), RQ 6.2.3 ({len(df_gamma_mean)} rows)")

    return merged


# ============================================================================
# STEP 1: Create Accuracy Tertiles
# ============================================================================

def step01_create_tertiles():
    """Split participants into Low/Med/High accuracy tertiles."""
    log("\n" + "=" * 60)
    log("STEP 01: Create Accuracy Tertiles")
    log("=" * 60)

    # Load merged metrics
    df = pd.read_csv(RQ_DIR / "data" / "step00_merged_metrics.csv")
    log(f"Loaded merged metrics: {len(df)} rows")

    # Create tertiles using qcut
    df['tertile_numeric'], bins = pd.qcut(
        df['baseline_accuracy'],
        q=3,
        labels=[1, 2, 3],
        retbins=True
    )
    df['tertile_numeric'] = df['tertile_numeric'].astype(int)

    # Map to labels
    tertile_map = {1: 'Low', 2: 'Med', 3: 'High'}
    df['tertile_label'] = df['tertile_numeric'].map(tertile_map)

    # Report tertile boundaries
    log(f"\nTertile boundaries:")
    log(f"  Low:  [{bins[0]:.4f}, {bins[1]:.4f}]")
    log(f"  Med:  [{bins[1]:.4f}, {bins[2]:.4f}]")
    log(f"  High: [{bins[2]:.4f}, {bins[3]:.4f}]")

    # Report N per tertile
    tertile_counts = df['tertile_label'].value_counts()
    log(f"\nTertile assignment complete: Low ({tertile_counts.get('Low', 0)}), "
        f"Med ({tertile_counts.get('Med', 0)}), High ({tertile_counts.get('High', 0)})")

    # Validate: mean accuracy should be ordered Low < Med < High
    mean_by_tertile = df.groupby('tertile_label')['baseline_accuracy'].mean()
    log(f"\nMean accuracy by tertile (sanity check):")
    log(f"  Low:  {mean_by_tertile.get('Low', 'N/A'):.4f}")
    log(f"  Med:  {mean_by_tertile.get('Med', 'N/A'):.4f}")
    log(f"  High: {mean_by_tertile.get('High', 'N/A'):.4f}")

    if not (mean_by_tertile['Low'] < mean_by_tertile['Med'] < mean_by_tertile['High']):
        log("WARNING: Mean accuracy not strictly ordered Low < Med < High")

    # Save tertiles CSV
    output_cols = ['UID', 'baseline_accuracy', 'tertile_label', 'tertile_numeric']
    output_path = RQ_DIR / "data" / "step01_accuracy_tertiles.csv"
    df[output_cols].to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    # Save summary text
    summary_path = RQ_DIR / "data" / "step01_tertile_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("RQ 6.2.4 - Accuracy Tertile Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Tertile Boundaries:\n")
        f.write(f"  Low:  [{bins[0]:.4f}, {bins[1]:.4f}]\n")
        f.write(f"  Med:  [{bins[1]:.4f}, {bins[2]:.4f}]\n")
        f.write(f"  High: [{bins[2]:.4f}, {bins[3]:.4f}]\n\n")
        f.write(f"N per Tertile:\n")
        for label in ['Low', 'Med', 'High']:
            n = tertile_counts.get(label, 0)
            mean_acc = mean_by_tertile.get(label, 0)
            f.write(f"  {label}: N={n}, Mean accuracy={mean_acc:.4f}\n")
    log(f"Saved: {summary_path}")

    return df[output_cols]


# ============================================================================
# STEP 2: Compare Calibration Metrics Across Tertiles
# ============================================================================

def step02_tertile_comparison():
    """Test calibration metrics across tertiles using ANOVA/Kruskal-Wallis."""
    log("\n" + "=" * 60)
    log("STEP 02: Compare Calibration Metrics Across Tertiles")
    log("=" * 60)

    # Load data
    df_metrics = pd.read_csv(RQ_DIR / "data" / "step00_merged_metrics.csv")
    df_tertiles = pd.read_csv(RQ_DIR / "data" / "step01_accuracy_tertiles.csv")

    # Merge
    df = df_metrics.merge(df_tertiles[['UID', 'tertile_label', 'tertile_numeric']], on='UID')
    log(f"Merged data: {len(df)} rows")

    # Compute absolute calibration
    df['abs_calibration'] = df['mean_calibration'].abs()

    # Metrics to test
    metrics = ['abs_calibration', 'mean_gamma']

    # Results storage
    comparison_results = []
    normality_results = []
    variance_results = []

    for metric in metrics:
        log(f"\n--- Testing: {metric} ---")

        # Get values by tertile
        groups = {
            label: df[df['tertile_label'] == label][metric].values
            for label in ['Low', 'Med', 'High']
        }

        # Test normality per tertile (Shapiro-Wilk)
        normality_ok = True
        for label, values in groups.items():
            stat, p = stats.shapiro(values)
            normality_results.append({
                'metric': metric,
                'tertile': label,
                'shapiro_statistic': stat,
                'shapiro_p': p
            })
            log(f"  Shapiro-Wilk ({label}): W={stat:.4f}, p={p:.4f}")
            if p < 0.05:
                normality_ok = False

        # Test homogeneity of variance (Levene)
        levene_stat, levene_p = stats.levene(groups['Low'], groups['Med'], groups['High'])
        variance_results.append({
            'metric': metric,
            'levene_statistic': levene_stat,
            'levene_p': levene_p
        })
        log(f"  Levene test: F={levene_stat:.4f}, p={levene_p:.4f}")

        variance_ok = levene_p >= 0.05

        # Choose test based on assumptions
        if normality_ok and variance_ok:
            test_used = 'ANOVA'
            stat, p_value = stats.f_oneway(groups['Low'], groups['Med'], groups['High'])
            log(f"  Using ANOVA: F={stat:.4f}, p={p_value:.6f}")
        else:
            test_used = 'Kruskal-Wallis'
            stat, p_value = stats.kruskal(groups['Low'], groups['Med'], groups['High'])
            log(f"  Using Kruskal-Wallis: H={stat:.4f}, p={p_value:.6f}")
            if not normality_ok:
                log(f"    (Normality violated)")
            if not variance_ok:
                log(f"    (Variance homogeneity violated)")

        # Compute descriptives per tertile
        for label in ['Low', 'Med', 'High']:
            values = groups[label]
            comparison_results.append({
                'metric': metric,
                'test_used': test_used,
                'statistic': stat,
                'p_value': p_value,
                'tertile': label,
                'mean': np.mean(values),
                'sd': np.std(values, ddof=1),
                'median': np.median(values),
                'iqr': stats.iqr(values)
            })

    # Save results
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(RQ_DIR / "data" / "step02_tertile_comparison.csv", index=False)
    log(f"\nSaved: data/step02_tertile_comparison.csv")

    df_normality = pd.DataFrame(normality_results)
    df_normality.to_csv(RQ_DIR / "data" / "step02_normality_tests.csv", index=False)
    log(f"Saved: data/step02_normality_tests.csv")

    df_variance = pd.DataFrame(variance_results)
    df_variance.to_csv(RQ_DIR / "data" / "step02_variance_tests.csv", index=False)
    log(f"Saved: data/step02_variance_tests.csv")

    # Summary
    log("\nTertile comparison complete:")
    for metric in metrics:
        row = df_comparison[df_comparison['metric'] == metric].iloc[0]
        log(f"  {metric} ({row['test_used']}, p={row['p_value']:.4f})")

    return df_comparison


# ============================================================================
# STEP 3: Test Dunning-Kruger Hypothesis (One-Sample t-Tests)
# ============================================================================

def step03_dunning_kruger_test():
    """Test if low performers show overconfidence (calibration > 0)."""
    log("\n" + "=" * 60)
    log("STEP 03: Test Dunning-Kruger Hypothesis (One-Sample t-Tests)")
    log("=" * 60)

    # Load data
    df_metrics = pd.read_csv(RQ_DIR / "data" / "step00_merged_metrics.csv")
    df_tertiles = pd.read_csv(RQ_DIR / "data" / "step01_accuracy_tertiles.csv")

    # Merge
    df = df_metrics.merge(df_tertiles[['UID', 'tertile_label']], on='UID')

    results = []

    for tertile in ['Low', 'Med', 'High']:
        values = df[df['tertile_label'] == tertile]['mean_calibration'].values
        n = len(values)
        mean_cal = np.mean(values)
        sd_cal = np.std(values, ddof=1)

        # One-sample t-test against mu=0
        t_stat, p_two = stats.ttest_1samp(values, 0)
        df_val = n - 1

        # 95% CI
        se = sd_cal / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df_val)
        ci_lower = mean_cal - t_crit * se
        ci_upper = mean_cal + t_crit * se

        # Bonferroni correction (3 comparisons)
        p_bonf = min(p_two * 3, 1.0)

        results.append({
            'tertile': tertile,
            'N': n,
            'mean_calibration': mean_cal,
            'sd_calibration': sd_cal,
            't_statistic': t_stat,
            'df': df_val,
            'p_uncorrected': p_two,
            'p_bonferroni': p_bonf,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })

        # Interpretation
        sig_label = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else ""
        direction = "OVERCONFIDENT" if mean_cal > 0 else "UNDERCONFIDENT" if mean_cal < 0 else "CALIBRATED"
        log(f"\n{tertile} tertile (N={n}):")
        log(f"  Mean calibration: {mean_cal:.4f} ({direction})")
        log(f"  t({df_val}) = {t_stat:.3f}, p_uncorrected = {p_two:.4f}, p_bonferroni = {p_bonf:.4f} {sig_label}")
        log(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save
    df_dk = pd.DataFrame(results)
    df_dk.to_csv(RQ_DIR / "data" / "step03_dunning_kruger_test.csv", index=False)
    log(f"\nSaved: data/step03_dunning_kruger_test.csv")

    log("\nBonferroni correction applied (alpha = 0.05 / 3 = 0.0167)")

    # Key finding summary
    low_result = df_dk[df_dk['tertile'] == 'Low'].iloc[0]
    high_result = df_dk[df_dk['tertile'] == 'High'].iloc[0]

    log("\n--- DUNNING-KRUGER HYPOTHESIS TEST ---")
    if low_result['mean_calibration'] > 0 and low_result['p_bonferroni'] < 0.05:
        log("SUPPORTED: Low performers show SIGNIFICANT OVERCONFIDENCE")
    elif low_result['mean_calibration'] > 0:
        log(f"PARTIAL: Low performers show overconfidence (mean={low_result['mean_calibration']:.3f}) but NOT SIGNIFICANT (p_bonf={low_result['p_bonferroni']:.3f})")
    else:
        log("NOT SUPPORTED: Low performers do NOT show overconfidence")

    return df_dk


# ============================================================================
# STEP 4: Compute Correlations
# ============================================================================

def step04_correlation():
    """Compute correlations: baseline_accuracy vs abs_calibration, vs mean_gamma."""
    log("\n" + "=" * 60)
    log("STEP 04: Compute Correlations (Baseline Accuracy vs Calibration)")
    log("=" * 60)

    # Load data
    df = pd.read_csv(RQ_DIR / "data" / "step00_merged_metrics.csv")
    df['abs_calibration'] = df['mean_calibration'].abs()

    # Variables
    baseline = df['baseline_accuracy'].values
    abs_cal = df['abs_calibration'].values
    gamma = df['mean_gamma'].values

    # Normality tests
    normality_results = []
    for name, values in [('baseline_accuracy', baseline), ('abs_calibration', abs_cal), ('mean_gamma', gamma)]:
        stat, p = stats.shapiro(values)
        normality_results.append({
            'variable': name,
            'shapiro_statistic': stat,
            'shapiro_p': p
        })
        log(f"Shapiro-Wilk ({name}): W={stat:.4f}, p={p:.4f}")

    df_norm = pd.DataFrame(normality_results)
    df_norm.to_csv(RQ_DIR / "data" / "step04_normality_tests.csv", index=False)
    log(f"\nSaved: data/step04_normality_tests.csv")

    # Check if all normal
    all_normal = all(r['shapiro_p'] >= 0.05 for r in normality_results)

    correlation_results = []
    comparisons = [
        ('baseline_accuracy vs abs_calibration', baseline, abs_cal),
        ('baseline_accuracy vs mean_gamma', baseline, gamma)
    ]

    for comp_name, x, y in comparisons:
        log(f"\n--- {comp_name} ---")

        # Determine method based on normality of both variables
        var1 = comp_name.split(' vs ')[0]
        var2 = comp_name.split(' vs ')[1]
        norm1 = df_norm[df_norm['variable'] == var1]['shapiro_p'].values[0]
        norm2 = df_norm[df_norm['variable'] == var2]['shapiro_p'].values[0]

        if norm1 >= 0.05 and norm2 >= 0.05:
            method = 'Pearson'
            r, p = stats.pearsonr(x, y)
            log(f"  Using Pearson (both variables normal)")

            # Fisher z-transform for CI
            n = len(x)
            z = np.arctanh(r)
            se_z = 1.0 / np.sqrt(n - 3)
            z_lower = z - 1.96 * se_z
            z_upper = z + 1.96 * se_z
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
        else:
            method = 'Spearman'
            r, p = stats.spearmanr(x, y)
            log(f"  Using Spearman (normality violated)")

            # Bootstrap CI for Spearman
            n = len(x)
            np.random.seed(42)
            boot_r = []
            for _ in range(1000):
                idx = np.random.choice(n, n, replace=True)
                boot_r.append(stats.spearmanr(x[idx], y[idx])[0])
            ci_lower, ci_upper = np.percentile(boot_r, [2.5, 97.5])

        # Bonferroni correction (2 comparisons)
        p_bonf = min(p * 2, 1.0)

        correlation_results.append({
            'comparison': comp_name,
            'method': method,
            'r_or_rho': r,
            'p_uncorrected': p,
            'p_bonferroni': p_bonf,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'N': len(x)
        })

        sig_label = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else ""
        log(f"  {method} r/rho = {r:.4f}, p_uncorrected = {p:.4f}, p_bonferroni = {p_bonf:.4f} {sig_label}")
        log(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    df_corr = pd.DataFrame(correlation_results)
    df_corr.to_csv(RQ_DIR / "data" / "step04_correlation.csv", index=False)
    log(f"\nSaved: data/step04_correlation.csv")

    log("\nBonferroni correction applied (alpha = 0.05 / 2 = 0.025)")

    return df_corr


# ============================================================================
# STEP 5: Prepare Plot Data
# ============================================================================

def step05_prepare_plot_data():
    """Create plot source CSV for visualization."""
    log("\n" + "=" * 60)
    log("STEP 05: Prepare Calibration by Accuracy Plot Data")
    log("=" * 60)

    # Load data
    df_metrics = pd.read_csv(RQ_DIR / "data" / "step00_merged_metrics.csv")
    df_tertiles = pd.read_csv(RQ_DIR / "data" / "step01_accuracy_tertiles.csv")

    # Merge
    df = df_metrics.merge(df_tertiles[['UID', 'tertile_label']], on='UID')

    # Compute absolute calibration
    df['abs_calibration'] = df['mean_calibration'].abs()

    # Add color codes
    df['tertile_color'] = df['tertile_label'].map(TERTILE_COLORS)

    # Select output columns
    output_cols = ['UID', 'baseline_accuracy', 'abs_calibration', 'mean_gamma', 'tertile_label', 'tertile_color']
    df_plot = df[output_cols]

    # Validate
    if len(df_plot) != 100:
        raise ValueError(f"Expected 100 rows, got {len(df_plot)}")

    if df_plot.isna().any().any():
        raise ValueError("NaN values in plot data")

    tertile_counts = df_plot['tertile_label'].value_counts()
    log(f"Tertiles represented: Low ({tertile_counts.get('Low', 0)}), "
        f"Med ({tertile_counts.get('Med', 0)}), High ({tertile_counts.get('High', 0)})")

    # Save
    output_path = RQ_DIR / "data" / "step05_calibration_by_accuracy_plot_data.csv"
    df_plot.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"Plot data preparation complete: 100 rows created")
    log(f"All columns validated: {', '.join(output_cols)}")

    return df_plot


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    log("=" * 70)
    log("RQ 6.2.4: Calibration by Accuracy Level (Dunning-Kruger Test)")
    log("=" * 70)
    log(f"Start time: {datetime.now()}")
    log(f"RQ Directory: {RQ_DIR}")

    try:
        # Execute all steps
        step00_merge_metrics()
        step01_create_tertiles()
        step02_tertile_comparison()
        step03_dunning_kruger_test()
        step04_correlation()
        step05_prepare_plot_data()

        log("\n" + "=" * 70)
        log("ALL STEPS COMPLETED SUCCESSFULLY")
        log("=" * 70)
        log(f"End time: {datetime.now()}")

    except Exception as e:
        log(f"\nERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
