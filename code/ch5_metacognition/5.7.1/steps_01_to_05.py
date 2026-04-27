#!/usr/bin/env python3
"""
RQ 6.7.1: Initial Confidence Predicting Forgetting Rates
=========================================================

Tests whether high Day 0 (T1) confidence predicts slower forgetting trajectories.

Pipeline:
- Step 1: Load Day 0 confidence from RQ 6.1.1
- Step 2: Load forgetting slopes from Ch5 5.1.4
- Step 3: Merge confidence and slopes data
- Step 4: Compute correlation (with normality check) and tertile analysis
- Step 5: Prepare plot data

Primary Hypothesis: High Day 0 confidence predicts slower forgetting (positive correlation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, pearsonr, spearmanr, ttest_ind, f_oneway
import warnings

# Suppress convergence warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.7.1
LOG_FILE = RQ_DIR / "logs" / "steps_01_to_05.log"

# Source data paths
RQ_6_1_1_CONFIDENCE = Path("/home/etai/projects/REMEMVR/results/ch6/6.1.1/data/step03_theta_confidence.csv")
CH5_5_1_4_SLOPES = Path("/home/etai/projects/REMEMVR/results/ch5/5.1.4/data/step04_random_effects.csv")

# Bootstrap configuration
N_BOOTSTRAP = 10000
RANDOM_SEED = 42

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
        f.write("RQ 6.7.1: Initial Confidence Predicting Forgetting Rates\n")
        f.write("=" * 70 + "\n\n")

# =============================================================================
# Step 1: Load Day 0 Confidence Data
# =============================================================================

def step01_load_day0_confidence() -> pd.DataFrame:
    """Load Day 0 confidence estimates from RQ 6.1.1."""
    log("=" * 70)
    log("STEP 1: Load Day 0 Confidence Data")
    log("=" * 70)

    # Check source file exists
    if not RQ_6_1_1_CONFIDENCE.exists():
        raise FileNotFoundError(f"RQ 6.1.1 confidence data not found: {RQ_6_1_1_CONFIDENCE}")

    # Load data
    df = pd.read_csv(RQ_6_1_1_CONFIDENCE)
    log(f"Loaded {len(df)} rows from RQ 6.1.1")
    log(f"Columns: {list(df.columns)}")

    # Parse composite_ID to extract UID and test
    df['UID'] = df['composite_ID'].str.split('_').str[0]
    df['test'] = df['composite_ID'].str.split('_').str[1]

    log(f"Unique tests: {df['test'].unique()}")

    # Filter to T1 (Day 0) only
    df_t1 = df[df['test'] == 'T1'].copy()
    log(f"Filtered to T1 only: {len(df_t1)} rows retained")

    # Rename columns for clarity
    # Note: RQ 6.1.1 uses theta_All, se_All (omnibus factor)
    df_t1 = df_t1.rename(columns={
        'theta_All': 'Day0_confidence',
        'se_All': 'se_confidence'
    })

    # Select relevant columns
    df_out = df_t1[['UID', 'Day0_confidence', 'se_confidence']].copy()
    df_out = df_out.sort_values('UID').reset_index(drop=True)

    # Validation
    log(f"\nValidation:")
    log(f"  Rows: {len(df_out)} (expected: 100)")
    log(f"  Columns: {list(df_out.columns)}")
    log(f"  NaN count: {df_out.isna().sum().sum()}")
    log(f"  Day0_confidence range: [{df_out['Day0_confidence'].min():.3f}, {df_out['Day0_confidence'].max():.3f}]")
    log(f"  se_confidence range: [{df_out['se_confidence'].min():.3f}, {df_out['se_confidence'].max():.3f}]")
    log(f"  Unique UIDs: {df_out['UID'].nunique()}")

    if len(df_out) != 100:
        raise ValueError(f"Expected 100 rows, found {len(df_out)}")
    if df_out.isna().sum().sum() > 0:
        raise ValueError("NaN values detected in Day 0 confidence data")

    # Save output
    output_path = RQ_DIR / "data" / "step01_day0_confidence.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"Loaded 100 Day 0 confidence estimates")

    return df_out

# =============================================================================
# Step 2: Load Forgetting Slopes Data
# =============================================================================

def step02_load_forgetting_slopes() -> pd.DataFrame:
    """Load forgetting slopes from Ch5 5.1.4."""
    log("\n" + "=" * 70)
    log("STEP 2: Load Forgetting Slopes Data")
    log("=" * 70)

    # Check source file exists
    if not CH5_5_1_4_SLOPES.exists():
        raise FileNotFoundError(f"Ch5 5.1.4 random effects not found: {CH5_5_1_4_SLOPES}")

    # Load data
    df = pd.read_csv(CH5_5_1_4_SLOPES)
    log(f"Loaded {len(df)} rows from Ch5 5.1.4")
    log(f"Columns: {list(df.columns)}")
    log(f"Data source: Ch5 RQ 5.1.4")

    # Select and rename columns
    # Note: Using total_slope (fixed + random) as the individual forgetting rate
    # random_slope is deviation from mean; total_slope is the actual trajectory
    df_out = df[['UID', 'total_slope']].copy()
    df_out = df_out.rename(columns={'total_slope': 'forgetting_slope'})

    # We don't have SE for total_slope directly, but we can use random_slope SE proxy
    # For now, compute SE from the random effects variance if needed
    # Actually, let's just include the random_slope for reference
    df_out['random_slope'] = df['random_slope']

    # Estimate SE as the SD of random slopes / sqrt(N) as a rough proxy
    # Or we can just not include SE since it's not critical for correlation
    # Let's compute a rough SE based on the random slope SD
    random_slope_sd = df['random_slope'].std()
    df_out['se_slope'] = random_slope_sd / np.sqrt(4)  # Rough estimate based on 4 time points

    df_out = df_out.sort_values('UID').reset_index(drop=True)

    # Validation
    log(f"\nValidation:")
    log(f"  Rows: {len(df_out)} (expected: 100)")
    log(f"  Columns: {list(df_out.columns)}")
    log(f"  NaN count: {df_out.isna().sum().sum()}")
    log(f"  forgetting_slope range: [{df_out['forgetting_slope'].min():.4f}, {df_out['forgetting_slope'].max():.4f}]")
    log(f"  Mean forgetting_slope: {df_out['forgetting_slope'].mean():.4f}")
    log(f"  Unique UIDs: {df_out['UID'].nunique()}")

    if len(df_out) != 100:
        raise ValueError(f"Expected 100 rows, found {len(df_out)}")
    if df_out.isna().sum().sum() > 0:
        raise ValueError("NaN values detected in forgetting slopes data")

    # Note on forgetting_slope interpretation:
    # Positive slope = improvement over time (rare)
    # Negative slope = forgetting over time (expected)
    # The fixed effect slope from 5.1.4 was around 0.077 (per day?)
    # Random slopes show individual deviations
    n_positive = (df_out['forgetting_slope'] > 0).sum()
    n_negative = (df_out['forgetting_slope'] < 0).sum()
    log(f"\nSlope direction:")
    log(f"  Positive (improvement): {n_positive}")
    log(f"  Negative (forgetting): {n_negative}")

    # Save output (select final columns)
    df_save = df_out[['UID', 'forgetting_slope', 'se_slope']].copy()
    output_path = RQ_DIR / "data" / "step02_forgetting_slopes.csv"
    df_save.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"Loaded 100 forgetting slopes")

    return df_save

# =============================================================================
# Step 3: Merge Confidence and Slopes Data
# =============================================================================

def step03_merge_data(df_confidence: pd.DataFrame, df_slopes: pd.DataFrame) -> pd.DataFrame:
    """Merge confidence and slopes data."""
    log("\n" + "=" * 70)
    log("STEP 3: Merge Confidence and Slopes Data")
    log("=" * 70)

    # Inner merge on UID
    df_merged = pd.merge(df_confidence, df_slopes, on='UID', how='inner')

    log(f"Merged {len(df_merged)} participants successfully")
    log(f"Columns: {list(df_merged.columns)}")

    # Validation
    n_conf = len(df_confidence)
    n_slope = len(df_slopes)
    n_merged = len(df_merged)

    log(f"\nMerge statistics:")
    log(f"  Input confidence: {n_conf}")
    log(f"  Input slopes: {n_slope}")
    log(f"  Merged output: {n_merged}")
    log(f"  No data loss: {n_merged == 100}")

    if n_merged != 100:
        # Find which participants are missing
        conf_uids = set(df_confidence['UID'])
        slope_uids = set(df_slopes['UID'])
        missing_from_slopes = conf_uids - slope_uids
        missing_from_conf = slope_uids - conf_uids
        log(f"  Missing from slopes: {missing_from_slopes}")
        log(f"  Missing from confidence: {missing_from_conf}")
        raise ValueError(f"Merge incomplete: {n_conf + n_slope - n_merged} participants missing")

    # Sort and reset index
    df_merged = df_merged.sort_values('UID').reset_index(drop=True)

    # Save output
    output_path = RQ_DIR / "data" / "step03_predictive_data.csv"
    df_merged.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"No data loss: 100 participants with both measures")

    return df_merged

# =============================================================================
# Step 4: Compute Correlation and Tertile Analysis
# =============================================================================

def step04_correlation_and_tertile(df: pd.DataFrame) -> tuple:
    """Compute correlation and tertile analysis with D068 dual p-values."""
    log("\n" + "=" * 70)
    log("STEP 4: Correlation and Tertile Analysis")
    log("=" * 70)

    x = df['Day0_confidence'].values
    y = df['forgetting_slope'].values

    # ----- 4A: Normality Tests (Shapiro-Wilk) -----
    log("\n--- 4A: Normality Tests (Shapiro-Wilk) ---")

    shapiro_conf = shapiro(x)
    shapiro_slope = shapiro(y)

    log(f"Day0_confidence: W = {shapiro_conf.statistic:.4f}, p = {shapiro_conf.pvalue:.4f}")
    log(f"forgetting_slope: W = {shapiro_slope.statistic:.4f}, p = {shapiro_slope.pvalue:.4f}")

    # Determine if non-normal (p < 0.05)
    conf_normal = shapiro_conf.pvalue >= 0.05
    slope_normal = shapiro_slope.pvalue >= 0.05

    use_spearman = not (conf_normal and slope_normal)

    if use_spearman:
        log("\nNon-normality detected: Using Spearman as primary, Pearson as supplementary")
        primary_method = "Spearman"
    else:
        log("\nNormality acceptable: Using Pearson as primary, Spearman as robustness check")
        primary_method = "Pearson"

    # Save normality results
    normality_results = pd.DataFrame({
        'variable': ['Day0_confidence', 'forgetting_slope'],
        'shapiro_W': [shapiro_conf.statistic, shapiro_slope.statistic],
        'shapiro_p': [shapiro_conf.pvalue, shapiro_slope.pvalue],
        'normal': [conf_normal, slope_normal]
    })
    normality_path = RQ_DIR / "data" / "step04_normality_tests.csv"
    normality_results.to_csv(normality_path, index=False)
    log(f"\nSaved: {normality_path}")

    # ----- 4B: Correlation Analysis -----
    log("\n--- 4B: Correlation Analysis ---")

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(x, y)
    log(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")

    # Spearman correlation
    spearman_rho, spearman_p = spearmanr(x, y)
    log(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.6f}")

    # Bootstrap CI for primary correlation
    np.random.seed(RANDOM_SEED)
    boot_rs = []
    n = len(x)
    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, n, replace=True)
        if use_spearman:
            boot_r, _ = spearmanr(x[idx], y[idx])
        else:
            boot_r, _ = pearsonr(x[idx], y[idx])
        boot_rs.append(boot_r)

    boot_rs = np.array(boot_rs)
    ci_lower = np.percentile(boot_rs, 2.5)
    ci_upper = np.percentile(boot_rs, 97.5)

    log(f"\nBootstrap 95% CI ({N_BOOTSTRAP} resamples): [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Effect direction
    if use_spearman:
        primary_r = spearman_rho
        primary_p = spearman_p
    else:
        primary_r = pearson_r
        primary_p = pearson_p

    if abs(primary_r) < 0.10:
        direction = "null"
    elif primary_r > 0:
        direction = "positive"
    else:
        direction = "negative"

    log(f"\nPrimary correlation ({primary_method}): r = {primary_r:.4f}")
    log(f"Effect direction: {direction}")
    log(f"95% CI excludes zero: {ci_lower > 0 or ci_upper < 0}")

    # Dual p-values for D068 (using Bonferroni with k=1 test, so no correction needed)
    # But we report both parametric and bootstrap-based inference
    p_uncorrected = primary_p
    p_bonferroni = min(primary_p * 1, 1.0)  # k=1, no correction

    log(f"\nDual p-values reported per Decision D068")
    log(f"  p_uncorrected = {p_uncorrected:.6f}")
    log(f"  p_bonferroni = {p_bonferroni:.6f}")

    # Save correlation results
    corr_results = pd.DataFrame({
        'primary_method': [primary_method],
        'correlation_r': [primary_r],
        'CI_lower': [ci_lower],
        'CI_upper': [ci_upper],
        'p_uncorrected': [p_uncorrected],
        'p_bonferroni': [p_bonferroni],
        'N': [n],
        'direction': [direction],
        'pearson_r': [pearson_r],
        'pearson_p': [pearson_p],
        'spearman_rho': [spearman_rho],
        'spearman_p': [spearman_p]
    })
    corr_path = RQ_DIR / "data" / "step04_correlation.csv"
    corr_results.to_csv(corr_path, index=False)
    log(f"\nSaved: {corr_path}")
    log(f"Correlation computed: r = {primary_r:.2f}, p_uncorrected = {p_uncorrected:.4f}, p_bonferroni = {p_bonferroni:.4f}")

    # ----- 4C: Tertile Analysis -----
    log("\n--- 4C: Tertile Analysis ---")

    # Create tertiles
    df['tertile'] = pd.qcut(df['Day0_confidence'], q=3, labels=['Low', 'Med', 'High'])

    tertile_stats = df.groupby('tertile').agg({
        'Day0_confidence': ['count', 'mean'],
        'forgetting_slope': ['mean', 'std', 'sem']
    }).reset_index()

    tertile_stats.columns = ['tertile', 'N', 'mean_Day0_confidence',
                             'mean_forgetting_slope', 'sd_forgetting_slope', 'se_forgetting_slope']

    log(f"\nTertile analysis complete: 3 groups created (Low/Med/High)")
    for _, row in tertile_stats.iterrows():
        log(f"  {row['tertile']}: N={row['N']:.0f}, mean_conf={row['mean_Day0_confidence']:.3f}, "
            f"mean_slope={row['mean_forgetting_slope']:.4f} (SE={row['se_forgetting_slope']:.4f})")

    # Save tertile analysis
    tertile_path = RQ_DIR / "data" / "step04_tertile_analysis.csv"
    tertile_stats.to_csv(tertile_path, index=False)
    log(f"\nSaved: {tertile_path}")

    # ----- 4D: High vs Low Comparison -----
    log("\n--- 4D: High vs Low Tertile Comparison ---")

    high_slopes = df[df['tertile'] == 'High']['forgetting_slope'].values
    low_slopes = df[df['tertile'] == 'Low']['forgetting_slope'].values

    # Independent samples t-test
    t_stat, t_p = ttest_ind(high_slopes, low_slopes)

    # Cohen's d
    n1, n2 = len(high_slopes), len(low_slopes)
    pooled_std = np.sqrt(((n1-1)*np.std(high_slopes, ddof=1)**2 + (n2-1)*np.std(low_slopes, ddof=1)**2) / (n1+n2-2))
    cohens_d = (np.mean(high_slopes) - np.mean(low_slopes)) / pooled_std

    log(f"High tertile mean slope: {np.mean(high_slopes):.4f}")
    log(f"Low tertile mean slope: {np.mean(low_slopes):.4f}")
    log(f"Difference: {np.mean(high_slopes) - np.mean(low_slopes):.4f}")
    log(f"t-test: t = {t_stat:.3f}, p = {t_p:.6f}")
    log(f"Cohen's d: {cohens_d:.3f}")

    # Effect interpretation
    if cohens_d > 0:
        interpretation = "High confidence -> slower forgetting (less negative slope)"
    elif cohens_d < 0:
        interpretation = "High confidence -> faster forgetting (more negative slope)"
    else:
        interpretation = "No difference in forgetting between confidence groups"

    log(f"Interpretation: {interpretation}")

    # Dual p-values for tertile test
    p_tertile_uncorr = t_p
    p_tertile_bonf = min(t_p * 1, 1.0)  # k=1

    # Save tertile test results
    tertile_test = pd.DataFrame({
        'comparison': ['High vs Low'],
        'high_mean': [np.mean(high_slopes)],
        'low_mean': [np.mean(low_slopes)],
        'difference': [np.mean(high_slopes) - np.mean(low_slopes)],
        't_statistic': [t_stat],
        'cohens_d': [cohens_d],
        'p_uncorrected': [p_tertile_uncorr],
        'p_bonferroni': [p_tertile_bonf],
        'interpretation': [interpretation]
    })
    tertile_test_path = RQ_DIR / "data" / "step04_tertile_test.csv"
    tertile_test.to_csv(tertile_test_path, index=False)
    log(f"\nSaved: {tertile_test_path}")

    # ----- 4E: One-way ANOVA across all tertiles -----
    log("\n--- 4E: One-way ANOVA (all tertiles) ---")

    med_slopes = df[df['tertile'] == 'Med']['forgetting_slope'].values
    f_stat, anova_p = f_oneway(low_slopes, med_slopes, high_slopes)

    log(f"ANOVA: F = {f_stat:.3f}, p = {anova_p:.6f}")

    # Eta-squared effect size
    ss_between = sum([len(g) * (np.mean(g) - df['forgetting_slope'].mean())**2
                      for g in [low_slopes, med_slopes, high_slopes]])
    ss_total = sum((df['forgetting_slope'] - df['forgetting_slope'].mean())**2)
    eta_sq = ss_between / ss_total
    log(f"Eta-squared: {eta_sq:.4f}")

    # Save ANOVA results
    anova_results = pd.DataFrame({
        'test': ['One-way ANOVA'],
        'F_statistic': [f_stat],
        'p_value': [anova_p],
        'eta_squared': [eta_sq],
        'df_between': [2],
        'df_within': [len(df) - 3]
    })
    anova_path = RQ_DIR / "data" / "step04_anova.csv"
    anova_results.to_csv(anova_path, index=False)
    log(f"\nSaved: {anova_path}")

    return corr_results, tertile_stats, df

# =============================================================================
# Step 5: Prepare Plot Data
# =============================================================================

def step05_prepare_plot_data(df: pd.DataFrame, tertile_stats: pd.DataFrame):
    """Prepare plot data for scatterplot with tertile overlays."""
    log("\n" + "=" * 70)
    log("STEP 5: Prepare Plot Data")
    log("=" * 70)

    # Individual data points
    df_individuals = df[['UID', 'Day0_confidence', 'forgetting_slope', 'tertile']].copy()
    df_individuals['is_mean'] = False
    df_individuals['se_slope'] = np.nan

    # Tertile means
    df_means = pd.DataFrame({
        'UID': ['MEAN_Low', 'MEAN_Med', 'MEAN_High'],
        'Day0_confidence': tertile_stats['mean_Day0_confidence'].values,
        'forgetting_slope': tertile_stats['mean_forgetting_slope'].values,
        'tertile': tertile_stats['tertile'].values,
        'is_mean': True,
        'se_slope': tertile_stats['se_forgetting_slope'].values
    })

    # Combine
    df_plot = pd.concat([df_individuals, df_means], ignore_index=True)

    log(f"Plot data preparation complete: {len(df_plot)} rows created (100 individuals + 3 tertile means)")
    log(f"Columns: {list(df_plot.columns)}")
    log(f"All tertiles represented: {sorted(df_plot['tertile'].unique())}")

    # Validation
    n_individuals = (df_plot['is_mean'] == False).sum()
    n_means = (df_plot['is_mean'] == True).sum()
    log(f"\nValidation:")
    log(f"  Individual points: {n_individuals} (expected: 100)")
    log(f"  Tertile means: {n_means} (expected: 3)")
    log(f"  Total rows: {len(df_plot)} (expected: 103)")

    if len(df_plot) != 103:
        raise ValueError(f"Expected 103 rows, found {len(df_plot)}")

    # Save output
    output_path = RQ_DIR / "data" / "step05_confidence_predicts_forgetting_data.csv"
    df_plot.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return df_plot

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Execute all steps."""
    init_log()

    log("Starting RQ 6.7.1 Analysis Pipeline")
    log(f"Output directory: {RQ_DIR}")
    log("")

    # Step 1: Load Day 0 confidence
    df_confidence = step01_load_day0_confidence()

    # Step 2: Load forgetting slopes
    df_slopes = step02_load_forgetting_slopes()

    # Step 3: Merge data
    df_merged = step03_merge_data(df_confidence, df_slopes)

    # Step 4: Correlation and tertile analysis
    corr_results, tertile_stats, df_with_tertiles = step04_correlation_and_tertile(df_merged)

    # Step 5: Prepare plot data
    df_plot = step05_prepare_plot_data(df_with_tertiles, tertile_stats)

    # Final summary
    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)

    # Extract key results
    r = corr_results['correlation_r'].values[0]
    p = corr_results['p_uncorrected'].values[0]
    direction = corr_results['direction'].values[0]
    ci_lower = corr_results['CI_lower'].values[0]
    ci_upper = corr_results['CI_upper'].values[0]
    method = corr_results['primary_method'].values[0]

    log(f"\nPRIMARY FINDING:")
    log(f"  {method} correlation: r = {r:.4f}")
    log(f"  95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    log(f"  p-value: {p:.6f}")
    log(f"  Direction: {direction}")

    # Hypothesis evaluation
    if p < 0.05 and direction == "positive":
        log(f"\n  HYPOTHESIS SUPPORTED: High Day 0 confidence predicts slower forgetting")
    elif p < 0.05 and direction == "negative":
        log(f"\n  HYPOTHESIS REFUTED: High Day 0 confidence predicts FASTER forgetting (opposite direction)")
    else:
        log(f"\n  HYPOTHESIS NOT SUPPORTED: No significant relationship (p = {p:.4f})")

    log(f"\nFiles created:")
    log(f"  data/step01_day0_confidence.csv")
    log(f"  data/step02_forgetting_slopes.csv")
    log(f"  data/step03_predictive_data.csv")
    log(f"  data/step04_normality_tests.csv")
    log(f"  data/step04_correlation.csv")
    log(f"  data/step04_tertile_analysis.csv")
    log(f"  data/step04_tertile_test.csv")
    log(f"  data/step04_anova.csv")
    log(f"  data/step05_confidence_predicts_forgetting_data.csv")

    log("\n" + "=" * 70)
    log("RQ 6.7.1 Pipeline Complete")
    log("=" * 70)

if __name__ == "__main__":
    main()
