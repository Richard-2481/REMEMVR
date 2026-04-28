#!/usr/bin/env python3
"""
RQ 6.4.2: Paradigm Confidence Calibration
==========================================
Tests whether calibration quality differs across retrieval paradigms (IFR/ICR/IRE).
Calibration = theta_confidence_z - theta_accuracy_z (signed difference).

Hypothesis: Recognition shows highest overconfidence (fluency-familiarity heuristic),
Free Recall shows best calibration (retrieval difficulty provides accurate cue).

Steps:
  00: Load and merge accuracy (Ch5 5.3.1) and confidence (Ch6 6.4.1) data
  01: Fit LMM with Paradigm × Time interaction
  02: Compute post-hoc pairwise paradigm contrasts
  03: Rank paradigms by calibration quality (|calibration|)
  04: Prepare calibration trajectory plot data

Dependencies:
  - Ch5 5.3.1: step03_theta_scores.csv (accuracy theta by paradigm)
  - Ch6 6.4.1: step03_theta_confidence.csv (confidence theta by paradigm)
  - Ch6 6.4.1: step00_tsvr_mapping.csv (TSVR hours)
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.4.2
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"
DATA_DIR = RQ_DIR / "data"

# Source files
ACCURACY_FILE = RQ_DIR.parents[1] / "ch5" / "5.3.1" / "data" / "step03_theta_scores.csv"
CONFIDENCE_FILE = RQ_DIR.parents[1] / "ch6" / "6.4.1" / "data" / "step03_theta_confidence.csv"
TSVR_FILE = RQ_DIR.parents[1] / "ch6" / "6.4.1" / "data" / "step00_tsvr_mapping.csv"

# Paradigm mapping: accuracy file uses lowercase, confidence file uses abbreviations
PARADIGM_MAP_ACC = {
    'free_recall': 'IFR',
    'cued_recall': 'ICR',
    'recognition': 'IRE'
}

PARADIGMS = ['IFR', 'ICR', 'IRE']

# LOGGING

def log(msg: str):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')
        f.flush()
    print(log_msg, flush=True)

def log_section(title: str):
    """Log section header."""
    log("=" * 70)
    log(title)
    log("=" * 70)

# LOAD AND MERGE DATA

def step00_load_merge_data() -> pd.DataFrame:
    """
    Load paradigm-stratified accuracy theta from Ch5 5.3.1 and confidence theta
    from Ch6 6.4.1, merge by UID × TEST × Paradigm, compute calibration metrics.
    """
    log_section("STEP 00: Load and Merge Accuracy/Confidence Data")

    # -------------------------------------------------------------------------
    # Load accuracy data (Ch5 5.3.1) - long format
    # -------------------------------------------------------------------------
    log(f"Loading accuracy data from: {ACCURACY_FILE}")
    if not ACCURACY_FILE.exists():
        raise FileNotFoundError(f"Accuracy file not found: {ACCURACY_FILE}")

    df_acc = pd.read_csv(ACCURACY_FILE)
    log(f"  Loaded accuracy data: {len(df_acc)} rows (long format)")
    log(f"  Columns: {list(df_acc.columns)}")

    # Parse composite_ID: format is UID_TestNum (e.g., A010_1 for Test 1)
    df_acc['UID'] = df_acc['composite_ID'].str.split('_').str[0]
    df_acc['test_num'] = df_acc['composite_ID'].str.split('_').str[1].astype(int)
    df_acc['TEST'] = 'T' + df_acc['test_num'].astype(str)

    # Map paradigm names to standard abbreviations
    df_acc['Paradigm'] = df_acc['domain_name'].map(PARADIGM_MAP_ACC)
    df_acc['theta_accuracy'] = df_acc['theta']

    # Keep only needed columns
    df_acc = df_acc[['UID', 'TEST', 'Paradigm', 'theta_accuracy']]

    log(f"  Accuracy data (processed): {len(df_acc)} rows")
    log(f"  Paradigms: {df_acc['Paradigm'].unique().tolist()}")

    # -------------------------------------------------------------------------
    # Load confidence data (Ch6 6.4.1) - wide format
    # -------------------------------------------------------------------------
    log(f"Loading confidence data from: {CONFIDENCE_FILE}")
    if not CONFIDENCE_FILE.exists():
        raise FileNotFoundError(f"Confidence file not found: {CONFIDENCE_FILE}")

    df_conf_wide = pd.read_csv(CONFIDENCE_FILE)
    log(f"  Loaded confidence data: {len(df_conf_wide)} rows (wide format)")
    log(f"  Columns: {list(df_conf_wide.columns)}")

    # Parse composite_ID: format is UID_T# (e.g., A010_T1)
    df_conf_wide['UID'] = df_conf_wide['composite_ID'].str.split('_').str[0]
    df_conf_wide['TEST'] = df_conf_wide['composite_ID'].str.split('_').str[1]

    # Melt to long format
    paradigm_cols_conf = ['theta_IFR', 'theta_ICR', 'theta_IRE']
    df_conf_long = df_conf_wide.melt(
        id_vars=['UID', 'TEST'],
        value_vars=paradigm_cols_conf,
        var_name='paradigm_col',
        value_name='theta_confidence'
    )

    # Extract paradigm name
    df_conf_long['Paradigm'] = df_conf_long['paradigm_col'].str.replace('theta_', '')
    df_conf_long = df_conf_long.drop(columns=['paradigm_col'])

    log(f"  Confidence data (long format): {len(df_conf_long)} rows")
    log(f"  Paradigms: {df_conf_long['Paradigm'].unique().tolist()}")

    # -------------------------------------------------------------------------
    # Load TSVR mapping
    # -------------------------------------------------------------------------
    log(f"Loading TSVR mapping from: {TSVR_FILE}")
    if not TSVR_FILE.exists():
        raise FileNotFoundError(f"TSVR file not found: {TSVR_FILE}")

    df_tsvr = pd.read_csv(TSVR_FILE)
    df_tsvr = df_tsvr[['UID', 'test', 'TSVR_hours']].drop_duplicates()
    df_tsvr = df_tsvr.rename(columns={'test': 'TEST'})
    log(f"  TSVR mapping: {len(df_tsvr)} rows")

    # -------------------------------------------------------------------------
    # Merge datasets
    # -------------------------------------------------------------------------
    log("Merging accuracy and confidence data...")

    # Inner join on UID, TEST, Paradigm
    df_merged = pd.merge(
        df_acc,
        df_conf_long,
        on=['UID', 'TEST', 'Paradigm'],
        how='inner'
    )
    log(f"  After merge (accuracy + confidence): {len(df_merged)} rows")

    # Merge with TSVR
    df_merged = pd.merge(df_merged, df_tsvr, on=['UID', 'TEST'], how='left')
    log(f"  After TSVR merge: {len(df_merged)} rows")

    # Check for missing TSVR values
    missing_tsvr = df_merged['TSVR_hours'].isna().sum()
    if missing_tsvr > 0:
        log(f"  WARNING: {missing_tsvr} rows missing TSVR_hours")

    # -------------------------------------------------------------------------
    # Z-standardize theta values
    # -------------------------------------------------------------------------
    log("Computing z-standardized theta values...")

    theta_acc_mean = df_merged['theta_accuracy'].mean()
    theta_acc_std = df_merged['theta_accuracy'].std()
    df_merged['theta_accuracy_z'] = (df_merged['theta_accuracy'] - theta_acc_mean) / theta_acc_std

    theta_conf_mean = df_merged['theta_confidence'].mean()
    theta_conf_std = df_merged['theta_confidence'].std()
    df_merged['theta_confidence_z'] = (df_merged['theta_confidence'] - theta_conf_mean) / theta_conf_std

    log(f"  theta_accuracy: mean={theta_acc_mean:.4f}, SD={theta_acc_std:.4f}")
    log(f"  theta_accuracy_z: mean={df_merged['theta_accuracy_z'].mean():.4f}, SD={df_merged['theta_accuracy_z'].std():.4f}")
    log(f"  theta_confidence: mean={theta_conf_mean:.4f}, SD={theta_conf_std:.4f}")
    log(f"  theta_confidence_z: mean={df_merged['theta_confidence_z'].mean():.4f}, SD={df_merged['theta_confidence_z'].std():.4f}")

    # -------------------------------------------------------------------------
    # Compute calibration metrics
    # -------------------------------------------------------------------------
    log("Computing calibration metrics...")

    # Calibration = confidence_z - accuracy_z
    # Positive = overconfidence, Negative = underconfidence, Zero = well-calibrated
    df_merged['calibration'] = df_merged['theta_confidence_z'] - df_merged['theta_accuracy_z']
    df_merged['abs_calibration'] = df_merged['calibration'].abs()

    log(f"  Calibration range: [{df_merged['calibration'].min():.3f}, {df_merged['calibration'].max():.3f}]")
    log(f"  Mean calibration: {df_merged['calibration'].mean():.4f}")
    log(f"  Mean |calibration|: {df_merged['abs_calibration'].mean():.4f}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    log("Validating merged data...")

    n_expected = 1200  # 3 paradigms × 4 tests × 100 participants

    if len(df_merged) < n_expected:
        log(f"  WARNING: Fewer rows than expected: {len(df_merged)} < {n_expected}")

    if df_merged.isna().any().any():
        na_cols = df_merged.columns[df_merged.isna().any()].tolist()
        log(f"  WARNING: NaN values detected in columns: {na_cols}")

    # Check paradigms
    paradigms_found = df_merged['Paradigm'].unique().tolist()
    log(f"  Paradigms in merged data: {paradigms_found}")

    # Check UIDs
    n_uids = df_merged['UID'].nunique()
    log(f"  Unique participants: {n_uids}")

    # Check tests
    tests_found = sorted(df_merged['TEST'].unique().tolist())
    log(f"  Tests: {tests_found}")

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    output_path = DATA_DIR / "step00_calibration_by_paradigm.csv"
    df_merged.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_merged)} rows)")

    return df_merged

# FIT LMM WITH PARADIGM × TIME INTERACTION

def step01_fit_lmm(df: pd.DataFrame) -> dict:
    """
    Fit Linear Mixed Model: calibration ~ Paradigm * TSVR_centered + (TSVR_centered | UID)
    Test Paradigm main effect and Paradigm × Time interaction.
    """
    log_section("STEP 01: Fit LMM with Paradigm × Time Interaction")

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------
    df_lmm = df.copy()

    # Center TSVR_hours
    tsvr_mean = df_lmm['TSVR_hours'].mean()
    df_lmm['TSVR_centered'] = df_lmm['TSVR_hours'] - tsvr_mean
    log(f"TSVR centering: mean = {tsvr_mean:.2f} hours")

    # Convert Paradigm to categorical with proper reference (IFR = Free Recall as reference)
    df_lmm['Paradigm'] = pd.Categorical(df_lmm['Paradigm'], categories=['IFR', 'ICR', 'IRE'])

    # -------------------------------------------------------------------------
    # Fit full model with random intercept and slope
    # -------------------------------------------------------------------------
    log("Fitting LMM: calibration ~ Paradigm * TSVR_centered + (TSVR_centered | UID)")

    formula = "calibration ~ C(Paradigm) * TSVR_centered"

    try:
        # Try with random slopes first
        model = mixedlm(
            formula,
            data=df_lmm,
            groups=df_lmm['UID'],
            re_formula="~TSVR_centered"
        )
        result = model.fit(reml=False)
        random_slope = True
        log("  Model with random slopes converged successfully")
    except Exception as e:
        log(f"  WARNING: Random slope model failed ({str(e)[:50]}), trying intercept-only...")
        # Fall back to random intercept only
        model = mixedlm(
            formula,
            data=df_lmm,
            groups=df_lmm['UID']
        )
        result = model.fit(reml=False)
        random_slope = False
        log("  Model with random intercept only converged")

    # -------------------------------------------------------------------------
    # Extract fixed effects
    # -------------------------------------------------------------------------
    log("Extracting fixed effects...")

    n_fe = len(result.fe_params)
    fe_names = list(result.fe_params.index)
    fe_params = result.fe_params.values
    fe_se = result.bse_fe.values
    fe_z = fe_params / fe_se
    fe_pvals = 2 * (1 - stats.norm.cdf(np.abs(fe_z)))

    df_fe = pd.DataFrame({
        'term': fe_names,
        'estimate': fe_params,
        'SE': fe_se,
        'z': fe_z,
        'p_value': fe_pvals
    })

    log("\nFixed Effects:")
    for _, row in df_fe.iterrows():
        log(f"  {row['term']}: β={row['estimate']:.4f}, SE={row['SE']:.4f}, z={row['z']:.2f}, p={row['p_value']:.4f}")

    # -------------------------------------------------------------------------
    # Test Paradigm main effect using LRT
    # -------------------------------------------------------------------------
    log("\nTesting Paradigm main effect (LRT)...")

    # Reduced model without Paradigm terms
    formula_reduced = "calibration ~ TSVR_centered"
    try:
        if random_slope:
            model_reduced = mixedlm(
                formula_reduced,
                data=df_lmm,
                groups=df_lmm['UID'],
                re_formula="~TSVR_centered"
            )
        else:
            model_reduced = mixedlm(
                formula_reduced,
                data=df_lmm,
                groups=df_lmm['UID']
            )
        result_reduced = model_reduced.fit(reml=False)

        # LRT for Paradigm main effect
        lrt_paradigm = 2 * (result.llf - result_reduced.llf)
        df_paradigm = 2  # 2 Paradigm levels (IFR is reference, so 2 coefficients for ICR and IRE)
        p_paradigm_uncorr = 1 - stats.chi2.cdf(lrt_paradigm, df_paradigm)
        p_paradigm_bonf = min(p_paradigm_uncorr * 2, 1.0)  # Bonferroni for 2 tests

        log(f"  Paradigm main effect: χ²({df_paradigm})={lrt_paradigm:.2f}, p_uncorrected={p_paradigm_uncorr:.4f}, p_bonferroni={p_paradigm_bonf:.4f}")
    except Exception as e:
        log(f"  WARNING: LRT for Paradigm failed: {str(e)[:50]}")
        lrt_paradigm = np.nan
        df_paradigm = 2
        p_paradigm_uncorr = np.nan
        p_paradigm_bonf = np.nan

    # -------------------------------------------------------------------------
    # Test Paradigm × Time interaction using LRT
    # -------------------------------------------------------------------------
    log("\nTesting Paradigm × Time interaction (LRT)...")

    # Reduced model without interaction
    formula_no_interaction = "calibration ~ C(Paradigm) + TSVR_centered"
    try:
        if random_slope:
            model_no_int = mixedlm(
                formula_no_interaction,
                data=df_lmm,
                groups=df_lmm['UID'],
                re_formula="~TSVR_centered"
            )
        else:
            model_no_int = mixedlm(
                formula_no_interaction,
                data=df_lmm,
                groups=df_lmm['UID']
            )
        result_no_int = model_no_int.fit(reml=False)

        # LRT for interaction
        lrt_interaction = 2 * (result.llf - result_no_int.llf)
        df_interaction = 2  # 2 interaction terms (IFR×Time is reference)
        p_interaction_uncorr = 1 - stats.chi2.cdf(lrt_interaction, df_interaction)
        p_interaction_bonf = min(p_interaction_uncorr * 2, 1.0)  # Bonferroni for 2 tests

        log(f"  Paradigm × Time interaction: χ²({df_interaction})={lrt_interaction:.2f}, p_uncorrected={p_interaction_uncorr:.4f}, p_bonferroni={p_interaction_bonf:.4f}")
    except Exception as e:
        log(f"  WARNING: LRT for interaction failed: {str(e)[:50]}")
        lrt_interaction = np.nan
        df_interaction = 2
        p_interaction_uncorr = np.nan
        p_interaction_bonf = np.nan

    # -------------------------------------------------------------------------
    # Save paradigm effects
    # -------------------------------------------------------------------------
    df_effects = pd.DataFrame({
        'term': ['Paradigm_main', 'Paradigm_x_Time_interaction'],
        'statistic': [lrt_paradigm, lrt_interaction],
        'df': [df_paradigm, df_interaction],
        'p_uncorrected': [p_paradigm_uncorr, p_interaction_uncorr],
        'p_bonferroni': [p_paradigm_bonf, p_interaction_bonf],
        'interpretation': [
            'significant' if p_paradigm_bonf < 0.05 else 'not significant',
            'significant' if p_interaction_bonf < 0.05 else 'not significant'
        ]
    })

    effects_path = DATA_DIR / "step01_paradigm_effects.csv"
    df_effects.to_csv(effects_path, index=False)
    log(f"\nSaved: {effects_path}")

    # Save fixed effects
    fe_path = DATA_DIR / "step01_lmm_fixed_effects.csv"
    df_fe.to_csv(fe_path, index=False)
    log(f"Saved: {fe_path}")

    # -------------------------------------------------------------------------
    # Save model summary
    # -------------------------------------------------------------------------
    summary_path = DATA_DIR / "step01_lmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ 6.4.2: LMM Model Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Groups: UID\n")
        f.write(f"Random Effects: {'Intercept + Slope' if random_slope else 'Intercept only'}\n")
        f.write(f"Estimation: ML (REML=False)\n\n")
        f.write("Fixed Effects:\n")
        f.write("-" * 70 + "\n")
        f.write(df_fe.to_string(index=False) + "\n\n")
        f.write("Hypothesis Tests (LRT with dual p-values, Decision D068):\n")
        f.write("-" * 70 + "\n")
        f.write(df_effects.to_string(index=False) + "\n\n")
        f.write("Model Fit:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Log-likelihood: {result.llf:.2f}\n")
        f.write(f"AIC: {result.aic:.2f}\n")
        f.write(f"BIC: {result.bic:.2f}\n")

    log(f"Saved: {summary_path}")

    return {
        'result': result,
        'df_lmm': df_lmm,
        'df_effects': df_effects,
        'df_fe': df_fe,
        'random_slope': random_slope
    }

# POST-HOC PARADIGM CONTRASTS

def step02_post_hoc_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise post-hoc contrasts between paradigms.
    """
    log_section("STEP 02: Post-Hoc Paradigm Contrasts")

    # Compute paradigm-level means and SEs
    paradigm_stats = df.groupby('Paradigm')['calibration'].agg(['mean', 'std', 'count']).reset_index()
    paradigm_stats.columns = ['Paradigm', 'mean', 'std', 'n']
    paradigm_stats['se'] = paradigm_stats['std'] / np.sqrt(paradigm_stats['n'])

    log("Paradigm-level calibration statistics:")
    for _, row in paradigm_stats.iterrows():
        cal_type = "overconfidence" if row['mean'] > 0 else "underconfidence" if row['mean'] < 0 else "well-calibrated"
        log(f"  {row['Paradigm']}: mean={row['mean']:.4f} ({cal_type}), SD={row['std']:.4f}, N={row['n']}")

    # Define contrasts (hypothesis-driven: IRE vs IFR is primary)
    contrasts = [
        ('IRE vs IFR', 'IRE', 'IFR'),
        ('ICR vs IFR', 'ICR', 'IFR'),
        ('IRE vs ICR', 'IRE', 'ICR')
    ]

    results = []

    for contrast_name, paradigm1, paradigm2 in contrasts:
        # Check if both paradigms exist
        if paradigm1 not in paradigm_stats['Paradigm'].values or paradigm2 not in paradigm_stats['Paradigm'].values:
            log(f"  Skipping {contrast_name}: one or both paradigms not available")
            continue

        stats1 = paradigm_stats[paradigm_stats['Paradigm'] == paradigm1].iloc[0]
        stats2 = paradigm_stats[paradigm_stats['Paradigm'] == paradigm2].iloc[0]

        # Estimate = mean1 - mean2
        estimate = stats1['mean'] - stats2['mean']

        # Pooled SE for two-sample comparison
        se = np.sqrt(stats1['se']**2 + stats2['se']**2)

        # z-test
        z = estimate / se
        p_uncorr = 2 * (1 - stats.norm.cdf(np.abs(z)))

        # Bonferroni correction (3 comparisons)
        n_comparisons = 3
        p_bonf = min(p_uncorr * n_comparisons, 1.0)

        # Cohen's d
        pooled_std = np.sqrt(((stats1['n']-1)*stats1['std']**2 + (stats2['n']-1)*stats2['std']**2) /
                            (stats1['n'] + stats2['n'] - 2))
        cohens_d = estimate / pooled_std if pooled_std > 0 else 0

        results.append({
            'contrast': contrast_name,
            'estimate': estimate,
            'SE': se,
            'z': z,
            'p_uncorrected': p_uncorr,
            'p_bonferroni': p_bonf,
            'cohens_d': cohens_d,
            'interpretation': 'significant' if p_bonf < 0.05 else 'not significant'
        })

        log(f"  {contrast_name}: Δ={estimate:.4f}, z={z:.2f}, p_uncorr={p_uncorr:.4f}, p_bonf={p_bonf:.4f}, d={cohens_d:.3f}")

    df_contrasts = pd.DataFrame(results)

    # Save
    output_path = DATA_DIR / "step02_post_hoc_contrasts.csv"
    df_contrasts.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return df_contrasts

# RANK PARADIGMS BY CALIBRATION QUALITY

def step03_rank_paradigms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank paradigms by calibration quality (lower |calibration| = better calibrated).
    """
    log_section("STEP 03: Rank Paradigms by Calibration Quality")

    # Compute paradigm-level statistics on abs_calibration
    paradigm_ranking = df.groupby('Paradigm')['abs_calibration'].agg(['mean', 'std', 'count']).reset_index()
    paradigm_ranking.columns = ['Paradigm', 'mean_abs_calibration', 'sd_abs_calibration', 'N']

    # Sort by mean_abs_calibration (ascending = better calibrated first)
    paradigm_ranking = paradigm_ranking.sort_values('mean_abs_calibration').reset_index(drop=True)

    # Assign ranks (1 = best calibrated)
    paradigm_ranking['rank'] = range(1, len(paradigm_ranking) + 1)

    # Interpretations
    n_paradigms = len(paradigm_ranking)
    interpretations = []
    for i in range(n_paradigms):
        if i == 0:
            interpretations.append('Best calibrated')
        elif i == n_paradigms - 1:
            interpretations.append('Worst calibrated')
        else:
            interpretations.append('Middle')
    paradigm_ranking['interpretation'] = interpretations

    log("Paradigm ranking by calibration quality:")
    for _, row in paradigm_ranking.iterrows():
        log(f"  Rank {row['rank']}: {row['Paradigm']} - mean |calibration|={row['mean_abs_calibration']:.4f} ({row['interpretation']})")

    # Save
    output_path = DATA_DIR / "step03_paradigm_ranking.csv"
    paradigm_ranking.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return paradigm_ranking

# PREPARE PLOT DATA

def step04_prepare_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare calibration trajectory plot data (calibration by Paradigm × Time).
    """
    log_section("STEP 04: Prepare Calibration Trajectory Plot Data")

    # Aggregate by Paradigm × TEST
    plot_data = df.groupby(['Paradigm', 'TEST']).agg({
        'calibration': ['mean', 'std', 'count'],
        'TSVR_hours': 'mean'
    }).reset_index()

    # Flatten column names
    plot_data.columns = ['Paradigm', 'TEST', 'mean_calibration', 'std_calibration', 'N', 'TSVR_hours']

    # Compute 95% CI
    plot_data['SE'] = plot_data['std_calibration'] / np.sqrt(plot_data['N'])
    plot_data['CI_lower'] = plot_data['mean_calibration'] - 1.96 * plot_data['SE']
    plot_data['CI_upper'] = plot_data['mean_calibration'] + 1.96 * plot_data['SE']

    # Select and sort
    plot_data = plot_data[['Paradigm', 'TEST', 'TSVR_hours', 'mean_calibration', 'CI_lower', 'CI_upper']]
    plot_data = plot_data.sort_values(['Paradigm', 'TSVR_hours']).reset_index(drop=True)

    log("Plot data summary:")
    log(f"  Rows: {len(plot_data)}")
    log(f"  Paradigms: {plot_data['Paradigm'].unique().tolist()}")
    log(f"  Tests: {plot_data['TEST'].unique().tolist()}")

    # Log trajectory patterns
    for paradigm in plot_data['Paradigm'].unique():
        paradigm_data = plot_data[plot_data['Paradigm'] == paradigm]
        t1_cal = paradigm_data[paradigm_data['TEST'] == 'T1']['mean_calibration'].values[0]
        t4_cal = paradigm_data[paradigm_data['TEST'] == 'T4']['mean_calibration'].values[0]
        log(f"  {paradigm}: T1={t1_cal:.3f} → T4={t4_cal:.3f} (Δ={t4_cal-t1_cal:.3f})")

    # Save
    output_path = DATA_DIR / "step04_calibration_trajectory_data.csv"
    plot_data.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")

    return plot_data

# MAIN

def main():
    """Execute all analysis steps."""
    log_section("RQ 6.4.2: Paradigm Confidence Calibration")
    log(f"Started: {datetime.now()}")
    log(f"RQ_DIR: {RQ_DIR}")

    try:
        # Step 00: Load and merge data
        df_calibration = step00_load_merge_data()

        # Step 01: Fit LMM
        lmm_results = step01_fit_lmm(df_calibration)

        # Step 02: Post-hoc contrasts
        df_contrasts = step02_post_hoc_contrasts(df_calibration)

        # Step 03: Rank paradigms
        df_ranking = step03_rank_paradigms(df_calibration)

        # Step 04: Prepare plot data
        df_plot_data = step04_prepare_plot_data(df_calibration)

        log_section("EXECUTION COMPLETE")
        log(f"All steps completed successfully")
        log(f"Ended: {datetime.now()}")

    except Exception as e:
        log(f"ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
