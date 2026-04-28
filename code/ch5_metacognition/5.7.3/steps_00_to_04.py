"""
RQ 6.7.3: Calibration Predicts Trajectory Stability
=====================================================

Purpose: Test whether Day 0 calibration quality predicts forgetting trajectory
         stability (SD of residuals from LMM fit).

Hypothesis: Good calibration (low |calibration|) → lower trajectory variability
            (more stable, predictable forgetting patterns).

Dependencies:
- RQ 6.2.1: Day 0 calibration scores (confidence - accuracy alignment)
- Ch5 5.1.1: LMM trajectory model (to compute residuals)

Steps:
- Step 00: Extract calibration from 6.2.1, compute residuals from 5.1.1
- Step 01: Compute trajectory variability (SD of residuals per participant)
- Step 02: Merge calibration and variability
- Step 03: Compute correlation with dual p-values (D068)
- Step 04: Prepare scatterplot data

Created: 2025-12-12
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from datetime import datetime

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.7.3
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"

# Dependency paths
CALIBRATION_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.2.1/data/step02_calibration_scores.csv")
LMM_INPUT_FILE = Path("/home/etai/projects/REMEMVR/results/ch5/5.1.1/data/step04_lmm_input.csv")

# Best model from Ch5 5.1.1: PowerLaw_04 (theta ~ (TSVR_hours+1)^(-0.4))
POWER_LAW_ALPHA = 0.4


def log(msg):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + "\n")
        f.flush()
    print(log_msg, flush=True)


# Extract Calibration and Compute Residuals

def step00_extract_data():
    """
    Load Day 0 calibration from RQ 6.2.1 and compute trajectory residuals
    from Ch5 5.1.1 best model (PowerLaw_04).
    """
    log("=" * 70)
    log("STEP 00: Extract Calibration and Compute Residuals")
    log("=" * 70)

    # -----------------------------
    # 1. Load and filter calibration scores (Day 0 / T1 only)
    # -----------------------------
    log(f"Loading calibration scores from {CALIBRATION_FILE}")

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_FILE}")

    df_calib = pd.read_csv(CALIBRATION_FILE)
    log(f"Loaded {len(df_calib)} calibration scores from RQ 6.2.1")

    # Filter to T1 (Day 0) only
    df_calib_day0 = df_calib[df_calib['test'] == 'T1'].copy()
    log(f"Filtered to {len(df_calib_day0)} Day 0 (T1) calibration scores")

    # Validate 100 participants
    if len(df_calib_day0) != 100:
        raise ValueError(f"Expected 100 Day 0 calibration scores, found {len(df_calib_day0)}")

    # Keep only UID and calibration
    df_calib_day0 = df_calib_day0[['UID', 'calibration']].reset_index(drop=True)

    # Save
    output_path = RQ_DIR / "data" / "step00_calibration_day0.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_calib_day0.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_calib_day0)} rows)")

    # -----------------------------
    # 2. Load LMM input and compute residuals from best model
    # -----------------------------
    log(f"Loading LMM input from {LMM_INPUT_FILE}")

    if not LMM_INPUT_FILE.exists():
        raise FileNotFoundError(f"LMM input file not found: {LMM_INPUT_FILE}")

    df_lmm = pd.read_csv(LMM_INPUT_FILE)
    log(f"Loaded {len(df_lmm)} observations from Ch5 5.1.1")

    # Create power law time predictor: (TSVR_hours + 1)^(-0.4)
    df_lmm['time_powerlaw'] = (df_lmm['TSVR_hours'] + 1) ** (-POWER_LAW_ALPHA)

    # Fit PowerLaw_04 LMM: theta ~ time_powerlaw + (1 | UID)
    log(f"Refitting best model: theta ~ time_powerlaw + (1 | UID), alpha={POWER_LAW_ALPHA}")
    model = smf.mixedlm("theta ~ time_powerlaw", df_lmm, groups=df_lmm['UID'])
    result = model.fit(method='powell', reml=True)

    log(f"Model converged: {result.converged}")
    log(f"Fixed effects: intercept={result.fe_params['Intercept']:.4f}, slope={result.fe_params['time_powerlaw']:.4f}")

    # Compute residuals
    df_lmm['fitted'] = result.fittedvalues
    df_lmm['residual'] = df_lmm['theta'] - df_lmm['fitted']

    # Extract UID and test from composite_ID
    # composite_ID format: "A010_1" -> UID="A010", test="T1"
    df_lmm['test_num'] = df_lmm['composite_ID'].str.split('_').str[1].astype(int)
    df_lmm['test'] = 'T' + df_lmm['test_num'].astype(str)

    log(f"Residuals computed: mean={df_lmm['residual'].mean():.4f}, SD={df_lmm['residual'].std():.4f}")

    # Select columns for output
    df_residuals = df_lmm[['composite_ID', 'UID', 'test', 'residual']].copy()

    # Validate 400 rows (100 × 4 tests)
    if len(df_residuals) != 400:
        raise ValueError(f"Expected 400 residuals, found {len(df_residuals)}")

    # Save
    output_path = RQ_DIR / "data" / "step00_trajectory_residuals.csv"
    df_residuals.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_residuals)} rows)")

    # Validate UID matching
    uids_calib = set(df_calib_day0['UID'])
    uids_resid = set(df_residuals['UID'])
    if uids_calib != uids_resid:
        raise ValueError(f"UID mismatch: {len(uids_calib)} calibration vs {len(uids_resid)} residuals")

    log("Step 00 COMPLETE: Extracted 100 Day 0 calibration scores and 400 residuals")
    log("All 100 participants have complete data")

    return df_calib_day0, df_residuals


# Compute Trajectory Variability (SD of Residuals per Participant)

def step01_compute_variability(df_residuals):
    """
    Compute SD of residuals across 4 timepoints for each participant.
    This represents trajectory variability (stability of forgetting pattern).
    """
    log("=" * 70)
    log("STEP 01: Compute Trajectory Variability")
    log("=" * 70)

    # Group by UID and compute SD
    df_variability = df_residuals.groupby('UID').agg(
        n_tests=('residual', 'count'),
        trajectory_variability=('residual', 'std')
    ).reset_index()

    log(f"Computed trajectory variability for {len(df_variability)} participants")

    # Validate all have 4 tests
    if not (df_variability['n_tests'] == 4).all():
        bad_uids = df_variability[df_variability['n_tests'] != 4]['UID'].tolist()
        raise ValueError(f"UIDs with != 4 tests: {bad_uids}")

    # Drop n_tests column (validation only)
    df_variability = df_variability[['UID', 'trajectory_variability']]

    # Validate range (SD must be >= 0)
    if (df_variability['trajectory_variability'] < 0).any():
        raise ValueError("Negative trajectory variability values detected (impossible)")

    # Log summary statistics
    mean_var = df_variability['trajectory_variability'].mean()
    sd_var = df_variability['trajectory_variability'].std()
    min_var = df_variability['trajectory_variability'].min()
    max_var = df_variability['trajectory_variability'].max()

    log(f"Mean variability: {mean_var:.4f}, SD variability: {sd_var:.4f}")
    log(f"Range: [{min_var:.4f}, {max_var:.4f}]")

    # Save
    output_path = RQ_DIR / "data" / "step01_trajectory_variability.csv"
    df_variability.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_variability)} rows)")

    log("Step 01 COMPLETE: Computed trajectory variability for 100 participants")

    return df_variability


# Merge Calibration and Variability

def step02_merge_data(df_calib_day0, df_variability):
    """
    Merge Day 0 calibration with trajectory variability for correlation analysis.
    """
    log("=" * 70)
    log("STEP 02: Merge Calibration and Variability")
    log("=" * 70)

    # Inner join on UID
    df_merged = pd.merge(df_calib_day0, df_variability, on='UID', how='inner')

    log(f"Merged calibration and variability: {len(df_merged)} participants with complete data")

    # Validate 100 rows
    if len(df_merged) != 100:
        raise ValueError(f"Expected 100 participants after merge, found {len(df_merged)}")

    # Check for NaN
    if df_merged.isnull().any().any():
        raise ValueError("NaN values detected in merged data")

    # Check for duplicates
    if df_merged.duplicated(subset=['UID']).any():
        raise ValueError("Duplicate UIDs detected in merged data")

    # Log summary
    log(f"Calibration: mean={df_merged['calibration'].mean():.4f}, SD={df_merged['calibration'].std():.4f}")
    log(f"Variability: mean={df_merged['trajectory_variability'].mean():.4f}, SD={df_merged['trajectory_variability'].std():.4f}")

    # Save
    output_path = RQ_DIR / "data" / "step02_calibration_variability.csv"
    df_merged.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_merged)} rows)")

    log("Step 02 COMPLETE: Merged calibration and variability (100 participants)")

    return df_merged


# Compute Correlation with Dual P-Values (D068)

def step03_compute_correlation(df_merged):
    """
    Test Pearson correlation between Day 0 calibration and trajectory variability.
    Implements Decision D068: dual p-value reporting (one-tailed and two-tailed).

    Hypothesis: Better calibration (lower |calibration|) → lower variability
    Since calibration = confidence - accuracy (z-scores), we're testing
    whether extreme calibration (over/underconfidence) predicts instability.

    NOTE: The hypothesis tests raw calibration (not absolute value) vs variability.
    Negative r would mean: overconfidence → higher variability
    Positive r would mean: overconfidence → lower variability
    """
    log("=" * 70)
    log("STEP 03: Compute Correlation with Dual P-Values (D068)")
    log("=" * 70)

    x = df_merged['calibration'].values
    y = df_merged['trajectory_variability'].values
    n = len(x)

    # Compute Pearson correlation
    r, p_two_tailed = stats.pearsonr(x, y)

    # Compute one-tailed p-value
    # Hypothesis expects negative r (better calibration [closer to 0] → lower variability)
    # But raw calibration isn't directional, so we report one-tailed for EITHER direction
    # If r < 0: one-tailed = p_two_tailed / 2 (testing r < 0)
    # If r > 0: one-tailed = p_two_tailed / 2 (testing r > 0)
    p_one_tailed = p_two_tailed / 2

    # Classify effect size
    abs_r = abs(r)
    if abs_r >= 0.50:
        effect_size = "large"
    elif abs_r >= 0.30:
        effect_size = "moderate"
    elif abs_r >= 0.20:
        effect_size = "small"
    else:
        effect_size = "negligible"

    # Classify direction
    if abs_r < 0.10:
        direction = "null"
    elif r > 0:
        direction = "positive"
    else:
        direction = "negative"

    # Log results
    log(f"Pearson r = {r:.4f}, p (one-tailed) = {p_one_tailed:.4f}, p (two-tailed) = {p_two_tailed:.4f}")
    log(f"Effect size: {effect_size}")
    log(f"Direction: {direction}")
    log(f"Sample size: N = {n}")

    # Interpret significance
    if p_two_tailed < 0.05:
        log(f"SIGNIFICANT at alpha=0.05 (two-tailed)")
    else:
        log(f"NOT significant at alpha=0.05 (two-tailed)")

    # Create result DataFrame
    result = pd.DataFrame({
        'r': [r],
        'p_one_tailed': [p_one_tailed],
        'p_two_tailed': [p_two_tailed],
        'n': [n],
        'effect_size': [effect_size],
        'direction': [direction]
    })

    # Save
    output_path = RQ_DIR / "data" / "step03_correlation.csv"
    result.to_csv(output_path, index=False)
    log(f"Saved: {output_path} (1 row)")

    log("Step 03 COMPLETE: Correlation computed with D068 dual p-values")

    return result


# Prepare Scatterplot Data

def step04_prepare_plot_data(df_merged):
    """
    Prepare scatterplot data with regression line for visualization.
    """
    log("=" * 70)
    log("STEP 04: Prepare Scatterplot Data")
    log("=" * 70)

    x = df_merged['calibration'].values
    y = df_merged['trajectory_variability'].values

    # Compute regression line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_predicted = slope * x + intercept

    log(f"Regression line: y = {slope:.4f}x + {intercept:.4f}")

    # Create plot DataFrame
    df_plot = pd.DataFrame({
        'calibration': x,
        'trajectory_variability': y,
        'y_predicted': y_predicted
    })

    # Validate
    if len(df_plot) != 100:
        raise ValueError(f"Expected 100 rows, found {len(df_plot)}")

    if df_plot.isnull().any().any():
        raise ValueError("NaN values in plot data")

    # Save
    output_path = RQ_DIR / "data" / "step04_scatterplot_data.csv"
    df_plot.to_csv(output_path, index=False)
    log(f"Saved: {output_path} ({len(df_plot)} rows)")

    log("Step 04 COMPLETE: Plot data prepared with regression line")

    return df_plot


# MAIN EXECUTION

def main():
    """Execute all steps for RQ 6.7.3."""
    log("=" * 70)
    log("RQ 6.7.3: Calibration Predicts Trajectory Stability")
    log("=" * 70)
    log("")

    try:
        # Step 00: Extract data
        df_calib_day0, df_residuals = step00_extract_data()
        log("")

        # Step 01: Compute variability
        df_variability = step01_compute_variability(df_residuals)
        log("")

        # Step 02: Merge data
        df_merged = step02_merge_data(df_calib_day0, df_variability)
        log("")

        # Step 03: Compute correlation
        result = step03_compute_correlation(df_merged)
        log("")

        # Step 04: Prepare plot data
        df_plot = step04_prepare_plot_data(df_merged)
        log("")

        log("=" * 70)
        log("ALL STEPS COMPLETE")
        log("=" * 70)

        # Final summary
        r = result['r'].values[0]
        p_two = result['p_two_tailed'].values[0]
        effect = result['effect_size'].values[0]
        direction = result['direction'].values[0]

        log("")
        log("SUMMARY:")
        log(f"  Pearson r = {r:.4f}")
        log(f"  p (two-tailed) = {p_two:.4f}")
        log(f"  Effect size: {effect}")
        log(f"  Direction: {direction}")
        log("")

        if p_two < 0.05:
            log("HYPOTHESIS STATUS: SUPPORTED - Calibration predicts trajectory stability")
        else:
            log("HYPOTHESIS STATUS: NOT SUPPORTED - No significant relationship found")

    except Exception as e:
        log(f"ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
