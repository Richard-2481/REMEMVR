#!/usr/bin/env python3
"""
RQ 6.8.2: Source-Destination Calibration
=========================================
Computes calibration = confidence - accuracy by location type (Source vs Destination)
Tests whether calibration quality differs between source and destination spatial memories.

Steps:
  00: Merge accuracy (Ch5 5.5.1) and confidence (Ch6 6.8.1) by UID x TEST x LocationType
  01: Compute calibration = Z_confidence - Z_accuracy (z-standardized within location type)
  02: Fit LMM: calibration ~ LocationType * log_TSVR + (1 | UID)
  03: Prepare plot data (calibration by LocationType x Time)

Dependencies:
  - Ch5 5.5.1: results/ch5/5.5.1/data/step04_lmm_input.csv (accuracy theta)
  - Ch6 6.8.1: results/ch6/6.8.1/data/step04_lmm_input.csv (confidence theta)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.8.2
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR project root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_03.log"

# Dependency paths
ACCURACY_PATH = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step04_lmm_input.csv"
CONFIDENCE_PATH = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step04_lmm_input.csv"

def log(msg):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Merge Accuracy and Confidence by Location Type

def step00_merge_accuracy_confidence():
    """
    Merge accuracy theta (Ch5 5.5.1) with confidence theta (Ch6 6.8.1).
    Both files have 800 rows: 100 UID x 4 tests x 2 location types.
    """
    log("\n" + "="*70)
    log("STEP 00: Merge Accuracy and Confidence by Location Type")
    log("="*70)

    # Check dependencies exist
    if not ACCURACY_PATH.exists():
        raise FileNotFoundError(f"Accuracy file not found: {ACCURACY_PATH}")
    if not CONFIDENCE_PATH.exists():
        raise FileNotFoundError(f"Confidence file not found: {CONFIDENCE_PATH}")

    # Load accuracy data (Ch5 5.5.1)
    log(f"\nLoading accuracy data from: {ACCURACY_PATH}")
    df_acc = pd.read_csv(ACCURACY_PATH)
    log(f"  Accuracy data: {len(df_acc)} rows, columns: {list(df_acc.columns)}")

    # Standardize column names for accuracy
    # Ch5 has: UID, test, LocationType, theta, TSVR_hours
    df_acc = df_acc.rename(columns={
        'test': 'TEST',
        'theta': 'theta_accuracy',
        'se': 'SE_accuracy'
    })

    # Convert TEST to string format (T1, T2, T3, T4)
    df_acc['TEST'] = 'T' + df_acc['TEST'].astype(str)

    # Standardize LocationType capitalization
    df_acc['LocationType'] = df_acc['LocationType'].str.capitalize()

    # Select relevant columns
    df_acc = df_acc[['UID', 'TEST', 'LocationType', 'theta_accuracy', 'TSVR_hours']].copy()
    log(f"  After cleanup: {len(df_acc)} rows")

    # Load confidence data (Ch6 6.8.1)
    log(f"\nLoading confidence data from: {CONFIDENCE_PATH}")
    df_conf = pd.read_csv(CONFIDENCE_PATH)
    log(f"  Confidence data: {len(df_conf)} rows, columns: {list(df_conf.columns)}")

    # Standardize column names for confidence
    # Ch6 6.8.1 has: UID, TEST, location, theta, log_TSVR
    df_conf = df_conf.rename(columns={
        'location': 'LocationType',
        'theta': 'theta_confidence'
    })

    # Select relevant columns
    df_conf = df_conf[['UID', 'TEST', 'LocationType', 'theta_confidence']].copy()
    log(f"  After cleanup: {len(df_conf)} rows")

    # Merge on UID x TEST x LocationType
    log("\nMerging accuracy + confidence on UID x TEST x LocationType...")
    df_merged = pd.merge(
        df_acc,
        df_conf,
        on=['UID', 'TEST', 'LocationType'],
        how='inner'
    )

    log(f"  Merged data: {len(df_merged)} rows (expected: 800)")

    if len(df_merged) != 800:
        log(f"  WARNING: Expected 800 rows, got {len(df_merged)}")
        # Check what's missing
        n_uid = df_merged['UID'].nunique()
        n_test = df_merged['TEST'].nunique()
        n_loc = df_merged['LocationType'].nunique()
        log(f"  Unique UIDs: {n_uid}, Tests: {n_test}, LocationTypes: {n_loc}")

    # Validate merge
    assert df_merged['theta_accuracy'].notna().all(), "Missing accuracy values after merge"
    assert df_merged['theta_confidence'].notna().all(), "Missing confidence values after merge"

    # Save merged data
    output_path = RQ_DIR / "data" / "step00_accuracy_confidence_merged.csv"
    df_merged.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"  Columns: {list(df_merged.columns)}")

    # Log descriptives
    log("\nDescriptive Statistics:")
    for loc in ['Source', 'Destination']:
        subset = df_merged[df_merged['LocationType'] == loc]
        log(f"  {loc}:")
        log(f"    theta_accuracy:   mean={subset['theta_accuracy'].mean():.3f}, SD={subset['theta_accuracy'].std():.3f}")
        log(f"    theta_confidence: mean={subset['theta_confidence'].mean():.3f}, SD={subset['theta_confidence'].std():.3f}")

    log(f"\nMerge complete: {len(df_merged)} rows created")
    return df_merged

# Compute Calibration per Location Type

def step01_compute_calibration(df_merged):
    """
    Z-standardize accuracy and confidence within each LocationType.
    Compute calibration = Z_confidence - Z_accuracy.

    Positive calibration = overconfidence (confidence > accuracy)
    Negative calibration = underconfidence (accuracy > confidence)
    Zero calibration = well-calibrated
    """
    log("\n" + "="*70)
    log("STEP 01: Compute Calibration per Location Type")
    log("="*70)

    df = df_merged.copy()

    # Z-standardize within each LocationType
    log("\nZ-standardizing within each LocationType...")

    df['Z_accuracy'] = np.nan
    df['Z_confidence'] = np.nan

    for loc in ['Source', 'Destination']:
        mask = df['LocationType'] == loc

        # Z-standardize accuracy
        mean_acc = df.loc[mask, 'theta_accuracy'].mean()
        std_acc = df.loc[mask, 'theta_accuracy'].std()
        df.loc[mask, 'Z_accuracy'] = (df.loc[mask, 'theta_accuracy'] - mean_acc) / std_acc

        # Z-standardize confidence
        mean_conf = df.loc[mask, 'theta_confidence'].mean()
        std_conf = df.loc[mask, 'theta_confidence'].std()
        df.loc[mask, 'Z_confidence'] = (df.loc[mask, 'theta_confidence'] - mean_conf) / std_conf

        log(f"\n  {loc}:")
        log(f"    Accuracy:   mean={mean_acc:.3f}, SD={std_acc:.3f}")
        log(f"    Confidence: mean={mean_conf:.3f}, SD={std_conf:.3f}")
        log(f"    Z_accuracy:   mean={df.loc[mask, 'Z_accuracy'].mean():.4f}, SD={df.loc[mask, 'Z_accuracy'].std():.4f}")
        log(f"    Z_confidence: mean={df.loc[mask, 'Z_confidence'].mean():.4f}, SD={df.loc[mask, 'Z_confidence'].std():.4f}")

    # Compute calibration
    df['calibration'] = df['Z_confidence'] - df['Z_accuracy']

    log("\nCalibration computed:")
    log(f"  calibration = Z_confidence - Z_accuracy")
    log(f"  Positive = overconfidence, Negative = underconfidence, Zero = well-calibrated")

    # Calibration descriptives by location
    log("\nCalibration by LocationType:")
    for loc in ['Source', 'Destination']:
        subset = df[df['LocationType'] == loc]
        cal_mean = subset['calibration'].mean()
        cal_sd = subset['calibration'].std()
        log(f"  {loc}: mean={cal_mean:.3f}, SD={cal_sd:.3f}")
        if cal_mean > 0.1:
            log(f"    -> OVERCONFIDENT (confidence > accuracy)")
        elif cal_mean < -0.1:
            log(f"    -> UNDERCONFIDENT (accuracy > confidence)")
        else:
            log(f"    -> WELL-CALIBRATED (|mean| < 0.1)")

    # Save calibration data
    output_path = RQ_DIR / "data" / "step01_calibration_by_location.csv"
    df.to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"  Rows: {len(df)}")
    log(f"  Columns: {list(df.columns)}")

    log(f"\nCalibration computed: {len(df)} observations")
    return df

# Fit LMM Testing Location Effects on Calibration

def step02_fit_lmm_calibration(df_calibration):
    """
    Fit LMM: calibration ~ LocationType * log_TSVR + (1 | UID)

    Tests:
    - LocationType main effect: Does source vs destination differ in overall calibration?
    - LocationType x log_TSVR interaction: Does calibration diverge over time?
    """
    log("\n" + "="*70)
    log("STEP 02: Fit LMM Testing Location Effects on Calibration")
    log("="*70)

    df = df_calibration.copy()

    # Compute log_TSVR for time variable
    df['log_TSVR'] = np.log(df['TSVR_hours'] + 1)

    log(f"\nTime variable: log_TSVR = log(TSVR_hours + 1)")
    log(f"  Range: {df['log_TSVR'].min():.2f} to {df['log_TSVR'].max():.2f}")

    # Fit LMM
    log("\nFitting LMM: calibration ~ C(LocationType) * log_TSVR")
    log("  Random effects: (1 | UID) - random intercepts only")

    # Create dummy variable for LocationType (Source=1, Destination=0)
    df['LocationType_Source'] = (df['LocationType'] == 'Source').astype(int)

    formula = "calibration ~ LocationType_Source * log_TSVR"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df['UID'],
            re_formula="~1"  # Random intercept only
        )
        result = model.fit(reml=True)

    log("\n" + "="*50)
    log("LMM Results")
    log("="*50)
    log(str(result.summary()))

    # Save full summary
    summary_path = RQ_DIR / "data" / "step02_lmm_calibration_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"\nSaved: {summary_path}")

    # Extract fixed effects
    log("\nFixed Effects:")

    # Get number of fixed effects
    n_fe = len(result.model.exog_names)
    fixed_names = result.model.exog_names
    fixed_params = result.params[:n_fe]
    fixed_bse = result.bse[:n_fe]
    fixed_tvalues = result.tvalues[:n_fe]
    fixed_pvalues = result.pvalues[:n_fe]

    # Create fixed effects table
    fe_df = pd.DataFrame({
        'Effect': fixed_names,
        'Estimate': fixed_params.values,
        'SE': fixed_bse.values,
        't': fixed_tvalues.values,
        'p_uncorrected': fixed_pvalues.values
    })

    # Add Bonferroni correction (4 tests: intercept, location, time, interaction)
    fe_df['p_bonferroni'] = np.minimum(fe_df['p_uncorrected'] * 4, 1.0)

    # Compute 95% CI
    fe_df['CI_lower'] = fe_df['Estimate'] - 1.96 * fe_df['SE']
    fe_df['CI_upper'] = fe_df['Estimate'] + 1.96 * fe_df['SE']

    log(fe_df.to_string(index=False))

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "step02_location_effects.csv"
    fe_df.to_csv(fe_path, index=False)
    log(f"\nSaved: {fe_path}")

    # Key hypothesis tests
    log("\n" + "="*50)
    log("KEY HYPOTHESIS TESTS")
    log("="*50)

    # LocationType main effect
    loc_row = fe_df[fe_df['Effect'].str.contains('LocationType')].iloc[0]
    log(f"\nLocationType Main Effect (Source vs Destination):")
    log(f"  Estimate: {loc_row['Estimate']:.4f}")
    log(f"  SE: {loc_row['SE']:.4f}")
    log(f"  t = {loc_row['t']:.2f}")
    log(f"  p_uncorrected = {loc_row['p_uncorrected']:.4f}")
    log(f"  p_bonferroni = {loc_row['p_bonferroni']:.4f}")

    if loc_row['p_uncorrected'] < 0.05:
        log(f"  -> SIGNIFICANT (p < 0.05)")
        if loc_row['Estimate'] > 0:
            log(f"  -> Source MORE overconfident than Destination")
        else:
            log(f"  -> Source LESS overconfident than Destination (better calibrated)")
    else:
        log(f"  -> NOT SIGNIFICANT (p >= 0.05)")

    # Interaction effect
    int_row = fe_df[fe_df['Effect'].str.contains(':')].iloc[0] if any(fe_df['Effect'].str.contains(':')) else None
    if int_row is not None:
        log(f"\nLocationType x Time Interaction:")
        log(f"  Estimate: {int_row['Estimate']:.4f}")
        log(f"  SE: {int_row['SE']:.4f}")
        log(f"  t = {int_row['t']:.2f}")
        log(f"  p_uncorrected = {int_row['p_uncorrected']:.4f}")
        log(f"  p_bonferroni = {int_row['p_bonferroni']:.4f}")

        if int_row['p_uncorrected'] < 0.05:
            log(f"  -> SIGNIFICANT: Calibration trajectories DIVERGE over time")
        else:
            log(f"  -> NOT SIGNIFICANT: Calibration trajectories PARALLEL")

    # Model fit indices
    log(f"\nModel Fit Indices:")
    log(f"  AIC: {result.aic:.2f}")
    log(f"  BIC: {result.bic:.2f}")
    log(f"  Log-Likelihood: {result.llf:.2f}")

    # Variance components
    log(f"\nVariance Components:")
    log(f"  Participant intercept variance: {result.cov_re.iloc[0, 0]:.4f}")
    log(f"  Residual variance: {result.scale:.4f}")

    # Compute effect sizes (Cohen's f² approximation)
    # f² = R² / (1 - R²), where R² approximated from t² / (t² + df)
    df_residual = len(df) - n_fe
    effect_sizes = []
    for _, row in fe_df.iterrows():
        if row['Effect'] != 'Intercept':
            t_sq = row['t'] ** 2
            partial_r2 = t_sq / (t_sq + df_residual)
            cohens_f2 = partial_r2 / (1 - partial_r2) if partial_r2 < 1 else np.nan
            effect_sizes.append({
                'Effect': row['Effect'],
                'partial_R2': partial_r2,
                'Cohens_f2': cohens_f2
            })

    es_df = pd.DataFrame(effect_sizes)
    es_path = RQ_DIR / "data" / "step02_effect_sizes.csv"
    es_df.to_csv(es_path, index=False)
    log(f"\nSaved: {es_path}")

    log("\nEffect Sizes (Cohen's f²):")
    log(f"  Small: f² = 0.02, Medium: f² = 0.15, Large: f² = 0.35")
    for _, row in es_df.iterrows():
        size_label = "negligible" if row['Cohens_f2'] < 0.02 else \
                     "small" if row['Cohens_f2'] < 0.15 else \
                     "medium" if row['Cohens_f2'] < 0.35 else "large"
        log(f"  {row['Effect']}: f² = {row['Cohens_f2']:.4f} ({size_label})")

    log(f"\nLMM fitting complete")
    return result, fe_df

# Prepare Calibration Plot Data

def step03_prepare_plot_data(df_calibration):
    """
    Aggregate calibration by LocationType x TEST for trajectory visualization.
    """
    log("\n" + "="*70)
    log("STEP 03: Prepare Calibration Plot Data")
    log("="*70)

    df = df_calibration.copy()

    # Get mean TSVR for each TEST
    tsvr_by_test = df.groupby('TEST')['TSVR_hours'].mean().to_dict()
    log(f"\nMean TSVR by test: {tsvr_by_test}")

    # Aggregate by LocationType x TEST
    plot_data = df.groupby(['LocationType', 'TEST']).agg({
        'calibration': ['mean', 'std', 'count'],
        'TSVR_hours': 'mean'
    }).reset_index()

    # Flatten column names
    plot_data.columns = ['LocationType', 'TEST', 'mean_calibration', 'std_calibration', 'n', 'TSVR_hours']

    # Compute 95% CI
    plot_data['SE'] = plot_data['std_calibration'] / np.sqrt(plot_data['n'])
    plot_data['CI_lower'] = plot_data['mean_calibration'] - 1.96 * plot_data['SE']
    plot_data['CI_upper'] = plot_data['mean_calibration'] + 1.96 * plot_data['SE']

    # Sort by LocationType then TSVR
    plot_data = plot_data.sort_values(['LocationType', 'TSVR_hours'])

    log("\nPlot Data:")
    log(plot_data[['LocationType', 'TEST', 'TSVR_hours', 'mean_calibration', 'CI_lower', 'CI_upper', 'n']].to_string(index=False))

    # Save plot data
    output_path = RQ_DIR / "data" / "step03_calibration_plot_data.csv"
    plot_data[['LocationType', 'TEST', 'TSVR_hours', 'mean_calibration', 'CI_lower', 'CI_upper', 'n']].to_csv(output_path, index=False)
    log(f"\nSaved: {output_path}")
    log(f"  Rows: {len(plot_data)} (expected: 8 = 2 LocationTypes x 4 tests)")

    # Interpretation
    log("\n" + "="*50)
    log("CALIBRATION PATTERN INTERPRETATION")
    log("="*50)

    for loc in ['Source', 'Destination']:
        subset = plot_data[plot_data['LocationType'] == loc]
        t1_cal = subset[subset['TEST'] == 'T1']['mean_calibration'].values[0]
        t4_cal = subset[subset['TEST'] == 'T4']['mean_calibration'].values[0]
        change = t4_cal - t1_cal

        log(f"\n{loc}:")
        log(f"  T1 calibration: {t1_cal:.3f}")
        log(f"  T4 calibration: {t4_cal:.3f}")
        log(f"  Change (T4 - T1): {change:.3f}")

        if t1_cal > 0.1:
            log(f"  -> Initially OVERCONFIDENT")
        elif t1_cal < -0.1:
            log(f"  -> Initially UNDERCONFIDENT")
        else:
            log(f"  -> Initially WELL-CALIBRATED")

        if change > 0.1:
            log(f"  -> Overconfidence INCREASES over time")
        elif change < -0.1:
            log(f"  -> Overconfidence DECREASES over time (calibration improves)")
        else:
            log(f"  -> Calibration STABLE over time")

    log(f"\nPlot data preparation complete: {len(plot_data)} rows")
    return plot_data

# MAIN EXECUTION

def main():
    """Execute all steps."""
    log("\n" + "="*70)
    log("RQ 6.8.2: Source-Destination Calibration")
    log("="*70)
    log(f"RQ_DIR: {RQ_DIR}")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")

    # Step 00: Merge accuracy and confidence
    df_merged = step00_merge_accuracy_confidence()

    # Step 01: Compute calibration
    df_calibration = step01_compute_calibration(df_merged)

    # Step 02: Fit LMM
    lmm_result, fe_df = step02_fit_lmm_calibration(df_calibration)

    # Step 03: Prepare plot data
    plot_data = step03_prepare_plot_data(df_calibration)

    log("\n" + "="*70)
    log("ALL STEPS COMPLETE")
    log("="*70)

    # Final summary
    log("\nFiles Created:")
    log(f"  data/step00_accuracy_confidence_merged.csv")
    log(f"  data/step01_calibration_by_location.csv")
    log(f"  data/step02_lmm_calibration_summary.txt")
    log(f"  data/step02_location_effects.csv")
    log(f"  data/step02_effect_sizes.csv")
    log(f"  data/step03_calibration_plot_data.csv")

if __name__ == "__main__":
    main()
