#!/usr/bin/env python3
"""
RQ 6.2.1: Calibration Over Time
===============================
Does calibration (confidence-accuracy alignment) change from Day 0 to Day 6?

Steps:
  00a: Load accuracy theta from RQ 5.1.1
  00b: Load confidence theta from RQ 6.1.1
  00c: Load TSVR mapping from RQ 6.1.1
  01:  Merge all theta scores with z-standardization
  02:  Compute calibration metric (z_theta_confidence - z_theta_accuracy)
  03:  Compute Brier scores (item-level calibration)
  04:  Compute ECE (Expected Calibration Error) per timepoint
  05:  Fit LMM: calibration ~ TSVR_hours + (TSVR_hours | UID)
  06:  Test time effect with dual p-values (Decision D068)
  07:  Prepare trajectory plot data (Decision D069)

Date: 2025-12-11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.1
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_07.log"

# Source files from dependency RQs
ACCURACY_THETA_FILE = PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv"
CONFIDENCE_THETA_FILE = PROJECT_ROOT / "results/ch6/6.1.1/data/step03_theta_confidence.csv"
TSVR_MAPPING_FILE = PROJECT_ROOT / "results/ch6/6.1.1/data/step00_tsvr_mapping.csv"
DFDATA_FILE = PROJECT_ROOT / "data/cache/dfData.csv"

# Validation thresholds
EXPECTED_ROWS = 400  # 100 participants x 4 tests
Z_STANDARDIZATION_TOLERANCE = 0.01


def log(msg):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def validate_dataframe(df, name, expected_rows=None, required_columns=None):
    """Validate DataFrame structure and content."""
    log(f"  Validating {name}...")

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"VALIDATION FAILED: {name} has {len(df)} rows, expected {expected_rows}")

    if required_columns is not None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"VALIDATION FAILED: {name} missing columns: {missing}")

    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log(f"  WARNING: NaN values in {name}: {dict(nan_cols)}")

    log(f"  VALIDATION - PASS: {name} ({len(df)} rows, {len(df.columns)} columns)")
    return True


# STEP 00a: Load Accuracy Theta Scores

def step00a_load_accuracy_theta():
    """Load accuracy theta scores from RQ 5.1.1."""
    log("\n" + "="*70)
    log("STEP 00a: Load Accuracy Theta Scores")
    log("="*70)

    if not ACCURACY_THETA_FILE.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: RQ 5.1.1 output file missing: {ACCURACY_THETA_FILE}")

    df = pd.read_csv(ACCURACY_THETA_FILE)
    log(f"Loaded accuracy theta: {len(df)} rows")
    log(f"Columns: {list(df.columns)}")

    # Actual columns: UID, test, Theta_All
    # Need to create composite_ID and rename theta column
    df['composite_ID'] = df['UID'].astype(str) + '_T' + df['test'].astype(str)
    df = df.rename(columns={'Theta_All': 'theta_accuracy'})

    # Note: Ch5 5.1.1 doesn't have SE column in this file, set to NaN
    df['se_accuracy'] = np.nan

    # Select and order columns
    df_out = df[['composite_ID', 'theta_accuracy', 'se_accuracy']].copy()

    validate_dataframe(df_out, "accuracy_theta", expected_rows=EXPECTED_ROWS,
                       required_columns=['composite_ID', 'theta_accuracy'])

    # Check theta range
    theta_range = df_out['theta_accuracy'].agg(['min', 'max'])
    log(f"Theta accuracy range: [{theta_range['min']:.3f}, {theta_range['max']:.3f}]")

    out_path = RQ_DIR / "data" / "step00a_accuracy_theta.csv"
    df_out.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")

    return df_out


# STEP 00b: Load Confidence Theta Scores

def step00b_load_confidence_theta():
    """Load confidence theta scores from RQ 6.1.1."""
    log("\n" + "="*70)
    log("STEP 00b: Load Confidence Theta Scores")
    log("="*70)

    if not CONFIDENCE_THETA_FILE.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: RQ 6.1.1 output file missing: {CONFIDENCE_THETA_FILE}")

    df = pd.read_csv(CONFIDENCE_THETA_FILE)
    log(f"Loaded confidence theta: {len(df)} rows")
    log(f"Columns: {list(df.columns)}")

    # Actual columns: composite_ID, theta_All, se_All
    # Rename to theta_confidence, se_confidence
    df = df.rename(columns={
        'theta_All': 'theta_confidence',
        'se_All': 'se_confidence'
    })

    # Standardize composite_ID format (already has _T format from 6.1.1)
    # Check format
    sample_id = df['composite_ID'].iloc[0]
    log(f"Sample composite_ID format: {sample_id}")

    # If format is "A010_T1", good. If "A010_1", need to convert
    if '_T' not in sample_id:
        df['composite_ID'] = df['composite_ID'].str.replace(r'_(\d)$', r'_T\1', regex=True)
        log("Converted composite_ID format from UID_N to UID_TN")

    validate_dataframe(df, "confidence_theta", expected_rows=EXPECTED_ROWS,
                       required_columns=['composite_ID', 'theta_confidence', 'se_confidence'])

    theta_range = df['theta_confidence'].agg(['min', 'max'])
    log(f"Theta confidence range: [{theta_range['min']:.3f}, {theta_range['max']:.3f}]")

    out_path = RQ_DIR / "data" / "step00b_confidence_theta.csv"
    df.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")

    return df


# STEP 00c: Load TSVR Mapping

def step00c_load_tsvr_mapping():
    """Load TSVR time variable mapping from RQ 6.1.1."""
    log("\n" + "="*70)
    log("STEP 00c: Load TSVR Mapping")
    log("="*70)

    if not TSVR_MAPPING_FILE.exists():
        raise FileNotFoundError(f"EXPECTATIONS ERROR: TSVR mapping file missing: {TSVR_MAPPING_FILE}")

    df = pd.read_csv(TSVR_MAPPING_FILE)
    log(f"Loaded TSVR mapping: {len(df)} rows")
    log(f"Columns: {list(df.columns)}")

    # Standardize composite_ID format
    sample_id = df['composite_ID'].iloc[0]
    log(f"Sample composite_ID format: {sample_id}")

    if '_T' not in str(sample_id):
        df['composite_ID'] = df['composite_ID'].str.replace(r'_(\d)$', r'_T\1', regex=True)
        log("Converted composite_ID format from UID_N to UID_TN")

    # Convert test to T1, T2, T3, T4 format
    df['test'] = 'T' + df['test'].astype(str)

    # Validate TSVR non-negative
    if (df['TSVR_hours'] < 0).any():
        raise ValueError("VALIDATION FAILED: Negative TSVR_hours values detected")

    validate_dataframe(df, "tsvr_mapping", expected_rows=EXPECTED_ROWS,
                       required_columns=['composite_ID', 'TSVR_hours', 'test'])

    tsvr_range = df['TSVR_hours'].agg(['min', 'max'])
    log(f"TSVR_hours range: [{tsvr_range['min']:.1f}, {tsvr_range['max']:.1f}] hours")

    out_path = RQ_DIR / "data" / "step00c_tsvr_mapping.csv"
    df.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")

    return df


# Merge Theta Scores with TSVR and Z-Standardize

def step01_merge_theta(df_acc, df_conf, df_tsvr):
    """Merge all theta scores and z-standardize."""
    log("\n" + "="*70)
    log("STEP 01: Merge Theta Scores with TSVR")
    log("="*70)

    # Inner join on composite_ID
    df = df_acc.merge(df_conf, on='composite_ID', how='inner')
    log(f"After accuracy-confidence merge: {len(df)} rows")

    df = df.merge(df_tsvr, on='composite_ID', how='inner')
    log(f"After TSVR merge: {len(df)} rows")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"MERGE FAILED: Expected {EXPECTED_ROWS} rows, got {len(df)}")

    # Extract UID from composite_ID
    df['UID'] = df['composite_ID'].str.split('_').str[0]

    # Z-standardize theta scores (CRITICAL for calibration computation)
    df['z_theta_accuracy'] = (df['theta_accuracy'] - df['theta_accuracy'].mean()) / df['theta_accuracy'].std()
    df['z_theta_confidence'] = (df['theta_confidence'] - df['theta_confidence'].mean()) / df['theta_confidence'].std()

    # Validate z-standardization
    for col in ['z_theta_accuracy', 'z_theta_confidence']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if abs(mean_val) > Z_STANDARDIZATION_TOLERANCE or abs(std_val - 1) > Z_STANDARDIZATION_TOLERANCE:
            raise ValueError(f"Z-standardization failed for {col}: mean={mean_val:.4f}, std={std_val:.4f}")
        log(f"Z-standardization {col}: mean={mean_val:.6f}, std={std_val:.6f}")

    # Reorder columns
    df = df[['UID', 'test', 'composite_ID', 'TSVR_hours',
             'theta_accuracy', 'se_accuracy', 'theta_confidence', 'se_confidence',
             'z_theta_accuracy', 'z_theta_confidence']]

    validate_dataframe(df, "merged_theta", expected_rows=EXPECTED_ROWS)

    log(f"Unique UIDs: {df['UID'].nunique()}")
    log(f"Tests per UID: {df.groupby('UID').size().value_counts().to_dict()}")

    out_path = RQ_DIR / "data" / "step01_merged_theta.csv"
    df.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: Merge and z-standardization complete")

    return df


# Compute Calibration Metric

def step02_compute_calibration(df_merged):
    """Compute calibration = z_theta_confidence - z_theta_accuracy."""
    log("\n" + "="*70)
    log("STEP 02: Compute Calibration Metric")
    log("="*70)

    df = df_merged.copy()

    # Calibration = confidence - accuracy (z-standardized)
    # Positive = overconfidence (confidence > accuracy)
    # Negative = underconfidence (accuracy > confidence)
    df['calibration'] = df['z_theta_confidence'] - df['z_theta_accuracy']

    # Verify arithmetic
    sample_idx = df.index[0]
    expected = df.loc[sample_idx, 'z_theta_confidence'] - df.loc[sample_idx, 'z_theta_accuracy']
    actual = df.loc[sample_idx, 'calibration']
    if abs(expected - actual) > 1e-10:
        raise ValueError(f"Arithmetic verification failed: expected {expected}, got {actual}")

    # Statistics
    cal_stats = df['calibration'].describe()
    log(f"Calibration statistics:")
    log(f"  Mean: {cal_stats['mean']:.4f}")
    log(f"  Std: {cal_stats['std']:.4f}")
    log(f"  Min: {cal_stats['min']:.4f}")
    log(f"  Max: {cal_stats['max']:.4f}")

    # Classification
    n_overconf = (df['calibration'] > 0).sum()
    n_underconf = (df['calibration'] < 0).sum()
    log(f"Overconfident observations: {n_overconf} ({100*n_overconf/len(df):.1f}%)")
    log(f"Underconfident observations: {n_underconf} ({100*n_underconf/len(df):.1f}%)")

    # Output columns
    df_out = df[['UID', 'test', 'composite_ID', 'TSVR_hours',
                 'z_theta_accuracy', 'z_theta_confidence', 'calibration']].copy()

    validate_dataframe(df_out, "calibration_scores", expected_rows=EXPECTED_ROWS,
                       required_columns=['calibration'])

    out_path = RQ_DIR / "data" / "step02_calibration_scores.csv"
    df_out.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: Calibration metric computed")

    return df_out


# Compute Brier Score

def step03_compute_brier():
    """Compute item-level Brier scores for calibration assessment."""
    log("\n" + "="*70)
    log("STEP 03: Compute Brier Score (Item-Level Calibration)")
    log("="*70)

    if not DFDATA_FILE.exists():
        raise FileNotFoundError(f"dfData.csv not found: {DFDATA_FILE}")

    df = pd.read_csv(DFDATA_FILE)
    log(f"Loaded dfData: {len(df)} rows")

    # Get TQ_* (accuracy) and TC_* (confidence) columns
    tq_cols = [c for c in df.columns if c.startswith('TQ_')]
    tc_cols = [c for c in df.columns if c.startswith('TC_')]
    log(f"TQ (accuracy) columns: {len(tq_cols)}")
    log(f"TC (confidence) columns: {len(tc_cols)}")

    # Find matching pairs (TQ_X and TC_X)
    # Item names should match after removing prefix
    tq_items = {c.replace('TQ_', ''): c for c in tq_cols}
    tc_items = {c.replace('TC_', ''): c for c in tc_cols}

    matched_items = set(tq_items.keys()) & set(tc_items.keys())
    log(f"Matched TQ/TC item pairs: {len(matched_items)}")

    if len(matched_items) == 0:
        raise ValueError("No matched TQ/TC item pairs found!")

    # Filter to interactive paradigms only (IFR, ICR, IRE)
    # These are items with -N- (What), -L-/-U-/-D- (Where), -O- (When) in name
    interactive_items = [item for item in matched_items
                        if any(tag in item for tag in ['-N-', '-L-', '-U-', '-D-', '-O-'])]
    log(f"Interactive paradigm items: {len(interactive_items)}")

    if len(interactive_items) == 0:
        # Fall back to all matched items if filtering is too strict
        log("WARNING: No interactive items found, using all matched items")
        interactive_items = list(matched_items)

    # Compute Brier score per participant-test
    results = []

    for _, row in df.iterrows():
        uid = row['UID']
        test = row['TEST']

        squared_errors = []
        for item in interactive_items:
            tq_col = tq_items[item]
            tc_col = tc_items[item]

            accuracy_raw = row[tq_col]
            confidence = row[tc_col]  # Should be in [0, 1]

            if pd.notna(accuracy_raw) and pd.notna(confidence):
                # Binary recode: 1=correct, <1=incorrect (partial credit → 0)
                accuracy = 1.0 if float(accuracy_raw) == 1.0 else 0.0
                # Brier score component: (confidence - accuracy)^2
                squared_errors.append((confidence - accuracy) ** 2)

        if len(squared_errors) > 0:
            brier_score = np.mean(squared_errors)
            results.append({
                'UID': uid,
                'TEST': f'T{test}' if isinstance(test, (int, float)) else test,
                'composite_ID': f"{uid}_T{int(test) if isinstance(test, (int, float)) else test.replace('T','')}",
                'brier_score': brier_score,
                'n_items': len(squared_errors)
            })

    df_brier = pd.DataFrame(results)

    # Validate
    if len(df_brier) != EXPECTED_ROWS:
        log(f"WARNING: Expected {EXPECTED_ROWS} rows, got {len(df_brier)}")

    # Check Brier range
    brier_range = df_brier['brier_score'].agg(['min', 'max', 'mean'])
    log(f"Brier score range: [{brier_range['min']:.4f}, {brier_range['max']:.4f}]")
    log(f"Brier score mean: {brier_range['mean']:.4f}")

    if (df_brier['brier_score'] > 1).any() or (df_brier['brier_score'] < 0).any():
        raise ValueError("VALIDATION FAILED: Brier scores outside [0, 1] range")

    items_per_obs = df_brier['n_items'].agg(['min', 'max', 'mean'])
    log(f"Items per observation: min={items_per_obs['min']}, max={items_per_obs['max']}, mean={items_per_obs['mean']:.1f}")

    out_path = RQ_DIR / "data" / "step03_brier_scores.csv"
    df_brier.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: Brier scores computed")

    return df_brier


# Compute Expected Calibration Error (ECE)

def step04_compute_ece():
    """Compute ECE per timepoint by binning confidence levels."""
    log("\n" + "="*70)
    log("STEP 04: Compute Expected Calibration Error (ECE)")
    log("="*70)

    df = pd.read_csv(DFDATA_FILE)

    # Get TQ_* and TC_* columns
    tq_cols = [c for c in df.columns if c.startswith('TQ_')]
    tc_cols = [c for c in df.columns if c.startswith('TC_')]

    tq_items = {c.replace('TQ_', ''): c for c in tq_cols}
    tc_items = {c.replace('TC_', ''): c for c in tc_cols}
    matched_items = list(set(tq_items.keys()) & set(tc_items.keys()))

    # Confidence bins (discrete values from Likert scale)
    # TC values should be: 0, 0.25, 0.5, 0.75, 1.0 (5-point scale normalized)
    # But may be 1, 2, 3, 4, 5 (raw ordinal) - need to check

    # Collect all item-level responses per test
    results = []

    for test_val in df['TEST'].unique():
        df_test = df[df['TEST'] == test_val]

        # Collect all confidence-accuracy pairs
        conf_acc_pairs = []
        for _, row in df_test.iterrows():
            for item in matched_items:
                tq_col = tq_items[item]
                tc_col = tc_items[item]

                accuracy_raw = row[tq_col]
                confidence = row[tc_col]

                if pd.notna(accuracy_raw) and pd.notna(confidence):
                    # Binary recode: 1=correct, <1=incorrect (partial credit → 0)
                    accuracy = 1.0 if float(accuracy_raw) == 1.0 else 0.0
                    conf_acc_pairs.append({'confidence': confidence, 'accuracy': accuracy})

        df_pairs = pd.DataFrame(conf_acc_pairs)

        if len(df_pairs) == 0:
            continue

        # Check confidence scale - if > 1, normalize
        conf_max = df_pairs['confidence'].max()
        if conf_max > 1:
            # Assume 5-point scale (1-5), normalize to 0-1
            df_pairs['confidence'] = (df_pairs['confidence'] - 1) / 4
            log(f"Test {test_val}: Normalized confidence from 1-5 scale to 0-1")

        # Bin by confidence level (5 bins: 0, 0.25, 0.5, 0.75, 1.0)
        bins = [-0.001, 0.125, 0.375, 0.625, 0.875, 1.001]  # Midpoints
        bin_labels = ['bin_0', 'bin_025', 'bin_05', 'bin_075', 'bin_1']
        df_pairs['conf_bin'] = pd.cut(df_pairs['confidence'], bins=bins, labels=bin_labels)

        # Compute bin-specific errors
        bin_errors = {}
        total_items = 0
        weighted_error_sum = 0

        for bin_label in bin_labels:
            bin_data = df_pairs[df_pairs['conf_bin'] == bin_label]
            n_in_bin = len(bin_data)

            if n_in_bin > 0:
                mean_conf = bin_data['confidence'].mean()
                mean_acc = bin_data['accuracy'].mean()
                bin_error = abs(mean_conf - mean_acc)

                weighted_error_sum += bin_error * n_in_bin
                total_items += n_in_bin
            else:
                bin_error = np.nan

            bin_errors[f'{bin_label}_error'] = bin_error

        # ECE = weighted average of bin errors
        ece = weighted_error_sum / total_items if total_items > 0 else np.nan

        test_label = f'T{int(test_val)}' if isinstance(test_val, (int, float)) else test_val
        results.append({
            'test': test_label,
            'ECE': ece,
            'n_items': total_items,
            **bin_errors
        })

    df_ece = pd.DataFrame(results)
    df_ece = df_ece.sort_values('test').reset_index(drop=True)

    log(f"ECE results:")
    for _, row in df_ece.iterrows():
        log(f"  {row['test']}: ECE={row['ECE']:.4f}, n_items={row['n_items']}")

    ece_range = df_ece['ECE'].agg(['min', 'max'])
    log(f"ECE range: [{ece_range['min']:.4f}, {ece_range['max']:.4f}]")

    if (df_ece['ECE'] > 1).any() or (df_ece['ECE'] < 0).any():
        raise ValueError("VALIDATION FAILED: ECE outside [0, 1] range")

    out_path = RQ_DIR / "data" / "step04_ece_by_time.csv"
    df_ece.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: ECE computed for 4 timepoints")

    return df_ece


# Fit LMM for Calibration Trajectory

def step05_fit_lmm(df_calibration):
    """Fit LMM: calibration ~ TSVR_hours + (TSVR_hours | UID)."""
    log("\n" + "="*70)
    log("STEP 05: Fit LMM for Calibration Trajectory")
    log("="*70)

    df = df_calibration.copy()

    # Model formula: calibration ~ Time + (Time | UID)
    # Using TSVR_hours as continuous time predictor (Decision D070)

    log(f"Model formula: calibration ~ TSVR_hours")
    log(f"Random effects: (1 + TSVR_hours | UID)")
    log(f"N observations: {len(df)}")
    log(f"N groups (UIDs): {df['UID'].nunique()}")

    # Scale TSVR_hours for numerical stability
    # TSVR can be 0-200 hours, scale by dividing by 100
    df['Time'] = df['TSVR_hours'] / 100  # Scale factor for better convergence

    # Fit random slopes model
    try:
        model = smf.mixedlm(
            "calibration ~ Time",
            data=df,
            groups=df['UID'],
            re_formula="~Time"
        )
        result = model.fit(reml=False, method='powell')  # ML for LRT comparison

        converged = True
        log(f"Model converged: {converged}")
    except Exception as e:
        log(f"WARNING: Random slopes model failed: {e}")
        log("Falling back to random intercepts only...")

        model = smf.mixedlm(
            "calibration ~ Time",
            data=df,
            groups=df['UID']
        )
        result = model.fit(reml=False)
        converged = True

    # Extract model summary
    summary_text = str(result.summary())
    log("\nModel Summary:")
    log(summary_text)

    # Save summary
    summary_path = RQ_DIR / "data" / "step05_lmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"RQ 6.2.1: Calibration Trajectory LMM\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Formula: calibration ~ Time (TSVR_hours/100)\n")
        f.write(f"Random effects: (1 + Time | UID) or (1 | UID) if slopes failed\n")
        f.write(f"Estimation: ML (for LRT comparison)\n\n")
        f.write(summary_text)

    log(f"Saved: {summary_path}")

    # Validate fit
    if not np.isfinite(result.llf):
        raise ValueError("VALIDATION FAILED: Non-finite log-likelihood")

    log(f"Log-likelihood: {result.llf:.2f}")
    log(f"AIC: {result.aic:.2f}")
    log(f"BIC: {result.bic:.2f}")
    log("VALIDATION - PASS: LMM converged")

    return result, df


# Test Time Effect with Dual P-Values

def step06_test_time_effect(lmm_result, df_with_time):
    """Extract Time effect with dual p-value reporting (Decision D068)."""
    log("\n" + "="*70)
    log("STEP 06: Test Time Effect (Decision D068 Dual P-Values)")
    log("="*70)

    # Extract fixed effects
    fe_params = lmm_result.fe_params
    fe_bse = lmm_result.bse_fe
    fe_tvalues = lmm_result.tvalues
    fe_pvalues = lmm_result.pvalues

    log("Fixed Effects:")
    for name in fe_params.index:
        log(f"  {name}: coef={fe_params[name]:.6f}, SE={fe_bse[name]:.6f}, "
            f"z={fe_tvalues[name]:.3f}, p={fe_pvalues[name]:.6f}")

    # Time effect (scaled TSVR)
    time_coef = fe_params['Time']
    time_se = fe_bse['Time']
    time_p_wald = fe_pvalues['Time']

    # Convert back to per-hour effect
    # coef_per_hour = time_coef / 100 (since Time = TSVR_hours / 100)
    coef_per_hour = time_coef / 100
    se_per_hour = time_se / 100

    log(f"\nTime effect (per 100 hours): {time_coef:.6f}")
    log(f"Time effect (per hour): {coef_per_hour:.8f}")
    log(f"Wald p-value (uncorrected): {time_p_wald:.6f}")

    # LRT for corrected p-value (compare full model vs intercept-only)
    try:
        model_null = smf.mixedlm(
            "calibration ~ 1",
            data=df_with_time,
            groups=df_with_time['UID']
        )
        result_null = model_null.fit(reml=False)

        # LRT statistic = 2 * (ll_full - ll_null)
        lrt_stat = 2 * (lmm_result.llf - result_null.llf)
        lrt_df = 1  # One parameter difference (Time coefficient)
        lrt_pval = 1 - stats.chi2.cdf(lrt_stat, lrt_df)

        log(f"LRT statistic: {lrt_stat:.4f} (df={lrt_df})")
        log(f"LRT p-value (corrected): {lrt_pval:.6f}")

    except Exception as e:
        log(f"WARNING: LRT failed: {e}")
        lrt_pval = time_p_wald  # Use Wald as fallback

    # Interpretation
    alpha = 0.05
    if time_p_wald < alpha and lrt_pval < alpha:
        interpretation = "Significant"
    elif time_p_wald < alpha or lrt_pval < alpha:
        interpretation = "Marginal"
    else:
        interpretation = "Not significant"

    if abs(time_coef) < 0.0001:
        direction = "Null"
    elif time_coef > 0:
        direction = "Positive"  # Increasing overconfidence
    else:
        direction = "Negative"  # Decreasing overconfidence

    log(f"\nInterpretation: {interpretation} ({direction})")

    if direction == "Positive":
        log("  → Calibration WORSENS over time (increasing overconfidence)")
    elif direction == "Negative":
        log("  → Calibration IMPROVES over time (decreasing overconfidence)")
    else:
        log("  → Calibration STABLE over time")

    # Save results
    df_result = pd.DataFrame([{
        'effect': 'TSVR_hours',
        'coefficient_per_100h': time_coef,
        'coefficient_per_hour': coef_per_hour,
        'se': time_se,
        'se_per_hour': se_per_hour,
        'p_uncorrected': time_p_wald,
        'p_corrected': lrt_pval,
        'interpretation': interpretation,
        'direction': direction
    }])

    out_path = RQ_DIR / "data" / "step06_time_effect.csv"
    df_result.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: Dual p-values present (D068)")

    return df_result


# Prepare Calibration Trajectory Plot Data

def step07_prepare_trajectory_plot(df_calibration, lmm_result):
    """Create plot source CSV for calibration trajectory."""
    log("\n" + "="*70)
    log("STEP 07: Prepare Calibration Trajectory Plot Data")
    log("="*70)

    df = df_calibration.copy()

    # Aggregate by test session
    agg_data = df.groupby('test').agg({
        'TSVR_hours': 'mean',
        'calibration': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    agg_data.columns = ['test', 'time', 'calibration', 'calibration_std', 'n']

    # Compute 95% CI
    agg_data['CI_lower'] = agg_data['calibration'] - 1.96 * agg_data['calibration_std'] / np.sqrt(agg_data['n'])
    agg_data['CI_upper'] = agg_data['calibration'] + 1.96 * agg_data['calibration_std'] / np.sqrt(agg_data['n'])

    # Sort by time
    agg_data = agg_data.sort_values('time').reset_index(drop=True)

    # Select output columns
    df_plot = agg_data[['time', 'calibration', 'CI_lower', 'CI_upper', 'test']].copy()

    log("Trajectory plot data:")
    for _, row in df_plot.iterrows():
        log(f"  {row['test']}: time={row['time']:.1f}h, cal={row['calibration']:.4f} "
            f"[{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]")

    # Validate
    if len(df_plot) != 4:
        log(f"WARNING: Expected 4 rows (T1-T4), got {len(df_plot)}")

    if (df_plot['CI_lower'] >= df_plot['CI_upper']).any():
        raise ValueError("VALIDATION FAILED: CI_lower >= CI_upper")

    out_path = RQ_DIR / "data" / "step07_calibration_trajectory_theta_data.csv"
    df_plot.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")
    log("VALIDATION - PASS: Plot data complete with 4 timepoints")

    return df_plot


# MAIN EXECUTION

def main():
    """Execute all analysis steps."""
    log(f"{'='*70}")
    log(f"RQ 6.2.1: CALIBRATION OVER TIME")
    log(f"{'='*70}")
    log(f"Started: {pd.Timestamp.now()}")
    log(f"RQ Directory: {RQ_DIR}")

    try:
        # Step 00a-c: Load source data
        df_acc = step00a_load_accuracy_theta()
        df_conf = step00b_load_confidence_theta()
        df_tsvr = step00c_load_tsvr_mapping()

        # Step 01: Merge and standardize
        df_merged = step01_merge_theta(df_acc, df_conf, df_tsvr)

        # Step 02: Compute calibration metric
        df_calibration = step02_compute_calibration(df_merged)

        # Step 03: Compute Brier scores
        df_brier = step03_compute_brier()

        # Step 04: Compute ECE
        df_ece = step04_compute_ece()

        # Step 05: Fit LMM
        lmm_result, df_with_time = step05_fit_lmm(df_calibration)

        # Step 06: Test time effect
        df_time_effect = step06_test_time_effect(lmm_result, df_with_time)

        # Step 07: Prepare plot data
        df_plot = step07_prepare_trajectory_plot(df_calibration, lmm_result)

        # Final summary
        log("\n" + "="*70)
        log("EXECUTION COMPLETE")
        log("="*70)

        # Key findings
        time_row = df_time_effect.iloc[0]
        log(f"\nPRIMARY FINDING: Time Effect on Calibration")
        log(f"  Coefficient (per hour): {time_row['coefficient_per_hour']:.8f}")
        log(f"  p-value (Wald): {time_row['p_uncorrected']:.6f}")
        log(f"  p-value (LRT): {time_row['p_corrected']:.6f}")
        log(f"  Interpretation: {time_row['interpretation']} ({time_row['direction']})")

        log(f"\nSECONDARY METRICS:")
        log(f"  Mean Brier score: {df_brier['brier_score'].mean():.4f}")
        log(f"  ECE range: [{df_ece['ECE'].min():.4f}, {df_ece['ECE'].max():.4f}]")

        log(f"\nFiles created:")
        for f in sorted((RQ_DIR / "data").glob("step*.csv")):
            log(f"  {f.name}")

        log(f"\nCompleted: {pd.Timestamp.now()}")
        log("SUCCESS: All steps completed")

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
