#!/usr/bin/env python3
"""
RQ 6.4.2: SEM-Based Paradigm-Stratified Calibration Computation
================================================================

REWRITTEN to use tools/sem_calibration.py (SEMCalibration class) for consistency
with RQ 6.2.1, 6.3.2, and 6.8.2. Previous version used direct semopy which
failed silently and fell back to raw z-difference (no error correction).

Approach: Compute SEM latent calibration SEPARATELY FOR EACH PARADIGM
using factor score regression with reliability-weighted shrinkage.

Input: step00_calibration_by_paradigm.csv (1200 rows: 100 UID x 4 tests x 3 paradigms)
Output: step11_calibration_scores_SEM.csv (1200 rows, with latent_calibration by paradigm)

Date: 2026-04-07
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add tools to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from sem_calibration import compute_difference_score_reliability, SEMCalibration

# SETUP

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.4.2
LOG_FILE = RQ_DIR / "logs" / "step11_SEM.log"
DATA_DIR = RQ_DIR / "data"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 1200  # 100 participants x 4 tests x 3 paradigms
EXPECTED_PARADIGMS = ['ICR', 'IFR', 'IRE']


def log(msg: str) -> None:
    """Log message to both file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


# Clear log file
LOG_FILE.write_text("")
log("=" * 70)
log("RQ 6.4.2: SEM-Based Paradigm-Stratified Calibration")
log("=" * 70)
log("Using tools/sem_calibration.py (SEMCalibration class)")
log("Same methodology as RQ 6.2.1, 6.3.2, 6.8.2")
log("=" * 70)

# Load Paradigm-Stratified Data

log("\n" + "=" * 70)
log("STEP 1: Load Paradigm-Stratified Calibration Data")
log("=" * 70)

input_file = DATA_DIR / "step00_calibration_by_paradigm.csv"
if not input_file.exists():
    log(f"ERROR: Input file not found: {input_file}")
    sys.exit(1)

df = pd.read_csv(input_file)
log(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
log(f"Columns: {list(df.columns)}")

# Validate structure
required_cols = ['UID', 'TEST', 'Paradigm', 'TSVR_hours',
                 'theta_accuracy', 'theta_confidence',
                 'theta_accuracy_z', 'theta_confidence_z',
                 'calibration']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    log(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)
log(f"All required columns present")

if len(df) != EXPECTED_ROWS:
    log(f"WARNING: Expected {EXPECTED_ROWS} rows, got {len(df)}")

paradigms_found = sorted(df['Paradigm'].unique())
log(f"Paradigms found: {paradigms_found}")

n_missing_acc = df['theta_accuracy'].isna().sum()
n_missing_conf = df['theta_confidence'].isna().sum()
if n_missing_acc > 0 or n_missing_conf > 0:
    log(f"ERROR: Missing theta values - accuracy: {n_missing_acc}, confidence: {n_missing_conf}")
    sys.exit(1)
log("No missing values in theta scores")
log("STEP 1 COMPLETE")

# Compute PRE-SEM Reliability BY PARADIGM

log("\n" + "=" * 70)
log("STEP 2: Compute PRE-SEM Difference Score Reliability (BY PARADIGM)")
log("=" * 70)

reliability_results = []

for paradigm in paradigms_found:
    log(f"\n{'=' * 70}")
    log(f"PARADIGM: {paradigm}")
    log(f"{'=' * 70}")

    df_par = df[df['Paradigm'] == paradigm].copy()
    log(f"N = {len(df_par)} observations ({df_par['UID'].nunique()} participants x {df_par['TEST'].nunique()} tests)")

    # Correlation between accuracy and confidence
    r_xy, p_value = stats.pearsonr(df_par['theta_accuracy_z'], df_par['theta_confidence_z'])
    log(f"Correlation (accuracy, confidence): r = {r_xy:.4f}, p = {p_value:.6f}")

    # ICC for accuracy
    person_means_acc = df_par.groupby('UID')['theta_accuracy_z'].mean()
    person_vars_acc = df_par.groupby('UID')['theta_accuracy_z'].var()
    between_var_acc = person_means_acc.var()
    within_var_acc = person_vars_acc.mean()
    r_xx_est = between_var_acc / (between_var_acc + within_var_acc)
    log(f"\nAccuracy ICC (r_xx): {r_xx_est:.4f}")

    # ICC for confidence
    person_means_conf = df_par.groupby('UID')['theta_confidence_z'].mean()
    person_vars_conf = df_par.groupby('UID')['theta_confidence_z'].var()
    between_var_conf = person_means_conf.var()
    within_var_conf = person_vars_conf.mean()
    r_yy_est = between_var_conf / (between_var_conf + within_var_conf)
    log(f"Confidence ICC (r_yy): {r_yy_est:.4f}")

    # Difference score reliability
    r_diff = compute_difference_score_reliability(
        r_xx=r_xx_est, r_yy=r_yy_est, r_xy=r_xy
    )

    if r_diff < 0:
        interpretation = "CATASTROPHIC (negative)"
    elif r_diff < 0.20:
        interpretation = "CRITICAL (< 0.20)"
    elif r_diff < 0.60:
        interpretation = "LOW (< 0.60)"
    elif r_diff < 0.70:
        interpretation = "MARGINAL (< 0.70)"
    else:
        interpretation = "ACCEPTABLE (>= 0.70)"

    log(f"Difference Score Reliability: r_diff = {r_diff:.4f} ({interpretation})")

    reliability_results.append({
        'Paradigm': paradigm,
        'r_xx': r_xx_est,
        'r_yy': r_yy_est,
        'r_xy': r_xy,
        'r_diff': r_diff,
        'interpretation': interpretation
    })

log("\nSUMMARY: PRE-SEM Reliability BY PARADIGM")
for res in reliability_results:
    log(f"  {res['Paradigm']:8s}: r_diff={res['r_diff']:+.4f} ({res['interpretation']})")

log("STEP 2 COMPLETE")

# Compute SEM-Based Latent Calibration (BY PARADIGM)

log("\n" + "=" * 70)
log("STEP 3: Compute SEM-Based Latent Calibration (BY PARADIGM)")
log("=" * 70)

df['latent_calibration'] = np.nan
sem_diagnostics = []

for paradigm in paradigms_found:
    log(f"\n{'=' * 70}")
    log(f"PARADIGM: {paradigm}")
    log(f"{'=' * 70}")

    df_par = df[df['Paradigm'] == paradigm].copy()

    # Get reliability estimates for this paradigm
    rel = [r for r in reliability_results if r['Paradigm'] == paradigm][0]
    r_xx_est = rel['r_xx']
    r_yy_est = rel['r_yy']
    r_xy = rel['r_xy']
    r_diff = rel['r_diff']

    # Prepare inputs (z-standardized theta)
    theta_accuracy = df_par['theta_accuracy_z'].values
    theta_confidence = df_par['theta_confidence_z'].values

    # SE from reliability
    se_accuracy = np.full(len(theta_accuracy), np.sqrt(max(1 - r_xx_est, 0.01)))
    se_confidence = np.full(len(theta_confidence), np.sqrt(max(1 - r_yy_est, 0.01)))

    log(f"SE_accuracy: {se_accuracy[0]:.4f} (from r_xx={r_xx_est:.4f})")
    log(f"SE_confidence: {se_confidence[0]:.4f} (from r_yy={r_yy_est:.4f})")

    # Prepare DataFrames for SEMCalibration
    df_acc_input = pd.DataFrame({
        'UID': df_par['UID'].values,
        'test': df_par['TEST'].values,
        'theta_accuracy': theta_accuracy,
        'se_accuracy': se_accuracy
    })

    df_conf_input = pd.DataFrame({
        'UID': df_par['UID'].values,
        'test': df_par['TEST'].values,
        'theta_confidence': theta_confidence,
        'se_confidence': se_confidence
    })

    try:
        log(f"\nInitializing SEMCalibration for {paradigm}...")
        sem = SEMCalibration(
            theta_accuracy=df_acc_input,
            theta_confidence=df_conf_input,
            measurement_error_acc=None,
            measurement_error_conf=None,
            reliability_acc=r_xx_est,
            reliability_conf=r_yy_est,
            id_vars=['UID', 'test']
        )

        log(f"Fitting latent difference model for {paradigm}...")
        fit_stats = sem.fit_latent_difference(method='ML', verbose=True)

        log(f"\nSEM calibration SUCCESSFUL for {paradigm}")

        # Get latent calibration scores
        latent_calibration = sem.get_latent_calibration(approach='difference')

        # Validate
        if len(latent_calibration) != len(df_par):
            log(f"ERROR: Length mismatch - expected {len(df_par)}, got {len(latent_calibration)}")
            sys.exit(1)

        # Assign to main dataframe
        for i, idx in enumerate(df_par.index):
            df.at[idx, 'latent_calibration'] = latent_calibration[i]

        n_assigned = df.loc[df['Paradigm'] == paradigm, 'latent_calibration'].notna().sum()
        log(f"Assigned {n_assigned}/{len(df_par)} latent_calibration values for {paradigm}")

        # POST-SEM reliability (split-half)
        df_par_sem = df_par.copy()
        df_par_sem['latent_calibration'] = latent_calibration
        df_par_sem['latent_cal_z'] = (latent_calibration - latent_calibration.mean()) / latent_calibration.std()

        df_odd = df_par_sem[df_par_sem['TEST'].isin(['T1', 'T3'])].groupby('UID')['latent_cal_z'].mean().reset_index()
        df_odd.columns = ['UID', 'mean_cal_odd']
        df_even = df_par_sem[df_par_sem['TEST'].isin(['T2', 'T4'])].groupby('UID')['latent_cal_z'].mean().reset_index()
        df_even.columns = ['UID', 'mean_cal_even']
        df_split = df_odd.merge(df_even, on='UID')

        odd_sd = df_split['mean_cal_odd'].std()
        even_sd = df_split['mean_cal_even'].std()

        if odd_sd < 0.001 or even_sd < 0.001 or np.isnan(odd_sd) or np.isnan(even_sd):
            log(f"  WARNING: Zero/NaN variance in split-half - using ICC method")
            person_means_lat = df_par_sem.groupby('UID')['latent_cal_z'].mean()
            person_vars_lat = df_par_sem.groupby('UID')['latent_cal_z'].var()
            between_var_lat = person_means_lat.var()
            within_var_lat = person_vars_lat.mean()
            r_full_length = between_var_lat / (between_var_lat + within_var_lat) if within_var_lat > 0 else np.nan
            r_split_half = np.nan
        else:
            try:
                r_split_half, _ = stats.pearsonr(df_split['mean_cal_odd'], df_split['mean_cal_even'])
                if np.isnan(r_split_half):
                    raise ValueError("NaN")
                r_full_length = (2 * r_split_half) / (1 + r_split_half)
            except Exception:
                person_means_lat = df_par_sem.groupby('UID')['latent_cal_z'].mean()
                person_vars_lat = df_par_sem.groupby('UID')['latent_cal_z'].var()
                between_var_lat = person_means_lat.var()
                within_var_lat = person_vars_lat.mean()
                r_full_length = between_var_lat / (between_var_lat + within_var_lat) if within_var_lat > 0 else np.nan
                r_split_half = np.nan

        log(f"\nPOST-SEM Reliability for {paradigm}:")
        log(f"  Split-half r: {r_split_half}")
        log(f"  Full-length r: {r_full_length:.4f}")

        reliability_gain = r_full_length - r_diff
        log(f"  Improvement: {r_diff:.4f} -> {r_full_length:.4f} ({reliability_gain:+.4f})")

        # Correlation with simple difference
        simple_cal = df_par['theta_confidence_z'] - df_par['theta_accuracy_z']
        r_sem_vs_simple, _ = stats.pearsonr(latent_calibration, simple_cal)
        log(f"  Correlation (SEM vs Simple): r={r_sem_vs_simple:.4f}")

        sem_diagnostics.append({
            'Paradigm': paradigm,
            'r_xx_accuracy': r_xx_est,
            'r_yy_confidence': r_yy_est,
            'r_xy_correlation': r_xy,
            'r_diff_PRE_SEM': r_diff,
            'r_split_half_POST_SEM': r_split_half,
            'r_full_length_POST_SEM': r_full_length,
            'reliability_improvement': reliability_gain,
            'r_sem_vs_simple': r_sem_vs_simple,
            'method': fit_stats.get('method', 'latent_difference'),
            'n_observations': len(df_par)
        })

    except Exception as e:
        log(f"\nERROR during SEM for {paradigm}: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

log("\n" + "=" * 70)
log("SUMMARY: POST-SEM Reliability BY PARADIGM")
log("=" * 70)
for diag in sem_diagnostics:
    log(f"  {diag['Paradigm']:8s}: r_PRE={diag['r_diff_PRE_SEM']:+.4f} -> r_POST={diag['r_full_length_POST_SEM']:+.4f} ({diag['reliability_improvement']:+.4f})")

log("STEP 3 COMPLETE")

# Save

log("\n" + "=" * 70)
log("STEP 4: Save SEM Calibration Scores")
log("=" * 70)

n_missing = df['latent_calibration'].isna().sum()
if n_missing > 0:
    log(f"ERROR: {n_missing} missing latent_calibration values")
    sys.exit(1)
log("No missing values in latent_calibration")

df_out = df[['UID', 'TEST', 'Paradigm', 'TSVR_hours',
             'theta_accuracy', 'theta_confidence',
             'theta_accuracy_z', 'theta_confidence_z',
             'latent_calibration']].copy()

output_file = DATA_DIR / "step11_calibration_scores_SEM.csv"
df_out.to_csv(output_file, index=False)
log(f"Saved: {output_file} ({len(df_out)} rows)")

diagnostics_df = pd.DataFrame(sem_diagnostics)
diagnostics_file = DATA_DIR / "step11_SEM_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_file, index=False)
log(f"Saved diagnostics: {diagnostics_file}")

log("\n" + "=" * 70)
log("ALL STEPS COMPLETE")
log("=" * 70)
