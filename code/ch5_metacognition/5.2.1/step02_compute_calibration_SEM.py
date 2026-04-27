#!/usr/bin/env python3
"""
RQ 6.2.1: SEM-Based Calibration Computation
============================================

Replaces simple difference scores (calibration = confidence - accuracy) with
Structural Equation Modeling approach that properly accounts for IRT measurement error.

**Problem:** Simple difference scores have catastrophic reliability (r_diff=-0.16 for RQ 6.2.2)
**Solution:** SEM with latent variables produces reliable calibration scores (target r>0.70)

Input: step01_merged_theta.csv (400 rows, theta + SE for accuracy and confidence)
Output: step02_calibration_scores_SEM.csv (400 rows, with latent_calibration)

Author: Claude Code (Phase 2 SEM Prototype)
Date: 2025-12-28
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add tools to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from sem_calibration import quick_sem_calibration, compute_difference_score_reliability

# ============================================================================
# SETUP
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.1
LOG_FILE = RQ_DIR / "logs" / "step02_SEM.log"
DATA_DIR = RQ_DIR / "data"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Expected validation
EXPECTED_ROWS = 400  # 100 participants x 4 tests
Z_STANDARDIZATION_TOLERANCE = 0.01


def log(msg: str) -> None:
    """Log message to both file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


# Clear log file
LOG_FILE.write_text("")
log("=" * 70)
log("RQ 6.2.1: SEM-Based Calibration Computation")
log("=" * 70)
log(f"Phase 2 Prototype: Replacing simple difference with latent variables")
log(f"Target reliability: r > 0.70 (vs r_diff=-0.16 for simple difference)")
log("=" * 70)

# ============================================================================
# STEP 1: Load Merged Theta Scores
# ============================================================================

log("\n" + "=" * 70)
log("STEP 1: Load Merged Theta Scores")
log("=" * 70)

input_file = DATA_DIR / "step01_merged_theta.csv"
if not input_file.exists():
    log(f"ERROR: Input file not found: {input_file}")
    log("FAIL: Run steps_00_to_07.py first to generate step01_merged_theta.csv")
    sys.exit(1)

df = pd.read_csv(input_file)
log(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
log(f"Columns: {list(df.columns)}")

# Validate structure
required_cols = ['UID', 'test', 'composite_ID', 'TSVR_hours',
                 'theta_accuracy', 'se_accuracy',
                 'theta_confidence', 'se_confidence',
                 'z_theta_accuracy', 'z_theta_confidence']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    log(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)
log(f"All required columns present: {len(required_cols)} columns")

# Validate row count
if len(df) != EXPECTED_ROWS:
    log(f"WARNING: Expected {EXPECTED_ROWS} rows, found {len(df)}")

# Check for missing values
n_missing_acc = df['theta_accuracy'].isna().sum()
n_missing_conf = df['theta_confidence'].isna().sum()
if n_missing_acc > 0 or n_missing_conf > 0:
    log(f"ERROR: Missing theta values - accuracy: {n_missing_acc}, confidence: {n_missing_conf}")
    sys.exit(1)
log("No missing values in theta scores")

# Check SE values (may be NaN from RQ 5.1.1, will be estimated)
n_missing_se_acc = df['se_accuracy'].isna().sum()
n_missing_se_conf = df['se_confidence'].isna().sum()
log(f"SE values - accuracy NaN: {n_missing_se_acc}/{len(df)}, confidence NaN: {n_missing_se_conf}/{len(df)}")

if n_missing_se_acc > 0:
    log("WARNING: Missing SE for accuracy, will use conservative estimate")
if n_missing_se_conf > 0:
    log("WARNING: Missing SE for confidence, will use conservative estimate")

log("STEP 1 COMPLETE")

# ============================================================================
# STEP 2: Compute PRE-SEM Difference Score Reliability (Baseline)
# ============================================================================

log("\n" + "=" * 70)
log("STEP 2: Compute PRE-SEM Difference Score Reliability (Baseline)")
log("=" * 70)

# Correlation between accuracy and confidence
r_xy, p_value = stats.pearsonr(df['z_theta_accuracy'], df['z_theta_confidence'])
log(f"Correlation (z_theta_accuracy, z_theta_confidence): r = {r_xy:.4f}, p = {p_value:.6f}")

# Compute ACTUAL IRT reliabilities using ICC (Intraclass Correlation)
# Method: Variance decomposition (between-person vs within-person)
# This is the CORRECT approach for repeated measures (100 participants × 4 tests)

log("\nComputing IRT reliabilities via ICC (variance decomposition)...")

# For accuracy: Compute ICC
# Between-person variance = variance of person means
# Within-person variance = mean of person variances
person_means_acc = df.groupby('UID')['z_theta_accuracy'].mean()
person_vars_acc = df.groupby('UID')['z_theta_accuracy'].var()

between_var_acc = person_means_acc.var()
within_var_acc = person_vars_acc.mean()

r_xx_est = between_var_acc / (between_var_acc + within_var_acc)
log(f"\nAccuracy IRT Reliability (ICC):")
log(f"  Between-person variance: {between_var_acc:.4f}")
log(f"  Within-person variance: {within_var_acc:.4f}")
log(f"  ICC (r_xx): {r_xx_est:.4f}")

# For confidence: Compute ICC
person_means_conf = df.groupby('UID')['z_theta_confidence'].mean()
person_vars_conf = df.groupby('UID')['z_theta_confidence'].var()

between_var_conf = person_means_conf.var()
within_var_conf = person_vars_conf.mean()

r_yy_est = between_var_conf / (between_var_conf + within_var_conf)
log(f"\nConfidence IRT Reliability (ICC):")
log(f"  Between-person variance: {between_var_conf:.4f}")
log(f"  Within-person variance: {within_var_conf:.4f}")
log(f"  ICC (r_yy): {r_yy_est:.4f}")

# Compute difference score reliability
r_diff = compute_difference_score_reliability(
    r_xx=r_xx_est,
    r_yy=r_yy_est,
    r_xy=r_xy
)

# Interpret
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

log(f"\nDifference Score Reliability (PRE-SEM):")
log(f"  r_xx (accuracy reliability): {r_xx_est:.4f}")
log(f"  r_yy (confidence reliability): {r_yy_est:.4f}")
log(f"  r_xy (correlation): {r_xy:.4f}")
log(f"  r_diff: {r_diff:.4f} ({interpretation})")

if r_diff < 0:
    log(f"\n🔴 CATASTROPHIC: r_diff={r_diff:.4f} (NEGATIVE!)")
    log("    Difference score is WORSE than random measurement!")
    log("    SEM implementation is MANDATORY")
elif r_diff < 0.60:
    log(f"\n⚠️  CRITICAL: r_diff={r_diff:.4f} (LOW)")
    log("    SEM implementation is strongly recommended")
elif r_diff < 0.70:
    log(f"\n⚠️  MARGINAL: r_diff={r_diff:.4f}")
    log("    SEM implementation recommended for publication quality")
else:
    log(f"\n✅ ACCEPTABLE: r_diff={r_diff:.4f}")
    log("    SEM will still improve reliability")

log("STEP 2 COMPLETE")

# ============================================================================
# STEP 3: Compute SEM-Based Latent Calibration
# ============================================================================

log("\n" + "=" * 70)
log("STEP 3: Compute SEM-Based Latent Calibration")
log("=" * 70)

log("Using tools/sem_calibration.py - quick_sem_calibration() function")
log("Approach: Latent difference (Latent_Calibration = Latent_Confidence - Latent_Accuracy)")
log("Measurement error properly accounted for via factor score regression")

# Prepare inputs (using z-standardized theta for consistency with original analysis)
theta_accuracy = df['z_theta_accuracy'].values
theta_confidence = df['z_theta_confidence'].values

# SE: Use actual values if available, otherwise estimate from reliability
se_accuracy = df['se_accuracy'].values.copy()
se_confidence = df['se_confidence'].values.copy()

# Fill missing SE with estimates
if np.isnan(se_accuracy).any():
    # SE = sqrt(1 - reliability) for standardized theta
    se_est_acc = np.sqrt(1 - r_xx_est)
    se_accuracy = np.where(np.isnan(se_accuracy), se_est_acc, se_accuracy)
    log(f"Filled {np.isnan(df['se_accuracy'].values).sum()} missing SE_accuracy with estimate: {se_est_acc:.4f}")

if np.isnan(se_confidence).any():
    se_est_conf = np.sqrt(1 - r_yy_est)
    se_confidence = np.where(np.isnan(se_confidence), se_est_conf, se_confidence)
    log(f"Filled {np.isnan(df['se_confidence'].values).sum()} missing SE_confidence with estimate: {se_est_conf:.4f}")

# Import SEM class
from sem_calibration import SEMCalibration

# Prepare data for SEM (need DataFrames with theta columns)
df_acc_input = pd.DataFrame({
    'UID': df['UID'],
    'test': df['test'],
    'theta_accuracy': theta_accuracy,
    'se_accuracy': se_accuracy
})

df_conf_input = pd.DataFrame({
    'UID': df['UID'],
    'test': df['test'],
    'theta_confidence': theta_confidence,
    'se_confidence': se_confidence
})

# Initialize SEM
try:
    log("\nInitializing SEMCalibration class...")
    sem = SEMCalibration(
        theta_accuracy=df_acc_input,
        theta_confidence=df_conf_input,
        measurement_error_acc=None,  # Will use SE
        measurement_error_conf=None,  # Will use SE
        reliability_acc=r_xx_est,
        reliability_conf=r_yy_est,
        id_vars=['UID', 'test']
    )

    log("\nFitting latent difference model...")
    fit_stats = sem.fit_latent_difference(method='ML', verbose=True)

    log("\n✅ SEM calibration computation SUCCESSFUL")

    # Get latent calibration scores
    latent_calibration = sem.get_latent_calibration(approach='difference')

    # Collect diagnostics
    diagnostics = {
        'method': fit_stats.get('method', 'latent_difference'),
        'reliability_acc': fit_stats.get('reliability_acc', r_xx_est),
        'reliability_conf': fit_stats.get('reliability_conf', r_yy_est),
        'shrinkage_applied': fit_stats.get('shrinkage_applied', False),
        **fit_stats
    }

except Exception as e:
    log(f"\n❌ ERROR during SEM computation: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

# Add latent_calibration to dataframe
df['latent_calibration'] = latent_calibration

# Validate output
if len(latent_calibration) != len(df):
    log(f"ERROR: Output length mismatch - expected {len(df)}, got {len(latent_calibration)}")
    sys.exit(1)

if np.isnan(latent_calibration).any():
    log(f"WARNING: {np.isnan(latent_calibration).sum()} NaN values in latent_calibration")

# Log diagnostics
log("\nSEM Diagnostics:")
for key, value in diagnostics.items():
    if isinstance(value, (int, float, np.number)):
        log(f"  {key}: {value:.4f}" if abs(value) < 1000 else f"  {key}: {value}")
    else:
        log(f"  {key}: {value}")

log("STEP 3 COMPLETE")

# ============================================================================
# STEP 4: Validate SEM Calibration Reliability (Target r>0.70)
# ============================================================================

log("\n" + "=" * 70)
log("STEP 4: Validate SEM Calibration Reliability (Target r>0.70)")
log("=" * 70)

# Method 1: Correlation with original simple difference (should be high but not perfect)
simple_calibration = df['z_theta_confidence'] - df['z_theta_accuracy']
r_sem_vs_simple, p_val = stats.pearsonr(latent_calibration, simple_calibration)
log(f"\nCorrelation (SEM vs Simple Difference): r = {r_sem_vs_simple:.4f}, p = {p_val:.6f}")

if r_sem_vs_simple > 0.95:
    log("  ✅ High correlation (>0.95): SEM preserves core signal")
elif r_sem_vs_simple > 0.80:
    log("  ⚠️  Moderate correlation (0.80-0.95): SEM applies substantial correction")
else:
    log(f"  🔴 Low correlation (<0.80): SEM substantially different from simple difference")

# Method 2: Test-retest reliability (split-half: T1+T3 vs T2+T4)
# This is a rough estimate since we don't have true test-retest data
df['latent_cal_z'] = (df['latent_calibration'] - df['latent_calibration'].mean()) / df['latent_calibration'].std()

# Compute person-level means for odd vs even sessions
df_odd = df[df['test'].isin(['T1', 'T3'])].groupby('UID')['latent_cal_z'].mean().reset_index()
df_odd.columns = ['UID', 'mean_cal_odd']

df_even = df[df['test'].isin(['T2', 'T4'])].groupby('UID')['latent_cal_z'].mean().reset_index()
df_even.columns = ['UID', 'mean_cal_even']

df_split = df_odd.merge(df_even, on='UID')

r_split_half, p_split = stats.pearsonr(df_split['mean_cal_odd'], df_split['mean_cal_even'])
# Spearman-Brown prophecy formula for full-length reliability
r_full_length = (2 * r_split_half) / (1 + r_split_half)

log(f"\nSplit-Half Reliability (T1+T3 vs T2+T4):")
log(f"  Split-half r: {r_split_half:.4f}")
log(f"  Full-length r (Spearman-Brown): {r_full_length:.4f}")

if r_full_length >= 0.70:
    log(f"  ✅ GOOD reliability (>= 0.70) - Target achieved!")
elif r_full_length >= 0.60:
    log(f"  ⚠️  MARGINAL reliability (0.60-0.69) - Better than difference score")
else:
    log(f"  🔴 LOW reliability (< 0.60) - SEM improvement insufficient")

# Method 3: Descriptive statistics comparison
log(f"\nDescriptive Statistics Comparison:")
log(f"PRE-SEM (Simple Difference):")
log(f"  Mean: {simple_calibration.mean():+.4f}")
log(f"  SD: {simple_calibration.std():.4f}")
log(f"  Range: [{simple_calibration.min():+.4f}, {simple_calibration.max():+.4f}]")
log(f"  Reliability: r_diff = {r_diff:.4f}")

log(f"\nPOST-SEM (Latent Calibration):")
log(f"  Mean: {latent_calibration.mean():+.4f}")
log(f"  SD: {latent_calibration.std():.4f}")
log(f"  Range: [{latent_calibration.min():+.4f}, {latent_calibration.max():+.4f}]")
log(f"  Reliability (split-half): r = {r_full_length:.4f}")

# Reliability improvement
reliability_gain = r_full_length - r_diff
log(f"\n📊 RELIABILITY IMPROVEMENT: {reliability_gain:+.4f}")
if reliability_gain > 0:
    log(f"   SEM achieves {reliability_gain*100:.1f} percentage points gain in reliability")
    log(f"   From r_diff={r_diff:.4f} to r={r_full_length:.4f}")
else:
    log(f"   ⚠️  WARNING: SEM did not improve reliability")

log("STEP 4 COMPLETE")

# ============================================================================
# STEP 5: Save SEM-Based Calibration Scores
# ============================================================================

log("\n" + "=" * 70)
log("STEP 5: Save SEM-Based Calibration Scores")
log("=" * 70)

# Prepare output dataframe (same structure as original step02_calibration_scores.csv)
# But with latent_calibration instead of simple difference
df_out = df[['UID', 'test', 'composite_ID', 'TSVR_hours',
             'z_theta_accuracy', 'z_theta_confidence', 'latent_calibration']].copy()

# Rename for consistency with downstream code
df_out = df_out.rename(columns={'latent_calibration': 'calibration'})

# Validate output
if len(df_out) != EXPECTED_ROWS:
    log(f"WARNING: Expected {EXPECTED_ROWS} rows, got {len(df_out)}")

if df_out['calibration'].isna().sum() > 0:
    log(f"ERROR: {df_out['calibration'].isna().sum()} NaN values in calibration")
    sys.exit(1)

# Save to new file (preserving original)
output_file = DATA_DIR / "step02_calibration_scores_SEM.csv"
df_out.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_out)} rows, {len(df_out.columns)} columns)")

# Also save diagnostics
diagnostics_df = pd.DataFrame([{
    'r_xx_accuracy': r_xx_est,
    'r_yy_confidence': r_yy_est,
    'r_xy_correlation': r_xy,
    'r_diff_PRE_SEM': r_diff,
    'r_split_half_POST_SEM': r_split_half,
    'r_full_length_POST_SEM': r_full_length,
    'reliability_improvement': reliability_gain,
    'method': diagnostics.get('method', 'unknown'),
    'approach': 'latent_difference',
    'n_observations': len(df_out)
}])

diagnostics_file = DATA_DIR / "step02_SEM_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_file, index=False)
log(f"Saved diagnostics: {diagnostics_file}")

log("STEP 5 COMPLETE")

# ============================================================================
# SUMMARY
# ============================================================================

log("\n" + "=" * 70)
log("SEM-BASED CALIBRATION COMPUTATION COMPLETE")
log("=" * 70)

log("\n📊 KEY RESULTS:")
log(f"1. PRE-SEM reliability: r_diff = {r_diff:.4f} ({interpretation})")
log(f"2. POST-SEM reliability: r = {r_full_length:.4f} (split-half, Spearman-Brown corrected)")
log(f"3. Reliability improvement: {reliability_gain:+.4f} ({reliability_gain*100:+.1f} percentage points)")
log(f"4. Method used: {diagnostics.get('method', 'unknown')}")
log(f"5. Correlation with simple difference: r = {r_sem_vs_simple:.4f}")

if r_full_length >= 0.70:
    log("\n✅ SUCCESS: Target reliability (r>=0.70) ACHIEVED")
    log("   SEM calibration scores are suitable for RQ 6.2.2 analysis")
elif r_full_length >= 0.60:
    log("\n⚠️  MARGINAL: Reliability improved but below target")
    log("   SEM calibration still recommended over simple difference")
else:
    log("\n🔴 WARNING: Low reliability despite SEM")
    log("   Further investigation needed")

log("\n📁 OUTPUT FILES:")
log(f"   {output_file.name}")
log(f"   {diagnostics_file.name}")

log("\n🔄 NEXT STEPS:")
log("   1. Modify RQ 6.2.2 to use step02_calibration_scores_SEM.csv")
log("   2. Re-run RQ 6.2.2 analysis with SEM calibration")
log("   3. Compare PRE-SEM (p=0.230) vs POST-SEM (expected: stronger effect)")
log("   4. Document findings in Phase 2 prototype report")

log("\n" + "=" * 70)
log("ALL STEPS COMPLETE")
log("=" * 70)
