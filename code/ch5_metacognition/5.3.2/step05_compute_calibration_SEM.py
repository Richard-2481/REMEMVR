#!/usr/bin/env python3
"""
RQ 6.3.2: SEM-Based Domain-Stratified Calibration Computation
==============================================================

Tier 1 SEM Validation: CRITICAL FINDING AT RISK
- Original: χ²=59.60 crossover interaction (thesis centerpiece)
- Reliability: r_diff=0.037 to 0.336 across domains (all CATASTROPHIC)
- Risk: Entire crossover pattern may be measurement artifact

**Approach:** Compute SEM latent calibration SEPARATELY FOR EACH DOMAIN
- What domain: r_diff=0.083
- Where domain: r_diff=0.037 (WORST)
- When domain: r_diff=0.336 (best but still fails)

Input: step00_calibration_by_domain.csv (1200 rows: 100 UID × 4 tests × 3 domains)
Output: step05_calibration_scores_SEM.csv (1200 rows, with latent_calibration by domain)

Author: Claude Code (Tier 1 Batch - Domain-Stratified SEM)
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

from sem_calibration import compute_difference_score_reliability, SEMCalibration

# ============================================================================
# SETUP
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.2
LOG_FILE = RQ_DIR / "logs" / "step05_SEM.log"
DATA_DIR = RQ_DIR / "data"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Expected validation
EXPECTED_ROWS = 1200  # 100 participants × 4 tests × 3 domains
EXPECTED_DOMAINS = ['What', 'Where', 'When']
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
log("RQ 6.3.2: SEM-Based Domain-Stratified Calibration")
log("=" * 70)
log("TIER 1 VALIDATION: Crossover interaction at risk (χ²=59.60, p<0.0001)")
log("Difference score reliability: CATASTROPHIC across all domains")
log("  - What: r_diff=0.083")
log("  - Where: r_diff=0.037 (WORST)")
log("  - When: r_diff=0.336")
log("Target: r>0.70 for EACH domain after SEM")
log("=" * 70)

# ============================================================================
# STEP 1: Load Merged Domain-Stratified Data
# ============================================================================

log("\n" + "=" * 70)
log("STEP 1: Load Domain-Stratified Calibration Data")
log("=" * 70)

input_file = DATA_DIR / "step00_calibration_by_domain.csv"
if not input_file.exists():
    log(f"ERROR: Input file not found: {input_file}")
    log("FAIL: Run steps_00_to_04.py first to generate step00_calibration_by_domain.csv")
    sys.exit(1)

df = pd.read_csv(input_file)
log(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
log(f"Columns: {list(df.columns)}")

# Validate structure
required_cols = ['UID', 'TEST', 'Domain', 'TSVR_hours',
                 'theta_accuracy', 'theta_confidence',
                 'theta_accuracy_z', 'theta_confidence_z',
                 'calibration', 'abs_calibration']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    log(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)
log(f"All required columns present: {len(required_cols)} columns")

# Validate row count
if len(df) < 800:  # Minimum 2 domains × 4 tests × 100 participants
    log(f"ERROR: Too few rows ({len(df)} < 800 minimum)")
    sys.exit(1)
log(f"Row count OK: {len(df)} rows (expected ~{EXPECTED_ROWS})")

# Check domains
domains_found = sorted(df['Domain'].unique())
log(f"Domains found: {domains_found}")
n_domains = len(domains_found)

# Check for missing values in theta
n_missing_acc = df['theta_accuracy'].isna().sum()
n_missing_conf = df['theta_confidence'].isna().sum()
if n_missing_acc > 0 or n_missing_conf > 0:
    log(f"ERROR: Missing theta values - accuracy: {n_missing_acc}, confidence: {n_missing_conf}")
    sys.exit(1)
log("No missing values in theta scores")

log("STEP 1 COMPLETE")

# ============================================================================
# STEP 2: Compute PRE-SEM Reliability BY DOMAIN
# ============================================================================

log("\n" + "=" * 70)
log("STEP 2: Compute PRE-SEM Difference Score Reliability (BY DOMAIN)")
log("=" * 70)

reliability_results = []

for domain in domains_found:
    log(f"\n{'=' * 70}")
    log(f"DOMAIN: {domain}")
    log(f"{'=' * 70}")

    df_domain = df[df['Domain'] == domain].copy()
    log(f"N = {len(df_domain)} observations ({df_domain['UID'].nunique()} participants × {df_domain['TEST'].nunique()} tests)")

    # Correlation between accuracy and confidence
    r_xy, p_value = stats.pearsonr(df_domain['theta_accuracy_z'], df_domain['theta_confidence_z'])
    log(f"Correlation (accuracy, confidence): r = {r_xy:.4f}, p = {p_value:.6f}")

    # Compute ACTUAL IRT reliabilities using ICC (Intraclass Correlation)
    # Between-person variance vs within-person variance

    # For accuracy
    person_means_acc = df_domain.groupby('UID')['theta_accuracy_z'].mean()
    person_vars_acc = df_domain.groupby('UID')['theta_accuracy_z'].var()

    between_var_acc = person_means_acc.var()
    within_var_acc = person_vars_acc.mean()

    r_xx_est = between_var_acc / (between_var_acc + within_var_acc)
    log(f"\nAccuracy IRT Reliability (ICC):")
    log(f"  Between-person variance: {between_var_acc:.4f}")
    log(f"  Within-person variance: {within_var_acc:.4f}")
    log(f"  ICC (r_xx): {r_xx_est:.4f}")

    # For confidence
    person_means_conf = df_domain.groupby('UID')['theta_confidence_z'].mean()
    person_vars_conf = df_domain.groupby('UID')['theta_confidence_z'].var()

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

    log(f"\nDifference Score Reliability (PRE-SEM) for {domain}:")
    log(f"  r_xx (accuracy reliability): {r_xx_est:.4f}")
    log(f"  r_yy (confidence reliability): {r_yy_est:.4f}")
    log(f"  r_xy (correlation): {r_xy:.4f}")
    log(f"  r_diff: {r_diff:.4f} ({interpretation})")

    if r_diff < 0:
        log(f"\n🔴 CATASTROPHIC: r_diff={r_diff:.4f} (NEGATIVE!)")
    elif r_diff < 0.60:
        log(f"\n⚠️  CRITICAL/LOW: r_diff={r_diff:.4f}")
    elif r_diff < 0.70:
        log(f"\n⚠️  MARGINAL: r_diff={r_diff:.4f}")
    else:
        log(f"\n✅ ACCEPTABLE: r_diff={r_diff:.4f}")

    # Store results
    reliability_results.append({
        'Domain': domain,
        'r_xx': r_xx_est,
        'r_yy': r_yy_est,
        'r_xy': r_xy,
        'r_diff': r_diff,
        'interpretation': interpretation
    })

log("\n" + "=" * 70)
log("SUMMARY: PRE-SEM Reliability BY DOMAIN")
log("=" * 70)
for res in reliability_results:
    log(f"{res['Domain']:8s}: r_diff={res['r_diff']:+.4f} ({res['interpretation']})")

log("STEP 2 COMPLETE")

# ============================================================================
# STEP 3: Compute SEM-Based Latent Calibration (BY DOMAIN)
# ============================================================================

log("\n" + "=" * 70)
log("STEP 3: Compute SEM-Based Latent Calibration (BY DOMAIN)")
log("=" * 70)

log("Approach: Latent difference for EACH domain separately")
log("Method: Factor score regression with measurement error")

# Will accumulate latent calibration scores by domain
df['latent_calibration'] = np.nan
sem_diagnostics = []

for domain in domains_found:
    log(f"\n{'=' * 70}")
    log(f"DOMAIN: {domain}")
    log(f"{'=' * 70}")

    df_domain = df[df['Domain'] == domain].copy()

    # Get reliability estimates for this domain
    rel = [r for r in reliability_results if r['Domain'] == domain][0]
    r_xx_est = rel['r_xx']
    r_yy_est = rel['r_yy']
    r_xy = rel['r_xy']
    r_diff = rel['r_diff']

    # Prepare inputs
    theta_accuracy = df_domain['theta_accuracy_z'].values
    theta_confidence = df_domain['theta_confidence_z'].values

    # SE: Estimate from reliability (standardized theta)
    se_accuracy = np.full(len(theta_accuracy), np.sqrt(max(1 - r_xx_est, 0.01)))
    se_confidence = np.full(len(theta_confidence), np.sqrt(max(1 - r_yy_est, 0.01)))

    log(f"Using estimated SE from ICC:")
    log(f"  SE_accuracy: {se_accuracy[0]:.4f} (from r_xx={r_xx_est:.4f})")
    log(f"  SE_confidence: {se_confidence[0]:.4f} (from r_yy={r_yy_est:.4f})")

    # Prepare DataFrames for SEM
    df_acc_input = pd.DataFrame({
        'UID': df_domain['UID'].values,
        'test': df_domain['TEST'].values,
        'theta_accuracy': theta_accuracy,
        'se_accuracy': se_accuracy
    })

    df_conf_input = pd.DataFrame({
        'UID': df_domain['UID'].values,
        'test': df_domain['TEST'].values,
        'theta_confidence': theta_confidence,
        'se_confidence': se_confidence
    })

    # Initialize SEM
    try:
        log(f"\nInitializing SEMCalibration for {domain}...")
        sem = SEMCalibration(
            theta_accuracy=df_acc_input,
            theta_confidence=df_conf_input,
            measurement_error_acc=None,  # Will use SE
            measurement_error_conf=None,  # Will use SE
            reliability_acc=r_xx_est,
            reliability_conf=r_yy_est,
            id_vars=['UID', 'test']
        )

        log(f"Fitting latent difference model for {domain}...")
        fit_stats = sem.fit_latent_difference(method='ML', verbose=True)

        log(f"\n✅ SEM calibration SUCCESSFUL for {domain}")

        # Get latent calibration scores
        latent_calibration = sem.get_latent_calibration(approach='difference')

        # Validate
        if len(latent_calibration) != len(df_domain):
            log(f"ERROR: Length mismatch - expected {len(df_domain)}, got {len(latent_calibration)}")
            sys.exit(1)

        if np.isnan(latent_calibration).any():
            log(f"WARNING: {np.isnan(latent_calibration).sum()} NaN values in latent_calibration")

        # Assign to main dataframe using index alignment
        # CRITICAL: Use df_domain.index to ensure alignment
        if len(latent_calibration) != len(df_domain):
            log(f"ERROR: Length mismatch - latent_calibration has {len(latent_calibration)}, df_domain has {len(df_domain)}")
            sys.exit(1)

        # Direct assignment using index
        for i, idx in enumerate(df_domain.index):
            df.at[idx, 'latent_calibration'] = latent_calibration[i]

        # Verify assignment
        n_assigned = df.loc[df['Domain'] == domain, 'latent_calibration'].notna().sum()
        log(f"Assigned {n_assigned}/{len(df_domain)} latent_calibration values for {domain}")

        # Compute POST-SEM reliability (split-half)
        df_domain_sem = df_domain.copy()
        df_domain_sem['latent_calibration'] = latent_calibration
        df_domain_sem['latent_cal_z'] = (latent_calibration - latent_calibration.mean()) / latent_calibration.std()

        # Odd vs even sessions (T1+T3 vs T2+T4)
        df_odd = df_domain_sem[df_domain_sem['TEST'].isin(['T1', 'T3'])].groupby('UID')['latent_cal_z'].mean().reset_index()
        df_odd.columns = ['UID', 'mean_cal_odd']

        df_even = df_domain_sem[df_domain_sem['TEST'].isin(['T2', 'T4'])].groupby('UID')['latent_cal_z'].mean().reset_index()
        df_even.columns = ['UID', 'mean_cal_even']

        df_split = df_odd.merge(df_even, on='UID')

        # Check for variance in split-half means
        odd_sd = df_split['mean_cal_odd'].std()
        even_sd = df_split['mean_cal_even'].std()

        if odd_sd < 0.001 or even_sd < 0.001 or np.isnan(odd_sd) or np.isnan(even_sd):
            log(f"  WARNING: Zero/NaN variance in split-half means (odd_SD={odd_sd:.4f}, even_SD={even_sd:.4f})")
            log(f"  Cannot compute split-half correlation - using ICC method")
            # Use ICC across all sessions instead
            person_means_lat = df_domain_sem.groupby('UID')['latent_cal_z'].mean()
            person_vars_lat = df_domain_sem.groupby('UID')['latent_cal_z'].var()
            between_var_lat = person_means_lat.var()
            within_var_lat = person_vars_lat.mean()
            r_full_length = between_var_lat / (between_var_lat + within_var_lat) if within_var_lat > 0 else np.nan
            r_split_half = np.nan
            log(f"  Using ICC method: r_full_length={r_full_length:.4f}")
        else:
            try:
                r_split_half, _ = stats.pearsonr(df_split['mean_cal_odd'], df_split['mean_cal_even'])
                if np.isnan(r_split_half):
                    raise ValueError("Pearson correlation returned NaN")
                r_full_length = (2 * r_split_half) / (1 + r_split_half)
            except Exception as e:
                log(f"  WARNING: Split-half correlation failed ({str(e)[:50]}) - using ICC method")
                person_means_lat = df_domain_sem.groupby('UID')['latent_cal_z'].mean()
                person_vars_lat = df_domain_sem.groupby('UID')['latent_cal_z'].var()
                between_var_lat = person_means_lat.var()
                within_var_lat = person_vars_lat.mean()
                r_full_length = between_var_lat / (between_var_lat + within_var_lat) if within_var_lat > 0 else np.nan
                r_split_half = np.nan

        log(f"\nSplit-Half Reliability (POST-SEM) for {domain}:")
        log(f"  Split-half r: {r_split_half:.4f}")
        log(f"  Full-length r (Spearman-Brown): {r_full_length:.4f}")

        if r_full_length >= 0.70:
            log(f"  ✅ GOOD reliability (>= 0.70) - Target achieved!")
        elif r_full_length >= 0.60:
            log(f"  ⚠️  MARGINAL reliability (0.60-0.69)")
        else:
            log(f"  🔴 LOW reliability (< 0.60)")

        # Reliability improvement
        reliability_gain = r_full_length - r_diff
        log(f"\n📊 RELIABILITY IMPROVEMENT for {domain}: {reliability_gain:+.4f}")
        log(f"   From r_diff={r_diff:.4f} to r={r_full_length:.4f}")

        # Correlation with simple difference
        simple_cal = df_domain['theta_confidence_z'] - df_domain['theta_accuracy_z']
        r_sem_vs_simple, _ = stats.pearsonr(latent_calibration, simple_cal)
        log(f"   Correlation (SEM vs Simple): r={r_sem_vs_simple:.4f}")

        # Store diagnostics
        sem_diagnostics.append({
            'Domain': domain,
            'r_xx_accuracy': r_xx_est,
            'r_yy_confidence': r_yy_est,
            'r_xy_correlation': r_xy,
            'r_diff_PRE_SEM': r_diff,
            'r_split_half_POST_SEM': r_split_half,
            'r_full_length_POST_SEM': r_full_length,
            'reliability_improvement': reliability_gain,
            'r_sem_vs_simple': r_sem_vs_simple,
            'method': fit_stats.get('method', 'latent_difference'),
            'n_observations': len(df_domain)
        })

    except Exception as e:
        log(f"\n❌ ERROR during SEM for {domain}: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

log("\n" + "=" * 70)
log("SUMMARY: POST-SEM Reliability BY DOMAIN")
log("=" * 70)
for diag in sem_diagnostics:
    log(f"{diag['Domain']:8s}: r_PRE={diag['r_diff_PRE_SEM']:+.4f} → r_POST={diag['r_full_length_POST_SEM']:+.4f} (Δ={diag['reliability_improvement']:+.4f})")

log("STEP 3 COMPLETE")

# ============================================================================
# STEP 4: Validate & Save SEM Calibration Scores
# ============================================================================

log("\n" + "=" * 70)
log("STEP 4: Validate & Save SEM Calibration Scores")
log("=" * 70)

# Check for any missing latent_calibration values
n_missing = df['latent_calibration'].isna().sum()
if n_missing > 0:
    log(f"ERROR: {n_missing} missing latent_calibration values")
    sys.exit(1)
log("No missing values in latent_calibration")

# Prepare output dataframe
df_out = df[['UID', 'TEST', 'Domain', 'TSVR_hours',
             'theta_accuracy', 'theta_confidence',
             'theta_accuracy_z', 'theta_confidence_z',
             'latent_calibration']].copy()

# Validate
if len(df_out) < 800:
    log(f"WARNING: Expected ~{EXPECTED_ROWS} rows, got {len(df_out)}")

# Save SEM calibration scores
output_file = DATA_DIR / "step05_calibration_scores_SEM.csv"
df_out.to_csv(output_file, index=False)
log(f"\nSaved: {output_file} ({len(df_out)} rows, {len(df_out.columns)} columns)")

# Save diagnostics
diagnostics_df = pd.DataFrame(sem_diagnostics)
diagnostics_file = DATA_DIR / "step05_SEM_diagnostics.csv"
diagnostics_df.to_csv(diagnostics_file, index=False)
log(f"Saved diagnostics: {diagnostics_file}")

log("STEP 4 COMPLETE")

# ============================================================================
# SUMMARY
# ============================================================================

log("\n" + "=" * 70)
log("SEM-BASED DOMAIN-STRATIFIED CALIBRATION COMPLETE")
log("=" * 70)

log("\n📊 KEY RESULTS (BY DOMAIN):")
for i, diag in enumerate(sem_diagnostics, 1):
    log(f"\n{i}. {diag['Domain']} Domain:")
    log(f"   PRE-SEM: r_diff = {diag['r_diff_PRE_SEM']:.4f}")
    log(f"   POST-SEM: r = {diag['r_full_length_POST_SEM']:.4f} (split-half)")
    log(f"   Improvement: {diag['reliability_improvement']:+.4f} ({diag['reliability_improvement']*100:+.1f} pp)")
    log(f"   Correlation with simple: r = {diag['r_sem_vs_simple']:.4f}")

# Overall success assessment
all_pass = all(d['r_full_length_POST_SEM'] >= 0.70 for d in sem_diagnostics)
any_marginal = any(0.60 <= d['r_full_length_POST_SEM'] < 0.70 for d in sem_diagnostics)
any_fail = any(d['r_full_length_POST_SEM'] < 0.60 for d in sem_diagnostics)

if all_pass:
    log("\n✅ SUCCESS: All domains achieve target reliability (r>=0.70)")
elif any_marginal and not any_fail:
    log("\n⚠️  MARGINAL: Some domains below target but all >= 0.60")
else:
    log("\n🔴 WARNING: Some domains remain low reliability (<0.60)")

log("\n📁 OUTPUT FILES:")
log(f"   {output_file.name}")
log(f"   {diagnostics_file.name}")

log("\n🔄 NEXT STEPS:")
log("   1. Create steps_00_to_04_SEM.py (modified to use latent_calibration)")
log("   2. Re-run LMM: calibration ~ Domain × TSVR + (TSVR | UID)")
log("   3. Compare PRE-SEM (χ²=59.60) vs POST-SEM crossover interaction")
log("   4. Classify outcome: ROBUST / NULL / MARGINAL")

log("\n" + "=" * 70)
log("ALL STEPS COMPLETE")
log("=" * 70)
