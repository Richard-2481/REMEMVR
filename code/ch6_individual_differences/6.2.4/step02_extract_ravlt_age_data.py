#!/usr/bin/env python3
"""
Step 2: Extract and Prepare RAVLT and Age Data
RQ 7.2.4 - VR Scaffolding Validation

Purpose: Extract RAVLT scores and age from dfnonvr.csv with total score calculation
         Applies ceiling fix to trial scores and computes percent retention
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_extract_ravlt.log"

def log(msg):
    """Log to both file and stdout"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    When a participant scores 14 or 15 on a trial, the next trial may not be
    administered (recorded as 0). This substitutes the maximum possible score (15)
    for those unadministered trials. Applied to trials 1-5 only, not delayed recall.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0
    for idx in df.index:
        for i in range(1, 5):
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df.at[idx, prev_col]})")
    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df


def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT percent retention: delayed recall / best learning trial * 100.

    Uses the highest non-zero trial score (searching from trial 5 backwards) as
    the denominator. Returns NaN if no valid denominator is found.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    dr_col = 'ravlt-delayed-recall-score'
    pct_ret = np.full(len(df), np.nan)
    for i, idx in enumerate(df.index):
        dr = df.at[idx, dr_col]
        denom = np.nan
        for trial_col in reversed(trial_cols):
            val = df.at[idx, trial_col]
            if val > 0:
                denom = val
                break
        if denom > 0:
            pct_ret[i] = (dr / denom) * 100
    n_valid = np.sum(~np.isnan(pct_ret))
    log_fn(f"RAVLT Percent Retention: {n_valid}/{len(df)} valid")
    return pct_ret


def main():
    log("=" * 60)
    log("Step 2: Extract RAVLT and Age Data")
    log("=" * 60)

    # Read dfnonvr.csv
    dfnonvr_path = Path("data/dfnonvr.csv")
    log(f"Reading participant data from: {dfnonvr_path}")

    df_master = pd.read_csv(dfnonvr_path)
    log(f"Loaded {len(df_master)} rows with {len(df_master.columns)} columns")

    # Define RAVLT trial columns and age column
    ravlt_cols = ['ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score',
                  'ravlt-trial-4-score', 'ravlt-trial-5-score', 'ravlt-delayed-recall-score']
    age_col = 'age'

    # Verify columns exist
    missing_cols = [col for col in ravlt_cols + [age_col] if col not in df_master.columns]
    if missing_cols:
        log(f"ERROR: Missing columns: {missing_cols}")
        sys.exit(1)

    log("All required columns found")

    # Extract relevant columns
    df_ravlt = df_master[['UID'] + ravlt_cols + [age_col]].copy()

    # Ensure UID is string type
    df_ravlt['UID'] = df_ravlt['UID'].astype(str)

    # Apply ceiling fix to trial scores BEFORE computing totals
    log("\nApplying ceiling fix to RAVLT trial scores...")
    df_ravlt = fix_ravlt_ceiling(df_ravlt, log)

    # Calculate RAVLT_Total as sum of trials 1-5 + delayed recall
    # NOTE: This intentionally includes delayed recall in the total (non-standard)
    log("\nCalculating RAVLT_Total as sum of trials 1-5 + delayed recall")
    df_ravlt['RAVLT_Total'] = df_ravlt[ravlt_cols].sum(axis=1)

    # Compute RAVLT Percent Retention as separate variable
    log("\nComputing RAVLT Percent Retention...")
    df_ravlt['RAVLT_Pct_Ret'] = compute_ravlt_percent_retention(df_ravlt, log)

    # Rename Age column for consistency
    df_ravlt = df_ravlt.rename(columns={'age': 'Age'})

    # Check for missing data
    n_before = len(df_ravlt)
    df_ravlt = df_ravlt.dropna(subset=['RAVLT_Total', 'Age'])
    n_after = len(df_ravlt)

    if n_before != n_after:
        log(f"Removed {n_before - n_after} participants with missing data")

    # Standardize RAVLT scores (z-score transformation)
    ravlt_mean = df_ravlt['RAVLT_Total'].mean()
    ravlt_std = df_ravlt['RAVLT_Total'].std()
    df_ravlt['RAVLT_Total_z'] = (df_ravlt['RAVLT_Total'] - ravlt_mean) / ravlt_std

    # Standardize RAVLT Percent Retention
    pct_ret_valid = df_ravlt['RAVLT_Pct_Ret'].dropna()
    pct_ret_mean = pct_ret_valid.mean()
    pct_ret_std = pct_ret_valid.std()
    df_ravlt['RAVLT_Pct_Ret_z'] = (df_ravlt['RAVLT_Pct_Ret'] - pct_ret_mean) / pct_ret_std

    # Quality checks
    n_participants = len(df_ravlt)
    age_min, age_max = df_ravlt['Age'].min(), df_ravlt['Age'].max()
    age_mean, age_std = df_ravlt['Age'].mean(), df_ravlt['Age'].std()
    age_range = age_max - age_min
    age_variance = df_ravlt['Age'].var()

    log(f"\nRESULTS:")
    log(f"RAVLT data extracted: {n_participants} participants")
    log(f"RAVLT Total descriptives: mean={ravlt_mean:.1f}, sd={ravlt_std:.1f}")
    log(f"RAVLT Total range: min={df_ravlt['RAVLT_Total'].min():.1f}, max={df_ravlt['RAVLT_Total'].max():.1f}")
    log(f"RAVLT Pct Ret descriptives: mean={pct_ret_mean:.1f}, sd={pct_ret_std:.1f}")
    log(f"RAVLT Pct Ret range: min={df_ravlt['RAVLT_Pct_Ret'].min():.1f}, max={df_ravlt['RAVLT_Pct_Ret'].max():.1f}")
    n_pct_ret_valid = df_ravlt['RAVLT_Pct_Ret'].notna().sum()
    log(f"RAVLT Pct Ret valid: {n_pct_ret_valid}/{n_participants}")
    log(f"Age descriptives: mean={age_mean:.1f}, sd={age_std:.1f}")
    log(f"Age range: {age_min:.0f}-{age_max:.0f} years (range={age_range:.0f})")
    log(f"Age variance: {age_variance:.1f}")

    # Check for adequate age variance for correlation
    if age_range < 20:
        log(f"WARNING: Insufficient age variance - range only {age_range:.0f} years")
    else:
        log(f"Age range {age_range:.0f} years: adequate variance for correlation")

    # Check for ceiling/floor effects in RAVLT
    ravlt_min_possible = 0
    ravlt_max_possible = 90  # 15 words * 6 trials
    pct_at_floor = (df_ravlt['RAVLT_Total'] <= 10).mean() * 100
    pct_at_ceiling = (df_ravlt['RAVLT_Total'] >= 80).mean() * 100

    log(f"\nCeiling/floor effects check:")
    log(f"  - At floor (<=10): {pct_at_floor:.1f}%")
    log(f"  - At ceiling (>=80): {pct_at_ceiling:.1f}%")

    if pct_at_floor > 5 or pct_at_ceiling > 5:
        log("WARNING: Potential ceiling/floor effects detected")

    # Verify standardization
    z_mean = df_ravlt['RAVLT_Total_z'].mean()
    z_std = df_ravlt['RAVLT_Total_z'].std()
    log(f"\nStandardization verification (Total): z_mean={z_mean:.6f}, z_std={z_std:.6f}")
    pct_z_mean = df_ravlt['RAVLT_Pct_Ret_z'].dropna().mean()
    pct_z_std = df_ravlt['RAVLT_Pct_Ret_z'].dropna().std()
    log(f"Standardization verification (Pct Ret): z_mean={pct_z_mean:.6f}, z_std={pct_z_std:.6f}")

    # Select final columns and save
    df_final = df_ravlt[['UID', 'RAVLT_Total', 'RAVLT_Total_z', 'RAVLT_Pct_Ret', 'RAVLT_Pct_Ret_z', 'Age']]

    output_path = RQ_DIR / "data" / "step02_ravlt_age_data.csv"
    df_final.to_csv(output_path, index=False)
    log(f"\nSaved to: {output_path}")
    log(f"Output shape: {df_final.shape}")

    # Final verification
    log("\nFinal data check:")
    log(f"  - Participants: {n_participants}")
    log(f"  - Columns: {list(df_final.columns)}")
    log(f"  - No missing values (Total/Age): {df_final[['RAVLT_Total', 'Age']].isnull().sum().sum() == 0}")
    log(f"  - Pct Ret missing: {df_final['RAVLT_Pct_Ret'].isnull().sum()}")
    log(f"  - All finite values (Total/Age): {np.isfinite(df_final[['RAVLT_Total', 'RAVLT_Total_z', 'Age']]).all().all()}")

    log("\nStep 2 completed successfully")

if __name__ == "__main__":
    main()