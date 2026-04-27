#!/usr/bin/env python3
"""
Step 01: Extract Cognitive Tests
RQ 7.1.4: Unique REMEMVR variance unexplained by all predictors

Extract RAVLT, BVMT, NART, and RPM scores from dfnonvr.csv

v2 CHANGES (2026-03-22):
1. RAVLT ceiling fix: participants with unadministered trials stored as 0.
   Substitutes 15 where trial N == 0 and trial N-1 >= 14 (ceiling performance).
2. BVMT Total recomputed explicitly from sum(trials 1-3) instead of pre-computed column.
3. Added RAVLT Percent Retention (Delayed Recall / best available trial x 100).
4. Added BVMT Percent Retained (from pre-computed column in dfnonvr.csv).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
RQ_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = RQ_DIR.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# Set up logging
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    """Log to both console and file."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    Logic: If a participant scored >= 14 on trial N-1 and trial N == 0,
    trial N was not administered (ceiling). Substitute 15.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0

    for idx in df.index:
        for i in range(1, 5):  # Check trials 2,3,4,5 (compare with previous)
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                old_val = df.at[idx, current_col]
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} {old_val} -> 15 "
                       f"(prev trial = {df.at[idx, prev_col]})")

    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df

def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT Percent Retention = Delayed Recall / best available trial x 100.

    Denominator: last non-zero trial score (trial 5 -> 4 -> 3 fallback).
    After ceiling fix, this should always be trial 5 for all participants.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    dr_col = 'ravlt-delayed-recall-score'

    pct_ret = np.full(len(df), np.nan)
    for i, idx in enumerate(df.index):
        dr = df.at[idx, dr_col]
        # Find best available trial (last non-zero, working backwards)
        denom = np.nan
        for trial_col in reversed(trial_cols):
            val = df.at[idx, trial_col]
            if val > 0:
                denom = val
                break
        if denom > 0:
            pct_ret[i] = (dr / denom) * 100

    n_valid = np.sum(~np.isnan(pct_ret))
    log_fn(f"[COMPUTED] RAVLT Percent Retention: {n_valid}/{len(df)} valid, "
           f"M={np.nanmean(pct_ret):.1f}%, SD={np.nanstd(pct_ret):.1f}%")
    return pct_ret

def main():
    """Main execution."""
    log("[START] Step 01: Extract cognitive tests (v2 - ceiling fix + percent retention)")

    # Load participant data
    log("[LOAD] Reading dfnonvr.csv...")
    df = pd.read_csv(PROJ_ROOT / "data" / "dfnonvr.csv")
    log(f"[INFO] Loaded {len(df)} participants from dfnonvr.csv")

    # =========================================================================
    # Apply RAVLT ceiling fix BEFORE computing totals
    # =========================================================================
    log("\n[CEILING] Applying RAVLT ceiling fix (0 -> 15 for unadministered trials)...")
    df = fix_ravlt_ceiling(df, log)

    # =========================================================================
    # Extract and compute cognitive test scores
    # =========================================================================
    log("\n[EXTRACT] Extracting cognitive test scores...")

    # Initialize output dataframe
    cognitive_df = pd.DataFrame()
    cognitive_df['uid'] = df['UID'].astype(str)

    # RAVLT Total: Sum trials 1-5 (after ceiling fix)
    ravlt_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    missing_cols = [c for c in ravlt_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing RAVLT trial columns: {missing_cols}")

    cognitive_df['RAVLT_T'] = df[ravlt_cols].sum(axis=1)
    log(f"[SUCCESS] RAVLT Total (ceiling-fixed): "
        f"M={cognitive_df['RAVLT_T'].mean():.1f}, SD={cognitive_df['RAVLT_T'].std():.1f}")

    # RAVLT Delayed Recall
    if 'ravlt-delayed-recall-score' in df.columns:
        cognitive_df['RAVLT_DR_T'] = df['ravlt-delayed-recall-score']
        log("[SUCCESS] RAVLT delayed recall extracted")
    else:
        raise ValueError("Column 'ravlt-delayed-recall-score' not found")

    # RAVLT Percent Retention (new predictor)
    cognitive_df['RAVLT_Pct_Ret'] = compute_ravlt_percent_retention(df, log)

    # BVMT Total: Explicitly sum trials 1-3 (not pre-computed column)
    bvmt_trial_cols = [f'bvmt-trial-{i}-score' for i in range(1, 4)]
    missing_cols = [c for c in bvmt_trial_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing BVMT trial columns: {missing_cols}")

    cognitive_df['BVMT_T'] = df[bvmt_trial_cols].sum(axis=1)
    log(f"[SUCCESS] BVMT Total (sum trials 1-3): "
        f"M={cognitive_df['BVMT_T'].mean():.1f}, SD={cognitive_df['BVMT_T'].std():.1f}")

    # BVMT Percent Retained (new predictor, pre-computed in CSV)
    if 'bvmt-percent-retained' not in df.columns:
        raise ValueError("Column 'bvmt-percent-retained' not found in dfnonvr.csv")
    cognitive_df['BVMT_Pct_Ret'] = df['bvmt-percent-retained']
    log(f"[SUCCESS] BVMT Percent Retained: "
        f"M={cognitive_df['BVMT_Pct_Ret'].mean():.1f}%, SD={cognitive_df['BVMT_Pct_Ret'].std():.1f}%")

    # NART
    if 'nart-score' in df.columns:
        cognitive_df['NART_T'] = df['nart-score']
        log("[SUCCESS] NART score extracted")
    else:
        raise ValueError("Column 'nart-score' not found")

    # RPM
    if 'rpm-score' in df.columns:
        cognitive_df['RPM_T'] = df['rpm-score']
        log("[SUCCESS] RPM score extracted")
    else:
        raise ValueError("Column 'rpm-score' not found")

    # Check for missing values
    log("\n[CHECK] Checking for missing values...")
    missing = cognitive_df.isnull().sum()
    for col, n_missing in missing.items():
        if n_missing > 0:
            log(f"  - {col}: {n_missing} missing values")

    # Summary statistics
    log("\n[SUMMARY] Cognitive test scores (raw):")
    for col in ['RAVLT_T', 'RAVLT_DR_T', 'RAVLT_Pct_Ret', 'BVMT_T', 'BVMT_Pct_Ret', 'NART_T', 'RPM_T']:
        if col in cognitive_df.columns:
            mean_val = cognitive_df[col].mean()
            std_val = cognitive_df[col].std()
            min_val = cognitive_df[col].min()
            max_val = cognitive_df[col].max()
            log(f"  - {col}: M={mean_val:.1f}, SD={std_val:.1f}, Range=[{min_val:.0f}, {max_val:.0f}]")

    # Save output
    output_path = RQ_DIR / "data" / "step01_cognitive_tests.csv"
    output_path.parent.mkdir(exist_ok=True)
    cognitive_df.to_csv(output_path, index=False)
    log(f"\n[SAVE] Saved cognitive tests to {output_path}")
    log(f"[INFO] Shape: {cognitive_df.shape}")
    log(f"[INFO] Columns: {cognitive_df.columns.tolist()}")

    log("\n[SUCCESS] Step 01 complete (v2 - ceiling fix + percent retention)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
