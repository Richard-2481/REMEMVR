#!/usr/bin/env python3
"""
Step 01: Extract and Prepare Cognitive Test Data (v2 - Percent Retention)
RQ: ch7/7.1.2
Purpose: Extract cognitive test scores from dfnonvr.csv and standardize to T-scores
Output: results/ch7/7.1.2/data/step01_cognitive_tests.csv

v2 CHANGES (2026-03-22):
1. RAVLT ceiling fix: participants with unadministered trials stored as 0.
   Substitutes 15 where trial N == 0 and trial N-1 >= 14 (ceiling performance).
2. BVMT Total recomputed explicitly from sum(trials 1-3) instead of pre-computed column.
3. Added RAVLT Percent Retention (Delayed Recall / best available trial x 100).
4. Added BVMT Percent Retained (from pre-computed column in dfnonvr.csv).
5. Output column names match downstream expectations (RAVLT_T, BVMT_T, not RAVLT_Total_T).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import utilities
sys.path.insert(0, str(PROJECT_ROOT / "results" / "ch7"))
from column_name_fix import get_correct_column_name, COLUMN_MAPPINGS
from missing_data_handler import (
    analyze_missing_pattern,
    create_missing_data_report,
    handle_missing_data,
    document_excluded_participants
)

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_cognitive_tests.csv"
MISSING_REPORT_FILE = RQ_DIR / "data" / "step01_missing_data_report.txt"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def compute_t_score(raw_scores):
    """Convert raw scores to T-scores (M=50, SD=10)."""
    if len(raw_scores) == 0:
        return []
    mean = np.nanmean(raw_scores)
    std = np.nanstd(raw_scores)
    if std == 0:
        return np.full_like(raw_scores, 50.0)
    return 50 + 10 * (raw_scores - mean) / std

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    Logic: If a participant scored >= 14 on trial N-1 and trial N == 0,
    trial N was not administered (ceiling). Substitute 15.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0

    for idx in df.index:
        for i in range(1, 5):  # Check trials 2,3,4,5 (index 1,2,3,4)
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

if __name__ == "__main__":
    try:
        log("[START] Step 01: Extract and Prepare Cognitive Test Data - v2 (Percent Retention)")
        log("[INFO] Includes RAVLT ceiling fix + percent retention predictors")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        log(f"[SETUP] Output will be saved to: {OUTPUT_FILE}")

        # =========================================================================
        # STEP 1: Load participant data
        # =========================================================================
        log("\n[DATA] Loading participant data from dfnonvr.csv...")

        data_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        cognitive_df = pd.read_csv(data_path)
        log(f"[DATA] Loaded data for {len(cognitive_df)} participants")

        # =========================================================================
        # STEP 2: Apply RAVLT ceiling fix BEFORE computing totals
        # =========================================================================
        log("\n[CEILING] Applying RAVLT ceiling fix (0 -> 15 for unadministered trials)...")
        cognitive_df = fix_ravlt_ceiling(cognitive_df, log)

        # =========================================================================
        # STEP 3: Extract and compute cognitive test scores
        # =========================================================================
        log("\n[PROCESS] Extracting cognitive test scores...")

        extracted_data = pd.DataFrame()
        extracted_data['UID'] = cognitive_df['UID'].astype(str)

        # RAVLT Total: Sum trials 1-5 (after ceiling fix)
        ravlt_trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
        missing_cols = [c for c in ravlt_trial_cols if c not in cognitive_df.columns]
        if missing_cols:
            raise ValueError(f"Missing RAVLT trial columns: {missing_cols}")

        extracted_data['RAVLT_Total'] = cognitive_df[ravlt_trial_cols].sum(axis=1)
        log(f"[SUCCESS] RAVLT Total (ceiling-fixed): "
            f"M={extracted_data['RAVLT_Total'].mean():.1f}, SD={extracted_data['RAVLT_Total'].std():.1f}")

        # RAVLT Delayed Recall
        extracted_data['RAVLT_DR'] = cognitive_df['ravlt-delayed-recall-score']
        log(f"[SUCCESS] RAVLT Delayed Recall extracted")

        # RAVLT Percent Retention (new predictor)
        extracted_data['RAVLT_Pct_Ret'] = compute_ravlt_percent_retention(cognitive_df, log)

        # BVMT Total: Explicitly sum trials 1-3 (not pre-computed column)
        bvmt_trial_cols = [f'bvmt-trial-{i}-score' for i in range(1, 4)]
        missing_cols = [c for c in bvmt_trial_cols if c not in cognitive_df.columns]
        if missing_cols:
            raise ValueError(f"Missing BVMT trial columns: {missing_cols}")

        extracted_data['BVMT_Total'] = cognitive_df[bvmt_trial_cols].sum(axis=1)
        log(f"[SUCCESS] BVMT Total (sum trials 1-3): "
            f"M={extracted_data['BVMT_Total'].mean():.1f}, SD={extracted_data['BVMT_Total'].std():.1f}")

        # BVMT Percent Retained (new predictor, pre-computed in CSV)
        if 'bvmt-percent-retained' not in cognitive_df.columns:
            raise ValueError("Column 'bvmt-percent-retained' not found in dfnonvr.csv")
        extracted_data['BVMT_Pct_Ret'] = cognitive_df['bvmt-percent-retained']
        log(f"[SUCCESS] BVMT Percent Retained: "
            f"M={extracted_data['BVMT_Pct_Ret'].mean():.1f}%, SD={extracted_data['BVMT_Pct_Ret'].std():.1f}%")

        # NART
        if 'nart-score' in cognitive_df.columns:
            extracted_data['NART_Score'] = cognitive_df['nart-score']
            log(f"[SUCCESS] NART score extracted")
        else:
            log("[ERROR] NART score column not found")
            extracted_data['NART_Score'] = np.nan

        # RPM
        if 'rpm-score' in cognitive_df.columns:
            extracted_data['RPM_Score'] = cognitive_df['rpm-score']
            log(f"[SUCCESS] RPM score extracted")
        else:
            log("[ERROR] RPM score column not found")
            extracted_data['RPM_Score'] = np.nan

        # =========================================================================
        # STEP 4: Analyze missing data BEFORE processing
        # =========================================================================
        log("\n[MISSING DATA] Analyzing missing data patterns...")

        key_columns = ['RAVLT_Total', 'RAVLT_DR', 'RAVLT_Pct_Ret',
                       'BVMT_Total', 'BVMT_Pct_Ret', 'NART_Score', 'RPM_Score']

        missing_report = create_missing_data_report(extracted_data, key_columns)
        log("\n" + missing_report)

        with open(MISSING_REPORT_FILE, 'w') as f:
            f.write(missing_report)
        log(f"\n[SAVE] Missing data report saved to: {MISSING_REPORT_FILE}")

        complete_data = extracted_data.dropna(subset=key_columns)
        n_excluded = len(extracted_data) - len(complete_data)

        if n_excluded > 0:
            log(f"\n[EXCLUSION] {n_excluded} participants excluded due to missing data")

            demo_cols = ['age', 'sex', 'education']
            for col in demo_cols:
                if col in cognitive_df.columns:
                    extracted_data[col] = cognitive_df[col]

            comparison = document_excluded_participants(
                extracted_data, complete_data, 'UID', demo_cols
            )

            if not comparison.empty:
                log("\n[EXCLUSION] Comparison of included vs excluded participants:")
                log(comparison.to_string())
                comparison_file = RQ_DIR / "data" / "step01_exclusion_comparison.csv"
                comparison.to_csv(comparison_file, index=False)
                log(f"[SAVE] Exclusion comparison saved to: {comparison_file}")

        # =========================================================================
        # STEP 5: Convert to T-scores
        # =========================================================================
        log("\n[PROCESS] Converting raw scores to T-scores (M=50, SD=10)...")

        t_score_df = complete_data.copy()

        # T-score conversion for all measures
        raw_to_t = {
            'RAVLT_Total': 'RAVLT_T',
            'RAVLT_DR': 'RAVLT_DR_T',
            'RAVLT_Pct_Ret': 'RAVLT_Pct_Ret_T',
            'BVMT_Total': 'BVMT_T',
            'BVMT_Pct_Ret': 'BVMT_Pct_Ret_T',
            'NART_Score': 'NART_T',
            'RPM_Score': 'RPM_T',
        }

        for raw_col, t_col in raw_to_t.items():
            t_score_df[t_col] = compute_t_score(t_score_df[raw_col].values)

        # Report T-score statistics
        log("\n[T-SCORES] Summary statistics:")
        for t_col in raw_to_t.values():
            mean = t_score_df[t_col].mean()
            std = t_score_df[t_col].std()
            log(f"  {t_col:20} M={mean:.1f}, SD={std:.1f}")

        # =========================================================================
        # STEP 6: Save final dataset
        # =========================================================================

        # Output columns: names match downstream expectations
        final_columns = ['UID', 'RAVLT_T', 'RAVLT_DR_T', 'RAVLT_Pct_Ret_T',
                         'BVMT_T', 'BVMT_Pct_Ret_T', 'NART_T', 'RPM_T']
        final_df = t_score_df[final_columns]

        final_df.to_csv(OUTPUT_FILE, index=False)
        log(f"\n[SAVE] Saved cognitive test T-scores to: {OUTPUT_FILE}")
        log(f"[INFO] Final dataset: {final_df.shape[0]} participants, {final_df.shape[1]} columns")
        log(f"[INFO] Columns: {final_df.columns.tolist()}")

        # =========================================================================
        # STEP 7: Summary
        # =========================================================================
        log("\n" + "=" * 60)
        log("SUMMARY")
        log("=" * 60)
        log(f"Total participants loaded: {len(cognitive_df)}")
        log(f"Participants with complete data: {len(complete_data)}")
        log(f"Participants excluded: {n_excluded}")
        log(f"Exclusion rate: {(n_excluded / len(cognitive_df) * 100):.1f}%")
        log("\nCeiling fixes applied for participants with trial N == 0 and trial N-1 >= 14")
        log("New predictors: RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T")
        log("\n[SUCCESS] Step 01 complete - v2 with ceiling fix + percent retention")

    except Exception as e:
        log(f"\n[ERROR] Script failed: {str(e)}")
        import traceback
        log(f"[TRACEBACK]\n{traceback.format_exc()}")
        sys.exit(1)
