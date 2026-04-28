#!/usr/bin/env python3
"""extract_and_merge_data: Extract theta_All scores and cognitive test scores (RAVLT, BVMT), then merge by UID"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.1
LOG_FILE = RQ_DIR / "logs" / "step01_extract_merge_data.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

def fix_ravlt_ceiling(df, log_fn):
    """If trial N == 0 and trial N-1 >= 14, set trial N = 15.

    Addresses ceiling effect where participants at maximum recall (14-15)
    on one trial show 0 on the next due to data recording issues.
    Known affected: A064, A070, A077, A103 (7 total fixes).
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


if __name__ == "__main__":
    try:
        log("Step 01: Extract and Merge Data")
        # Load Ch5 Theta Scores

        log("Loading Ch5 theta scores...")
        theta_path = PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv'

        df_theta = pd.read_csv(theta_path)
        log(f"Ch5 theta scores: {len(df_theta)} rows, {len(df_theta.columns)} cols")
        log(f"Columns: {df_theta.columns.tolist()}")
        log(f"Unique UIDs: {df_theta['UID'].nunique()}")
        log(f"Unique tests: {sorted(df_theta['test'].unique())}")
        # Aggregate Theta by UID

        log("Computing participant-level theta means...")

        # Keep UID as string (gcode_lessons.md: never convert to int)
        df_theta_agg = df_theta.groupby('UID', as_index=False)['Theta_All'].mean()
        df_theta_agg.rename(columns={'Theta_All': 'theta_all_mean'}, inplace=True)

        log(f"Theta means: {len(df_theta_agg)} participants")
        log(f"Theta range: [{df_theta_agg['theta_all_mean'].min():.3f}, {df_theta_agg['theta_all_mean'].max():.3f}]")
        log(f"Theta mean: {df_theta_agg['theta_all_mean'].mean():.3f}, SD: {df_theta_agg['theta_all_mean'].std():.3f}")
        # Load Cognitive Test Data

        log("Loading cognitive test data from dfnonvr.csv...")
        cognitive_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'

        df_cognitive = pd.read_csv(cognitive_path)
        log(f"dfnonvr.csv: {len(df_cognitive)} rows, {len(df_cognitive.columns)} cols")
        # Apply RAVLT Ceiling Fix
        # Fix ceiling effect BEFORE computing totals
        # If trial N == 0 and trial N-1 >= 14, set trial N = 15

        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        df_cognitive = fix_ravlt_ceiling(df_cognitive, log)
        # Extract and Compute Cognitive Totals
        # RAVLT: Sum trials 1-5 (exclude distraction/delay trials)
        # BVMT: Sum trials 1-3 (exclude delay trial)

        log("Computing RAVLT and BVMT totals...")

        # RAVLT Total: Explicit selection of trials 1-5 (gcode_lessons.md #6)
        ravlt_trial_cols = []
        for i in range(1, 6):
            col = f'ravlt-trial-{i}-score'
            if col in df_cognitive.columns:
                ravlt_trial_cols.append(col)
            else:
                log(f"Expected column not found: {col}")

        if len(ravlt_trial_cols) == 5:
            df_cognitive['ravlt_total'] = df_cognitive[ravlt_trial_cols].sum(axis=1)
            log(f"RAVLT total from columns: {ravlt_trial_cols}")
            log(f"RAVLT total range: [{df_cognitive['ravlt_total'].min():.0f}, {df_cognitive['ravlt_total'].max():.0f}]")
        else:
            log(f"Found only {len(ravlt_trial_cols)} RAVLT trial columns, expected 5")
            raise ValueError("Missing RAVLT trial columns")

        # BVMT Total: Sum trials 1-3
        bvmt_trial_cols = []
        for i in range(1, 4):
            col = f'bvmt-trial-{i}-score'
            if col in df_cognitive.columns:
                bvmt_trial_cols.append(col)
            else:
                log(f"Expected column not found: {col}")

        if len(bvmt_trial_cols) == 3:
            df_cognitive['bvmt_total'] = df_cognitive[bvmt_trial_cols].sum(axis=1)
            log(f"BVMT total from columns: {bvmt_trial_cols}")
            log(f"BVMT total range: [{df_cognitive['bvmt_total'].min():.0f}, {df_cognitive['bvmt_total'].max():.0f}]")
        else:
            log(f"Found only {len(bvmt_trial_cols)} BVMT trial columns, expected 3")
            raise ValueError("Missing BVMT trial columns")
        # Compute Percent Retention Scores
        # RAVLT: DR / max(trials 1-5 after ceiling fix) * 100
        # BVMT: Pre-computed column 'bvmt-percent-retained' from dfnonvr.csv

        log("Computing percent retention scores...")

        # RAVLT percent retention
        dr_col = 'ravlt-delayed-recall-score'
        if dr_col not in df_cognitive.columns:
            log(f"Expected column not found: {dr_col}")
            raise ValueError(f"Missing column: {dr_col}")

        ravlt_max_trial = df_cognitive[ravlt_trial_cols].max(axis=1)
        df_cognitive['ravlt_pct_ret'] = np.where(
            ravlt_max_trial > 0,
            (df_cognitive[dr_col] / ravlt_max_trial) * 100,
            0.0
        )
        log(f"RAVLT pct retention: mean={df_cognitive['ravlt_pct_ret'].mean():.1f}%, "
            f"range=[{df_cognitive['ravlt_pct_ret'].min():.1f}, {df_cognitive['ravlt_pct_ret'].max():.1f}]")

        # BVMT percent retention (pre-computed in dfnonvr.csv)
        bvmt_pct_col = 'bvmt-percent-retained'
        if bvmt_pct_col not in df_cognitive.columns:
            log(f"Expected column not found: {bvmt_pct_col}")
            raise ValueError(f"Missing column: {bvmt_pct_col}")

        df_cognitive['bvmt_pct_ret'] = df_cognitive[bvmt_pct_col]
        log(f"BVMT pct retention: mean={df_cognitive['bvmt_pct_ret'].mean():.1f}%, "
            f"range=[{df_cognitive['bvmt_pct_ret'].min():.1f}, {df_cognitive['bvmt_pct_ret'].max():.1f}]")

        # Select relevant columns
        df_cognitive_subset = df_cognitive[['UID', 'ravlt_total', 'bvmt_total', 'ravlt_pct_ret', 'bvmt_pct_ret']].copy()
        log(f"Cognitive data subset: {len(df_cognitive_subset)} participants")
        # Merge Datasets

        log("Merging theta and cognitive data by UID...")

        df_merged = pd.merge(
            df_theta_agg,
            df_cognitive_subset,
            on='UID',
            how='inner'
        )

        log(f"Final dataset: {len(df_merged)} rows, {len(df_merged.columns)} cols")
        log(f"Columns: {df_merged.columns.tolist()}")

        # Log merge statistics
        n_theta_only = len(df_theta_agg) - len(df_merged)
        n_cognitive_only = len(df_cognitive_subset) - len(df_merged)
        if n_theta_only > 0:
            log(f"{n_theta_only} participants with theta but no cognitive data")
        if n_cognitive_only > 0:
            log(f"{n_cognitive_only} participants with cognitive but no theta data")
        # Validate Merged Dataset
        # Check: no missing values, reasonable ranges, no duplicates

        log("Validating merged dataset...")

        # Check for duplicates
        if df_merged['UID'].duplicated().any():
            log("Duplicate UIDs found in merged dataset")
            raise ValueError("Duplicate UIDs in merged dataset")
        else:
            log("No duplicate UIDs")

        # Check for missing values
        missing_counts = df_merged.isnull().sum()
        if missing_counts.any():
            log(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        else:
            log("No missing values")

        # Validate ranges
        validation_passed = True

        # Theta range check
        theta_min = df_merged['theta_all_mean'].min()
        theta_max = df_merged['theta_all_mean'].max()
        if theta_min < -5 or theta_max > 5:
            log(f"Theta outside typical range: [{theta_min:.3f}, {theta_max:.3f}]")
            validation_passed = False
        else:
            log(f"Theta range valid: [{theta_min:.3f}, {theta_max:.3f}]")

        # RAVLT range check (0-75)
        ravlt_min = df_merged['ravlt_total'].min()
        ravlt_max = df_merged['ravlt_total'].max()
        if ravlt_min < 0 or ravlt_max > 75:
            log(f"RAVLT outside valid range: [{ravlt_min:.0f}, {ravlt_max:.0f}]")
            validation_passed = False
        else:
            log(f"RAVLT range valid: [{ravlt_min:.0f}, {ravlt_max:.0f}]")

        # BVMT range check (0-36)
        bvmt_min = df_merged['bvmt_total'].min()
        bvmt_max = df_merged['bvmt_total'].max()
        if bvmt_min < 0 or bvmt_max > 36:
            log(f"BVMT outside valid range: [{bvmt_min:.0f}, {bvmt_max:.0f}]")
            validation_passed = False
        else:
            log(f"BVMT range valid: [{bvmt_min:.0f}, {bvmt_max:.0f}]")

        # RAVLT pct retention range check (0-200)
        ravlt_pct_min = df_merged['ravlt_pct_ret'].min()
        ravlt_pct_max = df_merged['ravlt_pct_ret'].max()
        if ravlt_pct_min < 0 or ravlt_pct_max > 200:
            log(f"RAVLT pct retention outside valid range: [{ravlt_pct_min:.1f}, {ravlt_pct_max:.1f}]")
            validation_passed = False
        else:
            log(f"RAVLT pct retention range valid: [{ravlt_pct_min:.1f}, {ravlt_pct_max:.1f}]")

        # BVMT pct retention range check (0-200)
        bvmt_pct_min = df_merged['bvmt_pct_ret'].min()
        bvmt_pct_max = df_merged['bvmt_pct_ret'].max()
        if bvmt_pct_min < 0 or bvmt_pct_max > 200:
            log(f"BVMT pct retention outside valid range: [{bvmt_pct_min:.1f}, {bvmt_pct_max:.1f}]")
            validation_passed = False
        else:
            log(f"BVMT pct retention range valid: [{bvmt_pct_min:.1f}, {bvmt_pct_max:.1f}]")
        # Save Merged Dataset
        # Output: CSV with UID, theta_all_mean, ravlt_total, bvmt_total, ravlt_pct_ret, bvmt_pct_ret

        log("Saving merged dataset...")
        output_path = RQ_DIR / 'data' / 'step01_merged_dataset.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_merged.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"Output shape: {df_merged.shape}")

        # Summary statistics
        log("Dataset statistics:")
        log(f"  N participants: {len(df_merged)}")
        log(f"  Theta mean: {df_merged['theta_all_mean'].mean():.3f} (SD={df_merged['theta_all_mean'].std():.3f})")
        log(f"  RAVLT mean: {df_merged['ravlt_total'].mean():.1f} (SD={df_merged['ravlt_total'].std():.1f})")
        log(f"  BVMT mean: {df_merged['bvmt_total'].mean():.1f} (SD={df_merged['bvmt_total'].std():.1f})")
        log(f"  RAVLT pct ret mean: {df_merged['ravlt_pct_ret'].mean():.1f} (SD={df_merged['ravlt_pct_ret'].std():.1f})")
        log(f"  BVMT pct ret mean: {df_merged['bvmt_pct_ret'].mean():.1f} (SD={df_merged['bvmt_pct_ret'].std():.1f})")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
