#!/usr/bin/env python3
"""extract_ravlt_scores: Extract RAVLT test scores from dfnonvr.csv and compute alternative scoring metrics:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_ravlt_scores.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_ravlt_scores.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# RAVLT Ceiling Fix

def fix_ravlt_ceiling(df, log_fn):
    """If trial N == 0 and trial N-1 >= 14, set trial N = 15.

    Addresses scoring artifact where participants at ceiling (14-15) on one trial
    occasionally record 0 on the next trial due to administration/recording error.
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

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract RAVLT Scores")
        # Load dfnonvr.csv
        log("Loading dfnonvr.csv...")

        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        df = pd.read_csv(dfnonvr_path)

        log(f"dfnonvr.csv ({len(df)} participants, {len(df.columns)} columns)")
        # STEP 1b: Apply RAVLT Ceiling Fix
        log("[CEILING FIX] Applying ceiling fix to raw trial scores...")
        df = fix_ravlt_ceiling(df, log)
        # Extract RAVLT Columns
        # Use exact column names from DATA_DICTIONARY.md (avoid bug #2)
        log("Extracting RAVLT columns...")

        ravlt_data = pd.DataFrame()
        ravlt_data['UID'] = df['UID']

        # Trials 1-5 (CRITICAL: Do NOT include distraction trial - avoid bug #6)
        ravlt_data['ravlt_trial_1'] = df['ravlt-trial-1-score']
        ravlt_data['ravlt_trial_2'] = df['ravlt-trial-2-score']
        ravlt_data['ravlt_trial_3'] = df['ravlt-trial-3-score']
        ravlt_data['ravlt_trial_4'] = df['ravlt-trial-4-score']
        ravlt_data['ravlt_trial_5'] = df['ravlt-trial-5-score']

        # Delayed recall and recognition
        ravlt_data['ravlt_delayed'] = df['ravlt-delayed-recall-score']
        ravlt_data['ravlt_recognition'] = df['ravlt-recognition-hits-']

        log(f"RAVLT columns ({len(ravlt_data)} participants)")
        # Compute Derived Metrics
        log("Computing alternative scoring metrics...")

        # Total: Sum of trials 1-5 only (NOT distraction trial)
        ravlt_data['Total'] = (
            ravlt_data['ravlt_trial_1'] +
            ravlt_data['ravlt_trial_2'] +
            ravlt_data['ravlt_trial_3'] +
            ravlt_data['ravlt_trial_4'] +
            ravlt_data['ravlt_trial_5']
        )

        # Learning: Raw change from T1 to T5
        ravlt_data['Learning'] = ravlt_data['ravlt_trial_5'] - ravlt_data['ravlt_trial_1']

        # LearningSlope: Proportional change (avoid division by zero)
        # If T1=0, set LearningSlope to NaN (can't compute proportional change)
        ravlt_data['LearningSlope'] = np.where(
            ravlt_data['ravlt_trial_1'] == 0,
            np.nan,
            (ravlt_data['ravlt_trial_5'] - ravlt_data['ravlt_trial_1']) / ravlt_data['ravlt_trial_1']
        )

        # Forgetting: Decline from T5 to delayed recall
        ravlt_data['Forgetting'] = ravlt_data['ravlt_trial_5'] - ravlt_data['ravlt_delayed']

        # Recognition: Use recognition hits directly
        ravlt_data['Recognition'] = ravlt_data['ravlt_recognition']

        # PctRet: Delayed Recall / Best Learning Trial * 100
        best_trial = ravlt_data[['ravlt_trial_1', 'ravlt_trial_2', 'ravlt_trial_3', 'ravlt_trial_4', 'ravlt_trial_5']].max(axis=1)
        ravlt_data['PctRet'] = np.where(best_trial > 0, ravlt_data['ravlt_delayed'] / best_trial * 100, np.nan)

        log("Total, Learning, LearningSlope, Forgetting, Recognition, PctRet")
        # Standardize Metrics (Z-scores)
        log("Computing z-scores for regression models...")

        ravlt_data['Total_z'] = (ravlt_data['Total'] - ravlt_data['Total'].mean()) / ravlt_data['Total'].std()
        ravlt_data['Learning_z'] = (ravlt_data['Learning'] - ravlt_data['Learning'].mean()) / ravlt_data['Learning'].std()
        ravlt_data['LearningSlope_z'] = (ravlt_data['LearningSlope'] - ravlt_data['LearningSlope'].mean()) / ravlt_data['LearningSlope'].std()
        ravlt_data['Forgetting_z'] = (ravlt_data['Forgetting'] - ravlt_data['Forgetting'].mean()) / ravlt_data['Forgetting'].std()
        ravlt_data['Recognition_z'] = (ravlt_data['Recognition'] - ravlt_data['Recognition'].mean()) / ravlt_data['Recognition'].std()
        ravlt_data['PctRet_z'] = (ravlt_data['PctRet'] - ravlt_data['PctRet'].mean()) / ravlt_data['PctRet'].std()

        log("All metrics converted to z-scores")
        # Validation Checks
        log("Running validation checks...")

        # Check trial score ranges (0-15)
        trial_cols = ['ravlt_trial_1', 'ravlt_trial_2', 'ravlt_trial_3', 'ravlt_trial_4', 'ravlt_trial_5', 'ravlt_delayed', 'ravlt_recognition']
        for col in trial_cols:
            min_val = ravlt_data[col].min()
            max_val = ravlt_data[col].max()
            if min_val < 0 or max_val > 15:
                log(f"{col} out of expected range: [{min_val}, {max_val}] (expected [0, 15])")
            else:
                log(f"{col} within range [{min_val}, {max_val}]")

        # Check z-scores (approximately -3 to 3)
        z_cols = ['Total_z', 'Learning_z', 'LearningSlope_z', 'Forgetting_z', 'Recognition_z', 'PctRet_z']
        for col in z_cols:
            min_z = ravlt_data[col].min()
            max_z = ravlt_data[col].max()
            log(f"{col} range: [{min_z:.2f}, {max_z:.2f}]")

        # Check for missing data
        n_missing = ravlt_data.isnull().sum().sum()
        if n_missing > 0:
            log(f"{n_missing} missing values detected")
            log(f"Missing by column:\n{ravlt_data.isnull().sum()[ravlt_data.isnull().sum() > 0]}")
        else:
            log("No missing values")
        # Save Output
        log("Saving RAVLT scores...")

        ravlt_data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        log(f"{OUTPUT_FILE} ({len(ravlt_data)} participants, {len(ravlt_data.columns)} columns)")
        # Summary Statistics
        log("Descriptive statistics:")
        log(f"  Total: M={ravlt_data['Total'].mean():.2f}, SD={ravlt_data['Total'].std():.2f}, Range=[{ravlt_data['Total'].min():.0f}, {ravlt_data['Total'].max():.0f}]")
        log(f"  Learning: M={ravlt_data['Learning'].mean():.2f}, SD={ravlt_data['Learning'].std():.2f}")
        log(f"  LearningSlope: M={ravlt_data['LearningSlope'].mean():.2f}, SD={ravlt_data['LearningSlope'].std():.2f}")
        log(f"  Forgetting: M={ravlt_data['Forgetting'].mean():.2f}, SD={ravlt_data['Forgetting'].std():.2f}")
        log(f"  Recognition: M={ravlt_data['Recognition'].mean():.2f}, SD={ravlt_data['Recognition'].std():.2f}")
        log(f"  PctRet: M={ravlt_data['PctRet'].mean():.2f}, SD={ravlt_data['PctRet'].std():.2f}")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
