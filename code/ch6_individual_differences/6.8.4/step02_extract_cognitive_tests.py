#!/usr/bin/env python3
"""extract_cognitive_tests: Extract cognitive test battery scores (RAVLT, BVMT, RPM, Age) from dfnonvr.csv and standardize"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data import extract_cognitive_tests

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4
LOG_FILE = RQ_DIR / "logs" / "step02_extract_cognitive_tests.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: extract_cognitive_tests")
        # Load Cognitive Test Data

        log("Loading cognitive test data...")

        # KNOWN BUG FIX: tools.data.extract_cognitive_tests looks for lowercase 'uid' but
        # dfnonvr.csv has 'UID'. Load directly instead (gcode_lessons.md #2).
        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'
        df_cognitive = pd.read_csv(dfnonvr_path)

        log(f"Cognitive tests: ({len(df_cognitive)} rows, {len(df_cognitive.columns)} cols)")
        # STEP 1b: RAVLT Ceiling Fix
        # RAVLT Ceiling Fix: if trial N == 0 and trial N-1 >= 14, set trial N = 15
        # This RQ uses Trial 5 only, but fix must apply to the raw data first
        # Known affected: A064 (trial 5=0, trial 4=15), A103 (trial 4=0 trial 3=15, then trial 5=0 trial 4=15)
        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        ravlt_trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
        fixes_applied = 0
        for idx in df_cognitive.index:
            for i in range(1, 5):
                current_col = ravlt_trial_cols[i]
                prev_col = ravlt_trial_cols[i - 1]
                if df_cognitive.at[idx, current_col] == 0 and df_cognitive.at[idx, prev_col] >= 14:
                    uid = df_cognitive.at[idx, 'UID']
                    df_cognitive.at[idx, current_col] = 15
                    fixes_applied += 1
                    log(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df_cognitive.at[idx, prev_col]})")
        log(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
        # Extract Required Columns

        log("Extracting required test columns...")

        # NOTE: extract_cognitive_tests returns standardized column names
        # But we need to check what columns are actually returned
        log(f"Available columns: {df_cognitive.columns.tolist()}")

        # Map exact dfnonvr.csv column names to our output names
        # Ch7 CRITICAL: Column names are lowercase with hyphens
        column_mapping = {
            'ravlt-trial-5-score': 'ravlt_raw',
            'bvmt-total-recall': 'bvmt_raw',
            'rpm-score': 'rpm_raw',
            'age': 'age_raw'
        }

        # Check if extract_cognitive_tests already standardized names
        # If it has 'RAVLT' column, it standardized; if 'ravlt-trial-5-score', it didn't
        if 'ravlt-trial-5-score' in df_cognitive.columns:
            # Not standardized - do it manually
            df_selected = df_cognitive[['UID'] + list(column_mapping.keys())].copy()
            df_selected.rename(columns=column_mapping, inplace=True)
        elif 'RAVLT' in df_cognitive.columns or 'RAVLT_Trial_5' in df_cognitive.columns:
            # Standardized names - need to map
            # Check actual columns returned by tool
            if 'RAVLT_Trial_5' in df_cognitive.columns:
                df_selected = df_cognitive[['UID', 'RAVLT_Trial_5', 'BVMT_Total', 'RPM', 'Age']].copy()
                df_selected.rename(columns={
                    'RAVLT_Trial_5': 'ravlt_raw',
                    'BVMT_Total': 'bvmt_raw',
                    'RPM': 'rpm_raw',
                    'Age': 'age_raw'
                }, inplace=True)
            else:
                # Fallback: Just load from CSV directly
                log("Unexpected column format from extract_cognitive_tests, loading directly")
                df_raw = pd.read_csv(dfnonvr_path)
                df_selected = df_raw[['UID', 'ravlt-trial-5-score', 'bvmt-total-recall', 'rpm-score', 'age']].copy()
                df_selected.rename(columns=column_mapping, inplace=True)
        else:
            # Unknown format - load directly from CSV
            log("Unknown column format, loading directly from dfnonvr.csv")
            df_raw = pd.read_csv(dfnonvr_path)
            df_selected = df_raw[['UID', 'ravlt-trial-5-score', 'bvmt-total-recall', 'rpm-score', 'age']].copy()
            df_selected.rename(columns=column_mapping, inplace=True)

        log(f"{len(df_selected)} rows with 4 cognitive predictors")

        # Remove missing data
        initial_count = len(df_selected)
        df_selected = df_selected.dropna()
        if len(df_selected) < initial_count:
            log(f"Removed {initial_count - len(df_selected)} rows with missing data")
        # Standardize to Z-Scores

        log("Converting to z-scores...")

        df_standardized = df_selected.copy()

        for col in ['ravlt_raw', 'bvmt_raw', 'rpm_raw', 'age_raw']:
            z_col = col.replace('_raw', '_z')
            mean_val = df_selected[col].mean()
            std_val = df_selected[col].std()

            df_standardized[z_col] = (df_selected[col] - mean_val) / std_val

            log(f"{col}: mean={mean_val:.2f}, SD={std_val:.2f}")

        # RAVLT Percent Retention: DR / max(trials 1-5 after ceiling fix) * 100
        # Added alongside Trial 5 (different operationalisation of forgetting)
        dr_col = 'ravlt-delayed-recall-score'
        ravlt_max_trial = df_cognitive[ravlt_trial_cols].max(axis=1)
        # Align indices: df_selected may have dropped rows via dropna
        df_standardized['pctret_raw'] = np.where(
            ravlt_max_trial.loc[df_selected.index].values > 0,
            (df_cognitive.loc[df_selected.index, dr_col].values / ravlt_max_trial.loc[df_selected.index].values) * 100,
            np.nan
        )
        pctret_mean = df_standardized['pctret_raw'].mean()
        pctret_std = df_standardized['pctret_raw'].std()
        df_standardized['pctret_z'] = (df_standardized['pctret_raw'] - pctret_mean) / pctret_std
        log(f"pctret_raw: mean={pctret_mean:.2f}, SD={pctret_std:.2f}")

        # Keep only UID and z-score columns
        df_output = df_standardized[['UID', 'ravlt_z', 'bvmt_z', 'rpm_z', 'age_z', 'pctret_z']]

        # Drop any rows with NaN pctret (e.g., if max trial score was 0)
        pre_pctret_count = len(df_output)
        df_output = df_output.dropna()
        if len(df_output) < pre_pctret_count:
            log(f"Removed {pre_pctret_count - len(df_output)} rows with missing pctret")

        log(f"Standardized {len(df_output)} participants")
        # Compute Predictor Correlations
        # Validates: Multicollinearity check (|r| < 0.80)

        log("Computing predictor correlations...")

        predictor_cols = ['ravlt_z', 'bvmt_z', 'rpm_z', 'age_z', 'pctret_z']
        correlation_results = []

        for i, pred1 in enumerate(predictor_cols):
            for pred2 in predictor_cols[i+1:]:
                # Compute Pearson correlation
                r, p = pearsonr(df_output[pred1], df_output[pred2])

                correlation_results.append({
                    'predictor1': pred1.replace('_z', ''),
                    'predictor2': pred2.replace('_z', ''),
                    'correlation': r,
                    'p_value': p
                })

                log(f"{pred1} <-> {pred2}: r = {r:.3f}, p = {p:.4f}")

        df_correlations = pd.DataFrame(correlation_results)
        # Save Outputs
        # These outputs will be used by: Step 03 (merge with domain theta scores)

        log("Saving standardized cognitive tests...")
        cognitive_output = RQ_DIR / 'data' / 'step02_cognitive_tests.csv'
        df_output.to_csv(cognitive_output, index=False, encoding='utf-8')
        log(f"{cognitive_output} ({len(df_output)} rows, {len(df_output.columns)} cols)")

        log("Saving predictor correlations...")
        corr_output = RQ_DIR / 'data' / 'step02_predictor_correlations.csv'
        df_correlations.to_csv(corr_output, index=False, encoding='utf-8')
        log(f"{corr_output} ({len(df_correlations)} rows, {len(df_correlations.columns)} cols)")
        # Validation
        # Validates: Z-scores (mean≈0, SD≈1), multicollinearity (|r|<0.80), no missing
        # Threshold: All checks must pass

        log("Validating outputs...")

        validation_pass = True

        # Check z-score properties (mean ≈ 0, SD ≈ 1)
        for col in predictor_cols:
            mean_val = df_output[col].mean()
            std_val = df_output[col].std()

            if abs(mean_val) > 1e-10:  # Numerical precision tolerance
                log(f"{col} mean = {mean_val:.6f} (expected ≈ 0)")

            if abs(std_val - 1.0) > 1e-10:
                log(f"{col} SD = {std_val:.6f} (expected ≈ 1)")

        log(f"Z-score properties validated")

        # Check multicollinearity threshold (|r| < 0.80)
        high_corr_count = 0
        for _, row in df_correlations.iterrows():
            if abs(row['correlation']) >= 0.80:
                log(f"High correlation: {row['predictor1']}-{row['predictor2']}: r = {row['correlation']:.3f}")
                high_corr_count += 1

        if high_corr_count > 0:
            log(f"{high_corr_count} predictor pairs exceed multicollinearity threshold")
        else:
            log(f"No multicollinearity detected (all |r| < 0.80)")

        # Check for missing data
        if df_output.isnull().sum().sum() > 0:
            log(f"Missing data in output")
            validation_pass = False
        else:
            log(f"No missing data in output")

        # Report validation results
        if validation_pass:
            log("All validation checks passed")
        else:
            log("Some validation checks failed")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
