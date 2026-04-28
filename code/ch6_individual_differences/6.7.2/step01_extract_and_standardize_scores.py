#!/usr/bin/env python3
"""extract_and_standardize_scores: Load theta scores from Ch5 5.1.1 and RAVLT Total scores from dfnonvr.csv,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import traceback
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_and_standardize_scores.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract and Standardize Scores")
        # Load Validation Results to Get File Paths
        log("Reading dependency validation results...")

        validation_file = RQ_DIR / "data" / "step00_dependency_validation.txt"
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation_text = f.read()

        # Extract theta file path from validation text
        theta_match = re.search(r'Path:\s+([^\n]+theta[^\n]+\.csv)', validation_text)
        if theta_match:
            theta_rel_path = theta_match.group(1).strip()
            theta_path = PROJECT_ROOT / theta_rel_path
        else:
            # Fallback to default
            theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"

        log(f"Theta file path: {theta_path.relative_to(PROJECT_ROOT)}")
        # Load Ch5 Theta Scores and Aggregate by UID
        # Rationale: RAVLT is single-session, need comparable structure

        log("Loading Ch5 theta scores...")
        theta_df = pd.read_csv(theta_path)

        # Find theta column (handle both 'Theta_All' and 'theta_all')
        theta_col = None
        for col in theta_df.columns:
            if col.lower() == 'theta_all':
                theta_col = col
                break

        if theta_col is None:
            raise ValueError(f"No theta column found in {theta_path}. Columns: {theta_df.columns.tolist()}")

        log(f"Using theta column: '{theta_col}'")
        log(f"Theta scores: {len(theta_df)} rows")

        # Aggregate theta scores by UID (mean across sessions)
        theta_agg = theta_df.groupby('UID')[theta_col].mean().reset_index()
        theta_agg.columns = ['UID', 'theta_raw']

        log(f"{len(theta_agg)} unique participants (mean theta across sessions)")
        # Load dfnonvr.csv and Calculate RAVLT Total
        # CRITICAL: Trials 1-5 only (from gcode_lessons.md Bug #6)

        log("Loading dfnonvr.csv for RAVLT data...")
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        dfnonvr = pd.read_csv(dfnonvr_path)

        log(f"dfnonvr.csv: {len(dfnonvr)} participants")

        # Calculate RAVLT total from trials 1-5 using exact column names
        ravlt_cols = [
            "ravlt-trial-1-score",
            "ravlt-trial-2-score",
            "ravlt-trial-3-score",
            "ravlt-trial-4-score",
            "ravlt-trial-5-score"
        ]

        # Verify columns exist (including delayed recall for percent retention)
        required_ravlt_cols = ravlt_cols + ['ravlt-delayed-recall-score']
        missing_ravlt = [col for col in required_ravlt_cols if col not in dfnonvr.columns]
        if missing_ravlt:
            raise ValueError(f"Missing RAVLT columns in dfnonvr.csv: {missing_ravlt}")
        # CEILING FIX: If trial N == 0 and trial N-1 >= 14, set trial N = 15
        # Rationale: A score of 0 following a near-perfect trial (>=14) is
        # almost certainly a data entry error or administration issue, not a
        # genuine score. Known affected: A064, A070, A077, A103 (7 total fixes).
        log("[CEILING FIX] Applying RAVLT ceiling fix before computing totals...")
        fixes_applied = 0
        for idx in dfnonvr.index:
            for i in range(1, 5):  # Check trials 2-5 (index 1-4)
                current_col = ravlt_cols[i]
                prev_col = ravlt_cols[i - 1]
                if dfnonvr.at[idx, current_col] == 0 and dfnonvr.at[idx, prev_col] >= 14:
                    uid = dfnonvr.at[idx, 'UID']
                    dfnonvr.at[idx, current_col] = 15
                    fixes_applied += 1
                    log(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {dfnonvr.at[idx, prev_col]})")
        log(f"[CEILING FIX] Total fixes applied: {fixes_applied}")

        # Calculate total (sum of trials 1-5, AFTER ceiling fix)
        dfnonvr['RAVLT_Total'] = dfnonvr[ravlt_cols].sum(axis=1)

        log(f"RAVLT Total from trials 1-5 (post-ceiling-fix)")
        log(f"RAVLT Total range: {dfnonvr['RAVLT_Total'].min():.1f} to {dfnonvr['RAVLT_Total'].max():.1f}")
        # PERCENT RETENTION: DR / max(trials 1-5) * 100
        # Captures forgetting rate rather than absolute learning capacity.
        # Uses max of trials 1-5 (post-ceiling-fix) as the best learning estimate.
        log("Computing RAVLT Percent Retention...")
        dfnonvr['RAVLT_max_trial'] = dfnonvr[ravlt_cols].max(axis=1)
        dfnonvr['RAVLT_Pct_Ret'] = (dfnonvr['ravlt-delayed-recall-score'] / dfnonvr['RAVLT_max_trial']) * 100

        log(f"RAVLT Percent Retention = DR / max(trials 1-5) * 100")
        log(f"RAVLT Pct Ret range: {dfnonvr['RAVLT_Pct_Ret'].min():.1f}% to {dfnonvr['RAVLT_Pct_Ret'].max():.1f}%")
        log(f"RAVLT Pct Ret M={dfnonvr['RAVLT_Pct_Ret'].mean():.1f}%, SD={dfnonvr['RAVLT_Pct_Ret'].std():.1f}%")
        # Merge REMEMVR and RAVLT Data
        log("Merging REMEMVR theta scores with RAVLT scores...")

        merged = theta_agg.merge(
            dfnonvr[['UID', 'RAVLT_Total', 'RAVLT_Pct_Ret']],
            on='UID',
            how='inner'
        )

        merged['RAVLT_raw'] = merged['RAVLT_Total']
        merged['RAVLT_Pct_Ret_raw'] = merged['RAVLT_Pct_Ret']

        log(f"{len(merged)} participants with both REMEMVR and RAVLT scores")

        # Check for participants without matches
        theta_only = set(theta_agg['UID']) - set(merged['UID'])
        ravlt_only = set(dfnonvr['UID']) - set(merged['UID'])

        if theta_only:
            log(f"{len(theta_only)} participants in theta data but not in RAVLT data")
        if ravlt_only:
            log(f"{len(ravlt_only)} participants in RAVLT data but not in theta data")
        # Standardize Both Measures to Z-Scores
        # Method: Sample-based z-scores (M=0, SD=1)

        log("Converting raw scores to z-scores...")

        merged['REMEMVR_z'] = stats.zscore(merged['theta_raw'])
        merged['RAVLT_z'] = stats.zscore(merged['RAVLT_raw'])
        merged['RAVLT_Pct_Ret_z'] = stats.zscore(merged['RAVLT_Pct_Ret_raw'])

        log(f"REMEMVR_z: M={merged['REMEMVR_z'].mean():.4f}, SD={merged['REMEMVR_z'].std():.4f}")
        log(f"RAVLT_z: M={merged['RAVLT_z'].mean():.4f}, SD={merged['RAVLT_z'].std():.4f}")
        log(f"RAVLT_Pct_Ret_z: M={merged['RAVLT_Pct_Ret_z'].mean():.4f}, SD={merged['RAVLT_Pct_Ret_z'].std():.4f}")
        # Bootstrap 95% CIs for Standardization Parameters
        # Method: Participant-level resampling (1000 iterations, seed=42)

        log("Computing 95% CIs for standardization parameters...")

        n_iterations = 1000
        np.random.seed(42)

        rememvr_means = []
        rememvr_sds = []
        ravlt_means = []
        ravlt_sds = []
        pctret_means = []
        pctret_sds = []

        for i in range(n_iterations):
            # Resample participants with replacement
            boot_indices = np.random.choice(len(merged), size=len(merged), replace=True)
            boot_sample = merged.iloc[boot_indices]

            rememvr_means.append(boot_sample['theta_raw'].mean())
            rememvr_sds.append(boot_sample['theta_raw'].std())
            ravlt_means.append(boot_sample['RAVLT_raw'].mean())
            ravlt_sds.append(boot_sample['RAVLT_raw'].std())
            pctret_means.append(boot_sample['RAVLT_Pct_Ret_raw'].mean())
            pctret_sds.append(boot_sample['RAVLT_Pct_Ret_raw'].std())

        # Calculate percentile CIs
        rememvr_mean_ci = np.percentile(rememvr_means, [2.5, 97.5])
        rememvr_sd_ci = np.percentile(rememvr_sds, [2.5, 97.5])
        ravlt_mean_ci = np.percentile(ravlt_means, [2.5, 97.5])
        ravlt_sd_ci = np.percentile(ravlt_sds, [2.5, 97.5])
        pctret_mean_ci = np.percentile(pctret_means, [2.5, 97.5])
        pctret_sd_ci = np.percentile(pctret_sds, [2.5, 97.5])

        log(f"REMEMVR mean 95% CI: [{rememvr_mean_ci[0]:.4f}, {rememvr_mean_ci[1]:.4f}]")
        log(f"REMEMVR SD 95% CI: [{rememvr_sd_ci[0]:.4f}, {rememvr_sd_ci[1]:.4f}]")
        log(f"RAVLT mean 95% CI: [{ravlt_mean_ci[0]:.2f}, {ravlt_mean_ci[1]:.2f}]")
        log(f"RAVLT SD 95% CI: [{ravlt_sd_ci[0]:.2f}, {ravlt_sd_ci[1]:.2f}]")
        log(f"RAVLT Pct Ret mean 95% CI: [{pctret_mean_ci[0]:.2f}, {pctret_mean_ci[1]:.2f}]")
        log(f"RAVLT Pct Ret SD 95% CI: [{pctret_sd_ci[0]:.2f}, {pctret_sd_ci[1]:.2f}]")

        # Create standardization statistics DataFrame
        stats_df = pd.DataFrame({
            'measure': ['REMEMVR', 'RAVLT', 'RAVLT_Pct_Ret'],
            'mean': [merged['theta_raw'].mean(), merged['RAVLT_raw'].mean(), merged['RAVLT_Pct_Ret_raw'].mean()],
            'sd': [merged['theta_raw'].std(), merged['RAVLT_raw'].std(), merged['RAVLT_Pct_Ret_raw'].std()],
            'mean_ci_lower': [rememvr_mean_ci[0], ravlt_mean_ci[0], pctret_mean_ci[0]],
            'mean_ci_upper': [rememvr_mean_ci[1], ravlt_mean_ci[1], pctret_mean_ci[1]],
            'sd_ci_lower': [rememvr_sd_ci[0], ravlt_sd_ci[0], pctret_sd_ci[0]],
            'sd_ci_upper': [rememvr_sd_ci[1], ravlt_sd_ci[1], pctret_sd_ci[1]]
        })
        # Save Outputs
        log("Saving standardized scores...")

        # Final dataset with required columns
        final_df = merged[['UID', 'theta_raw', 'RAVLT_raw', 'RAVLT_Pct_Ret_raw', 'REMEMVR_z', 'RAVLT_z', 'RAVLT_Pct_Ret_z']]

        output_scores = RQ_DIR / "data" / "step01_standardized_scores.csv"
        final_df.to_csv(output_scores, index=False, encoding='utf-8')
        log(f"{output_scores.name} ({len(final_df)} rows)")

        output_stats = RQ_DIR / "data" / "step01_standardization_stats.csv"
        stats_df.to_csv(output_stats, index=False, encoding='utf-8')
        log(f"{output_stats.name} ({len(stats_df)} rows)")
        # Validate Standardization
        log("Running validate_standardization...")

        validation_result = validate_standardization(
            df=final_df,
            column_names=['REMEMVR_z', 'RAVLT_z', 'RAVLT_Pct_Ret_z'],
            tolerance=0.01
        )

        if validation_result.get('valid', False):
            log("Standardization validation successful")
        else:
            log(f"Standardization validation warnings: {validation_result}")

        # Check for extreme outliers (|z| > 4)
        rememvr_outliers = (final_df['REMEMVR_z'].abs() > 4).sum()
        ravlt_outliers = (final_df['RAVLT_z'].abs() > 4).sum()
        pctret_outliers = (final_df['RAVLT_Pct_Ret_z'].abs() > 4).sum()

        if rememvr_outliers > 0:
            log(f"{rememvr_outliers} REMEMVR extreme outliers (|z| > 4)")
        if ravlt_outliers > 0:
            log(f"{ravlt_outliers} RAVLT extreme outliers (|z| > 4)")
        if pctret_outliers > 0:
            log(f"{pctret_outliers} RAVLT Pct Ret extreme outliers (|z| > 4)")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
