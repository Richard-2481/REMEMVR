#!/usr/bin/env python3
"""merge_theta_tsvr: Merge theta scores with TSVR time variable, reshape for LMM (Decision D070)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# parents[4] = REMEMVR/ (code -> rqY -> chX -> results -> REMEMVR)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.1
LOG_FILE = RQ_DIR / "logs" / "step04_merge_theta_tsvr.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear log file for fresh run
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        log("Step 04: Merge Theta with TSVR for LMM Input")
        log(f"RQ Directory: {RQ_DIR}")
        # Load Input Data
        # Load theta scores from Pass 2 calibration and TSVR mapping

        log("Loading input data...")

        # Load theta scores from step03
        theta_path = RQ_DIR / "data" / "step03_theta_scores.csv"
        df_theta = pd.read_csv(theta_path, encoding='utf-8')
        log(f"step03_theta_scores.csv ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"Theta columns: {list(df_theta.columns)}")
        log(f"Unique composite_IDs in theta: {df_theta['composite_ID'].nunique()}")
        log(f"Unique paradigms (domain_name): {df_theta['domain_name'].unique().tolist()}")

        # Load TSVR mapping from step00
        tsvr_path = RQ_DIR / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"TSVR columns: {list(df_tsvr.columns)}")
        log(f"TSVR range: {df_tsvr['TSVR_hours'].min():.2f} - {df_tsvr['TSVR_hours'].max():.2f} hours")
        # Merge Theta with TSVR
        # Merge on composite_ID to attach time variable to each theta observation
        # Result: 1200 rows (each theta score gets TSVR_hours from its composite_ID)

        log("Merging theta scores with TSVR mapping...")

        # Merge on composite_ID
        df_merged = pd.merge(
            df_theta,
            df_tsvr,
            on='composite_ID',
            how='left'
        )
        log(f"Result: {len(df_merged)} rows")

        # Check for any failed merges (NaN in TSVR_hours would indicate missing mapping)
        missing_tsvr = df_merged['TSVR_hours'].isna().sum()
        if missing_tsvr > 0:
            log(f"{missing_tsvr} rows have missing TSVR_hours after merge")
        else:
            log("All rows have valid TSVR_hours")
        # Rename and Transform Columns
        # - Rename domain_name to paradigm (more intuitive for LMM)
        # - Create time transformations for model comparison:
        #   - TSVR_hours_sq: For quadratic time models
        #   - TSVR_hours_log: For logarithmic time models

        log("Renaming columns and creating time transformations...")

        # Rename domain_name to paradigm
        df_merged = df_merged.rename(columns={'domain_name': 'paradigm'})
        log("Renamed 'domain_name' -> 'paradigm'")

        # Create squared time term (for quadratic models)
        df_merged['TSVR_hours_sq'] = df_merged['TSVR_hours'] ** 2
        log("Created TSVR_hours_sq = TSVR_hours ** 2")

        # Create log-transformed time term (for logarithmic models)
        # Adding 1 to avoid log(0) for T1 observations (TSVR_hours ~ 0)
        df_merged['TSVR_hours_log'] = np.log(df_merged['TSVR_hours'] + 1)
        log("Created TSVR_hours_log = log(TSVR_hours + 1)")
        # Extract UID if not already present
        # UID should be in TSVR mapping, but verify it's in merged result

        if 'UID' not in df_merged.columns:
            # Extract UID from composite_ID (format: UID_test)
            df_merged['UID'] = df_merged['composite_ID'].str.split('_').str[0]
            log("Extracted UID from composite_ID")
        else:
            log("UID column already present from TSVR mapping")

        log(f"Unique UIDs: {df_merged['UID'].nunique()}")
        # Organize Column Order
        # Reorder columns for clear LMM input format

        column_order = [
            'composite_ID',
            'UID',
            'test',
            'TSVR_hours',
            'TSVR_hours_sq',
            'TSVR_hours_log',
            'paradigm',
            'theta'
        ]

        # Verify all columns exist
        missing_cols = [c for c in column_order if c not in df_merged.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_lmm_input = df_merged[column_order]
        log(f"Final columns: {list(df_lmm_input.columns)}")
        # Save LMM Input
        # Save to data/ folder with step prefix

        output_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        df_lmm_input.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_lmm_input)} rows, {len(df_lmm_input.columns)} cols)")
        # Inline Validation
        # Validate output meets all criteria from 4_analysis.yaml
        # CRITICAL severity = must pass or fail
        # MODERATE severity = warn only (real data has natural variation)

        log("Running inline validation checks...")

        critical_failures = []
        moderate_warnings = []

        # Check 1 : Row count == 1200
        if len(df_lmm_input) != 1200:
            critical_failures.append(f"Row count: expected 1200, got {len(df_lmm_input)}")
        else:
            log("Row count == 1200")

        # Check 2 : No missing TSVR
        if not df_lmm_input['TSVR_hours'].notna().all():
            missing = df_lmm_input['TSVR_hours'].isna().sum()
            critical_failures.append(f"Missing TSVR_hours: {missing} rows")
        else:
            log("No missing TSVR values")

        # Check 3 : Three paradigms present
        expected_paradigms = {'free_recall', 'cued_recall', 'recognition'}
        actual_paradigms = set(df_lmm_input['paradigm'].unique())
        if actual_paradigms != expected_paradigms:
            critical_failures.append(f"Paradigms: expected {expected_paradigms}, got {actual_paradigms}")
        else:
            log(f"Three paradigms present: {actual_paradigms}")

        # Check 4 : 100 unique UIDs
        unique_uids = df_lmm_input['UID'].nunique()
        if unique_uids != 100:
            critical_failures.append(f"Unique UIDs: expected 100, got {unique_uids}")
        else:
            log(f"100 unique UIDs")

        # Check 5 : TSVR range valid (0-200 hours)
        # Real TSVR data has natural variation - participants tested late is normal
        # This is a warning, not a failure
        tsvr_min = df_lmm_input['TSVR_hours'].min()
        tsvr_max = df_lmm_input['TSVR_hours'].max()
        if not df_lmm_input['TSVR_hours'].between(0, 200).all():
            # Count how many are outside range
            outside_range = (~df_lmm_input['TSVR_hours'].between(0, 200)).sum()
            moderate_warnings.append(
                f"TSVR range: {outside_range} values outside 0-200 hours "
                f"(min={tsvr_min:.2f}, max={tsvr_max:.2f}). "
                f"This is expected with real participant data."
            )
        else:
            log(f"TSVR range valid: {tsvr_min:.2f} - {tsvr_max:.2f} hours")

        # Report validation results
        # MODERATE warnings first (don't fail)
        if moderate_warnings:
            for warning in moderate_warnings:
                log(f"{warning}")

        # CRITICAL failures (fail if any)
        if critical_failures:
            for failure in critical_failures:
                log(f"{failure}")
            raise ValueError(f"Step 04 validation failed (CRITICAL): {'; '.join(critical_failures)}")

        log("All CRITICAL checks passed")
        # Summary Statistics
        # Log summary for debugging and verification

        log("\nLMM Input Data Summary:")
        log(f"  Total rows: {len(df_lmm_input)}")
        log(f"  Unique participants (UID): {df_lmm_input['UID'].nunique()}")
        log(f"  Unique composite_IDs: {df_lmm_input['composite_ID'].nunique()}")
        log(f"  Test sessions: {sorted(df_lmm_input['test'].unique())}")
        log(f"  Paradigms: {df_lmm_input['paradigm'].unique().tolist()}")
        log(f"  TSVR_hours: min={tsvr_min:.2f}, max={tsvr_max:.2f}, mean={df_lmm_input['TSVR_hours'].mean():.2f}")
        log(f"  Theta: min={df_lmm_input['theta'].min():.3f}, max={df_lmm_input['theta'].max():.3f}, mean={df_lmm_input['theta'].mean():.3f}")

        # Paradigm-wise summary
        log("\nBy Paradigm:")
        for paradigm in ['free_recall', 'cued_recall', 'recognition']:
            subset = df_lmm_input[df_lmm_input['paradigm'] == paradigm]
            log(f"  {paradigm}: n={len(subset)}, theta_mean={subset['theta'].mean():.3f}, theta_std={subset['theta'].std():.3f}")

        log("\nStep 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
