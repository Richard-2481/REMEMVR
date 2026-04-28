#!/usr/bin/env python3
"""Extract Domain Theta Scores: Extract What and Where domain theta scores from Ch5 5.2.1 (EXCLUDE When due to"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step01_extract_domain_theta_scores.log"
OUTPUT_DIR = RQ_DIR / "data"

# Input from Ch5 5.2.1
INPUT_FILE = PROJECT_ROOT / 'results' / 'ch5' / '5.2.1' / 'data' / 'step03_theta_scores.csv'

# Output
OUTPUT_FILE = OUTPUT_DIR / 'step01_domain_theta_scores.csv'

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract Domain Theta Scores")
        # Load Ch5 5.2.1 Theta Scores

        log("\nLoading Ch5 5.2.1 theta scores...")
        log(f"{INPUT_FILE}")

        df_theta = pd.read_csv(INPUT_FILE)
        log(f"{len(df_theta)} rows, {len(df_theta.columns)} columns")
        log(f"{df_theta.columns.tolist()}")

        # Scientific Mantra Checkpoint
        log("\nData structure validation")
        expected_cols = ['composite_ID', 'theta_what', 'theta_where', 'theta_when']
        actual_cols = df_theta.columns.tolist()

        if actual_cols != expected_cols:
            raise ValueError(f"Column mismatch. Expected: {expected_cols}, Got: {actual_cols}")

        log(f"Columns match expected structure")
        log(f"First 5 rows:")
        log(f"{df_theta.head().to_string()}")
        # Parse composite_ID to Extract UID
        # composite_ID format: "A010_1" → UID="A010", test=1

        log("\nExtracting UID from composite_ID...")

        df_theta['UID'] = df_theta['composite_ID'].str.split('_').str[0]
        df_theta['test'] = df_theta['composite_ID'].str.split('_').str[1].astype(int)

        log(f"{df_theta['UID'].nunique()} unique participants")
        log(f"{df_theta['test'].nunique()} unique test sessions")
        log(f"Test distribution:")
        log(f"{df_theta['test'].value_counts().sort_index().to_string()}")

        # Verify expected structure: 100 UIDs × 4 tests = 400 rows
        n_uids = df_theta['UID'].nunique()
        n_tests = df_theta.groupby('UID')['test'].count()

        if n_uids != 100:
            log(f"Expected 100 UIDs, found {n_uids}")

        incomplete_uids = n_tests[n_tests != 4]
        if len(incomplete_uids) > 0:
            log(f"{len(incomplete_uids)} UIDs with incomplete tests:")
            log(f"{incomplete_uids.to_string()}")

        # Scientific Mantra Checkpoint
        log(f"Data completeness check")
        log(f"UID extraction complete")
        # Aggregate by UID (Mean Theta Across 4 Tests)
        # Exclude When domain (77% floor effect documented in Ch5)

        log("\nComputing participant-level mean theta scores...")
        log("When domain (77% floor effect per Ch5 documentation)")

        # Group by UID and compute mean for What and Where domains ONLY
        df_agg = df_theta.groupby('UID').agg({
            'theta_what': 'mean',
            'theta_where': 'mean'
            # EXCLUDE theta_when (floor effect)
        }).reset_index()

        log(f"{len(df_agg)} participants (reduced from {len(df_theta)} test-level rows)")
        log(f"{df_agg.columns.tolist()}")
        log(f"Descriptive statistics (raw means):")
        log(f"{df_agg[['theta_what', 'theta_where']].describe().to_string()}")

        # Scientific Mantra Checkpoint
        log("\nAggregation validation")
        if len(df_agg) != 100:
            log(f"Expected 100 participants, got {len(df_agg)}")
        else:
            log(f"Expected participant count (N=100)")
        # Apply Z-Score Standardization
        # Formula: z = (X - mean(X)) / sd(X)

        log("\nApplying z-score standardization...")

        # Rename columns to final format before standardization
        df_agg = df_agg.rename(columns={
            'theta_what': 'theta_What',
            'theta_where': 'theta_Where'
        })

        # Apply z-score transformation
        for domain in ['theta_What', 'theta_Where']:
            mean_val = df_agg[domain].mean()
            sd_val = df_agg[domain].std()

            log(f"{domain} - Mean: {mean_val:.4f}, SD: {sd_val:.4f}")

            df_agg[domain] = (df_agg[domain] - mean_val) / sd_val

        log(f"Z-scored domains")
        log(f"Post-standardization statistics:")
        log(f"{df_agg[['theta_What', 'theta_Where']].describe().to_string()}")

        # Scientific Mantra Checkpoint
        log("\nStandardization validation")
        for domain in ['theta_What', 'theta_Where']:
            z_mean = df_agg[domain].mean()
            z_sd = df_agg[domain].std()

            # Check mean~0, SD~1 (tolerance: 1e-10 for floating point)
            if abs(z_mean) > 1e-10:
                log(f"{domain} z-score mean not ~0: {z_mean}")
            if abs(z_sd - 1.0) > 1e-10:
                log(f"{domain} z-score SD not ~1: {z_sd}")

        log(f"Z-score properties validated")
        # Save Output

        log(f"\nWriting output to {OUTPUT_FILE}")

        df_agg.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        log(f"{len(df_agg)} rows, {len(df_agg.columns)} columns")
        log(f"{df_agg.columns.tolist()}")
        # VALIDATION: Check Output Properties

        log("\nFinal output validation...")

        # Check 1: Row count
        if len(df_agg) != 100:
            raise ValueError(f"Expected 100 rows, got {len(df_agg)}")
        log(f"Row count = 100")

        # Check 2: No missing values
        missing_count = df_agg.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"Found {missing_count} missing values")
        log(f"No missing values")

        # Check 3: Z-score properties
        for domain in ['theta_What', 'theta_Where']:
            z_mean = abs(df_agg[domain].mean())
            z_sd = abs(df_agg[domain].std() - 1.0)

            if z_mean > 1e-10 or z_sd > 1e-10:
                log(f"{domain} standardization slightly off (mean={z_mean:.2e}, SD_diff={z_sd:.2e})")

        log(f"Z-score properties validated")

        log("\nStep 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
