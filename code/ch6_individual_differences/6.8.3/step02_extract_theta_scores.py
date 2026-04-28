#!/usr/bin/env python3
"""Extract Theta Scores: Load Theta_All scores from Ch5 5.1.1 results and merge with cognitive test data"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_extract_theta_scores.log"
OUTPUT_DIR = RQ_DIR / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Theta range check (typical IRT range)
THETA_MIN = -4.0
THETA_MAX = 4.0

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract Theta Scores")
        # Load Ch5 Theta Scores
        log("Loading Ch5 5.1.1 theta scores...")

        theta_file = PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv'
        df_theta = pd.read_csv(theta_file)

        log(f"Ch5 theta scores: {len(df_theta)} rows, {len(df_theta.columns)} columns")

        # Verify Theta_All column exists
        if 'Theta_All' not in df_theta.columns:
            raise ValueError("Theta_All column not found in Ch5 theta scores")

        log(f"Theta_all (all rows): mean={df_theta['Theta_All'].mean():.3f}, SD={df_theta['Theta_All'].std():.3f}, range=[{df_theta['Theta_All'].min():.3f}, {df_theta['Theta_All'].max():.3f}]")

        # Aggregate by UID (Ch5 file has 100 UIDs × 4 tests = 400 rows, need participant-level)
        log("Averaging theta scores across 4 tests per participant...")
        df_theta_agg = df_theta.groupby('UID', as_index=False)['Theta_All'].mean()
        log(f"{len(df_theta_agg)} unique participants (from {len(df_theta)} test-level rows)")
        log(f"Theta_all (participant-level): mean={df_theta_agg['Theta_All'].mean():.3f}, SD={df_theta_agg['Theta_All'].std():.3f}, range=[{df_theta_agg['Theta_All'].min():.3f}, {df_theta_agg['Theta_All'].max():.3f}]")

        # Check for extreme values (using aggregated data)
        extreme_theta = (df_theta_agg['Theta_All'] < THETA_MIN) | (df_theta_agg['Theta_All'] > THETA_MAX)
        if extreme_theta.any():
            n_extreme = extreme_theta.sum()
            log(f"{n_extreme} theta values outside [{THETA_MIN}, {THETA_MAX}] range")
            extreme_uids = df_theta_agg.loc[extreme_theta, 'UID'].tolist()
            log(f"Extreme UIDs: {extreme_uids}")
        else:
            log(f"All theta values within [{THETA_MIN}, {THETA_MAX}] range")
        # Load Cognitive Test Data
        log("Loading cognitive test data from step 1...")

        cognitive_file = RQ_DIR / 'data' / 'step01_cognitive_tests.csv'
        df_cognitive = pd.read_csv(cognitive_file)

        log(f"Cognitive tests: {len(df_cognitive)} rows, {len(df_cognitive.columns)} columns")
        log(f"Cognitive columns: {df_cognitive.columns.tolist()}")
        # Merge Datasets
        log("Merging theta scores with cognitive tests on UID...")

        # Inner merge (requires matching UID) - using aggregated theta scores
        df_merged = pd.merge(
            df_cognitive,
            df_theta_agg[['UID', 'Theta_All']],
            on='UID',
            how='inner'
        )

        log(f"{len(df_merged)} rows after merge")

        # Verify merge completeness
        if len(df_merged) != len(df_cognitive):
            n_missing = len(df_cognitive) - len(df_merged)
            missing_uids = set(df_cognitive['UID']) - set(df_merged['UID'])
            log(f"Merge incomplete: {n_missing} participants missing theta scores")
            log(f"Missing UIDs: {missing_uids}")
            raise ValueError(f"Merge failed: {n_missing} participants missing theta scores")

        if len(df_merged) != len(df_theta_agg):
            n_missing = len(df_theta_agg) - len(df_merged)
            missing_uids = set(df_theta_agg['UID']) - set(df_merged['UID'])
            log(f"{n_missing} Ch5 participants not in cognitive test data")
            log(f"Missing UIDs: {missing_uids}")

        log("All cognitive test participants have theta scores")
        # Save Outputs
        # Save theta scores subset (aggregated by UID)
        theta_output = OUTPUT_DIR / "step02_Theta_All_scores.csv"
        log(f"Saving theta scores to {theta_output}")
        df_theta_agg[['UID', 'Theta_All']].to_csv(theta_output, index=False, encoding='utf-8')
        log(f"{len(df_theta_agg)} rows")

        # Save merged analysis dataset
        analysis_output = OUTPUT_DIR / "step02_analysis_input.csv"
        log(f"Saving merged analysis dataset to {analysis_output}")
        df_merged.to_csv(analysis_output, index=False, encoding='utf-8')
        log(f"{len(df_merged)} rows, {len(df_merged.columns)} columns")
        # Validation Check
        log("Validating merged dataset...")

        # Check for missing data
        total_missing = df_merged.isna().sum().sum()
        if total_missing > 0:
            log(f"{total_missing} missing values in merged dataset")
            for col in df_merged.columns:
                n_missing = df_merged[col].isna().sum()
                if n_missing > 0:
                    log(f"  {col}: {n_missing} missing")
        else:
            log("No missing data in merged dataset")

        # Validate theta range
        validation_result = validate_numeric_range(
            data=df_merged['Theta_All'],
            min_val=THETA_MIN,
            max_val=THETA_MAX,
            column_name='Theta_All'
        )

        if validation_result.get('valid', False):
            log(f"Theta range validation passed")
        else:
            log(f"Theta range validation: {validation_result.get('message', 'Unknown issue')}")

        # Check column order matches expected
        expected_columns = ['UID', 'Age', 'Sex', 'Education', 'RAVLT_T', 'BVMT_T',
                           'RPM_T', 'DASS_Total', 'Sleep', 'Theta_All']
        actual_columns = df_merged.columns.tolist()

        if actual_columns == expected_columns:
            log(f"Column order matches expected")
        else:
            log(f"Expected columns: {expected_columns}")
            log(f"Actual columns: {actual_columns}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
