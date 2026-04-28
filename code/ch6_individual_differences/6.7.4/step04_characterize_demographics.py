#!/usr/bin/env python3
"""characterize_demographics: Compute descriptive statistics for false negative demographic characterization."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.7.4
LOG_FILE = RQ_DIR / "logs" / "step04_characterize_demographics.log"
INPUT_FALSE_NEG = RQ_DIR / "data" / "step03_false_negatives.csv"
INPUT_ALL = RQ_DIR / "data" / "step02_standardized_scores.csv"
OUTPUT_FILE = RQ_DIR / "data" / "step04_demographic_summary.csv"

# Variables to characterize
DEMOGRAPHIC_VARS = ['Age', 'Education', 'VR_Experience', 'NART_Score', 'RAVLT_Total', 'RAVLT_Pct_Ret', 'REMEMVR_theta']

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: characterize_demographics")
        # Load False Negatives
        log("Loading false negative cases from step03...")

        df_false_neg = pd.read_csv(INPUT_FALSE_NEG)
        log(f"False negatives: {len(df_false_neg)} cases")
        # Load All Participants
        log("Loading all participants from step02...")

        df_all = pd.read_csv(INPUT_ALL)
        log(f"All participants: {len(df_all)} cases")
        # Create "Other" Group (non-false-negatives)
        log("Creating comparison group (non-false-negatives)...")

        # Extract UIDs of false negatives
        false_neg_uids = df_false_neg['UID'].tolist()
        log(f"False negative UIDs: {false_neg_uids}")

        # Create "Other" group by excluding false negatives
        df_other = df_all[~df_all['UID'].isin(false_neg_uids)].copy()
        log(f"Other group: {len(df_other)} participants")

        # Verify counts
        if len(df_false_neg) + len(df_other) != len(df_all):
            log(f"Group counts don't sum to total: {len(df_false_neg)} + {len(df_other)} != {len(df_all)}")
            raise ValueError("Group assignment error - counts don't match")
        else:
            log(f"Group counts verified: {len(df_false_neg)} + {len(df_other)} = {len(df_all)}")
        # Compute Descriptive Statistics
        log("Computing descriptive statistics by group...")

        summary_rows = []

        for var in DEMOGRAPHIC_VARS:
            log(f"Processing variable: {var}")

            # False negative group statistics
            fn_data = df_false_neg[var]
            fn_n = fn_data.count()  # Count non-missing
            fn_mean = fn_data.mean()
            fn_sd = fn_data.std()

            # Other group statistics
            other_data = df_other[var]
            other_n = other_data.count()
            other_mean = other_data.mean()
            other_sd = other_data.std()

            # Create summary row
            summary_row = {
                'Variable': var,
                'False_Neg_N': fn_n,
                'False_Neg_Mean': fn_mean,
                'False_Neg_SD': fn_sd,
                'Other_N': other_n,
                'Other_Mean': other_mean,
                'Other_SD': other_sd
            }
            summary_rows.append(summary_row)

            # Log comparison
            log(f"  False Neg: N={fn_n}, Mean={fn_mean:.2f}, SD={fn_sd:.2f}")
            log(f"  Other:     N={other_n}, Mean={other_mean:.2f}, SD={other_sd:.2f}")
        # Create Summary DataFrame
        log("Creating demographic summary table...")

        df_summary = pd.DataFrame(summary_rows)
        log(f"Summary table: {len(df_summary)} rows, {len(df_summary.columns)} columns")
        # Save Summary
        log("Saving demographic summary...")

        df_summary.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        log(f"{OUTPUT_FILE}")
        # Validate Output
        log("Running validate_dataframe_structure...")

        expected_columns = ['Variable', 'False_Neg_N', 'False_Neg_Mean', 'False_Neg_SD',
                          'Other_N', 'Other_Mean', 'Other_SD']

        validation_result = validate_dataframe_structure(
            df=df_summary,
            expected_rows=7,  # 7 demographic variables
            expected_columns=expected_columns
        )

        if validation_result.get('valid', False):
            log(f"Demographic summary structure validated")
        else:
            log(f"Validation: {validation_result.get('message', 'Unknown issue')}")

        # Check for missing values
        missing_count = df_summary.isnull().sum().sum()
        if missing_count > 0:
            log(f"Summary contains {missing_count} missing values")
        else:
            log(f"No missing values in summary")

        # Verify sample sizes
        total_false_neg = df_summary['False_Neg_N'].iloc[0]  # Should be same for all vars
        total_other = df_summary['Other_N'].iloc[0]

        if total_false_neg != len(df_false_neg):
            log(f"False_Neg_N ({total_false_neg}) != actual false neg count ({len(df_false_neg)})")
        else:
            log(f"False negative sample size verified: {total_false_neg}")

        if total_other != len(df_other):
            log(f"Other_N ({total_other}) != actual other count ({len(df_other)})")
        else:
            log(f"Other group sample size verified: {total_other}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
