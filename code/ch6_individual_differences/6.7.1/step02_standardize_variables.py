#!/usr/bin/env python3
"""standardize_variables: Convert all variables to T-scores (M=50, SD=10) for meaningful effect size comparison"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.data import standardize_to_t_scores
from tools.validation import validate_standardization

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_standardize_variables.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Standardize Variables")

        # Load merged data
        log("Loading merged dataset...")
        input_path = RQ_DIR / 'data' / 'step01_merged_dataset.csv'
        df = pd.read_csv(input_path)
        log(f"{len(df)} rows, {len(df.columns)} cols")

        # Compute sample statistics and standardize
        log("Converting to T-scores (M=50, SD=10)...")

        # Standardize each variable using SAMPLE statistics
        df['REMEMVR_T'] = standardize_to_t_scores(
            scores=df['theta_all_mean'],
            population_mean=df['theta_all_mean'].mean(),
            population_sd=df['theta_all_mean'].std()
        )
        log(f"REMEMVR_T: M={df['REMEMVR_T'].mean():.2f}, SD={df['REMEMVR_T'].std():.2f}")

        df['RAVLT_T'] = standardize_to_t_scores(
            scores=df['ravlt_total'],
            population_mean=df['ravlt_total'].mean(),
            population_sd=df['ravlt_total'].std()
        )
        log(f"RAVLT_T: M={df['RAVLT_T'].mean():.2f}, SD={df['RAVLT_T'].std():.2f}")

        df['BVMT_T'] = standardize_to_t_scores(
            scores=df['bvmt_total'],
            population_mean=df['bvmt_total'].mean(),
            population_sd=df['bvmt_total'].std()
        )
        log(f"BVMT_T: M={df['BVMT_T'].mean():.2f}, SD={df['BVMT_T'].std():.2f}")

        df['RAVLT_PctRet_T'] = standardize_to_t_scores(
            scores=df['ravlt_pct_ret'],
            population_mean=df['ravlt_pct_ret'].mean(),
            population_sd=df['ravlt_pct_ret'].std()
        )
        log(f"RAVLT_PctRet_T: M={df['RAVLT_PctRet_T'].mean():.2f}, SD={df['RAVLT_PctRet_T'].std():.2f}")

        df['BVMT_PctRet_T'] = standardize_to_t_scores(
            scores=df['bvmt_pct_ret'],
            population_mean=df['bvmt_pct_ret'].mean(),
            population_sd=df['bvmt_pct_ret'].std()
        )
        log(f"BVMT_PctRet_T: M={df['BVMT_PctRet_T'].mean():.2f}, SD={df['BVMT_PctRet_T'].std():.2f}")

        # Validate standardization
        log("Checking T-score properties...")

        validation_result = validate_standardization(
            df=df,
            column_names=['REMEMVR_T', 'RAVLT_T', 'BVMT_T', 'RAVLT_PctRet_T', 'BVMT_PctRet_T'],
            tolerance=0.01
        )

        if validation_result.get('valid', False):
            log(f"Standardization validation passed")
        else:
            log(f"Standardization validation: {validation_result.get('message', 'Unknown issue')}")

        # Save standardized data
        log("Saving T-scored variables...")
        output_path = RQ_DIR / 'data' / 'step02_standardized_data.csv'

        # Select final columns
        df_output = df[['UID', 'REMEMVR_T', 'RAVLT_T', 'BVMT_T', 'RAVLT_PctRet_T', 'BVMT_PctRet_T']].copy()
        df_output.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
