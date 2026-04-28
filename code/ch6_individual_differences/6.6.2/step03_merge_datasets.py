#!/usr/bin/env python3
"""merge_analysis_dataset: Merge RAVLT forgetting scores with REMEMVR slope estimates to create"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step03_merge_datasets.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: merge_analysis_dataset")
        # Load Input Data

        log("Loading input datasets...")

        # Load RAVLT scores (Step 01 output)
        ravlt_path = RQ_DIR / "data" / "step01_ravlt_scores.csv"
        ravlt_df = pd.read_csv(ravlt_path, encoding='utf-8')
        log(f"RAVLT scores: {len(ravlt_df)} rows, {len(ravlt_df.columns)} cols")
        log(f"RAVLT columns: {ravlt_df.columns.tolist()}")

        # Load REMEMVR slopes (Step 02 output)
        slopes_path = RQ_DIR / "data" / "step02_rememvr_slopes.csv"
        slopes_df = pd.read_csv(slopes_path, encoding='utf-8')
        log(f"REMEMVR slopes: {len(slopes_df)} rows, {len(slopes_df.columns)} cols")
        log(f"REMEMVR columns: {slopes_df.columns.tolist()}")
        # Merge Datasets

        log("Merging datasets on UID...")

        # Inner join to ensure both RAVLT and slope data present
        merged_df = pd.merge(ravlt_df, slopes_df, on='UID', how='inner')

        log(f"Input 1 (RAVLT): {len(ravlt_df)} rows")
        log(f"Input 2 (Slopes): {len(slopes_df)} rows")
        log(f"Output (Merged): {len(merged_df)} rows")

        # Check for data loss
        if len(merged_df) < len(ravlt_df) or len(merged_df) < len(slopes_df):
            missing_from_ravlt = len(ravlt_df) - len(merged_df)
            missing_from_slopes = len(slopes_df) - len(merged_df)
            log(f"Data loss detected:")
            log(f"  - Missing from RAVLT: {missing_from_ravlt} participants")
            log(f"  - Missing from Slopes: {missing_from_slopes} participants")
        else:
            log("No data loss - all participants retained")

        # Report merged dataset statistics
        log(f"Merged dataset columns: {merged_df.columns.tolist()}")
        log(f"RAVLT_Forgetting - Mean: {merged_df['RAVLT_Forgetting'].mean():.3f}, SD: {merged_df['RAVLT_Forgetting'].std():.3f}")
        if 'RAVLT_Pct_Ret' in merged_df.columns:
            log(f"RAVLT_Pct_Ret - Mean: {merged_df['RAVLT_Pct_Ret'].mean():.3f}, SD: {merged_df['RAVLT_Pct_Ret'].std():.3f}")
        log(f"REMEMVR_Slope - Mean: {merged_df['REMEMVR_Slope'].mean():.4f}, SD: {merged_df['REMEMVR_Slope'].std():.4f}")
        # Save Merged Dataset
        # This output will be used by: Steps 04-08 (correlation analyses)

        log(f"Saving merged dataset...")
        output_path = RQ_DIR / "data" / "step03_analysis_input.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Required columns present for correlation analysis
        # Required columns: UID, RAVLT_Forgetting, REMEMVR_Slope

        log("Running validate_data_columns...")

        required_columns = ["UID", "RAVLT_Forgetting", "REMEMVR_Slope"]

        validation_result = validate_data_columns(
            df=merged_df,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")

            if validation_result.get('valid', False):
                log("PASS - All required columns present")
            else:
                log("FAIL - Missing required columns")
                log(f"Missing: {validation_result.get('missing_columns', [])}")
        else:
            log(f"{validation_result}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
