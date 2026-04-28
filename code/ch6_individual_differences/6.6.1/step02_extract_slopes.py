#!/usr/bin/env python3
"""Extract Individual Slope Estimates: Extract per-participant slope estimates from Ch5 5.1.4 model-averaged results."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.6.1
LOG_FILE = RQ_DIR / "logs" / "step02_extract_slopes.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Extract Individual Slope Estimates")
        # Load Ch5 5.1.4 Random Effects Data

        log("Loading Ch5 5.1.4 random effects...")
        ch5_path = PROJECT_ROOT / 'results' / 'ch5' / '5.1.4' / 'data' / 'step04_random_effects.csv'

        if not ch5_path.exists():
            log(f"Ch5 dependency file not found: {ch5_path}")
            log("Ensure Ch5 5.1.4 has been completed successfully")
            sys.exit(1)

        df = pd.read_csv(ch5_path)
        log(f"{ch5_path.name} ({len(df)} rows, {len(df.columns)} cols)")
        log(f"{', '.join(df.columns.tolist())}")
        # Verify Required Columns
        # Per 4_analysis.yaml lines 148-149: Expected columns: UID, random_slope

        log("Checking for required columns...")
        required_cols = ['UID', 'random_slope']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            log(f"Missing required columns: {missing_cols}")
            log(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

        log(f"All required columns present: {', '.join(required_cols)}")
        # Extract and Rename Columns
        # Per 4_analysis.yaml lines 141-142:
        # - columns_to_extract: ["UID", "random_slope"]
        # - rename_columns: random_slope -> slope

        log("Selecting UID and random_slope columns...")
        output_df = df[required_cols].copy()

        log("Renaming random_slope -> slope...")
        output_df.rename(columns={'random_slope': 'slope'}, inplace=True)

        log(f"{len(output_df)} participants with slope estimates")
        # Validate Slope Range
        # Per 4_analysis.yaml line 144: slope_range: [-0.5, 0.1]
        # Slopes represent forgetting rate (negative = performance decline)

        log("Validating slope range...")

        slope_min = output_df['slope'].min()
        slope_max = output_df['slope'].max()
        slope_mean = output_df['slope'].mean()
        slope_sd = output_df['slope'].std()

        log(f"Slope: mean={slope_mean:.4f}, SD={slope_sd:.4f}, range=[{slope_min:.4f}, {slope_max:.4f}]")

        # Check expected range [-0.5, 0.1]
        expected_min = -0.5
        expected_max = 0.1

        out_of_range = (output_df['slope'] < expected_min) | (output_df['slope'] > expected_max)
        n_out = out_of_range.sum()

        if n_out > 0:
            log(f"{n_out} slopes outside expected range [{expected_min}, {expected_max}]")
            log(f"Min slope: {slope_min:.4f}, Max slope: {slope_max:.4f}")
        else:
            log(f"All slopes within expected range [{expected_min}, {expected_max}]")

        # Check for missing values
        missing_count = output_df['slope'].isnull().sum()
        if missing_count > 0:
            log(f"{missing_count} participants have missing slopes")
            sys.exit(1)
        else:
            log("No missing slope values")

        # Check participant count
        expected_n = 100
        actual_n = len(output_df)
        if actual_n == expected_n:
            log(f"Participant count: {actual_n} (expected {expected_n})")
        else:
            log(f"Participant count: {actual_n} (expected {expected_n})")
        # Save Output
        # Output: results/ch7/7.6.1/data/step02_slopes_extracted.csv
        # Per 4_analysis.yaml lines 152-154

        output_path = RQ_DIR / 'data' / 'step02_slopes_extracted.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"{len(output_df)} participants with slope estimates")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
