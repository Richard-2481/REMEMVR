#!/usr/bin/env python3
"""
Step 06: Descriptive Statistics by Paradigm × Time
RQ 6.9.7 - Paradigm-Specific Calibration Trajectory

PURPOSE: Compute mean calibration by paradigm × timepoint with rankings
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_descriptive_by_time.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 6: descriptive_by_time")

        # Load calibration scores
        data_path = RQ_DIR / "data" / "step02_calibration_scores.csv"
        df = pd.read_csv(data_path, encoding='utf-8')
        log(f"{data_path.name} ({len(df)} rows)")

        # Compute descriptive statistics by paradigm × test
        log("Computing descriptive statistics by paradigm × timepoint...")

        desc = df.groupby(['paradigm', 'test']).agg(
            TSVR_hours_mean=('TSVR_hours', 'mean'),
            n=('calibration', 'count'),
            mean_calibration=('calibration', 'mean'),
            sd_calibration=('calibration', 'std'),
            se_calibration=('calibration', lambda x: x.std() / np.sqrt(len(x)))
        ).reset_index()

        # Compute 95% CIs
        desc['ci_lower'] = desc['mean_calibration'] - 1.96 * desc['se_calibration']
        desc['ci_upper'] = desc['mean_calibration'] + 1.96 * desc['se_calibration']

        # Compute absolute mean for ranking
        desc['abs_mean_calibration'] = desc['mean_calibration'].abs()

        # Rank paradigms within each timepoint (1=best calibrated, closest to 0)
        desc['rank'] = desc.groupby('test')['abs_mean_calibration'].rank(method='min').astype(int)

        log(f"{len(desc)} cells (3 paradigms × 4 timepoints)")

        # Display results
        log("Descriptive statistics by paradigm × time:")
        for _, row in desc.sort_values(['test', 'rank']).iterrows():
            log(f"  {row['test']}, {row['paradigm']}: n={row['n']}, mean={row['mean_calibration']:.3f}, SD={row['sd_calibration']:.3f}, rank={row['rank']}")

        # Validate structure
        validation_result = validate_dataframe_structure(
            desc,
            expected_rows=12,
            expected_columns=['paradigm', 'test', 'n', 'mean_calibration']
        )

        if not validation_result.get('valid', False):
            log(f"Validation failed: {validation_result.get('message', 'Unknown')}")
            sys.exit(1)

        log("Structure validation successful")

        # Check balanced cell sizes
        if (desc['n'] != 100).any():
            log("Unequal cell sizes detected (expected n=100 per cell)")

        # Save results
        out_path = RQ_DIR / "data" / "step06_paradigm_by_time.csv"
        desc.to_csv(out_path, index=False, encoding='utf-8')
        log(f"{out_path.name} ({len(desc)} rows)")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
