#!/usr/bin/env python3
"""Extract Theta Scores from RQ 5.4.1: Extract IRT theta scores by congruence from RQ 5.5 (DERIVED data dependency)."""

import sys
from pathlib import Path
import pandas as pd
import yaml
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.4.2
RQ5_DIR = PROJECT_ROOT / "results" / "ch5" / "5.4.1"
LOG_FILE = RQ_DIR / "logs" / "step00_extract_theta_from_rq5.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 0: Extract Theta Scores from RQ 5.5")
        # Check RQ 5.5 Dependency

        log("[DEPENDENCY CHECK] Validating RQ 5.5 completion status...")

        # Check status.yaml exists
        rq5_status_path = RQ5_DIR / "status.yaml"
        if not rq5_status_path.exists():
            raise FileNotFoundError(
                f"EXPECTATIONS ERROR: To extract theta from RQ 5.4.1 I expect "
                f"results/ch5/5.4.1/status.yaml, but file missing"
            )

        # Read status.yaml
        with open(rq5_status_path, 'r', encoding='utf-8') as f:
            rq5_status = yaml.safe_load(f)

        # Check results analysis.status = 'success'
        if 'results analysis' not in rq5_status:
            raise ValueError(
                f"EXPECTATIONS ERROR: RQ 5.5 status.yaml missing 'results analysis' section"
            )

        if rq5_status['results analysis'].get('status') != 'success':
            actual_status = rq5_status['results analysis'].get('status', 'UNKNOWN')
            raise ValueError(
                f"EXPECTATIONS ERROR: To extract theta from RQ 5.5 I expect "
                f"results analysis.status = 'success', but found '{actual_status}'. "
                f"RQ 5.5 must complete successfully before RQ 5.6 can proceed."
            )

        log(f"RQ 5.5 dependency satisfied (status = success)")
        # Load RQ 5.5 Theta Scores

        log("Loading RQ 5.5 theta scores...")

        rq5_theta_path = RQ5_DIR / "data" / "step03_theta_scores.csv"
        if not rq5_theta_path.exists():
            raise FileNotFoundError(
                f"EXPECTATIONS ERROR: To extract theta from RQ 5.4.1 I expect "
                f"results/ch5/5.4.1/data/step03_theta_scores.csv, but file missing"
            )

        df_theta = pd.read_csv(rq5_theta_path, encoding='utf-8')
        log(f"{rq5_theta_path.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        # Validate Structure

        log("Validating theta scores structure...")

        expected_columns = [
            'composite_ID',
            'theta_common', 'theta_congruent', 'theta_incongruent',
            'se_common', 'se_congruent', 'se_incongruent'
        ]

        # Check column count
        if len(df_theta.columns) != 7:
            raise ValueError(
                f"Column count incorrect: expected 7 columns, found {len(df_theta.columns)}"
            )
        log(f"Column count correct (7 columns)")

        # Check column names
        if list(df_theta.columns) != expected_columns:
            raise ValueError(
                f"Column names incorrect: expected {expected_columns}, "
                f"found {list(df_theta.columns)}"
            )
        log(f"Column names correct")

        # Check row count
        if len(df_theta) != 400:
            raise ValueError(
                f"Row count incorrect: expected 400 rows (100 participants x 4 tests), "
                f"found {len(df_theta)}"
            )
        log(f"Row count correct (400 rows)")

        # Check theta value ranges
        theta_cols = ['theta_common', 'theta_congruent', 'theta_incongruent']
        for col in theta_cols:
            if df_theta[col].min() < -3.0 or df_theta[col].max() > 3.0:
                raise ValueError(
                    f"Theta range invalid for {col}: "
                    f"min={df_theta[col].min():.3f}, max={df_theta[col].max():.3f}, "
                    f"expected [-3, 3]"
                )
        log(f"Theta value ranges valid (all in [-3, 3])")

        # Check SE value ranges
        se_cols = ['se_common', 'se_congruent', 'se_incongruent']
        for col in se_cols:
            if df_theta[col].min() < 0.1 or df_theta[col].max() > 1.0:
                raise ValueError(
                    f"SE range invalid for {col}: "
                    f"min={df_theta[col].min():.3f}, max={df_theta[col].max():.3f}, "
                    f"expected [0.1, 1.0]"
                )
        log(f"SE value ranges valid (all in [0.1, 1.0])")

        # Check for missing data
        if df_theta[theta_cols + se_cols].isna().any().any():
            n_missing = df_theta[theta_cols + se_cols].isna().sum().sum()
            raise ValueError(
                f"Missing data detected: {n_missing} NaN values in theta/SE columns"
            )
        log(f"No missing data (all theta and SE columns complete)")
        # Save Cached Copy
        # Output: data/step00_theta_scores_from_rq5.csv

        log("Saving cached theta scores...")

        output_path = RQ_DIR / "data" / "step00_theta_scores_from_rq5.csv"
        df_theta.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        # Summary Report

        log("All checks passed:")
        log(f"  - RQ 5.5 dependency satisfied")
        log(f"  - Theta file exists and loaded")
        log(f"  - Row count correct (400)")
        log(f"  - Column count correct (7)")
        log(f"  - Theta ranges valid ([-3, 3])")
        log(f"  - SE ranges valid ([0.1, 1.0])")
        log(f"  - No missing data")

        log("Step 0 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
