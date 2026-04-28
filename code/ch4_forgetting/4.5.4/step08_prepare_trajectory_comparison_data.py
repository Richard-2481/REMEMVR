#!/usr/bin/env python3
"""prepare_trajectory_comparison_data: Create plot source CSV for trajectory comparison showing mean IRT theta and mean"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_trajectory_comparison_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 8: Prepare trajectory comparison data")
        # Load Input Data

        log("Loading IRT theta scores from step 0...")
        theta_long = pd.read_csv(RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv", encoding='utf-8')
        log(f"IRT theta data ({len(theta_long)} rows, {len(theta_long.columns)} cols)")

        log("Loading CTT scores from step 1...")
        ctt_scores = pd.read_csv(RQ_DIR / "data" / "step01_ctt_scores.csv", encoding='utf-8')
        log(f"CTT scores data ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        # Aggregate IRT Data
        #               95% CI using SE = std / sqrt(n)

        log("Aggregating IRT data by location_type + test...")
        irt_agg = theta_long.groupby(['location_type', 'test']).agg({
            'irt_theta': ['mean', 'std', 'count'],
            'TSVR_hours': 'mean'
        }).reset_index()

        # Flatten column names
        irt_agg.columns = ['location_type', 'test', 'mean_score', 'std_score', 'n', 'time']

        # Compute 95% CI: mean ± 1.96 × SE, where SE = std / sqrt(n)
        irt_agg['se'] = irt_agg['std_score'] / np.sqrt(irt_agg['n'])
        irt_agg['ci_lower'] = irt_agg['mean_score'] - 1.96 * irt_agg['se']
        irt_agg['ci_upper'] = irt_agg['mean_score'] + 1.96 * irt_agg['se']

        # Add method label
        irt_agg['method'] = 'IRT'

        # Select final columns
        irt_agg = irt_agg[['location_type', 'test', 'method', 'mean_score', 'ci_lower', 'ci_upper', 'time', 'n']]

        log(f"IRT data ({len(irt_agg)} rows)")
        # Aggregate CTT Data
        #               mean TSVR, 95% CI using SE = std / sqrt(n)

        log("Aggregating CTT data by location_type + test...")
        ctt_agg = ctt_scores.groupby(['location_type', 'test']).agg({
            'ctt_mean_score': ['mean', 'std', 'count'],
            'TSVR_hours': 'mean'
        }).reset_index()

        # Flatten column names
        ctt_agg.columns = ['location_type', 'test', 'mean_score', 'std_score', 'n', 'time']

        # Compute 95% CI: mean ± 1.96 × SE, where SE = std / sqrt(n)
        ctt_agg['se'] = ctt_agg['std_score'] / np.sqrt(ctt_agg['n'])
        ctt_agg['ci_lower'] = ctt_agg['mean_score'] - 1.96 * ctt_agg['se']
        ctt_agg['ci_upper'] = ctt_agg['mean_score'] + 1.96 * ctt_agg['se']

        # Add method label
        ctt_agg['method'] = 'CTT'

        # Select final columns
        ctt_agg = ctt_agg[['location_type', 'test', 'method', 'mean_score', 'ci_lower', 'ci_upper', 'time', 'n']]

        log(f"CTT data ({len(ctt_agg)} rows)")
        # Stack Both Datasets
        # Combine IRT and CTT aggregated data

        log("Stacking IRT and CTT aggregated data...")
        trajectory_data = pd.concat([irt_agg, ctt_agg], ignore_index=True)
        log(f"Combined data ({len(trajectory_data)} rows)")
        # Save Output
        # Output: data/step08_trajectory_comparison_data.csv
        # Contains: Aggregated means and 95% CIs for trajectory plotting

        output_path = RQ_DIR / "data" / "step08_trajectory_comparison_data.csv"
        log(f"Saving trajectory comparison data to {output_path}...")
        trajectory_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(trajectory_data)} rows, {len(trajectory_data.columns)} cols)")
        # Run Validation (Inline)
        # Validates: Row count, combinations, ranges, CI bounds

        log("Running inline validation...")

        # Check 1: Exactly 16 rows
        if len(trajectory_data) != 16:
            raise ValueError(f"Expected 16 rows, got {len(trajectory_data)}")
        log("Row count: 16 ")

        # Check 2: All combinations present
        expected_combinations = set()
        for location in ['source', 'destination']:
            for test in [1, 2, 3, 4]:
                for method in ['IRT', 'CTT']:
                    expected_combinations.add((location, test, method))

        actual_combinations = set(zip(trajectory_data['location_type'],
                                     trajectory_data['test'],
                                     trajectory_data['method']))

        if expected_combinations != actual_combinations:
            missing = expected_combinations - actual_combinations
            extra = actual_combinations - expected_combinations
            raise ValueError(f"Combination mismatch. Missing: {missing}, Extra: {extra}")
        log("All combinations present ")

        # Check 3: No NaN in critical columns
        critical_cols = ['mean_score', 'ci_lower', 'ci_upper', 'time']
        for col in critical_cols:
            if trajectory_data[col].isna().any():
                raise ValueError(f"NaN values found in {col}")
        log("No NaN in critical columns ")

        # Check 4: CI bounds valid (ci_upper > ci_lower)
        if not (trajectory_data['ci_upper'] > trajectory_data['ci_lower']).all():
            invalid_rows = trajectory_data[trajectory_data['ci_upper'] <= trajectory_data['ci_lower']]
            raise ValueError(f"Invalid CI bounds in {len(invalid_rows)} rows:\n{invalid_rows}")
        log("CI bounds valid (ci_upper > ci_lower) ")

        # Check 5: n = 100 for all rows
        if not (trajectory_data['n'] == 100).all():
            invalid_counts = trajectory_data[trajectory_data['n'] != 100]
            raise ValueError(f"Expected n=100 for all rows, found violations:\n{invalid_counts}")
        log("All n = 100 ")

        # Check 6: IRT mean_score in [-3, 3]
        irt_rows = trajectory_data[trajectory_data['method'] == 'IRT']
        if not ((irt_rows['mean_score'] >= -3) & (irt_rows['mean_score'] <= 3)).all():
            invalid_irt = irt_rows[(irt_rows['mean_score'] < -3) | (irt_rows['mean_score'] > 3)]
            raise ValueError(f"IRT mean_score out of [-3, 3] range:\n{invalid_irt}")
        log("IRT mean_score in [-3, 3] ")

        # Check 7: CTT mean_score in [0, 1]
        ctt_rows = trajectory_data[trajectory_data['method'] == 'CTT']
        if not ((ctt_rows['mean_score'] >= 0) & (ctt_rows['mean_score'] <= 1)).all():
            invalid_ctt = ctt_rows[(ctt_rows['mean_score'] < 0) | (ctt_rows['mean_score'] > 1)]
            raise ValueError(f"CTT mean_score out of [0, 1] range:\n{invalid_ctt}")
        log("CTT mean_score in [0, 1] ")

        # Check 8: time in [0, 168] hours
        if not ((trajectory_data['time'] >= 0) & (trajectory_data['time'] <= 168)).all():
            invalid_time = trajectory_data[(trajectory_data['time'] < 0) | (trajectory_data['time'] > 168)]
            raise ValueError(f"TSVR_hours out of [0, 168] range:\n{invalid_time}")
        log("time in [0, 168] hours ")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
