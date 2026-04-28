#!/usr/bin/env python3
"""extract_theta_scores: Load omnibus theta_all scores from Ch5 5.1.1 outputs and aggregate by participant."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_extract_theta_scores.log"
OUTPUT_FILE = RQ_DIR / "data" / "step02_theta_scores.csv"

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

        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        theta_df = pd.read_csv(theta_path)

        log(f"{theta_path.name} ({len(theta_df)} rows, {len(theta_df.columns)} columns)")
        log(f"Columns: {theta_df.columns.tolist()}")
        # Aggregate by Participant
        # Ch5 data is test-level (4 rows per participant: test 1-4)
        # Need participant-level means for regression analysis
        log("Computing participant-level theta means...")

        # Verify column name (actual file has 'Theta_All' not 'theta_all')
        if 'Theta_All' not in theta_df.columns:
            log(f"Expected column 'Theta_All' not found")
            log(f"Available columns: {theta_df.columns.tolist()}")
            sys.exit(1)

        # Group by UID and compute mean theta across 4 tests
        participant_theta = theta_df.groupby('UID')['Theta_All'].mean().reset_index()
        participant_theta.columns = ['UID', 'theta_all']

        log(f"{len(participant_theta)} participants")
        # Standardize Theta Scores
        log("Computing z-scores...")

        participant_theta['theta_standardized'] = (
            (participant_theta['theta_all'] - participant_theta['theta_all'].mean()) /
            participant_theta['theta_all'].std()
        )

        log("Theta scores converted to z-scores")
        # Validation Checks
        log("Running validation checks...")

        # Check theta range (IRT ability scale typically -4 to 4)
        min_theta = participant_theta['theta_all'].min()
        max_theta = participant_theta['theta_all'].max()
        mean_theta = participant_theta['theta_all'].mean()
        sd_theta = participant_theta['theta_all'].std()

        log(f"Theta range: [{min_theta:.2f}, {max_theta:.2f}]")
        log(f"Theta M={mean_theta:.2f}, SD={sd_theta:.2f}")

        if min_theta < -4 or max_theta > 4:
            log(f"Theta scores outside typical IRT range (-4 to 4)")
        else:
            log("Theta scores within expected IRT range")

        # Check for missing data
        n_missing = participant_theta.isnull().sum().sum()
        if n_missing > 0:
            log(f"{n_missing} missing values detected")
        else:
            log("No missing values")

        # Check participant count
        expected_n = 100
        actual_n = len(participant_theta)
        if actual_n != expected_n:
            log(f"Expected {expected_n} participants, found {actual_n}")
        else:
            log(f"N={actual_n} participants (as expected)")
        # Save Output
        log("Saving participant-level theta scores...")

        participant_theta.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        log(f"{OUTPUT_FILE} ({len(participant_theta)} participants, {len(participant_theta.columns)} columns)")
        # Summary Statistics
        log("Descriptive statistics:")
        log(f"  Theta_all: M={participant_theta['theta_all'].mean():.2f}, SD={participant_theta['theta_all'].std():.2f}")
        log(f"  Theta_all: Range=[{participant_theta['theta_all'].min():.2f}, {participant_theta['theta_all'].max():.2f}]")
        log(f"  Theta_standardized: M={participant_theta['theta_standardized'].mean():.2f}, SD={participant_theta['theta_standardized'].std():.2f}")

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
