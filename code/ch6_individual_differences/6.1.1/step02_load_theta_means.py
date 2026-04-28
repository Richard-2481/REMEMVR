#!/usr/bin/env python3
"""Load Theta Means from Ch5: Load IRT theta scores from Ch5 5.1.1 and compute mean theta per participant across test sessions"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step02_load_theta_means.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Load Theta Means from Ch5")
        # Load Input Data from Ch5

        log("Loading theta scores from Ch5 5.1.1...")
        # Load Ch5 theta scores from results/ch5/5.1.1/data/step03_theta_scores.csv
        # Expected columns: UID, test, Theta_All
        # Expected rows: ~400 (100 participants x 4 test sessions)
        ch5_theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        theta_scores = pd.read_csv(ch5_theta_path, encoding='utf-8')
        log(f"{ch5_theta_path} ({len(theta_scores)} rows, {len(theta_scores.columns)} cols)")
        log(f"Columns: {theta_scores.columns.tolist()}")
        log(f"Unique UIDs: {theta_scores['UID'].nunique()}")
        log(f"Test sessions: {sorted(theta_scores['test'].unique())}")
        # Custom Theta Mean Computation

        log("Computing mean theta per participant...")
        
        # Check data structure
        required_cols = ['UID', 'test', 'Theta_All']
        missing_cols = [col for col in required_cols if col not in theta_scores.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        missing_data = theta_scores[required_cols].isnull().sum()
        if missing_data.any():
            log(f"Missing data found: {missing_data.to_dict()}")
        
        # Group by UID and compute mean theta across test sessions
        # Parameters: aggregation_method=mean, min_sessions_required=3
        participant_sessions = theta_scores.groupby('UID').size()
        log(f"Sessions per participant - Min: {participant_sessions.min()}, Max: {participant_sessions.max()}, Mean: {participant_sessions.mean():.1f}")
        
        # Filter participants with at least 3 sessions (min_sessions_required=3)
        valid_participants = participant_sessions[participant_sessions >= 3].index
        log(f"Participants with >=3 sessions: {len(valid_participants)} of {len(participant_sessions)}")
        
        # Compute mean theta for valid participants
        theta_means = (theta_scores[theta_scores['UID'].isin(valid_participants)]
                      .groupby('UID')['Theta_All']
                      .mean()
                      .reset_index())
        theta_means.columns = ['UID', 'theta_mean']
        
        log(f"Computed theta means for {len(theta_means)} participants")
        log(f"Theta mean stats - Mean: {theta_means['theta_mean'].mean():.3f}, Std: {theta_means['theta_mean'].std():.3f}")
        log(f"Theta range: [{theta_means['theta_mean'].min():.3f}, {theta_means['theta_mean'].max():.3f}]")
        # Save Analysis Output
        # These outputs will be used by: Step 03 (merge with cognitive tests)

        output_path = RQ_DIR / "data" / "step02_theta_means.csv"
        log(f"Saving {output_path}...")
        # Output: step02_theta_means.csv
        # Contains: Mean theta scores per participant for regression analysis
        # Columns: ['UID', 'theta_mean']
        theta_means.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(theta_means)} rows, {len(theta_means.columns)} cols)")
        # Run Validation Tool
        # Validates: Theta score range, distribution normality, participant count
        # Threshold: theta_range [-3, 3], expected_mean 0.0, tolerance 0.5

        log("Running validate_data_columns...")
        
        # Validate theta score characteristics
        validation_result = {
            'valid': True,
            'messages': []
        }
        
        # Check theta range [-3, 3]
        theta_min, theta_max = theta_means['theta_mean'].min(), theta_means['theta_mean'].max()
        if theta_min < -3.0 or theta_max > 3.0:
            validation_result['valid'] = False
            validation_result['messages'].append(f"Theta values outside IRT range [-3, 3]: [{theta_min:.3f}, {theta_max:.3f}]")
        else:
            log(f"Theta range check PASSED: [{theta_min:.3f}, {theta_max:.3f}] within [-3, 3]")
        
        # Check distribution approximately normal with mean near 0 (tolerance 0.5)
        theta_mean = theta_means['theta_mean'].mean()
        if abs(theta_mean) > 0.5:
            validation_result['valid'] = False
            validation_result['messages'].append(f"Theta mean {theta_mean:.3f} not near 0 (tolerance 0.5)")
        else:
            log(f"Theta mean check PASSED: {theta_mean:.3f} within tolerance 0.5 of 0")
        
        # Check participant count (90-100 expected)
        n_participants = len(theta_means)
        if n_participants < 90:
            validation_result['valid'] = False
            validation_result['messages'].append(f"Low participant count: {n_participants} < 90")
        else:
            log(f"Participant count check PASSED: {n_participants} >= 90")
        
        # Check for missing theta_mean values
        missing_theta = theta_means['theta_mean'].isnull().sum()
        if missing_theta > 0:
            validation_result['valid'] = False
            validation_result['messages'].append(f"Missing theta_mean values: {missing_theta}")
        else:
            log(f"Missing values check PASSED: No missing theta_mean values")

        # Report validation results
        if validation_result['valid']:
            log("All checks PASSED")
        else:
            log("Some checks FAILED:")
            for msg in validation_result['messages']:
                log(f"- {msg}")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)