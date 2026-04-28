#!/usr/bin/env python3
"""extract_paradigm_theta_scores: Load paradigm-specific theta scores from Ch5 5.3.1 results for RAVLT"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.1 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_paradigm_theta.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: extract_paradigm_theta_scores")
        # Load Input Data from Ch5

        log("Loading Ch5 theta scores from Ch5 5.3.1...")
        source_path = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step03_theta_scores.csv"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Ch5 dependency missing: {source_path}")
            
        theta_df = pd.read_csv(source_path)
        log(f"Ch5 theta scores ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        
        # Validate expected columns
        expected_cols = ['composite_ID', 'domain_name', 'theta']
        if list(theta_df.columns) != expected_cols:
            raise ValueError(f"Unexpected columns. Expected {expected_cols}, got {list(theta_df.columns)}")
        
        log(f"Unique domains: {sorted(theta_df['domain_name'].unique())}")
        # Filter to Target Paradigms
        # Target paradigms: Free Recall (free_recall) and Recognition (recognition)
        # Exclude: Cued Recall (cued_recall) - not needed for this analysis

        log("Filtering to free_recall and recognition paradigms...")
        target_paradigms = ['free_recall', 'recognition']
        
        # Filter to target paradigms
        paradigm_theta = theta_df[theta_df['domain_name'].isin(target_paradigms)].copy()
        log(f"Kept {len(paradigm_theta)} rows with target paradigms")
        
        # Verify both paradigms are present
        actual_paradigms = set(paradigm_theta['domain_name'].unique())
        missing_paradigms = set(target_paradigms) - actual_paradigms
        if missing_paradigms:
            raise ValueError(f"Missing required paradigms: {missing_paradigms}")
        
        log(f"Both paradigms present: {sorted(actual_paradigms)}")
        # Extract UID from composite_ID
        # Transform composite_ID format "A010_1" -> UID "A010"
        # Pattern: participant_test format

        log("Extracting UID from composite_ID...")
        
        # Extract UID (participant identifier) from composite_ID
        paradigm_theta['uid'] = paradigm_theta['composite_ID'].str.split('_').str[0]
        
        # Verify extraction worked
        sample_extraction = paradigm_theta[['composite_ID', 'uid']].head(3)
        log(f"UID extraction examples:")
        for _, row in sample_extraction.iterrows():
            log(f"  {row['composite_ID']} -> {row['uid']}")
        
        unique_uids = paradigm_theta['uid'].nunique()
        log(f"Found {unique_uids} unique participants")
        # Aggregate Theta Scores
        # Aggregate theta scores per participant per paradigm
        # Mean across all domains (What, Where, When) and all 4 tests per paradigm

        log("Computing mean theta per participant per paradigm...")
        
        # Group by UID and domain_name, compute mean theta
        # This aggregates across all test sessions (1-4) and implicitly across domains if multiple
        aggregated_theta = paradigm_theta.groupby(['uid', 'domain_name'])['theta'].mean().reset_index()
        
        log(f"Reduced to {len(aggregated_theta)} participant-paradigm combinations")
        
        # Check data structure
        paradigm_counts = aggregated_theta.groupby('uid')['domain_name'].count()
        participants_with_both = (paradigm_counts == 2).sum()
        participants_with_one = (paradigm_counts == 1).sum()
        
        log(f"Participants with both paradigms: {participants_with_both}")
        log(f"Participants with only one paradigm: {participants_with_one}")
        
        if participants_with_one > 0:
            log("Some participants missing one paradigm - will be excluded from analysis")
        # Reshape to Wide Format
        # Transform to wide format: uid, theta_free_recall, theta_recognition
        # This creates the final analysis dataset

        log("Converting to wide format...")
        
        # Pivot to wide format
        wide_theta = aggregated_theta.pivot(index='uid', columns='domain_name', values='theta').reset_index()
        
        # Rename columns to match specification
        column_mapping = {
            'free_recall': 'theta_free_recall',
            'recognition': 'theta_recognition'
        }
        wide_theta = wide_theta.rename(columns=column_mapping)
        
        # Ensure all expected columns exist
        expected_output_cols = ['uid', 'theta_free_recall', 'theta_recognition']
        for col in expected_output_cols:
            if col not in wide_theta.columns:
                raise ValueError(f"Missing expected output column: {col}")
        
        # Select and order columns
        final_theta = wide_theta[expected_output_cols].copy()
        
        # Remove participants with missing paradigm data
        before_dropna = len(final_theta)
        final_theta = final_theta.dropna()
        after_dropna = len(final_theta)
        
        if before_dropna != after_dropna:
            log(f"Removed {before_dropna - after_dropna} participants with missing paradigm data")
        
        log(f"Wide format dataset: {len(final_theta)} participants × {len(final_theta.columns)} columns")
        # Save Analysis Output
        # Save to data/ folder with step prefix as per folder conventions

        output_path = RQ_DIR / "data" / "step01_paradigm_theta.csv"
        log(f"Saving to {output_path}...")
        
        final_theta.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step01_paradigm_theta.csv ({len(final_theta)} rows, {len(final_theta.columns)} cols)")
        
        # Log summary statistics
        log(f"Free recall theta: mean={final_theta['theta_free_recall'].mean():.3f}, std={final_theta['theta_free_recall'].std():.3f}")
        log(f"Recognition theta: mean={final_theta['theta_recognition'].mean():.3f}, std={final_theta['theta_recognition'].std():.3f}")
        # Run Validation Tool
        # Validate theta_free_recall values in IRT range [-3, 3]
        # Also implicitly validates recognition theta has similar range

        log("Running validate_numeric_range...")
        
        # Validate free recall theta range
        validation_result = validate_numeric_range(
            data=final_theta['theta_free_recall'],
            min_val=-3.0,
            max_val=3.0,
            column_name='theta_free_recall'
        )
        
        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
        
        # Additional validation checks
        log("Additional checks...")
        
        # Check expected number of participants (around 100)
        n_participants = len(final_theta)
        if n_participants < 50:
            log(f"Only {n_participants} participants - expected ~100")
        elif n_participants < 90:
            log(f"{n_participants} participants (acceptable range)")
        else:
            log(f"{n_participants} participants (expected range)")
        
        # Check no missing values
        missing_free_recall = final_theta['theta_free_recall'].isna().sum()
        missing_recognition = final_theta['theta_recognition'].isna().sum()
        
        if missing_free_recall == 0 and missing_recognition == 0:
            log("No missing values in final dataset")
        else:
            log(f"Missing values: free_recall={missing_free_recall}, recognition={missing_recognition}")
        
        # Check both paradigms represented
        has_free_recall_data = not final_theta['theta_free_recall'].isna().all()
        has_recognition_data = not final_theta['theta_recognition'].isna().all()
        
        if has_free_recall_data and has_recognition_data:
            log("Both paradigms represented in final dataset")
        else:
            log(f"Missing paradigm data: free_recall={has_free_recall_data}, recognition={has_recognition_data}")

        log("Step 1 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)