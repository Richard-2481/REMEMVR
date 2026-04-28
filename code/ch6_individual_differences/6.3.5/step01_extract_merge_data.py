#!/usr/bin/env python3
"""extract_merge_data: Extract and merge theta scores, confidence ratings, and cognitive reserve indicators"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_merge_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract and merge theta scores, confidence, and cognitive reserve data")
        # Load and Process Ch5 Theta Scores

        log("Loading Ch5 theta scores...")
        ch5_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        ch5_df = pd.read_csv(ch5_path)
        log(f"Ch5 data ({len(ch5_df)} rows, {len(ch5_df.columns)} cols)")
        log(f"Ch5 columns: {list(ch5_df.columns)}")
        
        # Aggregate Theta_All by UID (mean across test sessions)
        # Note: Ch5 file has 'Theta_All' not 'theta_all' - case sensitive fix
        ch5_agg = ch5_df.groupby('UID')['Theta_All'].mean().reset_index()
        ch5_agg.rename(columns={'Theta_All': 'theta_all'}, inplace=True)
        log(f"Ch5 aggregated to {len(ch5_agg)} participants")
        # Load and Process Ch6 Confidence Scores

        log("Loading Ch6 confidence scores...")
        ch6_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv"
        ch6_df = pd.read_csv(ch6_path)
        log(f"Ch6 data ({len(ch6_df)} rows, {len(ch6_df.columns)} cols)")
        log(f"Ch6 columns: {list(ch6_df.columns)}")
        
        # Extract UID from composite_ID (format: "A001_1" -> "A001")
        # Note: Ch6 file has 'composite_ID' not 'UID' - format fix needed
        ch6_df['UID'] = ch6_df['composite_ID'].str.split('_').str[0]
        
        # Rename theta_All to confidence_theta for clarity
        # Note: Ch6 file has 'theta_All' not 'theta_confidence' - column name fix
        ch6_df.rename(columns={'theta_All': 'confidence_theta'}, inplace=True)
        
        # Aggregate Ch6 across tests (mean confidence per participant)
        ch6_processed = ch6_df.groupby('UID')['confidence_theta'].mean().reset_index()
        log(f"Ch6 UIDs extracted and aggregated, {len(ch6_processed)} participants")
        # Load Cognitive Reserve Data

        log("Loading cognitive reserve data...")
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        dfnonvr_df = pd.read_csv(dfnonvr_path)
        log(f"dfnonvr data ({len(dfnonvr_df)} rows, {len(dfnonvr_df.columns)} cols)")
        
        # Extract cognitive reserve columns
        # Note: dfnonvr.csv has 'rpm-score' not 'rpm' - exact column name needed
        cognitive_cols = ['UID', 'rpm-score', 'age', 'education']
        missing_cols = [col for col in cognitive_cols if col not in dfnonvr_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dfnonvr.csv: {missing_cols}")
        
        cognitive_df = dfnonvr_df[cognitive_cols].copy()
        cognitive_df.rename(columns={'rpm-score': 'rpm'}, inplace=True)  # Rename for output consistency
        log(f"Cognitive reserve data extracted, {len(cognitive_df)} participants")
        # Merge All Datasets

        log("Merging datasets on UID...")
        
        # Start with Ch5 theta scores
        merged_df = ch5_agg.copy()
        
        # Merge with Ch6 confidence scores
        merged_df = merged_df.merge(ch6_processed, on='UID', how='inner')
        log(f"After Ch6 merge: {len(merged_df)} participants")
        
        # Merge with cognitive reserve data
        merged_df = merged_df.merge(cognitive_df, on='UID', how='inner')
        log(f"Final merge: {len(merged_df)} participants")
        
        # Reorder columns for output specification
        final_columns = ['UID', 'theta_all', 'confidence_theta', 'education', 'rpm', 'age']
        merged_df = merged_df[final_columns]
        log("Merge complete")
        # Save Merged Output
        # Output: step01_merged_data.csv
        # Contains: All variables needed for confidence-accuracy calibration analysis
        # Columns: UID, theta_all, confidence_theta, education, rpm, age

        log(f"Saving merged dataset...")
        output_path = RQ_DIR / "data" / "step01_merged_data.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        
        # Log basic descriptive statistics
        log("Final dataset summary:")
        log(f"  Participants: {len(merged_df)}")
        log(f"  Theta range: {merged_df['theta_all'].min():.2f} to {merged_df['theta_all'].max():.2f}")
        log(f"  Confidence range: {merged_df['confidence_theta'].min():.2f} to {merged_df['confidence_theta'].max():.2f}")
        log(f"  Age range: {merged_df['age'].min():.0f} to {merged_df['age'].max():.0f}")
        log(f"  Education range: {merged_df['education'].min():.0f} to {merged_df['education'].max():.0f}")
        log(f"  RPM range: {merged_df['rpm'].min():.0f} to {merged_df['rpm'].max():.0f}")
        # Run Validation Tool
        # Validates: All required columns are present in output dataset
        # Threshold: All 6 required columns must exist

        log("Running validate_data_columns...")
        required_columns = ['UID', 'theta_all', 'confidence_theta', 'education', 'rpm', 'age']
        validation_result = validate_data_columns(
            merged_df, 
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")
            
        # Check for missing data in key variables
        missing_counts = merged_df.isnull().sum()
        if missing_counts.sum() > 0:
            log("Missing data detected:")
            for col, count in missing_counts.items():
                if count > 0:
                    log(f"  {col}: {count} missing values")
        else:
            log("No missing data in merged dataset")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)