#!/usr/bin/env python3
"""extract_merge_data: Extract theta scores from Ch5 5.1.1 and merge with DASS subscales from dfnonvr.csv."""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_merge_data.log"

# Create directories if they don't exist
(RQ_DIR / "data").mkdir(parents=True, exist_ok=True)
(RQ_DIR / "logs").mkdir(parents=True, exist_ok=True)


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_merge_data")
        # Load Ch5 Theta Scores

        log("Loading Ch5 theta scores...")
        theta_path = PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
        
        if not theta_path.exists():
            raise FileNotFoundError(f"Ch5 theta file not found: {theta_path}")
        
        theta_df = pd.read_csv(theta_path)
        log(f"Ch5 theta scores ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        log(f"Theta columns: {theta_df.columns.tolist()}")
        log(f"Unique UIDs: {theta_df['UID'].nunique()}, Unique tests: {sorted(theta_df['test'].unique())}")
        # Aggregate Theta Scores by Participant

        log("Aggregating theta scores by participant...")
        
        # Calculate mean theta across tests for each participant
        # Standard practice: Use mean for stable ability estimate across test sessions
        theta_aggregated = theta_df.groupby('UID')['Theta_All'].agg(['mean', 'count', 'std']).reset_index()
        theta_aggregated.columns = ['UID', 'theta_all', 'n_tests', 'theta_sd']
        
        log(f"Theta scores: {len(theta_aggregated)} participants")
        log(f"Tests per participant: min={theta_aggregated['n_tests'].min()}, max={theta_aggregated['n_tests'].max()}, mean={theta_aggregated['n_tests'].mean():.1f}")
        log(f"Theta range: {theta_aggregated['theta_all'].min():.3f} to {theta_aggregated['theta_all'].max():.3f}")
        # Load DASS Subscales and Controls from dfnonvr.csv

        log("Loading DASS subscales and controls from dfnonvr.csv...")
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        if not dfnonvr_path.exists():
            raise FileNotFoundError(f"dfnonvr.csv not found: {dfnonvr_path}")
        
        # Load only required columns for efficiency
        dass_controls_cols = ['UID', 'total-dass-depression-items', 'total-dass-anxiety-items', 
                             'total-dass-stress-items', 'age', 'nart-score']
        
        dfnonvr = pd.read_csv(dfnonvr_path, usecols=dass_controls_cols)
        log(f"dfnonvr.csv subset ({len(dfnonvr)} rows, {len(dfnonvr.columns)} cols)")
        
        # Rename columns to standardized analysis names
        column_mapping = {
            'total-dass-depression-items': 'dass_dep',
            'total-dass-anxiety-items': 'dass_anx', 
            'total-dass-stress-items': 'dass_str',
            'nart-score': 'nart_score'
        }
        dfnonvr = dfnonvr.rename(columns=column_mapping)
        log(f"Columns standardized: {dfnonvr.columns.tolist()}")
        # Merge Datasets and Create Analysis Dataset
        # Validates: Complete cases only (no missing data for hierarchical regression)

        log("Merging theta scores with DASS subscales and controls...")
        
        # Inner join to ensure complete cases only
        analysis_df = pd.merge(theta_aggregated[['UID', 'theta_all']], dfnonvr, on='UID', how='inner')
        log(f"Analysis dataset: {len(analysis_df)} complete cases")
        
        # Check for missing data
        missing_counts = analysis_df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            log(f"Missing data found:")
            for col, count in missing_counts.items():
                if count > 0:
                    log(f"  {col}: {count} missing values")
            
            # Remove rows with any missing data
            before_dropna = len(analysis_df)
            analysis_df = analysis_df.dropna()
            after_dropna = len(analysis_df)
            log(f"Dropped {before_dropna - after_dropna} rows with missing data")
        else:
            log(f"No missing data found - all {len(analysis_df)} cases complete")
        # Save Analysis Dataset
        # These outputs will be used by: Step 02 descriptive statistics and all downstream analyses

        log(f"Saving analysis dataset...")
        # Output: Analysis dataset with standardized variable names
        # Contains: Participant-level data ready for hierarchical regression
        # Columns: UID, theta_all (outcome), DASS subscales + controls (predictors)
        analysis_output_path = RQ_DIR / "data" / "step01_analysis_dataset.csv"
        analysis_df.to_csv(analysis_output_path, index=False, encoding='utf-8')
        log(f"{analysis_output_path} ({len(analysis_df)} rows, {len(analysis_df.columns)} cols)")

        # Save extraction log with detailed merge statistics
        log_output_path = RQ_DIR / "data" / "step01_extraction_log.txt"
        with open(log_output_path, 'w', encoding='utf-8') as f:
            f.write("RQ 7.5.2 Step 01 - Data Extraction and Merge Log\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"INPUT DATA SUMMARY:\n")
            f.write(f"Ch5 theta scores: {len(theta_df)} rows, {theta_df['UID'].nunique()} unique UIDs\n")
            f.write(f"dfnonvr.csv: {len(dfnonvr)} rows\n\n")
            
            f.write(f"THETA AGGREGATION:\n")
            f.write(f"Aggregated to: {len(theta_aggregated)} participants\n")
            f.write(f"Tests per participant: {theta_aggregated['n_tests'].min()}-{theta_aggregated['n_tests'].max()} (mean: {theta_aggregated['n_tests'].mean():.1f})\n")
            f.write(f"Theta range: {theta_aggregated['theta_all'].min():.3f} to {theta_aggregated['theta_all'].max():.3f}\n\n")
            
            f.write(f"MERGE RESULTS:\n")
            f.write(f"Final sample size: {len(analysis_df)} complete cases\n")
            f.write(f"Retention rate: {len(analysis_df)/theta_aggregated['UID'].nunique()*100:.1f}% of theta participants\n\n")
            
            f.write(f"VARIABLE RANGES (for validation):\n")
            for col in ['theta_all', 'dass_dep', 'dass_anx', 'dass_str', 'age', 'nart_score']:
                if col in analysis_df.columns:
                    f.write(f"{col}: {analysis_df[col].min():.2f} to {analysis_df[col].max():.2f} (mean: {analysis_df[col].mean():.2f})\n")
            
        log(f"{log_output_path}")
        # Run Validation Tool
        # Validates: Required columns present, sample size adequate, variable ranges sensible
        # Threshold: Sample size 90-100 expected

        log("Running validate_data_columns...")
        
        # Define validation criteria
        required_columns = ['UID', 'theta_all', 'dass_dep', 'dass_anx', 'dass_str', 'age', 'nart_score']
        
        validation_result = validate_data_columns(
            df=analysis_df,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Additional custom validation for this step
        log("Additional checks...")
        
        # Sample size check
        if len(analysis_df) < 90:
            log(f"Sample size {len(analysis_df)} below expected minimum of 90")
        elif len(analysis_df) > 100:
            log(f"Sample size {len(analysis_df)} above expected maximum of 100")
        else:
            log(f"Sample size {len(analysis_df)} within expected range [90, 100]")
        
        # Range checks
        if analysis_df['theta_all'].min() < -3.0 or analysis_df['theta_all'].max() > 3.0:
            log(f"Theta values outside expected range [-3.0, 3.0]: {analysis_df['theta_all'].min():.3f} to {analysis_df['theta_all'].max():.3f}")
        else:
            log(f"Theta range within expected bounds [-3.0, 3.0]")
            
        dass_cols = ['dass_dep', 'dass_anx', 'dass_str']
        for col in dass_cols:
            if analysis_df[col].min() < 0 or analysis_df[col].max() > 21:
                log(f"{col} outside DASS-21 range [0, 21]: {analysis_df[col].min()} to {analysis_df[col].max()}")
            else:
                log(f"{col} within DASS-21 range [0, 21]")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)