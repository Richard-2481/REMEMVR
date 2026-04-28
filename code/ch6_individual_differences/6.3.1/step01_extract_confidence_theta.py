#!/usr/bin/env python3
"""extract_confidence_theta: Load and validate Ch6-derived confidence theta scores for N=100 participants."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_confidence_theta.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Custom Validation Functions

def validate_theta_range(df, theta_col, min_val=-3.0, max_val=3.0):
    """Validate theta scores are in valid IRT range."""
    valid_range = (df[theta_col] >= min_val) & (df[theta_col] <= max_val)
    n_valid = valid_range.sum()
    n_total = len(df)
    
    result = {
        'valid': n_valid == n_total,
        'n_valid': int(n_valid),
        'n_total': int(n_total),
        'min_observed': float(df[theta_col].min()),
        'max_observed': float(df[theta_col].max()),
        'mean_observed': float(df[theta_col].mean()),
        'message': f'{n_valid}/{n_total} theta scores in range [{min_val}, {max_val}]'
    }
    return result

def validate_se_range(df, se_col, min_val=0.01, max_val=2.0):
    """Validate standard errors are positive and reasonable."""
    valid_range = (df[se_col] >= min_val) & (df[se_col] <= max_val)
    n_valid = valid_range.sum()
    n_total = len(df)
    
    result = {
        'valid': n_valid == n_total,
        'n_valid': int(n_valid),
        'n_total': int(n_total),
        'min_observed': float(df[se_col].min()),
        'max_observed': float(df[se_col].max()),
        'mean_observed': float(df[se_col].mean()),
        'message': f'{n_valid}/{n_total} standard errors in range [{min_val}, {max_val}]'
    }
    return result

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: Extract Confidence Theta")
        # Load Input Data from Ch6

        log("Loading Ch6 confidence theta data...")
        
        # Use actual file name from Ch6 (corrected from specification)
        input_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Ch6 input file not found: {input_path}")
            
        ch6_theta_df = pd.read_csv(input_path)
        log(f"Ch6 theta data ({len(ch6_theta_df)} rows, {len(ch6_theta_df.columns)} cols)")
        log(f"Columns: {list(ch6_theta_df.columns)}")
        # Data Processing - Extract UIDs and Aggregate by Participant
        # Transform: 400 rows (composite_ID format: A010_T1) → 100 rows (UID format: A010)
        # Aggregate: Mean theta across 4 tests per participant

        log("Extracting UIDs from composite_ID...")
        
        # Extract UID from composite_ID (e.g., A010_T1 → A010)
        ch6_theta_df['UID'] = ch6_theta_df['composite_ID'].str.extract(r'([A-Z]\d{3})')
        
        # Check extraction worked
        n_unique_uids = ch6_theta_df['UID'].nunique()
        n_null_uids = ch6_theta_df['UID'].isnull().sum()
        
        log(f"Extracted {n_unique_uids} unique UIDs")
        if n_null_uids > 0:
            log(f"{n_null_uids} rows failed UID extraction")
            
        # Aggregate confidence theta across tests (mean)
        log("Aggregating theta scores by participant...")
        
        confidence_theta_agg = ch6_theta_df.groupby('UID').agg({
            'theta_All': 'mean',     # Mean theta across 4 tests
            'se_All': 'mean'         # Mean SE across 4 tests (conservative)
        }).reset_index()
        
        # Rename columns to match expected output format
        confidence_theta_agg = confidence_theta_agg.rename(columns={
            'theta_All': 'confidence_theta',
            'se_All': 'se_theta'
        })
        
        log(f"Aggregated to {len(confidence_theta_agg)} participants")
        # Save Analysis Output
        # Output: participant-level confidence theta for downstream regression

        output_path = RQ_DIR / "data" / "step01_confidence_theta.csv"
        
        log(f"Saving participant-level confidence data...")
        confidence_theta_agg.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(confidence_theta_agg)} rows, {len(confidence_theta_agg.columns)} cols)")
        
        # Log summary statistics
        log(f"Confidence theta - Mean: {confidence_theta_agg['confidence_theta'].mean():.3f}, "
            f"SD: {confidence_theta_agg['confidence_theta'].std():.3f}, "
            f"Range: [{confidence_theta_agg['confidence_theta'].min():.3f}, {confidence_theta_agg['confidence_theta'].max():.3f}]")
        # Run Custom Validation
        # Validate: theta in IRT range, SEs reasonable, complete data
        # Using custom validation due to tools.validation signature mismatch

        log("Running custom validation...")
        
        # Validate theta scores in IRT range [-3, 3]
        theta_validation = validate_theta_range(confidence_theta_agg, 'confidence_theta')
        log(f"Theta range: {theta_validation['message']}")
        if not theta_validation['valid']:
            log(f"Some theta scores outside valid range")
            
        # Validate standard errors
        se_validation = validate_se_range(confidence_theta_agg, 'se_theta')
        log(f"SE range: {se_validation['message']}")
        if not se_validation['valid']:
            log(f"Some standard errors outside reasonable range")
            
        # Validate completeness
        n_complete = confidence_theta_agg.dropna().shape[0]
        n_total = len(confidence_theta_agg)
        log(f"Complete cases: {n_complete}/{n_total}")
        
        # Validate expected sample size (should be ~100 participants)
        if n_total < 90:
            log(f"Sample size ({n_total}) lower than expected (~100)")
        elif n_total > 110:
            log(f"Sample size ({n_total}) higher than expected (~100)")
        else:
            log(f"Sample size appropriate: {n_total} participants")

        # Overall validation status
        all_valid = (theta_validation['valid'] and se_validation['valid'] and 
                    n_complete == n_total and 90 <= n_total <= 110)
        
        if all_valid:
            log("All validation criteria passed")
        else:
            log("Some validation warnings - check log details")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)