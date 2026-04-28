#!/usr/bin/env python3
"""extract_domain_theta_scores: Extract Where and What domain-specific theta scores from Ch5 analyses."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_domain_theta_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Custom Analysis Functions (Due to Signature Mismatches)

def extract_domain_theta_scores_custom(input_path: str, domains: List[str]) -> pd.DataFrame:
    """
    Custom function to extract and aggregate domain theta scores.
    
    Parameters
    ----------
    input_path : str
        Path to Ch5 theta scores CSV
    domains : List[str]
        List of domains to extract (e.g., ['Where', 'What'])
    
    Returns
    -------
    pd.DataFrame
        Aggregated domain scores with columns: UID, {domain}_mean, ...
    """
    # Load Ch5 theta scores
    df_theta = pd.read_csv(input_path)
    log(f"Ch5 theta scores: {len(df_theta)} rows, columns: {df_theta.columns.tolist()}")
    
    # Extract UID from composite_ID (e.g., A010_1 -> A010)
    df_theta['UID'] = df_theta['composite_ID'].str.split('_').str[0]
    log(f"Extracted UID from composite_ID")
    
    # Check unique participants
    n_participants = df_theta['UID'].nunique()
    log(f"Found {n_participants} unique participants")
    
    # Prepare domain columns mapping (handle case sensitivity)
    domain_cols = {}
    for domain in domains:
        # Map domain names to actual column names (lowercase theta_)
        col_name = f"theta_{domain.lower()}"
        if col_name in df_theta.columns:
            domain_cols[domain] = col_name
        else:
            log(f"Domain column not found: {col_name}")
            raise ValueError(f"Column {col_name} not found in theta scores")
    
    log(f"Domain column mapping: {domain_cols}")
    
    # Group by UID and compute mean theta scores across tests
    agg_data = []
    for uid, group in df_theta.groupby('UID'):
        row_data = {'UID': uid}
        
        for domain, col_name in domain_cols.items():
            # Compute mean across all tests for this participant and domain
            domain_mean = group[col_name].mean()
            row_data[f"{domain}_mean"] = domain_mean
        
        agg_data.append(row_data)
    
    df_aggregated = pd.DataFrame(agg_data)
    log(f"Created aggregated dataset: {len(df_aggregated)} participants")
    
    return df_aggregated

def validate_domain_theta_scores(df: pd.DataFrame, theta_range: Tuple[float, float] = (-3.0, 3.0)) -> Dict[str, Any]:
    """
    Custom validation for domain theta scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Domain theta scores to validate
    theta_range : Tuple[float, float]
        Valid IRT theta range (min, max)
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    results = {
        'valid': True,
        'n_participants': len(df),
        'missing_values': {},
        'out_of_range': {},
        'variance_check': {},
        'issues': []
    }
    
    min_theta, max_theta = theta_range
    
    # Check participant count
    if len(df) < 100:
        results['valid'] = False
        results['issues'].append(f"Insufficient participants: {len(df)} < 100")
    
    # Check for missing values
    for col in df.columns:
        if col != 'UID':
            missing_count = df[col].isna().sum()
            results['missing_values'][col] = missing_count
            if missing_count > 0:
                results['valid'] = False
                results['issues'].append(f"Missing values in {col}: {missing_count}")
    
    # Check theta ranges
    for col in df.columns:
        if col != 'UID' and 'mean' in col:
            out_of_range = ((df[col] < min_theta) | (df[col] > max_theta)).sum()
            results['out_of_range'][col] = out_of_range
            if out_of_range > 0:
                results['valid'] = False
                results['issues'].append(f"Out of range values in {col}: {out_of_range}")
            
            # Check variance
            variance = df[col].var()
            results['variance_check'][col] = variance
            if variance < 0.1:  # Very low variance threshold
                results['issues'].append(f"Low variance in {col}: {variance:.4f}")
    
    # Check for duplicate UIDs
    duplicates = df['UID'].duplicated().sum()
    if duplicates > 0:
        results['valid'] = False
        results['issues'].append(f"Duplicate UIDs found: {duplicates}")
    
    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_domain_theta_scores")
        # Load Ch5 Domain Theta Scores
        
        log("Loading Ch5 theta scores...")
        ch5_theta_path = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
        
        if not ch5_theta_path.exists():
            raise FileNotFoundError(f"Ch5 theta scores not found: {ch5_theta_path}")
        
        log(f"Ch5 theta file: {ch5_theta_path}")
        # Extract and Aggregate Domain Scores
        
        log("Extracting and aggregating domain theta scores...")
        domains = ["Where", "What"]  # Focus on spatial and object domains
        
        df_domain_scores = extract_domain_theta_scores_custom(
            input_path=str(ch5_theta_path),
            domains=domains
        )
        log("Domain score extraction complete")
        # Save Domain Theta Scores
        # These outputs will be used by: Step 03 (merge with BVMT scores)
        
        output_path = RQ_DIR / "data" / "step01_domain_theta_scores.csv"
        log(f"Saving domain theta scores to {output_path}...")
        
        # Ensure data directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with UTF-8 encoding
        df_domain_scores.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(df_domain_scores)} rows, {len(df_domain_scores.columns)} cols)")
        
        # Log sample of data for verification
        log("First 3 rows:")
        for i, row in df_domain_scores.head(3).iterrows():
            log(f"  {dict(row)}")
        # Run Validation
        # Validates: Participant count, theta ranges, missing values, variance
        # Threshold: 100 participants, theta in [-3, 3], adequate variance
        
        log("Validating domain theta scores...")
        validation_result = validate_domain_theta_scores(
            df=df_domain_scores,
            theta_range=(-3.0, 3.0)  # Standard IRT theta range
        )

        # Report validation results
        log(f"Overall valid: {validation_result['valid']}")
        log(f"Participants: {validation_result['n_participants']}")
        log(f"Missing values: {validation_result['missing_values']}")
        log(f"Out of range: {validation_result['out_of_range']}")
        log(f"Variance check: {validation_result['variance_check']}")
        
        if validation_result['issues']:
            log("Issues found:")
            for issue in validation_result['issues']:
                log(f"  - {issue}")
        
        if not validation_result['valid']:
            log("Validation failed - see issues above")
            sys.exit(1)
        else:
            log("All checks passed")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)