#!/usr/bin/env python3
"""validate_dependencies: Validate Ch5 domain outputs and dfnonvr.csv BVMT data exist before proceeding"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import check_file_exists, validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.2 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: validate_dependencies")
        # Validate Ch5 Domain Outputs

        log("Checking Ch5 domain output files...")
        
        # Primary path and fallback paths for Ch5 data
        ch5_paths = [
            PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv",
            PROJECT_ROOT / "results" / "ch5" / "5.2.2" / "data" / "step03_theta_scores.csv",
            PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "theta_scores.csv"
        ]
        
        ch5_file_found = None
        ch5_validation_result = None
        
        for ch5_path in ch5_paths:
            log(f"Trying Ch5 path: {ch5_path}")
            file_result = check_file_exists(file_path=ch5_path, min_size_bytes=1000)  # Minimum 1KB
            
            if file_result.get('valid', False):
                log(f"Ch5 file located: {ch5_path}")
                ch5_file_found = ch5_path
                ch5_validation_result = file_result
                break
            else:
                log(f"Ch5 file not found: {file_result.get('message', 'Unknown error')}")
        
        if ch5_file_found is None:
            log("No Ch5 domain output files found in any fallback location")
            raise FileNotFoundError("Ch5 domain analysis not complete - theta scores file missing")
        # Validate Ch5 Column Structure
        # Critical: Ch5 has lowercase column names (theta_what, theta_where not theta_What, theta_Where)
        
        log("Checking Ch5 column structure...")
        ch5_df = pd.read_csv(ch5_file_found, nrows=5)  # Read header + few rows for validation
        
        # Ch5 may have composite_ID instead of UID
        if 'composite_ID' in ch5_df.columns and 'UID' not in ch5_df.columns:
            log("Ch5 has composite_ID instead of UID - will extract UID during analysis")
            uid_col = 'composite_ID'
        else:
            uid_col = 'UID'
        
        # Note: Ch5 uses lowercase - theta_what, theta_where (from lessons learned)
        required_ch5_columns = ["theta_what", "theta_where"]  # Don't check UID column name
        ch5_column_result = validate_data_columns(df=ch5_df, required_columns=required_ch5_columns)
        
        if not ch5_column_result.get('valid', False):
            missing_cols = ch5_column_result.get('missing_columns', [])
            existing_cols = ch5_df.columns.tolist()
            log(f"Ch5 missing required columns: {missing_cols}")
            log(f"Ch5 existing columns: {existing_cols}")
            raise ValueError(f"Ch5 theta scores missing required columns: {missing_cols}")
        
        log(f"Ch5 columns validated: {uid_col}, {required_ch5_columns}")
        # Validate BVMT Data Source
        
        log("Checking dfnonvr.csv BVMT data...")
        bvmt_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        bvmt_file_result = check_file_exists(file_path=bvmt_path, min_size_bytes=10000)  # Minimum 10KB
        
        if not bvmt_file_result.get('valid', False):
            log(f"BVMT data file not found: {bvmt_file_result.get('message', 'Unknown error')}")
            raise FileNotFoundError("dfnonvr.csv not found - cognitive test data missing")
        
        log(f"BVMT data located: {bvmt_path} ({bvmt_file_result.get('size_bytes', 0)} bytes)")
        # Validate BVMT Column Structure
        # Critical: Column name is 'bvmt-total-recall' (from DATA_DICTIONARY.md)
        
        log("Checking BVMT column structure...")
        bvmt_df = pd.read_csv(bvmt_path, nrows=5)  # Read header + few rows for validation
        
        required_bvmt_columns = ["UID", "bvmt-total-recall"]
        bvmt_column_result = validate_data_columns(df=bvmt_df, required_columns=required_bvmt_columns)
        
        if not bvmt_column_result.get('valid', False):
            missing_cols = bvmt_column_result.get('missing_columns', [])
            existing_cols = bvmt_df.columns.tolist()
            log(f"BVMT missing required columns: {missing_cols}")
            log(f"BVMT existing columns: {existing_cols}")
            raise ValueError(f"dfnonvr.csv missing required columns: {missing_cols}")
        
        log(f"BVMT columns validated: {required_bvmt_columns}")
        # Validate Participant Counts
        
        log("Checking participant counts...")
        
        # Load full datasets to check participant counts
        ch5_full_df = pd.read_csv(ch5_file_found)
        bvmt_full_df = pd.read_csv(bvmt_path)
        
        # Handle composite_ID in Ch5 if needed
        if 'composite_ID' in ch5_full_df.columns and 'UID' not in ch5_full_df.columns:
            # Extract UID from composite_ID (e.g., A010_1 -> A010)
            ch5_full_df['UID'] = ch5_full_df['composite_ID'].str.split('_').str[0]
        
        # Remove any rows with missing UIDs or key data
        ch5_complete = ch5_full_df.dropna(subset=["UID", "theta_what", "theta_where"])
        bvmt_complete = bvmt_full_df.dropna(subset=["UID", "bvmt-total-recall"])
        
        n_ch5_participants = len(ch5_complete)
        n_bvmt_participants = len(bvmt_complete)
        
        log(f"Ch5 participants with complete data: {n_ch5_participants}")
        log(f"BVMT participants with complete data: {n_bvmt_participants}")
        
        # Check minimum participant threshold
        min_participants = 100
        if n_ch5_participants < min_participants:
            log(f"Ch5 insufficient participants: {n_ch5_participants} < {min_participants}")
            raise ValueError(f"Ch5 dataset has only {n_ch5_participants} participants (minimum {min_participants} required)")
            
        if n_bvmt_participants < min_participants:
            log(f"BVMT insufficient participants: {n_bvmt_participants} < {min_participants}")
            raise ValueError(f"BVMT dataset has only {n_bvmt_participants} participants (minimum {min_participants} required)")

        # Check participant overlap
        ch5_uids = set(ch5_complete['UID'].astype(str))
        bvmt_uids = set(bvmt_complete['UID'].astype(str))
        overlap_uids = ch5_uids.intersection(bvmt_uids)
        n_overlap = len(overlap_uids)
        
        log(f"Overlapping participants: {n_overlap}")
        
        if n_overlap < min_participants:
            log(f"Insufficient overlap: {n_overlap} < {min_participants}")
            raise ValueError(f"Only {n_overlap} participants in both datasets (minimum {min_participants} required)")

        log(f"Participant counts validated: {n_overlap} participants ready for analysis")
        # Save Validation Summary
        # Output: Summary of all validation results for downstream reference
        
        log("Writing validation summary...")
        
        validation_summary = f"""DEPENDENCY VALIDATION SUMMARY - RQ 7.4.2
Step: step00_validate_dependencies

FILES VALIDATED:
  Ch5 Domain Data: {ch5_file_found}
    Size: {ch5_validation_result.get('size_bytes', 0)} bytes
    Participants: {n_ch5_participants}
    Columns: {required_ch5_columns}
    
  BVMT Cognitive Data: {bvmt_path}
    Size: {bvmt_file_result.get('size_bytes', 0)} bytes
    Participants: {n_bvmt_participants}
    Columns: {required_bvmt_columns}

PARTICIPANT OVERLAP:
  Ch5 UIDs: {len(ch5_uids)}
  BVMT UIDs: {len(bvmt_uids)}
  Overlap: {n_overlap}
  Analysis-Ready: {n_overlap} participants

VALIDATION RESULTS:
  Ch5 domain files located and accessible
  dfnonvr.csv contains bvmt-total-recall column
  Minimum 100 participants present in both sources
  Required columns present and accessible

STATUS: ALL DEPENDENCIES VALIDATED - READY FOR ANALYSIS

Next Steps:
  1. Extract domain theta scores (step01)
  2. Extract BVMT scores (step02) 
  3. Merge datasets (step03)
  4. Compute correlations (step04)
  5. Run Steiger test (step05)
"""

        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(validation_summary)
        
        log(f"Validation summary: {output_path}")

        log("Step 00 complete - all dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)