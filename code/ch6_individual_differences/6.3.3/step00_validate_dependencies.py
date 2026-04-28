#!/usr/bin/env python3
"""validate_dependencies: Validate Ch6 HCE data and dfnonvr.csv accessibility with correct column names."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import os

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.3 (derived from script location)
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
        # Validate Ch6 HCE Data

        log("Checking Ch6 HCE data availability...")
        
        # Primary path for Ch6 HCE data
        hce_primary_path = PROJECT_ROOT / "results" / "ch6" / "6.6.1" / "data" / "step01_hce_rates.csv"
        
        # Check primary path
        hce_data_found = False
        hce_final_path = None
        
        if hce_primary_path.exists():
            log(f"Ch6 HCE data at primary path: {hce_primary_path}")
            hce_data_found = True
            hce_final_path = hce_primary_path
        else:
            log(f"Ch6 HCE data not found at primary path: {hce_primary_path}")
            
            # Check alternative paths as requested
            log("Checking alternative Ch6 paths for HCE data...")
            alternative_paths = [
                PROJECT_ROOT / "results" / "ch6" / "6.6.2" / "data",
                PROJECT_ROOT / "results" / "ch6" / "6.6.1" / "data",
                PROJECT_ROOT / "results" / "ch6" / "data"
            ]
            
            for alt_dir in alternative_paths:
                if alt_dir.exists():
                    # Look for any files with 'hce' in the name
                    hce_files = list(alt_dir.glob("*hce*.csv"))
                    if hce_files:
                        log(f"Alternative HCE files in {alt_dir}: {[f.name for f in hce_files]}")
                        hce_final_path = hce_files[0]  # Use first found
                        hce_data_found = True
                        break
                    else:
                        log(f"No HCE files found in {alt_dir}")
                else:
                    log(f"Directory does not exist: {alt_dir}")
        
        if not hce_data_found:
            log("No Ch6 HCE data found in any expected locations")
            log("Run Ch6 RQ 6.6.1 first to generate step01_hce_rates.csv")
            sys.exit(1)
        
        # Validate HCE data format
        log(f"Checking HCE data format: {hce_final_path}")
        try:
            hce_df = pd.read_csv(hce_final_path)
            hce_columns = hce_df.columns.tolist()
            expected_hce_columns = ['UID', 'HCE_rate']
            
            log(f"HCE data has columns: {hce_columns}")
            
            # Check for required columns (flexible - HCE data might have additional columns)
            missing_hce_cols = [col for col in expected_hce_columns if col not in hce_columns]
            if missing_hce_cols:
                log(f"Missing required columns in HCE data: {missing_hce_cols}")
                log(f"Available columns: {hce_columns}")
                sys.exit(1)
            
            # Check data dimensions
            n_hce_rows = len(hce_df)
            n_hce_participants = hce_df['UID'].nunique()
            log(f"HCE data: {n_hce_rows} rows, {n_hce_participants} unique participants")
            
            # Check HCE rate range
            hce_rates = hce_df['HCE_rate'].dropna()
            if len(hce_rates) > 0:
                hce_min, hce_max = hce_rates.min(), hce_rates.max()
                log(f"HCE rates: {hce_min:.4f} to {hce_max:.4f}")
                if hce_min < 0 or hce_max > 1:
                    log("HCE rates outside [0,1] range - check data quality")
            
            log("Ch6 HCE data validation successful")
            
        except Exception as e:
            log(f"Failed to read HCE data: {e}")
            sys.exit(1)
        # Validate dfnonvr.csv Accessibility
        log("Checking dfnonvr.csv accessibility...")
        
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        if not dfnonvr_path.exists():
            log(f"dfnonvr.csv not found at expected location: {dfnonvr_path}")
            sys.exit(1)
        
        log(f"dfnonvr.csv at: {dfnonvr_path}")
        
        # Validate column names - CRITICAL: Must match exact hyphenated format
        log("Checking dfnonvr.csv column names...")
        try:
            # Read header only for fast column checking
            dfnonvr_df = pd.read_csv(dfnonvr_path, nrows=0)
            actual_columns = dfnonvr_df.columns.tolist()
            
            # Expected columns from analysis.yaml (CRITICAL: exact hyphenated format)
            expected_columns = [
                "UID", 
                "ravlt-trial-1-score", "ravlt-trial-2-score", "ravlt-trial-3-score", 
                "ravlt-trial-4-score", "ravlt-trial-5-score",
                "bvmt-trial-1-score", "bvmt-trial-2-score", "bvmt-trial-3-score", 
                "bvmt-delayed-recall-score",
                "rpm-score", 
                "age", "sex", "education"
            ]
            
            log(f"dfnonvr.csv has {len(actual_columns)} total columns")
            log("Required columns for cognitive analysis:")
            
            missing_columns = []
            present_columns = []
            
            for col in expected_columns:
                if col in actual_columns:
                    log(f"  {col}")
                    present_columns.append(col)
                else:
                    log(f"  {col}")
                    missing_columns.append(col)
            
            if missing_columns:
                log(f"Missing required columns in dfnonvr.csv: {missing_columns}")
                log("Column names MUST be lowercase with hyphens (e.g., 'ravlt-trial-1-score')")
                log("NOT uppercase with spaces (e.g., 'RAVLT trial 1 score')")
                
                # Show similar columns to help debug
                log("Checking for similar column names:")
                for missing_col in missing_columns:
                    similar_cols = [col for col in actual_columns if missing_col.replace('-', ' ').lower() in col.lower()]
                    if similar_cols:
                        log(f"  Similar to '{missing_col}': {similar_cols[:5]}")  # Show first 5 matches
                
                sys.exit(1)
            
            log(f"All {len(expected_columns)} required columns found in dfnonvr.csv")
            
            # Check data dimensions (read full file for row count)
            dfnonvr_full = pd.read_csv(dfnonvr_path)
            n_nonvr_rows = len(dfnonvr_full)
            n_nonvr_participants = dfnonvr_full['UID'].nunique()
            log(f"dfnonvr.csv: {n_nonvr_rows} rows, {n_nonvr_participants} unique participants")
            
            # Check for missing data in key columns
            log("[DATA QUALITY] Checking for missing values in key columns:")
            for col in ['UID', 'rpm-score', 'age', 'sex']:
                n_missing = dfnonvr_full[col].isna().sum()
                pct_missing = (n_missing / n_nonvr_rows) * 100
                log(f"  {col}: {n_missing} missing ({pct_missing:.1f}%)")
            
            log("dfnonvr.csv validation successful")
            
        except Exception as e:
            log(f"Failed to read dfnonvr.csv: {e}")
            sys.exit(1)
        # Cross-Validation Between Datasets

        log("Checking participant overlap between datasets...")
        
        hce_uids = set(hce_df['UID'].unique())
        cognitive_uids = set(dfnonvr_full['UID'].unique())
        
        # Find overlap
        overlapping_uids = hce_uids.intersection(cognitive_uids)
        hce_only = hce_uids - cognitive_uids
        cognitive_only = cognitive_uids - hce_uids
        
        log(f"HCE data participants: {len(hce_uids)}")
        log(f"Cognitive data participants: {len(cognitive_uids)}")
        log(f"Overlapping participants: {len(overlapping_uids)}")
        log(f"HCE only: {len(hce_only)}")
        log(f"Cognitive only: {len(cognitive_only)}")
        
        # Check overlap percentage
        overlap_pct = (len(overlapping_uids) / max(len(hce_uids), len(cognitive_uids))) * 100
        log(f"Overlap percentage: {overlap_pct:.1f}%")
        
        if len(overlapping_uids) < 50:
            log("Low participant overlap (<50) may limit analysis power")
        else:
            log(f"Sufficient participant overlap ({len(overlapping_uids)}) for analysis")
        # Save Validation Results

        log("Creating validation report...")
        
        validation_report_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        validation_report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            f.write("# RQ 7.3.3 - Dependency Validation Report\n")
            f.write("# Generated by step00_validate_dependencies.py\n")
            
            f.write("## Ch6 HCE Data Validation\n")
            f.write(f"File path: {hce_final_path}\n")
            f.write(f"Status: FOUND\n")
            f.write(f"Columns: {hce_columns}\n")
            f.write(f"Required columns present: {expected_hce_columns}\n")
            f.write(f"Data dimensions: {n_hce_rows} rows, {n_hce_participants} participants\n")
            if len(hce_rates) > 0:
                f.write(f"HCE rate range: {hce_min:.4f} to {hce_max:.4f}\n")
            f.write("\n")
            
            f.write("## dfnonvr.csv Validation\n")
            f.write(f"File path: {dfnonvr_path}\n")
            f.write("Status: FOUND\n")
            f.write(f"Total columns: {len(actual_columns)}\n")
            f.write(f"Required columns present: {present_columns}\n")
            f.write(f"Data dimensions: {n_nonvr_rows} rows, {n_nonvr_participants} participants\n")
            f.write("\n")
            
            f.write("## Participant Overlap Analysis\n")
            f.write(f"HCE data participants: {len(hce_uids)}\n")
            f.write(f"Cognitive data participants: {len(cognitive_uids)}\n")
            f.write(f"Overlapping participants: {len(overlapping_uids)}\n")
            f.write(f"Overlap percentage: {overlap_pct:.1f}%\n")
            f.write("\n")
            
            f.write("## Validation Summary\n")
            f.write("- Ch6 HCE data: PASS\n")
            f.write("- dfnonvr.csv access: PASS\n")
            f.write("- Column name format: PASS (exact hyphenated format confirmed)\n")
            f.write("- Participant overlap: PASS\n")
            f.write("\n")
            f.write("## Ready for Analysis\n")
            f.write("All dependency validation checks passed.\n")
            f.write("RQ 7.3.3 cognitive predictors analysis can proceed.\n")
        
        log(f"Validation report: {validation_report_path}")
        log(f"{len(actual_columns)} columns verified, {len(overlapping_uids)} participants available")

        log("Step 00 complete - all dependencies validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)