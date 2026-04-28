#!/usr/bin/env python3
"""
Step 00: Validate Cross-RQ Dependencies
RQ: ch7/7.1.3
Purpose: Verify required Ch5 domain-specific outputs exist before proceeding
Output: results/ch7/7.1.3/data/step00_dependency_validation.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Go up 4 levels from code file
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"
OUTPUT_FILE = RQ_DIR / "data" / "step00_dependency_validation.txt"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Validate Cross-RQ Dependencies")
        log(f"RQ Directory: {RQ_DIR}")
        log(f"Output will be saved to: {OUTPUT_FILE}")
        # Check Ch5 5.2.1 theta scores file
        log("\nChecking Ch5 5.2.1 domain theta scores...")
        
        validation_results = []
        all_valid = True
        
        # Check Ch5 5.2.1 theta scores (contains all three domains)
        ch5_theta_file = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
        
        if ch5_theta_file.exists():
            log(f"Found Ch5 5.2.1 theta scores: {ch5_theta_file}")
            validation_results.append(f"✓ Ch5 5.2.1 theta scores found: {ch5_theta_file}")
            
            # Load and validate structure
            try:
                theta_df = pd.read_csv(ch5_theta_file)
                log(f"Theta file shape: {theta_df.shape}")
                log(f"Columns: {theta_df.columns.tolist()}")
                
                # Check for required domain columns
                required_cols = ['composite_ID', 'theta_what', 'theta_where', 'theta_when']
                missing_cols = [col for col in required_cols if col not in theta_df.columns]
                
                if missing_cols:
                    log(f"Missing columns in theta file: {missing_cols}")
                    validation_results.append(f"⚠ Missing columns: {missing_cols}")
                    all_valid = False
                else:
                    log(f"All required columns present: {required_cols}")
                    validation_results.append(f"✓ All required columns present")
                    
                    # Check data quality
                    n_records = len(theta_df)
                    n_unique_composite = theta_df['composite_ID'].nunique()
                    log(f"Total records: {n_records}")
                    log(f"Unique composite IDs: {n_unique_composite}")
                    
                    # Extract unique participants (UID from composite_ID)
                    theta_df['UID'] = theta_df['composite_ID'].str.split('_').str[0]
                    n_unique_participants = theta_df['UID'].nunique()
                    log(f"Unique participants: {n_unique_participants}")
                    validation_results.append(f"✓ {n_unique_participants} unique participants found")
                    
                    # Check for missing values
                    missing_what = theta_df['theta_what'].isna().sum()
                    missing_where = theta_df['theta_where'].isna().sum()
                    missing_when = theta_df['theta_when'].isna().sum()
                    
                    if missing_what > 0 or missing_where > 0 or missing_when > 0:
                        log(f"Missing theta values: What={missing_what}, Where={missing_where}, When={missing_when}")
                        validation_results.append(f"⚠ Missing values detected")
                    else:
                        log("No missing theta values")
                        validation_results.append("✓ No missing theta values")
                        
            except Exception as e:
                log(f"Failed to read theta file: {e}")
                validation_results.append(f"✗ Failed to read theta file: {e}")
                all_valid = False
        else:
            log(f"Ch5 5.2.1 theta scores not found: {ch5_theta_file}")
            validation_results.append(f"✗ Ch5 5.2.1 theta scores not found")
            all_valid = False
        # Check cognitive test data availability
        log("\nChecking cognitive test data...")
        
        dfnonvr_file = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        if dfnonvr_file.exists():
            log(f"Found dfnonvr.csv: {dfnonvr_file}")
            validation_results.append(f"✓ dfnonvr.csv found")
            
            try:
                # Load and check for cognitive test columns
                df_cog = pd.read_csv(dfnonvr_file)
                log(f"dfnonvr shape: {df_cog.shape}")
                
                # Check for cognitive test columns (T-scores)
                required_cog_tests = ['RAVLT_T', 'BVMT_T', 'RPM_T']
                
                # Also check for raw scores if T-scores not found
                alt_cog_tests = ['RAVLT_Total', 'BVMT_Total', 'Ravens_Score']
                
                found_tests = []
                for test in required_cog_tests:
                    if test in df_cog.columns:
                        found_tests.append(test)
                        log(f"Found {test} column")
                    else:
                        # Check for alternative column names
                        alt_found = False
                        for alt in alt_cog_tests:
                            if alt in df_cog.columns:
                                log(f"Found alternative: {alt} (will need T-score conversion)")
                                found_tests.append(alt)
                                alt_found = True
                                break
                        if not alt_found:
                            log(f"{test} not found in dfnonvr.csv")
                            
                if len(found_tests) >= 3:
                    validation_results.append(f"✓ Cognitive test data available: {found_tests}")
                else:
                    validation_results.append(f"⚠ Some cognitive tests missing")
                    all_valid = False
                    
                # Check UID column
                if 'UID' in df_cog.columns:
                    n_participants = df_cog['UID'].nunique()
                    log(f"Participants in dfnonvr: {n_participants}")
                    validation_results.append(f"✓ {n_participants} participants in dfnonvr.csv")
                else:
                    log("UID column not found in dfnonvr.csv")
                    validation_results.append("✗ UID column missing")
                    all_valid = False
                    
            except Exception as e:
                log(f"Failed to read dfnonvr.csv: {e}")
                validation_results.append(f"✗ Failed to read dfnonvr.csv: {e}")
                all_valid = False
        else:
            log(f"dfnonvr.csv not found: {dfnonvr_file}")
            validation_results.append("✗ dfnonvr.csv not found")
            all_valid = False
        # Write validation results
        log("\nWriting validation results...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("DEPENDENCY VALIDATION REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"RQ: 7.1.3\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Ch5 5.2.1 Dependencies:\n")
            f.write("-"*30 + "\n")
            for result in validation_results:
                f.write(f"{result}\n")
            
            f.write("\n" + "="*50 + "\n")
            if all_valid:
                f.write("STATUS: ALL DEPENDENCIES VALIDATED ✓\n")
                log("All dependencies validated successfully")
            else:
                f.write("STATUS: VALIDATION FAILED ✗\n")
                log("Some dependencies failed validation")
                
        log(f"Validation report written to: {OUTPUT_FILE}")
        
        # Exit with error if validation failed
        if not all_valid:
            log("Exiting due to dependency validation failure")
            sys.exit(1)
            
        log("Step 00 completed successfully")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"{traceback.format_exc()}")
        sys.exit(1)