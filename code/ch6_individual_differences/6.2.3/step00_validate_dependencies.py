#!/usr/bin/env python3
"""
Step 0: Validate Cross-RQ Dependencies for RQ 7.2.3
Purpose: Verify Ch5 5.1.1 theta scores and cognitive test data exist before proceeding

Scientific Context:
- RQ 7.2.3 tests Age x Cognitive Test interactions on REMEMVR performance
- Requires mean theta_all scores from Ch5 5.1.1 (omnibus memory performance)
- Requires cognitive test scores (RAVLT, BVMT, NART, RPM) from dfnonvr.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Define paths
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.3
CH5_DIR = PROJECT_ROOT / "results" / "ch5" / "5.1.1"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = RQ_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Output file
VALIDATION_FILE = OUTPUT_DIR / "step00_dependency_validation.txt"

def validate_dependencies():
    """Validate all required dependencies exist and are accessible."""
    
    validation_results = []
    all_valid = True
    
    # Check Ch5 5.1.1 theta scores
    validation_results.append("=" * 60)
    validation_results.append("DEPENDENCY VALIDATION FOR RQ 7.2.3")
    validation_results.append("=" * 60)
    validation_results.append("")
    
    # Primary theta file path
    theta_file = CH5_DIR / "data" / "step03_theta_scores.csv"
    
    if theta_file.exists():
        try:
            theta_df = pd.read_csv(theta_file)
            n_rows = len(theta_df)
            
            # Check expected columns
            required_cols = ['UID', 'Theta_All']
            missing_cols = [col for col in required_cols if col not in theta_df.columns]
            
            if missing_cols:
                validation_results.append(f"WARNING: Ch5 theta file missing columns: {missing_cols}")
                all_valid = False
            else:
                validation_results.append(f"✓ Ch5 5.1.1 theta scores found: {theta_file}")
                validation_results.append(f"  - Rows: {n_rows} (expecting 400 for 100 participants x 4 tests)")
                validation_results.append(f"  - Columns: {list(theta_df.columns)}")
                
                # Check data quality
                n_participants = theta_df['UID'].nunique()
                validation_results.append(f"  - Unique participants: {n_participants}")
                
                if n_participants != 100:
                    validation_results.append(f"WARNING: Expected 100 participants, found {n_participants}")
                
        except Exception as e:
            validation_results.append(f"ERROR reading Ch5 theta file: {e}")
            all_valid = False
    else:
        validation_results.append("ERROR: Ch5 5.1.1 theta scores not found!")
        validation_results.append(f"  Expected at: {theta_file}")
        all_valid = False
    
    validation_results.append("")
    
    # Check cognitive test data
    dfnonvr_file = DATA_DIR / "dfnonvr.csv"
    
    if dfnonvr_file.exists():
        try:
            # Read first row to check columns
            df_sample = pd.read_csv(dfnonvr_file, nrows=5)
            
            validation_results.append(f"✓ Cognitive test data found: {dfnonvr_file}")
            validation_results.append(f"  - Shape: {pd.read_csv(dfnonvr_file).shape}")
            
            # Check for required columns
            required_vars = {
                'Age': 'age',
                'NART': 'nart-score',
                'RPM': 'rpm-score', 
                'BVMT': 'bvmt-total-recall',
                'RAVLT': ['ravlt-trial-1-score', 'ravlt-trial-2-score', 
                          'ravlt-trial-3-score', 'ravlt-trial-4-score', 
                          'ravlt-trial-5-score']
            }
            
            for var_name, col_names in required_vars.items():
                if isinstance(col_names, list):
                    # Check all RAVLT trial columns
                    missing = [col for col in col_names if col not in df_sample.columns]
                    if missing:
                        validation_results.append(f"  WARNING: Missing {var_name} columns: {missing}")
                        all_valid = False
                    else:
                        validation_results.append(f"  ✓ {var_name} columns found (will sum trials 1-5)")
                else:
                    if col_names in df_sample.columns:
                        validation_results.append(f"  ✓ {var_name} found: column '{col_names}'")
                    else:
                        validation_results.append(f"  ERROR: {var_name} not found (looking for '{col_names}')")
                        all_valid = False
                        
        except Exception as e:
            validation_results.append(f"ERROR reading cognitive test file: {e}")
            all_valid = False
    else:
        validation_results.append("ERROR: Cognitive test data (dfnonvr.csv) not found!")
        all_valid = False
    
    validation_results.append("")
    
    # Final status
    if all_valid:
        validation_results.append("VALIDATION STATUS: PASS ✓")
        validation_results.append("Ch5 5.1.1 validation PASS")
        validation_results.append("Files accessible")
    else:
        validation_results.append("VALIDATION STATUS: FAIL ✗")
        validation_results.append("Please resolve issues before proceeding")
    
    # Write results
    with open(VALIDATION_FILE, 'w') as f:
        f.write('\n'.join(validation_results))
    
    # Also print to console
    print('\n'.join(validation_results))
    
    return all_valid

if __name__ == "__main__":
    valid = validate_dependencies()
    if not valid:
        print("\nDependency validation failed. Exiting.")
        sys.exit(1)
    else:
        print("\nAll dependencies validated successfully. Ready to proceed with analysis.")