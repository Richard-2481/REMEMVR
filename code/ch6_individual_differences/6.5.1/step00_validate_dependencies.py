#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.1.1 theta scores and dfnonvr.csv accessibility before proceeding with"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import os

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.1 (derived from script location)
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
        # Validate Ch5 5.1.1 Dependencies

        log("Starting dependency validation...")
        
        # Validate Ch5 dependency
        ch5_status_path = PROJECT_ROOT / 'results/ch5/5.1.1/status.yaml'
        ch5_theta_path = PROJECT_ROOT / 'results/ch5/5.1.1/data/step03_theta_scores.csv'
        dfnonvr_path = PROJECT_ROOT / 'data/dfnonvr.csv'
        
        validation_results = []
        
        # Check Ch5 status
        if ch5_status_path.exists():
            with open(ch5_status_path, encoding='utf-8') as f:
                content = f.read()
                if 'results analysis:' in content and 'success' in content:
                    validation_results.append("Ch5 5.1.1 dependency VALIDATED")
                    log("Ch5 5.1.1 status shows success")
                else:
                    validation_results.append("Ch5 5.1.1 dependency FAILED - not complete")
                    log("Ch5 5.1.1 status not successful")
        else:
            validation_results.append("Ch5 5.1.1 dependency FAILED - status not found")
            log("Ch5 5.1.1 status.yaml not found")
        
        # Check Ch5 theta file
        if ch5_theta_path.exists():
            theta_df = pd.read_csv(ch5_theta_path)
            if len(theta_df) >= 400 and 'Theta_All' in theta_df.columns:
                validation_results.append("Theta scores VALIDATED - 400 rows with Theta_All")
                log(f"Theta file exists with {len(theta_df)} rows and Theta_All column")
            else:
                validation_results.append("Theta scores FAILED - insufficient data")
                log(f"Theta file has {len(theta_df)} rows, missing required data")
        else:
            validation_results.append("Theta scores FAILED - file not found")
            log("Ch5 theta scores file not found")
        
        # Check dfnonvr.csv
        if dfnonvr_path.exists():
            participant_df = pd.read_csv(dfnonvr_path)
            required_cols = ['UID', 'education', 'vr-exposure', 'typical-sleep-hours', 'age']
            missing_cols = [col for col in required_cols if col not in participant_df.columns]
            if not missing_cols and len(participant_df) >= 100:
                validation_results.append("dfnonvr.csv dependency VALIDATED")
                validation_results.append(f"Sample size N={len(participant_df)} CONFIRMED")
                log(f"dfnonvr.csv exists with {len(participant_df)} participants")
                log(f"All required columns present: {required_cols}")
            else:
                validation_results.append(f"dfnonvr.csv FAILED - missing columns: {missing_cols}")
                log(f"dfnonvr.csv missing columns: {missing_cols}")
        else:
            validation_results.append("dfnonvr.csv dependency FAILED - file not found")
            log("dfnonvr.csv not found")
        # Write Validation Report
        # Output: Hierarchical path validation report

        log("Writing validation report...")
        
        # Write validation report
        output_path = RQ_DIR / 'data' / 'step00_dependency_validation.txt'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RQ 7.5.1 Dependency Validation Report\n")
            f.write("=====================================\n\n")
            for result in validation_results:
                f.write(f"{result}\n")
            f.write(f"\nValidation completed at: {pd.Timestamp.now()}\n")
            
            if any("FAILED" in result for result in validation_results):
                f.write("\nERROR: Dependencies not met - cannot proceed\n")
                log("Critical dependencies missing")
                raise ValueError("Critical dependencies missing")
            else:
                f.write("\nSUCCESS: All dependencies validated\n")
                log("All dependencies validated")
        
        log(f"Validation report: {output_path}")
        # Run Validation Tool
        # Validates: dfnonvr.csv has required columns for self-report analysis

        log("Running validate_data_columns...")
        
        # Validate dfnonvr.csv structure
        required_columns = ['UID', 'education', 'vr-exposure', 'typical-sleep-hours', 'age']
        # Load the data first since function expects DataFrame, not path
        df_for_validation = pd.read_csv(dfnonvr_path)
        validation_result = validate_data_columns(
            df=df_for_validation,
            required_columns=required_columns
        )

        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)