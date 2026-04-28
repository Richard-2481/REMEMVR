#!/usr/bin/env python3
"""validate_dependencies: Validate cross-RQ dependencies and data sources exist before proceeding with analysis"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Import validation tool  
from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.4.3 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()  # Critical for real-time monitoring
    print(msg, flush=True)  # -u flag compatibility

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: validate_dependencies")
        # Define Dependency Files to Check

        log("Defining dependency files to check...")
        
        files_to_check = [
            {
                'path': PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv",
                'min_size_bytes': 1000,
                'description': "What-domain theta scores from Ch5 5.2.1"
            },
            {
                'path': PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv",
                'min_size_bytes': 1000,
                'description': "Overall theta scores from Ch5 5.1.1"
            },
            {
                'path': PROJECT_ROOT / "data" / "dfnonvr.csv",
                'min_size_bytes': 50000,
                'description': "Non-VR data including RPM scores"
            }
        ]
        
        log(f"Will check {len(files_to_check)} dependency files")
        # Validate Each File Exists and Meets Requirements

        log("Running dependency validation checks...")
        
        validation_results = []
        all_valid = True
        
        for file_info in files_to_check:
            file_path = file_info['path']
            min_size = file_info['min_size_bytes']
            description = file_info['description']
            
            log(f"Validating {description}...")
            log(f"Path: {file_path}")
            log(f"Min size: {min_size} bytes")
            
            # Use actual signature: check_file_exists(file_path, min_size_bytes=0)
            result = check_file_exists(file_path=str(file_path), min_size_bytes=min_size)
            
            # Extract validation status
            is_valid = result.get('valid', False)
            actual_size = result.get('size_bytes', 0)
            message = result.get('message', 'Unknown')
            
            if is_valid:
                status = "PASS"
                log(f"{description} - {actual_size} bytes")
            else:
                status = "FAIL"
                all_valid = False
                log(f"{description} - {message}")
            
            validation_results.append({
                'file': str(file_path.relative_to(PROJECT_ROOT)),
                'description': description,
                'status': status,
                'size_bytes': actual_size,
                'min_size_required': min_size,
                'message': message
            })
        # Save Validation Report
        # These outputs will be used by: Step 1 and subsequent analysis steps
        
        log(f"Saving dependency validation report...")
        
        output_file = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RQ 7.4.3 Dependency Validation Report\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write(f"# Overall Status: {'PASS' if all_valid else 'FAIL'}\n")
            f.write("\n")
            
            for result in validation_results:
                f.write(f"File: {result['file']}\n")
                f.write(f"Description: {result['description']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Size: {result['size_bytes']} bytes (min: {result['min_size_required']})\n")
                f.write(f"Message: {result['message']}\n")
                f.write("\n")
                
            f.write(f"Summary: {sum(1 for r in validation_results if r['status'] == 'PASS')} of {len(validation_results)} dependencies valid\n")
            
        log(f"{output_file} ({len(validation_results)} validation results)")
        # Run Validation Tool (Basic Format Check)
        # Validates: Basic report structure and content format
        # Note: This is a simple format check since we generated a text file

        log("Running basic format validation...")
        
        # Create a minimal DataFrame representation for validation
        # Since validate_data_format expects a DataFrame, we'll create one from our results
        results_df = pd.DataFrame(validation_results)
        
        # Use actual signature: validate_data_format(df, required_cols)
        format_validation = validate_data_format(
            df=results_df,
            required_cols=['file', 'description', 'status', 'size_bytes', 'message']
        )
        
        # Report validation results
        if isinstance(format_validation, dict):
            valid = format_validation.get('valid', False)
            message = format_validation.get('message', 'Unknown validation result')
            
            if valid:
                log("Format validation PASSED")
            else:
                log(f"Format validation FAILED: {message}")
        else:
            log(f"Format validation result: {format_validation}")
        # FINAL STATUS
        
        if all_valid:
            log("Step 00 complete - All dependencies valid")
            log("Analysis can proceed to Step 01")
        else:
            failed_files = [r['file'] for r in validation_results if r['status'] == 'FAIL']
            log(f"Step 00 FAILED - Invalid dependencies: {', '.join(failed_files)}")
            log("Analysis cannot proceed until dependencies resolved")
            sys.exit(1)
        
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)