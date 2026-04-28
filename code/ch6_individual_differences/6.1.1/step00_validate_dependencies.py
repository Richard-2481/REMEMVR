#!/usr/bin/env python3
"""step00_validate_dependencies: Verify required Ch5 outputs exist and cognitive test data accessible before proceeding."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import traceback
import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Import validation tool (same as analysis tool)
# from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1 (derived from script location)
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
        # Define Dependency Files to Check

        log("Defining dependency files to check...")
        
        # Files to check from 4_analysis.yaml parameters
        file_paths_to_check = [
            "results/ch5/5.1.1/data/step03_theta_scores.csv",
            "data/cache/dfData.csv", 
            "data/dfnonvr.csv"
        ]
        min_size_bytes = 1000  # Files must be >1KB to be considered valid
        
        log(f"Will check {len(file_paths_to_check)} dependency files")
        log(f"Minimum size threshold: {min_size_bytes} bytes")
        # Run Analysis Tool - Check Each File

        validation_results = {}
        all_files_valid = True
        
        for file_path in file_paths_to_check:
            log(f"Checking file: {file_path}")
            
            # Convert to absolute path for checking
            abs_file_path = PROJECT_ROOT / file_path
            
            result = check_file_exists(
                file_path=abs_file_path,
                min_size_bytes=min_size_bytes
            )
            
            validation_results[file_path] = result
            
            if result.get('valid', False):
                log(f"{file_path} - {result.get('message', 'file check')}")
            else:
                log(f"{file_path} - {result.get('message', 'unknown error')}")
                all_files_valid = False
        # Check Ch5 5.1.1 Status (Additional Validation)
        # Check if Ch5 5.1.1 shows completion status

        log("Checking Ch5 5.1.1 completion status...")
        ch5_status_path = PROJECT_ROOT / "results/ch5/5.1.1/status.yaml"
        
        ch5_status_ok = False
        if ch5_status_path.exists():
            try:
                with open(ch5_status_path, 'r', encoding='utf-8') as f:
                    status_data = yaml.safe_load(f)
                
                results analysis_status = status_data.get('results analysis', {})
                if isinstance(results analysis_status, dict) and results analysis_status.get('status') == 'success':
                    actual_status = 'success'
                elif results analysis_status == 'success':
                    actual_status = 'success'
                else:
                    actual_status = results analysis_status
                
                if actual_status == 'success':
                    log("Ch5 5.1.1 status shows results analysis = success")
                    ch5_status_ok = True
                else:
                    log(f"Ch5 5.1.1 status shows results analysis = {results analysis_status} (expected: success)")
                    all_files_valid = False
                    ch5_status_ok = False
            except Exception as e:
                log(f"Could not read Ch5 5.1.1 status.yaml: {e}")
                all_files_valid = False
        else:
            log("Ch5 5.1.1 status.yaml not found")
            all_files_valid = False
        # Save Analysis Outputs
        # Output: Dependency validation results summary
        # Contains: File check results, status check, overall validation outcome

        log("Saving dependency validation results...")
        
        output_path = RQ_DIR / "data/step00_dependency_validation.txt"
        
        # Create validation summary
        validation_summary = []
        validation_summary.append("DEPENDENCY VALIDATION RESULTS")
        validation_summary.append("=" * 40)
        validation_summary.append(f"Validation timestamp: {pd.Timestamp.now()}")
        validation_summary.append(f"RQ: {RQ_DIR}")
        validation_summary.append("")
        
        validation_summary.append("FILE EXISTENCE CHECKS:")
        for file_path, result in validation_results.items():
            status = "PASS" if result.get('valid', False) else "FAIL"
            size_info = f"({result.get('size_bytes', 0)} bytes)" if result.get('valid', False) else "(not found)"
            validation_summary.append(f"  {status}: {file_path} {size_info}")
        
        validation_summary.append("")
        validation_summary.append("STATUS CHECKS:")
        status = "PASS" if ch5_status_ok else "FAIL"
        validation_summary.append(f"  {status}: Ch5 5.1.1 completion status")
        
        validation_summary.append("")
        overall_status = "PASS" if (all_files_valid and ch5_status_ok) else "FAIL"
        validation_summary.append(f"OVERALL VALIDATION: {overall_status}")
        
        if all_files_valid and ch5_status_ok:
            validation_summary.append("All cross-RQ dependencies satisfied. Ch7 analysis can proceed.")
        else:
            validation_summary.append("CRITICAL: Cross-RQ dependencies missing. Ch7 analysis cannot proceed.")
            validation_summary.append("Required actions:")
            validation_summary.append("  1. Complete Ch5 5.1.1 analysis (results analysis = success)")
            validation_summary.append("  2. Ensure all required data files exist and are >1KB")
            validation_summary.append("  3. Re-run this dependency validation step")
        
        # Write summary to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(validation_summary))
        
        log(f"{output_path} ({len(validation_summary)} lines)")
        # Run Validation Tool
        # Validates: The dependency validation output file was created successfully
        # Threshold: Output file should exist and be >1KB

        log("Verifying dependency validation output...")
        
        final_validation = check_file_exists(
            file_path=output_path,
            min_size_bytes=100  # Output file should be at least 100 bytes
        )
        
        if final_validation.get('valid', False):
            log(f"Output file created successfully ({final_validation.get('size_bytes', 0)} bytes)")
        else:
            log(f"Output file validation failed: {final_validation.get('message', 'unknown error')}")

        # Final status check
        if all_files_valid and ch5_status_ok:
            log("Step 00 complete - All dependencies validated")
            sys.exit(0)
        else:
            log("Step 00 complete - Dependency validation failed")
            raise FileNotFoundError("Cross-RQ dependencies missing - cannot proceed without Ch5 5.1.1 theta scores")

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)