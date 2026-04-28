#!/usr/bin/env python3
"""validate_dependencies: Verify Ch5 5.1.1 outputs and data availability before proceeding"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.2 (derived from script location)
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
        # Prepare Required Files List

        log("Preparing required files for validation...")
        
        required_files = [
            PROJECT_ROOT / "results/ch5/5.1.1/status.yaml",
            PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv", 
            PROJECT_ROOT / "data/dfnonvr.csv"
        ]
        
        validation_results = []
        # Run Analysis Tool - File Existence Validation

        log("Running check_file_exists for each required file...")
        
        for file_path in required_files:
            log(f"Validating {file_path.name}...")
            
            # Use analysis tool to check file existence
            file_result = check_file_exists(
                file_path=file_path,
                min_size_bytes=100  # Ensure files aren't empty stubs
            )
            
            # Store result with additional context
            result_entry = {
                'file': str(file_path.relative_to(PROJECT_ROOT)),
                'valid': file_result.get('valid', False),
                'size_bytes': file_result.get('size_bytes', 0),
                'message': file_result.get('message', 'No message'),
                'full_path': str(file_path)
            }
            
            validation_results.append(result_entry)
            
            # Log individual result
            status = "" if result_entry['valid'] else ""
            log(f"{status} {file_path.name}: {result_entry['message']}")

        log("File existence validation complete")
        # Advanced Validation - Check Ch5 5.1.1 Status
        # Check Ch5 5.1.1 completion status from status.yaml
        # LESSON LEARNED: Handle both dict and string formats for results analysis (from gcode_lessons.md item 5)
        
        log("Validating Ch5 5.1.1 completion status...")
        
        status_yaml_path = PROJECT_ROOT / "results/ch5/5.1.1/status.yaml"
        ch5_completion_valid = False
        ch5_status_message = "Unknown"
        
        if status_yaml_path.exists():
            try:
                with open(status_yaml_path, 'r', encoding='utf-8') as f:
                    status_data = yaml.safe_load(f)
                
                # Handle both dict and string formats for results analysis (LESSON LEARNED from gcode_lessons.md)
                results analysis_status = status_data.get('results analysis', {})
                if isinstance(results analysis_status, dict):
                    actual_status = results analysis_status.get('status', 'unknown')
                elif isinstance(results analysis_status, str):
                    actual_status = results analysis_status
                else:
                    actual_status = 'unknown'
                
                # Check if Ch5 5.1.1 is complete (success status)
                ch5_completion_valid = actual_status == 'success'
                ch5_status_message = f"Ch5 5.1.1 status: {actual_status}"
                
                log(f"{ch5_status_message}")
                
            except Exception as e:
                ch5_status_message = f"Error reading status.yaml: {str(e)}"
                log(f"{ch5_status_message}")
        else:
            ch5_status_message = "status.yaml not found"
            log(f"{ch5_status_message}")
        # Advanced Validation - Check CSV Column Names
        # LESSON LEARNED: Check actual column names (case sensitivity from gcode_lessons.md item 2)
        
        log("Validating CSV file columns...")
        
        csv_validation_results = []
        
        # Validate theta scores CSV
        theta_csv_path = PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv"
        if theta_csv_path.exists():
            try:
                theta_df = pd.read_csv(theta_csv_path, nrows=0)  # Header only
                theta_cols = theta_df.columns.tolist()
                # ACTUAL column names in Ch5 5.1.1: UID, test, Theta_All
                expected_theta_cols = ["UID", "test", "Theta_All"]

                theta_cols_valid = all(col in theta_cols for col in expected_theta_cols)
                theta_message = f"Theta CSV columns: {theta_cols} (expected: {expected_theta_cols})"
                
                csv_validation_results.append({
                    'file': 'step03_theta_scores.csv',
                    'valid': theta_cols_valid,
                    'message': theta_message
                })
                
                status = "" if theta_cols_valid else ""
                log(f"{status} {theta_message}")
                
            except Exception as e:
                csv_validation_results.append({
                    'file': 'step03_theta_scores.csv',
                    'valid': False,
                    'message': f"Error reading CSV: {str(e)}"
                })
                log(f"Error reading theta CSV: {str(e)}")
        
        # Validate dfnonvr.csv
        dfnonvr_path = PROJECT_ROOT / "data/dfnonvr.csv"
        if dfnonvr_path.exists():
            try:
                dfnonvr_df = pd.read_csv(dfnonvr_path, nrows=0)  # Header only
                dfnonvr_cols = dfnonvr_df.columns.tolist()
                expected_dfnonvr_cols = ["UID", "total-dass-depression-items", "total-dass-anxiety-items", "total-dass-stress-items", "age", "nart-score"]
                
                dfnonvr_cols_valid = all(col in dfnonvr_cols for col in expected_dfnonvr_cols)
                dfnonvr_message = f"dfnonvr.csv has required DASS columns: {dfnonvr_cols_valid}"
                
                csv_validation_results.append({
                    'file': 'dfnonvr.csv',
                    'valid': dfnonvr_cols_valid,
                    'message': dfnonvr_message
                })
                
                status = "" if dfnonvr_cols_valid else ""
                log(f"{status} {dfnonvr_message}")
                
                # Log first few available columns for debugging
                log(f"dfnonvr.csv first 10 columns: {dfnonvr_cols[:10]}")
                
            except Exception as e:
                csv_validation_results.append({
                    'file': 'dfnonvr.csv',
                    'valid': False,
                    'message': f"Error reading CSV: {str(e)}"
                })
                log(f"Error reading dfnonvr.csv: {str(e)}")
        # Save Analysis Outputs
        # Output: dependency validation log with comprehensive PASS/FAIL status
        # Contains: Detailed validation results for downstream debugging

        log("Saving dependency validation results...")
        
        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== RQ 7.5.2 DEPENDENCY VALIDATION LOG ===\n")
            f.write(f"Validation tool: tools.validation.check_file_exists\n\n")
            
            # Overall summary
            all_files_valid = all(result['valid'] for result in validation_results)
            all_columns_valid = all(result['valid'] for result in csv_validation_results)
            overall_valid = all_files_valid and ch5_completion_valid and all_columns_valid
            
            f.write("=== OVERALL STATUS ===\n")
            f.write(f"Ch5 5.1.1 status: {'SUCCESS' if ch5_completion_valid else 'FAIL'}\n")
            f.write(f"File existence: {'SUCCESS' if all_files_valid else 'FAIL'}\n")
            f.write(f"Column validation: {'SUCCESS' if all_columns_valid else 'FAIL'}\n")
            f.write(f"Ready for analysis: {'YES' if overall_valid else 'NO'}\n\n")
            
            # Detailed file results
            f.write("=== FILE EXISTENCE VALIDATION ===\n")
            for result in validation_results:
                status = "PASS" if result['valid'] else "FAIL"
                f.write(f"[{status}] {result['file']}\n")
                f.write(f"    Size: {result['size_bytes']} bytes\n")
                f.write(f"    Message: {result['message']}\n")
                f.write(f"    Path: {result['full_path']}\n\n")
            
            # Ch5 status details
            f.write("=== CH5 5.1.1 STATUS VALIDATION ===\n")
            f.write(f"Status: {ch5_status_message}\n")
            f.write(f"Valid: {'YES' if ch5_completion_valid else 'NO'}\n\n")
            
            # CSV column details
            f.write("=== CSV COLUMN VALIDATION ===\n")
            for result in csv_validation_results:
                status = "PASS" if result['valid'] else "FAIL"
                f.write(f"[{status}] {result['file']}\n")
                f.write(f"    Message: {result['message']}\n\n")
            
            # Required patterns summary (for validation_call)
            f.write("=== VALIDATION PATTERNS ===\n")
            patterns_found = []
            if ch5_completion_valid:
                patterns_found.append("Ch5 5.1.1 status: success")
            if any(result['file'] == 'data/dfnonvr.csv' and result['valid'] for result in validation_results):
                patterns_found.append("dfnonvr.csv accessible")
            if any(result['file'] == 'results/ch5/5.1.1/data/step03_theta_scores.csv' and result['valid'] for result in validation_results):
                patterns_found.append("theta file found")
                
            f.write(f"Required patterns found: {patterns_found}\n")
            f.write(f"Forbidden patterns found: {'none' if overall_valid else 'validation failures present'}\n")

        log(f"step00_dependency_validation.txt ({output_path.stat().st_size} bytes)")
        # Run Validation Tool
        # Validates: Presence of required patterns, absence of forbidden patterns
        # Threshold: All required patterns must be present

        log("Running validation on output file...")
        
        # Read validation output for pattern checking
        with open(output_path, 'r', encoding='utf-8') as f:
            validation_content = f.read()
        
        # Check for required patterns
        required_patterns = ["Ch5 5.1.1 status: success", "dfnonvr.csv accessible", "theta file found"]
        forbidden_patterns = ["ERROR", "not found", "missing"]
        
        patterns_present = [pattern for pattern in required_patterns if pattern in validation_content]
        patterns_missing = [pattern for pattern in required_patterns if pattern not in validation_content]
        forbidden_present = [pattern for pattern in forbidden_patterns if pattern in validation_content]
        
        validation_passed = len(patterns_missing) == 0 and len(forbidden_present) == 0
        
        # Report validation results
        if validation_passed:
            log("All required patterns found, no forbidden patterns detected")
            log(f"Required patterns present: {patterns_present}")
        else:
            log("Validation issues detected:")
            if patterns_missing:
                log(f"Missing required patterns: {patterns_missing}")
            if forbidden_present:
                log(f"Forbidden patterns found: {forbidden_present}")
        # Final Decision
        
        if overall_valid:
            log("Step 00 complete - All dependencies validated successfully")
            log(f"Ch5 5.1.1 outputs available, dfnonvr.csv accessible")
            log(f"RQ 7.5.2 analysis pipeline can proceed")
            sys.exit(0)
        else:
            log("Step 00 failed - Dependencies not met")
            log(f"Ch5 5.1.1 complete: {ch5_completion_valid}")
            log(f"Files exist: {all_files_valid}")
            log(f"Columns valid: {all_columns_valid}")
            log("Fix dependencies before proceeding with RQ 7.5.2")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)