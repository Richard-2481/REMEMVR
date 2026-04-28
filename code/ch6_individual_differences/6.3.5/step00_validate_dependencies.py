#!/usr/bin/env python3
"""validate_dependencies: Validate cross-RQ dependencies exist before proceeding with confidence-accuracy"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.5 (derived from script location)
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
        log("Step 00: Validate Dependencies")
        # Define Dependencies to Check
        
        log("Defining dependency files...")
        dependencies = [
            {
                'path': PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv',
                'description': 'Ch5 5.1.1 theta scores (memory accuracy)',
                'required_columns': ['UID', 'theta_all'],
                'source': 'Ch5 5.1.1 IRT calibration'
            },
            {
                'path': PROJECT_ROOT / 'results' / 'ch6' / '6.1.1' / 'data' / 'step03_theta_confidence.csv',
                'description': 'Ch6 confidence scores on IRT scale',
                'required_columns': ['UID', 'theta_confidence'],
                'source': 'Ch6 6.1.1 confidence calibration'
            },
            {
                'path': PROJECT_ROOT / 'data' / 'dfnonvr.csv',
                'description': 'CRITICAL v5.3.0: Use dfnonvr.csv, NEVER master.xlsx',
                'required_columns': ['UID', 'rpm-score', 'age', 'education'],
                'source': 'Base cognitive reserve indicators'
            }
        ]
        
        log(f"{len(dependencies)} dependencies to validate")
        # Run File Existence Validation
        
        validation_results = []
        all_valid = True
        
        for i, dep in enumerate(dependencies, 1):
            log(f"[VALIDATION {i}/3] Checking {dep['description']}...")
            log(f"{dep['path']}")
            
            # Run validation tool
            result = check_file_exists(
                file_path=dep['path'],
                min_size_bytes=200  # Ensures files are not empty
            )
            
            # Store results
            dep_result = {
                'dependency': dep['description'],
                'path': str(dep['path']),
                'source': dep['source'],
                'valid': result.get('valid', False),
                'size_bytes': result.get('size_bytes', 0),
                'message': result.get('message', ''),
                'required_columns': dep['required_columns']
            }
            validation_results.append(dep_result)
            
            # Report individual result
            if dep_result['valid']:
                log(f"{dep['description']} - {dep_result['size_bytes']} bytes")
            else:
                log(f"{dep['description']} - {dep_result['message']}")
                all_valid = False
        # Column Validation (if files exist)
        # Validates: CSV files have expected columns with exact names
        
        log("Checking CSV column structure...")
        column_validation_results = []
        
        for i, dep in enumerate(dependencies, 1):
            dep_result = validation_results[i-1]
            
            if dep_result['valid'] and str(dep['path']).endswith('.csv'):
                try:
                    log(f"[COLUMN CHECK {i}/3] Reading {dep['description']} columns...")
                    
                    # Read header only (fast)
                    df_cols = pd.read_csv(dep['path'], nrows=0).columns.tolist()
                    expected_cols = dep['required_columns']
                    
                    # Check exact match (order matters)
                    missing_cols = [col for col in expected_cols if col not in df_cols]
                    extra_cols = [col for col in df_cols if col not in expected_cols]
                    
                    col_result = {
                        'dependency': dep['description'],
                        'expected_columns': expected_cols,
                        'actual_columns': df_cols,
                        'missing_columns': missing_cols,
                        'extra_columns': extra_cols,
                        'columns_valid': len(missing_cols) == 0
                    }
                    column_validation_results.append(col_result)
                    
                    if col_result['columns_valid']:
                        log(f"{dep['description']} - All required columns present")
                        log(f"{df_cols}")
                    else:
                        log(f"{dep['description']} - Missing columns: {missing_cols}")
                        if extra_cols:
                            log(f"Extra columns found: {extra_cols}")
                        all_valid = False
                        
                except Exception as e:
                    log(f"Column validation failed for {dep['description']}: {e}")
                    col_result = {
                        'dependency': dep['description'],
                        'columns_valid': False,
                        'error': str(e)
                    }
                    column_validation_results.append(col_result)
                    all_valid = False
        # Save Validation Results
        # These outputs will be used by: Master to determine if analysis can proceed
        
        log("Writing validation summary...")
        output_file = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RQ 7.3.5 Dependency Validation Results\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write(f"# Overall Status: {'PASS' if all_valid else 'FAIL'}\n")
            f.write("\n")
            
            f.write("## File Existence Validation\n")
            for result in validation_results:
                status = "PASS" if result['valid'] else "FAIL"
                f.write(f"[{status}] {result['dependency']}\n")
                f.write(f"  Path: {result['path']}\n")
                f.write(f"  Source: {result['source']}\n")
                f.write(f"  Size: {result['size_bytes']} bytes\n")
                if not result['valid']:
                    f.write(f"  Error: {result['message']}\n")
                f.write("\n")
            
            f.write("## Column Structure Validation\n")
            for result in column_validation_results:
                status = "PASS" if result['columns_valid'] else "FAIL"
                f.write(f"[{status}] {result['dependency']}\n")
                if 'expected_columns' in result:
                    f.write(f"  Expected: {result['expected_columns']}\n")
                    f.write(f"  Actual: {result['actual_columns']}\n")
                    if result['missing_columns']:
                        f.write(f"  Missing: {result['missing_columns']}\n")
                    if result['extra_columns']:
                        f.write(f"  Extra: {result['extra_columns']}\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")
            
            f.write("## Summary\n")
            if all_valid:
                f.write("All dependencies validated successfully. Analysis can proceed.\n")
            else:
                f.write("CRITICAL FAILURE: One or more dependencies missing or invalid.\n")
                f.write("Analysis CANNOT proceed. Fix dependencies first.\n")
        
        log(f"{output_file} (validation summary)")
        # Final Status Check
        # Validates: All critical dependencies are available
        # Expected behavior on failure: QUIT immediately with specific error
        
        if all_valid:
            log("All dependencies validated successfully")
            log("RQ 7.3.5 analysis can proceed")
            sys.exit(0)
        else:
            log("[CRITICAL FAILURE] One or more dependencies missing or invalid")
            log("Cannot proceed with analysis")
            log("Fix missing dependencies before running step01")
            
            # List specific failures for master
            failed_deps = []
            for result in validation_results:
                if not result['valid']:
                    failed_deps.append(f"{result['dependency']}: {result['message']}")
            for result in column_validation_results:
                if not result['columns_valid']:
                    if 'missing_columns' in result and result['missing_columns']:
                        failed_deps.append(f"{result['dependency']}: Missing columns {result['missing_columns']}")
            
            log(f"[FAILED DEPENDENCIES] {'; '.join(failed_deps)}")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)