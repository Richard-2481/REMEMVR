#!/usr/bin/env python3
"""validate_dependencies: Verify all cross-RQ dependencies exist before proceeding with DASS analysis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step00_validate_dependencies.log"


# Dependency File Paths

DEPENDENCY_FILES = [
    {
        'path': PROJECT_ROOT / 'results' / 'ch5' / '5.1.1' / 'data' / 'step03_theta_scores.csv',
        'description': 'Memory theta scores (accuracy outcomes)',
        'required_columns': ['UID', 'theta'],
        'expected_rows': '~400 (100 participants × 4 tests)'
    },
    {
        'path': PROJECT_ROOT / 'results' / 'ch6' / '6.1.1' / 'data' / 'step03_theta_confidence.csv',
        'description': 'Confidence theta scores (metacognitive outcomes)', 
        'required_columns': ['UID', 'theta_confidence'],
        'expected_rows': '~400 (100 participants × 4 tests)'
    },
    {
        'path': PROJECT_ROOT / 'results' / 'ch6' / '6.2.1' / 'data' / 'step02_calibration_scores.csv',
        'description': 'Calibration metrics (metacognitive outcomes)',
        'required_columns': ['UID', 'calibration'],
        'expected_rows': '~100 participants'
    },
    {
        'path': PROJECT_ROOT / 'data' / 'dfnonvr.csv',
        'description': 'DASS scores in prepared data (Ch7 uses dfnonvr.csv, NOT master.xlsx)',
        'required_columns': ['UID', 'total-dass-anxiety-items', 'total-dass-stress-items', 'total-dass-depression-items'],
        'expected_rows': '~100 participants',
        'note': 'CORRECTED - All 3 DASS columns available with lowercase hyphenated names'
    }
]

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
        # Validate Each Dependency File

        log("Checking 4 critical dependency files...")
        validation_results = []
        min_size_bytes = 1000  # Files must be >1KB to be considered valid
        
        for i, dependency in enumerate(DEPENDENCY_FILES, 1):
            file_path = dependency['path']
            description = dependency['description']
            
            log(f"[CHECK {i}/4] {description}")
            log(f"{file_path}")
            
            # Run analysis tool: check_file_exists
            # Tool checks file existence and size requirements
            file_check = check_file_exists(str(file_path), min_size_bytes=min_size_bytes)
            
            # Extract results from validation function return
            exists = file_check.get('valid', False)
            size_bytes = file_check.get('size_bytes', 0)
            message = file_check.get('message', 'No message')
            
            # Determine status for CSV output
            if exists:
                status = "FOUND"
                note = f"Valid file ({size_bytes} bytes)"
                log(f"{description} - {size_bytes} bytes")
            else:
                status = "MISSING" 
                note = f"Failed: {message}"
                log(f"{description} - {message}")
            
            # Store results for CSV output
            validation_results.append({
                'file_path': str(file_path),
                'exists': exists,
                'size_bytes': size_bytes,
                'status': status,
                'note': note
            })
        # Special Check for DASS Columns in dfnonvr.csv

        log("Verifying DASS column names in dfnonvr.csv...")
        
        # Check if dfnonvr.csv passed basic file existence test
        dfnonvr_result = validation_results[3]  # Last entry is dfnonvr.csv
        
        if dfnonvr_result['exists']:
            try:
                # Load just the header to check column names
                dfnonvr = pd.read_csv(PROJECT_ROOT / 'data' / 'dfnonvr.csv', nrows=0)
                actual_columns = dfnonvr.columns.tolist()
                
                # Required DASS columns (exact case-sensitive match)
                required_dass_columns = [
                    'total-dass-anxiety-items',
                    'total-dass-stress-items', 
                    'total-dass-depression-items'
                ]
                
                # Check for exact column name matches
                missing_dass_columns = [col for col in required_dass_columns if col not in actual_columns]
                
                if len(missing_dass_columns) == 0:
                    log("All 3 DASS columns found with correct lowercase hyphenated names")
                    dfnonvr_result['note'] += f" | DASS columns verified: {required_dass_columns}"
                else:
                    log(f"Missing DASS columns: {missing_dass_columns}")
                    log(f"Available columns with 'dass' in name: {[col for col in actual_columns if 'dass' in col.lower()]}")
                    dfnonvr_result['status'] = "COLUMN_ERROR"
                    dfnonvr_result['note'] = f"Missing DASS columns: {missing_dass_columns}"
                    
            except Exception as e:
                log(f"Could not read dfnonvr.csv columns: {str(e)}")
                dfnonvr_result['status'] = "READ_ERROR"
                dfnonvr_result['note'] = f"Cannot read file: {str(e)}"
        else:
            log("dfnonvr.csv does not exist, skipping column check")
        # Save Validation Results
        # These outputs will be used by: Master to verify all dependencies before starting analysis
        
        log("Saving validation results...")
        
        # Output: step00_dependency_validation.csv
        # Contains: Dependency validation results with file paths and status  
        # Columns: ['file_path', 'exists', 'size_bytes', 'status', 'note']
        validation_df = pd.DataFrame(validation_results)
        output_path = RQ_DIR / 'data' / 'step00_dependency_validation.csv'
        validation_df.to_csv(output_path, index=False, encoding='utf-8')
        
        log(f"{output_path} ({len(validation_df)} rows, {len(validation_df.columns)} cols)")
        # Run Validation Tool
        # Validates: Output CSV has expected columns and format
        # Threshold: All required columns present

        log("Running validate_data_format...")
        required_output_columns = ['file_path', 'exists', 'size_bytes', 'status', 'note']
        
        validation_result = validate_data_format(validation_df, required_output_columns)

        # Report validation results
        if isinstance(validation_result, dict):
            if validation_result.get('valid', False):
                log("Output CSV format valid - all required columns present")
                for key, value in validation_result.items():
                    log(f"{key}: {value}")
            else:
                log(f"Output CSV format invalid: {validation_result.get('message', 'Unknown error')}")
        else:
            log(f"{validation_result}")

        # Summary of validation results
        total_files = len(validation_results)
        valid_files = sum(1 for result in validation_results if result['exists'])
        
        log(f"Dependency validation complete: {valid_files}/{total_files} files valid")
        
        if valid_files == total_files:
            log("All 4 critical dependencies found - RQ 7.3.4 ready to proceed")
        else:
            log(f"{total_files - valid_files} dependencies missing - analysis may fail")

        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)