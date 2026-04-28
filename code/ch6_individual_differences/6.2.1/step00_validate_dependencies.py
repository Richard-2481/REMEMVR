#!/usr/bin/env python3
"""validate_dependencies: Validate cross-RQ dependencies for age moderation analysis. Checks that Ch5 5.1.1"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import check_file_exists

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.1 (derived from script location)
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
        # Define Dependency Files to Check

        log("Defining dependency files to validate...")
        
        # Files to validate (using project root for absolute paths)
        dependency_files = [
            {
                'path': PROJECT_ROOT / "results/ch5/5.1.1/data/step03_theta_scores.csv",
                'description': "Ch5 5.1.1 theta_all scores (IRT ability estimates)",
                'expected_columns': ["composite_ID", "UID", "theta_all", "SE"],
                'source': "Ch5 5.1.1 IRT calibration"
            },
            {
                'path': PROJECT_ROOT / "data/dfnonvr.csv", 
                'description': "Preprocessed participant data with cognitive tests",
                'expected_columns': ["UID", "age", "rpm-score", "bvmt-trial-1-score", "ravlt-trial-1-score", "ravlt-delayed-recall-score", "bvmt-percent-retained"],
                'source': "Ch7 mandate (dfnonvr.csv only - NO master.xlsx)"
            }
        ]
        
        log(f"Will validate {len(dependency_files)} dependency files")
        # Run Analysis Tool (check_file_exists for each dependency)

        log("Running check_file_exists for each dependency...")
        
        validation_results = []
        min_size_bytes = 1000  # Files must be non-trivial size
        
        for file_info in dependency_files:
            file_path = file_info['path']
            log(f"Checking {file_path}...")
            
            # Run check_file_exists tool
            result = check_file_exists(file_path=file_path, min_size_bytes=min_size_bytes)
            
            # Add metadata to result
            result['description'] = file_info['description']
            result['expected_columns'] = file_info['expected_columns']
            result['source'] = file_info['source']
            
            validation_results.append(result)
            
            # Log result
            if result.get('valid', False):
                size_mb = result.get('size_bytes', 0) / (1024 * 1024)
                log(f"{file_path.name} exists ({size_mb:.2f} MB)")
            else:
                log(f"{file_path.name}: {result.get('message', 'Unknown error')}")

        log("File existence checks complete")
        # Additional Structure Validation for Existing Files
        # For files that exist, perform basic structure validation
        # Check column names (remembering gcode_lessons.md case sensitivity warnings)

        log("Performing structure validation for existing files...")
        
        for i, (file_info, result) in enumerate(zip(dependency_files, validation_results)):
            if result.get('valid', False) and str(file_info['path']).endswith('.csv'):
                try:
                    file_path = file_info['path']
                    log(f"Validating structure of {file_path.name}...")
                    
                    # Load CSV header only (fast)
                    df_header = pd.read_csv(file_path, nrows=0)
                    actual_columns = df_header.columns.tolist()
                    expected_columns = file_info['expected_columns']
                    
                    # Check if expected columns exist (allowing extra columns)
                    missing_columns = [col for col in expected_columns if col not in actual_columns]
                    
                    if missing_columns:
                        log(f"{file_path.name} missing expected columns: {missing_columns}")
                        log(f"Actual columns: {actual_columns}")
                        validation_results[i]['structure_warning'] = f"Missing columns: {missing_columns}"
                    else:
                        log(f"{file_path.name} has all expected columns")
                        validation_results[i]['structure_valid'] = True
                    
                    # Add actual column info to result
                    validation_results[i]['actual_columns'] = actual_columns
                    validation_results[i]['n_columns'] = len(actual_columns)
                    
                    # Get row count if file is reasonably sized
                    if result.get('size_bytes', 0) < 50 * 1024 * 1024:  # < 50MB
                        try:
                            df_size = pd.read_csv(file_path, usecols=[actual_columns[0]] if actual_columns else None)
                            validation_results[i]['n_rows'] = len(df_size)
                            log(f"{file_path.name} has {len(df_size)} rows")
                        except Exception as e:
                            log(f"Could not determine row count for {file_path.name}: {e}")
                    
                except Exception as e:
                    log(f"Structure validation failed for {file_path.name}: {e}")
                    validation_results[i]['structure_error'] = str(e)
        # Save Validation Output
        # Output: step00_dependency_validation.txt
        # Contains: Comprehensive validation report with file status and recommendations

        log("Generating dependency validation report...")
        
        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        # Create comprehensive validation report
        report_lines = []
        report_lines.append("# DEPENDENCY VALIDATION REPORT")
        report_lines.append("# Generated by step00_validate_dependencies.py")
        report_lines.append(f"# RQ: ch7/7.2.1 (Age Moderation of Test-VR Relationship)")
        report_lines.append("")
        report_lines.append("## VALIDATION SUMMARY")
        report_lines.append("")
        
        all_valid = True
        for result in validation_results:
            if not result.get('valid', False):
                all_valid = False
                break
        
        if all_valid:
            report_lines.append("STATUS: PASS - All dependencies validated successfully")
            report_lines.append("RECOMMENDATION: Proceed with analysis pipeline")
        else:
            report_lines.append("STATUS: FAIL - One or more dependencies missing or invalid")
            report_lines.append("RECOMMENDATION: Resolve dependency issues before proceeding")
        
        report_lines.append("")
        report_lines.append("## DETAILED RESULTS")
        report_lines.append("")
        
        for i, (file_info, result) in enumerate(zip(dependency_files, validation_results)):
            report_lines.append(f"### File {i+1}: {file_info['path'].name}")
            report_lines.append(f"Path: {file_info['path']}")
            report_lines.append(f"Source: {result.get('source', 'Unknown')}")
            report_lines.append(f"Description: {result.get('description', 'No description')}")
            report_lines.append("")
            
            # File existence and size
            if result.get('valid', False):
                size_mb = result.get('size_bytes', 0) / (1024 * 1024)
                report_lines.append(f"Existence: PASS")
                report_lines.append(f"Size: {size_mb:.2f} MB (meets {min_size_bytes} byte minimum)")
            else:
                report_lines.append(f"Existence: FAIL")
                report_lines.append(f"Issue: {result.get('message', 'Unknown error')}")
            
            # Structure validation (if applicable)
            if 'actual_columns' in result:
                expected_cols = file_info['expected_columns']
                actual_cols = result['actual_columns']
                n_rows = result.get('n_rows', 'Unknown')
                
                report_lines.append(f"Structure: {'PASS' if result.get('structure_valid') else 'WARN'}")
                report_lines.append(f"Rows: {n_rows}")
                report_lines.append(f"Columns: {len(actual_cols)}")
                report_lines.append(f"Expected columns: {expected_cols}")
                report_lines.append(f"Actual columns: {actual_cols}")
                
                if 'structure_warning' in result:
                    report_lines.append(f"Warning: {result['structure_warning']}")
                if 'structure_error' in result:
                    report_lines.append(f"Error: {result['structure_error']}")
            
            report_lines.append("")
        
        # Add next steps section
        report_lines.append("## NEXT STEPS")
        report_lines.append("")
        if all_valid:
            report_lines.append("1. All dependencies validated - ready for step01_extract_merge_data.py")
            report_lines.append("2. Expected to merge ~100 participants from both sources")
            report_lines.append("3. Ch7 analysis can proceed with hierarchical regression pipeline")
        else:
            report_lines.append("1. RESOLVE DEPENDENCY ISSUES:")
            for result in validation_results:
                if not result.get('valid', False):
                    report_lines.append(f"   - {result.get('message', 'Fix missing dependency')}")
            report_lines.append("2. Re-run step00 validation after fixes")
            report_lines.append("3. Do NOT proceed to step01 until all dependencies pass")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        log(f"{output_path.name} ({len(report_lines)} lines)")
        # Run Validation Tool (validate_data_format on report structure)
        # Validates: That our validation results have expected structure
        # Note: We'll create a simple DataFrame version of results for validation

        log("Running validate_data_format on validation results...")
        
        # Create DataFrame of validation results for validation tool
        validation_df_data = []
        for file_info, result in zip(dependency_files, validation_results):
            validation_df_data.append({
                'file_name': file_info['path'].name,
                'valid': result.get('valid', False),
                'size_bytes': result.get('size_bytes', 0),
                'message': result.get('message', ''),
                'structure_valid': result.get('structure_valid', False)
            })
        
        validation_df = pd.DataFrame(validation_df_data)
        
        # Required columns for our validation results
        required_cols = ['file_name', 'valid', 'size_bytes', 'message']
        
        validation_result = validate_data_format(df=validation_df, required_cols=required_cols)
        
        # Report validation results
        if isinstance(validation_result, dict):
            for key, value in validation_result.items():
                log(f"{key}: {value}")
        else:
            log(f"{validation_result}")

        # Final status report
        if all_valid:
            log("Step 00 complete - All dependencies validated successfully")
            log("  -> Ready to proceed with step01_extract_merge_data.py")
            sys.exit(0)
        else:
            failed_files = [result.get('file_path', 'unknown') for result in validation_results if not result.get('valid', False)]
            log(f"Step 00 complete - Dependency validation failed")
            log(f"  -> Failed files: {failed_files}")
            log(f"  -> Review {output_path.name} for details")
            log(f"  -> Do NOT proceed to step01 until dependencies are resolved")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)