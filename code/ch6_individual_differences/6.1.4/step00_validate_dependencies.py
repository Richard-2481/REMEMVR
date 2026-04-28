#!/usr/bin/env python3
"""validate_dependencies: Validate that all required dependency files exist and contain expected data"""

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.4 (derived from script location)
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
        # Define Dependencies to Validate

        log("Defining dependency files to validate...")
        
        dependencies = {
            "ch5_theta_scores": {
                "path": PROJECT_ROOT / "results" / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv",
                "description": "Ch5 VR ability theta scores",
                "expected_columns": ["UID", "theta"],
                "expected_participants": 100
            },
            "dfnonvr_data": {
                "path": PROJECT_ROOT / "data" / "dfnonvr.csv", 
                "description": "Cognitive and demographic data",
                "expected_columns": ["UID"],  # Will check actual columns later
                "expected_participants": 100
            }
        }
        
        log(f"{len(dependencies)} dependency files to validate")
        # Run Validation Tool for Each Dependency

        log("Running check_file_exists for each dependency...")
        
        validation_results = {}
        all_valid = True
        
        for dep_name, dep_info in dependencies.items():
            log(f"Checking {dep_name}: {dep_info['path']}")
            
            # Run analysis tool
            result = check_file_exists(
                file_path=str(dep_info["path"]),  # Convert Path to str
                min_size_bytes=1000  # Ensure file is not empty (>1KB)
            )
            
            validation_results[dep_name] = result
            
            if result.get("valid", False):
                log(f"{dep_name}: File exists ({result.get('size_bytes', 0)} bytes)")
            else:
                log(f"{dep_name}: {result.get('message', 'Validation failed')}")
                all_valid = False
        # Additional Data Quality Checks
        # Check participant counts and basic data structure

        log("Performing additional data quality checks...")
        
        data_quality_results = {}
        
        for dep_name, dep_info in dependencies.items():
            if validation_results[dep_name].get("valid", False):
                try:
                    # Load file to check data quality
                    df = pd.read_csv(dep_info["path"])
                    
                    # Check participant count
                    n_participants = len(df)
                    has_uid = "UID" in df.columns
                    
                    data_quality_results[dep_name] = {
                        "participants": n_participants,
                        "has_uid": has_uid,
                        "columns": df.columns.tolist()[:10],  # First 10 columns only
                        "valid": has_uid and n_participants >= 50  # Allow some flexibility
                    }
                    
                    if data_quality_results[dep_name]["valid"]:
                        log(f"{dep_name}: {n_participants} participants, UID column present")
                    else:
                        log(f"{dep_name}: Issues detected (participants={n_participants}, UID={has_uid})")
                        all_valid = False
                        
                except Exception as e:
                    log(f"{dep_name}: Error reading file - {str(e)}")
                    data_quality_results[dep_name] = {"valid": False, "error": str(e)}
                    all_valid = False
            else:
                data_quality_results[dep_name] = {"valid": False, "reason": "File validation failed"}
        # Generate Validation Report
        # Output: Validation report text file with detailed status
        # Contains: File existence, size, participant counts, overall status

        log("Generating dependency validation report...")
        
        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DEPENDENCY VALIDATION REPORT\n")
            f.write("RQ: Ch7 7.1.4 - Incremental Validity of VR Assessment\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write("="*60 + "\n\n")
            
            f.write("OVERALL STATUS: ")
            if all_valid:
                f.write("PASS - All dependencies validated successfully\n\n")
            else:
                f.write("FAIL - One or more dependencies failed validation\n\n")
            
            f.write("DEPENDENCY DETAILS:\n")
            f.write("-"*40 + "\n")
            
            for dep_name, dep_info in dependencies.items():
                f.write(f"\n{dep_name.upper()}: {dep_info['description']}\n")
                f.write(f"Path: {dep_info['path']}\n")
                
                # File existence results
                file_result = validation_results[dep_name]
                f.write(f"File exists: {'YES' if file_result.get('valid', False) else 'NO'}\n")
                if file_result.get('valid', False):
                    f.write(f"File size: {file_result.get('size_bytes', 0)} bytes\n")
                else:
                    f.write(f"Issue: {file_result.get('message', 'Unknown error')}\n")
                
                # Data quality results
                if dep_name in data_quality_results:
                    data_result = data_quality_results[dep_name]
                    if data_result.get("valid", False):
                        f.write(f"Participants: {data_result.get('participants', 'unknown')}\n")
                        f.write(f"Has UID column: {'YES' if data_result.get('has_uid', False) else 'NO'}\n")
                        f.write(f"Sample columns: {', '.join(data_result.get('columns', []))}\n")
                    else:
                        f.write(f"Data quality: FAIL ({data_result.get('error', data_result.get('reason', 'unknown'))})\n")
                
                f.write(f"Status: {'PASS' if validation_results[dep_name].get('valid', False) and data_quality_results.get(dep_name, {}).get('valid', False) else 'FAIL'}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("NEXT STEPS:\n")
            if all_valid:
                f.write("- All dependencies validated\n")
                f.write("- Ready to proceed with RQ 7.1.4 analysis steps\n")
                f.write("- Expected ~100 participants from Ch5 + cognitive/demographic data\n")
            else:
                f.write("- FIX DEPENDENCY ISSUES before proceeding\n")
                f.write("- Check that Ch5 5.1.1 completed successfully\n") 
                f.write("- Ensure dfnonvr.csv contains expected data\n")
        
        log(f"step00_dependency_validation.txt ({output_path.stat().st_size} bytes)")
        # Run Manual Validation
        # Manual validation: Check overall validation status
        # Validates: All dependencies pass both file existence and data quality checks

        log("Running manual validation...")
        
        if all_valid:
            log("Overall status: PASS")
            log(f"Dependencies validated: {len(dependencies)}")
            log("Ready for RQ 7.1.4 analysis")
        else:
            failed_deps = [name for name, result in validation_results.items() 
                          if not result.get("valid", False) or 
                          not data_quality_results.get(name, {}).get("valid", False)]
            log(f"Overall status: FAIL")
            log(f"Failed dependencies: {', '.join(failed_deps)}")
            log("Must fix dependency issues before proceeding")

        log("Step 00 complete")
        
        # Exit with appropriate code
        if all_valid:
            sys.exit(0)
        else:
            log("Dependency validation failed - stopping execution")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)