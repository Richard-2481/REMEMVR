#!/usr/bin/env python3
"""validate_dependencies: Verify Ch6 calibration data and dfnonvr.csv accessibility for RQ 7.3.2 analysis."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import os
import glob

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.2 (derived from script location)
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
        # Validate Dependencies Using Standard Library Implementation

        log("Running dependency validation...")
        
        # Standard library implementation for dependency checking
        # (from 4_analysis.yaml inline implementation)
        results = []
        
        # Check Ch6 calibration data
        log("Looking for Ch6 calibration data...")
        ch6_patterns = [
            str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*calibration*.csv'),
            str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*resolution*.csv'), 
            str(PROJECT_ROOT / 'results' / 'ch6' / '*' / 'data' / '*brier*.csv')
        ]
        
        ch6_found = False
        ch6_source = None
        for pattern in ch6_patterns:
            files = glob.glob(pattern)
            if files:
                ch6_found = True
                ch6_source = files[0]
                results.append(f"Ch6 calibration source: {ch6_source}")
                log(f"Ch6 calibration data: {ch6_source}")
                break
        
        if not ch6_found:
            results.append("Ch6 calibration source: NOT FOUND")
            log("Ch6 calibration data: NOT FOUND")
            
        # Check dfnonvr.csv (Ch7 uses this, NOT master.xlsx per gcode_lessons.md)
        log("Looking for dfnonvr.csv...")
        dfnonvr_path = PROJECT_ROOT / 'data' / 'dfnonvr.csv'
        if dfnonvr_path.exists():
            results.append("dfnonvr.csv: ACCESSIBLE")
            log("dfnonvr.csv: ACCESSIBLE")
            
            # Verify cognitive test columns exist
            log("Verifying cognitive test columns in dfnonvr.csv...")
            df = pd.read_csv(dfnonvr_path, nrows=1)  # Header only for fast check
            required_cols = ['ravlt-trial-1-score', 'bvmt-trial-1-score', 'rpm-score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results.append(f"Missing columns: {missing_cols}")
                log(f"Missing cognitive test columns: {missing_cols}")
            else:
                results.append("Cognitive test columns: VERIFIED")
                log("Cognitive test columns: VERIFIED")
        else:
            results.append("dfnonvr.csv: NOT FOUND")
            log("dfnonvr.csv: NOT FOUND")
        
        # Summary
        if ch6_found and dfnonvr_path.exists():
            results.append("Dependency validation: PASS")
            log("Dependency validation: PASS")
        else:
            results.append("Dependency validation: FAIL")
            log("Dependency validation: FAIL")
        # Save Dependency Validation Report
        # Output: results/ch7/7.3.2/data/step00_dependency_validation.txt
        # Contains: Validation results for Ch6 and dfnonvr.csv dependencies
        
        output_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        log(f"Saving dependency validation to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== RQ 7.3.2 DEPENDENCY VALIDATION ===\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"RQ Directory: {RQ_DIR}\n")
            f.write(f"Project Root: {PROJECT_ROOT}\n\n")
            
            for result in results:
                f.write(f"{result}\n")
            
            f.write("\n=== DETAILED FINDINGS ===\n")
            if ch6_found:
                f.write(f"Ch6 Data Source: {ch6_source}\n")
                f.write("Ch6 Status: Available for analysis\n")
            else:
                f.write("Ch6 Data Source: None found\n")
                f.write("Ch6 Status: Missing - check Ch6 analysis completion\n")
            
            f.write(f"dfnonvr.csv Path: {dfnonvr_path}\n")
            if dfnonvr_path.exists():
                f.write("dfnonvr.csv Status: Accessible\n")
                # Report actual columns for verification
                df_check = pd.read_csv(dfnonvr_path, nrows=0)
                cognitive_cols = [col for col in df_check.columns if any(test in col.lower() 
                                 for test in ['ravlt', 'bvmt', 'rpm'])]
                f.write(f"Available cognitive columns: {len(cognitive_cols)} found\n")
                f.write(f"Cognitive column examples: {cognitive_cols[:5]}\n")
            else:
                f.write("dfnonvr.csv Status: Missing\n")
            
            f.write("\n=== NEXT STEPS ===\n")
            if ch6_found and dfnonvr_path.exists():
                f.write("Dependencies validated - proceed with Step 01\n")
            else:
                f.write("Dependencies incomplete - resolve missing data before proceeding\n")
        
        log(f"{output_path} ({len(results)} validation results)")
        # Run Validation of Dependency Check
        # Validates: Dependency validation file was created with required patterns
        
        log("Running validation of dependency check...")
        
        # Standard library implementation for validation
        # (from 4_analysis.yaml inline validation implementation)
        validation_file = output_path
        if not validation_file.exists():
            validation_result = {"valid": False, "message": "Dependency validation file missing"}
        else:
            with open(validation_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_patterns = ["Ch6 calibration source", "dfnonvr.csv: ACCESSIBLE", "Dependency validation: PASS"]
            missing_patterns = [p for p in required_patterns if p not in content]
            
            if missing_patterns:
                validation_result = {"valid": False, "message": f"Missing patterns: {missing_patterns}"}
            else:
                validation_result = {"valid": True, "message": "All dependencies validated"}

        # Report validation results
        if validation_result["valid"]:
            log(f"[VALIDATION PASS] {validation_result['message']}")
        else:
            log(f"[VALIDATION FAIL] {validation_result['message']}")
        
        log("Step 00 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)