#!/usr/bin/env python3
"""Validate Dependencies: Validate required Ch6 6.1.1 outputs and dfnonvr.csv exist before proceeding with RQ 7.3.1."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.1 (derived from script location)
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
        log("Step 0: Validate Dependencies")
        # Initialize Validation Report

        log("Initializing dependency validation...")
        validation_results = {
            'ch6_file_check': {},
            'dfnonvr_file_check': {},
            'column_validation': {},
            'data_quality': {},
            'overall_status': 'PENDING'
        }
        # Check Ch6 6.1.1 Confidence Theta File

        log("Validating Ch6 6.1.1 confidence theta file...")
        ch6_file_path = PROJECT_ROOT / "results" / "ch6" / "6.1.1" / "data" / "step03_theta_confidence.csv"
        
        if ch6_file_path.exists():
            log(f"Ch6 file exists: {ch6_file_path}")
            
            # Load and examine Ch6 data
            ch6_df = pd.read_csv(ch6_file_path)
            ch6_cols = ch6_df.columns.tolist()
            log(f"Ch6 file ({len(ch6_df)} rows, {len(ch6_df.columns)} cols)")
            log(f"Ch6 actual columns: {ch6_cols}")
            
            # Expected columns per 4_analysis.yaml: ["UID", "confidence_theta", "se_theta"]
            # Actual columns found: ["composite_ID", "theta_All", "se_All"]
            log("Column name mismatch detected:")
            log("  Expected: ['UID', 'confidence_theta', 'se_theta']")
            log("  Actual:   ['composite_ID', 'theta_All', 'se_All']")
            
            # Validate using actual column names
            required_ch6_cols = ['composite_ID', 'theta_All', 'se_All']
            ch6_validation = validate_data_columns(ch6_df, required_ch6_cols)
            
            if ch6_validation.get('valid', False):
                log("Ch6 file has all required columns")
                
                # Check data quality
                n_participants = len(ch6_df)
                theta_range = (ch6_df['theta_All'].min(), ch6_df['theta_All'].max())
                se_range = (ch6_df['se_All'].min(), ch6_df['se_All'].max())
                
                log(f"Ch6 participants: {n_participants}")
                log(f"Theta range: [{theta_range[0]:.3f}, {theta_range[1]:.3f}]")
                log(f"SE range: [{se_range[0]:.3f}, {se_range[1]:.3f}]")
                
                # Validate theta scores are in reasonable IRT range [-3, 3]
                valid_theta = ((ch6_df['theta_All'] >= -3) & (ch6_df['theta_All'] <= 3)).all()
                valid_se = (ch6_df['se_All'] > 0).all()
                
                if valid_theta and valid_se and n_participants >= 90:
                    validation_results['ch6_file_check'] = {
                        'status': 'PASS',
                        'path': str(ch6_file_path),
                        'participants': n_participants,
                        'theta_range': theta_range,
                        'se_range': se_range,
                        'notes': 'Column names differ from spec but data is valid'
                    }
                    log("Ch6 data quality validation successful")
                else:
                    validation_results['ch6_file_check'] = {
                        'status': 'FAIL',
                        'reason': f'Data quality issues: valid_theta={valid_theta}, valid_se={valid_se}, n_participants={n_participants}'
                    }
                    log("Ch6 data quality validation failed")
            else:
                validation_results['ch6_file_check'] = {
                    'status': 'FAIL',
                    'reason': f"Missing columns: {ch6_validation.get('missing_columns', [])}"
                }
                log(f"Ch6 column validation failed: {ch6_validation}")
        else:
            validation_results['ch6_file_check'] = {
                'status': 'FAIL',
                'reason': 'File does not exist'
            }
            log(f"Ch6 file not found: {ch6_file_path}")
        # Check dfnonvr.csv Cognitive Test Data

        log("Validating dfnonvr.csv cognitive test data...")
        dfnonvr_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        
        if dfnonvr_path.exists():
            log(f"dfnonvr.csv exists: {dfnonvr_path}")
            
            # Load and examine dfnonvr.csv
            dfnonvr_df = pd.read_csv(dfnonvr_path)
            log(f"dfnonvr.csv ({len(dfnonvr_df)} rows, {len(dfnonvr_df.columns)} cols)")
            
            # Required columns from DATA_DICTIONARY.md (exact names verified)
            required_cognitive_cols = [
                'UID',
                'ravlt-trial-1-score', 'ravlt-trial-2-score', 'ravlt-trial-3-score', 
                'ravlt-trial-4-score', 'ravlt-trial-5-score',
                'bvmt-trial-1-score', 'bvmt-trial-2-score', 'bvmt-trial-3-score', 
                'bvmt-delayed-recall-score',
                'rpm-score',
                'age', 'sex', 'education'
            ]
            
            cognitive_validation = validate_data_columns(dfnonvr_df, required_cognitive_cols)
            
            if cognitive_validation.get('valid', False):
                log("dfnonvr.csv has all required cognitive test columns")
                
                # Check data availability
                n_participants = len(dfnonvr_df)
                
                # Sample some cognitive test score ranges
                ravlt_scores = dfnonvr_df[[col for col in dfnonvr_df.columns if 'ravlt-trial' in col and 'score' in col]]
                bvmt_scores = dfnonvr_df[[col for col in dfnonvr_df.columns if 'bvmt-trial' in col and 'score' in col] + ['bvmt-delayed-recall-score']]
                rpm_scores = dfnonvr_df['rpm-score']
                
                log(f"dfnonvr.csv participants: {n_participants}")
                log(f"RAVLT trials available: {len(ravlt_scores.columns)}")
                log(f"BVMT trials available: {len(bvmt_scores.columns)}")
                log(f"RPM score range: [{rpm_scores.min():.1f}, {rpm_scores.max():.1f}]")
                
                # Check for reasonable data ranges
                rpm_valid = (rpm_scores >= 0).all() and (rpm_scores <= 12).all()  # RPM is 0-12 scale
                age_valid = (dfnonvr_df['age'] >= 18).all() and (dfnonvr_df['age'] <= 80).all()  # Reasonable age range
                
                if rpm_valid and age_valid and n_participants >= 90:
                    validation_results['dfnonvr_file_check'] = {
                        'status': 'PASS',
                        'path': str(dfnonvr_path),
                        'participants': n_participants,
                        'ravlt_trials': len(ravlt_scores.columns),
                        'bvmt_trials': len(bvmt_scores.columns),
                        'rpm_range': (float(rpm_scores.min()), float(rpm_scores.max())),
                        'age_range': (float(dfnonvr_df['age'].min()), float(dfnonvr_df['age'].max()))
                    }
                    log("dfnonvr.csv data quality validation successful")
                else:
                    validation_results['dfnonvr_file_check'] = {
                        'status': 'FAIL',
                        'reason': f'Data quality issues: rpm_valid={rpm_valid}, age_valid={age_valid}, n_participants={n_participants}'
                    }
                    log("dfnonvr.csv data quality validation failed")
            else:
                validation_results['dfnonvr_file_check'] = {
                    'status': 'FAIL',
                    'reason': f"Missing columns: {cognitive_validation.get('missing_columns', [])}"
                }
                log(f"dfnonvr.csv column validation failed: {cognitive_validation}")
        else:
            validation_results['dfnonvr_file_check'] = {
                'status': 'FAIL',
                'reason': 'File does not exist'
            }
            log(f"dfnonvr.csv not found: {dfnonvr_path}")
        # Generate Validation Report

        log("Generating dependency validation report...")
        
        # Determine overall status
        ch6_pass = validation_results['ch6_file_check'].get('status') == 'PASS'
        dfnonvr_pass = validation_results['dfnonvr_file_check'].get('status') == 'PASS'
        
        if ch6_pass and dfnonvr_pass:
            validation_results['overall_status'] = 'PASS'
            log("All dependencies validated successfully")
        else:
            validation_results['overall_status'] = 'FAIL'
            log("One or more dependencies failed validation")

        # Write validation report
        report_path = RQ_DIR / "data" / "step00_dependency_validation.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=============================================================================\n")
            f.write("DEPENDENCY VALIDATION REPORT - RQ 7.3.1\n")
            f.write("=============================================================================\n")
            f.write(f"Purpose: Validate Ch6 6.1.1 outputs and dfnonvr.csv for confidence prediction\n")
            f.write(f"Overall Status: {validation_results['overall_status']}\n")
            f.write("\n")
            
            f.write("CH6 6.1.1 CONFIDENCE THETA SCORES:\n")
            f.write("-----------------------------------\n")
            ch6_check = validation_results['ch6_file_check']
            f.write(f"Status: {ch6_check.get('status', 'UNKNOWN')}\n")
            if ch6_check.get('status') == 'PASS':
                f.write(f"File: {ch6_check['path']}\n")
                f.write(f"Participants: {ch6_check['participants']}\n")
                f.write(f"Theta Range: [{ch6_check['theta_range'][0]:.3f}, {ch6_check['theta_range'][1]:.3f}]\n")
                f.write(f"SE Range: [{ch6_check['se_range'][0]:.3f}, {ch6_check['se_range'][1]:.3f}]\n")
                f.write(f"Notes: {ch6_check['notes']}\n")
            else:
                f.write(f"Failure Reason: {ch6_check.get('reason', 'Unknown')}\n")
            f.write("\n")
            
            f.write("DFNONVR.CSV COGNITIVE TESTS:\n")
            f.write("----------------------------\n")
            dfnonvr_check = validation_results['dfnonvr_file_check']
            f.write(f"Status: {dfnonvr_check.get('status', 'UNKNOWN')}\n")
            if dfnonvr_check.get('status') == 'PASS':
                f.write(f"File: {dfnonvr_check['path']}\n")
                f.write(f"Participants: {dfnonvr_check['participants']}\n")
                f.write(f"RAVLT Trials: {dfnonvr_check['ravlt_trials']}\n")
                f.write(f"BVMT Trials: {dfnonvr_check['bvmt_trials']}\n")
                f.write(f"RPM Range: [{dfnonvr_check['rpm_range'][0]:.1f}, {dfnonvr_check['rpm_range'][1]:.1f}]\n")
                f.write(f"Age Range: [{dfnonvr_check['age_range'][0]:.1f}, {dfnonvr_check['age_range'][1]:.1f}]\n")
            else:
                f.write(f"Failure Reason: {dfnonvr_check.get('reason', 'Unknown')}\n")
            f.write("\n")
            
            f.write("CRITICAL NOTES:\n")
            f.write("---------------\n")
            f.write("1. Ch6 file uses different column names than spec:\n")
            f.write("   - Expected: UID, confidence_theta, se_theta\n")
            f.write("   - Actual: composite_ID, theta_All, se_All\n")
            f.write("2. All downstream steps must use ACTUAL column names\n")
            f.write("3. Column mapping will be needed for UID merging\n")
            f.write("\n")
            
            f.write("RECOMMENDATION:\n")
            f.write("---------------\n")
            if validation_results['overall_status'] == 'PASS':
                f.write("PROCEED with RQ 7.3.1 analysis steps.\n")
                f.write("Note column name differences for Step 1 implementation.\n")
            else:
                f.write("DO NOT PROCEED - resolve dependency issues first.\n")
                f.write("Check failed components and re-run validation.\n")
            f.write("\n")
            f.write("=============================================================================\n")

        log(f"Validation report: {report_path}")
        # Final Status and Exit
        # DEPENDENCY CHECK COMPLETE (log pattern per validation_criteria)

        log("DEPENDENCY CHECK COMPLETE")
        
        if validation_results['overall_status'] == 'PASS':
            log("Step 0 complete - all dependencies validated")
            sys.exit(0)
        else:
            log("Step 0 failed - dependency validation incomplete")
            sys.exit(1)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)