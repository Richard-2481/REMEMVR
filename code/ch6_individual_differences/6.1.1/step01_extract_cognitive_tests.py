#!/usr/bin/env python3
"""
Step 01: Extract and Prepare Cognitive Test Data (FIXED)
RQ: ch7/7.1.1
Purpose: Extract cognitive test scores from dfnonvr.csv and standardize to T-scores
Output: results/ch7/7.1.1/data/step01_cognitive_tests.csv

Changes:
1. Uses correct column names from DATA_DICTIONARY.md
2. Adds proper missing data analysis and documentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import utilities
sys.path.insert(0, str(PROJECT_ROOT / "results" / "ch7"))
from column_name_fix import get_correct_column_name, COLUMN_MAPPINGS
from missing_data_handler import (
    analyze_missing_pattern, 
    create_missing_data_report,
    handle_missing_data,
    document_excluded_participants
)

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_cognitive_tests.csv"
MISSING_REPORT_FILE = RQ_DIR / "data" / "step01_missing_data_report.txt"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def compute_t_score(raw_scores):
    """Convert raw scores to T-scores (M=50, SD=10)."""
    if len(raw_scores) == 0:
        return []
    mean = np.nanmean(raw_scores)
    std = np.nanstd(raw_scores)
    if std == 0:
        return np.full_like(raw_scores, 50.0)
    return 50 + 10 * (raw_scores - mean) / std

if __name__ == "__main__":
    try:
        log("Step 01: Extract and Prepare Cognitive Test Data - FIXED VERSION")
        log("Using correct column names from DATA_DICTIONARY.md")
        log("Adding proper missing data analysis")
        log(f"RQ Directory: {RQ_DIR}")
        log(f"Output will be saved to: {OUTPUT_FILE}")
        # Load participant data with CORRECT column names
        log("\nLoading participant data from dfnonvr.csv...")
        
        data_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        cognitive_df = pd.read_csv(data_path)
        log(f"Loaded data for {len(cognitive_df)} participants")
        # Extract cognitive tests with CORRECT column names
        log("\nExtracting cognitive test scores with correct column names...")
        
        extracted_data = pd.DataFrame()
        extracted_data['UID'] = cognitive_df['UID'].astype(str)
        
        # RAVLT: Sum trials 1-5 using CORRECT column names
        ravlt_trial_cols = []
        for i in range(1, 6):
            # CORRECT column name format
            col = f'ravlt-trial-{i}-score'
            if col in cognitive_df.columns:
                ravlt_trial_cols.append(col)
            else:
                log(f"Column '{col}' not found")
        
        if len(ravlt_trial_cols) == 5:
            log(f"Found all 5 RAVLT trial columns")
            extracted_data['RAVLT_Total'] = cognitive_df[ravlt_trial_cols].sum(axis=1)
            log(f"  RAVLT Total: M={extracted_data['RAVLT_Total'].mean():.1f}, SD={extracted_data['RAVLT_Total'].std():.1f}")
        else:
            log(f"Only found {len(ravlt_trial_cols)}/5 RAVLT trial columns")
            extracted_data['RAVLT_Total'] = np.nan
        
        # RAVLT Delayed Recall - CORRECT column name
        if 'ravlt-delayed-recall-score' in cognitive_df.columns:
            extracted_data['RAVLT_DR'] = cognitive_df['ravlt-delayed-recall-score']
            log(f"RAVLT delayed recall extracted")
        else:
            log("RAVLT delayed recall column not found")
            extracted_data['RAVLT_DR'] = np.nan
        
        # BVMT Total - CORRECT column name
        if 'bvmt-total-recall' in cognitive_df.columns:
            extracted_data['BVMT_Total'] = cognitive_df['bvmt-total-recall']
            log(f"BVMT total recall extracted")
        else:
            log("BVMT total recall column not found")
            extracted_data['BVMT_Total'] = np.nan
        
        # NART - CORRECT column name
        if 'nart-score' in cognitive_df.columns:
            extracted_data['NART_Score'] = cognitive_df['nart-score']
            log(f"NART score extracted")
        else:
            log("NART score column not found")
            extracted_data['NART_Score'] = np.nan
        
        # RPM - CORRECT column name
        if 'rpm-score' in cognitive_df.columns:
            extracted_data['RPM_Score'] = cognitive_df['rpm-score']
            log(f"RPM score extracted")
        else:
            log("RPM score column not found")
            extracted_data['RPM_Score'] = np.nan
        # Analyze missing data BEFORE processing
        log("\n[MISSING DATA] Analyzing missing data patterns...")
        
        key_columns = ['RAVLT_Total', 'RAVLT_DR', 'BVMT_Total', 'NART_Score', 'RPM_Score']
        
        # Generate missing data report
        missing_report = create_missing_data_report(extracted_data, key_columns)
        log("\n" + missing_report)
        
        # Save missing data report
        with open(MISSING_REPORT_FILE, 'w') as f:
            f.write(missing_report)
        log(f"\nMissing data report saved to: {MISSING_REPORT_FILE}")
        
        # Document excluded participants if any
        complete_data = extracted_data.dropna(subset=key_columns)
        n_excluded = len(extracted_data) - len(complete_data)
        
        if n_excluded > 0:
            log(f"\n{n_excluded} participants will be excluded due to missing data")
            
            # Get demographic data for comparison
            demo_cols = ['age', 'sex', 'education']
            for col in demo_cols:
                if col in cognitive_df.columns:
                    extracted_data[col] = cognitive_df[col]
            
            # Compare included vs excluded
            comparison = document_excluded_participants(
                extracted_data, complete_data, 'UID', demo_cols
            )
            
            if not comparison.empty:
                log("\nComparison of included vs excluded participants:")
                log(comparison.to_string())
                
                # Save comparison
                comparison_file = RQ_DIR / "data" / "step01_exclusion_comparison.csv"
                comparison.to_csv(comparison_file, index=False)
                log(f"Exclusion comparison saved to: {comparison_file}")
        # Convert to T-scores
        log("\nConverting raw scores to T-scores (M=50, SD=10)...")
        
        # Use complete cases for T-score calculation
        t_score_df = complete_data.copy()
        
        # Convert each test to T-scores
        t_score_df['RAVLT_Total_T'] = compute_t_score(t_score_df['RAVLT_Total'].values)
        t_score_df['RAVLT_DR_T'] = compute_t_score(t_score_df['RAVLT_DR'].values)
        t_score_df['BVMT_Total_T'] = compute_t_score(t_score_df['BVMT_Total'].values)
        t_score_df['NART_T'] = compute_t_score(t_score_df['NART_Score'].values)
        t_score_df['RPM_T'] = compute_t_score(t_score_df['RPM_Score'].values)
        
        # Report T-score statistics
        log("\n[T-SCORES] Summary statistics:")
        for col in ['RAVLT_Total_T', 'RAVLT_DR_T', 'BVMT_Total_T', 'NART_T', 'RPM_T']:
            mean = t_score_df[col].mean()
            std = t_score_df[col].std()
            log(f"  {col:15} M={mean:.1f}, SD={std:.1f}")
        # Save final dataset
        
        # Select final columns
        final_columns = ['UID', 'RAVLT_Total_T', 'RAVLT_DR_T', 'BVMT_Total_T', 'NART_T', 'RPM_T']
        final_df = t_score_df[final_columns]
        
        # Save
        final_df.to_csv(OUTPUT_FILE, index=False)
        log(f"\nSaved cognitive test T-scores to: {OUTPUT_FILE}")
        log(f"Final dataset: {final_df.shape[0]} participants, {final_df.shape[1]} columns")
        # Create summary
        log("\n" + "="*60)
        log("SUMMARY")
        log("="*60)
        log(f"Total participants loaded: {len(cognitive_df)}")
        log(f"Participants with complete data: {len(complete_data)}")
        log(f"Participants excluded: {n_excluded}")
        log(f"Exclusion rate: {(n_excluded/len(cognitive_df)*100):.1f}%")
        log("\nColumn name fixes applied:")
        for old, new in COLUMN_MAPPINGS.items():
            if 'ravlt' in new.lower() or 'bvmt' in new.lower() or 'nart' in new.lower() or 'rpm' in new.lower():
                log(f"  {old:30} → {new}")
        log("\nStep 01 complete - All column names correct, missing data documented")
        
    except Exception as e:
        log(f"\nScript failed: {str(e)}")
        import traceback
        log(f"\n{traceback.format_exc()}")
        sys.exit(1)