#!/usr/bin/env python3
"""
Step 01: Extract and Prepare Cognitive Test Data
RQ: ch7/7.1.1
Purpose: Extract cognitive test scores from dfnonvr.csv and standardize to T-scores
Output: results/ch7/7.1.1/data/step01_cognitive_tests.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Go up 4 levels from code file
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.1
LOG_FILE = RQ_DIR / "logs" / "step01_extract_cognitive_tests.log"
OUTPUT_FILE = RQ_DIR / "data" / "step01_cognitive_tests.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 01: Extract and Prepare Cognitive Test Data")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        log(f"[SETUP] Output will be saved to: {OUTPUT_FILE}")
        
        # =========================================================================
        # STEP 1: Load participant data with cognitive tests
        # =========================================================================
        log("[DATA] Loading participant data from dfnonvr.csv...")
        
        # Load the preprocessed participant data directly
        data_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load the CSV directly to avoid function signature issues
        cognitive_df = pd.read_csv(data_path)
        
        log(f"[DATA] Loaded cognitive test data for {len(cognitive_df)} participants")
        log(f"[DATA] Columns extracted: {list(cognitive_df.columns)}")
        
        # =========================================================================
        # STEP 2: Check what we got and compute derived scores if needed
        # =========================================================================
        log("[PROCESS] Computing derived scores...")
        
        # Check if RAVLT_Total already exists or needs to be computed
        if 'RAVLT_Total' not in cognitive_df.columns:
            log("[PROCESS] Computing RAVLT_Total from trial scores...")
            # Look for RAVLT trial 1-5 columns specifically (not distraction trial)
            ravlt_trial_cols = []
            for i in range(1, 6):
                col = f'RAVLT trial {i} score'
                if col in cognitive_df.columns:
                    ravlt_trial_cols.append(col)
            
            if len(ravlt_trial_cols) == 5:
                log(f"[PROCESS] Summing RAVLT trials 1-5: {ravlt_trial_cols}")
                cognitive_df['RAVLT_Total'] = cognitive_df[ravlt_trial_cols].sum(axis=1)
            else:
                log("[WARNING] RAVLT trial columns not found, checking for RAVLT_Total...")
        
        # Check what cognitive test columns we have
        test_cols = []
        for test in ['RAVLT', 'BVMT', 'NART', 'RPM']:
            matching = [col for col in cognitive_df.columns if test in col]
            log(f"[DATA] Found {test} columns: {matching}")
            test_cols.extend(matching)
        
        # =========================================================================
        # STEP 3: Select and rename columns for analysis
        # =========================================================================
        log("[PROCESS] Selecting and renaming cognitive test columns...")
        
        # Create standardized column names
        rename_map = {}
        final_df = pd.DataFrame()
        final_df['UID'] = cognitive_df['UID'].astype(str)
        
        # RAVLT - use total score
        if 'RAVLT_Total' in cognitive_df.columns:
            rename_map['RAVLT_Total'] = 'RAVLT_Raw'
            final_df['RAVLT_Raw'] = cognitive_df['RAVLT_Total']
        elif 'RAVLT total' in cognitive_df.columns:
            final_df['RAVLT_Raw'] = cognitive_df['RAVLT total']
        else:
            log("[WARNING] RAVLT_Total not found")
        
        # BVMT - look for total recall score
        bvmt_col = next((col for col in cognitive_df.columns if 'BVMT' in col and ('total' in col.lower() or 'TotR' in col)), None)
        if bvmt_col:
            final_df['BVMT_Raw'] = cognitive_df[bvmt_col]
            log(f"[DATA] Using {bvmt_col} for BVMT")
        else:
            log("[WARNING] BVMT total score not found")
        
        # NART - should be in the extracted data
        nart_col = next((col for col in cognitive_df.columns if 'NART' in col), None)
        if nart_col:
            final_df['NART_Raw'] = cognitive_df[nart_col]
            log(f"[DATA] Using {nart_col} for NART")
        else:
            log("[WARNING] NART score not found")
        
        # RPM - Raven's Progressive Matrices
        rpm_col = next((col for col in cognitive_df.columns if 'RPM' in col or 'Raven' in col), None)
        if rpm_col:
            final_df['RPM_Raw'] = cognitive_df[rpm_col]
            log(f"[DATA] Using {rpm_col} for RPM")
        else:
            log("[WARNING] RPM score not found")
        
        # =========================================================================
        # STEP 4: Convert to T-scores (M=50, SD=10)
        # =========================================================================
        log("[PROCESS] Converting raw scores to T-scores (M=50, SD=10)...")
        
        for test in ['RAVLT', 'BVMT', 'NART', 'RPM']:
            raw_col = f'{test}_Raw'
            t_col = f'{test}_T'
            
            if raw_col in final_df.columns:
                # Remove missing values for calculation
                valid_scores = final_df[raw_col].dropna()
                
                if len(valid_scores) > 0:
                    # Calculate T-score: T = 50 + 10 * ((raw - mean) / sd)
                    mean = valid_scores.mean()
                    sd = valid_scores.std()
                    
                    log(f"[STATS] {test}: Mean={mean:.2f}, SD={sd:.2f}, N={len(valid_scores)}")
                    
                    if sd > 0:
                        final_df[t_col] = 50 + 10 * ((final_df[raw_col] - mean) / sd)
                        
                        # Verify T-score properties
                        t_mean = final_df[t_col].mean()
                        t_sd = final_df[t_col].std()
                        log(f"[VERIFY] {test}_T: Mean={t_mean:.2f}, SD={t_sd:.2f}")
                    else:
                        log(f"[WARNING] {test} has zero variance, cannot compute T-scores")
                        final_df[t_col] = 50  # Set to mean if no variance
                else:
                    log(f"[WARNING] No valid {test} scores found")
        
        # =========================================================================
        # STEP 5: Handle missing data
        # =========================================================================
        log("[PROCESS] Checking for missing data...")
        
        # Report missing data
        for col in final_df.columns:
            if col != 'UID':
                n_missing = final_df[col].isna().sum()
                if n_missing > 0:
                    log(f"[MISSING] {col}: {n_missing} missing values ({n_missing/len(final_df)*100:.1f}%)")
        
        # Count complete cases
        t_cols = [col for col in final_df.columns if col.endswith('_T')]
        complete_cases = final_df[t_cols].notna().all(axis=1).sum()
        log(f"[DATA] Complete cases (all 4 T-scores): {complete_cases}/{len(final_df)}")
        
        # =========================================================================
        # STEP 6: Save output
        # =========================================================================
        log(f"[SAVE] Saving cognitive test T-scores to {OUTPUT_FILE}...")
        
        # Select final columns for output
        output_cols = ['UID'] + [col for col in final_df.columns if col.endswith('_T')]
        output_df = final_df[output_cols]
        
        # Save to CSV
        output_df.to_csv(OUTPUT_FILE, index=False)
        log(f"[SAVED] {len(output_df)} participants x {len(output_cols)} columns")
        
        # =========================================================================
        # STEP 7: Validation
        # =========================================================================
        log("[VALIDATION] Verifying output file...")
        
        # Reload and check
        check_df = pd.read_csv(OUTPUT_FILE)
        log(f"[VALIDATION] Output file has {len(check_df)} rows, {len(check_df.columns)} columns")
        log(f"[VALIDATION] Columns: {list(check_df.columns)}")
        
        # Check T-score properties
        for test in ['RAVLT', 'BVMT', 'NART', 'RPM']:
            t_col = f'{test}_T'
            if t_col in check_df.columns:
                mean = check_df[t_col].mean()
                sd = check_df[t_col].std()
                log(f"[VALIDATION] {t_col}: Mean={mean:.2f} (target=50), SD={sd:.2f} (target=10)")
        
        log("[SUCCESS] Step 01 complete - Cognitive test T-scores extracted and saved")
        
    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise