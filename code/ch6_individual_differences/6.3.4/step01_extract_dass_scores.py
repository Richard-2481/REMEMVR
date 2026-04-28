#!/usr/bin/env python3
"""extract_dass_scores: Extract DASS Depression, Anxiety and Stress scores from dfnonvr.csv (all 3 predictors)"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.3.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step01_extract_dass.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_dass_scores")
        # Load Input Data

        log("Loading participant data from dfnonvr.csv...")
        input_path = PROJECT_ROOT / "data" / "dfnonvr.csv"
        df = pd.read_csv(input_path)
        log(f"dfnonvr.csv ({len(df)} participants, {len(df.columns)} columns)")
        
        # Verify DASS columns exist (exact lowercase hyphenated names)
        dass_columns = {
            'depression': 'total-dass-depression-items',
            'anxiety': 'total-dass-anxiety-items', 
            'stress': 'total-dass-stress-items'
        }
        
        missing_cols = [col for col in dass_columns.values() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing DASS columns: {missing_cols}")
        
        log(f"Found all 3 DASS columns: {list(dass_columns.values())}")
        # Extract DASS Scores (Custom Implementation)

        log("Extracting DASS Depression, Anxiety, and Stress scores...")
        
        # Extract DASS columns and rename for output
        dass_data = df[['UID'] + list(dass_columns.values())].copy()
        
        # Rename to standard format (DASS_Dep, DASS_Anx, DASS_Str)
        dass_data = dass_data.rename(columns={
            'total-dass-depression-items': 'DASS_Dep',
            'total-dass-anxiety-items': 'DASS_Anx', 
            'total-dass-stress-items': 'DASS_Str'
        })
        
        # Remove rows with missing DASS data
        initial_n = len(dass_data)
        dass_data = dass_data.dropna(subset=['DASS_Dep', 'DASS_Anx', 'DASS_Str'])
        final_n = len(dass_data)
        log(f"Retained {final_n}/{initial_n} participants with complete DASS data")
        
        # Compute z-standardized versions
        log("Computing z-scores for all 3 DASS subscales...")
        for subscale in ['DASS_Dep', 'DASS_Anx', 'DASS_Str']:
            z_col = subscale.replace('DASS_', 'z_')
            dass_data[z_col] = (dass_data[subscale] - dass_data[subscale].mean()) / dass_data[subscale].std()
            
        # Check for outliers (z > 3.29)
        outlier_threshold = 3.29
        log(f"Checking for extreme z-scores (|z| > {outlier_threshold})...")
        for z_col in ['z_Dep', 'z_Anx', 'z_Str']:
            outliers = np.abs(dass_data[z_col]) > outlier_threshold
            n_outliers = outliers.sum()
            if n_outliers > 0:
                log(f"{z_col}: {n_outliers} extreme values (|z| > {outlier_threshold})")
            else:
                log(f"{z_col}: No extreme outliers")
        
        log("DASS extraction complete")
        # Save Analysis Outputs
        # These outputs will be used by: Step 2 (regression data preparation)

        output_path = RQ_DIR / "data" / "step01_dass_scores.csv"
        log(f"Saving {output_path}...")
        # Output: step01_dass_scores.csv
        # Contains: Raw and z-standardized DASS scores for all 3 subscales
        # Columns: UID, DASS_Dep, DASS_Anx, DASS_Str, z_Dep, z_Anx, z_Str
        dass_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path} ({len(dass_data)} participants, {len(dass_data.columns)} columns)")
        
        # Compute descriptive statistics
        log("DASS raw score statistics:")
        for subscale in ['DASS_Dep', 'DASS_Anx', 'DASS_Str']:
            mean_val = dass_data[subscale].mean()
            std_val = dass_data[subscale].std()
            min_val = dass_data[subscale].min()
            max_val = dass_data[subscale].max()
            log(f"{subscale}: Mean={mean_val:.2f}, SD={std_val:.2f}, Range=[{min_val:.1f}, {max_val:.1f}]")
        
        # Compute intercorrelations
        log("DASS subscale intercorrelations:")
        corr_matrix = dass_data[['DASS_Dep', 'DASS_Anx', 'DASS_Str']].corr()
        dep_anx_corr = corr_matrix.loc['DASS_Dep', 'DASS_Anx']
        dep_str_corr = corr_matrix.loc['DASS_Dep', 'DASS_Str']  
        anx_str_corr = corr_matrix.loc['DASS_Anx', 'DASS_Str']
        log(f"Depression-Anxiety: r={dep_anx_corr:.3f}")
        log(f"Depression-Stress: r={dep_str_corr:.3f}")
        log(f"Anxiety-Stress: r={anx_str_corr:.3f}")
        # Run Validation Tool
        # Validates: DASS raw scores in reasonable range, z-scores properly standardized
        # Threshold: Check z-scores have mean ~0, SD ~1

        log("Running validate_numeric_range...")
        
        # Validate z-scores are properly standardized
        validation_results = {}
        for z_col in ['z_Dep', 'z_Anx', 'z_Str']:
            z_data = dass_data[z_col].values
            # Z-scores should be roughly in [-4, 4] range for psychological data
            validation_result = validate_numeric_range(
                data=z_data,
                min_val=-4.0,
                max_val=4.0, 
                column_name=z_col
            )
            validation_results[z_col] = validation_result
            
            # Check standardization quality
            z_mean = np.mean(z_data)
            z_std = np.std(z_data, ddof=1)
            log(f"{z_col}: Mean={z_mean:.4f}, SD={z_std:.4f}")
            
            if abs(z_mean) > 0.01:
                log(f"WARNING: {z_col} mean not close to 0 (mean={z_mean:.4f})")
            if abs(z_std - 1.0) > 0.01:
                log(f"WARNING: {z_col} SD not close to 1 (SD={z_std:.4f})")

        # Report validation results
        all_valid = all(result.get('valid', False) for result in validation_results.values())
        if all_valid:
            log("All z-scores within expected range [-4, 4]")
        else:
            log("WARNING: Some z-scores outside expected range")
            
        # Check intercorrelations are in reasonable range
        corr_range_ok = all(0.3 <= abs(corr) <= 0.8 for corr in [dep_anx_corr, dep_str_corr, anx_str_corr])
        if corr_range_ok:
            log("DASS intercorrelations in expected range [0.3, 0.8]")
        else:
            log("WARNING: Some DASS correlations outside expected range")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)